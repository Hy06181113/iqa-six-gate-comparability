# -*- coding: utf-8 -*-
"""MUSIQ（KonIQ-10k 支）推理封装：图 -> 质量分标量。

方法出处
--------
Ke, Wang, Wang, Milanfar, Yang. "MUSIQ: Multi-Scale Image Quality Transformer",
ICCV 2021, pp. 5128–5137. 官方实现在 google-research/musiq；
本封装用的是 **pyiqa（chaofengc/IQA-PyTorch）的非官方 PyTorch 重实现**，
因为本课题要统一走 pyiqa 的权重与推理口径（与 MANIQA / TOPIQ-NR 同一来源）。

本封装复现的配置（**改动即不再是 MUSIQ/KonIQ**）：
    patch      patch_size = patch_stride = 32
    尺度       longer_side_lengths = [224, 384] + **原分辨率**尺度
               （max_seq_len_from_original_res = -1 ⇒ 原分辨率那一支不截断，全用）
    位置编码   hashed spatial position, hse_grid_size = 10；尺度编码打开
    编码器     hidden 384 / mlp 1152 / 6 头 / 14 层 / 3 尺度
    归一化     (x − 0.5) × 2 → [−1, 1]，在模型 `forward` 内部完成（**不是** ImageNet 归一化）
    输出       num_class = 1 ⇒ `dist_to_mos` 退化为恒等 ⇒ 输出即最后一层的标量；
               名义区间见 pyiqa 配置 `score_range: '~0, ~100'`（不是 [0,1]！）
    权重       pyiqa 的 `musiq_koniq_ckpt-e95806b9.pth`（`pretrained='koniq10k'`）

★ 三处与 TOPIQ-NR 不同、必须单独记的口径
    1) **归一化**：TOPIQ-NR 用 ImageNet mean/std；MUSIQ 用 (x−0.5)×2。两者都在模型内部做。
    2) **输入是分块序列**，不是整图：一张 512×384 的图会变成「224 尺度 + 384 尺度 +
       原分辨率尺度」三组 patch 拼接的长序列，序列长度远大于 1。
       ⇒ 显存与耗时随**原图分辨率**增长，不是常数（这与 MANIQA 的固定 20 crop 也不同）。
    3) **输出量纲**：本模块返回的是 MUSIQ 自己的分数，**名义区间 ~[0,100]**，
       与 TOPIQ-NR 的 ~[0,1] **不可直接比大小**。跨评委比较一律先做秩/映射。

★ 污染状态（必读）
    MUSIQ 的训练 split 是**全库随机 80/20**，种子与图像清单**均未公布**
    ⇒ **无法重建干净子集**，本评委在「同源对」上只能保守记账，见
    评委元数据表。**不得**把它当干净基线用。

★ 为什么 `load_pretrained_network(..., strict=True)`
    与 TOPIQ-NR 同理：strict 加载本身就是一次**架构等价性检验**。若我们 vendored 的
    MUSIQ 与上游结构有任何出入（少一层、多一个 buffer），这里会直接抛错，
    而不是静默加载一部分权重算出错分数。键集实测见 README「复现验证」。

缓存字段（每条）
    image, score, dist, pretrained, num_class, patch_size, longer_side_lengths,
    max_seq_len_from_original_res, weight_keys, ckpt, device, dtype, seed, img_size

用法
----
    python infer.py --image <图或目录> --out <缓存.jsonl>
    python infer.py --image <目录> --list <名单.txt> --out <缓存.jsonl>   # 官方子集
    python infer.py --inspect-ckpt          # 只打印 ckpt 键集，不跑图
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

from vendor_musiq import MUSIQ
from vendor_musiq.arch_util import load_pretrained_network

# --- 方法常量（改动即不再是 MUSIQ/KonIQ，勿动）------------------------------
PRETRAINED = "koniq10k"      # pyiqa default_model_configs['musiq'] 的 pretrained 取值
NUM_CLASS = 1                # koniq10k 支；只有 ava 支才用 10 类分布
PATCH_SIZE = 32              # MUSIQ.__init__ 默认
PATCH_STRIDE = 32            # 由 data_preprocess_opts 显式传给 get_multiscale_patches
HSE_GRID_SIZE = 10           # hashed spatial position 网格
LONGER_SIDE_LENGTHS = [224, 384]
MAX_SEQ_LEN_FROM_ORIGINAL_RES = -1   # -1 ⇒ 原分辨率尺度全取；None ⇒ 完全不用原分辨率
# 上游 MUSIQ.__init__ 调的是 `load_pretrained_network(self, path, True)`
# ⇒ weight_keys 取默认 None，即 ckpt **本身就是** state_dict，没有外层包裹。
# 这一点由 `--inspect-ckpt` 实测确认（见 README），不靠猜。
WEIGHT_KEYS = None
# ---------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Windows 控制台默认 GBK，装不下 `⇒` `≤` 这类符号，直接 print 会 UnicodeEncodeError。
# 统一把标准输出改 UTF-8（重定向到文件时同样正确）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_image(path):
    """图 -> float32 [0,1] 的 CHW tensor。

    MUSIQ 的 `forward` 内部做 `(x − 0.5) × 2` 再分块（`get_multiscale_patches`），
    此处**不做任何缩放、不做归一化**——与 TOPIQ-NR 一样按原分辨率进模型。
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0             # HWC, [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # CHW


def build_model(ckpt, device, dtype):
    net = MUSIQ(
        patch_size=PATCH_SIZE,
        num_class=NUM_CLASS,
        pretrained=False,          # 不走 URL 分支（vendored 已禁用）；手工加载，见下
        longer_side_lengths=LONGER_SIDE_LENGTHS,
        max_seq_len_from_original_res=MAX_SEQ_LEN_FROM_ORIGINAL_RES,
    )
    # strict=True：既是加载，也是对「vendored 架构 == 上游架构」的检验
    load_pretrained_network(net, ckpt, strict=True, weight_keys=WEIGHT_KEYS)
    net.eval().to(device)
    if dtype == "fp16":
        net.half()
    return net


def score_one(net, device, dt, img_path):
    x = load_image(img_path).unsqueeze(0).to(device, dtype=dt)
    with torch.no_grad():
        mos, dist = net(x, return_mos=True, return_dist=True)
    return (float(mos.reshape(-1)[0].float().item()),
            float(dist.reshape(-1)[0].float().item()))


def inspect_ckpt(ckpt):
    """打印 ckpt 的顶层结构，并核对「vendored 架构键集 == ckpt 键集」。

    只读结构、不做前向。MUSIQ 与 CFANet 不同，**没有** semantic_model 之类的
    外部预训练骨干，所以这里只判一件事：键集是否逐键对齐 + 是否需要 weight_keys 包裹。
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("顶层类型: %s" % type(sd).__name__)
    if isinstance(sd, dict):
        top = list(sd.keys())
        print("顶层键（前 5）: %s" % top[:5])
        is_wrapped = len(top) < 20 and all(isinstance(sd[k], dict) for k in top)
        print("是否外层包裹（如 {'params': {...}}）: %s" % is_wrapped)
        inner = sd[top[0]] if is_wrapped else sd
    else:
        inner = sd
    keys = list(inner.keys())
    print("参数键数: %d" % len(keys))
    print("前 3 个键: %s" % keys[:3])
    print("后 3 个键: %s" % keys[-3:])

    net = MUSIQ(
        patch_size=PATCH_SIZE, num_class=NUM_CLASS, pretrained=False,
        longer_side_lengths=LONGER_SIDE_LENGTHS,
        max_seq_len_from_original_res=MAX_SEQ_LEN_FROM_ORIGINAL_RES,
    )
    want = set(net.state_dict().keys())
    got = set(inner.keys())
    print("架构键集 == ckpt 键集: %s" % (want == got))
    if want != got:
        print("  仅架构有 (%d): %s" % (len(want - got), sorted(want - got)[:5]))
        print("  仅 ckpt 有 (%d): %s" % (len(got - want), sorted(got - want)[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="图片文件或目录")
    ap.add_argument("--out", help="缓存 .jsonl 路径")
    ap.add_argument("--ckpt",
                    default=r"D:\models\MUSIQ\musiq_koniq_ckpt-e95806b9.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="忽略已有缓存，从头重算（默认跳过缓存里已算过的图）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect-ckpt", action="store_true",
                    help="只打印 ckpt 键集与架构等价性，不跑图")
    args = ap.parse_args()

    if not pathlib.Path(args.ckpt).exists():
        sys.exit("权重不存在: %s\n先跑 download_weights.py" % args.ckpt)
    if args.inspect_ckpt:
        inspect_ckpt(args.ckpt)
        return
    if not args.image or not args.out:
        sys.exit("需要 --image 与 --out（含 --list 时也要 --image，"
                 "它指向图片所在目录）；或用 --inspect-ckpt")

    # MUSIQ 的 eval 路径是确定性的（无随机裁剪、无 dropout——已 eval）；固定种子只为留痕
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，降级 CPU", file=sys.stderr)
        args.device = "cpu"
    dt = torch.float32 if args.dtype == "fp32" else torch.float16
    if args.device == "cpu":
        dt = torch.float32

    p = pathlib.Path(args.image)
    if args.list:
        # 按名单取图，绝不对目录 glob——KonIQ-10k 目录里有全部 10,073 张，
        # glob 会把训练集混进来——校准结论必须只在官方 test 子集上做。
        if not p.is_dir():
            sys.exit("--list 需要 --image 指向图片目录，收到: %s" % p)
        names = [ln.strip() for ln in open(args.list, encoding="utf-8") if ln.strip()]
        imgs = [p / nm for nm in names]
        missing = [x.name for x in imgs if not x.exists()]
        if missing:
            sys.exit("--list 里 %d 张图在 %s 下找不到，例：%s"
                     % (len(missing), p, ", ".join(missing[:3])))
    elif p.is_dir():
        imgs = sorted(x for x in p.glob("*") if x.suffix.lower() in EXTS)
    else:
        imgs = [p]
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        sys.exit("没有找到图片: %s" % p)

    print("加载 MUSIQ (%s, patch=%d, 尺度=%s+原分辨率, %s) 到 %s ..."
          % (PRETRAINED, PATCH_SIZE, LONGER_SIDE_LENGTHS, args.dtype, args.device))
    t0 = time.time()
    net = build_model(args.ckpt, args.device, dt)
    print("模型就绪（strict 加载通过 ⇒ 架构与上游一致），耗时 %.1f s" % (time.time() - t0))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：几千张图一趟跑几小时，中途挂掉不能从头来。
    # 已算过的图按文件名跳过，缓存以追加方式续写。
    done = set()
    if out.exists() and not args.force:
        with open(out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(pathlib.Path(json.loads(line)["image"]).name)
                    except (json.JSONDecodeError, KeyError):
                        pass          # 半行/坏行：忽略，重算即可
    if done:
        before = len(imgs)
        imgs = [x for x in imgs if x.name not in done]
        print("[RESUME] %s 已有 %d 条，跳过；本轮待算 %d / %d 张"
              % (out.name, len(done), len(imgs), before))

    n = 0
    with open(out, "a" if done else "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            try:
                w, h = Image.open(img).size
                score, dist = score_one(net, args.device, dt, img)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[ERROR] 显存不足，停在第 %d 张。改用 --dtype fp16 或先缩图" % i,
                      file=sys.stderr)
                break
            f.write(json.dumps({
                "image": str(img),
                "score": round(score, 6),
                "dist": round(dist, 6),
                "pretrained": PRETRAINED,
                "num_class": NUM_CLASS,
                "patch_size": PATCH_SIZE,
                "longer_side_lengths": LONGER_SIDE_LENGTHS,
                "max_seq_len_from_original_res": MAX_SEQ_LEN_FROM_ORIGINAL_RES,
                "weight_keys": WEIGHT_KEYS,
                "ckpt": args.ckpt,
                "device": args.device,
                "dtype": args.dtype,
                "seed": args.seed,
                "img_size": [h, w],
            }, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("  [%d/%d] %.4f  %s  (%dx%d, %.2f s)"
                  % (i, len(imgs), score, img.name, h, w, time.time() - t1), flush=True)
    print("完成 %d 张，缓存 -> %s" % (n, out))


if __name__ == "__main__":
    main()
