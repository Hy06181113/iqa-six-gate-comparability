# -*- coding: utf-8 -*-
"""TOPIQ-NR（CFANet resnet50 支）推理封装：图 -> 质量分标量。

方法出处
--------
Chen, Mo, Hou, Wu, Liao, Sun, Yan, Lin.
"TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality
Assessment", IEEE TIP 2024. arXiv:2308.03060。官方实现 chaofengc/IQA-PyTorch。

本封装复现的配置（**改动即不再是 TOPIQ-NR/KonIQ**）：
    骨干      timm `resnet50`（features_only），取全部中间尺度特征做跨尺度注意力融合
    裁剪      num_crop = 1 ⇒ **不裁剪、不缩放，按原分辨率整图进**
    归一化    (x − ImageNet_mean) / ImageNet_std，在模型 `preprocess` 内部完成
    权重      pyiqa 的 `cfanet_nr_koniq_res50`（KonIQ-10k 支）
    输出      dist_to_mos 在 num_class=1 时退化为恒等 ⇒ 输出即模型最后一层的标量，
              名义区间 ≈ [0, 1]（pyiqa 配置写作 `score_range: '~0, ~1'`）

★ 与 MANIQA 封装的一处结构差别（值得记）
    MANIQA 是 **20 个 224 crop** 后对 patch 加权平均；TOPIQ-NR 是 **单张原分辨率整图**。
    这恰好是 C 组内部两种不同的输入口径，写 P1 的「可互换区间」时要分开处理，
    不能因为同属「回归型」就默认它们的批处理特性一样。
    ⇒ 批处理加速比、显存占用、CPU 侧开销三者在这两个评委身上**不能互相外推**。

★ 为什么 `load_pretrained_network(..., strict=True)`
    上游 CFANet 有两条加载路径：`pretrained_model_path` 走 `strict=False`，
    `pretrained=True` 走 URL + `strict=True`（本封装禁用了 URL 分支）。
    本封装**手工以 `strict=True` 加载本地文件**，因为 strict 加载本身就是一次
    **架构等价性检验**：若我们 vendored 的 CFANet 与上游结构有任何出入（少一层、
    多一个 buffer），这里会直接抛错，而不是静默加载一部分权重算出错分数。
    实测该 ckpt 的键集与 `net.state_dict()` **完全一致**（见 README「复现验证」）。

★ `backbone_pretrain`（见 `--inspect-ckpt`）
    若 ckpt 含 `semantic_model.*`（实测：含），则 timm 的 ImageNet 权重会被整体覆盖，
    下它纯属浪费 ⇒ `BACKBONE_PRETRAIN = False`。判定不靠猜，用 `--inspect-ckpt` 实测。

与本课题其它评委的差别
    Q-ReAlign Mini  等级期望型（A 组）   Qwen3.5-VL 多模态大模型
    CLIP-IQA        相似度型（B 组）     零样本，未在 KonIQ 训练
    MANIQA          回归型（C 组）       20×224 crop；**在 KonIQ-10k 上微调过**，污染
    TOPIQ-NR        回归型（C 组）       单张整图、原分辨率；训练用**官方 split** ⇒ 推定干净

缓存字段（每条）
    image, score, dist, model_name, crop_size, num_crop, backbone_pretrain,
    ckpt, device, dtype, seed, img_size

用法
----
    python infer.py --image <图或目录> --out <缓存.jsonl>
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

from vendor_cfanet import CFANet
from vendor_cfanet.arch_util import load_pretrained_network

# --- 方法常量（改动即不再是 TOPIQ-NR/KonIQ，勿动）---------------------------
MODEL_NAME = "cfanet_nr_koniq_res50"   # pyiqa default_model_configs['topiq_nr'] 的 model_name
SEMANTIC_MODEL = "resnet50"
NUM_CROP = 1                           # pyiqa 配置未覆盖 ⇒ 用 CFANet 默认 1
CROP_SIZE = 256                        # 同上；num_crop=1 时该值不起作用，仅记录
TEST_IMG_SIZE = None                   # pyiqa 配置未覆盖 ⇒ 不缩放，按原分辨率
BACKBONE_PRETRAIN = False              # 由 --inspect-ckpt 判定，见模块 docstring
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

    归一化在模型 `preprocess` 内部做（(x − ImageNet_mean) / ImageNet_std），
    此处**不做任何缩放、不做归一化**——TOPIQ-NR 走原分辨率整图。
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0             # HWC, [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # CHW


def build_model(ckpt, device, dtype):
    net = CFANet(
        semantic_model_name=SEMANTIC_MODEL,
        model_name=MODEL_NAME,
        use_ref=False,
        num_crop=NUM_CROP,
        crop_size=CROP_SIZE,
        test_img_size=TEST_IMG_SIZE,
        backbone_pretrain=BACKBONE_PRETRAIN,
        pretrained=False,          # 不走 URL 分支（vendored 已禁用）；手工加载，见下
        align_crop_face=False,     # 非 gfiqa 支，关掉以免触碰未 vendor 的 facexlib
    )
    # strict=True：既是加载，也是对「vendored 架构 == 上游架构」的检验
    load_pretrained_network(net, ckpt, strict=True, weight_keys="params")
    net.eval().to(device)
    if dtype == "fp16":
        net.half()
    return net


def score_one(net, device, dt, img_path):
    x = load_image(img_path).unsqueeze(0).to(device, dtype=dt)
    with torch.no_grad():
        mos, dist = net(x, return_mos=True, return_dist=True)
    return float(mos.reshape(-1)[0].float().item()), float(dist.reshape(-1)[0].float().item())


def inspect_ckpt(ckpt):
    """打印 ckpt 的顶层结构，用于判定 backbone_pretrain 与架构等价性。

    只读结构、不做前向。`params` 键是上游 CFANet 的约定（weight_keys='params'）。
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("顶层类型: %s" % type(sd).__name__)
    if isinstance(sd, dict):
        print("顶层键（前 5）: %s" % list(sd.keys())[:5])
        inner = sd.get("params", sd)
    else:
        inner = sd
    keys = list(inner.keys())
    print("参数键数: %d" % len(keys))
    has_sem = [k for k in keys if k.startswith("semantic_model.")]
    print("semantic_model.* 键数: %d" % len(has_sem))
    print("⇒ backbone_pretrain 应为 False（骨架权重已在 ckpt 内）"
          if has_sem else "⇒ ckpt 不含骨架权重，backbone_pretrain 必须为 True")
    if keys:
        print("前 3 个键: %s" % keys[:3])
        print("后 3 个键: %s" % keys[-3:])
    # 与 vendored 架构的键集比对（不加载权重，只比结构）
    # backbone_pretrain=False：键集与它无关，而 True 会去下 timm 的 ImageNet 权重
    # （本机不可达）。架构本身不因该开关而变。
    net = CFANet(
        semantic_model_name=SEMANTIC_MODEL, model_name=MODEL_NAME, use_ref=False,
        num_crop=NUM_CROP, crop_size=CROP_SIZE, test_img_size=TEST_IMG_SIZE,
        backbone_pretrain=False, pretrained=False, align_crop_face=False,
    )
    from vendor_cfanet.arch_util import clean_state_dict
    want = set(net.state_dict().keys())
    got = set(clean_state_dict(dict(inner)).keys())
    print("架构键集 == ckpt 键集: %s" % (want == got))
    if want != got:
        print("  仅架构有 (%d): %s" % (len(want - got), sorted(want - got)[:5]))
        print("  仅 ckpt 有 (%d): %s" % (len(got - want), sorted(got - want)[:5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="图片文件或目录")
    ap.add_argument("--out", help="缓存 .jsonl 路径")
    ap.add_argument("--ckpt",
                    default=r"D:\models\TOPIQ-NR\cfanet_nr_koniq_res50-9a73138b.pth")
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
        sys.exit("需要 --image 与 --out（或用 --inspect-ckpt）")

    # CFANet 的 eval 路径是确定性的（num_crop=1，无随机裁剪）；固定种子只为留痕
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

    print("加载 TOPIQ-NR (resnet50, num_crop=%d, %s) 到 %s ..."
          % (NUM_CROP, args.dtype, args.device))
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
                print("[ERROR] 显存不足，停在第 %d 张。改用 --dtype fp16" % i,
                      file=sys.stderr)
                break
            f.write(json.dumps({
                "image": str(img),
                "score": round(score, 6),
                "dist": round(dist, 6),
                "model_name": MODEL_NAME,
                "semantic_model": SEMANTIC_MODEL,
                "num_crop": NUM_CROP,
                "crop_size": CROP_SIZE,
                "backbone_pretrain": BACKBONE_PRETRAIN,
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
