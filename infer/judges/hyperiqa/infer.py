# -*- coding: utf-8 -*-
"""HyperIQA（HyperNet，KonIQ-10k 支）推理封装：图 -> 质量分标量。

方法出处
--------
Su, Yan, Zhu, Zhang, Ge, Sun, Zhang. "Blindly Assess Image Quality in the Wild
Guided by a Self-Adaptive Hyper Network", CVPR 2020, pp. 3667–3676.
官方实现 SSL92/hyperIQA（**未发布 KonIQ-10k 权重**）。
本封装用的是 **pyiqa（chaofengc/IQA-PyTorch）重新训练并发布的 PyTorch 实现**。

★ **本评委的权重出身与另两个 C 组评委都不同，必须单独记**（判污染看「我们加载的那份权重的训练 split」，不看原论文）
    上游 `docs/benchmark.rst` 原文把 `hyperiqa` 与 `cnniqa` / `dbcnn` 一起列在
    「**不提供官方权重、由我们（pyiqa）重训**」一栏 ⇒ 本权重的训练 split
    **不看原论文的「全库随机 80/20」**，而看 pyiqa 训练配置的 `split_file`：
    `options/train/HyperNet/train_HyperNet.yml` 写的是
    **`./datasets/meta_info/koniq10k_official.pkl`（官方 split）**
    ⇒ **推定它没见过我们的官方 test 图**（中置信，与 TOPIQ-NR 同档）。
    ⚠️ 未走到逐图清单，无法确认该 yml 就是生成这份 ckpt 的那一次跑。
    详见本目录 `README.md` 与评委元数据表 1。

本封装复现的配置（**改动即不再是 HyperIQA/KonIQ**）：
    骨干       timm 的 resnet50，`features_only=True`
    crop       num_crop = 25（5×5 均匀网格），crop_size = 224 —— **上游硬编码 224**，
               因为 HyperNet 的自适应超网络结构决定了输入必须是 224×224
    归一化     ImageNet mean/std，在模型 `preprocess` 内部完成（**不是** (x−0.5)×2）
    输出       pyiqa 配置 `score_range: '~0, ~1'` ⇒ 名义区间 **[0,1]**，
               与 MUSIQ 的 ~[0,100] **不可直接比大小**
    权重       pyiqa 的 `HyperIQA-resnet50-koniq10k-c96c41b1.pth`（**pyiqa 重训**）

★ 与同组另两个评委的口径差别（C 组内部三样东西都不一样）
    |            | MUSIQ            | HyperIQA          | MANIQA          |
    |---|---|---|---|
    | 输入       | 多尺度 patch 序列 | **25 × 224 crop** | 20 × 224 crop   |
    | 归一化     | (x−0.5)×2        | ImageNet mean/std | 见 maniqa/README |
    | 量纲       | ~[0,100]         | ~[0,1]            | ~[0,1]          |

★ 一处**必须靠 `--inspect-ckpt` 判定、不能靠猜**的地方
    上游 `timm.create_model(..., pretrained=True, ...)` 是**硬编码**的，会去 HF 下
    ImageNet resnet50。本环境 huggingface.co 不可达、HF 缓存里也没有该权重（已核实），
    故 vendored 架构把它改成 `BACKBONE_PRETRAIN = False`。
    **该改动成立的判据只有一个**：ckpt 内是否自带 `base_model.*` 全部参数——
    即 `--inspect-ckpt` 报出的 `架构键集 == ckpt 键集: True`。

★ 为什么 `load_pretrained_network(..., strict=True)`
    与 TOPIQ-NR / MUSIQ 同理：strict 加载本身就是一次**架构等价性检验**。
    上游 HyperNet 加载用的就是 `weight_keys='params'`（见 `hypernet_arch.py` 第 147–154 行），
    本封装沿用；ckpt 若为裸 state_dict，这里会直接 `KeyError`，不会静默出错。

缓存字段（每条）
    image, score, base_model_name, num_crop, backbone_pretrain, weight_keys,
    ckpt, device, dtype, seed, img_size

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

from vendor_hypernet import HyperNet
from vendor_hypernet.arch_util import load_pretrained_network

# --- 方法常量（改动即不再是 HyperIQA/KonIQ，勿动）---------------------------
BASE_MODEL_NAME = "resnet50"
NUM_CROP = 25                # HyperNet.__init__ 默认；pyiqa 配置未覆盖
BACKBONE_PRETRAIN = False    # ★ 见文件头「一处必须靠 --inspect-ckpt 判定的地方」
# 上游 HyperNet.__init__ 调的是
#   load_pretrained_network(self, default_model_urls['resnet50-koniq'], True, weight_keys='params')
# ⇒ weight_keys 取 'params'（ckpt 外层有 params 包裹）。由 `--inspect-ckpt` 实测确认。
WEIGHT_KEYS = "params"
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

    HyperNet 在 `eval()` 下自己做 `uniform_crop(..., 224, num_crop)`，
    切出的 patch 已是 224×224，`preprocess` 再做 ImageNet 归一化。
    此处**不做任何缩放、不做归一化**——按原分辨率进模型（与 TOPIQ-NR / MUSIQ 同）。
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0             # HWC, [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # CHW


def build_model(ckpt, device, dtype):
    net = HyperNet(
        base_model_name=BASE_MODEL_NAME,
        num_crop=NUM_CROP,
        pretrained=False,          # 不走 URL 分支；手工加载，见下
    )
    # strict=True：既是加载，也是对「vendored 架构 == 上游架构」的检验，
    # 同时判定 BACKBONE_PRETRAIN=False 是否成立（base_model.* 必须全在 ckpt 里）。
    load_pretrained_network(net, ckpt, strict=True, weight_keys=WEIGHT_KEYS)
    net.eval().to(device)
    if dtype == "fp16":
        net.half()
    return net


def score_one(net, device, dt, img_path):
    x = load_image(img_path).unsqueeze(0).to(device, dtype=dt)
    with torch.no_grad():
        out = net(x)
    return float(out.reshape(-1)[0].float().item())


def inspect_ckpt(ckpt):
    """打印 ckpt 的顶层结构，并核对「vendored 架构键集 == ckpt 键集」。

    只读结构、不做前向。本评委这里要判**两件事**：
      ① 是否需要 `weight_keys='params'` 包裹（上游约定是 'params'）；
      ② `base_model.*` 是否全在 ckpt 内 —— 这决定 `BACKBONE_PRETRAIN=False` 是否成立。
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
    print("base_model.* 键数: %d" % sum(1 for k in keys if k.startswith("base_model.")))
    print("前 3 个键: %s" % keys[:3])
    print("后 3 个键: %s" % keys[-3:])

    net = HyperNet(base_model_name=BASE_MODEL_NAME, num_crop=NUM_CROP, pretrained=False)
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
                    default=r"D:\models\HyperIQA\HyperIQA-resnet50-koniq10k-c96c41b1.pth")
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

    # HyperIQA 的 eval 路径是确定性的（`uniform_crop` 均匀网格、无随机裁剪；已 eval）
    # 固定种子只为留痕
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

    print("加载 HyperIQA (%s, num_crop=%d, backbone_pretrain=%s, %s) 到 %s ..."
          % (BASE_MODEL_NAME, NUM_CROP, BACKBONE_PRETRAIN, args.dtype, args.device))
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
                score = score_one(net, args.device, dt, img)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[ERROR] 显存不足，停在第 %d 张。改用 --dtype fp16 或先缩图" % i,
                      file=sys.stderr)
                break
            f.write(json.dumps({
                "image": str(img),
                "score": round(score, 6),
                "base_model_name": BASE_MODEL_NAME,
                "num_crop": NUM_CROP,
                "backbone_pretrain": BACKBONE_PRETRAIN,
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
