# -*- coding: utf-8 -*-
"""MANIQA（KonIQ-10k 微调）推理封装：图 -> 质量分标量。

方法出处
--------
Yang, Wu, Shi, Lao, Gong, Cao, Wang, Yang. "MANIQA: Multi-dimension Attention
Network for No-Reference Image Quality Assessment", CVPR Workshop 2022
（NTIRE2022 NRIQA 挑战赛冠军）。官方实现 IIGROUP/MANIQA。

本封装复现的配置（**必须与官方/pyiqa 一致，否则静默算错**）：
    骨干      ViT-B/8 (`vit_base_patch8_224`)，取第 6~9 个 Block 的输出拼接
    scale     **0.8**
    crop      224×224，用 `uniform_crop` 采 **20** 个（等步长网格，确定性）
    归一化    (x − 0.5) / 0.5，在模型 `forward` 内部完成
    权重      作者在 KonIQ-10k 上微调的 `ckpt_koniq10k.pt`

★ 陷阱一：`scale` 静默影响结果
    `scale` 在 Swin 里用于 `x = self.scale * x + _x`，是**运行时标量、不进 state_dict**，
    所以 `load_state_dict(strict=True)` **抓不到它**。而 `maniqa_arch.py` 的默认值是
    **0.13**，官方 `predict_one_image.py` 与 pyiqa 的 `maniqa` 配置都是 **0.8**。
    构造时忘传就静默算错。本封装硬编码 `SCALE = 0.8`，不留 CLI 开关。

★ 陷阱二：归一化到底做几次（已查清，勿再改）
    官方 `predict_one_image.py` / `inference.py` 的 transform 是
        `Compose([Normalize(0.5, 0.5), ToTensor()])`  → 喂给模型的是 **[-1, 1]**
    而官方 `models/maniqa.py` 的 `forward` **不做**归一化（直接从 `self.vit(x)` 开始）。
    pyiqa 版则把这次归一化**搬进了 forward**（`default_mean/std = (0.5,0.5,0.5)`），
    并且**没有**数据管线层的归一化。
    ⇒ 两者数学等价：官方喂 [-1,1] 与 pyiqa 喂 [0,1]，进模型后是同一条路径。
    本封装走 pyiqa 路径（省掉自建数据管线）：**喂 [0,1]，不做任何预处理归一化**。

★ 关于 crop 的确定性
    pyiqa **新版** eval 分支用 `uniform_crop`（等步长网格）——**无随机性，可复现**；
    旧版是 `random_crop`，官方 demo 也是 20 个随机 crop。本封装走新版。
    20 个 crop 在 forward 内被折进 batch 维，最后对**全部 20×784 个 patch 加权平均**，
    每张图返回**一个**分数（不是 20 个分数再平均）。
    实测这种「联合平均」与官方 demo 的「逐 crop 平均再取均值」差别很小（各图 ≤ 0.01），
    因为它俩的差别只在加权口径，而 fc_weight 是 sigmoid、patch 间分布平坦。
    见 README「复现验证」。

与本课题其它评委的差别
    Q-ReAlign Mini  等级期望型（A 组）  Qwen3.5-VL 多模态大模型
    CLIP-IQA        相似度型（B 组）    零样本，未在 KonIQ 训练
    MANIQA          回归型（C 组）      **在 KonIQ-10k 上微调过**，有训练污染风险

缓存字段（每条）
    image, score, crops, scale, ckpt, device, dtype, seed, img_size

用法
----
    python infer.py --image <图或目录> --out <缓存.jsonl>
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

from vendor_iqa import MANIQA
from vendor_iqa.arch_util import load_pretrained_network

# --- 方法常量（改动即不再是 MANIQA/KonIQ，勿动）-----------------------------
SCALE = 0.8            # 见 docstring「陷阱一」；不是 MANIQA 类默认的 0.13
CROP_SIZE = 224
DEFAULT_CROPS = 20     # pyiqa `maniqa` 配置与官方 demo 的取值
# ---------------------------------------------------------------------------

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Windows 控制台默认 GBK，装不下 `⇒` `≤` 这类符号，直接 print 会 UnicodeEncodeError。
# 统一把标准输出改 UTF-8（重定向到文件时同样正确）。
# 〔2026-10-10 回填〕此块先落在 topiq-nr / musiq / hyperiqa，本文件后补；纯输出路径，不碰数值。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load_image(path):
    """图 -> float32 [0,1] 的 CHW tensor。

    官方用 `cv2.imread` + `cvtColor(BGR2RGB)`；PIL 的 `convert("RGB")` 与之等价。
    **不做归一化** —— 归一化在模型 forward 内部（见模块 docstring「陷阱二」）。
    """
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0            # HWC, [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def build_model(ckpt, device, crops, dtype):
    net = MANIQA(
        embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8,
        img_size=CROP_SIZE, window_size=4, depths=[2, 2], num_heads=[4, 4],
        num_tab=2, scale=SCALE, test_sample=crops,
        # pretrained=False：不下载 timm 的 ImageNet 权重。见 README「权重完整性」——
        # 官方 ckpt 含全部 316 个键（其中 152 个是 vit.*），strict 加载会整体覆盖初始化。
        pretrained=False,
        # ★ 这里**不能**传 pretrained_model_path：vendored 的 __init__ 走该分支时会以
        #   weight_keys='params' 去取 sd['params']，而官方 ckpt 是**裸 state_dict**
        #   （实测顶层键就是 'vit.cls_token' 等），会直接 KeyError。
        #   故改为构造后手工加载（等价，只是不套那层 weight_keys）。
    )
    load_pretrained_network(net, ckpt, strict=True)
    net.eval().to(device)
    if dtype == "fp16":
        net.half()
    return net


def score_one(net, device, dt, img_path):
    x = load_image(img_path).unsqueeze(0).to(device, dtype=dt)
    with torch.no_grad():
        s = net(x)
    return float(s.reshape(-1)[0].float().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="图片文件或目录")
    ap.add_argument("--out", required=True, help="缓存 .jsonl 路径")
    ap.add_argument("--ckpt", default=r"D:\models\MANIQA\ckpt_koniq10k.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                    help="GTX 1650 显存只有 4 GB；20 crop 的 ViT-B/8 若 OOM 可试 fp16")
    ap.add_argument("--crops", type=int, default=DEFAULT_CROPS,
                    help="uniform_crop 的 crop 数，默认 20（= pyiqa / 官方 demo）")
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="忽略已有缓存，从头重算（默认会跳过缓存里已算过的图）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # uniform_crop 无随机性，这里固定种子只是为了让缓存里的该字段有意义
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

    if not pathlib.Path(args.ckpt).exists():
        sys.exit("权重不存在: %s\n先跑 download_weights.py" % args.ckpt)

    print("加载 MANIQA (scale=%s, crops=%d, %s) 到 %s ..."
          % (SCALE, args.crops, args.dtype, args.device))
    t0 = time.time()
    net = build_model(args.ckpt, args.device, args.crops, args.dtype)
    print("模型就绪，加载耗时 %.1f s" % (time.time() - t0))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 断点续传：2,015 张图按 3.8 s/图 要跑 2 小时，中途挂掉不能从头来。
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
                if min(h, w) <= CROP_SIZE:
                    print("[warn] %s 只有 %dx%d —— min 边 ≤ %d 会触发 uniform_crop 的"
                          "隐式双线性放大（先放大到 %d 再裁回，见 README 陷阱三）"
                          % (img.name, w, h, CROP_SIZE, CROP_SIZE + 1), file=sys.stderr)
                score = score_one(net, args.device, dt, img)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[ERROR] 显存不足，停在第 %d 张。减小 --crops 或改用 --dtype fp16"
                      % i, file=sys.stderr)
                break
            f.write(json.dumps({
                "image": str(img),
                "score": round(score, 6),
                "crops": args.crops,
                "scale": SCALE,
                "crop_size": CROP_SIZE,
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
