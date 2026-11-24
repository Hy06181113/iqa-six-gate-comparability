# -*- coding: utf-8 -*-
"""CLIP-IQA（training-free）推理封装：图 -> 质量分标量 + 二类概率。

方法出处
--------
Wang, Chan, Loy. "Exploring CLIP for Assessing the Look and Feel of Images",
AAAI 2023 (arXiv:2207.12396)。官方实现 IceClear/CLIP-IQA。

本封装复现的是官方 `CLIPIQAFixed` 分支（无需训练的那一支）：
    骨干            OpenAI CLIP **RN50**
    pos_embedding   **False**  ← 论文核心技巧：跳过 attn-pool 的位置编码
    prompt 对       ["Good photo.", "Bad photo."]
    分数            softmax([s_good, s_bad])[0] ∈ [0,1]（即"像好照片"的概率）

★ 与 Q-ReAlign Mini 的差别正是本课题的研究对象
    Q-ReAlign Mini：五级粒度概率 → 按固定权重折算标量（**等级期望型**）
    CLIP-IQA      ：二元反义 prompt 的图文相似度 softmax（**相似度型**）
    —— 两者输出都落在 [0,1]、名义量纲相同，语义却不同。

缓存字段（每条）
    image, score, probs{good,bad}, logits{good,bad}, preprocess, backbone,
    pos_embedding, device, seed

用法
----
    python infer.py --image <图或目录> --out <缓存.jsonl> [--preprocess native|clip]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

from vendor_clip import clip as vclip

# --- 方法常量（改动即不再是 CLIP-IQA，勿动）--------------------------------
BACKBONE = "RN50"
POS_EMBEDDING = False
PROMPT_PAIR = ["Good photo.", "Bad photo."]
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
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


def build_preprocess(mode: str, resize: int = None):
    """PIL.Image -> tensor[C,H,W]。

    native : 官方 KonIQ/LIVE-IWT 评测协议——原分辨率 + CLIP 归一化。
             见官方 `configs/clipiqa/clipiqa_attribute_test.py` 的 test_pipeline
             （LoadImageFromFile -> RescaleToZeroOne -> Normalize(CLIP mean/std)，
             全程 **不缩放**；不缩放之所以可行，正是因为 attn-pool 的位置编码被去掉）。
    clip   : CLIP 官方 preprocess——Resize(224, bicubic) + CenterCrop(224) + 归一化。

    两者的差异本身就是一次"预处理敏感性"对照，留作后续消融。
    """
    from torchvision.transforms import (Compose, Resize, CenterCrop, ToTensor,
                                        Normalize)
    try:
        from torchvision.transforms import InterpolationMode
        bicubic = InterpolationMode.BICUBIC
    except ImportError:  # 老版 torchvision
        bicubic = Image.BICUBIC

    to_rgb = lambda im: im.convert("RGB")  # noqa: E731
    norm = Normalize(CLIP_MEAN, CLIP_STD)

    if mode == "native":
        return Compose([to_rgb, ToTensor(), norm])
    if mode == "clip":
        n = resize or 224
        return Compose([Resize(n, interpolation=bicubic), CenterCrop(n),
                        to_rgb, ToTensor(), norm])
    raise ValueError("未知 preprocess: %s" % mode)


def build_model(ckpt: str, device: str):
    model, _ = vclip.load(ckpt, device=device, jit=False)
    model.eval()
    return model


def score_one(model, preprocess, device, img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    tokens = vclip.tokenize(PROMPT_PAIR).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image.type(model.dtype), tokens, POS_EMBEDDING)
        logits = logits_per_image[0].float().cpu().numpy()
        probs = torch.from_numpy(logits).softmax(-1).numpy()
    return float(probs[0]), probs, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="图片文件或目录")
    ap.add_argument("--out", required=True, help="缓存 .jsonl 路径")
    ap.add_argument("--ckpt", default=r"D:\models\CLIP-RN50\RN50.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preprocess", default="native", choices=["native", "clip"],
                    help="native=官方评测协议(原分辨率)；clip=CLIP 官方 224 预处理")
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--resize", type=int, default=None,
                    help="仅 --preprocess clip 时生效，默认 224")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，降级 CPU", file=sys.stderr)
        args.device = "cpu"

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

    preprocess = build_preprocess(args.preprocess, args.resize)

    print("加载 CLIP %s 到 %s ..." % (BACKBONE, args.device))
    t0 = time.time()
    model = build_model(args.ckpt, args.device)
    print("模型就绪，dtype=%s，加载耗时 %.1f s" % (model.dtype, time.time() - t0))
    print("preprocess=%s  pos_embedding=%s  prompts=%s"
          % (args.preprocess, POS_EMBEDDING, PROMPT_PAIR))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            score, probs, logits = score_one(model, preprocess, args.device, img)
            f.write(json.dumps({
                "image": str(img),
                "score": round(score, 6),
                "probs": {k: round(float(v), 6)
                          for k, v in zip(["good", "bad"], probs)},
                "logits": {k: round(float(v), 4)
                           for k, v in zip(["good", "bad"], logits)},
                "preprocess": args.preprocess,
                "backbone": BACKBONE,
                "pos_embedding": POS_EMBEDDING,
                "device": args.device,
                "seed": args.seed,
            }, ensure_ascii=False) + "\n")
            n += 1
            print("  [%d/%d] %.3f  %s  (%.2f s)"
                  % (i, len(imgs), score, img.name, time.time() - t1))
    print("完成 %d 张，缓存 -> %s" % (n, out))


if __name__ == "__main__":
    main()
