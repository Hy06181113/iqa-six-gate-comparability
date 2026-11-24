# -*- coding: utf-8 -*-
"""Q-ReAlign Mini (0.8B) 推理封装：图 -> 五级概率分布 + 质量分标量。

用法:
    python infer.py --image <图或目录> --out <缓存.jsonl> [--limit N]

输出缓存字段（每条）:
    image, score, probs{excellent,good,fair,poor,bad}, device, dtype, seed

设计约束:
  - 必须缓存五级概率分布（需含中间量），不能只存标量；
    C12 的可靠性估计要用到五级分布。
  - GTX 1650 无 bf16，dtype 用 fp16；OOM 时降级 CPU。
  - batch=1，限制输出长度（0.8B 在 4GB 显存上很紧）。
"""
import argparse, json, sys, time, pathlib
import numpy as np
import torch

LEVELS = ["excellent", "good", "fair", "poor", "bad"]
WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]
PROMPT = "How would you rate the quality of this image?"
STEM = "The quality of the image is"


def build_model(ckpt: str, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(ckpt)
    if device == "cuda":
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                ckpt, dtype=torch.float16).to("cuda").eval()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print("  [warn] CUDA 加载/推理失败，降级 CPU:", type(e).__name__, file=sys.stderr)
            model = AutoModelForImageTextToText.from_pretrained(
                ckpt, dtype=torch.float32).to("cpu").eval()
            device = "cpu"
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            ckpt, dtype=torch.float32).to("cpu").eval()
    return processor, model, device


def score_one(processor, model, device, img_path, seed):
    from PIL import Image

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True) + STEM
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    ids = [processor.tokenizer(" " + w, add_special_tokens=False).input_ids[0]
           for w in LEVELS]
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, ids]
        probs = logits.softmax(-1).cpu().float().numpy()
    score = float((probs * np.array(WEIGHTS)).sum())
    return score, dict(zip(LEVELS, [round(float(p), 6) for p in probs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=r"D:\models\Q-ReAlign-Mini-0.8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    imgs = []
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
        imgs = sorted(p.glob("*"))  # 先跑通，再收窄到图像扩展名
        imgs = [x for x in imgs if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    else:
        imgs = [p]
    if args.limit:
        imgs = imgs[: args.limit]

    print("加载模型 %s 到 %s ..." % (args.ckpt, args.device))
    t0 = time.time()
    processor, model, device = build_model(args.ckpt, args.device)
    print("模型就绪，device=%s，加载耗时 %.1f s" % (device, time.time() - t0))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            score, probs = score_one(processor, model, device, img, args.seed)
            f.write(json.dumps({"image": str(img), "score": round(score, 6),
                                "probs": probs, "device": device,
                                "seed": args.seed}, ensure_ascii=False) + "\n")
            n += 1
            print("  [%d/%d] %.3f  %s  (%.1f s)" % (i, len(imgs), score, img.name, time.time() - t1))
    print("完成 %d 张，缓存 -> %s" % (n, out))


if __name__ == "__main__":
    main()
