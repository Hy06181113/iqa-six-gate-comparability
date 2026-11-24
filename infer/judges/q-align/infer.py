# -*- coding: utf-8 -*-
"""Q-Align (mPLUG-Owl2-7B, IQA 支, q-future/q-align-iqa) 推理封装。

    python infer.py --image <图或目录> --out <缓存.jsonl> [--limit N] [--list 名单.txt]
    python infer.py --inspect-ckpt                 # 只看权重分片，不加载
    python infer.py --tokenizer-only [--image 图]  # 只验 token 级口径，不碰权重

输出缓存字段（每条）:
    image, score, probs{excellent,good,fair,poor,bad}, device, dtype, seed

★ 无卡模式（AutoDL）装不下 7B —— 2026-10-13 实测它的 cgroup 限额是
  **0.5 核 + 2 GiB 内存**（`cpu.max=50000 100000`、`memory.max=2147483648`），
  而权重 fp16 要 16.41 GB、fp32 要 28 GB，`Loading checkpoint shards: 0%` 后即被内存配额杀掉。
  ⇒ 无卡模式里**只能**跑 `--inspect-ckpt` 与 `--tokenizer-only`；真正的前向必须上有卡模式。
  另：无卡模式下 torch 默认线程数=56 去抢 0.5 核，实测比单线程**慢 199×**，必须 `--threads 1`。

设计约束:
  - **必须缓存五级概率分布**，不能只存标量（C12 的可靠性估计要用到分布）。
    官方的 `QAlignScorer.forward` 在 `@ weight_tensor` 之后就只剩标量、**分布被丢掉**，
    所以本文件**逐字复刻 scorer.py 的计算**（见下 SCORER_PY 常量），只多做一件事：
    把加权**之前**的 softmax 结果留下来。这是本文件与官方唯一的功能差异。
  - `--list` 按名单取图、不对目录 glob——校准结论只许在官方 test 子集上做。
  - CLI 与其它评委保持一致（`--image --out --ckpt --device --dtype --list --limit --force --seed`）。

★ 加载方式（2026-10-13 核定，见 README.md）:
  `q-future/q-align-iqa` 的 config.json **没有 auto_map**、仓库内也没有 `modeling_*.py`
  ⇒ 官方 README 那条 `trust_remote_code=True` 的 AutoModel 快捷方式**对它无效**（只对
  `one-align` 有效）。正解是走 GitHub 仓库的 `q_align` 包，其 `modeling_mplug_owl2.py`
  第 377-378 行向 transformers 注册了 `mplug_owl2`，故 `import q_align` 之后
  `AutoModelForCausalLM.from_pretrained(<本 checkpoint>)` 即用仓库自带的类加载。

★ 分数量纲:
  本文件输出的 score 与官方 `scorer.py` 一致，是 **0–1**（权重 `[1,.75,.5,.25,0]`），
  与 `../q-realign-mini/infer.py` 的 `WEIGHTS` **同量纲、可直接比**。
  ⚠️ 官方 README 的 `wa5()` 片段用的是 `[5,4,3,2,1]`（1–5 制）；两者是**仿射关系**
  `score(1-5) = 4 * score(0-1) + 1`。**引用数字时必须带量纲。**
"""
import argparse
import json
import os
import pathlib
import sys
import time

import torch

# --- 与官方 scorer.py 逐字一致的常量 -----------------------------------------
# 出处: Q-Align 仓库 q_align/evaluate/scorer.py 的 QAlignScorer.__init__ 与 forward
# （2026-10-13 一手取回；commit 见 /root/autodl-tmp/Q-Align_COMMIT.txt）
LEVELS = ["excellent", "good", "fair", "poor", "bad"]
WEIGHTS = [1.0, 0.75, 0.5, 0.25, 0.0]          # ★ 官方是 [1,.75,.5,.25,0]，不是 [5,4,3,2,1]
PROMPT = ("USER: How would you rate the quality of this image?\n"
          "<|image|>\n"
          "ASSISTANT: The quality of the image is")   # ★ 连 "ASSISTANT: ... is" 都要带上
DEFAULT_REPO = "/root/autodl-tmp/Q-Align"

SCORER_PY = "q_align/evaluate/scorer.py"


def _add_repo_to_path(repo: str):
    """把 Q-Align 仓库根目录放进 sys.path —— 这样就**不必** pip install -e .。

    ⚠️ 用 sys.path 注入而非安装，是为了**不动**我们 pin 好的依赖栈
    （`pip install -e .` 会去解析仓库的 pyproject.toml，有改动 transformers==4.36.1
     与 torch==2.0.1 的风险）。
    """
    repo = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo, "q_align")):
        sys.exit("在 %s 下找不到 q_align/ —— 请用 --repo 指向 Q-Align 仓库根目录" % repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def _import_qalign(repo: str):
    """导入 q_align 的三个入口，返回 (load_pretrained_model, IMAGE_TOKEN_INDEX, tokenizer_image_token)。

    ★ 单独抽出来是为了 `--tokenizer-only`：**验 tokenizer 口径不需要加载模型**。
    为什么要这条捷径（2026-10-13 实测）：AutoDL **无卡模式的内存上限是 2 GiB**
    （`/sys/fs/cgroup/memory.max = 2147483648`），7B 权重（fp16 16.41 GB / fp32 28 GB）
    根本装不进来——`Loading checkpoint shards: 0%` 之后进程就被内存配额杀掉了。
    而 tokenizer 只有 499,723 B，其口径（五级 token id、prompt 的 input_ids）**可以在无卡模式里先验掉**，
    把「必须上 GPU 才能验」的范围压到只剩一次前向。
    """
    _add_repo_to_path(repo)
    # ↓ 这一句既是导入、也是**注册**：modeling_mplug_owl2.py:377-378 在此触发
    #   AutoConfig.register("mplug_owl2", ...) / AutoModelForCausalLM.register(...)
    from q_align.model.builder import load_pretrained_model
    from q_align.constants import IMAGE_TOKEN_INDEX
    from q_align.mm_utils import tokenizer_image_token
    return load_pretrained_model, IMAGE_TOKEN_INDEX, tokenizer_image_token


def tokenizer_and_processor(ckpt, repo):
    """只加载 tokenizer + 图像处理器（**不碰权重**），并算出口径量。"""
    _import_qalign(repo)
    from transformers import AutoTokenizer
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(ckpt)
    return tokenizer, image_processor


def level_ids_of(tokenizer):
    """官方 `scorer.py` 原文: [id_[1] for id_ in tokenizer(LEVELS)["input_ids"]]"""
    return [id_[1] for id_ in tokenizer(LEVELS)["input_ids"]]


def prompt_ids_of(tokenizer, IMAGE_TOKEN_INDEX, device="cpu"):
    # 局部导入：`tokenizer_image_token` 是 `_import_qalign` 的局部返回值，
    # 不能在模块级直接用（会被当成未定义名）。此处依赖调用方已把 repo 放进 sys.path。
    from q_align.mm_utils import tokenizer_image_token
    return tokenizer_image_token(
        PROMPT, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)


def build_model(ckpt: str, device: str, repo: str, dtype: str):
    """按官方 scorer.py 的路径加载权重。

    - dtype="fp16"（默认）：完全走官方 `load_pretrained_model(ckpt, None, "mplug_owl2", device=...)`，
      其内部写死 `torch_dtype=torch.float16`。
    - dtype="fp32"：**明确偏离官方**，改为直接用注册后的模型类以 float32 加载。
      ⚠️ 取 fp32 时**必须在缓存里留痕**（dtype 字段），且该次结果**不得**与 fp16 的直接互比。
    """
    load_pretrained_model, IMAGE_TOKEN_INDEX, _t = _import_qalign(repo)

    if dtype == "fp32":
        from transformers import AutoTokenizer
        from q_align.model import MPLUGOwl2LlamaForCausalLM
        print("  [warn] --dtype fp32：偏离官方（官方写死 fp16）。", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
            ckpt, low_cpu_mem_usage=True, torch_dtype=torch.float32,
            device_map={"": device})
    else:
        tokenizer, model, _ip_unused, _ctx = load_pretrained_model(
            ckpt, None, "mplug_owl2", device=device)

    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained(ckpt)

    level_ids = level_ids_of(tokenizer)
    input_ids = prompt_ids_of(tokenizer, IMAGE_TOKEN_INDEX, model.device)

    model.eval()
    return tokenizer, model, image_processor, level_ids, input_ids


def run_tokenizer_only(ckpt, repo, image):
    """`--tokenizer-only`：把**不依赖权重**的可核验项全部打出来后退出。

    核验对象是「最容易错且错了看不出来」的那几件事：
      ① 五级词的 token id 是否真取到「第 2 个」（第 1 个是 BOS）——官方是 id_[1]，
         若 BOS 不存在或分词结果与预期不同，这一下标就会静默取错。
      ② prompt 里 <|image|> 是否被替换成 IMAGE_TOKEN_INDEX(-200)、出现几次。
      ③ 图像预处理的口径（mean/std/size/crop）—— expand2square 的背景色由 mean 算出来。
    """
    tokenizer, image_processor = tokenizer_and_processor(ckpt, repo)
    _, IMAGE_TOKEN_INDEX, tokenizer_image_token = _import_qalign(repo)

    print("tokenizer      :", type(tokenizer).__name__, "| vocab =", tokenizer.vocab_size)
    print("IMAGE_TOKEN_INDEX = %d" % IMAGE_TOKEN_INDEX)
    print()

    print("① 五级词的逐词分词（官方取 id_[1]，即下表第 2 个 id）：")
    for w in LEVELS:
        ids = tokenizer(w)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print("   %-10s ids=%-18s tokens=%s" % (w, ids, toks))
    print("   ⇒ level_ids = %s" % level_ids_of(tokenizer))
    print()

    ids = prompt_ids_of(tokenizer, IMAGE_TOKEN_INDEX)
    n_img = int((ids == IMAGE_TOKEN_INDEX).sum())
    print("② prompt 的 input_ids：长度 %d，其中 IMAGE_TOKEN_INDEX 出现 %d 次" % (ids.shape[1], n_img))
    print("   前 8 个 id = %s" % ids[0, :8].tolist())
    print("   prompt 原文 = %r" % PROMPT)
    print("   解码回来（-200 换成 <image>）= %r"
          % tokenizer.decode([i if i != IMAGE_TOKEN_INDEX else 0 for i in ids[0].tolist()]))
    print()

    print("③ 图像预处理口径（CLIPImageProcessor）：")
    print("   image_mean  = %s  ⇒ expand2square 背景色 = %s"
          % (image_processor.image_mean,
             [int(x * 255) for x in image_processor.image_mean]))
    print("   image_std   = %s" % (image_processor.image_std,))
    print("   size        = %s" % (image_processor.size,))
    print("   crop_size   = %s" % (getattr(image_processor, "crop_size", None),))
    print("   do_center_crop = %s | do_resize = %s | do_rescale = %s"
          % (image_processor.do_center_crop, image_processor.do_resize,
             getattr(image_processor, "do_rescale", None)))
    print()

    if image and os.path.isfile(image):
        from PIL import Image
        img = Image.open(image).convert("RGB")
        bg = tuple(int(x * 255) for x in image_processor.image_mean)
        sq = _expand2square(img, bg)
        pv = image_processor.preprocess(sq, return_tensors="pt")["pixel_values"]
        print("④ 对 %s 实际预处理：" % os.path.basename(image))
        print("   原始 %s → expand2square %s → pixel_values %s  dtype=%s"
              % (img.size, sq.size, tuple(pv.shape), pv.dtype))
    print("\n[tokenizer-only] 以上全部不依赖权重，无卡模式（2 GiB）即可核验。")


def _expand2square(pil_img, background_color):
    """官方 scorer.py 的同名方法，逐字复刻。"""
    from PIL import Image
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def score_one(model, image_processor, level_ids, input_ids, device, img_path):
    """返回 (score 0-1, probs 五级 dict)。计算与官方 forward 逐字一致。"""
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    img = _expand2square(img, bg)
    with torch.inference_mode():
        pixel_values = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        # 官方在此 .half()；fp32 路径下 model 是 fp32，故跟随 model.dtype
        pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)
        logits = model(input_ids, images=pixel_values)["logits"][:, -1, level_ids]
        probs_t = torch.softmax(logits, -1)                       # ← 官方在加权前丢掉的中间量
        weight_t = torch.tensor(WEIGHTS, dtype=probs_t.dtype, device=probs_t.device)
        score = (probs_t @ weight_t).item()                       # ← 官方原样
    probs = probs_t.float().cpu().numpy()[0]                      # 缓存用高精度副本
    return float(score), dict(zip(LEVELS, [round(float(p), 6) for p in probs]))


def inspect_ckpt(ckpt: str):
    """打印分片与参数规模 —— 用于权重完整性核对（不加载模型）。"""
    import glob
    idx = os.path.join(ckpt, "pytorch_model.bin.index.json")
    print("ckpt 目录      :", ckpt)
    shards = sorted(glob.glob(os.path.join(ckpt, "*.bin")))
    total = 0
    for s in shards:
        n = os.path.getsize(s)
        total += n
        # 注意：不能写成 "%15,d"——`%` 旧式格式化**不支持**千分位 `,` 标志，
        # 会抛 ValueError: unsupported format character ','。用 str.format 的 ",d"。
        print("  %-42s %15s B (%.2f GiB)"
              % (os.path.basename(s), format(n, ",d"), n / 2 ** 30))
    print("分片合计       : %s B (%.2f GiB)" % (format(total, ",d"), total / 2 ** 30))
    if os.path.isfile(idx):
        meta = json.load(open(idx, encoding="utf-8"))
        wm = meta.get("weight_map", {})
        print("index.json     : %d 个参数键，metadata.total_size=%s"
              % (len(wm), meta.get("metadata", {}).get("total_size")))
    else:
        print("index.json     : 不存在 —— 本 checkpoint 可能是单文件 ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="图或目录；--inspect-ckpt / --tokenizer-only 时可省略")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default="q-future/q-align-iqa")
    ap.add_argument("--repo", default=os.environ.get("QALIGN_REPO", DEFAULT_REPO),
                    help="Q-Align GitHub 仓库根目录（提供 q_align 包，见 README）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="已存在则覆盖（默认拒绝覆盖缓存）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inspect-ckpt", action="store_true", help="只打印权重信息后退出")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="只加载 tokenizer + 图像处理器，打印 token 级口径后退出（**不碰权重**）。"
                         "★ 无卡模式（2 GiB）下只能用这条做验证，见 README")
    # ★ AutoDL「无卡模式」的 CPU 配额是 0.5 核，而容器里 `nproc` 报 112 ⇒ torch 默认起
    #   56 个线程去抢半个核。实测（2026-10-13，`bench_threads.py`，同一台无卡机）：
    #       threads=56: outer×300 = 21.501s   gemm2048×3 = 7.599s
    #       threads= 1: outer×300 =  0.108s   gemm2048×3 = 0.910s
    #   ⇒ **单线程快 199×（零碎小 op）/ 8.4×（真实 GEMM）**。模型构造阶段全是小 op，
    #   所以线程越多越慢，表现为「加载卡死」（实为一个也卡在
    #   `_set_cos_sin_cache` 这类微秒级算子上）。
    #   ⇒ 无卡模式**必须** --threads 1；有卡模式跑 GPU 推理时不传即可（走 torch 默认）。
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 的 CPU 线程数；0=不干预（保持 torch 默认）。"
                         "★ AutoDL 无卡模式请传 1 —— 否则 56 线程抢 0.5 核，慢约两个数量级")
    args = ap.parse_args()

    if args.threads > 0:
        _nt_before = torch.get_num_threads()      # 必须在 set 之前取，否则打印的是新值
        torch.set_num_threads(args.threads)
        print("[note] torch CPU 线程数 %d -> %d" % (_nt_before, args.threads))

    if args.inspect_ckpt:
        inspect_ckpt(args.ckpt)
        return

    if args.tokenizer_only:
        run_tokenizer_only(args.ckpt, args.repo, args.image)
        return

    if not args.image or not args.out:
        ap.error("--image 与 --out 在推理模式下是必需的")

    if os.path.exists(args.out) and not args.force:
        sys.exit("缓存已存在，拒绝覆盖（分数缓存一旦生成不要覆盖，新版本另起目录）：%s"
                 % args.out)

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
        imgs = [x for x in sorted(p.glob("*"))
                if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    else:
        imgs = [p]
    if args.limit:
        imgs = imgs[: args.limit]

    print("加载 %s 到 %s (dtype=%s) ..." % (args.ckpt, args.device, args.dtype))
    t0 = time.time()
    tokenizer, model, image_processor, level_ids, input_ids = build_model(
        args.ckpt, args.device, args.repo, args.dtype)
    dev = str(model.device)
    print("模型就绪，device=%s，加载耗时 %.1f s" % (dev, time.time() - t0))
    print("五级 token ids = %s  （顺序 %s）" % (level_ids, LEVELS))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            score, probs = score_one(model, image_processor, level_ids,
                                     input_ids, dev, img)
            f.write(json.dumps({"image": str(img), "score": round(score, 6),
                                "probs": probs, "device": dev, "dtype": args.dtype,
                                "seed": args.seed}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("  [%d/%d] %.4f  %s  (%.1f s)"
                  % (i, len(imgs), score, img.name, time.time() - t1))
    print("完成 %d 张，缓存 -> %s" % (n, out))


if __name__ == "__main__":
    main()
