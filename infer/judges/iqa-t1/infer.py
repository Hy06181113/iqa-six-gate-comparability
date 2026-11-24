# -*- coding: utf-8 -*-
r"""IQA-T1 (Qwen3-VL-4B-Instruct, HF `zibuyu-02/IQA-T1`) 推理封装。

    python infer.py --image <图或目录> --out <缓存.jsonl> [--limit N] [--list 名单.txt]
    python infer.py --inspect-ckpt                  # 只看权重/配置/上游指纹，不加载
    python infer.py --tokenizer-only                # 只验 token 与提示词口径，不碰权重

输出缓存字段（每条）:
    image, score, score_path, raw_response, tools_used, num_tool_calls,
    device, dtype, attn, seed, enable_tools, max_tool_calls, seconds

设计约束:
  - **不复刻工具管线，直接复用上游 `IQAInference` 类**。上游的工具函数在
    `scripts/tools.py`（28,811 B / 956 行 / 15 个分析工具），复刻必然漂移。
    本文件只做四件事：批处理循环、名单取图、缓存落盘、口径留痕。
    ★ 这是本文件与 `../deqa-score/infer.py` 的**根本差别**（那份是逐字复刻官方
      score() 以保留五级分布；这份是**原样调用**上游类，因为上游没有中间量可留）。
  - `--list` 按名单取图、不对目录 glob——校准结论只许在官方 test 子集上做。
  - **必须逐图保留 `raw_response` 与 `tools_used`** —— 见下方「两处必须留痕的理由」。
  - CLI 与其它评委保持一致（`--image --out --ckpt --device --dtype --list --limit --force --seed`）。

★ 官方推理协议（2026-10-13 一手核定，逐处可核）:
  `inference/infer.py:377-384`
      self.gen_config = GenerationConfig(
          do_sample=False, temperature=1.0, top_k=0, top_p=1.0,
          max_new_tokens=2560, pad_token_id=self.processor.tokenizer.pad_token_id)
  ⇒ **`do_sample=False` 即贪心**；`temperature` / `top_k` / `top_p` 在 `do_sample=False`
    下**全部失效**（HF 语义）。默认 `--seed 42`（`parse_args` 第 588-593 行），
    `__init__` 第 371 行调 `set_seed(seed)`。
  ⇒ 🔴 **「IQA-T1 是 9 个评委里唯一的非确定性评委」这一说法作废**：
    官方口径下它与其余 8 个一样是确定函数。
  ⚠️ **但「协议确定」≠「逐位可复现」**，本文件因此**不假定确定性**、而是**实测**它：
    ① 贪心在 GPU 上仍可能因 kernel 非确定性在近平分处翻转；
    ② 更要紧的是它有**工具调用的离散分支**（见下）——选错一个工具整条轨迹就分叉。
    ⇒ 判据是**两次运行的逐字节比对，且必须比对 `tools_used` 序列，而非只比分数**。

★ 本文件在官方口径之上**唯一增加的一层留痕**（不改任何行为）:
  ① `score_path` —— 记下 `extract_score` 走的是**哪条分支**：
       "json"     = 正则 `"score"\\s*:\\s*([0-9]+\.?[0-9]*)` 命中（**预期路径**）
       "fallback" = 正则未命中，回退取「回复里第一个落在 [1,5] 的数」（**可能抓错数**）
       "none"     = 两条都没命中，score = None
     官方 `inference/infer.py:306-318` 的回退分支**在工具返回的中间数值上可能抓错**，
     所以「回退命中的比例」必须单独报一个数。
  ② `tools_used` / `num_tool_calls` —— 上游 `generate_with_tools` 已经在返回值里给了
     （第 269 行 `return full_response, tools_used, tool_images`），**不是我们算的**。

★ 工具调用（本评委独有的第二层口径，影响预计远大于采样）:
  官方 `--max_tool_calls` 默认 **6**（第 605-610 行；注意 `generate_with_tools` 的
  形参默认是 5，但 `IQAInference` 与 CLI 都传 6 ⇒ 生效值 **6**）；
  `--enable_tools` 默认 True、`--no_tools` 可关闭（第 594-604 行）。
  🔴 **工具开关不是「同一输入换个策略」——它把图像预处理管路一起换掉**：
    开工具：`generate_with_tools` 第 215-221 行走
            `processor(text=[current_text], images=all_images, ...)`，
            `all_images` 是脚本自己 `PIL.Image.open(...).convert("RGB")` 出来的
            **原图 + 逐轮追加的工具图**，且每轮把一个
            `<|vision_start|><|image_pad|><|vision_end|>` 占位符追加进上下文；
    关工具：第 517-529 行走 `qwen_vl_utils.process_vision_info(messages)` 另一条管路。
  ⇒ 两条路的输入像素口径不同、token 数也不同。**本课题逐图记录
    `enable_tools` / `max_tool_calls`，并把它当「接口层 + 输入口径」的双重变更报告。**
  ⚠️ `qwen_vl_utils` **不在官方 `demo/requirements.txt` 里**（已逐行核对该文件，
     129 B，列了 gradio/torch/torchvision/transformers>=4.57.0/accelerate/Pillow/numpy/
     opencv-python-headless/scipy/matplotlib/tqdm，**无 qwen_vl_utils**）——性质同
     Q-Align 的 `icecream`。它**只在 `--no_tools` 路上被 import** ⇒ 走默认路径不装也能跑。

★ 分数量纲: **1–5**（官方系统提示原文 *"Score range: 1.00 to 5.00"*，
  `inference/infer.py:290`；输出格式 `<answer_start>{"score": X.XX}<answer_end>` 见 :288）。
  本文件**不做任何仿射变换**（理由同 `../deqa-score/infer.py`：量纲转换属指标层）。

★ 加载参数（上游写死，本文件唯一覆盖处，逐字对照）:
  上游 `inference/infer.py:442-453`:
      self.model = Qwen3VLForConditionalGeneration.from_pretrained(
          load_path,
          dtype=torch.bfloat16,                          # ⇒ 权重实际是 bf16
          attn_implementation="flash_attention_2",       # ⇒ 硬依赖 flash-attn
          device_map={"": 0} if self.device == "cuda" else None,
      )
      self.model.eval()
      self.processor = AutoProcessor.from_pretrained(load_path, fix_mistral_regex=True)
  ⚠️ `config.json` 里写的是 `dtype: float32`，与磁盘 9.65 GB 矛盾（fp32 应约 19 GB）
     ⇒ **以官方代码的 bf16 为准**，不得沿用 config 的声明。
  ⚠️ `flash_attention_2` 需要 `flash-attn` 包；租卡环境若没有会加载失败。
     **`demo/infer.py:444` 官方自己用的是 `sdpa`** ⇒ 回退到 `sdpa` **有官方先例**，
     不是本课题的发明。但 `--attn` 的默认值仍是官方 `flash_attention_2`，
     要改必须**显式传参**，且卡上跑时要由 `run_l4.sh` 打印一行响的说明。

★ 训练污染（判污染看「我们加载的那份权重见过哪些图」）: `train/grpo/EasyR1-IQA-T1/data/` 下的
  `train_koniq_7k_rl.json`（GRPO，7,046 条）与 SFT 的
  `koniq_training-multimodal-all.json`（11,315 条 / 3,540 张不同图）整份取回后，
  在我们官方 test 的 2,015 个文件名上原始文本匹配 = **0 命中** ⇒ **实测干净**。
  审计登记见评委元数据表。⚠️ 但**本机的 L4 复现仍只在 KonIQ-10k 上**，
  这是本机栈的公共局限，其余 8 个评委相同。
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

# Windows 控制台默认 GBK，而本文件的输出里有 GBK 装不下的字符（★ ⇒ ⚠ 🔴），
# 于是 `python infer.py --help` 会直接抛 UnicodeEncodeError: 'gbk' codec can't encode…
# 把标准流降到「编码不了就替换」——不改变既有输出，只是不再因为**打印**而崩。
# （同一个坑在 `../deqa-score/infer.py` 与 `40-data/scripts/download_tid2013.py` 上踩过。）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):                    # 非 TextIOWrapper / 已关闭
        pass


# --- 上游锚点（引用必须验真）-----------------------------------------------------
UPSTREAM_REPO = "https://github.com/zibuyu-02/IQA-T1"
UPSTREAM_SHA = "3cabcdae2edabb40e0ffa4f5f85efbe1b7762f0d"
# 路径 -> (字节数, sha256)。字节数取自 GitHub Trees API（`?recursive=1`，truncated=false，
# 406 项），sha256 是 2026-10-13 经 jsDelivr 镜像取回同一 commit 的实测值——**两者互证**
# （四个文件的字节数逐一与 API 相符 ⇒ 镜像副本与仓库逐字节相同）。
UPSTREAM_FILES = {
    "inference/infer.py": (23306, "0b87e849ca9b282b215ec82ef6ea5bd3a871114a1aa3ee39f692cee8fff31e1d"),
    "scripts/tools.py":   (28811, "ba391a5ee90ad3bd8553dcaf06bfba5487f78535c0dc3b499cc40a6861bf5189"),
}
# 供 `--tokenizer-only` 打印用；同上，字节数与 API 相符。
UPSTREAM_FILES_EXTRA = {
    "demo/infer.py": (23310, "b3fd35e0cb7217fabb3e01e2c4af049b998e6aee4a1f68e6fc1cdc65ce5928ec"),
}

HF_REPO = "zibuyu-02/IQA-T1"
DEFAULT_CKPT = os.environ.get("IQAT1_CKPT", "/root/autodl-tmp/models/iqa-t1")
DEFAULT_REPO = os.environ.get("IQAT1_REPO", "/root/autodl-tmp/IQA-T1")

# 上游 `_load_model` 写死的注意力后端（:445）。默认照抄，改必须显式传 --attn。
OFFICIAL_ATTN = "flash_attention_2"


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# --- 上游模块的加载（唯一入口）--------------------------------------------------
def load_upstream(repo):
    """把上游 `inference/infer.py` 作为模块加载并返回。

    ★ 为什么是 import 而不是复刻：上游的工具管线在 `scripts/tools.py`（956 行 /
      15 个工具 / 依赖 cv2 + matplotlib），复刻必然漂移。**原样调用是唯一不漂移的做法。**

    ⚠️ 上游在模块顶层插 `scripts/` 进 `sys.path` 再 `from tools import ...`
      （`inference/infer.py:23-29`）⇒ **必须在克隆下来的完整仓库目录上加载**，
      单独拷一个 `infer.py` 出来是不行的。
    """
    entry = os.path.join(repo, "inference", "infer.py")
    if not os.path.isfile(entry):
        sys.exit("找不到上游入口 %s。\n  先 clone 仓库：见 README「怎么拿到上游代码」/ "
                 "setup_env.sh 的 repo 阶段。" % entry)
    spec = importlib.util.spec_from_file_location("_iqat1_upstream_infer", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                     # 上游内部可能自引用
    spec.loader.exec_module(mod)
    return mod


def verify_upstream(repo, verbose=True):
    """比对上游文件指纹 —— 口径漂移检查。

    ★ 这是「实验必须可复现」在**上游代码**这一层的落点：本课题的分数不只取决于权重，
      也取决于这份 `scripts/tools.py` 里 15 个工具的实现。上游改了而我们没察觉，
      比我们自己写错更难发现。**指纹不符即拒绝继续**（除非显式 --allow-drift）。
    """
    bad = []
    for rel, (n_expect, sha_expect) in UPSTREAM_FILES.items():
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            print("  [X] 缺文件 %s" % rel)
            bad.append(rel)
            continue
        n = os.path.getsize(p)
        sha = _sha256_of(p)
        ok = (n == n_expect and sha == sha_expect)
        if verbose or not ok:
            print("  %s %-20s %9d B (期望 %d)  sha256=%s%s"
                  % ("✅" if ok else "***", rel, n, n_expect, sha[:16],
                     "" if ok else "  期望 " + sha_expect[:16]))
        if not ok:
            bad.append(rel)
    return bad


# --- `extract_score` 的逐字复刻 + 分支判定 --------------------------------------
# 上游 `inference/infer.py:306-318` 原文：
#     m = re.search(r'"score"\s*:\s*([0-9]+\.?[0-9]*)', response_text)
#     if m: return float(m.group(1))
#     nums = re.findall(r'([0-9]+\.?[0-9]*)', response_text)
#     for n in nums:
#         v = float(n)
#         if 1 <= v <= 5: return v
#     return None
_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]+\.?[0-9]*)')
_NUM_RE = re.compile(r"([0-9]+\.?[0-9]*)")


def extract_score_verbose(text):
    """返回 (score, path)。`path` ∈ {"json", "fallback", "none"}。

    ★ 与上游分叉的唯一处是**多返回一个分支名**；分数本身与上游逐字同逻辑，
      并在运行时由 `_assert_extract_matches_upstream` 对拍验证（拿上游的真函数比）。
    """
    m = _SCORE_RE.search(text)
    if m:
        return float(m.group(1)), "json"
    for n in _NUM_RE.findall(text):
        v = float(n)
        if 1 <= v <= 5:
            return v, "fallback"
    return None, "none"


def _assert_extract_matches_upstream(up, samples):
    """把本文件的 `extract_score_verbose` 与上游真函数在若干用例上对拍。

    ★ 对拍的是**分数**，不是分支名（上游不返回分支名）。
      **任一处不等即退出** —— 说明上游改过 `extract_score`，我们的留痕口径已漂移。
    """
    bad = 0
    for s in samples:
        ours = extract_score_verbose(s)[0]
        theirs = up.extract_score(s)
        if ours != theirs:
            print("  *** 对拍失败：输入 %.60r ⇒ 本文件 %r / 上游 %r" % (s, ours, theirs))
            bad += 1
    if bad:
        sys.exit("*** extract_score 与上游不一致 —— 口径已漂移，禁止继续 ***")
    print("  对拍 extract_score          : ✅ %d 个用例（含 json 命中 / 回退 / 无解）分数一致"
          % len(samples))


# 用例集：覆盖三条分支，含**回退分支会抓错**的那种（分数写在文本中间）。
# ⚠️ 最后一条是「回退分支的危险」的活证据：格式没按 JSON 给时，回退只能扫第一个落在
#    [1,5] 的数——那里 `12` 被跳过、`5` 被当成分数，**真值 5.00 只是碰巧**。
EXTRACT_CASES = [
    '<answer_start>{"score": 3.46}<answer_end>',
    'blah <think>2.5 looks sharp</think> <answer_start>{"score": 4.12}<answer_end> tra',
    'the image scores 4.25 overall',                       # 回退：4.25
    'I rate 1.75 after checking',                          # 回退：1.75
    'analysis done. The score is 12 out of 5',             # 回退：12 被跳过，抓 5.0（碰巧）
    '<answer_start>{"score": 4.99}<answer_end> 对比度偏 12%',  # json 命中，且尾部有干扰数
]


# --- 唯一覆盖上游的一处：加载参数 ------------------------------------------------
def make_inference_class(up):
    """产出 `IQAInference` 的子类，只重写 `_load_model`。

    上游 `_load_model`（`inference/infer.py:436-455`）其余各行**逐字保留**，
    只把两个写死的加载参数变成可指定的。**除这两处，行为与上游完全相同。**
    """

    class IQAT1Inference(up.IQAInference):
        def __init__(self, *a, attn=OFFICIAL_ATTN, dtype_name="bf16", **kw):
            self._attn = attn
            self._dtype_name = dtype_name
            super().__init__(*a, **kw)

        def _load_model(self):
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            print("[1/3] Loading model from %s ..." % self.model_path)
            load_path = self._resolve_load_path()

            t_dtype = torch.bfloat16 if self._dtype_name == "bf16" else torch.float16
            if t_dtype is not torch.bfloat16:
                print("  [warn] --dtype %s：**偏离官方**（上游写死 bfloat16）。"
                      "该次结果不得与 bf16 的直接互比。" % self._dtype_name,
                      file=sys.stderr)

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                load_path,
                dtype=t_dtype,
                attn_implementation=self._attn,
                device_map={"": 0} if self.device == "cuda" else None,
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                load_path,
                fix_mistral_regex=True,
            )
            print("Model loaded.")

    return IQAT1Inference


# --- 不依赖权重的核验 -----------------------------------------------------------
def inspect_ckpt(ckpt, repo):
    """打印权重分片、config、preprocessor 与**上游指纹** —— 纯文件读取，不加载模型。

    无卡模式（AutoDL 0.5 核 + 2 GiB）就能跑，是本文件在取数前能做的第一层核对。
    """
    print("ckpt 目录      :", ckpt)
    print("上游仓库       :", repo)
    print("上游 pin       : %s @ %s" % (UPSTREAM_REPO, UPSTREAM_SHA))

    print("\n--- 上游文件指纹（口径漂移检查）---")
    bad = verify_upstream(repo)

    if not os.path.isdir(ckpt):
        print("\n[skip] ckpt 目录不存在（%s）—— 无卡阶段若权重还没下，这一半跳过是正常的。" % ckpt)
        print("\n--- 本文件使用的常量（须与上游一致）---")
        print("  贪心协议       = GenerationConfig(do_sample=False, max_new_tokens=2560)  "
              "（上游 :377-384）")
        print("  量纲           = 1–5（上游系统提示 :290 \"Score range: 1.00 to 5.00\"）")
        print("  工具默认       = enable_tools=True / max_tool_calls=6（上游 :594-610）")
        print("  默认注意力后端 = %s（上游 :445；demo/infer.py:444 官方用 sdpa）" % OFFICIAL_ATTN)
        if bad:
            sys.exit("*** 上游文件指纹不符或缺文件 —— 口径不可确认，禁止继续"
                     "（除非确认后显式 --allow-drift）***")
        return

    print("\n--- 权重文件清单 ---")
    for pat in ("*.safetensors", "*.bin", "*.json", "*.model", "*.txt", "*.jinja"):
        for f in sorted(glob.glob(os.path.join(ckpt, pat))):
            n = os.path.getsize(f)
            print("  %-46s %15s B (%.3f GiB)"
                  % (os.path.basename(f), format(n, ",d"), n / 2 ** 30))
    for pat in ("*.safetensors", "*.bin"):
        shards = sorted(glob.glob(os.path.join(ckpt, pat)))
        if shards:
            tot = sum(os.path.getsize(s) for s in shards)
            print("  %s 合计: %d 片 / %s B (%.2f GiB)"
                  % (pat, len(shards), format(tot, ",d"), tot / 2 ** 30))

    for ix in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(ckpt, ix)
        if os.path.isfile(p):
            meta = json.load(open(p, encoding="utf-8"))
            wm = meta.get("weight_map", {})
            print("  %s: %d 个参数键, metadata.total_size=%s"
                  % (ix, len(wm), meta.get("metadata", {}).get("total_size")))

    cfg_p = os.path.join(ckpt, "config.json")
    if os.path.isfile(cfg_p):
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        print("\n--- config.json ---")
        for k in ("_name_or_path", "architectures", "model_type", "dtype", "torch_dtype",
                  "transformers_version", "tie_word_embeddings"):
            print("  %-22s = %r" % (k, cfg.get(k)))
        print("  auto_map               = %s"
              % ("（无，架构是 transformers 原生的 Qwen3VL，不需要 trust_remote_code）"
                 if not cfg.get("auto_map") else json.dumps(cfg["auto_map"])))
        tc = cfg.get("text_config") or {}
        vc = cfg.get("vision_config") or {}
        print("  text_config: hidden=%s layers=%s vocab=%s"
              % (tc.get("hidden_size"), tc.get("num_hidden_layers"), tc.get("vocab_size")))
        print("  vision_config: depth=%s hidden=%s patch=%s"
              % (vc.get("depth"), vc.get("hidden_size"), vc.get("patch_size")))
        # ★ 这处矛盾必须打出来，否则下一个读 config 的人会以为权重是 fp32。
        declared = cfg.get("dtype") or cfg.get("torch_dtype")
        if declared in ("float32", "fp32"):
            print("  🔴 **矛盾**：config 声明 dtype=%s（fp32 应约 19 GB），但上游代码写死 "
                  "bfloat16\n     （`inference/infer.py:444`）⇒ **以代码为准**。本封装默认 "
                  "--dtype bf16。" % declared)

    pp = os.path.join(ckpt, "preprocessor_config.json")
    if os.path.isfile(pp):
        d = json.load(open(pp, encoding="utf-8"))
        print("\n--- preprocessor_config.json ---")
        for k in ("image_processor_type", "patch_size", "merge_size", "temporal_patch_size",
                  "min_pixels", "max_pixels", "do_resize", "do_rescale", "do_normalize"):
            if k in d:
                print("  %-22s = %r" % (k, d[k]))
        mp, xp = d.get("min_pixels"), d.get("max_pixels")
        if mp and xp:
            patch = (d.get("patch_size") or 16) * (d.get("merge_size") or 2)
            print("  ⚠️ **图像 token 数随输入分辨率变**：patch/merge = %d ⇒ "
                  "token 数 ≈ 宽×高 / %d²\n     min_pixels=%s / max_pixels=%s 会把它夹住。"
                  % (patch, patch, mp, xp))
            print("     ⇒ **本课题 L4 走 512×384 的缩略图**，其 token 数与官方 demo 的原始尺寸"
                  "**不同**，\n        这是输入口径必须与其它评委一并钉死的一处（双口径）。")

    print("\n--- 本文件使用的常量（须与上游一致）---")
    print("  贪心协议       = GenerationConfig(do_sample=False, max_new_tokens=2560)  "
          "（上游 :377-384）")
    print("  量纲           = 1–5（上游系统提示 :290 \"Score range: 1.00 to 5.00\"）")
    print("  工具默认       = enable_tools=True / max_tool_calls=6（上游 :594-610）")
    print("  默认注意力后端 = %s（上游 :445；demo/infer.py:444 官方用 sdpa）" % OFFICIAL_ATTN)
    print("\n[inspect-ckpt] 以上**全部不依赖权重**，无卡模式（2 GiB）即可核验。")
    if bad:
        sys.exit("*** 上游文件指纹不符 —— 口径已漂移，禁止继续（除非确认后显式 --allow-drift）***")


def run_tokenizer_only(up, ckpt):
    """`--tokenizer-only`：把**不依赖权重**的 token / 提示词口径全部打出来后退出。

    核验对象是「最容易错且错了看不出来」的那几件事：
      ① 上游的 `SYSTEM_PROMPT` / `USER_PROMPT` **原文**（我们的 messages 必须逐字一致）。
      ② chat template 渲染后的**完整提示词字符串**——这是真正喂给模型的输入。
      ③ `<|vision_start|><|image_pad|><|vision_end|>` 占位符与每个占位符展开的 token 数。
      ④ 15 个工具名与 `TOOL_PATTERN`（工具分支的触发面）。
      ⑤ processor 的 min/max_pixels（决定图像 token 数，随分辨率变）。
    """
    from transformers import AutoProcessor

    print("上游 SYSTEM_PROMPT 长度 = %d 字符" % len(up.SYSTEM_PROMPT))
    print("上游 USER_PROMPT       = %r" % up.USER_PROMPT)
    print("上游 ALL_TOOL_NAMES    = %d 个" % len(up.ALL_TOOL_NAMES))
    for i, n in enumerate(up.ALL_TOOL_NAMES, 1):
        print("   %2d. %s" % (i, n))
    print("上游 TOOL_PATTERN      = %s" % up.TOOL_PATTERN.pattern[:70] + " …")
    print("   ⇒ 回退分支的危险面：工具返回的任何 1–5 的数都可能被 `extract_score` 抓走")
    print()

    proc = AutoProcessor.from_pretrained(ckpt, fix_mistral_regex=True)
    tk = proc.tokenizer
    print("processor      : %s" % type(proc).__name__)
    print("tokenizer      : %s | vocab = %s | pad = %s | bos = %s"
          % (type(tk).__name__, tk.vocab_size, tk.pad_token_id, tk.bos_token_id))
    ip = getattr(proc, "image_processor", None)
    if ip is not None:
        for k in ("min_pixels", "max_pixels", "patch_size", "merge_size"):
            if hasattr(ip, k):
                print("  image_processor.%s = %r" % (k, getattr(ip, k)))
    print()

    # ② 与上游 `_load_model` 之后的 `infer()` 第 482-491 行**逐字同构**的 messages。
    #    ⚠️ 图片用 `file://` 前缀；渲染成文本不需要那张图真的存在。
    probe = "/probe/does-not-need-to-exist.jpg"
    messages = [
        {"role": "system", "content": up.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": "file://" + probe},
            {"type": "text", "text": up.USER_PROMPT},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("② chat template 渲染后的**完整输入**（%d 字符，逐字）：" % len(text))
    print("----------------------------------------------------------")
    print(text)
    print("----------------------------------------------------------")
    print("   ⚠️ 须确认：① 系统提示是**原文**、未被模板改写；② 图片占位符出现且只出现一次；")
    print("              ③ 结尾是 assistant 的起手（add_generation_prompt=True 的效果）。")
    print()

    vs, ve, pad = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
    ids = tk(text, return_tensors=None)["input_ids"]
    print("③ 提示词 tokens = %d 个" % len(ids))
    for name, tok in (("vision_start", vs), ("image_pad", pad), ("vision_end", ve)):
        print("   %-12s 在文本中出现 %d 次 / 在 token 表中存在=%s"
              % (name, text.count(tok), tok in tk.get_vocab()))
    print("   ⚠️ `<|image_pad|>` 的**实际展开数**由 processor 按图像尺寸算（见上 min/max_pixels），")
    print("      因此**不在本命令里给出**——要真实的数，须在卡上对一张真图跑一次 processor。")
    print()
    print("[tokenizer-only] 以上全部不依赖权重，无卡模式（2 GiB）即可核验。")
    print("[tokenizer-only] ⚠️ 它**不能**替代 `run_l4.sh` 第 [2] 段的两次运行确定性实测。")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="图或目录；--inspect-ckpt / --tokenizer-only 时可省略")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="IQA-T1 权重目录（HF 仓 %s 的 snapshot）。默认走 $IQAT1_CKPT" % HF_REPO)
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="IQA-T1 **上游仓库**的本地 clone 路径（需要其中的 inference/ 与 "
                         "scripts/）。默认走 $IQAT1_REPO")
    ap.add_argument("--device", default="cpu")
    # ★ 默认 bf16 = 官方（上游 :444 写死）。fp16 只为本机冒烟留，且会打警告。
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    # ★ 默认 = 官方写死的 flash_attention_2；sdpa 有官方先例（demo/infer.py:444）。
    ap.add_argument("--attn", default=OFFICIAL_ATTN,
                    choices=[OFFICIAL_ATTN, "sdpa", "eager"])
    ap.add_argument("--list", default=None,
                    help="图片名单文件（每行一个文件名）。给定则只跑名单里的图，"
                         "不再 glob 目录——用于官方 train/test 子集")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="已存在则覆盖（默认拒绝覆盖缓存）")
    ap.add_argument("--seed", type=int, default=42, help="官方默认 42（上游 :590）")
    ap.add_argument("--no-tools", action="store_true",
                    help="关闭工具调用（**上游叫 --no_tools**，:600-604）。"
                         "⚠️ 关掉会同时换掉图像预处理管路，见文件头说明")
    ap.add_argument("--max-tool-calls", type=int, default=6,
                    help="最大工具调用轮数（官方默认 6，上游 :605-610）")
    ap.add_argument("--tool-save-dir", default=None,
                    help="工具图保存目录。默认 = <out> 同级的 <out 名>.tools/，"
                         "**每次运行用新目录**以免复用上一轮的陈旧工具图")
    ap.add_argument("--inspect-ckpt", action="store_true", help="只打印权重与上游指纹后退出")
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="只加载 processor，打印 token / 提示词口径后退出（**不碰权重**）")
    ap.add_argument("--allow-drift", action="store_true",
                    help="上游指纹不符时仍继续（**须先人工确认漂移内容**，否则别用）")
    # ★ 无卡模式的 CPU 配额是 0.5 核，而容器里 `nproc` 报的是 host 的核心数
    #   （Q-Align 实测：56 线程抢 0.5 核比单线程**慢 199×**，表现为「加载卡死」）。
    #   ⇒ 无卡模式一律 --threads 1；有卡模式下要跑 GPU 前向时不必传（走 torch 默认）。
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 的 CPU 线程数；0=不干预（保持 torch 默认）。"
                         "★ AutoDL 无卡模式请传 1")
    args = ap.parse_args()

    if args.threads > 0:
        import torch
        _n = torch.get_num_threads()
        torch.set_num_threads(args.threads)
        print("[note] torch CPU 线程数 %d -> %d" % (_n, args.threads))

    # ★ `--inspect-ckpt` 必须在 load_upstream **之前**——它本身就要报「仓库在不在、
    #   指纹对不对」，若先 import 上游，仓库没 clone 时会先 exit、什么也看不到。
    if args.inspect_ckpt:
        inspect_ckpt(args.ckpt, args.repo)
        return

    up = load_upstream(args.repo)

    if args.tokenizer_only:
        run_tokenizer_only(up, args.ckpt)
        return

    if not args.image or not args.out:
        ap.error("--image 与 --out 在推理模式下是必需的")

    # 上游指纹：推理前先验一次。漂移就停——本课题的分数也取决于那份 tools.py。
    bad = verify_upstream(args.repo, verbose=False)
    if bad:
        if not args.allow_drift:
            print("  [X] 上游文件指纹不符：%s" % ", ".join(bad))
            print("      ⇒ 先用 --inspect-ckpt 看清差异，确认后再显式 --allow-drift。")
            sys.exit("上游代码已漂移，拒绝继续")
        print("  [warn] --allow-drift：上游指纹不符但被要求继续，**本次结果须标注**",
              file=sys.stderr)
    else:
        print("上游指纹       : ✅ 与 pin %s 相符" % UPSTREAM_SHA[:12])

    if os.path.exists(args.out) and not args.force:
        sys.exit("缓存已存在，拒绝覆盖（分数缓存一旦生成不要覆盖，新版本另起目录）：%s"
                 % args.out)

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
        print("  ⚠️ 未给 --list，正在 glob 整个目录（%d 张）。**若该目录含训练集图，"
              "本次结果不得用于校准结论**。" % len(imgs), file=sys.stderr)
    else:
        imgs = [p]
    if args.limit:
        imgs = imgs[: args.limit]

    enable_tools = not args.no_tools
    tool_dir = args.tool_save_dir
    if tool_dir is None and enable_tools:
        tool_dir = str(pathlib.Path(args.out).with_suffix("")) + ".tools"

    print("=" * 72)
    print("IQA-T1 推理封装（调用上游 IQAInference，不改其行为）")
    print("  ckpt        : %s" % args.ckpt)
    print("  device      : %s | dtype=%s | attn=%s" % (args.device, args.dtype, args.attn))
    print("  seed        : %d" % args.seed)
    print("  tools       : %s%s" % ("enabled" if enable_tools else "DISABLED",
                                   ("  max_tool_calls=%d" % args.max_tool_calls)
                                   if enable_tools else ""))
    print("  工具图目录  : %s" % tool_dir)
    print("  共 %d 张图" % len(imgs))
    print("=" * 72)

    print("--- 与上游对拍（口径漂移检查）---")
    _assert_extract_matches_upstream(up, EXTRACT_CASES)

    if args.attn != OFFICIAL_ATTN:
        print("  ⚠️ 注意力后端 = %s，**偏离官方写死的 %s**。"
              "官方 demo 入口用的是 sdpa，故这不是本课题的发明；"
              "但本次结果须记明后端（缓存里已存 attn 字段）。"
              % (args.attn, OFFICIAL_ATTN))

    print("加载权重 ...")
    t0 = time.time()
    cls = make_inference_class(up)
    eng = cls(model_path=args.ckpt, device=args.device, seed=args.seed,
              enable_tools=enable_tools, max_tool_calls=args.max_tool_calls,
              tool_save_dir=tool_dir, keep_tool_images=True,
              attn=args.attn, dtype_name=args.dtype)
    print("模型就绪，加载耗时 %.1f s" % (time.time() - t0))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_json = 0
    n_fb = 0
    n_none = 0
    n_err = 0
    n_tools = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            try:
                r = eng.infer(str(img))
            except Exception as e:                                  # 不吞：记进缓存再继续
                print("  [%d/%d] *** 异常：%s: %s" % (i, len(imgs), type(e).__name__, e),
                      file=sys.stderr)
                f.write(json.dumps({"image": str(img), "score": None,
                                    "score_path": "error", "error": "%s: %s"
                                    % (type(e).__name__, e),
                                    "device": args.device, "dtype": args.dtype,
                                    "attn": args.attn, "seed": args.seed,
                                    "enable_tools": enable_tools,
                                    "max_tool_calls": args.max_tool_calls,
                                    "seconds": round(time.time() - t1, 3)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                n_err += 1
                continue

            # ★ 用我们这份带分支名的版本再算一次（已与上游对拍过），
            #   只为把「走的是哪条分支」记下来；分数与上游 `r["score"]` 相同。
            score, path_kind = extract_score_verbose(r["raw_response"])
            if score != r["score"]:
                print("  [warn] 本文件与上游取分不一致：%r vs %r（以本文件为准并留痕）"
                      % (score, r["score"]), file=sys.stderr)

            rec = {
                "image": str(img),
                "score": score,
                "score_path": path_kind,
                "raw_response": r["raw_response"],
                "tools_used": r["tools_used"],
                "num_tool_calls": r["num_tool_calls"],
                "device": args.device, "dtype": args.dtype, "attn": args.attn,
                "seed": args.seed, "enable_tools": enable_tools,
                "max_tool_calls": args.max_tool_calls,
                "seconds": round(time.time() - t1, 3),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            n_json += (path_kind == "json")
            n_fb += (path_kind == "fallback")
            n_none += (path_kind == "none")
            n_tools += len(r["tools_used"])
            print("  [%d/%d] %-8s %s  tools=%s  (%.1f s)"
                  % (i, len(imgs), ("%s" % score), img.name,
                     ",".join(r["tools_used"]) if r["tools_used"] else "-",
                     time.time() - t1))

    n_ok = n - n_err
    print("\n完成 %d 张（其中异常 %d 张），缓存 -> %s" % (n, n_err, out))
    print("  score_path 统计：json=%d / ⚠️ fallback=%d / none=%d / error=%d"
          % (n_json, n_fb, n_none, n_err))
    if n_ok:
        print("  平均工具调用数 = %.2f 次/图" % (n_tools / n_ok))
    print("  ⚠️ fallback 比例不为 0 即说明有图的回复没按 `<answer_start>{\"score\": …}` 格式，"
          "该数必须写进实验记录（回退分支在工具数值上可能抓错）。")


if __name__ == "__main__":
    main()
