# -*- coding: utf-8 -*-
"""DeQA-Score (mPLUG-Owl2-7B, zhiyuanyou/DeQA-Score-Mix3) 推理封装。

    python infer.py --image <图或目录> --out <缓存.jsonl> [--limit N] [--list 名单.txt]
    python infer.py --inspect-ckpt                  # 只看权重分片与 config，不加载
    python infer.py --tokenizer-only [--image 图]   # 只验 token 级口径，不碰权重

输出缓存字段（每条）:
    image, score, probs{excellent,good,fair,poor,bad}, logits{...},
    device, dtype, seed

设计约束:
  - **必须缓存五级概率分布**，不能只存标量（C12 的可靠性估计要用到分布）。
    官方的 `model.score()` 在 `@ weight_tensor` 之后就只剩标量、**分布被丢掉**，
    所以本文件**逐字复刻 score() 的计算**（见下），只多做一件事：
    把加权**之前**的 softmax 结果留下来。这是本文件与官方唯一的功能差异。
  - `--list` 按名单取图、不对目录 glob——校准结论只许在官方 test 子集上做。
  - CLI 与其它评委保持一致（`--image --out --ckpt --device --dtype --list --limit --force --seed`）。

★ 加载方式（2026-10-13 核定，见 README.md）:
  走 HF 仓的 `trust_remote_code=True` —— **不需要 clone GitHub 仓库**。
  依据（全部一手取回）:
    · `config.json` 的 `auto_map` 完整；
    · `modeling_mplug_owl2_huggingface.py:378-379` 自注册
      `AutoConfig.register("mplug_owl2", …)` / `AutoModelForCausalLM.register(…)`；
    · 该文件**自足**：`IMAGE_TOKEN_INDEX`(37 行) / `DEFAULT_IMAGE_TOKEN`(38 行) /
      `tokenizer_image_token`(41 行) / `expand2square`(62 行) 全在它自己里定义，
      **不依赖 `src/` 包**；
    · 仓内自带 `preprocessor_config.json`。
  对比：`q-future/q-align-iqa` 的 config.json **没有** auto_map、仓内也**没有** `modeling_*.py`
  ⇒ 那条路必须 clone Q-Align 仓库（见 `../q-align/README.md`）。**两者是不同的加载方式，不是同一个坑。**

★ 两处会静默出错的坑（本文件已处理，见 README「加载方式」）:
  ① `modeling_mplug_owl2_huggingface.py:256-257` 用 `config._name_or_path` 去
     `AutoTokenizer.from_pretrained()` / `CLIPImageProcessor.from_pretrained()`，
     而该字段是**hub id** `zhiyuanyou/DeQA-Score-Mix3` ⇒ 不覆盖就会联网。
     处置：加载 config 后把它改成本地目录，再去加载模型；并靠 `HF_HUB_OFFLINE=1`
     把「覆盖有没有生效」变成一次**会响的检查**而不是看不见的假设。
  ② 官方 demo 图标称分数 = **1.9404**（模型卡 README，[1,5] 制）。
     这是本课题**唯一一个有一手出处的 demo 判据**，它同时覆盖
     prompt + tokenizer + 预处理 + 权重四段链路 —— 任一段口径错都会让它偏掉。
     `run_l4.sh` 的 [2] 段以它作硬门槛。

★ 分数量纲:
  本文件输出的 score 是 **1–5**（权重 `[5,4,3,2,1]`），即 DeQA 的原生量纲。
  ⚠️ 与 `../q-align/infer.py` 的 **0–1** 不同：两者是**仿射关系**
  `score(1-5) = 4 * score(0-1) + 1`。**引用数字时必须带量纲。**
  ⚠️ **这里刻意不做仿射变换**：量纲转换属指标层，不属「图 → 分数」的封装层；
  且本课题的命题恰恰是「分数尺度不能假定可比」，若在此折成 0–1，等于把
  「两者的仿射关系成立」这个**待检验假设预先写死进数据**。
  （L4 的 SRCC/PLCC 对仿射不变，此选择不影响任何既有指标。）

★ 训练污染: 模型卡原文 Mix3 是
  「fully fine-tuned on **KonIQ, SPAQ, and KADID** datasets」⇒ 三个数据集上都被污染。
  详见评委元数据表（那里把「实测」与「推定」分开写）。
"""
import argparse
import glob
import json
import os
import pathlib
import sys
import time

import torch

# Windows 控制台默认 GBK，而本文件的输出里有 GBK 装不下的字符（★ ⇒ ⚠），
# 于是 `python infer.py --help` 会直接抛 UnicodeEncodeError: 'gbk' codec can't encode…
# 把标准流降到「编码不了就替换」——不改变既有输出，只是不再因为**打印**而崩。
# （同一个坑在 `40-data/scripts/download_tid2013.py` 上踩过。）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):                    # 非 TextIOWrapper / 已关闭
        pass


# --- 与官方 modeling_mplug_owl2_huggingface.py 逐字一致的常量 ------------------
# 出处: zhiyuanyou/DeQA-Score-Mix3 仓内 modeling_mplug_owl2_huggingface.py
#       第 37/38/41/62 行（token 与几何工具）、第 256-291 行（__init__ 与 score()）
#       （2026-10-13 经 hf-mirror.com 一手取回；当时留的本机副本已随同日 `C:\tmp` 清理删除，
#         可按同一渠道重取——**结论不依赖该副本**，见 judges/deqa-score/README.md）
IMAGE_TOKEN_INDEX = -200                       # 37 行
DEFAULT_IMAGE_TOKEN = "<|image|>"              # 38 行
LEVELS = ["excellent", "good", "fair", "poor", "bad"]     # __init__ 里逐字
WEIGHTS = [5.0, 4.0, 3.0, 2.0, 1.0]            # ★ 官方 [5.,4.,3.,2.,1.]，1-5 制
TASK_, INPUT_ = "quality", "image"             # score() 的默认实参
# score() 第 277 行 f-string 在默认实参下的结果（**逐字符**，含 "\n" 与结尾的 " is"）
PROMPT = ("USER: How would you rate the quality of this image?\n"
          "<|image|>\n"
          "ASSISTANT: The quality of the image is")

DEFAULT_CKPT = os.environ.get("DEQA_CKPT", "/root/autodl-tmp/models/deqa-score-mix3")
HF_REPO = "zhiyuanyou/DeQA-Score-Mix3"


# --- 官方工具函数的逐字复刻 ---------------------------------------------------
# ★ 为什么复刻而不是 import：这两个函数定义在 HF 仓的 remote-code 模块里，
#   而那个模块名是 huggingface_hub 在运行时动态生成的。复刻 + 运行期对拍
#   （见 `_assert_tools_match_official`）比 import 稳，也比「相信上游没改」稳。
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX,
                          return_tensors=None):
    """官方 `modeling_mplug_owl2_huggingface.py:41` 的**逐字**复刻。

    行为要点：把 prompt 按 `<|image|>` 切成若干 chunk 分别分词，再用
    `image_token_index` 当分隔符拼回去。**第一段若以 BOS 开头，offset=1**
    ——即 BOS 保留一次、且分隔符写两次（`[sep] * (offset+1)`），
    随后每段切掉自己的前 offset 个 id 以免 BOS 重复。这个 offset 逻辑是
    官方特有的一处细节，写错会让 input_ids 长度对但**内容错位**，且不报错。
    """
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else []
                     for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == tokenizer.bos_token_id):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError("Unsupported tensor type: %s" % return_tensors)
    return input_ids


def expand2square(pil_img, background_color):
    """官方 `modeling_mplug_owl2_huggingface.py:62` 的逐字复刻。

    长边撑成正方形、短边居中填充，**背景色由 image_mean × 255 算出**
    （调用处见 score() 第 277 行）。
    """
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


def _official_module_of(model):
    """取回加载本模型的那个 remote-code 模块，用于与官方实现**对拍**。

    取不到返回 None（不抛）——对拍是加分项，不该因为拿不到模块就让推理失败。
    """
    try:
        return sys.modules[type(model).__module__]
    except KeyError:
        return None


def _assert_tools_match_official(model, prompt_ids, tokenizer):
    """把「我们复刻的」与「官方模块里的」逐个对拍。**这是本文件最值钱的一段。**"""
    mod = _official_module_of(model)
    if mod is None:
        print("  [warn] 取不到 remote-code 模块，跳过与官方实现对拍", file=sys.stderr)
        return
    ok = True

    if hasattr(mod, "tokenizer_image_token"):
        theirs = mod.tokenizer_image_token(PROMPT, tokenizer, IMAGE_TOKEN_INDEX,
                                           return_tensors="pt").unsqueeze(0)
        same = torch.equal(theirs, prompt_ids.cpu())
        print("  对拍 tokenizer_image_token : %s（长度 %d vs %d）"
              % ("✅ 逐 id 相同" if same else "*** 不同 ***",
                 theirs.shape[1], prompt_ids.shape[1]))
        ok &= bool(same)
    else:
        print("  [warn] 官方模块里没有 tokenizer_image_token", file=sys.stderr)

    # 等级词 id：官方在 __init__ 里算好存在 self.preferential_ids_
    theirs_ids = getattr(model, "preferential_ids_", None)
    if theirs_ids is not None:
        theirs_ids = [int(x) for x in theirs_ids]
        ours = level_ids_of(tokenizer)
        same = theirs_ids == ours
        print("  对拍 preferential_ids_   : %s（官方 %s / 本文件 %s）"
              % ("✅ 相同" if same else "*** 不同 ***", theirs_ids, ours))
        ok &= same
    else:
        print("  [warn] 模型上没有 preferential_ids_", file=sys.stderr)

    # 权重张量：官方在 score() 里**惰性**创建，正常情况下此刻还不存在，
    # 所以这里只能核对我们自己的常量与官方字面量一致。
    print("  权重张量 WEIGHTS          : %s（官方 score() 第 274 行字面量 [5.,4.,3.,2.,1.]）"
          % WEIGHTS)

    if not ok:
        sys.exit("*** 与官方实现对拍失败 —— 口径已漂移，禁止继续 ***")


def level_ids_of(tokenizer):
    """官方 `__init__` 原文: [id_[1] for id_ in tokenizer(LEVELS)["input_ids"]]

    ⚠️ `id_[1]` 取的是**第 2 个** id（第 1 个是 BOS）。若 BOS 缺失或分词结果与预期
    不同，这个下标会**静默**取到错的 token。`--tokenizer-only` 会把逐词分词打全,
    供人工确认那个「第 2 个」确实是我们想的那一个。
    """
    return [id_[1] for id_ in tokenizer(LEVELS)["input_ids"]]


def prompt_ids_of(tokenizer, device="cpu"):
    return tokenizer_image_token(PROMPT, tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt").unsqueeze(0).to(device)


# --- 加载 ---------------------------------------------------------------------
def load_config(ckpt, override_name_or_path=True):
    """读 config 并按需把 `_name_or_path` 覆盖为本地目录（**坑 ①** 的处置）。"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)
    if override_name_or_path:
        before = getattr(config, "_name_or_path", None)
        if before != str(ckpt):
            print("  [fix] config._name_or_path: %r -> %r" % (before, str(ckpt)))
            config._name_or_path = str(ckpt)
    return config


def tokenizer_and_processor(ckpt):
    """只加载 tokenizer + 图像处理器（**不碰权重**）。

    ⚠️ 参数与官方 `__init__` 第 256-257 行逐字相同 —— 官方那里写的是
    `AutoTokenizer.from_pretrained(config._name_or_path)`，**没有 `use_fast=False`**。
    本函数刻意**不加** `use_fast`，以保持与官方解析到同一个类。
    （对比：`../q-align/infer.py` 用了 `use_fast=False`，因为 Q-Align 官方那条路径就是慢分词。）
    """
    from transformers import AutoTokenizer, CLIPImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    image_processor = CLIPImageProcessor.from_pretrained(ckpt)
    return tokenizer, image_processor


def build_model(ckpt, device, dtype):
    """按官方 README 的 quick-start 路径加载权重。

    ★ **2026-10-13 实测更正**：原写「`attn_implementation="eager"` 照抄模型卡，
    transformers 4.36.1 不用改代码」——**后半句已被实测推翻**，本函数据此改写。

    模型卡把 `attn_implementation="eager"` 当 `from_pretrained` 的 kwarg 传，
    在 transformers **4.36.1** 上**必崩**（就是本函数第一次上卡的报错）：

        TypeError: MPLUGOwl2LlamaForCausalLM.__init__() got an unexpected
                   keyword argument 'attn_implementation'

    机制（已在卡上逐行读 `modeling_utils.py` 确认，不是推测）：
      · `from_pretrained`（`:2483`）在 `:3444` 调 `_autoset_attn_implementation` 时
        **只读 config 上已有的 `_attn_implementation_internal`，并不看 kwargs**；
      · 随后 `:3450` 直接 `cls(config, *model_args, **model_kwargs)`，而 `model_kwargs`
        里那个 `attn_implementation` **没有任何人 pop 掉**（4.36.1 里 pop 它的是
        `_from_config` `:1235`——那是**另一条**代码路径；我们显式传了 `config=`，
        走的就是 `from_pretrained`）；
      · 于是它被送进 remote-code 的 `MPLUGOwl2LlamaForCausalLM.__init__(self, config)`
        （`modeling_mplug_owl2_huggingface.py:252`，**签名只收 config、无 `**kwargs`**）。
    模型卡注明 4.46.3 / 4.57.6「需要改代码」指的是**另一处**，**不是**说 4.36.1 原样能跑。

    修法（钉根因，不是绕过报错）：上游真正读后端的是 **config 上的属性** ——
    `modeling_llama2.py:515`
        `self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](...)`
    故把后端**写在 config 上**，`from_pretrained` 不再收 `attn_implementation`。
    语义与官方意图一致（eager），而这条路径不经过任何 kwargs 转发。
    （4.36.1 下不写该属性时默认回退也是 "eager"——`PretrainedConfig` 的向后兼容默认；
    但**显式写出来**才不会被将来升级 transformers 后的 sdpa 默认值悄悄抢走。）

    `torch_dtype=torch.float16` 照抄模型卡，不变。
    """
    from transformers import AutoModelForCausalLM
    t_dtype = torch.float32 if dtype == "fp32" else torch.float16
    if dtype == "fp32":
        print("  [warn] --dtype fp32：**偏离官方**（模型卡 quick-start 写死 float16，"
              "且官方 weight_tensor 硬编码 .half()）。本文件改为跟随 model.dtype "
              "以免 matmul dtype 不匹配。该次结果**不得**与 fp16 的直接互比。",
              file=sys.stderr)

    config = load_config(ckpt)
    # ★ 见 docstring：后端只能钉在 config 上，**不能**当 from_pretrained 的 kwarg 传
    #   （4.36.1 不 pop 它，会透传给不吃它的 remote-code 模型类 ⇒ TypeError）。
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, config=config, trust_remote_code=True,
        torch_dtype=t_dtype,
        device_map={"": device},
    )
    model.eval()
    # 把**生效的**后端打进日志（判据的可复核性：上游 modeling_llama2.py:515 按它选注意力类）
    print("  attn 后端      : %s（期望 eager）"
          % getattr(model.config, "_attn_implementation", "<未知>"))

    # ★ 模型自己带 tokenizer / image_processor（官方 __init__ 256-257 行建的）。
    #   用它俩而不是另建一份，是因为 `score()` 用的就是这俩 —— 这样「我们复刻的计算」
    #   与「官方 score() 的计算」在对象级别就是同一套。
    tokenizer = getattr(model, "tokenizer", None)
    image_processor = getattr(model, "image_processor", None)
    if tokenizer is None or image_processor is None:
        print("  [warn] 模型上没挂 tokenizer/image_processor，改为从 ckpt 目录自建",
              file=sys.stderr)
        tokenizer, image_processor = tokenizer_and_processor(ckpt)

    # 坑 ① 的**运行期验证**：模型自带的 tokenizer 应当就是从本地 ckpt 目录加载的。
    # 若 `_name_or_path` 覆盖没生效，这里会显示 hub id（或直接联网/报错，视 HF_HUB_OFFLINE 而定）。
    src = getattr(tokenizer, "name_or_path", None)
    print("  tokenizer      : %s | vocab=%s | 来源=%s"
          % (type(tokenizer).__name__, getattr(tokenizer, "vocab_size", "?"), src))
    if src is not None and os.path.abspath(str(src)) != os.path.abspath(str(ckpt)):
        print("  [warn] tokenizer 的来源不是本地 ckpt 目录（%r != %r）——"
              "`_name_or_path` 覆盖可能未生效，请检查" % (src, ckpt), file=sys.stderr)

    level_ids = level_ids_of(tokenizer)
    input_ids = prompt_ids_of(tokenizer, model.device)
    return tokenizer, model, image_processor, level_ids, input_ids


def score_one(model, image_processor, level_ids, input_ids, img_path):
    """返回 (score 1-5, probs 五级 dict, logits 五级 dict)。

    计算与官方 `score()` 第 277-283 行**逐字一致**，唯一差别是把加权之前的
    softmax 结果留下来（官方那里 `@ self.weight_tensor` 之后分布就没了）。
    """
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    img = expand2square(img, tuple(int(x * 255) for x in image_processor.image_mean))

    with torch.inference_mode():
        # 官方：.half()（写死）。fp32 路径下跟随 model.dtype，否则 matmul 会报 dtype 错。
        pixel_values = image_processor.preprocess(
            [img], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(dtype=model.dtype, device=model.device)

        # 官方：self(input_ids.repeat(n,1), images=...)...["logits"][:, -1, preferential_ids_]
        output_logits = model(input_ids.repeat(pixel_values.shape[0], 1),
                              images=pixel_values)["logits"][:, -1, level_ids]

        probs_t = torch.softmax(output_logits, -1)          # ← 官方在加权前丢掉的中间量
        weight_t = torch.tensor(WEIGHTS, dtype=probs_t.dtype, device=probs_t.device)
        score = (probs_t @ weight_t).item()                 # ← 官方原样

    probs = probs_t.float().cpu().numpy()[0]                 # 缓存用高精度副本
    lg = output_logits.float().cpu().numpy()[0]
    return (float(score),
            dict(zip(LEVELS, [round(float(p), 6) for p in probs])),
            dict(zip(LEVELS, [round(float(v), 6) for v in lg])))


# --- 不依赖权重的核验 ---------------------------------------------------------
def inspect_ckpt(ckpt):
    """打印权重分片与 config —— **纯文件读取，不加载模型、不需要 GPU**。

    无卡模式（AutoDL 0.5 核 + 2 GiB）就能跑，是本文件在取数前能做的第一层核对。
    """
    print("ckpt 目录      :", ckpt)
    if not os.path.isdir(ckpt):
        sys.exit("目录不存在：%s" % ckpt)

    print("\n--- 文件清单 ---")
    for pat in ("*.safetensors", "*.bin", "*.json", "*.model", "*.txt"):
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

    idxs = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
    for ix in idxs:
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
        for k in ("_name_or_path", "architectures", "model_type", "torch_dtype",
                  "transformers_version"):
            print("  %-22s = %r" % (k, cfg.get(k)))
        am = cfg.get("auto_map")
        print("  auto_map               = %s" % ("**缺失 ⇒ trust_remote_code 加载不了**"
                                                 if not am else json.dumps(am)))
        print("  ⚠️ `_name_or_path` 是 **hub id**，官方 256-257 行拿它去建 tokenizer/"
              "image_processor\n     ⇒ 加载前必须覆盖为本地目录，否则会联网（坑 ①）。")

    pp = os.path.join(ckpt, "preprocessor_config.json")
    if os.path.isfile(pp):
        d = json.load(open(pp, encoding="utf-8"))
        print("\n--- preprocessor_config.json ---")
        for k in ("image_processor_type", "size", "crop_size", "image_mean",
                  "image_std", "do_center_crop", "do_resize"):
            print("  %-22s = %r" % (k, d.get(k)))
        im = d.get("image_mean")
        if im:
            print("  ⇒ expand2square 背景色 = %s" % [int(x * 255) for x in im])

    print("\n--- 本文件使用的常量（须与官方 modeling_mplug_owl2_huggingface.py 一致）---")
    print("  IMAGE_TOKEN_INDEX  = %d      (官方 37 行)" % IMAGE_TOKEN_INDEX)
    print("  DEFAULT_IMAGE_TOKEN= %r   (官方 38 行)" % DEFAULT_IMAGE_TOKEN)
    print("  LEVELS             = %s   (官方 __init__)" % LEVELS)
    print("  WEIGHTS            = %s     (官方 score() 第 274 行, 1-5 制)" % WEIGHTS)
    print("  PROMPT             = %r" % PROMPT)
    print("\n[inspect-ckpt] 以上**全部不依赖权重**，无卡模式（2 GiB）即可核验。")


def run_tokenizer_only(ckpt, image):
    """`--tokenizer-only`：把**不依赖权重**的 token 级口径全部打出来后退出。

    核验对象是「最容易错且错了看不出来」的那几件事：
      ① 五级词的 token id 是否真取到「第 2 个」（第 1 个是 BOS）——官方是 id_[1]。
      ② prompt 里 <|image|> 是否被换成 IMAGE_TOKEN_INDEX(-200)、出现几次。
      ③ 图像预处理口径（mean/std/size/crop）—— expand2square 的背景色由 mean 算出。
      ④ `_name_or_path` 覆盖前后的值（坑 ①）。
    """
    cfg = load_config(ckpt, override_name_or_path=False)
    print("config._name_or_path（**未覆盖前**，即 config.json 里的原值） = %r"
          % getattr(cfg, "_name_or_path", None))
    print("⇒ 官方 256-257 行拿它去建 tokenizer/image_processor；本文件在加载前把它覆盖为 %r"
          % str(ckpt))
    print()

    tokenizer, image_processor = tokenizer_and_processor(ckpt)

    print("tokenizer      : %s | vocab = %s | bos = %s"
          % (type(tokenizer).__name__, tokenizer.vocab_size, tokenizer.bos_token_id))
    print("IMAGE_TOKEN_INDEX = %d" % IMAGE_TOKEN_INDEX)
    print()

    print("① 五级词的逐词分词（官方取 id_[1]，即下表第 2 个 id）：")
    for w in LEVELS:
        ids = tokenizer(w)["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        print("   %-10s ids=%-18s tokens=%s" % (w, ids, toks))
    print("   ⇒ level_ids = %s" % level_ids_of(tokenizer))
    print()

    ids = prompt_ids_of(tokenizer)
    n_img = int((ids == IMAGE_TOKEN_INDEX).sum())
    print("② prompt 的 input_ids：长度 %d，其中 IMAGE_TOKEN_INDEX 出现 %d 次"
          % (ids.shape[1], n_img))
    print("   前 8 个 id = %s" % ids[0, :8].tolist())
    print("   prompt 原文 = %r" % PROMPT)
    print("   ⚠️ 须与官方 score() 第 277 行的 f-string 在 task_='quality'/input_='image' "
          "下逐字符相同")
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
        sq = expand2square(img, bg)
        pv = image_processor.preprocess([sq], return_tensors="pt")["pixel_values"]
        print("④ 对 %s 实际预处理：" % os.path.basename(image))
        print("   原始 %s → expand2square %s → pixel_values %s  dtype=%s"
              % (img.size, sq.size, tuple(pv.shape), pv.dtype))
    print("\n[tokenizer-only] 以上全部不依赖权重，无卡模式（2 GiB）即可核验。")


# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="图或目录；--inspect-ckpt / --tokenizer-only 时可省略")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="DeQA-Score-Mix3 的**本地目录**（HF 仓 %s 的 snapshot）。"
                         "默认走环境变量 DEQA_CKPT" % HF_REPO)
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
                    help="只加载 tokenizer + 图像处理器，打印 token 级口径后退出（**不碰权重**）")
    # ★ 无卡模式的 CPU 配额是 0.5 核，而容器里 `nproc` 报的是 host 的核心数
    #   （Q-Align 实测：56 线程抢 0.5 核比单线程**慢 199×**，表现为「加载卡死」）。
    #   ⇒ 无卡模式一律 --threads 1；有卡模式下要跑 GPU 前向时不必传（走 torch 默认）。
    ap.add_argument("--threads", type=int, default=0,
                    help="torch 的 CPU 线程数；0=不干预（保持 torch 默认）。"
                         "★ AutoDL 无卡模式请传 1")
    args = ap.parse_args()

    if args.threads > 0:
        _n = torch.get_num_threads()
        torch.set_num_threads(args.threads)
        print("[note] torch CPU 线程数 %d -> %d" % (_n, args.threads))

    if args.inspect_ckpt:
        inspect_ckpt(args.ckpt)
        return

    if args.tokenizer_only:
        run_tokenizer_only(args.ckpt, args.image)
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
        args.ckpt, args.device, args.dtype)
    dev = str(model.device)
    print("模型就绪，device=%s，加载耗时 %.1f s" % (dev, time.time() - t0))
    print("五级 token ids = %s  （顺序 %s）" % (level_ids, LEVELS))

    print("--- 与官方实现对拍（口径漂移检查）---")
    _assert_tools_match_official(model, input_ids, tokenizer)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, img in enumerate(imgs, 1):
            t1 = time.time()
            score, probs, logits = score_one(model, image_processor, level_ids,
                                             input_ids, img)
            f.write(json.dumps({"image": str(img), "score": round(score, 6),
                                "probs": probs, "logits": logits,
                                "device": dev, "dtype": args.dtype,
                                "seed": args.seed}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            print("  [%d/%d] %.4f (1-5制)  %s  (%.1f s)"
                  % (i, len(imgs), score, img.name, time.time() - t1))
    print("完成 %d 张，缓存 -> %s" % (n, out))


if __name__ == "__main__":
    main()
