# 评委：IQA-T1（生成式解码型）

**E 组（生成式解码型）**，九个评委里**唯一的解码型**——其余八个都是打分／回归，直接给一个标量；IQA-T1 先生成
文本（多轮工具调用 ＋ 推理）再解析分数。它的对照价值在于：判据若在「生成一段话再抽数」这一支上也成立，就不是只在回归型上成立的产物。

**污染状态：实测干净。** SFT 数据（11,315 条）与 GRPO 数据（7,046 条）整份取回，在 KonIQ 官方 test 的
**2,015 个文件名**上做原始文本匹配，**命中 0**。⚠️ 样本数 ≠ 图像数：11,315 条 reasoning chain 只覆盖
**3,540 张**图；TID2013 侧未做同款匹配。

## 复现对象与许可

*IQA-T1: Incentivizing Image Quality Assessment in Multimodal LLMs*，arXiv:**2607.12375**（ECCV 2026），
NWPU ＋ HKUST，与其余 8 个评委无血缘。权重与代码同属 `zibuyu-02/IQA-T1`：HF 仓 `zibuyu-02/IQA-T1`，
代码仓 `https://github.com/zibuyu-02/IQA-T1`，pin 在 commit **`3cabcdae2edabb40e0ffa4f5f85efbe1b7762f0d`**。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者侧代码（`zibuyu-02/IQA-T1`） | **MIT** | 可商用 |
| 权重（`zibuyu-02/IQA-T1`） | **MIT**（权重卡记 `mit`） | 可商用 |
| 底座模型（`Qwen3-VL-4B-Instruct`） | **Apache-2.0**（卡记 `apache-2.0`） | 可商用 |

**复现的是 HF 发布的 `zibuyu-02/IQA-T1`。** 上游两处本地路径提示互相矛盾（`README.md` 写 `...-all-GRPO-v3/`、
`infer_qwen3vl_iqa.sh` 默认写 `...-all-SFT`，都是作者机器上的路径）⇒ 无法判断发布的是哪一支，故**不写**「复现了 GRPO 版／SFT 版」。

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | Qwen3-VL-**4B-Instruct**，`Qwen3VLForConditionalGeneration`（transformers 原生，**无 `auto_map`** ⇒ 不需要 `trust_remote_code`） |
| 权重 | 两个 `safetensors` 分片：**分片1 = 4,997,532,824 B**，**分片2 = 4,656,099,008 B** ⇒ **合计 9,653,631,832 B ≈ 9.65 GB** |
| 仓总 | ≈9.67 GB（其余约 20 MB 为 tokenizer / processor / chat template / 配置） |
| 规模 | hidden 2,560 / 36 层 / vocab 151,936 |
| 来源 | `huggingface_hub.snapshot_download`，只取 `safetensors` |
| 落地路径 | `/root/autodl-tmp/models/iqa-t1`（**不随本包释出**，权重留在发布方） |
| 上游 `scripts/tools.py` | 28,811 B，sha256 `ba391a5ee90ad3bd8553dcaf06bfba5487f78535c0dc3b499cc40a6861bf5189` |
| 上游 `inference/infer.py` | 23,306 B，sha256 `0b87e849ca9b282b215ec82ef6ea5bd3a871114a1aa3ee39f692cee8fff31e1d` |
| 上游 `demo/infer.py` | 23,310 B，sha256 `b3fd35e0cb7217fabb3e01e2c4af049b998e6aee4a1f68e6fc1cdc65ce5928ec` |
| 上游 `demo/requirements.txt` | 129 B，sha256 `84791c4cbe1bfb43f6fda71867121a72886e8930decad52fab58a38eedb6a510` |

**`config.json` 的 `dtype` 是错的**：声明 `float32`（那应约 19 GB），与磁盘 9.65 GB 矛盾；官方代码写死
`dtype=torch.bfloat16`（`inference/infer.py:444`）⇒ **实际是 bf16**，`--inspect-ckpt` 会把矛盾打印出来。
表末四行上游件的 (字节数, sha256) 已钉死，前两个由 `--inspect-ckpt` 强制校验（不符即停），两列来自同一 commit 的两条通道，互证。

## 预处理与打分口径

**解码参数（贪心）**，`inference/infer.py:377-384` 唯一的 `GenerationConfig`：

```python
GenerationConfig(do_sample=False, temperature=1.0, top_k=0, top_p=1.0,
                 max_new_tokens=2560, pad_token_id=self.processor.tokenizer.pad_token_id)
```

`do_sample=False` ⇒ **贪心解码**；`temperature` / `top_k` / `top_p` 在 HF 语义下**全部失效**，不可读成随机采样；
默认 `seed=42`。两处 `generate` 共用此 config，`demo/infer.py` 与主入口的差异只涉及 import 路径、HF hub id
透传、`attn_implementation` 与 `device_map`，**不涉及采样与提示词**。

**prompt 与模板**：不复制进本文件——`infer.py` 原样 import 上游 `SYSTEM_PROMPT` / `USER_PROMPT` 装进 `messages`，
推理侧唯一定义处是 `inference/infer.py:274-303`。系统提示写明 *"Score range: 1.00 to 5.00"*（`:290`），输出格式
`<answer_start>{"score": X.XX}<answer_end>`（`:288`）；`--tokenizer-only` 会打印 prompt 原文与渲染后的完整输入。

**工具调用（第二层，同样影响输入口径）**：官方默认 `enable_tools=True`、`max_tool_calls=6`（形参默认 5、调用处传
6 ⇒ **生效值 6**），`--no_tools` 可关。开工具走 `generate_with_tools`：脚本自取 PIL 原图并逐轮追加工具图，每轮把
`<|vision_start|><|image_pad|><|vision_end|>` 占位符追加进上下文；关工具走单轮 `generate` 与
`qwen_vl_utils.process_vision_info`（未在官方 requirements 中）。⇒ 开关工具是**接口层 ＋ 输入口径的双重变更**，
实测固定开工具并逐图记录。图像 token 数由 `preprocessor_config.json` 的 `min_pixels` / `max_pixels` / `patch_size` /
`merge_size` 决定，本次全量实测走 **512×384 缩略图**。

**分数怎么抽出来**（`inference/infer.py:306-318`，逐字复刻）：

```python
m = re.search(r'"score"\s*:\s*([0-9]+\.?[0-9]*)', response_text)
if m: return float(m.group(1))              # json（预期）
for n in re.findall(r'([0-9]+\.?[0-9]*)', response_text):
    v = float(n)
    if 1 <= v <= 5: return v                # fallback（可能抓错数）
return None                                 # none
```

回退分支的风险是实的：回复「score is 12 out of 5」时它会把 `5` 当分数，而 `raw_response` 天然含工具返回的中间数值，
抓错概率高于纯打分型评委。故**逐图记录** `score_path` ∈ {`json`, `fallback`, `none`, `error`} 并打印分布；
`error` 也写进缓存（不吞异常），`none` / `fallback` 原样保留、不修补。

**量纲 1–5，不做仿射变换**：量纲转换属指标层；本课题的命题正是「分数尺度不能假定可比」，折成 0–1 等于预先把「仿射关系成立」写死进数据。

## 运行栈

Python **3.11.17** / torch **2.6.0+cu124** / transformers **5.19.0**；RTX 3090，**24,576 MiB**；`device=cuda`、
`dtype=bf16`、`attn=sdpa`（未装 `flash-attn`）、`bs=1`。租卡配额实测：有卡模式 14 核 / 90 GiB 内存 / 24 GB 显存；
无卡模式 0.5 核 ＋ 2 GiB（装不下权重）。本机 GTX 1650（4 GB）与 7.4 GiB 内存都装不下 9.65 GB 权重 ⇒ 连 CPU 路径也不可行。

官方 `demo/requirements.txt` 要求 `torch>=2.6.0` 与 `transformers>=4.57.0`（Qwen3-VL 架构 4.57 才进 transformers）⇒ **不能 clone 其余评委那份 transformers 4.36.1 的环境**。九个评委因此是**三段栈**：本机栈 6 个
（3.11.15 / 2.6.0+cu124 / 5.19.0 / GTX 1650）、卡上 qalign 系 2 个（3.10.22 / 2.0.1+cu117 / 4.36.1 / RTX 3090）、本评委单独一段；引用任何实测数字必须注明它出自哪一段栈。

## 本包包含什么

本目录随包只有 `infer.py`（图 → 分数 ＋ `score_path` ＋ `tools_used` ＋ `raw_response`）与本 README。
**不随包释出**：权重（9.65 GB，走 `snapshot_download`，不带下载脚本）、上游代码克隆、租卡运维脚本、
工具输出 PNG（全量约 218 MB）、缓存与日志。**没有 `vendor_*`**：上游 `scripts/tools.py` 有 956 行 /
15 个工具，拷进来会让「上游改了什么」不可见，改用 clone ＋ sha256 指纹。

⇒ 运行前需按 pin 的 commit 克隆上游仓库（`infer.py` import 它的 `scripts.tools` 与 `inference.infer`）。
封装与上游的关系：工具管线、提示词**原样调用**；`extract_score` 逐字复刻、只多返回一个分支名（对拍 **2,006 例全部一致**）；
`_load_model` 仅让 `attn_implementation` / `dtype` 可指定；种子、贪心、`max_new_tokens` 不碰；
批处理、名单取图、缓存落盘、`--inspect-ckpt` / `--tokenizer-only` 为新增。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl [--list 名单.txt] \
    --ckpt <权重目录> --repo <上游仓库> --device cuda --dtype bf16 --attn sdpa
```

## 实测

KonIQ-10k 走官方 test split；TID2013 走可用集（无官方 split ⇒ `set` 记 `test`）；两列同一段栈、不同数据集，不可直接比。

| 指标 | KonIQ-10k（2,015 张，512×384） | TID2013（2,687 / 3,000 张，384×512） |
|---|---|---|
| 覆盖 | 2,015 / 2,015（异常 0，退出码 0） | 2,687 / 2,687（异常 0，`rc=0`） |
| **SRCC** | **0.8878** | **0.7034** |
| KROCC | **0.7227** | **0.5217** |
| PLCC（raw） | **0.9078** | **0.7500** |
| PLCC（4 参数 logistic） | **0.9099** | **0.7599** |
| 预测范围 | — | 1.2700 .. 4.4200（1–5 制） |
| 墙钟 | **2 h 42 min 54 s** | **12,544 s = 3 h 29 min 04 s** |
| 吞吐 | 稳态 **4.73 s/图** | **0.208–0.213 图/秒 ≈ 4.7–4.8 s/图**（末段 3.4–3.6） |
| `score_path` | `json` 2,015 / `fallback` 0 / `none` 0 / `error` 0 | `json` 2,687 / 0 / 0 / 0 |
| 平均工具调用 | 2.6025 次/图 | 2.42 次/图 |

TID2013 口径：排除 **313 张** = 内容 i01/i25 全部 **240** 张 ∪ `_BROKEN_FILES.txt` 净增的 **73** 张；
MOS 取 `tid2013_eval_labels.csv` 的 `MOS` 列（0–9 制，实测 0.2424 .. 7.2143）；名单 2,687 行、纯 LF，
sha256 `cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`；图集上传件 1,111,692,036 B，sha256 `0db74247d9c5b388fd7b90f469fb27929bc7461e26299bdae12642357e8b271c`。

**确定性（两个数据集各 20 张 × 2 次）**：① 剔除 `seconds` 计时字段后内容**逐字节相同**；② `tools_used` 序列逐图相同；
工具输出 PNG 也逐字节相同（KonIQ 53/53 张；TID2013 45/45 文件）。唯一差异字段就是 `seconds` ⇒ 该 GPU / bf16 / sdpa 下**逐位可复现**。
⚠️ 全量工具图未取回，全量工具输出的逐字节比对**没有做**；`score_path` 全 `json` 只是间接旁证。

**工具使用（KonIQ 全 2,015 张）**：共 5,244 次、9 种工具。`GradientMagnitudeHistogram` 1,981、
`PerceptualColorfulnessMap` 1,837、`LuminanceHistogram` 1,107，三者占 **93.9%**，长尾 6 种合计 6.1%。
每图工具数 = {1 次: 1 张, 2 次: 799 张, 3 次: 1,215 张}，**无一张触到上限 6 ⇒ 截断率 0**。

**算力**：本评委单列一档约 **34.4 GPU 小时**（3090 口径），**一个评委就占九评委首轮缓存的 83.6%**（其余 8 个合计约 6.7 h）。

**产物**（不随本包，`scores/` 只带每图标量）：`pred_iqat1_test.jsonl` 2,030,934 B，sha256 `eb9a0cc9…`；`run_l4.log` 816,214 B，
sha256 `05b19924…`；TID2013 侧 `pred_iqat1_tid2013.jsonl` 2,625,942 B，sha256 `043e9e0fd5475f59dd4384bbccc48d8955643b6493f3568dff12aee9d46651c6`，
`run.log` 1,025,424 B。产出这批数字的 `infer.py` / `run_l4.sh`（不随包）当日指纹为 `c02ebec3…` / `c097cbab…`。

## 与 pyiqa 基准的对照

**没有对照。** pyiqa 基准表 `tests/NR_benchmark_results.csv` **没有 `iqat1` 行** ⇒ 本评委的 TID2013 数字连旁证都没有，
**不得**写「复现了某基准」。论文自报的 2.34 工具/图 与 5.09 s/图 只能当旁证（协议与测速硬件均无从核实），不得当门槛。

## 已知限制

1. **没有 demo 级硬判据**：官方 README 与权重卡都没给示例图的预期分数；示例图文件名形如 `826373.jpg`，**是 KonIQ 的图**，不能当干净样本。
2. **没有可比的官方实测数字**：论文只写训练用 6×RTX 4090，自报 5.09 s/图 的测量硬件无从核实。
3. **TID2013 无官方 split**：排除 i01/i25 与损坏文件后的 2,687 张不是随机子集，**i01/i25 的选择性偏差未检验**。
4. **量纲 1–5**：与 Q-Align 的 0–1、DeQA-Score 的 1–5 并排时必须带量纲；跨评委比较须同时报告预处理与分辨率。
5. **栈不同**：与卡上 qalign 系差 transformers 4.36.1↔5.19.0、torch 2.0.1↔2.6.0，该栈差**未被测量**、**不得给出估计值**。
6. **注意力后端可能偏离官方**：官方 inference 写死 `flash_attention_2`，实测用 `sdpa`（官方 demo 本身就用 `sdpa`）；缓存里的 `attn` 字段可追溯。
7. **全量工具图未取回** ⇒ 全量工具输出的逐字节比对未做。
8. **工具数 2.60 vs 论文 2.34** 只记为旁证级差异，不得据此判复现失败。
