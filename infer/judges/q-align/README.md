# 评委：Q-Align（LMM 等级期望型，mPLUG-Owl2-7B）

**A 组（等级期望型）**。A 组三个评委里，Q-Align 是唯一能承接「Q-Future 组原创论文」身份的一支（Q-ReAlign Mini 无独立 arXiv；DeQA-Score 属另一研究组）；它与 DeQA-Score **同底座 mPLUG-Owl2、异血缘**，两者分数不可直接混比。7B、权重约 **16.41 GB** ⇒ 本机 GTX 1650（4 GB、无 bf16）物理装不下，全部推理在租卡机完成。**污染状态：官方 train 与我们的 test 交集实测 = 0，干净**——实测结论，与 MANIQA / MUSIQ 的污染结论不同。

## 复现对象与许可

Wu, Zhang, Zhu, Zhang, Wang, Chen, Sun. *Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels*, **ICML 2024**，arXiv:**2312.17090**。官方实现：[Q-Future/Q-Align](https://github.com/Q-Future/Q-Align)；权重：[`q-future/q-align-iqa`](https://huggingface.co/q-future/q-align-iqa)。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者侧代码（`Q-Future/Q-Align`） | **S-Lab License 1.0** | **不可商用**；第 4 条要求商用「please contact the contributor(s)」 |
| 权重（`q-future/q-align-iqa`） | **模型卡记 `license: mit`** | 卡正文仅 48 B（`license: mit` ＋ arXiv 号）；与代码的 S-Lab 1.0 是**两份声明、两个对象**，不得互相外推 |
| 底座 mPLUG-Owl2（LLaMA-2-7B ＋ CLIP ViT-L） | **模型卡记 `apache-2.0`** | 官方件中未见援引 LLaMA-2 条款；该句按转述处理、不引用 |

许可见 `LICENSING.md` 与 `LICENSE-NOTES.md`（2026-10-16 经镜像取回）。

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | mPLUG-Owl2：CLIP **ViT-L** ＋ **LLaMA-2-7B**；hidden 4096 / 32 层 |
| 加载类 | `MPLUGOwl2LlamaForCausalLM`（`q_align` 注册后由 `AutoModelForCausalLM` 加载） |
| HF 仓库 | `q-future/q-align-iqa`（**不是** `q-future/one-align`，见下） |
| 分片 | `pytorch_model-00001-of-00002.bin` **9,991,591,698 B** ＋ 第 2 片 **6,417,830,970 B**；合计 **16.41 GB** |
| 精度 / 注意力 | config 记 **bfloat16**，运行加载 **fp16**（`.half()`）；`attn_implementation="eager"` |
| 整体 SHA256 | **未算**；替代指纹 = 分片字节数（与 HF `Content-Length` 逐位核对）＋ 首 1 MiB SHA256 |
| 来源 / 落地 | `snapshot_download` 自 HF；`/root/autodl-tmp/models/q-align-iqa/`（**不随本包释出**） |
| 仓库 `lastModified` | **2023-12-29**；含训练产物（`trainer_state.json` / `training_args.bin` / `rng_state_0..7.pth` / `zero_to_fp32.py`）⇒ 训练输出目录直接上传 |

**必须加载 `q-align-iqa`，不是 `one-align`**：后者是 OneAlign，额外吞了 IAA 与 VQA 两任务，用错等于换评委。判据是各 checkpoint 首片前 1 MiB 的 SHA256 互不相同（2026-10-13 实测）：

| checkpoint | 分片字节数 | 首 1 MiB SHA256 |
|---|---|---|
| `q-align-iqa` ← **本包加载的** | 9,991,591,698 | `f9747ff4a7127b535ec2e9b485c7635af4555a3a807b66dd64af16f0cfeb18d0` |
| `q-align-quality` | 9,991,591,698 | `d8600c48a7293e5db52ec54b99f3ffd676bc56adb72bdcdfe7fd8b650d5d7132` |
| `q-align-image` | 9,991,591,698 | `c5285c34f4bf1262b4db7660f72cf37859ec7925a536dfd4eac5d196c5c49f74` |
| `q-align-aesthetic` | 9,991,591,698 | `0dda9e11f60dcb4d2a72da1f490d6bbb16106d40c8a5b8a6d3581efc395b0901` |
| `q-align-vqa` | 9,991,591,698 | `ebbf17791e7a967d7a5249c62b924e86177874dd024793386626d994644e288f` |
| `one-align` | 9,991,591,698 | `def8293a5132e06e438e1aa710b62051b519d0894bf62b5c90585332a887b411` |

首 1 MiB 摘要只用于区分 checkpoint，**不能**当整体完整性摘要。整份权重的 SHA256 未算：HF 镜像 API 不返回 LFS `oid`，算出来无从对照；分片字节数 ＋ 首 1 MiB 摘要已足以判「是不是同一份权重」。

`q-align-iqa` 的 `config.json` **没有 `auto_map`**、仓库内**没有 `modeling_*.py`** ⇒ 官方 README 的 `trust_remote_code=True` 快捷方式**对它无效**（只对 `one-align` 成立）。正解：把官方仓库根目录放进 `sys.path`，`import q_align` 触发 `q_align/model/modeling_mplug_owl2.py:377–378` 的注册，`infer.py --repo` 指向该目录；**钉 commit** `fd0c3def92da21cd124f8301722b6b83bb75eccd`（2026-10-13）。不执行 `pip install -e .`，以免改动已钉住的依赖版本。

`modeling_mplug_owl2.py:255–256` **硬编码**从 `q-future/one-align` 取 tokenizer 与 image processor，与 `--ckpt` 无关。逐位核对后两边四个文件相同：`tokenizer.model` sha256 `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347`（499,723 B）、`tokenizer_config.json` `cf01145c6de57f06…`、`special_tokens_map.json` `719833ff26ac897a…`、`preprocessor_config.json` `7eeb54e111e0c96d1629ac07a798fd5a7782658ed6468bc018fb16c4410358d1`（317 B）。⇒ 不存在「用错 tokenizer」的风险。

但离线时必须让这几个文件已在 `HF_HOME` 缓存里，否则 `from_pretrained` 会对每个文件重试 5 次（每个约 23 s）；运行前强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。两处**未声明的依赖**：`icecream`（`modeling_mplug_owl2.py:38` 的模块级 import）与 `protobuf`（`LlamaTokenizer` 慢分词路径需要）。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 质量 prompt（逐字） | `USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is` |
| 五级词表 / token id | `excellent, good, fair, poor, bad` → `[15129, 1781, 6534, 6460, 4319]` |
| 取 token / logits | `tokenizer(word)["input_ids"][1]`（第 2 个 token，第 1 个是 BOS）；取**最后一位**的五个等级 logits |
| 图像预处理 | `expand2square`（ImageNet mean 背景 `[122, 116, 104]`）→ `CLIPImageProcessor`：`shortest_edge=448`、`crop 448×448`、`do_center_crop=True` |
| 精度 | 预处理 fp32；模型 `.half()` / `torch_dtype=torch.float16` |
| 输出 | `score = softmax(logits_levels) @ [1.0, 0.75, 0.5, 0.25, 0.0]` ∈ **[0,1]**，并缓存五级 `probs` |

官方 README 的 `wa5()` 用 `[5, 4, 3, 2, 1]`（1–5 制），与代码里的 `[1, .75, .5, .25, 0]` 是仿射关系：**`score(1–5) = 4 × score(0–1) + 1`**。本封装以 0–1 为准；与 DeQA-Score 的 1–5 制并排读时用同一式。五级分布是必需中间量——官方 `forward` 加权后只剩标量，`infer.py` 在加权前留下 softmax，分数与官方路径逐位一致。

## 运行栈

Q-Align 的全部数字出自租卡机的独立环境 **`envs/qalign`**（Python 3.10 / torch 2.0.1 / transformers 4.36.1）。卡上另两个评委不在同一个环境：DeQA-Score 用 `envs/deqa`（由 `envs/qalign` clone，数值栈相同），IQA-T1 用 `envs/iqat1`（另一套 torch / transformers）。无卡模式实测配额仅 **0.5 核 ＋ 2 GiB**，7B 任何 dtype 都装不下、进程在 `Loading checkpoint shards: 0%` 被静默杀掉 ⇒ 前向只能上有卡模式。

| 数值栈 | 评委 | 环境 | Python | torch | transformers | 机器 |
|---|---|---|---|---|---|---|
| 本机栈（6 个） | Q-ReAlign Mini / CLIP-IQA / MANIQA / MUSIQ / HyperIQA / TOPIQ-NR | 本机 | 3.11.15 | 2.6.0+cu124 | 5.19.0 | GTX 1650 4 GB |
| 租卡 qalign 段 | **Q-Align** | `envs/qalign` | 3.10 | **2.0.1** | 4.36.1 | RTX 3090 24 GB |
| 租卡 qalign 段（clone） | DeQA-Score | `envs/deqa` | 3.10 | 2.0.1 | 4.36.1 | RTX 3090 24 GB |
| 租卡 iqat1 段 | IQA-T1 | `envs/iqat1` | 3.11.17 | 2.6.0+cu124 | 5.19.0 | RTX 3090 |

**跨栈数字不可混入同一张表**：本评委的 KonIQ 与 TID2013 两列同出 `envs/qalign`，可以并列；与其余评委并列时必须写明栈，不同栈的数字不得放进同一张表。

## 本包包含什么

本目录随包只有 `infer.py`（图 → 0–1 分数 ＋ 五级 `probs`）与本 README。**不随包释出**：`setup_env.sh`、`run_l4.sh`、权重下载命令、权重文件、推理缓存。

本评委**没有 `vendor_*` 目录**：架构代码来自 `git clone` 的 `Q-Future/Q-Align`（`--repo` 指向根目录，`sys.path` 注入），权重由 `snapshot_download` 从 `q-future/q-align-iqa` 取；推理走 transformers `AutoModelForCausalLM` ＋ 仓库自带建模类，**不经 pyiqa**。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl [--repo <Q-Align 仓库根>] [--list 名单.txt] [--limit N]
```

## 实测

两列均出自 `envs/qalign`（RTX 3090 / fp16 / bs=1）；两列来自不同数据集，**不构成跨域结论**。

| 指标 | KonIQ-10k 官方 test | TID2013 可用集 |
|---|---|---|
| 样本 | **2,015 / 2,015** | **2,687 / 3,000**（排除 313 张） |
| SRCC | **0.9263** | **0.8204** |
| KROCC | **0.7669** | **0.6252** |
| PLCC（raw） | **0.9354** | **0.8423** |
| PLCC（4 参数 logistic） | **0.9362** | **0.8458** |
| 预测范围 | 0.0023 .. 0.9980 | 0.0008 .. 0.9766 |

- **KonIQ-10k**：官方 `set` 列的 test **2,015 张**，MOS 取 `koniq10k_distributions_sets.csv` 的 `MOS` 列（0–100）；源图 512×384 → `expand2square` → 448×448；2,015/2,015 全部命中。
- **TID2013**：可用集 **2,687 / 3,000**，排除 **313 张** = 内容 i01/i25 全部 **240** 张 ∪ `_BROKEN_FILES.txt` 的 **73** 张净增项；原生 384×512 BMP，未做格式转换；MOS 取 `tid2013_eval_labels.csv` 的 `MOS` 列（0–9 制，实测范围 0.2424 .. 7.2143）；TID2013 **无官方 split** ⇒ `set` 一律记 `test`（口径决策，不是官方 test；若用含 25 张参考图的 3,025 口径报数必须另注）；2,687/2,687、`rc=0`。

证据件 SHA256：KonIQ 名单（2,015 行）`adf8f66539193539…14a815`、图集 tar 149,354,136 B `243e720976a96afa…d52a9`、预测（2,015 行）`2a289775aea57ec79c017965b006f951dd3873b772866addbd645fb4dd694e8c`；TID2013 名单（2,687 行，LF）`cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`、图集 tar 1,111,692,036 B `0db74247d9c5b388fd7b90f469fb27929bc7461e26299bdae12642357e8b271c`、预测（2,687 行、648,283 B）`ddcf7b81366b30f3607484e83e60fa78d8cbcb35b1e11a25f59f0695e6ab719a`。

Q-Align 论文在 KonIQ-10k 上自报 SRCC ≈ **0.940**，本 run 低 **≈0.014**。split 可排除（官方 train 与我们的 test 交集为 0，且官方测试集就是这 2,015 张）；指标算法可排除（raw 与 logistic 只差 0.0008）。剩下最可能是论文自报值所用的**输入口径**未完全公开，以及 fp16 概率量化。**不据此判实现有误，也不调参去凑这个数。**

## 与 pyiqa 基准的对照

pyiqa 基准 CSV 的 `qalign` 行 `tid2013` 列（列头 `tid2013(PLCC/SRCC/KROCC)`）为 **0.8529 / 0.8313 / 0.6368**；本 run 为 **0.8423 / 0.8204 / 0.6252** ⇒ 残差 **−0.0106 / −0.0109 / −0.0116**（三项同向、同量级）。⚠️ 该列按 `mos_with_names.txt` 覆盖 **3,000 张**，本可用集是 **2,687 张**，**不是同一组图上的复现对照** ⇒ **不得写「复现了 pyiqa 基准」**。该 CSV 的 `qalign` 行 `koniq10k` 一格为空，Q-Align 在 KonIQ 上的基准值在 `koniq10k-1024` 列；其输入口径与本 run 的 448×448 不同，未作同组对照。

## 已知限制

1. **输入口径不同**：本评委走 `expand2square` → 448×448，不是本机六位的 512×384 native ⇒ 任何跨评委比较都必须写明预处理。
2. **本机装不下**：7B / 16.41 GB 对 GTX 1650 4 GB、无 bf16 ⇒ 权重加载与前向全部在租卡机完成，本机没有可用于冒烟的同款权重。
3. **KonIQ 自报值只当旁证**：论文 SRCC ≈0.940 与本 run 0.9263 的差未完全归因（输入口径 / fp16 量化），**不得**写「复现了论文值」，也不得调参凑数。
4. **TID2013 无官方 split**：`set` 一律记 `test` 是口径决策；i01/i25 的选择性偏差未检验，本行数字不得默认与其余内容同分布。
5. **两列不可直接比**：数据集、标注谱系（TID2013 是成对比较，KonIQ 是单刺激）、MOS 量纲均不同，并列只为留锚点，不构成「下降 = 复现变差」的判断。
6. **外部基准不同源**：pyiqa 的 `qalign` 行覆盖 3,000 张、本可用集 2,687 张，残差只能当量级旁证，不是同组复现。

## 训练污染审计（逐图证据）

**结论：干净（实测）。** 实际加载的 `q-future/q-align-iqa` 训练清单为 `train_koniq_spaq_kadid.json`（**5,821,869 B**、sha256 `c1acfc59…`、**23,943 条**）；其中 KonIQ **7,046 条**，与我们的 KonIQ 官方 test 2,015 张逐名比对，**双向差集 = 0**。另一份 `train_koniq.json`（2,841,794 B、7,046 条）的 KonIQ 图集合与前一份**逐张相同**，按实际加载的权重判定结论不变。

训练清单逐条可加总：KonIQ **7,046** ＋ SPAQ **8,897** ＋ KADID **8,000** = **23,943**，**没有第四条来源**；`agiqa` / `aigc` / `csiq` / `live` / `challenge` 均 **0 命中**。AGIQA-3K 的 **2,982 张**出现在评测清单 `agi.json`（与我们的 `agiqa3k_eval_list` 交集 2,982、双向差集 0）⇒ 是测试集不是训练集；`scripts/agi-5split-lora.sh` 的 LoRA adapter 不是本包加载的权重，未加载。

⇒ **未训练 AGIQA-3K（档位极高）；CSIQ / LIVE Challenge 均未命中（档位极高）。** TID2013 不在训练清单三源之内 ⇒ **TID2013 侧无训练污染**。

## 冒烟、耗时与显存

| 项 | 实测 |
|---|---|
| demo 图 | `fig/singapore_flyer.jpg`，**144,442 B**，sha256 `09f86ecd97a2a16a79a8ccbc3acc8d8fa435e53e34da6d9fe144083446d2c644`；1620×1080 → `expand2square` 1620×1620 |
| demo 输出 | **0.175049**（1–5 制 **1.7002**）；probs `{excellent 3e-06, good 0.000273, fair 0.020508, poor 0.658203, bad 0.320801}`（和 **0.999788**） |
| 确定性 | 两次运行逐字节相同，sha256 `fb636896dfa267c3bc3e500413e9b204e32ac292fc6aef46f70db4d880c7b7d4` |
| 加载 / 显存 | **20.8 s**（run1）/ **18.6 s**（run2）；**16,154 MiB** |
| 吞吐 / 墙钟 | KonIQ L4 ≈**0.1 s/图**、整条流水线 **5 min 12 s**（含加载 ×3）；TID2013 墙钟 **401 s**、≈**0.1 s/图**（三次加载 18.3 / 18.5 / 18.3 s） |

单图 0.175 偏低**不是实现 bug**：同一次会话的 KonIQ L4 给出 SRCC 0.9263（正相关），方向或实现若是反的，不可能正相关到这个量级 ⇒ 单图结论不外推。`0.1 s/图` 是本评委唯一的租卡端吞吐实测，本机没有锚点（权重装不下），算力只能直接用这个实测值，**不得外推**。

