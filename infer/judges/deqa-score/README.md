# 评委：DeQA-Score（LMM 打分型，mPLUG-Owl2-7B / Mix3 权重）

**A 组（LMM 打分型）第三个评委**。选它是因为它与 Q-Align 构成本课题唯一一对
「同底座、异血缘」的纯血缘对照：两者同为 mPLUG-Owl2 底座，但分属不同研究组
（DeQA 作者与 Q-Future 组作者名单无交集）。
**污染状态**：模型卡写明权重在 **KonIQ、SPAQ、KADID** 上全量微调。KonIQ 训练清单
（7,046 条）与本次实测的官方 test 2,015 个文件名逐名比对，**交集为 0**（实测）；
SPAQ / KADID 同样被训练过，但训练 meta 未取回、污染范围未量化——这两个数据集上的
结论不得用于 DeQA。

## 复现对象与许可

Zhiyuan You, Xin Cai, Jinjin Gu, Tianfan Xue, Chao Dong. *DeQA-Score: Teaching Large
Multimodal Models to Evaluate Image Quality with Arbitrary Score Levels*. arXiv:2501.11561
（预印本；原文未记会议 / 期刊信息）。官方仓库：
[zhiyuanyou/DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score)；权重：
[zhiyuanyou/DeQA-Score-Mix3](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3)。

| 对象 | 许可 | 能否商用 |
|---|---|---|
| 官方仓库代码 | **MIT** | 可以 |
| 权重（Mix3） | 模型卡记 **mit** | 按卡可以 |
| 底座 mPLUG-Owl2（LLaMA-2-7B + ViT-L） | 模型卡记 **apache-2.0**；官方件中未见援引 LLaMA-2 条款 | 按卡可以；LLaMA-2 一层须另行确认 |

许可见 `LICENSING.md` 与 `LICENSE-NOTES.md`（2026-10-16 经镜像取回）。

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | mPLUG-Owl2：CLIP ViT-L 视觉编码器 + visual abstractor + LLaMA-2-7B |
| 权重仓 | [`zhiyuanyou/DeQA-Score-Mix3`](https://huggingface.co/zhiyuanyou/DeQA-Score-Mix3)（HuggingFace） |
| 权重文件 | 4 片 safetensors：`model-00001-of-00004` **4,981,079,736 B** / `-00002-` **4,920,263,064 B** / `-00003-` **4,989,501,960 B** / `-00004-` **1,518,368,296 B** |
| 合计 | **16,409,213,056 B（15.28 GiB）**；`model.safetensors.index.json` 记 **862 个参数键**、`metadata.total_size = 16409100288`，与实测字节数一致 |
| 取法 | `snapshot_download(..., ignore_patterns=["*.bin","*.h5","*.msgpack","*.ot"])`。该仓 `.safetensors` 与 `.bin` 两套并存，全下约 **32 GB**，`ignore_patterns` 只要一半 |
| 落地路径 | `/root/autodl-tmp/models/deqa-score-mix3`（**不随本包释出**） |
| 加载代码 | HF 仓自带 `modeling_mplug_owl2_huggingface.py`，**19,710 B**，SHA256 `4d31a7dd3298ea70bed6b68e3d0dd7c14285676ffa8838b3262d3121b3802745`（2026-10-13 从 `hf-mirror.com` 取回，两次逐字节相同） |
| 特殊加载项 | `trust_remote_code=True`；fp16；无量化；注意力后端钉在 `config._attn_implementation="eager"` |

加载时须把 `config._name_or_path` 从 hub id 改写为本地 ckpt 目录，否则 tokenizer 与
image processor 会去连 huggingface.co；`infer.py` 以 `HF_HUB_OFFLINE=1` 兜底，覆盖失效时
当场报错。注意力后端**不能**作为 `from_pretrained` 的 kwarg 传入——transformers 4.36.1
会把它透传给只收 `config` 的 remote-code 模型类。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 预处理 | `expand2square`（`modeling_mplug_owl2_huggingface.py:62`），背景色取 image processor 的 `image_mean` |
| 五级 token | `excellent/good/fair/poor/bad` → `[15129, 1781, 6534, 6460, 4319]`（与官方 `preferential_ids_` 逐项相同） |
| 打分 | `softmax(logits[level_ids]) @ [5,4,3,2,1]`，**原生 1–5 制，不做仿射变换** |
| 分布 | `infer.py` 逐字复刻 `score()` 到 softmax 之前，缓存五级 `probs`（softmax 后）与 `logits`（softmax 前） |

prompt 逐字符为 `USER: How would you rate the quality of this image?` + 换行 + `<|image|>` + 换行 + `ASSISTANT: The quality of the image is`（行尾无空格，`<|image|>` 单独一行）。
与 Q-Align 的 0–1 制并排读时用 `score(1–5) = 4 · score(0–1) + 1`；本封装刻意不做换算，
因为量纲转换属指标层，且是否可比正是待检验的假设。

## 运行栈

本评委的全部实测数字出自**租卡栈**（`envs/deqa` 由 `envs/qalign` clone，以保证跨评委数值栈逐位相同）：

| 项 | 值 |
|---|---|
| torch / transformers | **2.0.1** / **4.36.1** |
| 其余官方 pin | torchvision 0.15.2、tokenizers 0.15.0、sentencepiece 0.1.99、accelerate 0.21.0、timm 0.6.13、numpy <2 |
| 卡型 / dtype | **RTX 3090**，fp16，bs=1 |
| 未声明但必需 | `icecream`（模块级 import）与 `protobuf`（`LlamaTokenizer` 走 sentencepiece 慢分词）；`pip install transformers` 不会带入 |

**跨栈数字不可混比**：本评委的数只与同一租卡栈的比较。模型卡称 transformers
4.46.3 / 4.57.6 需改代码、4.36.1 不用；实测 4.36.1 下仍须按上节钉住注意力后端。

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → 分数 + 五级 probs/logits）与本 README。
**不随包释出**：`setup_env.sh`、`run_l4.sh`、权重下载命令、权重文件、推理缓存。

本评委**没有 `vendor_*` 目录**：推理走 HF 路径，`modeling_mplug_owl2_huggingface.py`
等建模代码由权重仓自带、经 `trust_remote_code=True` 加载，不需要 clone GitHub 仓库，
也没有另打补丁；权重由 `snapshot_download` 从 `zhiyuanyou/DeQA-Score-Mix3` 直接取。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl [--list 名单.txt] [--limit N]
```

## 实测

两组数据均出自同一租卡栈（torch 2.0.1 / transformers 4.36.1 / RTX 3090 / fp16 / bs=1）。

| 指标 | KonIQ-10k 官方 test | TID2013 可用集 |
|---|---|---|
| 样本 | **2,015 张** | **2,687 / 3,000 张** |
| SRCC | **0.9201** | **0.8318** |
| KROCC | **0.7591** | **0.6370** |
| PLCC（raw） | **0.9398** | **0.8527** |
| PLCC（4 参数 logistic） | **0.9401** | **0.8622** |

- **KonIQ-10k**：官方 `set` 列的 test 2,015 张，MOS 取 `koniq10k_distributions_sets.csv`
  的 `MOS` 列。用 `--list` 按名单取图（目录下有全部 10,073 张，glob 会把训练集混进来）。
  一次跑通 2015/2015；墙钟 **6 min 06 s**（含加载 **23.7 s**），稳态 **0.1–0.2 s/图**。
- **口径核对**：`gt_score` 逐图精确等于 `koniq10k_scores_and_distributions.csv` 的 `MOS`
  （r = 1.00000000，斜率 1.000，最大残差 1.3e-15）；DeQA 的 KonIQ test（2010 条）严格
  包含于本次实测的 2,015 张（A−B = 5、B−A = 0）。
- **TID2013**：可用集 2,687 张（排除 **313 张** = 内容 i01/i25 全部 **240** 张 ∪ 解压
  失败清单 **73** 张净增项）；原生 **384×512** BMP，未做格式转换；MOS 取
  `tid2013_eval_labels.csv` 的 `MOS` 列（**0–9 制**，可用集实测范围 0.2424 .. 7.2143）；
  TID2013 **无官方 split**，`set` 一律记 `test`（口径决策，不是官方 test）。预测范围
  **1.1768 .. 4.5273**（1–5 制）；墙钟 **399 s**，三次加载各 **9.1 / 9.1 / 9.2 s**，
  **≈0.1 s/图**。

证据件 SHA256：TID2013 名单（2,687 行，LF）`cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`；图集压缩包 1,111,692,036 B `0db74247d9c5b388fd7b90f469fb27929bc7461e26299bdae12642357e8b271c`；
预测缓存 944,714 B `6e22ee53ec446fd7125015a33d42be3df9667361db9e7645e0c14f43a6b57ba3`。

## 与 pyiqa 基准的对照

pyiqa 的基准 CSV **没有 `deqa` 行** ⇒ 本评委没有任何外部基准可对照，不得写
「复现了某基准」。模型卡自报 Non-reference IQA 为 KonIQ **PLCC 0.956 / SRCC 0.943**
（同表 Q-Align 0.945 / 0.938），但口径三件事只核了 **split**（模型卡的 KonIQ test
2010 条严格包含于本次实测的 2,015 张）；**输入口径**（分辨率 / 预处理 / `expand2square`
背景色）与 **指标算法**（PLCC 是否做非线性拟合）**未核** ⇒ 自报值只当旁证。

## 已知限制

1. **污染范围**：KonIQ 已实测零交集，但「`train_koniq_7k.json` 即 Mix3 的 KonIQ 训练
   清单」属推定（模型卡未点名）；SPAQ / KADID 的 meta 未取回 ⇒ 按「污染、范围未量化」处理。
2. **TID2013 无官方 split**：`set` 一律记 `test`，是口径决策；i01/i25 的选择性偏差未检验。
3. **跨数据集不可直接比**：KonIQ 与 TID2013 的 SRCC 差 **0.0883**（0.9201 vs 0.8318），
   但数据集、标注谱系、MOS 量纲均不同 ⇒ 不构成跨域结论。
4. **无外部基准**：pyiqa 基准 CSV 无 `deqa` 行；模型卡自报值仅当旁证（见上节）。
5. **本机装不下**：本机 GTX 1650 **4 GB**、无 bf16，权重约 **16 GB**（差约 4×）⇒ 权重
   加载与前向全部在租卡机完成；本包不带权重，复现需自备约 16 GB 显存的卡。
6. **路径选择**：官方另有 builder 路径（`src/evaluate/iqa_eval.py`），只 dump 每等级 token
   的 logits、不算分数；本包复现的是模型卡 quick-start 的 HF 路径，两者不得混称「官方分数」。

## 冒烟与对拍

| 判据 | 结果 |
|---|---|
| 官方 demo 图 `singapore_flyer.jpg` 标称 `score` | **1.9404**（1–5 制，模型卡原文）；图 **144,442 B**，SHA256 `09f86ecd97a2a16a79a8ccbc3acc8d8fa435e53e34da6d9fe144083446d2c644` |
| 实测 | **1.940430**，与标称差 **3.00e-05**（判据 ≤ 1e-3）；两次调用逐位相同；同时覆盖 prompt + tokenizer + 预处理 + 权重四段链路 |
| 对拍 | 与官方实现：`tokenizer_image_token` 逐 id 相同（28 vs 28）、`preferential_ids_` 相同、`WEIGHTS = [5,4,3,2,1]`、注意力后端 **eager**；上游函数：`tokenizer_image_token()` **18 例**、`expand2square()` **7 例**全部一致（AST 抽源码逐用例比对，本机完成） |
