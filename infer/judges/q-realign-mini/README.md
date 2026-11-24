# 评委：Q-ReAlign Mini（0.8B，Qwen3.5-VL 底座）

**A 组（等级期望型），0.8B。** 它是九个评委里**唯一 CPU 可跑、本机 GTX 1650（4 GB）装得下的 LMM**，因此是本课题本机验证的基准件——所有脚本的第一块试跑对象都是它。它与 Q-Align **同组、同数据、同配方，仅换底座**（MPLUG-Owl2 7B → Qwen3.5-VL 0.8B），训练数据直接指向 Q-Align 的同一个 `train_koniq.json`；因此它**不能**当作独立观测源，与 Q-Align 对照时可归因的只有「底座换了」这一处。

**污染状态**：在 **KonIQ-10k 上训练过**（`train_koniq.json`，官方 train 侧），但按「是否见过我们的评测图」这一口径，**0 张 ⇒ 干净**；**AGIQA 轴不可核（有风险）**，是九个评委里唯一在该轴判不了「否」的一个（见「已知限制」）。

## 复现对象与许可

**无独立论文**：出处只有官方仓库 [Q-Future/Q-ReAlign](https://github.com/Q-Future/Q-ReAlign)，其 README 称 Q-ReAlign 是 Q-Align 的 **"the retrained version on Qwen3.5-VL"**。`configs/onealign.yaml` 的 `source: /DATA/Q-Align/train_koniq.json` 是「同数据」的一手证据；`docs/METHOD.md` 记五级权重与「uses the `gpt` answer verbatim」。
官方 README 自述训练集为 ONE-ALIGN 六数据集 mix（KonIQ + SPAQ + KADID + AGIQA-20K + AVA + LSVQ），但同仓 yml 的 `mix:` **只列 5 个、缺 AGIQA-20K**，两处不一致如实登记。复现的是 `Q-ReAlign-Mini-0.8B`（HF：`q-future/Q-ReAlign-Mini-0.8B`），更大的 Q-ReAlign 变体未复现。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者侧仓库代码 | **官方未声明**（仓库无 LICENSE 文件；GitHub `license=None`，`contents/LICENSE` → 404，2026-10-14 一手） | **默认不可再分发** |
| 权重（HF 模型卡） | **Apache-2.0**（卡记 `license: apache-2.0`，经镜像，2026-10-16 一手） | 按卡可以 |
| 底座模型 | **Qwen3.5 族 0.8B**；底座许可 **Apache-2.0**（许可正文一手件，经镜像）。具体型号按 `config.json` 与 `Qwen/Qwen3.5-0.8B` 逐字段相同**推定**，非官方指名 | 按卡可以（底座型号为推定） |

完整许可见 `LICENSING.md`（逐项记录）与 `LICENSE-NOTES.md`（可否再分发）。

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | Qwen3.5-VL **0.8B**（`architectures = ["Qwen3_5ForConditionalGeneration"]`，`model_type: qwen3_5`） |
| 权重文件 | `model.safetensors`，**2,214,590,296 B**（474 分片加载） |
| SHA256 | `bde34df0375fff90d2dee716a127039c57d310c0c868b9f52f4fc2d1ead34aac` |
| 来源 | `q-future/Q-ReAlign-Mini-0.8B`（HF），`huggingface_hub.snapshot_download` 整目录快照，经 `hf-mirror.com` |
| 落地位置 | `D:\models\Q-ReAlign-Mini-0.8B\`（**不随本包释出**，权重留在发布方） |
| 加载要求 | 必须整目录加载：`tokenizer.json`、`processor_config.json`、`preprocessor_config.json`、`chat_template.jinja` 缺一不可 |

上游**未公布该文件的 sha256**，本包能做的核对只有一条：**卡上那份与本机副本逐位相同**（2026-10-15 实测）。这比「与上游摘要逐位相同」弱一档，如实登记。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 提示词 | `"How would you rate the quality of this image?"` ＋ 续写前缀 `"The quality of the image is"`；不生成整句，只取续写前缀后第一个词的分布 |
| 五级词 | `excellent / good / fair / poor / bad` |
| 打分 | 五级词末位 logit 的 softmax × 权重 `[1.0, 0.75, 0.5, 0.25, 0.0]` ⇒ 分数 ∈ [0,1] |
| 词 id | `tokenizer(" " + w).input_ids[0]`，**前导空格不可省**（Q-Align 系的既有坑） |
| dtype | **fp16**，由 `infer.py` 内部定死，**无 CLI 开关**（GTX 1650 无 bf16） |
| batch | **1**（全部封装均不支持 batch） |
| 输出 | `score` ∈ [0,1] ＋ `probs` 五级分布；五级分布是必需的中间量，不能只存标量 |

## 运行栈

| 项 | 值 |
|---|---|
| 本机栈（正典） | Python 3.11.15 / torch **2.6.0+cu124** / transformers **5.19.0** / GTX 1650 4 GB / PIL 11.1.0 |
| 卡上栈（非正典） | `envs/cardlocal`：Python 3.11.17 / torch 2.6.0+cu124 / transformers 5.19.0 / PIL 12.3.0 / RTX 3090 |
| 依赖要求 | 模型卡与 `config.json` 记 `transformers_version = 5.3.0` ⇒ 要求 **transformers ≥ 5.2/5.3**；卡上 `envs/qalign`（4.36.1）装不下本评委 |
| 预期告警 | 运行时两条 transformers 回退告警（`causal_conv1d` 与 `flash-linear-attention` 未装）**属预期，不要去装** |

正文的 L4 数字均出自**本机栈**；卡上那段只用于速率与栈差证据，不得替换正典数字。

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → `score` ＋ 五级 `probs`）与本 README。**不随包释出**：权重与整个模型目录、`download_weights.py`、`run_local.sh`、`run_l4.sh`、`make_smoke_images.py`、`real_images/`、各次 run 产物。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl
```

## 实测

**口径**：KonIQ-10k 官方 test split **2,015 张**（512×384 native）；TID2013 可用集 **2,687 / 3,000 张**（384×512 native，排除 i01/i25 全部 240 张 ∪ `_BROKEN_FILES.txt` 其余 73 张，TID2013 无官方 split）。MOS 分别为 `koniq10k_distributions_sets.csv` 的 `MOS` 列（0–100）与 `mos_with_names.txt`（0–9）。两者均在本机栈、fp16、`bs=1` 下取得。

| 指标 | KonIQ（2,015 张） | TID2013（2,687 张） |
|---|---|---|
| SRCC | **0.7583** | **0.7120** |
| KROCC | 0.5666 | 0.5248 |
| PLCC（raw） | **0.8088** | **0.7521** |
| PLCC（4 参数 logistic） | **0.8108** | **0.7522** |

KonIQ 2,015/2,015、TID2013 2,687/2,687 全部命中（TID2013 退出码 0）；预测范围分别为 **0.0511 .. 0.9831** 与 **0.0024 .. 0.9779**（0–1 制）。在本机栈的六个评委里，它的 TID2013 SRCC 最高（0.7120）。

**无对外基准可比**：pyiqa 的基准表里没有 `qrealign` 行（Q-Align 系只有 `qalign` / `qalign_4bit` / `qalign_8bit`）；官方仓库未发布 KonIQ 上的 SRCC/PLCC；Q-Align 论文的自报值属于**另一个 checkpoint**，不可借用。⇒ 上表只声明为「本机协议下的实测」，**不得**写成「复现了某基准」。本封装的硬判据只有两项：权重按本机副本逐位加载；分数落在 [0,1] 且梯度方向正确。

## 已知限制

1. **AGIQA 轴不可核（有风险）**。官方 README 自述的 ONE-ALIGN 六集 mix 含 AGIQA-20K；同仓 `configs/onealign.yaml` 的 `mix:` 只列 5 个（KonIQ / SPAQ / KADID / AVA / LSVQ）、缺 AGIQA-20K，且自注为参考配置（`source:` 全是 `/DATA/…` 占位）。HF 仓的 10 个文件全是权重 / tokenizer（`config.json` / `generation_config.json` / `model.safetensors` / `preprocessor_config.json` / `processor_config.json` / `tokenizer.json` / `tokenizer_config.json` / `chat_template.jinja` / `.gitattributes` / `README.md`），无 `training_args.bin`、无训练清单。AGIQA-20K ≠ AGIQA-3K，两者是否逐图重叠**没有取到任何清单**。⇒ 既不能写「干净」（名目重叠只是旁证、无逐图证据），也不能写「污染」（没有证据证明这份权重见过我们的 2,982 张评测图）。
2. **CSIQ / LIVE Challenge 均未命中**：官方六集枚举与同仓 yml 的 `mix: [koniq, spaq, kadid, ava, lsvq]` 都没有 csiq / live；同 yml 的 `live` 只出现在 `datasets:` 与 `eval.sets`，是**测试集不是训练集**。档位中高——该 yml 自注参考配置、`source:` 全是 `/DATA/…` 占位，不是实测。
3. **无对外基准可比**（见「实测」），只能声明为本机协议下的实测。
4. **两列不可直接比**：数据集、标注谱系（TID2013 是成对比较协议）、失真类型、MOS 量纲均不同；并列只为给跨域迁移留锚点，不构成「下降 = 复现变差」的判断。
5. **静默降级到 CPU 的风险**：`infer.py` 的 `build_model` 把 `(torch.cuda.OutOfMemoryError, RuntimeError)` 一并捕获后静默降级到 CPU fp32，只往 stderr 丢一行 `[warn]`，不改返回值与退出码。0.8B 在 CPU 上会慢到计时毫无意义，而 rc 仍为 0、行数仍对。**读产物 jsonl 的 `device` 字段是唯一能察觉降级的办法**；卡上实测 `device` 分布 = `{'cuda': 2015}`。
6. **必须传 `--list`**：KonIQ-10k 目录含全部 **10,073** 张，对目录 glob 会把训练集混进 test 口径。
7. **`bs=1`，单卡串行**：不要与其它评委同时跑；加载耗时与另一评委的推理窗口重叠时，该加载值不可引用。

## 冒烟对照（六张共用图，2026-10-07）

| 图 | 本机栈 | 卡上 `envs/cardlocal`（2026-10-15） | Δ |
|---|---|---|---|
| `p1_orig.png` | **0.904050** | 0.904050 | **0.000000** |
| `photo1.jpg` | **0.904050** | 0.910312 | **+0.006262** |
| `p1_blur1.png` | 0.369038 | 0.369011 | −0.000027 |
| `p1_blur2.png` | 0.087755 | 0.087755 | 0.000000 |
| `p1_jpeg8.png` | 0.261293 | 0.261293 | 0.000000 |
| `p1_noise.png` | 0.235682 | 0.235339 | −0.000343 |

质量梯度正确：原图 0.904 > 轻模糊 0.369 > JPEG8 0.261 > 噪声 0.236 > 重模糊 0.088。`p1_orig.png` 与 `photo1.jpg` 像素相同、仅编码不同，本机栈上两者逐位相同（同为 0.904050）；卡上差 +0.006262，根因是图像解码器版本差（本机 PIL 11.1.0 ↔ 卡上 PIL 12.3.0）：PNG 通道两机逐字节相同、JPEG 通道像素不同。这是运行栈的一项口径，不是「卡上跑歪了」。

## 与 CLIP-IQA 的对照

两者同为 [0,1] 名义量纲，但语义不同，排序可以冲突：`p1_jpeg8.png` 在本评委上排第三（0.261293），CLIP-IQA 把它与 `p1_noise.png` 的次序判反。六张图的冒烟**只说明值得做，不构成证据**。

## 耗时（本机栈，正典）

| 项 | KonIQ（2,015 张） | TID2013（2,687 张） |
|---|---|---|
| 加载 | 92.1 s | 60.3 s（474 分片） |
| 稳态 | 1.8 s/图（首图 55.1 s 预热） | 中位 1.8 s/图 |
| 墙钟 | ≈60 min | 83 min 45 s |

卡上 `envs/cardlocal`（RTX 3090）的 `bs=1` 速率为 **197.7 s**（加载 7.3 s，稳态 **0.0945 s/图**），26,180 张外推 ≈0.69 h；该段**不是**正典 L4，只用于速率与栈差证据。
