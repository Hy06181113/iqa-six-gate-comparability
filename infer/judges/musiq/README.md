# 评委：MUSIQ（C 组回归型，KonIQ-10k 支）

MUSIQ 属于 **C 组（回归型）**，输出量纲约 **[0, 100]**，与同组 TOPIQ-NR / MANIQA 的 [0,1] **不可直接比大小**。
选择它是因为它与 MANIQA 构成「同机制、异血缘、同污染」的组内对照点：同组、同污染状态，但训练代码各自独立（MANIQA 的训练代码反而借自 MUSIQ 的非官方实现）。
**污染状态：已污染。** 它在 KonIQ-10k 上训练，用全库随机 80/20、跑 10 次、未遵守官方 split，**种子未公布 ⇒ 无法重建干净子集**，只能保守记账（估计覆盖约 80% 的 test 图，无法逐张确认）。

## 复现对象与许可

Ke, Wang, Wang, Milanfar, Yang. *MUSIQ: Multi-Scale Image Quality Transformer*, ICCV 2021, pp. 5128–5137.
官方实现：[google-research/musiq](https://github.com/google-research/google-research/tree/master/musiq)（TensorFlow）。
**本封装复现的是 pyiqa（chaofengc/IQA-PyTorch）的 PyTorch 重实现，不是 Google 原版**——C 组各评委统一走同一份权重仓库与同一套推理约定，否则「同组」对照不再干净。这一取舍须写进论文 Limitations（「复现的是社区重实现」）。pyiqa 里有 `musiq`（KonIQ-10k）/ `musiq-ava` / `musiq-spaq` / `musiq-paq2piq` 四支，本封装只复现 **KonIQ-10k 支**（`pretrained='koniq10k'`），其余三支未复现。

| 对象 | 许可 | 能否商用 |
|---|---|---|
| 官方 TF 代码（`google-research` monorepo） | **Apache-2.0** | 可商用（仅作旁证，本封装未使用） |
| 本封装实际复制的 pyiqa 代码 | **PolyForm Noncommercial License 1.0.0**（Required Notice: Copyright (c) 2022 Chaofeng Chen） | **不可商用**；再分发须随附许可全文与 Required Notice |
| 权重 `musiq_koniq_ckpt-e95806b9.pth` | pyiqa 权重仓 **`cc-by-nc-sa-4.0`**（混合来源仓，非一揽子授权） | **不可商用** |
| 底座模型 | **无外部底座**（MUSIQ 自含多尺度架构，非预训练主干微调） | — |

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | 无外部骨干：5 层 ResNet patch encoder ＋ 轻量 Transformer，**全部参数都在这份 ckpt 里** |
| 权重文件 | `musiq_koniq_ckpt-e95806b9.pth` |
| 字节数 | **108,610,983 B** |
| SHA256 | `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa` |
| 来源 | pyiqa 权重仓 `chaofengc/IQA-PyTorch-Weights`（上游亦为 chaofengc/IQA-PyTorch 的 release 资产）；本机经 `hf-mirror.com` 取：`https://hf-mirror.com/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth` |
| 落地路径 | `D:\models\MUSIQ\musiq_koniq_ckpt-e95806b9.pth`（**不随本包释出**） |
| 加载键 | ckpt 为**裸 `OrderedDict`**（248 个参数键，非 `{'params': {...}}` 包裹）⇒ `weight_keys=None`；`strict=True` 加载，vendored 架构键集与 ckpt 键集**逐键一致** |
| 特殊加载项 | `patch_size=32`、`patch_stride=32`、`longer_side_lengths=[224, 384]`、`max_seq_len_from_original_res=-1`（原分辨率尺度全取、不截断）、`num_class=1`、`hse_grid_size=10`、`pretrained='koniq10k'` |

**权重出处记账（不得拔高）**：pyiqa 的 `benchmark.rst` 把「不提供官方权重、由我们重训」的 NR 模型列为 `cnniqa` / `dbcnn` / `hyperiqa`，**MUSIQ 不在这一列** ⇒ pyiqa 把这份权重当作既有的发布权重。但它是从 Google 原版 TF checkpoint 转换来的、还是 pyiqa 自己重训的，`benchmark.rst` / `ModelCard.md` / pyiqa README **三份文档都没有写**。⇒ 记账口径为「pyiqa 发布的 MUSIQ KonIQ 权重，转换或重训未说明」，**不得写成「Google 官方权重」**；倾向是移植而非重训，但未明文说明。

**完整性校验**：文件大小 = `X-Linked-Size` = 108,610,983 B；SHA256 前 8 位 `e95806b9` 与 HF 的 `X-Linked-Etag` 逐位相同。`--inspect-ckpt` 实测顶层 `OrderedDict`、248 键、架构键集 == ckpt 键集 `True`。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| 输入 | RGB → float32 [0,1] 的 CHW；**不做缩放、不做裁剪、不做外部归一化**，原分辨率进模型 |
| 输入结构 | **多尺度分块序列**：224 / 384 / 原分辨率三组 patch 拼接（不是单张整图） |
| 归一化 | `(x − 0.5) × 2 → [−1, 1]`，在模型 `forward` **内部**完成（**不是** ImageNet mean/std） |
| 打分 | `num_class=1` ⇒ `dist_to_mos` 退化为恒等，输出即最后一层标量；名义区间 **`~0, ~100`** |
| 官方子集 | 取 test 口径必须传 `--list` 名单：KonIQ-10k 目录含全部 10,073 张，直接 glob 会把训练集混进来 |

## 运行栈

本 README 的数字分属两段栈；**跨栈数字不得混入同一张表**。

| 栈 | Python | torch | 卡 | dtype |
|---|---|---|---|---|
| 本机栈 | 3.11.15 | **2.6.0+cu124** | GTX 1650 4 GB | fp32 |
| 租卡栈 | 3.10.22 | **2.0.1+cu117** | RTX 3090 | fp32 |

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → 分数 ＋ 分布标量）与本 README。**不随包释出**：`vendor_musiq/`、权重下载脚本 `download_weights.py`、`run_local.sh`、`runs/` 缓存，以及权重文件本身。

`vendor_musiq/` 五份文件全部取自 pyiqa commit **`6e48c5ca9e3e58bd632af52f06a2ca976ccf2722`**（2026-08-30，取回通道 `cdn.jsdelivr.net`）：

| 文件 | 上游路径 | 改动 | 正文 sha256（去头部注释后） |
|---|---|---|---|
| `musiq_arch.py` | `pyiqa/archs/musiq_arch.py` | 导入拉平 3 处 ＋ 去掉 `@ARCH_REGISTRY.register()`；**模型定义一字未改** | `acaaa597…41fd6c2` |
| `multiscale_trans_util.py` | `pyiqa/data/multiscale_trans_util.py` | **无** | `6d13c02e…904133`（＝上游 blob） |
| `matlab_padding.py` | `pyiqa/matlab_utils/padding.py` | **无**（仅改文件名） | `51693f91…bdd4f3`（＝上游 blob） |
| `arch_util.py` | `pyiqa/archs/arch_util.py` | 继承 MANIQA 副本的两处补丁（删远程下载导入、URL 分支改抛错） | `6764807c…58f0393`（与 MANIQA 副本相同） |
| `constants.py` | `pyiqa/archs/constants.py` | **无** | `811a450a…5f16681`（＝上游 blob） |

只 vendor `matlab_utils/padding.py` 一个文件（而非整个包），是因为上游 `matlab_utils/__init__.py` 会连带导入 `resize` / `functions` / `scfpyr_util`，其中 `scfpyr_util` 需要本封装用不到的依赖；MUSIQ 只用 `ExactPadding2d` / `exact_padding_2d`。五份文件行尾均为 **LF**。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl
python infer.py --image <目录> --list <名单.txt> --out <缓存>.jsonl
```

## 实测

### KonIQ-10k 官方 test split（本机栈）

口径：KonIQ-10k 官方 test split **2,015 张**（按 `koniq10k_distributions_sets.csv` 的 `set` 列），分辨率 **512×384**（native，不缩放不裁剪），MOS 取同一 CSV 的 `MOS` 列（0–100），raw 与 4 参数 logistic 两条都算；2,015 / 2,015 全部命中。

| 指标 | 本机栈（2,015 张，512×384 native） | 租卡栈（同 2,015 张，同协议） |
|---|---|---|
| SRCC | **0.8654** | **0.8699** |
| KROCC | **0.6817** | **0.6872** |
| PLCC (raw) | **0.8958** | **0.8977** |
| PLCC (log) | 0.8996 | — |

预测范围 **14.7191 .. 78.3511**（官方标签 4.1674 .. 88.2438，0–100 制）。

### TID2013 可用集（本机栈）

口径：TID2013 **可用集 2,687 / 3,000 张**（已排除内容 i01/i25 全部 240 张 ∪ `_BROKEN_FILES.txt` 其余 73 张）；TID2013 **无官方 split**，`set` 一律 `test`（对可用集整体评估）。分辨率 **384×512**（原生，不缩放不裁剪）；MOS 取 `mos_with_names.txt`（0–9 制，实测 0.24242 .. 7.21429，无缺失）；2,687 / 2,687 全部命中。标签件 sha256 `5a765bfe1bcaf63e773718714602c4c4886357332b97014934e23fe5e34af8d3`；名单件（2,687 行）sha256 `cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`。两者不随本包释出。加载 0.5 s；单图中位 0.0500 s/图、均 0.0577 s/图；墙钟 2.8 min（全部 384×512）。

| 指标 | KonIQ（2,015 张，本机栈） | TID2013（2,687 张，本机栈） |
|---|---|---|
| SRCC | 0.8654 | **0.5896** |
| KROCC | 0.6817 | **0.4261** |
| PLCC (raw) | 0.8958 | **0.6948** |
| PLCC (log) | 0.8996 | **0.6950** |

TID2013 预测范围 **11.8962 .. 78.5155**。两列**不可直接比**：数据集、标注谱系（TID2013 是成对比较协议）、失真类型与 MOS 量纲均不同，并列只为给跨域迁移留锚点。逐图分数见本包 `scores/p2_koniq10k.csv` 与 `scores/p2_tid2013.csv` 的 `musiq` 列。

**耗时（本机栈，GTX 1650，512×384 native，fp32）**：2,014 张预热后单图均值 **0.0450 s**（中位 / 最大 / 最小 = 0.040 / 0.090 / 0.030 s）；2,015 张合计 **≈91 s**（含加载 0.5 s），比同组 MANIQA（3.891 s/图）快 86 倍。同一封装换到租卡 RTX 3090：**≈0.05 s/图**（稳态 0.02–0.08），**≈1.0×，完全无加速**——`bs=1` 下由逐图固定开销主导，不是被算力卡住的（MANIQA 在 3090 上 ≈0.400 s/图，≈9.7×；其 3090 侧用 pyiqa 自身代码，只作方向参考）。

## 与 pyiqa 基准的对照

pyiqa 仓库自带 `tests/NR_benchmark_results.csv`，是用它自己发布的权重、在官方 test split 上实测的三元组（协议见 `docs/benchmark.rst`：NR 测 KonIQ-10k official split、PLCC 不做校正、full image single input）。

| 来源 | SRCC | KROCC | PLCC (raw) | 栈 |
|---|---|---|---|---|
| pyiqa 基准 CSV | 0.8654 | 0.6817 | 0.8958 | — |
| 本封装 · **本机栈** | **0.8654** | **0.6817** | **0.8958** | Python 3.11.15 / torch 2.6.0+cu124 / GTX 1650 |
| 本封装 · 租卡栈 | **0.8699** | **0.6872** | **0.8977** | Python 3.10.22 / torch 2.0.1+cu117 / RTX 3090 |
| pyiqa 自身代码 · 租卡栈 | **0.8699** | **0.6872** | **0.8977** | 同上 |

- **本机栈上三项到小数点后 4 位与 pyiqa 基准 CSV 完全相同**（残差 0.0000）⇒ 可写「与 pyiqa 基准 CSV 一致到 1e-4」，**但必须同时写明栈**：该一致性**只在本机栈成立**。
- 同一封装、同一批图、图像 sha256 逐字节相同、协议字段逐项相同（dtype / patch_size / longer_side_lengths / max_seq_len / num_class / seed / weight_keys / img_size 全同），**仅换栈**：2,015 张逐图分数**全部不同**（中位差 **0.363**、最大 **2.995**，0–100 量纲），**SRCC 位移 ≈5e-3**——比 0.0000 大得多 ⇒ **不得**不加栈条件地宣称「复现了基准」。
- 这不是实现错：pyiqa 自身代码在同一租卡栈上给出 PLCC 0.8977 / SRCC 0.8699 / KROCC 0.6872，与本封装**到小数点后 4 位完全相同**；逐图核对 2,015 张，**最大差 5.0e-7、中位 2.4e-7**（float32 精度级）⇒ **本封装与 pyiqa 自身实现数值等价**，≈5e-3 的位移完全来自**运行栈**（torch 版本 / 卡型 / 算子实现）。

**TID2013 侧（旁证，非判据）**：pyiqa 同一 CSV 的 `tid2013` 列中 `musiq-koniq` 行为 PLCC **0.6814** / SRCC **0.575** / KRCC **0.4131**；本机为 0.6948 / 0.5896 / 0.4261 ⇒ 残差 **+0.0134 / +0.0146 / +0.0130**，三项同号。但 pyiqa 的 TID2013 meta_info 覆盖全部 3,000 张、我们只跑 2,687 张，且输入口径只做了自述核对、未逐项实测 ⇒ **只作旁证，不作「复现成功」的对外宣称，也不把 pyiqa 的数字当门槛**。

## 已知限制

1. **污染不可逆**：MUSIQ 用全库随机 80/20 训练、未遵守官方 split，种子未公布 ⇒ **无法重建干净子集**；本包数字是在被污染评委上取得的，只能保守记账，**不得**当干净基线用。
2. **权重出身未明、且复现的是社区重实现**：pyiqa 发布的 KonIQ 权重是「转换或重训未说明」，**不得**写成「Google 官方权重」；本封装复现的是 pyiqa 的 PyTorch 重实现，不是 Google 原版 TF 实现，论文 Limitations 须写明。
3. **TID2013 集合不同**：pyiqa 覆盖全部 3,000 张，我们只跑 2,687 张（其评测时是否再取子集未核）；**i01/i25 的选择性偏差未检验**，TID2013 数字不得默认与其余 23 个内容同分布。
4. **两列不可直接比**（KonIQ vs TID2013）：数据集、标注谱系、失真类型、MOS 量纲均不同，并列只为留锚点，不构成「下降 = 复现变差」的判断。
5. **跨栈数字不得混入同一张表**：同一评委同一批图仅换栈，SRCC 位移 ≈5e-3，已大于本机栈上的基准残差 0.0000。
