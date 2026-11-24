# 评委：CLIP-IQA（training-free）

**B 组（相似度型）**。九个本地评委里 **8 个在 KonIQ-10k 上训练过，唯一零样本的是它**——
因此它是本课题中唯一**确定没有训练污染**的评委，也是最早暴露「效度缺口」的观察点。

## 复现对象与许可

Wang, Chan, Loy. *Exploring CLIP for Assessing the Look and Feel of Images*, AAAI 2023
（arXiv:2207.12396）。官方实现：[IceClear/CLIP-IQA](https://github.com/IceClear/CLIP-IQA)。
复现的是官方 **`CLIPIQAFixed`** 那一支（无需训练）。

| 对象 | 许可 | 说明 |
|---|---|---|
| 作者侧仓库代码 | **S-Lab License 1.0** | **不可商用** |
| 上游 `vendor_clip/`（MMEditing） | **Apache-2.0** | 可商用；license 正文取自 `@master`，本课题拷的是 `v2-3.8` 分支（该 ref 取不到，两 ref 未逐字比对） |
| CLIP RN50 权重 | **MIT**（`openai/CLIP`）＋ 官方 URL 直下 | 可商用 |

## 实际加载的 checkpoint

| 项 | 值 |
|---|---|
| 骨干 | OpenAI CLIP **RN50** |
| 权重文件 | `RN50.pt`，**255,827,503 B**，SHA256 校验通过 |
| 来源 | `openaipublic.azureedge.net`，URL 取自官方 `clip.py` 的 `_MODELS["RN50"]` |
| 落地位置 | `D:\models\CLIP-RN50\RN50.pt`（**不随本包释出**，权重留在发布方） |
| 位置编码 | **关掉**（`pos_embedding=False`）——论文核心技巧 |

「关掉位置编码」的作用：`AttentionPool2d` 原有的位置编码按 224×224（7×7 特征图）固定，
关掉后注意力池化对特征图尺寸无约束，**因此可以喂原分辨率图**；这也解释了官方评测
pipeline 为什么不缩放。

## 预处理与打分口径

| 项 | 值 |
|---|---|
| prompt 对 | `["Good photo.", "Bad photo."]`（单对，取自官方 config 的 `classnames`） |
| 打分 | `softmax([s_good, s_bad])[0]`，即「像好照片」的概率 ∈ [0,1]（`CLIPIQAFixed.forward` 的 `probs[:, 0]`） |
| 预处理 | **`--preprocess native`：原分辨率 ＋ CLIP 归一化，不缩放不裁剪** |

官方全程不缩放的证据有两条：`restoration_inference()`（`process_type=0`，即
`clipiqa_koniq_demo.py` 走的那条）用的 `cfg.test_pipeline` 为
`LoadImageFromFile(unchanged) → RescaleToZeroOne → Normalize(CLIP mean/std) → ImageToTensor`，
无 Resize/Crop；作者自己的 `resize_process()` 里 `Resize(...)` 与 `CenterCrop(...)` 两行是注释掉的。
⇒ `--preprocess clip`（Resize 224 ＋ CenterCrop）是敏感性对照，**不是**官方协议。

## 运行栈

| 项 | 值 |
|---|---|
| 本机栈 | Python 3.11.15 / torch **2.6.0+cu124** / GTX 1650 4 GB，`dtype` 自动取到 **fp16** |
| 框架 | 只用作者的补丁版 CLIP 模块，自写推理循环（等价于 `CLIPIQAFixed.forward`）；**不装 mmcv / mmengine**——装整套 MMEditing 0.16.1 会与 torch 2.6 冲突，收益为零 |
| 环境适配 | `clip.py` 原 `from pkg_resources import packaging`，setuptools ≥ 81 已删该模块 ⇒ 改从独立 `packaging` 取 `version.parse`；原代码只用它做一次 torch 版本告警，不改变数值行为 |
| stdout 编码 | `infer.py` 把 stdout/stderr 改成 UTF-8——Windows 控制台默认 GBK，装不下 `⇒` `≤`，不补则 `UnicodeEncodeError`。纯输出路径：100 张重跑与改前逐条相同、差异 0 处 |

## 本包包含什么

本目录随包只有 `infer.py`（推理封装：图 → 分数 ＋ 二类概率 ＋ logits）与本 README。
**不随包释出**（理由见 `LICENSE-NOTES.md`）：

- `vendor_clip/`——拷自 IceClear/CLIP-IQA **v2-3.8** 分支 `mmedit/models/components/clip/` 的四个文件
  （`clip.py` / `model.py` / `simple_tokenizer.py` / `bpe_simple_vocab_16e6.txt.gz`）。
  **必须 vendor**：`pos_embedding` 开关只存在于作者改过的 `model.py`，OpenAI 原版（含 JIT 归档）没有；
  不 vendor 就不是复现，而是另一个方法。`infer.py` 的 `from vendor_clip import clip` 需该目录存在。
- `download_weights.py`（16 线程 Range ＋ 断点续传 ＋ SHA256 校验）与权重文件本身。

```bash
python infer.py --image <图或目录> --out <缓存>.jsonl
```

## 实测：KonIQ-10k 与 TID2013

**口径**：KonIQ-10k 官方 test **2,015 张**（512×384）；TID2013 可用集 **2,687 / 3,000 张**（384×512），
排除内容 i01/i25 全部 **240** 张 ∪ `_BROKEN_FILES.txt` 其余 **73** 张。TID2013 **无官方 split**，
`set` 一律记 `test`（对可用集整体评估）。MOS 取 `mos_with_names.txt`（0–9 制，实测 0.24242 .. 7.21429，无缺失）。
标签件 sha256 `5a765bfe1bcaf63e773718714602c4c4886357332b97014934e23fe5e34af8d3`；
名单件（2,687 行）sha256 `cad2367a400c7daac8ac15f8e6cb862cca5ca91d0eb2a4ef608de19992cf4815`。
名单由排除清单算出，**不是**对目录 `*.bmp` glob。两项均不进本包（`scores/` 只带每图标量）。

| 指标 | KonIQ（2,015 张，512×384 native） | TID2013（2,687 张，384×512 native） |
|---|---|---|
| SRCC | 0.5894 | **0.5224** |
| KROCC | 0.4182 | **0.3676** |
| PLCC（raw） | 0.6412 | **0.6320** |
| PLCC（4 参数 logistic） | 0.6433 | **0.6423** |

TID2013 预测范围 **0.0180 .. 0.9964**；加载 **2.2 s**（fp16）；日志只打印 `n/2687` 进度、
不打印单图秒数 ⇒ 只有含加载与写盘的有效吞吐 **≈0.196 s/图**；墙钟 **8.8 min**，2,687/2,687 全部命中、退出码 0。

冒烟参考（6 张 512×512，本机 GTX 1650）：加载 2.0 s、**0.25 s/图**。

## 与 pyiqa 基准的残差：已查明是 prompt 口径差

同一张 pyiqa 基准表 `tests/NR_benchmark_results.csv` 里，`clipiqa` 行在 KonIQ 上为
PLCC **0.7211** / SRCC **0.6572**，在 `tid2013` 列为 PLCC **0.6471** / SRCC **0.5786** / KRCC **0.4107**。
本机对应为 0.6412/0.5894（KonIQ）与 0.6320/0.5224/0.3676（TID2013）⇒ 残差
**−0.0799 / −0.0678**（KonIQ）与 **−0.0151 / −0.0562 / −0.0431**（TID2013），三项同号且为负。

根因 = **prompt 集合**：pyiqa 的 `clipiqa` 用**它自己拼的 10 个 prompt（5 对取平均）**，
源码自注 *"Different from original paper, we assemble multiple prompts to improve the results"*；
本封装按论文/官方 demo 用**单对** `['Good photo.', 'Bad photo.']`，其余逐项相同
（backbone / `pos_embedding` / 归一化 / `imread2tensor` 不缩放不裁剪 / 打分公式，均已回源码核对）。
**正向确认**：把本封装只换成那 10 个 prompt ⇒ KonIQ 上 **0.7208 / 0.6566 / 0.4733**，与 pyiqa 基准差 **≤6e-4**。

⇒ 这 −0.068 是**口径差**，不是实现残差；CLIP-IQA **不得**写「复现了 pyiqa 基准」，
**也不得**把该差当「未解释残差」挂账。TID2013 侧未做同款 prompt 对照 ⇒ 那里的 −0.056
是「同因同向」的推断，**不是实测归因**。

## 已知限制

1. **集合不同**：pyiqa 的 TID2013 meta_info 按 `mos_with_names.txt` 逐行生成，覆盖**全部 3,000 张**
   （不排除 i01/i25、不排除损坏文件）；我们只跑 2,687 张。其评测时是否再取子集，未核。
2. **i01/i25 的选择性偏差未检验**——本节数字不得默认与其余 23 个内容同分布。
3. **两列不可直接比**：数据集、标注谱系（TID2013 是**成对比较**协议）、失真类型、MOS 量纲均不同。
   并列只为给跨域迁移留锚点，**不构成「下降 = 复现变差」的判断**。
4. **本评委先验就弱**（KonIQ 上 SRCC 0.5894，九个里最低一档）⇒ 它在 TID2013 上掉幅最小（−0.067）
   **不得解读成「更跨域稳定」**：低起点本身压缩了可掉空间，这是本课题要研究的混淆项。

## 预处理敏感性（同一评委内部）

同一批图，仅换预处理即可翻转排序：

| 图 | native（官方协议，512） | clip224（对照） |
|---|---|---|
| p1_orig | 0.963 | 0.979 |
| p1_blur1 | 0.848 | 0.920 |
| p1_blur2 | 0.302 | **0.867** |
| p1_jpeg8 | 0.207 | **0.877** |
| p1_noise | 0.696 | **0.272** |

缩到 224 会把模糊与 JPEG 块效应平滑掉（blur2/jpeg8 抬到 0.87/0.88），噪声被放大（0.696 → 0.272）。
⇒ **预处理必须钉死在 native**，且任何跨评委比较都要在报告里写明预处理。

## 数据卫生提醒

冒烟集里 `p1_orig.png` 与 `photo1.jpg` **像素完全相同**（max|Δ| = 0），只是编码不同，故两者分数
逐位相同——这是对的，不是 bug；正式跑前应清掉一张。
