---
领域: "计算机视觉 / RGB-D 语义分割（SAM 应用）"
发表时间: "2023-05-23（arXiv v1）"
读文章时间: "2026-08-23"
标题: "SAD: Segment Any RGBD"
链接: "https://arxiv.org/abs/2305.14207"
Zotero: "已入库（key: BE5HPRU9）"
页码说明: "印刷页码 = PDF 物理页码（正文 1–5）"
---

# SAD: Segment Any RGBD

## 1. TL;DR

SAM 分割 RGB 图像时主要靠纹理（颜色）线索，容易把物体过度分割（一个桌子被切碎成好几块）。SAD（Segment Any RGBD）的思路极简：把深度图用 colormap 渲染成伪彩色 RGB 图再喂给 SAM，让分割结果"只看几何、忽略纹理"，从而缓解过分割；再叠加 OVSeg 开放词汇语义分割，通过**语义投票**给 SAM 的每个片段赋予类别并聚类相邻同类片段，最后把语义分割结果按深度投影到 3D 世界做立体可视化。整个方法**零训练、即插即用**，只是组合 SAM + OVSeg + colormap 渲染 + 投票，没有引入任何可学习参数。

## 2. 作者与单位

- Jun Cen（港科大 HKUST¹；NTU³）
- Yizheng Wu（华科²；NTU³）
- Kewei Wang（华科²；NTU³）
- Xingyi Li（华科²；NTU³）
- Jingkang Yang（NTU³）
- Yixuan Pei（西安交大⁴）
- Lingdong Kong（NUS⁵）
- Ziwei Liu（NTU³）
- Qifeng Chen（HKUST¹）

（1=香港科技大学，2=华中科技大学，3=南洋理工大学，4=西安交通大学，5=新加坡国立大学，P.1。）

## 3. 紧密相关的工作与交叉点

- **SAM**（[6], P.2）：ViT 大模型、SA-1B 训练，zero-shot 分割任意 2D 图像。SAD 的直接基础。
- **OVSeg**（[7], P.2）：mask-adapted CLIP，开放词汇语义分割——给定文本类别候选就能分割未见过的类别。SAD 用它给 mask 赋语义。
- **同时代的 SAM 应用**（[9] 综述列举）：SSA（Semantic Segment Anything，[1]）、Anything-3D（[8]）、SAM 3D（[2]）等**全都把 RGB 图像当输入**。SAD 的差异化定位（P.2 原文）："we are the first to utilize SAM for directly segmenting rendered depth images"——第一个直接拿 SAM 分割渲染深度图的工作。

交叉点：站在"SAM 的分割能力"和"深度图的几何信息"两条线的交汇处，用最轻量的工程手段（colormap）完成 2D 分割到几何感知分割的迁移。它没有学任何新东西，本质上是**输入模态的切换**：把"几何信息"编码进 SAM 熟悉的 RGB 空间。

## 4. 问题与核心方法

### 4.1 要解决的问题

SAM 在分割 RGB 图时**过度依赖纹理信息**（颜色、图案），几何信息被忽视，导致过分割：一个语义上完整的物体被切成多个片段（P.1）。目标是让 SAM 的分割结果包含更多几何信息，并进一步升级成带类别的语义分割、乃至 3D 可视化。

### 4.2 核心方法（P.2-3，图 2）

**直觉**：人类看深度图能轻松认出物体（深度可视化天然突出几何、抹掉纹理）。那么与其改造 SAM，不如改造输入——把深度图渲染成 RGB 图喂给 SAM，SAM 就能"用几何的方式"分割。方法分四步：

1. **渲染深度图（Rendering depth maps）**：用 colormap 函数（Matplotlib，[5]）把深度图 $D \in \mathbb{R}^{H \times W}$ 映射成伪彩色 RGB 图 $D' \in \mathbb{R}^{H \times W \times 3}$。论文试了 Viridis、Gray、Plasma、Cividis、Purples 等 colormap（图 1）。关键性质：渲染后的深度图**保留了几何信息、丢掉了纹理信息**。
2. **SAM 分割（Segmentation with SAM）**：把渲染深度图喂给 SAM，得到初始 mask。这些 mask 仍是无类别（class-agnostic）的，且依然过分割（图 1 可见）。
3. **OVSeg 语义分割（Semantic segmentation with OVSeg）**：用 RGB 原图 + 文本提示跑 OVSeg，得到包含语义信息的**粗 mask**。粗 mask 有两个作用：① 引导后续对过分割片段的聚类；② 提供类别信息。
4. **语义投票与聚类（Semantic voting）**：对 SAM mask 中的每个像素，查它在 OVSeg mask 里对应的预测类别，按"片段内像素的多数类别"给整个片段赋类；然后把**相邻且同类**的片段聚类合并，得到最终的语义分割结果。
5. **3D 投影**：基于深度图把语义分割结果投影到 3D 世界，做立体可视化（图 4/5 的右侧列）。

整个流程没有训练、没有可学习参数，纯粹是现成模型的组合。注：论文正文的引用编号有些错乱——正文提到 SSA [1]、Anything-3D [8]、SAM 3D [2]，但参考文献 [1] 是 Semantic Segment Anything、[2] 是 Pointcept 代码库，"SAM 3D [2]" 对应关系存疑（P.2 正文 vs P.5 参考文献），读原文时留意。

## 5. 实验

**实验设置**：这篇是**纯定性短文**——没有定量实验、没有指标表格，只在两个数据集上展示可视化结果：SAIL-VOS 3D [4]（合成视频数据）和 ScanNet [3]（室内重建）。评价指标：无（看图说话）。

**RGB 输入 vs 深度输入对比（第 3 节, P.3, 图 3）**：

- 现象：RGB 图颜色丰富、纹理多，SAM 对 RGB 输入会产生**更多 mask**（过分割）；对深度输入产生的 mask 更少、更接近完整物体。
- 例子 1（桌子，黄圈）：RGB 输入把桌子切成 4 块，其中一块在语义结果里被误判成"椅子"；深度输入则正确识别为桌子。
- 例子 2（椅子，红圈）：**反过来**——两个紧挨着的物体在深度图里可能被 SAM 当成一个物体合并；此时 RGB 的纹理信息反而关键，能区分它们。
- 结论：深度输入缓解过分割，但**几何相近的邻近物体需要 RGB 纹理来区分**——说明两类输入各有优劣，理想方案是互补使用。

**定性结果（第 4 节, P.3）**：Sailvos3D（图 4）和 ScanNetV2（图 5）上展示了"SAM masks with class → 3D 可视化 → 语义 mask → 3D 可视化"的完整流程，说明深度输入下几何语义分割效果更好。无量化支撑，说服力有限。

## 6. 图表清单

- Fig.1 不同 colormap 渲染的深度图经 SAM 分割的结果对比；深度图分割天然带更多几何信息 (P.1)
- Fig.2 SAD 框架总览：渲染深度图 → SAM → OVSeg → 语义投票 → 3D 投影 (P.2)
- Fig.3 RGB 输入 vs 渲染深度图输入的 SAM 分割结果对比（桌子被过分割并误判、邻近椅子被合并两个反例）(P.3)
- Fig.4 SAIL-VOS 3D 数据集定性结果（Input to SAM / SAM Masks with Class / 3D Visualization / Semantic Masks / 3D Visualization）(P.4)
- Fig.5 ScanNetV2 数据集定性结果（同上布局）(P.4)

（无表格、无公式——全文没有量化实验。）

## 7. 快速总结

1. **核心贡献**：提出"把深度图 colormap 渲染后喂给 SAM"这一输入模态切换技巧，是首个直接用 SAM 分割渲染深度图的工作；配合 OVSeg + 语义投票，把 class-agnostic 的 SAM mask 升级成带类别的语义分割并投影到 3D。
2. **工程价值**：零训练、零参数、即插即用，组合的都是现成模型（SAM + OVSeg + Matplotlib colormap），门槛极低，很容易复现/复用。
3. **主要局限**：只有定性实验，没有定量指标；深度图会把邻近物体合并（需 RGB 纹理补足）；依赖深度输入（RGB-D 传感器或深度估计模型），没有深度就没有方法。
4. **影响**：作为 SAM 时代早期"几何感知分割"的简洁基线，被后续 3D 分割工作（如 MV-SAM 引用的 SA3D/SAGA/OmniSeg3D 一脉）频繁引用，属于承上启下的工作。
5. **可延伸**：colormap 选择的消融（论文试了 5 种但未系统比较）、与 SAM2/更稳的深度估计器组合、把语义投票换成更细粒度的融合（如 SAM mask 与 OVSeg 的边界对齐）都是可做方向。
