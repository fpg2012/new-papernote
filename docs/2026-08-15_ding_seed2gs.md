---
arXiv分类: "cs.CV"
领域: "3D 重建 / 神经渲染（3DGS 分割）"
发表时间: "2026-08-12（arXiv v1）"
读文章时间: "2026-08-15"
标题: "Seed2GS: Camera-Free, Training-Free Object Extraction from 3D Gaussian Scenes via a Single Reference-View Grounding"
链接: "https://arxiv.org/abs/2608.11928"
Zotero: "已入库 [zotero://select/library/items/QYH66LJX](zotero://select/library/items/QYH66LJX)"
页码说明: "PDF 页码"
---

# Seed2GS: 免相机、免训练地从 3DGS 场景中提取目标物体

## 1. TL;DR

解决「从一个**已经建好但拿不到原始重建相机**的 3DGS 场景里，按一句文本提示把某个目标物体对应的高斯抠出来」的问题。核心贡献是把「目标身份」和「3D 覆盖范围」两件事分开：用一次语义定位（QD-SAM3）固定身份，再用虚拟轨道 + 视频追踪扩展覆盖，最后只优化每个高斯一个**一次性前景 logit**。在 LERF-MASK 上达到 92.1% mIoU、单次查询 9.3 秒，超过所有免训练基线和最强的场景训练方法 ObjectGS。

## 2. 作者与单位

- 丁宗健 Zongjian Ding（中国科学院大学；中科院信工所，共同一作）
- 高宇东 Yudong Gao（香港科技大学，共同一作）
- 刘佳乐 Jiale Liu（浙江大学）
- 于兴林 Xinglin Yu（北京理工大学）
- 任俊星 Junxing Ren（中国科学院大学；中科院信工所）
- 魏东 Dong Wei（中国科学院大学；中科院信工所）
- 陈雅静 Yajing Chen（中国科学院大学）
- 黄珊 Shan Huang（浙江大学）
- 程明骏 Mingjun Cheng（浙江大学，通讯作者）
- 李敏 Min Li（中国科学院大学；中科院信工所，通讯作者）

## 3. 紧密相关的工作与交叉点

本文站在三条线交叉点上：

1. **场景特定 3DGS 分割**（语言特征挂到高斯上：LERF、LangSplat、OpenGaussian；实例身份学习：Gaussian Grouping、SAGA、ObjectGS 等）。这些方法准确率高，但每个场景要先花几十分钟构建语义/实例表示，无法对「刚收到的资产」做即席查询（P.2）。
2. **Mask 提升（mask lifting）类方法**（SA3D、FlashSplat、LBG）：把 2D mask 证据直接转到 3D 原语，免去持久语义场；但都依赖**原始重建相机**，而用户手里往往只有资产本身没有相机（P.2）。
3. **免相机方法**：B3-Seg 是最接近的工作——从冻结场景渲染虚拟视图，用解析期望信息增益选下一视图，再反复用开放词汇检测器重检，用 Beta-Bernoulli 更新每个高斯的置信。它免相机免训练，但精度仍落后场景训练方法（P.1）。

**差异点**：B3-Seg 把计算花在「每次选哪个视角 + 每视角重新检测」上；Seed2GS 认为身份问题（提示词指哪个实例）和覆盖问题（该实例在 3D 里延伸到哪）需要不同证据，反复重检同一语义问题反而可能在相似实例间跳变。因此它只做一次语义决策（选一个 seed），覆盖交给轨迹 + 追踪。

## 4. 问题与核心方法

### 4.1 要解决的问题

输入：冻结的 3DGS 场景 $\mathcal{G}=\{g_i\}_{i=1}^N$、文本提示 $p$、用户当前参考相机 $c_r$。输出：构成目标物体的高斯子集。约束：不用原始重建相机、不做场景级语义训练、不建持久表示，单次查询要在秒级完成（P.2）。

### 4.2 核心方法

**思路**：把任务拆成三步——ground once（定位一次身份）、propagate（沿虚拟轨迹传播）、fit（拟合高斯隶属度）。冻结场景的几何、颜色、不透明度全都不动，只对每个高斯估一个**用完即弃**的前景 logit。

**形式化基础（3DGS 预备）**：每个高斯原语 $g_i$ 有中心 $\mu_i$、协方差 $\Sigma_i$、球谐颜色系数 $\mathbf{h}_i$、不透明度 $\alpha_i$。对相机 $c$，光栅化把每个高斯对像素 $u$ 的合成贡献记为 $w_{u,i}^{(c)}$。若给每个高斯一个前景变量 $q_i \in [0,1]$，用同样的合成权重混合就得到软前景 mask（P.2）：

$$
P_{u}^{(c)}=\sum_{i}w_{u,i}^{(c)}q_{i}.
$$

把视图 $c$ 的 $U$ 个像素堆起来就是 $P^{(c)} = W^{(c)} q$，其中 $W^{(c)} \in \mathbb{R}^{U \times N}$ 是合成权重矩阵，$q=(q_1,\ldots,q_N)^\top$。**场景冻结意味着 $W^{(c)}$ 每个元素都已知**，所以提取目标变成一个带 box 约束的线性逆问题：给定固定算子 $\{W^{(c)}\}$ 和目标 mask $\{Y^{(c)}\}$，求 $q\in[0,1]^N$ 使每个视图 $W^{(c)}q \approx Y^{(c)}$。作者用 $q_i=\sigma(l_i)$（sigmoid）自动满足 box 约束，每个高斯只拟合一个临时 logit $l_i$（P.2-3）。

与同类公式化方法的区别：FlashSplat 用闭式解、B3-Seg 用贝叶斯更新，Seed2GS 最小化一个可靠度加权的目标；且 Seed2GS 的所有目标 mask $Y^{(c)}$ 都来自**同一次定位**（P.3）。

**Step 1 — QD-SAM3 种子获取**：整个管线唯一的语义阶段。组合三个发布模型：Qwen3-VL 和 GroundingDINO 各给一个定位框（语义/检测两种 cue），SAM3 把每个 cue 转成 mask；此外 SAM3 本身也接受文本提示作为第三个 cue。设这些 cue 为 $\{a_k\}_{k=1}^K$，用同一个 SAM3 处理全部：

$$
\mathcal{M}(p)=\{m_{k}=\mathrm{SAM3}(I_r,a_{k})\}_{k=1}^{K},
$$

其中 $I_r = \mathcal{R}(\mathcal{G}, c_r)$ 是参考视图渲染图。因为 mask 生成器固定，候选差异主要反映定位 cue 的质量。最后用三项加权分数选出 seed mask（P.2-3）：

$$
m^{\star}=\underset{m_{k}\in\mathcal{M}(p)}{\mathrm{argmax}}\; w_{s}S_{\mathrm{src}}(m_{k})+w_{c}S_{\mathrm{conf}}(m_{k})+w_{t}S_{\mathrm{lang}}(m_{k},p).
$$

- $S_{\mathrm{src}}$：候选来源（Qwen3-VL / GroundingDINO / SAM3-text）的先验权重；
- $S_{\mathrm{conf}}$：来源模型的置信度；
- $S_{\mathrm{lang}}$：来源与查询语义类型的兼容性（比如语义框更配"桌上有几个杯子"这类关系提示）。

**Step 2 — Seed lift（种子提升）**：把参考 mask $m^*$ 转成保守的 3D 高斯支撑 $\mathcal{G}_0$，同时估计物体中心 $\mathbf{o}$。三个约束定位参考视图中属于前景的高斯：投影中心满足 $m^*(\pi_i)=1$（$\pi_i$ 是 $g_i$ 的投影中心）、深度满足 $|z_i - D_r(\pi_i)| \le \epsilon_d$（$z_i$ 是 $g_i$ 深度，$D_r$ 是参考深度图）、再对 3D 候选图做连通分量过滤去掉投影离群点（P.3）。$\mathcal{G}_0$ 只用来初始化 logit 和参数化虚拟轨道，不改场景本身。

**Step 3 — VAAS 虚拟轨道监督（Visibility-Adaptive Ascending Spiral）**：既然没有原始相机，所有监督视图都用虚拟相机渲染。围绕 $\mathbf{o}$ 构造两条参考锚定轨迹：一条平直轨道（提供水平视角）、一条上升螺旋轨道（补充顶部/底部/遮挡面视角），都从参考相机 $c_r$ 出发。SAM2 视频追踪把 seed mask $m^*$ 传播到这些新视图，**不再做任何语义检测**（P.3）。

轨道高度自适应：设 $\bar{V}$ 为平直轨道各视角下 $\mathcal{G}_0$ 的平均可见比例（深度一致投影），上升俯仰幅度取

$$
A=A_{\max}(1-\bar{V})^{\gamma}.
$$

覆盖好的目标 $\bar{V}$ 大，轨迹接近平面；覆盖差的目标爬升更高。当 $A<5^\circ$ 时省略近似重复的上升片段，只渲染平直片段——把额外的视图预算花在确实缺覆盖的目标上（P.3-4）。

**Step 4 — 高斯细化（Gaussian refinement）**：用 VAAS 传播来的 mask 优化每高斯的隶属度。遮挡和追踪漂移会污染部分 mask，所以按各视图的运行期 BCE 残差 $D_c$ 加权。设 $m_D$ 为中位数残差、$s_D$ 为基于 MAD 的稳健尺度，视图权重为（P.4）：

$$
\omega_{c}=\mathrm{Norm}\left[r_{\min}+(1-r_{\min})\sigma\left(\frac{m_{D}+s_{D}-D_{c}}{\lambda_{\tau}s_{D}}\right)\right],
$$

其中 $\mathrm{Norm}[\cdot]$ 表示均值归一化，$r_{\min}$ 是非零下限（保证困难视角在优化早期仍有监督）。优化目标为可靠度加权的多视图 mask 损失：

$$
\mathcal{L}=\sum_{c,u}\omega_{c}\,\mathrm{BCE}\left(P_{u}^{(c)},Y_{u}^{(c)}\right),
$$

$Y^{(c)}$ 是视图 $c$ 的传播监督 mask。收敛后按阈值输出目标子集：

$$
\mathcal{G}_{f}=\{g_{i}\mid q_{i}>\tau\}.
$$

**整体流程（Algorithm 1, P.4）**：渲染参考画布 → QD-SAM3 选 seed → SeedLift 得 $\mathcal{G}_0,\mathbf{o}$ → VAAS 得监督 mask 集 → 初始化 logit → 最小化 $\mathcal{L}$ → sigmoid + 阈值输出。

## 5. 实验

- **数据集**：LERF-MASK（23 个文本查询目标，分布在 Figurines/Ramen/Teatime 三个 LERF 场景；含细薄结构、关系型提示、容器/内容歧义）和 3D-OVS（Bed/Bench/Sofa/Lawn 四场景，沿用 B3-Seg 的场景选择与测试划分）。指标 mIoU 与 mBIoU，沿用 Gaussian Grouping 公开协议；原生评估器把前景概率阈值设为 0.40（P.4-5）。
- **运行时**：H800 上，23 个 LERF-MASK 目标平均 $9.26 \pm 0.97$ 秒；QD-SAM3 占 47.7%、VAAS 占 52.3%（Table 3, P.6）。B3-Seg 无公开代码，12.1s 与 9.3s 不是同硬件对比。

**主结果 — LERF-MASK（Table 1, P.5）**：Seed2GS 92.1 mIoU / 88.6 mBIoU，9.3s，免训练免相机。对比：B3-Seg 84.5/81.0（+7.6）；最强的场景训练方法 ObjectGS 88.4/85.8（+3.7）；Gaga 74.7；LangSplat 57.6；FlashSplat-R 76.5；Gaussian Grouping 72.8。Ramen 场景上差距最大（Seed2GS 91.5 vs B3-Seg 75.3、ObjectGS 88.0），因为该场景查询含容器/内容歧义。

**主结果 — 3D-OVS（Table 2, P.6）**：Seed2GS 95.7 mIoU，仅比 B3-Seg（95.0）高 0.7；但所有方法都 >91%，分离度小。Seed2GS 在 Lawn 拿到最佳 97.1。说明 3D-OVS 的查询较简单，方法间差异主要靠 LERF-MASK 拉开。

**参考视图敏感性（Table 5, P.6；Fig.4, P.7）**：这是评估协议的关键控制实验。

- 每场景用固定测试参考（视图 0/2/0）：QD-SAM3 91.14 mIoU；把预测 seed 换成 GT mask 只提高 0.72 点——说明初始 mask 本身已接近上限，后续传播和拟合没引入大误差。
- 230 次有利参考扫描：平均 91.17 mIoU、目标内标准差 1.16 点；23 个目标中仅 5 个超过 1 点偏差；全部 230 次运行都返回有效提取。最差参考平均仍有 88.98 mIoU。

**消融（Table 6, P.7）**：

- 选择策略：三项目加权（92.14）比纯 max-confidence（87.73）高 4.41 点。
- 移除来源：去掉 Qwen3-VL 掉得最多（84.34，-7.8），去掉 GroundingDINO 89.64，去掉 SAM3-text 91.46。
- 移除分数项：去掉 $S_{conf}$ 掉得最多（89.16，-2.98），去掉 $S_{lang}$ 91.46，去掉 $S_{src}$ 92.10。→ Qwen3-VL 和置信度项是最强来源/项。

**消融（Table 7, P.7）**：

- 轨迹策略（相对平直轨道）：自适应 VAAS 在 LERF-MASK 上 +2.17 mIoU / +2.08 mBIoU，且只用平均 27.1 个视图而非固定的 48 个；固定 48 视图只 +0.76。→ 收益来自自适应轨迹设计而非多渲染视图。
- 视图数：LERF-MASK 上 24 视图最优（12 视图 -0.24，48 视图 -0.84）；3D-OVS 上 48 视图 +0.28 但 mBIoU 略降。

**消融（Table 8, P.7）**：

- 两条参考锚定片段 vs 单条连续片段：LERF-MASK +10.56、3D-OVS +17.50。
- 视图可靠度加权：+1.25（LERF-MASK）/ +1.40（3D-OVS）。

**运行时审计（正文 P.5-6, Table 4）**：初始支撑大小（Teatime 上从 212 到 26,874 个高斯，126.8×）与延迟几乎无关（$|\rho|\le0.321$）；耗时主要由场景级渲染和视图预算决定，而非目标大小。

## 6. 图表清单

- Fig.1 精度—运行时间权衡（LERF-MASK，断裂坐标轴区分秒级/分钟级）(P.1)
- Fig.2 方法总览：QD-SAM3 选 seed → seed lift + VAAS/SAM2 传播 → 冻结光栅化 + 可靠度加权 mask 拟合 (P.2)
- Fig.3 定性对比（3D-OVS 两行 + LERF-MASK 一行；对比 Gaussian Grouping/OpenSplat3D/FlashSplat/Gaga；Seed2GS 背景泄漏更少）(P.5)
- Fig.4 230 次参考视图鲁棒性：上为平均精度，下为目标内标准差 (P.6)
- Table 1 LERF-MASK 精度/时间/运行假设对比（13 个方法）(P.5)
- Table 2 3D-OVS mIoU 对比（6 个方法）(P.6)
- Table 3 分场景运行时间（QD-SAM3 / VAAS / 总时长）(P.6)
- Table 4 VAAS 后端重复计时（12 vs 24 视图/片段）(P.6)
- Table 5 参考视图研究（固定 QD-SAM3 / 固定 GT oracle / Top-10 扫描，mIoU+mBIoU）(P.6)
- Table 6 QD-SAM3 消融（选择策略、来源移除、分数项移除）(P.7)
- Table 7 VAAS 消融（轨迹策略、视图数）(P.7)
- Table 8 追踪与视图加权消融（单片段/无加权/完整）(P.7)
- Algorithm 1 SEED2GS 推理流程 (P.4)

## 7. 快速总结

**核心贡献**

1. 把「目标身份」与「3D 覆盖」解耦：一次语义定位 + 追踪传播，取代 B3-Seg 的逐视图重检。
2. 在冻结场景上只拟合每个高斯一个一次性 logit，同时满足免相机、免训练、秒级延迟三个条件，且在 LERF-MASK 上精度超过全部场景训练方法。
3. 做了相当扎实的协议控制实验（固定参考/多参考扫描），证明主结果不依赖单个精心挑选的参考视图。

**主要局限**

1. 单次定位是单点故障：seed 错了，提升、追踪、拟合全部被污染（严重遮挡、同类实例、复杂指代仍会失败）。
2. 虚拟视图无法恢复场景里缺失的几何；大视角变化会导致 SAM2 追踪漂移。
3. 230 次扫描只覆盖"有利参考"，任意视角（目标很小/被藏/不存在）未评估。
4. 继承离壳识别模型的弱点（如后门触发器），论文明确声明假定检测器可信、不做审计。

**可借鉴/可延伸**

1. 多参考初始化与不确定性感知传播，可缓解单 seed 故障点。
2. 用运行期 BCE 残差 + MAD 稳健尺度做视图加权，这个"证据加权"思路可移植到其他 mask 提升管线。
3. 与 B3-Seg 的思路对比很有启发：把查询期算力集中花在"一次做对的语义决策"上，而非反复检测。
