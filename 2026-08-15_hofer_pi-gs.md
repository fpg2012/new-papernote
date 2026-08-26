---
领域: 3D 重建 / 神经渲染（稀疏视角 Gaussian Splatting）
发表时间: 2026-02-03（arXiv v1）
读文章时间: 2026-08-15
标题: Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization
链接: https://arxiv.org/abs/2602.03327
Zotero: 已入库（arXiv:2602.03327 [cs.GR]，2026-02-03 入库）
页码说明: 正文无印刷页码的首页按 PDF 物理页 1 计；其余页印刷页码与 PDF 物理页码一致
---

# Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization

## 1. TL;DR

这篇文章解决 **3D Gaussian Splatting (3DGS) 在稀疏视角（sparse-view）下重建质量差**的问题：传统 SfM 在稀疏、低重叠输入下常常失败，给不出好的点云初始化和相机位姿。核心贡献是**用无参考（reference-free）的前馈点云估计网络 $\pi^{3}$ 替代 SfM 做稠密初始化**，再配合一套为稀疏视角定制的正则化方案（置信度加权的 Pearson 深度损失、法线监督、深度扭曲生成伪视图），在 Tanks and Temples、LLFF、DTU、MipNeRF360 上达到 SOTA 水平，显著减少 floaters 并提升高斯与真实表面的对齐。

## 2. 作者与单位

- Manuel Hofer（奥地利格拉茨技术大学 Graz University of Technology）
- Markus Steinberger（奥地利格拉茨技术大学）
- Thomas Köhler（奥地利格拉茨技术大学）

（P.1：三人均署名 Graz University of Technology, Austria）

## 3. 紧密相关的工作与交叉点

- **稀疏视角 3DGS 主线**：DNGaussian [14]、Few-shot NVS with Depth [13] 用单目深度估计做正则；FSGS [33] 用池化策略；DropGaussian [18]、DropoutGS [29] 随机停用高斯抗过拟合。这些方法**都依赖 SfM 给的准确相机位姿**。
- **免 SfM 主线**：COLMAP-Free 3DGS [8]、InstantSplat [31] 联合优化高斯与相机位姿；MASt3R [7]、DUSt3R [24] 学习式点云/位姿估计，但**需要好的参考视图、且要做耗时的迭代相机对齐**（几分钟级），位姿不准还会拉低重建质量 (P.2)。
- **扩散先验主线**：GenFusion [26]、SparseGS [28]、Gaussian Scenes [19]、Intern-GS [27] 用扩散先验合成额外视图，效果好但**高频纹理和视图一致性差** (P.2)。
- **交叉点**：本文站在「免 SfM 前馈初始化 + 显式几何正则」这个交叉点上——用 $\pi^{3}$ 一次前馈同时给出稠密点云和相机位姿（不做迭代对齐），再用 PGSR 的平面高斯渲染深度/法线来监督几何，明确**放弃扩散式视图生成**，走几何一致性路线 (P.3)。与 MASt3R/DUSt3R 类方法的关键差异是 $\pi^{3}$ 无参考视图依赖（reference-free）。

## 4. 问题与核心方法

### 4.1 要解决的问题

3DGS 训练严重依赖两点：**准确的相机位姿**和**高质量的点云初始化**（传统上来自 SfM）。稀疏视角下 SfM 经常配准失败，即使成功，点云也极稀疏、图像重叠不足，导致 3DGS 从一开始就陷入坏初始化、深度歧义、对训练视角过拟合，出现 floaters 和视图不一致 (P.1, P.3)。

### 4.2 核心方法

整体思路：**用前馈网络 $\pi^{3}$ 一次性稠密重建场景并估计相机参数 → 在 PGSR 框架上做稀疏视角适配 → 用三个几何正则（置信度加权深度损失、法线损失、深度扭曲伪视图）约束优化**。下面分四步。

**（0）基础：3DGS 与 PGSR 的深度/法线渲染**（P.3）

场景用一组 3D 高斯表示，每个高斯由 $3\times 3$ 协方差矩阵 $\Sigma$ 和世界坐标中心 $\mu \in \mathbb{R}^{3}$ 定义：

$$
G(x)=e^{-\frac{1}{2}(x-\mu_i)^T\Sigma^{-1}(x-\mu_i)}
$$

投影到 2D 时协方差变换为 $\Sigma' = JW\Sigma W^{T}J^{T}$（$J$ 为投影的仿射近似雅可比，$W$ 为视图变换矩阵）；为保证 $\Sigma$ 半正定，参数化为 $\Sigma = RSS^{T}R^{T}$（$S$ 缩放矩阵、$R$ 旋转矩阵，旋转用四元数、缩放用 3D 向量存储）。沿光线混合颜色：

$$
C=\sum_{i=1}^{N}T_i\alpha_i c_i
$$

$N$ 为光线上高斯数，$c_i$ 为第 $i$ 个高斯的颜色（球谐表示，处理视角相关效果），$\alpha_i$ 为加权不透明度，$T_i$ 为透射率 $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$。基础训练损失 $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-SSIM}}$。

本文用 PGSR [4] 渲染深度和法线（可反向传播）。PGSR 把 3D 高斯压平成 2D 平面（尺度损失 $\mathcal{L}_s = \|\min(s_1,s_2,s_3)\|_1$，最小尺度方向即法线 $n_i$），从而渲染**无偏**深度和法线：法线 $\mathcal{N} = \sum_{i=1}^{N}R_c^{T}n_i\alpha_iT_i$（$R_c$ 为相机到世界的旋转）；像素深度 $D = \sum_{i=1}^{N}d_i\alpha_iT_i$，其中 $d_i = (R_c^{T}(\mu_i - T_c))\cdot R_c^{T}n_i$ 是高斯平面到相机中心的（带符号）距离（$T_c$ 为世界系下的相机中心）。PGSR 还带单视图局部平面损失 $\mathcal{L}_{svgeo}$、多视图几何一致性 $\mathcal{L}_{mvgeo}$ 和光度一致性 $\mathcal{L}_{mvrgb}$ 三类损失。

**（1）PGSR 稀疏视角适配**（P.3-4）

默认 PGSR 假设每个点被多相机观测（multi-view observer trim），稀疏视角下不成立 → **停用 trim**。opacity reset 会把背景细节刷没并产生伪影（Fig.2：Ballroom 后墙细节完全丢失，窗口框出现伪影），PSNR 从 22.76 提到 23.73 → **停用 opacity reset**。点云已够稠密 → **停用 splitting 策略**。

**（2）稠密初始化（$\pi^{3}$）**（P.3-4）

用预训练前馈网络 $\pi^{3}$ [25] 一次性预测**每视图点云（稠密）+ 相机参数**，完全替代 SfM。效果对比（bicycle 场景，MipNeRF360，24 视图）：COLMAP 只有 **1,028 个点**，$\pi^{3}$ 有 **1,013,106 个点**（按默认 20% 置信度阈值过滤后）(P.4, Fig.3)。密度差三个数量级，且免去几分钟的迭代相机对齐。

**（3）置信度感知 Pearson 深度损失**（P.4-5）

$\pi^{3}$ 每视图输出点云可直接当深度图用。作者试过 L1/L2 深度损失（会过拟合到深度估计的有限保真度），也试过 DNGaussian 的全局-局部深度归一化（因预测自带尺度一致性而显得多余），最终用 **Pearson 相关损失**——强制结构一致性，同时允许恢复初始深度缺失的高频细节。再叠加 $\pi^{3}$ 输出的**置信度 $C_i$ 做加权**，低置信区域权重低、允许自由优化：

$$
\mu_p=\frac{\sum_{i=1}^{N}C_i D_i^p}{\sum_{i=1}^{N}C_i},\quad \mu_t=\frac{\sum_{i=1}^{N}C_i D_i^t}{\sum_{i=1}^{N}C_i}
$$

$$
\bar{D}_p=D_p-\mu_p,\quad \bar{D}_t=D_t-\mu_t
$$

$$
P_{conf}=\frac{\sum_{i=1}^{N}C_i\,\bar{D}_i^p\,\bar{D}_i^t}{\sqrt{\left(\sum_{i=1}^{N}C_i(\bar{D}_i^p)^2\right)\left(\sum_{i=1}^{N}C_i(\bar{D}_i^t)^2\right)}}
$$

$$
\mathcal{L}_{pearson}=1-P_{conf}
$$

$N$ 为像素数，$D_i^{p}$ 为第 $i$ 像素渲染出的深度（预测），$D_i^{t}$ 为第 $i$ 像素的"真值"深度（即 $\pi^{3}$ 估计的深度），$C_i$ 为第 $i$ 像素置信度，$P_{conf}$ 为置信度加权的 Pearson 相关系数。去均值后的 $\bar{D}$ 保证损失只衡量**结构一致性**而非绝对尺度。对照 Fig.4 可见置信度加权版在背景细节（尤其低分辨率深度估计）上明显更好。

**（4）法线监督（带网格伪影 mask）**（P.5）

表面法线由深度图逐像素偏导 $\partial z/\partial x$、$\partial z/\partial y$ 算出（$z$ 为深度，$x$、$y$ 为像素坐标）。但 $\pi^{3}$ 按 **14×14 像素 patch** 处理图像，patch 之间梯度不连续 → 法线图出现网格伪影（Fig.5a）。解决：构造 14×14 网格，**把每个 cell 的 1 像素内边框区域 mask 掉**，这些边界区域不做正则，伪影就不会进入场景表示（Fig.5b）。监督用 L1 损失：

$$
\mathcal{L}_{normal}=\frac{1}{N}\sum_{i=1}^{N}\|N_i^t-N_i^p\|_1
$$

$N$ 为像素数，$N_i^{t}$ 为第 $i$ 像素真值法线（由 $\pi^{3}$ 深度估计算出），$N_i^{p}$ 为渲染出的法线。

**（5）深度扭曲生成伪视图**（P.5）

把图像像素从源相机投影到 3D、再重投影到目标相机，生成伪视图。只投影**高置信度像素**，其余（含未见区域）mask 掉（Fig.6）。伪相机用**圆插值**生成：目标相机与其最近的两个相机共三点定圆，在相邻视图对之间按步长插值，每对得到 2 个伪视图（实验中 2 个最优）。最近邻相机直接复用 PGSR 已算好的。伪视图以权重 0.1 参与 SSIM + L1 监督，帮助泛化、防过拟合 (P.5)。

**实现要点**（P.6）：全数据集统一设置——$\pi^{3}$ 会自动降采样图像，作者把相机重缩放到全尺寸抵消；只把训练视图投影到 3D，测试视图仅用于初始相机位置（保证对比公平）；训练 7000 迭代，深度/法线/伪视图损失一起上；伪视图置信阈值 20%；停用高斯分裂。

## 5. 实验

**实验设置**：四个基准数据集，全部 3-view（MipNeRF360 另有 12-view 设置）。指标 PSNR↑ / SSIM↑ / LPIPS↓。
- **Tanks and Temples**：8 个真实室内外场景子集（同 Intern-GS/InstantSplat），测试集 12 张均匀采样（去首尾帧），其余为训练集再均匀抽 3 视图，不降采样。
- **MipNeRF360**：3-view 设置（同 Gaussian Scenes，9 场景全用）和 12-view 设置（同 SparseGS，只用其中 6 个场景），都用 4x 降采样。
- **LLFF**：同 DNGaussian 的 3-view 划分，8x 降采样。
- **DTU**：高度标定的物体中心实验室场景，用 $\pi^{3}$ 自推位姿，4x 降采样，评估时按惯例用提供的背景分离 mask。

**① 主结果——Tanks and Temples 3-view（Table 3, P.7）**：为了说明能否在真实稀疏场景拿到 SOTA。**Ours 全面第一：PSNR 22.87 / SSIM 0.764 / LPIPS 0.189**，超过 Intern-GS（22.67/0.736/0.191）、FSGS（22.31）、InstantSplat（22.20）等。作者强调不磨平高频纹理、高斯与表面几何对齐好。

**② 主结果——MipNeRF360 3-view（Table 4, P.7）**：为了对比免 SfM / 生成式方法。**LPIPS 最低（0.523）**，PSNR 14.14、SSIM 0.310 均第二，仅微逊于 FSGS（14.17/0.318/0.578）。注意这是 3-view 极端稀疏设置，绝对分数都低。反常点：Ours 不靠扩散生成未见内容，因此 PSNR 略低于会用扩散"脑补"的 FSGS。

**③ 主结果——MipNeRF360 12-view（Table 5, P.7）**：为了验证视图稍多时的表现。**PSNR 19.54 第一、LPIPS 0.362 第一**；SSIM 0.492 低于 SparseGS 的 0.577（唯一非最优项）。作者归因于表面对齐更准、地面视图一致、浮游伪影更少（Fig.8）。

**④ 主结果——LLFF 与 DTU 3-view（Table 6, P.8）**：DTU 上 **PSNR 23.52 全场第一、LPIPS 0.145 第一**，SSIM 0.815 第二（Intern-GS 0.851）。LLFF 上 PSNR 19.92、SSIM 0.664、LPIPS 0.254，**略低于 Intern-GS（20.49）和 FSGS（20.31）**——作者明确解释：模型只优化已见区域和已知信息，未见区域（如 Fig.7 的天花板）没有几何信息可重建，而生成式方法能"脑补"出来。

**⑤ 位姿精度**：Tanks and Temples 上 $\pi^{3}$ 的相机位姿 mean ATE = 0.0293、RMSE = 0.0325，说明位姿足够准，可比性成立 (P.6)。

**⑥ 消融（Table 1, P.7；Barn 场景）**：每加一个模块 PSNR 都涨——
| 配置 | PSNR |
|---|---|
| 原始 3DGS | 17.53 |
| + PGSR 框架 | 18.05 |
| + $\pi^{3}$ 稠密初始化 | 19.66 |
| + 深度正则 | 20.72 |
| + 法线正则 | 21.56 |
| + 深度扭曲（完整模型） | **22.15** |
| + 高斯分裂 densification | 21.97（反而降，故停用）|

**⑦ 框架消融 PGSR vs 3DGS（Table 2, P.7）**：证明 PGSR 的平面深度渲染在稀疏视角下价值明显——7000 迭代 T&T: 19.99 vs 18.00；**15000 迭代时差距拉大**（20.19 vs 17.04，3DGS 明显退化/过拟合），MipNeRF360 类似（23.41 vs 20.94）。说明 PGSR 让模型稳定、减少 floaters（Fig.9）。

**反常/有趣的点**：(a) 高斯分裂（splitting densification）在稀疏视角 + 稠密初始化下反而有害（21.97 < 22.15）；(b) DTU 上 PSNR 大幅领先（23.52 vs Intern-GS 20.34）但 SSIM 反而略低，两类指标取向不同；(c) 12-view 下 SSIM 输给 SparseGS，作者没有细究，只强调表面对齐和 LPIPS。

## 6. 图表清单

- Fig.1 稀疏视角下 3DGS 的 floaters/视图不一致，及深度/法线图中的几何错位；本文方法大幅改善 (P.1)
- Fig.2 Ballroom 场景有无 opacity reset 对比：2a 有（背景细节丢失、窗口伪影），2b 无（细节保留） (P.4)
- Fig.3 点云对比：3a $\pi^{3}$ 稠密点云 vs 3b COLMAP 点云（bicycle 场景 24 视图；1,013,106 vs 1,028 点） (P.5)
- Fig.4 深度渲染对比：4a 置信度感知 Pearson 损失 vs 4b 标准 Pearson 损失（前者背景细节更好） (P.5)
- Fig.5 法线图：5a 默认（$\pi^{3}$ 的 14×14 patch 导致网格伪影）vs 5b mask 后（伪影消除） (P.5)
- Fig.6 Barn 场景两个重投影伪视图示例（带置信度 mask） (P.6)
- Fig.7 定性对比（Ground Truth / Intern-GS / Ours）：反射准确、伪影少，但天花板未见区域无法重建 (P.7)
- Fig.8 定性对比 SparseGS vs Ours：背景和地面重建更准、伪影更少 (P.7)
- Fig.9 3DGS vs PGSR 不同迭代数可视化：PGSR 的平面深度显著去除 floaters、对齐几何 (P.8)
- Table 1 消融研究（Barn 场景，PSNR；各模块递增 + 高斯分裂负贡献） (P.7)
- Table 2 框架消融 PGSR vs 3DGS（7000/15000 迭代，T&T + MipNeRF360 的 PSNR/SSIM/LPIPS） (P.7)
- Table 3 Tanks and Temples 3-view 定量对比（SOTA） (P.7)
- Table 4 MipNeRF360 3-view 定量对比（Ours LPIPS 最低） (P.8)
- Table 5 MipNeRF360 12-view 定量对比（Ours PSNR/LPIPS 第一） (P.8)
- Table 6 LLFF + DTU 3-view 定量对比（DTU PSNR/LPIPS 第一，LLFF 略逊生成式方法） (P.8)

## 7. 快速总结

1. **核心贡献**：用 $\pi^{3}$ 无参考前馈网络替代 SfM 做稠密初始化（点云密度提升约 3 个数量级），并配套 PGSR 稀疏视角适配 + 三项几何正则，稀疏视角重建达到 SOTA（T&T 3-view 全面第一、DTU PSNR/LPIPS 第一）。
2. **方法亮点**：置信度加权 Pearson 深度损失既保结构又放低置信区域自由度；14×14 patch 网格伪影的 mask 方案很工程、很实用；深度扭曲伪视图是免扩散、纯几何的泛化手段。
3. **工程细节可借鉴**：稀疏视角下停用 opacity reset 和 splitting 反而更好（消融证实 splitting 是负贡献）；PGSR 在长迭代下比 3DGS 稳定得多。
4. **主要局限**：$\pi^{3}$ 处理多视图/大场景消耗大量 GPU 内存，消费级硬件不可行；个别场景深度估计不准（如 LLFF 的 leaves）；未见区域无法重建（对比生成式方法在 PSNR 上的差距）。
5. **可延伸方向**（作者自述）：联合优化相机位姿与高斯场景；引入生成式先验来保持遮挡/稀疏区域的照片与几何一致性。
6. **可借鉴点**：对于"初始化不靠谱"的下游任务（NeRF/3DGS 类），前馈稠密预测 + 不确定性（置信度）加权正则是一条很稳的路线；代码已开源（https://github.com/Mango0000/Pi-GS）。
