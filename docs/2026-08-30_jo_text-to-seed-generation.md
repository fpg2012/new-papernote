---
arXiv分类: "cs.CV"
领域: 图像分割 / 开放词汇语义分割（Open-Vocabulary Semantic Segmentation）
发表时间: 2026-08-27 (arXiv v1；期刊版 Knowledge-Based Systems, DOI 10.1016/j.knosys.2026.116698)
读文章时间: 2026-08-30
标题: Text-to-seed generation: Training-free open-vocabulary seeded semantic segmentation via re-purposing diffusion as text-guided seed generator
链接: https://arxiv.org/abs/2608.26624
Zotero: 已入库
页码说明: 页码为 PDF 印刷页码
---

# Text-to-Seed (T2S): 免训练开放词汇语义分割

## 1. TL;DR

开放词汇语义分割（OVSS）需要按任意文本查询分割图像区域。现有方法常让 SAM 去"精修"其他模型的粗掩码，初始掩码不准就全盘皆输。T2S 反其道而行：把 OVSS 重新表述为经典**种子分割**（seed localization + region expansion）——用 Stable Diffusion 的注意力图生成文本引导的**种子点**，再把种子作为 point prompt 喂给 SAM 做区域扩张。全程免训练、即插即用，在 VOC / Cityscapes / ADE20K / COCO / Pascal Context 上全指标 SOTA。

## 2. 作者与单位

- Kumju Jo（汉阳大学 人工智能系）
- Heesun Jung（汉阳大学 数据科学系）
- Sungyong Baik（汉阳大学 两系，通讯作者）

基金：韩国 IITP（RS-2025-25422680、RS-2020-II201373）与 NRF（RS-2025-24533064）。

## 3. 紧密相关的工作与交叉点

- **CLIP 路线**：CLIP 只有图像级对齐，空间一致性差；SCLIP/NACLIP/CLIPer 等改进其空间表征，但没有独立的文本条件定位信号，仍受限于 CLIP 初始预测质量。
- **扩散模型路线**：用 SD 直接估掩码（OVDiff、DiffSegmenter）容易得到粗掩码；用 SD 生成合成图做特征匹配（FreeDA、FreeSeg-Diff）依赖合成图像质量；NERVE 用 CLIP 交叉注意力 + SD 自注意力 + 熵引导随机游走修掩码。
- **SAM 路线**：SAM 空间连贯但无语义；Grounded-SAM 用 Grounding DINO 的框喂 SAM；CaR/CaR+SAM 用 SAM 后处理精修 CLIP 或扩散的粗掩码——受限于初始掩码质量；SAM 3 支持文本 prompt 但语义理解限于简单名词短语。
- **交叉点**：与同组前作 Seediff [48]（AAAI 2025，从扩散模型生成种子掩码）一脉相承，把 SD 的角色从"生成掩码"进一步降为"生成种子"，SAM 从"修掩码"变为"做区域扩张"——即用 SD 的空间文本对齐做定位（seed），用 SAM 的空间连贯做扩张（expansion），各取所长、互不拖累。

## 4. 问题与核心方法

### 4.1 要解决的问题

SAM 语义理解弱、CLIP 空间定位差、SD 直接出掩码粗且多主体识别难——三者单独都不行；组合时"粗掩码 → SAM 精修"的范式又受限于粗掩码质量。目标：**免训练**地把三者拼起来拿到高质掩码。

### 4.2 核心方法

四阶段（Fig.2, P.4）：初始化种子生成器 → 种子提取 → 迭代种子生成与扩散 → SAM 区域扩张。

**① 种子生成器初始化（Sec 4.1, P.3-4）**

用 SD v1.4。两个设计点：
- **[EOT] 加权**：SD 的注意力图中，End-of-Text ([EOT]) embedding 蕴含图中物体语义，把 [EOT] embedding 权重 ×2 增强目标类别信号。
- **聚合窗口取后半段**：SD 是图像生成模型，早期去噪步（t 接近 T）图还是高斯噪声、注意力图很噪；从 $t=T/2$（T=75）之后开始聚合注意力，此时图像已接近真实。Fig.3 (P.4) 验证了早聚合=噪声。

注意力按分辨率聚合、逐层归一化后平均：

$$
\bar{\mathcal{A}}_{s_{r}}=\frac{1}{|\mathbb{L}_{s_{r}}|\cdot T}\sum_{l\in\mathbb{L}_{s_{r}}}\sum_{1\leq t\leq T}\frac{\mathcal{A}_{l,t}}{\max(\mathcal{A}_{l,t})},
$$

其中 $\mathbb{L}_{s_r}$ 是产生分辨率 $s_r$ 注意力的层集合，$s_r\in\{8,16,32,64\}$。得到聚合交叉注意力（ACA）$\tilde{\mathcal{A}}_{s_r,c_{\mathrm{cls}}}^{\mathrm{CA}}$ 与聚合自注意力（ASA）$\tilde{\mathcal{A}}_{s_r,c_{\mathrm{cls}}}^{\mathrm{SA}}$（$c_{\mathrm{cls}}$ 是文本中类别 token 索引）。**注意力聚合完后 SD 就不再需要了**——后续迭代都在已聚合的图上做，很省。

CA/SA 的形式定义（第 $l$ 层、时间步 $t$、维度 $d_l$）：

$$
\mathcal{A}_{l,t}^{\mathrm{CA}}=\mathrm{softmax}\left(\frac{Q_{l,t}^{z}\cdot K_{l}^{\tau\top}}{\sqrt{d_{l}}}\right),\qquad
\mathcal{A}_{l,t}^{\mathrm{SA}}=\mathrm{softmax}\left(\frac{Q_{l,t}^{z}\cdot K_{l,t}^{z\top}}{\sqrt{d_{l}}}\right),
$$

$Q^z$/$K^z$ 是潜特征 $z_t^l$ 的 query/key 投影，$K^\tau$ 是文本 $\tau(y)\in\mathbb{R}^{P\times d_T}$ 的 key 投影（$P$=文本长度）。

**② 种子初始化（Sec 4.2, P.4）**

分辨率越高注意力越弥散、定位越差（Fig.4），所以只用 8×8 和 16×16 的 ACA。初始种子取 8×8 上超过阈值 $\alpha$ 的坐标：

$$
\mathbb{S}_{1}=\left\{(i,j)\mid\tilde{\mathcal{A}}_{8,c_{\mathrm{cls}}}^{\mathrm{CA}}[i,j]\geq\alpha\right\},\quad \alpha=0.5.
$$

**③ 迭代种子生成与扩散（SGS, Sec 4.3, P.4-5）**

低分辨率 ACA 定位好、高分辨率 ASA 细节细，于是交替使用：用 ASA 把当前种子"扩散"到相似特征，再上采样到更高分辨率继续提种子。

先用 ASA 聚合出与当前种子相似的注意力掩码：

$$
\mathcal{M}_{s_{k}}^{\prime}=\frac{1}{|\mathbb{S}_{k}|}\sum_{(i,j)\in\mathbb{S}_{k}}\tilde{\mathcal{A}}_{s_{k}}^{\mathrm{SA}}[i,j,:,:],
$$

双线性上采样到下一分辨率：

$$
\mathcal{M}_{s_{k+1}}=\mathrm{bilinear\text{-}upsample}_{s_{k+1}}\left(\mathcal{M}_{s_{k}}^{\prime}\right),
$$

到 16×16 时与 ACA 逐点取 max 融合（ACA 语义对应更好）：

$$
\mathcal{M}_{16}\leftarrow\max\left(\tilde{\mathcal{A}}_{16,c_{\mathrm{cls}}}^{\mathrm{CA}},\;\mathcal{M}_{16}\right),
$$

再提种子：

$$
\mathbb{S}_{k+1}=\left\{(i,j)\mid\mathcal{M}_{s_{k+1}}[i,j]\geq\alpha\right\}.
$$

重复到最高分辨率 $s_K=64$（Fig.6 展示了掩码逐轮变细变稀疏）。

**④ SAM 区域扩张（Sec 4.4, P.5）**

SAM 在"正点提示 + 负点提示"下分割更准。正点即类别种子；负点取注意力值**低**的位置（背景/无关物）：

$$
\mathbb{S}_{k+1}^{\mathrm{neg}}=\left\{(i,j)\mid\mathcal{M}_{s_{k+1}}^{\mathrm{neg}}[i,j]\leq\beta\right\},\quad \beta=0.3.
$$

把 $\mathbb{S}_{\mathrm{SAM}}=\mathbb{S}_K\cup\mathbb{S}_K^{\mathrm{neg}}$ 作为 point prompt 喂 SAM 得到掩码 $m$；再把掩码图 $m\odot x$ 喂 CLIP（ViT-L/14）做区域分类，过滤误检。

**⑤ 迭代式种子语义分割（ISSS, Sec 4.5, P.5）**

SD 交叉注意力对"多个主体"常常只注意其中一个。解法：迭代——每轮把已找到的掩码区域从 16×16 ACA 图上**抹掉**，迫使下一轮种子落在别处；直到 CLIP 连续拒绝 $n=10$ 次或全图覆盖。Fig.5 展示掩码图随迭代变细、定位更准。

## 5. 实验

数据集：VOC-20/21、Pascal Context（Pascal59/60）、Cityscapes（19 类无背景）、COCO object（80 类）、ADE20K（150 类无背景），指标 mIoU；另加 ISPRS Potsdam 航空影像做域外泛化测试。

**主结果（Tab.1, P.6）**——training-free 方法中全指标第一：

| 模型 | VOC-20 | VOC-21 | Cityscapes | ADE | Objects | Pascal59 | Pascal60 |
|---|---|---|---|---|---|---|---|
| CaR+SAM | 91.7 | 67.9 | 16.0 | 17.9 | 37.7 | 40.0 | 31.6 |
| CLIPer | 89.8 | 72.2 | 42.5 | 25.0 | 44.7 | 44.6 | 39.5 |
| NERVE | 90.1 | 69.7 | 34.1 | 24.0 | 43.3 | 43.4 | 37.7 |
| Trident | 88.7 | 70.8 | 47.6 | 26.7 | 42.2 | 44.3 | 40.1 |
| **T2S (Ours)** | **92.1** | **74.2** | **47.3** | **27.1** | **45.6** | **47.9** | **41.7** |

**消融（Tab.2, P.7，VOC-20）**：逐模块累加——仅种子生成 73.1 → +EOT 加权 76.8 → +负种子 82.3 → +ISSS 迭代 92.1。负种子和迭代提升最大。

**超参消融（P.8-9）**：
- 起始去噪步（Tab.3）：3/4T=90.4、1/2T=92.1（最优）、1/4T=89.7——过早过晚都差，1/2T 平衡了"物体信号已出现"与"噪声还不大"。
- ACA 分辨率（Tab.4）：仅 8=88.2、8+16=92.1（最优）、8+16+32=82.4、全=76.3——**分辨率越高越差**，印证"高分辨率注意力弥散"的假设。
- 阈值 $\alpha$（Tab.5）：0.5 最优（92.1），0.3→87.2、0.7→84.2，说明 0.5 的取值无需精细调参。
- 迭代轮数（Tab.6）：固定 R=1 → 82.3、R=5 → 86.8，自适应拒绝策略（n=10）→ 92.1——**自适应迭代远超固定轮数**。

**域外泛化（Tab.7, P.9）**：ISPRS Potsdam 航空影像（训练分布外）：CaR+SAM 81.3 vs T2S 83.2。

**计算开销（Tab.8-10, P.10）**——明确的短板：
- 188 s/图、峰值显存 ~20 GB、每类平均 9.1 轮迭代；SAM mask decoder 调用 311.7 次/图、CLIP 验证 21.8 次/图。
- GFLOPs：CaR 338、FreeDA 588、OVDiff 1492、T2S 3851——最贵（主要来自反复 SAM mask decoding + CLIP 验证）。
- 种子生成阶段对物体数不敏感（3.62–3.65 TFLOPs，变化 <1%），瓶颈在扩张/验证环节。

**定性**：多主体场景（Fig.8）、任意虚构角色（Fig.7，如动漫人物——训练过固定类别的方法做不了）、背景类（Fig.9）、小物体与遮挡（Fig.10，CaR 漏检小物体，T2S 边界更准）、与简单替代方案对比（Fig.11：直接阈值化注意力 / top-k 注意力点 / 手工提示 SAM3——要么掩码残缺要么需要人工）。失败模式（Fig.13）：严重遮挡、弱边界、目标与背景重叠 → 漏检或泄漏到邻近物体。

## 6. 图表清单

- Fig.1 框架总览：SD 提取种子 → SAM 区域扩张 (P.1)
- Fig.2 详细流程图：(a)总览 (b)种子生成器初始化 (c)SGS (d)迭代 SGS (e)SAM 扩张 (P.4)
- Fig.3 不同起始去噪步的 ACA 可视化（早聚合=噪声）(P.4)
- Fig.4 不同分辨率的 ACA 可视化（高分辨率弥散）(P.4)
- Fig.5 迭代聚合掩码过程可视化 (P.5)
- Fig.6 ASA 掩码随迭代演化（更细更稀疏）(P.5)
- Fig.7 开放词汇场景下与 CaR 的定性对比 (P.7)
- Fig.8 多主体场景定性结果 (P.7)
- Fig.9 Pascal59 背景类定性结果 (P.9)
- Fig.10 小物体与遮挡下 T2S vs CaR（T2S 更准）(P.9)
- Fig.11 与简单替代方案对比（直接阈值化 / top-k / SAM3 提示）(P.10)
- Fig.12 不同物体尺度与场景复杂度定性 (P.11)
- Fig.13 失败案例分析（弱边界、重叠、严重遮挡）(P.11)
- Table 1 主结果：7 个基准 × 4 类方法 (P.6)
- Table 2 模块消融（Init/SG/负种子/ISSS）(P.7)
- Table 3 起始去噪步消融 (P.8)
- Table 4 ACA 分辨率消融 (P.9)
- Table 5 种子阈值 α 消融 (P.9)
- Table 6 固定迭代 vs 自适应拒绝策略消融 (P.9)
- Table 7 ISPRS Potsdam 域外泛化 (P.10)
- Table 8 推理开销明细（时间/显存/调用次数）(P.10)
- Table 9 种子生成复杂度 vs 物体数 (P.10)
- Table 10 与代表性 training-free 基线的显存/GFLOPs 对比 (P.10)

## 7. 快速总结

- 核心贡献：把 OVSS 重新表述为"种子定位 + 区域扩张"两段式，SD 只做定位（种子）、SAM 只做扩张（点提示），回避了"粗掩码→精修"的脆弱范式；全流程免训练。
- 三个有效组件：EOT token 加权、负种子提示、CLIP 自适应拒绝的迭代分割（ISSS）——消融显示每加一个都显著涨点。
- 反直觉且扎实的发现：高分辨率注意力反而定位更差（Tab.4），起始去噪步取 1/2T 最优（Tab.3）——设计完全由这两个实验支撑。
- 明确局限：推理贵（3851 GFLOPs、188s/图、311 次 SAM 解码/图），远超 CaR；严重遮挡/弱边界/目标重叠场景会漏检。
- 可借鉴：与 3DGS 场景编辑（如 CoGeo-GS 用 Grounding DINO+SAM 提掩码）同属"生成模型/基础模型做定位提示，SAM 做像素级扩张"的编排思路，这一范式正成为免训练分割的主流；迭代抹除已分割区域的做法也可迁移到多实例分割。
