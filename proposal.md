1. 为什么要做这件事
随着 Video Diffusion Model 和 Video World Model 的快速发展，生成视频的视觉质量、时间一致性和指令遵循能力已经显著提升。然而，对于真正面向 world modeling 的模型，仅仅生成"看起来合理"的视频并不足够：模型还需要遵守真实世界中的物理规律，例如物体运动、碰撞、摩擦、流体、光学以及材料交互等。
现有视频物理评测大多依赖两类方案：一类使用人工评价，判断视频中的运动是否"符合物理直觉"；另一类使用 VLM/MLLM 作为自动 judge，通过问答、CoT 或 rubric scoring 判断视频是否物理合理。这类方法具有较强的语义理解能力，但本质上仍然是在判断 physical plausibility，而不是直接测量 physical validity。当视频整体视觉上足够自然时，judge 可能忽略局部但明确的物理错误。已有工作也观察到类似现象，例如 MechVerse (https://arxiv.org/pdf/2605.14843) 中随着任务难度增加，人类评价下降，但部分通用 world-model benchmark 分数反而上升，作者认为原因之一是 judge 会奖励整体视觉合理、但机械约束错误的视频。也就是说，VLM 本身对物理知识的理解也不是很好。
我们希望回答一个更加基础的问题：
是否可以不依赖或者少依赖human/VLM judge，而直接从生成视频的像素中验证其是否满足物理规律？
这一问题的关键困难在于，生成视频通常没有真实的空间和时间标定。我们不知道多少像素对应一米，也不知道一帧对应真实世界中的多少秒，因此很多经典物理公式不能直接使用。设计稿中的核心观察是：如果一个物理约束能够消除这些未知的空间、时间尺度，或者其真假在尺度变化下保持不变，那么就有可能仅依赖像素测量对它进行验证。
因此，我们提出 Judge-Free Physics Evaluation：不训练额外的 VLM judge，也不要求仿真器 ground truth，而是寻找能够从视频中直接观测、且对未知空间/时间标定不敏感的 calibration-invariant physical constraints，将物理评测从"模型觉得对不对"转化为"可测量量是否满足物理关系"。
这种评测方式有三个重要优势：
1. Objective：结果由可测物理量决定，而非 VLM 的主观判断。
2. Interpretable：可以明确指出模型违反了哪条物理关系，以及误差有多大。
3. Hard to Hack：锐化、提高视觉质量、增加纹理等操作无法直接改善物理不变量，因此比 perceptual/VLM score 更难通过视觉 shortcut 获得高分。
更进一步，这项工作也希望明确 judge-free physics evaluation 的能力边界：并非所有物理规律都能够从未标定视频中恢复。与几何量直接相关的力学、光学和表面张力更容易构造精确比值，而温度、电荷量、场强等无法直接由像素确定，因此对应的物理规律往往只能通过方向、对称性、拓扑、序关系或运动学变化间接验证。

---
2. 先前工作
表 1：Human / VLM-based Physics Evaluation
Work
Venue / Time
Coverage / Scale
Evaluation Signal
Main Limitation
VideoPhy
ICLR 2025
688 captions；固-固、固-流、流-流材料交互
Human SA + PC；VideoCon-Physics
主要判断 physical plausibility，而非直接验证物理关系
VideoPhy-2
ICLR 2026
200 actions，4K prompts
Human + VideoPhy-2-AutoEval
Learned evaluator，依赖 VLM 对物理现象的理解
PhyGenBench
2024
160 prompts；27 条规律；力学/光学/热学/材料
三级 VLM-based PhyGenEval
最终正确性仍由 VLM 判断
PhyWorldBench
2025
10 大类 × 5 子类
MLLM zero-shot judge + Anti-Physics
自动 judge 本身可能存在 physics reasoning error
VBench-2.0
2025
18 个能力维度
VLM + expert models
Physics 只是整体评价中的一个高层维度
WorldModelBench
NeurIPS 2025 D&B
7 domains，56 subdomains
Human-annotated learned judger
判断 physics adherence，但不是 law-level measurement
PhyJudge
2026
250 prompts；13 条物理规律
Physics-specialized PhyJudge-9B
专用 judge 更强，但仍属于 learned evaluator
MechVerse
2026
21,156 clips；1,357 assemblies
Video metrics + instruction following + human
外观/流畅度与 mechanical correctness 可以脱钩
WorldJen
2026
16 dimensions
Likert-style VLM evaluation
指出 binary VQA 存在 yes-bias，低分辨率 judge 会漏掉 temporal failure
Physion-Eval
2026
Ego + exo physical videos
Large-scale expert reasoning
高质量但依赖昂贵人工评价
共同问题：这些方法主要回答 "Does the video look physically plausible?"，而不是直接回答 "Does the observed video satisfy a particular physical law?"。
表 2：Reference / Simulator Ground-Truth-based Evaluation
Work
Reference / GT
Core Evaluation
Advantage
Limitation
Physics-IQ
Real physical experiment videos
Spatial IoU、spatiotemporal IoU、weighted IoU、MSE
用真实物理过程作为 reference
比较 trajectory/pixel similarity，不等价于 physical validity
Physics-IQ Verified
Re-verified real-video benchmark
修正样本与 prompt 后重新排名
揭示 reference benchmark 的稳定性问题
修订前后 ranking Kendall (τ) 仅 0.46
Morpheus
130 real physical experiment videos
Scenario-specific physics-informed metrics
比纯视觉 metric 更接近物理
需要真实 reference video
PISA
120 fps controlled real free-fall videos
高精度落体测量
测量可靠
只覆盖单一物理现象
RigidBench
Blender GT
3D trajectory、mask、depth 等
可以直接获得精确 dynamics GT
仅适用于 simulator-defined tasks
NewtonBench-60K
Simulation GT
Trajectory / Chamfer / velocity / acceleration errors
可以精确评价 Newtonian dynamics
需要 simulator ground truth
PhyParam
Explicit physical parameters
Force、mass、friction、restitution、gravity
可直接评价参数控制能力
依赖显式 physical state / parameter GT
与我们的区别：
$$\text{Reference/Simulator methods: Need physical GT}$$
而我们希望：
No Human/VLM Judge + No Reference Video + No Simulator GT
直接从生成视频本身进行物理验证。
表 3：Programmatic / Judge-Free Physics Measurement —— 最相关工作
Work
Core Mechanism
Calibration Handling
Coverage
与我们的关系
Gravity Evaluation
(2512.02016)
Unit-free two-object protocol：
$$\frac{t_1^2}{t_2^2}=\frac{h_1}{h_2}$$
通过无量纲比值直接消除空间尺度、时间尺度、g 等
自由落体 + Galileo equivalence
最强重叠。已经证明 calibration-invariant physical measurement 可行；我们的 novelty 不能再说首次提出该思想
NewtonRewards

Optical flow → velocity proxy → constant-acceleration constraints
在 pixel-frame 坐标中直接构造约束
自由落体、水平抛、斜抛、斜面上下滑
同样 judge-free，但主要集中于 Newtonian constant-acceleration primitives
CrashTwin
Calibration-free 3D reconstruction → momentum / energy evaluation
恢复 latent 3D physical state
Collision dynamics
我们不恢复 metric 3D，而是设计无需 metric state 的 invariant
Visual Chronometer
/ PhyFPS
从视觉 dynamics 回归 Physical FPS
估计未知 temporal calibration
通用 video dynamics
与我们形成两条路线：Estimate Calibration vs Eliminate Calibration
Parallax Number
Dimensionless gauge-invariant camera-motion descriptor
使用无量纲 quantity 消除 gauge ambiguity
Camera motion
方法论与我们的 invariant construction 同源
PhyCoBench
/ PhyCoPredictor
Optical-flow-guided future-frame predictor
学习预测 dynamics
120 prompts / 7 categories
判断"未来像素预测是否正确"，不是直接检查 physical relation
PPE
/ Likelihood Preference
用 diffusion model likelihood 判断 physics
不需要 external VLM judge
12 scenes / 4 domains
Judge-free，但依赖生成模型自身 likelihood，物理可解释性弱
Ours

Systematic calibration-invariant physical constraints
不估计 calibration，而系统寻找对 calibration invariant 的 observables
Mechanics, friction, optics, fluids, EM, surface tension, etc.
从单一 invariant 推进到 taxonomy + multi-domain tests + identifiability boundary
Proposal 定位：2512.02016 已经提出 unit-free physical evaluation，因此我们的贡献不是首次发现"尺度可以约掉"。
我们的区别应该重新定义为：
Isolated Unit-Free Test → Systematic Judge-Free Measurement Framework
表 4：Theory / Motivation：Identifiability 与 VLM Judge Reliability
Direction
Work
Key Finding
对我们的作用
Physics Identifiability
Physics from Video
(2606.00115)
研究视觉轨迹下 dynamics / physical parameters 的 identifiability
为"哪些物理属性可以/不可以从视频恢复"提供理论语言
Physics Identifiability
Physics-as-Inverse-Graphics
已知 physics equation 下从视频估计物理参数
支撑 video → physical parameter inference 的理论背景
Physics Identifiability
Unsupervised parameter estimation
从连续 dynamics 中恢复 latent physical parameters
与我们的 measurable / non-measurable boundary 相关
VLM Judge Reliability
TRAVL / ImplausiBench
VLM 对精心控制的 physical violation 识别能力接近随机
直接支撑"不应该完全依赖 VLM judge"的 motivation
VLM Judge Reliability
JudgeFit
不同 VLM 能感知的 error type 不同
说明单一通用 VLM judge taxonomy 可能产生 evaluator mismatch
VLM Judge Reliability
PhysBench
VLM 对真实物理世界理解仍明显不足
Judge 的 physics knowledge 本身不是 ground truth
VLM Judge Reliability
WorldJen
Binary VQA 存在 yes-bias；低分辨率 judge 漏掉 temporal failures
说明 automatic judge 有结构性 evaluation bias
VLM Judge Reliability
MechVerse
模型可保持 appearance / smoothness，同时产生 mechanically infeasible motion
支撑 visual quality 与 physical validity 的脱钩
定位总结

VLM Judge
Reference GT
Calibration Recovery
Direct Physical Constraint
Systematic Identifiability
VideoPhy / PhyGenBench
✓
✗
✗
✗
✗
Physics-IQ / Morpheus
✗/部分
✓
✗
部分
✗
RigidBench / PhyParam
✗
✓
—
✓
✗
Gravity Eval.
✗
✗
✗
✓
✗
NewtonRewards
✗
✗
✗
✓
✗
CrashTwin
✗
✗
✓
✓
✗
Ours
✗
✗
✗
✓
✓
核心定位：Prior work has demonstrated individual judge-free physical measurements; we aim to systematize them by asking which physical constraints are identifiable and directly testable from uncalibrated generated videos.
同时，还覆盖更多场景，不只是自由落体，还包括光学，热力学，电磁学等等；

---
3. 核心思想
3.1 从 Physics Judgment 到 Physics Measurement
假设真实世界中的空间位置和时间分别为 (x, t)，而生成视频中只能观察到：
$$x_{\mathrm{pix}}=\frac{x}{s}, \qquad n=\frac{t}{\tau}$$
其中 s 表示未知的米/像素比例，τ 表示未知的秒/帧比例。
因此，依赖绝对速度、加速度、力或能量的指标通常无法直接计算。我们的核心思想是：
如果一个物理约束的真假在未知空间尺度 s 和时间尺度 τ 变化下保持不变，则该物理约束可以通过生成视频进行 judge-free evaluation。
换言之，我们不尝试恢复真实的 s 和 τ，而是寻找能够消除这些 nuisance variables 的 observables。
我们总结出4种主要 measurement structure：
暂时无法在飞书文档外展示此内容
这些约束可以通过 SAM2、CoTracker、homography、边缘/直线拟合、颜色统计以及峰值检测等标准视觉工具得到，不需要额外的物理 judge，也不使用深度传感器、标定板或 simulator ground truth。
3.2 Benchmark Design
我们不会将所有可能的物理现象都作为主 benchmark，而是优先选择：
- 低阶甚至零阶测量；
- 单视频即可验证；
- 具有明确理论值或 hard bound；
- 不依赖绝对尺度；
- 在当前视频生成模型中具有较高可生成率；
- measurement noise 可以被可靠控制。
设计稿目前筛选出的高优先级例子包括：摩擦角质量无关性、Plateau 定律、铁屑磁场拓扑、磁铁在铜板上的涡流阻尼、双泡曲率关系、沙漏/水漏差异、浮冰水位、斜面上下运动不对称、滚动惯量、弹跳恢复系数交叉验证、折射一致性、刚体射影不变量以及抛体几何关系等。
第一版 benchmark 预计选取约 10–15 个最稳定、最容易测量的核心任务，覆盖多种物理领域和 measurement structures，其余任务作为 extended benchmark 或 supplementary。
3.3 Measurability Gate
Judge-free evaluation 的一个重要问题是：并不是所有生成视频都能够可靠测量。例如目标可能突然消失、发生严重形变、镜头发生移动，或者 tracking 失败。
因此，我们首先定义 Physical Measurability Rate (PMR)：
$$\text{PMR} = \frac{\#\text{measurable generated videos}}{\#\text{all generated videos}}$$
通过 temporal mask consistency、track length、SAM2/CoTracker disagreement 和 camera drift 等 QC signal 判断视频是否满足测量条件。设计稿中特别强调必须显式报告 PMR，而不能简单删除 tracking failure 的样本，否则模型可以通过"生成无法测量的视频"获得虚假的高 physics score。
最终每个模型同时报告：
$$\text{PMR}, \qquad \text{Physics Accuracy} \mid \text{measurable}$$
从而分别衡量"能不能生成一个稳定、可观察的世界"和"生成出来以后是否符合物理规律"。
3.4 具体指标
指标
含义
计算方式
趋势
Physical Measurability Rate (PMR)

有多少生成视频能够被可靠测量（需要可靠frames > 80%？之后可以再更改）
$$\mathrm{PMR}=\frac{N_{\text{measurable}}}{N_{\text{all}}}$$
↑
Physical Error (PE)
与理论物理关系偏离多少

对每个场景定义 calibration-free residual ($$e_i$$)，对可测样本取平均：
$$\mathrm{PE}=\frac{1}{N_m}\sum_i e_i$$
↓
Physical Pass Rate (PPR)
有多少视频在允许误差内满足物理规律
$$\mathrm{PPR}=\frac{1}{N_m}\sum_i\mathbf{1}[e_i<\epsilon_k]$$
↑
Hard Violation Rate (HVR)
有多少视频违反明确的物理硬约束
$$\mathrm{HVR}=\frac{N_{\text{hard violation}}}{N_m}$$
↓
目前暂定这4个指标

---
3.5 First Frame怎么来？
1. 来自于网络，进行筛选
2. 来自于GPT Image 2生成
3. 实拍
4. 预期结果
我们预期通过该 benchmark 得到以下四类主要结果。
4.1 当前 SOTA Video Models 仍存在明显物理错误
即使最新模型已经能够生成视觉上高度逼真的视频，在具体 calibration-invariant physics constraints 上仍可能产生系统性错误。例如：
- 不同质量但同材质物块在不同时间开始滑动；
- 沙漏的流速错误地表现出与液体类似的变化；
- 弹跳高度和弹跳周期给出的恢复系数彼此矛盾；
- 视频中的 rigid object 在运动过程中发生几何变形；
- 视觉上自然的运动违反明确的单侧物理 hard bound。
这些错误可以通过具体的 numerical residual 定位，而不仅仅得到一个"physics score"。
4.2 VLM Judge 与真实物理正确性存在系统性 Gap
我们预期存在一批：
$$\text{High VLM Score} \quad+\quad \text{Large Physics Violation}$$
的视频。
这些 case 将成为最重要的 qualitative result：视频在整体视觉上高度自然，因此能够获得较高的 VLM/perceptual score，但直接测量后却违反明确的物理不变量。
最终希望证明：
Visual plausibility is not equivalent to physical validity.
并进一步量化不同 VLM judge 与 judge-free physics score 的 correlation。
4.3 不同模型的 Physics Failure Mode 明显不同
除了提供一个总体排名，我们更关注模型的 physics profile。
例如某些模型可能在：
- geometry / projectile motion 上表现较好；
- contact / friction 上明显失败；
- fluid / granular dynamics 上较弱；
- symmetry constraint 正确，但 quantitative invariant 错误。
最终 benchmark 可以得到类似：
$$\text{Model} \rightarrow [\text{Geometry},\text{Dynamics},\text{Contact},\text{Fluid},\text{Material},\ldots]$$
的 diagnostic profile。
这比单一 Physics Score 更有助于分析不同模型究竟学到了哪些 physical priors。
4.4 建立 Judge-Free Physics Evaluation 的能力边界
最后，我们希望得到一个比 benchmark 排名更一般的结论：
Generated-video physics is directly measurable only when the target physical constraint remains identifiable under the unknown observation calibration.
因此，一部分物理规律可以通过 geometry、ratio、symmetry、topology、ordering 和 event consistency 直接验证；而需要绝对温度、质量、电荷量、场强等 latent physical quantities 的规律，则原则上无法仅从普通生成视频中唯一恢复。
因此 Judge-Free Physics 并不是要取代所有 VLM/human evaluation，而是把整个物理评测问题划分为两部分：
Directly measurable physics
Latent physics requiring semantic judgment
这一边界本身将成为论文的重要结论，并为未来 video world model evaluation 提供一种新的 measurement-centric framework。
4.5 我们benchmark的评测结果和人类对物理的理解对齐
希望我们benchmark测试来的“物理好的视频”，人类也觉得物理好