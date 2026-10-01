const PAPERS_DATA = {
  "last_updated": "2026-10-01 05:28:45 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Semifactual Credit-Augmented Policy Optimization",
      "authors": [
        "Junshu Pan",
        "Zhizhang Fu",
        "Shulin Huang",
        "Yiran Ding",
        "Zifan Cheng",
        "Wenqi Shao",
        "Qiaosheng Zhang",
        "Yue Zhang"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) has improved the reasoning capabilities of large language models (LLMs), yet their predictions remain sensitive to task-irrelevant prompt features. We investigate this sensitivity through semifactual prompt interventions that preserve the underlying problem and its answer. Our analysis reveals substantial variation in token-level sensitivity and shows that suppressing high-drift token candidates during decoding improves reasoning accuracy without updating model weights. These findings highlight a limitation of Group Relative Policy Optimization (GRPO), which assigns the same outcome-derived advantage to every response token and may reinforce potential spurious dependence alongside useful reasoning. Motivated by this observation, we introduce Semifactual Credit-Augmented Policy Optimization (SCAPO), a causally inspired variant of GRPO that incorporates semifactual stability into token-level credit assignment. SCAPO measures token probability drift for fixed responses under semifactual interventions and uses normalized stability scores to reduce advantages for relatively unstable tokens during early training, while granting no additional credit for stability alone. On Qwen3-4B-Base and Qwen3-1.7B-Base, SCAPO improves AIME 2024-2026 accuracy over GRPO by 5.63 and 4.17 percentage points, respectively. At both model scales, SCAPO achieves the best results on most evaluated mathematics benchmarks and all evaluated out-of-distribution benchmarks among the compared methods. These results suggest that semifactual stability provides an effective training signal for improving reasoning and generalization through finer-grained credit assignment in RLVR. The code is available at https://github.com/DtYXs/SCAPO.",
      "published": "2026-09-30T17:59:56Z",
      "abstract_url": "http://arxiv.org/abs/2609.40360v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40360v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Scaling Laws for Looped Mixture of Experts",
      "authors": [
        "Yanbei Chen",
        "Anirudh Goyal",
        "Raghuraman Krishnamoorthi"
      ],
      "abstract": "Looped transformers and Mixture-of-Experts (MoE) offer complementary routes to efficient scaling: recurrence increases computational depth at fixed parameters, while MoE sparsity expands total capacity at fixed active compute. Yet existing scaling laws model recurrence or sparsity in isolation. In this work, we introduce Loop Scaling Laws, the first scaling law to jointly model recurrence and sparsity alongside model size and data. At its core is a bounded, sparsity-conditional recurrence mapping that characterizes the effective-parameter gain from looping and how sparsity raises this gain. The laws predict the held-out loss of looped models more accurately than prior alternatives, and recover the standard dense and MoE scaling laws as special cases. Beyond prediction, the fitted laws provide a principled foundation for designing looped MoE models under compute and memory constraints. Downstream evaluations further demonstrate the complementary benefits of the two axes: sparsity delivers ~3x active-parameter efficiency, recurrence yields ~2x total-parameter efficiency on reasoning, and joint scaling further advances the performance frontier. As a practical extension, we show these gains hold at trillion-token scale: at matched training compute, a looped MoE with law-derived recurrence matches a ~2x larger non-looped MoE on the reasoning benchmarks, while enabling test-time scaling through recurrence.",
      "published": "2026-09-30T17:53:47Z",
      "abstract_url": "http://arxiv.org/abs/2609.40316v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40316v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "DynaHarness: A Dynamic Physical Harness for Self-Evolving Robot Agents",
      "authors": [
        "Haoyuan Deng",
        "Jiebin Liu",
        "Tengxiao Zhang",
        "Langning Yan",
        "Hongye Cao",
        "Ziwei Wang"
      ],
      "abstract": "Pretrained robot policies provide useful action priors, but long-horizon manipulation still requires coordination between semantic reasoning and physical execution. Semantic reasoning operates at a coarser timescale than physical interaction, while episode-level failures provide limited guidance on which system component should be revised. We propose DynaHarness, a dynamic physical harness that couples semantic reasoning with physical governance through a shared execution contract and turns failure evidence into validated capability revisions. To be more specific, the slow brain proposes capabilities and symbolic arguments, while the fast brain grounds and monitors commands, refuses unresolved actions, substitutes capabilities, and requests replans when needed. The physical execution contract bounds each accepted command and records execution evidence across analytic skills, recovery skills, and the frozen VLA. Failure attribution localizes faults in these records and directs targeted revisions of reusable capabilities or execution mechanisms. Paired regression checks govern admission or rejection, closing the self-evolution loop. On LIBERO-Pro, DynaHarness achieves 75.2% on 800 newly sampled initial states, compared with 17.5% for the frozen policy. With the same capability library, full dynamic execution reaches 74.0% versus 63.9% under nominal one-step replanning. This demonstrates the value of DynaHarness as a dynamic physical harness that governs how existing capabilities are grounded, monitored, and coordinated during execution. Our project page is at https://denghaoyuan123.github.io/Dynaharness_page/.",
      "published": "2026-09-30T17:52:09Z",
      "abstract_url": "http://arxiv.org/abs/2609.40306v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40306v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "How Much of a Harness Does a Strong Agent Need for Autonomous ML Engineering?",
      "authors": [
        "Kirill Brilliantov",
        "Alejandro Hernández-Cano",
        "Emmanuel Abbé"
      ],
      "abstract": "Recent autonomous machine learning engineering (MLE) agents have made significant progress on public leaderboards. Often motivated by progress stagnation over long-horizon cycles and limited Large Language Model (LLM) primitives, modern MLE agents are deployed on top of increasingly elaborate machinery: multi-agent orchestrators, dedicated retrieval subagents, and more. While such harnesses expand, the use of more primitive but improved coding agents - where LLMs have direct access to the execution environment through read, write, and bash primitives - has received little attention in the field. In this paper we find that, under an equal time budget and the same frontier LLM backbone, open-source state-of-the-art harnesses provide no advantages over a single session of a minimal-harness coding agent baseline, pointing to the backbone as the primary driver for performance. Via a series of large-scale systematic ablation studies, we argue that the machinery layers become redundant in the coding agent setting. We conclude that the effort spent elaborating hand-crafted harnesses around strong models yields poor returns for current MLE benchmarks.",
      "published": "2026-09-30T17:51:30Z",
      "abstract_url": "http://arxiv.org/abs/2609.40303v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40303v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "cua-speedrun: Standardized Benchmarking of the Speed of Computer-Use Agents",
      "authors": [
        "Pranjal Aggarwal",
        "Lawrence Keunho Jang",
        "Sean Welleck",
        "Daniel Fried",
        "Ruslan Salakhutdinov",
        "Jing Yu Koh"
      ],
      "abstract": "Computer use agents (CUAs), which use graphical user interfaces (GUIs) to complete tasks on a computer, have recently surpassed human performance on many standard benchmarks, including difficult long-horizon tasks. Their capabilities are undoubtedly impressive, however, a key barrier to the widespread adoption and deployment of CUAs remains their speed and cost. Progress towards faster yet capable CUAs requires reliable evaluation of their speed, but many CUA benchmarks currently face a reproducibility crisis. Benchmarks are based on complex infrastructure with varying machine and container configurations that confound the evaluation of the execution speed of CUAs. Towards addressing this gap, we propose cua-speedrun, which introduces standardized infrastructure and task sets, with a focus on evaluating the speed and efficiency of CUAs. cua-speedrun uses a uniform virtual machine setup and execution pipeline, along with a common agent interface that enables single-agent implementations to operate seamlessly across different benchmarks. Across four different CUA benchmarks, we evaluate how reasoning effort, agent harnesses, and environment latency affect performance, speed, and cost. We find no single model family is optimal for all three; none of the open-weight models are on the frontier, and also, unintuitively, for some models increasing the reasoning effort can speed up task completion, while faster environment input-output can slow down overall task completion time. We also demonstrate that we can effectively reduce the evaluation task set of most CUA benchmarks without degrading overall statistical power, allowing for more efficient benchmarking and comparison. We believe cua-speedrun will enable structured progress towards fast, efficient CUAs, unlocking new real-world use cases and applications. All code, infrastructure, and analysis are available at https://cuaspeedrun.com.",
      "published": "2026-09-30T17:48:06Z",
      "abstract_url": "http://arxiv.org/abs/2609.40284v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40284v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PhantomEnvironments: Training LLM Agents in Fictional Worlds",
      "authors": [
        "Anmol Kabra",
        "Swathi Saravana Selvam",
        "Albert Gong",
        "Chao Wan",
        "Christian Belardi",
        "Dongyoung Go",
        "Katie Z. Luo",
        "Kilian Q. Weinberger"
      ],
      "abstract": "Training LLM agents with reinforcement learning (RL) is bottlenecked by environments, which must provide verifiable rewards, support long-horizon interaction, and scale cheaply. Existing approaches rely on costly human-curated data or on LLM-generated environments that risk hallucinations and benchmark contamination. We show that LLMs can instead be trained into capable search agents using synthetic environments generated entirely by rules, whose generation requires no LLM and has zero marginal cost. We build PhantomEnvironments, multi-turn RL environments from fictional worlds, where agents must search a corpus of templated articles to answer multi-hop questions. Despite sharing no facts with the real world, these strikingly simple environments yield agents that transfer to real-world multi-hop search benchmarks, often outperforming real-world training data on newer benchmarks. Trained agents generalize to unseen fictional universes, and Qwen models learn to scale their search budget roughly linearly with question difficulty, suggesting emergent search scaling from environment interaction alone. Ablating environment complexity reveals that hop count drives transfer more than constraints or comparisons: even the simplest rule-generated environments are a surprisingly effective, free resource for training generalizable LLM agents.",
      "published": "2026-09-30T17:26:57Z",
      "abstract_url": "http://arxiv.org/abs/2609.40221v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40221v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PrefPI: Preference-Guided Steering into Out-of-Distribution Behaviors",
      "authors": [
        "Seungeun Rho",
        "Wontaek Kim",
        "Danfei Xu",
        "Sehoon Ha"
      ],
      "abstract": "We present PrefPI (Preference-Guided Policy Iteration), an iterative framework for steering pretrained generative robot policies using only relative preferences over self-generated trajectories. Unlike prior preference-learning methods that primarily sharpen modes already represented by the policy, we study steering beyond the initial effective support, where desired behaviors are rarely or never observed under the initial policy. Our key idea is to formulate preference learning as preference-conditioned generative modeling: preferred trajectories define a conditional distribution, whose density ratio with the broader behavior prior provides an implicit preference signal amplified by classifier-free guidance (CFG). Repeating this preference-conditioned modeling and guidance step yields a form of preference-guided policy iteration, turning incremental improvements toward previously inaccessible behaviors. Across diffusion policies and the PI0.5 flow- matching VLA in simulation and the real world, PrefPI produces substantial behavioral shifts with limited feedback. In particular, PrefPI increases object transport height from 10.7 cm to 19.8 cm on real hardware with only 150 preference-labeled trajectories.",
      "published": "2026-09-30T16:58:38Z",
      "abstract_url": "http://arxiv.org/abs/2609.40165v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40165v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Game-Guided Skill Discovery through Self-Play for Playable Agent Control",
      "authors": [
        "Seungeun Rho",
        "Jeonghwan Kim",
        "Xue Bin Peng",
        "Sehoon Ha"
      ],
      "abstract": "We present Game-Guided Skill Discovery (GGSD), a framework that uses self-play in games to discover motor skills that are directly playable by humans. Playable skills provide a compact abstraction for controlling embodied agents through a small set of learned behaviors rather than low-level actions. To be effective, these skills should be semantically distinct, interpretable, and expressive; properties that existing unsupervised skill-discovery methods often fail to achieve simultaneously. GGSD achieves these desiderata by grounding skill discovery in competitive gameplay. A hierarchical agent competes against its past selves, with a high-level policy selecting from a small discrete skill set and a skill-conditioned low-level policy learning the corresponding behaviors. After training, a human can replace the high-level policy and directly control the agent through the same discrete skills. Despite the small number of high-level actions, skill transitions give rise to emergent combo behaviors, expanding expressivity beyond individual primitives. Across Ant, Franka-arm, and Unitree G1 environments, we show that GGSD produces human-playable skills that humans can compose to solve unseen tasks, such as Maze and CubePush, without additional training. An interactive demo is available at https://ggsd-demo.github.io.",
      "published": "2026-09-30T16:49:51Z",
      "abstract_url": "http://arxiv.org/abs/2609.40137v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40137v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "On the (In)effectiveness of AMR Augmentation for Large Language Models",
      "authors": [
        "Hoa Quynh Nhung Nguyen",
        "Jacopo Staiano",
        "Michael Sullivan"
      ],
      "abstract": "While Abstract Meaning Representation (AMR) has historically improved performance on a range of NLP tasks, the benefit---or lack thereof---of AMR augmentation for modern LLMs is thus far unclear. In this paper, we attempt to reproduce recent work that reported substantial downstream gains from AMR augmentation, finding that these are likely due to specific choices in the experimental settings used: using a consistent and unified protocol for hyperparameter selection, we observe that text-only baselines consistently match or exceed the performance of AMR-augmented models. To investigate this null result, we introduce a perplexity-based probe measuring the degree to which AMR provides an LLM with supplemental relational knowledge not already available to the model. We find that AMR augmentation does not help LLMs improve their understanding of relational content in the sentence, indicating that augmenting these models with AMR offers no clear benefit on downstream tasks.",
      "published": "2026-09-30T16:45:14Z",
      "abstract_url": "http://arxiv.org/abs/2609.40121v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40121v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "PTNO: Training Neural Operators with Noisy Monte Carlo Estimates for Particle Transport Problems",
      "authors": [
        "Yubo Cao",
        "Xi Deng",
        "Mengqi Xia",
        "Vignesh Gopakumar",
        "Ander Gray",
        "Anima Anandkumar"
      ],
      "abstract": "Particle transport under multiple scattering is central to radiative transfer and plasma physics, yet high-fidelity Monte Carlo (MC) simulations must trace prohibitively many particles. Learning-based surrogates can amortize this cost, but typically train on expensive, well-converged MC solutions. We propose the Particle Transport Neural Operator (PTNO), a neural operator that learns particle transport surrogates directly from noisy, low-cost MC labels. Such labels pose two challenges: (1) high variance, which destabilizes standard supervised learning, and (2) a high dynamic range (HDR) spanning many orders of magnitude. For the first, we learn the solution operator from noisy labels of many configurations, amortizing MC cost and generalizing to unseen configurations. Because MC labels are unbiased, we show that the squared loss on them shares its minimizer with the loss on converged solutions, and our budget-allocation study over training scenes $M$, MC samples per render $N$, and independent renders per scene $K$ shows that many noisy scenes beat fewer converged ones. For the second, a nonlinear transform such as the logarithm biases noisy supervision. Instead, PTNO keeps labels in physical space and enforces positivity with a softplus output layer that represents small values effectively. We further train with a pointwise relative $L_2$ loss (PRelL2), the stop-gradient relative loss of HDR denoising and neural rendering, which normalizes each residual by the stop-gradient prediction instead of the noisy label. We demonstrate PTNO on neutron transport in fusion reactors and radiative transfer in participating media. On the two neutronics tasks, PTNO is $10^4$-$10^5\\times$ faster than converged MC on the same CPU and $10^3$-$10^5\\times$ cheaper than MC at matched accuracy; on the two radiative-transfer tasks, MC at matched accuracy costs $0.8$-$11\\times$ as much as PTNO.",
      "published": "2026-09-30T16:31:40Z",
      "abstract_url": "http://arxiv.org/abs/2609.40090v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40090v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "MeanVoiceFlow2: Joint Optimization of Mean Flow and Content Encoder for Fast One-Step Zero-Shot Voice Conversion",
      "authors": [
        "Takuhiro Kaneko",
        "Hirokazu Kameoka",
        "Kou Tanaka",
        "Yuto Kondo"
      ],
      "abstract": "Flow-matching approaches to voice conversion (VC) have gained attention owing to their high speech quality and strong speaker similarity. Among them, one-step models such as MeanVoiceFlow are particularly attractive because they enable efficient inference; however, their reliance on a computationally intensive content encoder remains a bottleneck. We therefore propose MeanVoiceFlow2, a framework that jointly optimizes a flow-based conversion module and a computationally efficient content encoder. The model is trained through conversion distillation using MeanVoiceFlow and the reconstruction of real data. We further incorporate diffusion-GAN training with sample mixing and teacher-guided conditioning augmentation to enhance realism and disentanglement. Experiments on zero-shot VC showed that MeanVoiceFlow2 achieved higher perceptual quality and approximately $9\\times$ faster inference than MeanVoiceFlow while maintaining comparable speaker similarity. Audio samples are available at https://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/meanvoiceflow2/.",
      "published": "2026-09-30T16:31:01Z",
      "abstract_url": "http://arxiv.org/abs/2609.40087v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40087v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.LG",
        "eess.AS"
      ]
    },
    {
      "title": "BatSLAM 2.0: Sequence-Verified Sonar Place Recognition in a Robust Pose Graph",
      "authors": [
        "Jan Steckel"
      ],
      "abstract": "Echolocating bats can navigate dark and cluttered spaces using echolocation. Over a decade ago, BatSLAM showed that a robot with a biomimetic binaural sonar can build a topological map of the environment, by recognizing places from the received acoustic signals. Sonar place recognition, however, is ambiguous by nature: corridors produce nearly identical echo trains, and wrong loop closure can collapse the topological map. In this paper, we introduce BatSLAM 2.0, a novel sonar-only SLAM system built from three elements: an updated acoustic front-end, a sequence verifier that tracks and verifies loop closure candidates and a pose graph implemented on a high performance factor graph framework. The system was thoroughly evaluated both in simulated as well as real world recordings. In both cases, the BatSLAM2.0 algorithm shows the capability of robust topological map creation, countering map collapse, and robust scaling of map size.",
      "published": "2026-09-30T16:29:57Z",
      "abstract_url": "http://arxiv.org/abs/2609.40085v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40085v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "LongEmo: Towards Emotion Understanding and Reasoning in Long Videos",
      "authors": [
        "Shuo Zhang",
        "Yifan Zhou",
        "Han Wang",
        "Jinsong Zhang",
        "Jingyu Li",
        "Hongbing Li",
        "Zhejun Zhang",
        "Chengyi Zhao",
        "Yuquan Hao",
        "Yitong Liu",
        "Jiyin Li",
        "Ruiqi Tang",
        "Zixuan Lin",
        "Yi Luo",
        "Xurui Zhang",
        "Ronghao Chen",
        "Huacan Wang",
        "Lei Li"
      ],
      "abstract": "While recent Multimodal Large Language Models (MLLMs) have shown promise in affective computing, their reasoning capabilities are largely confined to short video clips with limited interactions. However, real-world emotions are not merely isolated instantaneous reactions but dynamic and cumulative processes deeply shaped by past experiences and ongoing events. To bridge this gap, we introduce LongEmoBench, a benchmark dedicated to emotion understanding and reasoning in long videos. It assesses progressive capabilities scaling from continuous scene interactions to complex episodic developments. Furthermore, we propose LongEmo, a novel memory-augmented agentic framework designed to tackle the immense challenges of long-range affective reasoning. LongEmo processes continuous video streams to construct an Event Memory Graph, explicitly modeling long-range dependencies and capturing emotional dynamics across discrete events. Given a question, the agent retrieves a query-relevant event stream from the graph, iteratively integrating multimodal memories and relational dependencies to deduce the final answer. Extensive evaluations of 17 representative methods reveal that they struggle significantly with emotion understanding and reasoning in long videos. In contrast, LongEmo achieves state-of-the-art performance, demonstrating the efficacy of its event-centric memory architecture.",
      "published": "2026-09-30T16:27:25Z",
      "abstract_url": "http://arxiv.org/abs/2609.40079v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40079v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Inference Auctions",
      "authors": [
        "Keegan Harris",
        "Siddharth Prasad",
        "Asher Trockman",
        "Nika Haghtalab",
        "Michael I. Jordan"
      ],
      "abstract": "When inference demand exceeds available compute capacity, model providers must decide which requests should be served first. Users have different tolerances for delay from an LLM API, but current priority pricing schemes compress these differences into coarse fixed-price service tiers. We design an inference auction that allows users to bid for faster service. Our auction allocates priority in an economically efficient way without sacrificing latency, and we develop fast algorithms for implementing prices that incentivize truthful bidding. We also design an autobidding agent for our inference auction, where users specify an inference budget and the autobidder dynamically adjusts its bids over time to maximize user utility subject to the budget constraint. Experiments validate the practicality of our auction: it increases system welfare while maintaining the cache utilization and latency advantages of SGLang, a state-of-the-art inference serving framework.",
      "published": "2026-09-30T16:24:10Z",
      "abstract_url": "http://arxiv.org/abs/2609.40070v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40070v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.GT"
      ]
    },
    {
      "title": "Efficient Active Auditing of Multi-Group Fairness with Bias Probes",
      "authors": [
        "Ayoub Ajarra",
        "Debabrota Basu"
      ],
      "abstract": "Over the past decade, Machine Learning (ML) has been trained under dual objectives: minimizing prediction error via Empirical Risk Minimization (ERM) while controlling unfairness bias. In practice, however, fairness-aware training often yields limited improvements over standard ERM, making reliable post hoc auditing essential. Existing auditing approaches for black-box models either rely on model reconstruction --exposing systems to extraction attacks-- or directly estimate fairness metrics, offering limited insight into which regions of the data distribution drive bias. More fundamentally, property-specific auditing --aimed at extracting only targeted fairness information without reconstructing the model-- remains poorly understood. In this work, we introduce the bias probe framework, which enables targeted and adaptive querying to reveal bias structure while preserving model confidentiality. Building on this framework, we propose ALeBi, an active auditor that learns such probes to efficiently estimate multi-group fairness metrics. We establish novel sample complexity guarantees governed by a property-specific complexity measure, resolving a previously posed open question, and extend our analysis to adversarial settings where the model owner may strategically obscure bias. Our results uncover a fundamental trade-off between model confidentiality and reliable auditing, and show that property-specific probing enables both accurate estimation and interpretable identification of high and low-bias regions. Extensive experiments support our theoretical findings and demonstrate the practical effectiveness of our approach.",
      "published": "2026-09-30T16:05:25Z",
      "abstract_url": "http://arxiv.org/abs/2609.40034v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40034v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CY",
        "stat.AP",
        "stat.ML"
      ]
    },
    {
      "title": "Fenchel Tilting: Weighted Correction for Efficient Finetuning of Generative Models",
      "authors": [
        "Maksim Bobrin",
        "Maksim Zhdanov",
        "Dmitry Dylov"
      ],
      "abstract": "Adapting a pretrained generative model to an arbitrary preference expressed as a utility function underlies reward alignment, guided design, and constraint satisfaction, enabling diverse applications. Existing fine-tuning methods trade off generality against computational cost: they either restrict the family class of supported preferences to keep optimization simple or preserve generality at the expense of efficiency. We introduce Fenchel Tilt Flow Control (FTFC), which decouples utility optimization from generative-model fitting. FTFC first optimizes for a target distribution by jointly fitting an effective reward and density-ratio weights on pretrained samples. Method combines the utility's variational structure with Fenchel duality, supporting general $f$-divergence penalties that determine how rewards are transformed into an distribution-correction weights. These weights are then frozen and used to modify a diffusion or flow model in a single stage of importance-weighted denoising or flow matching, without differentiating through sampling trajectories. We establish exact duality for concave utilities under suitable conditions and show that weighted fitting reproduces the optimal target distribution for a given utility. Across image and molecule generation benchmarks, FTFC improves over baselines on diverse preference functions, while also being up to $20\\times$ more efficient. roposed method enables adaptation beyond expected-reward maximization without complex optimization, while preserving robustness for more general class of the utility functions compared to baselines.",
      "published": "2026-09-30T16:02:58Z",
      "abstract_url": "http://arxiv.org/abs/2609.40030v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40030v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Who Verifies the Graph? Misspecification Attacks on Causal Action Verification for Language Agents",
      "authors": [
        "Fabio Rovai"
      ],
      "abstract": "Causal action verifiers gate an agent's state-changing tool calls by checking whether each proposed intervention is identifiable against a committed action-state graph, and they issue a certificate that carries the identification argument and a one-sided lower confidence bound. One such verifier, CIVeX, reports zero false executions on a confounded tool-use benchmark. We red-team it by corrupting only the committed graph. Omitting a single bidirected edge takes it from zero false executions to 15.3% at the benchmark's published confounding strength, with 91% of its executions harmful and utility falling from +2.27 to +0.35. Reversing one arrowhead, so that a mediator is committed as a confounder, gives 48.9% false executions and no correct ones. Every one of these actions carries an internally valid certificate. An attestation step that tests each observationally certified execution against a bounded randomised sample detected both attacks, with 2 false alarms in 555 executions on a truthful graph; refusing what fails the test, or cannot be tested, gave zero false executions in every setting we measured. It does not restore beneficial execution: at the published strength 97.1% of beneficial actions are still never executed, because the same misspecification rejects them before attestation runs. Those rejections carry certificates too, and auditing them works, but its cost scales with the number of rejections rather than the number of executions. Recovering safety costs 127 experiments per 1,050 actions; recovering the lost value costs 614 more, at which point the audited verifier makes the honest graph's decisions on every instance and spends exactly its experiment budget. An audit that inspects only executions protects against wrongful action. Wrongful inaction has to be paid for separately.",
      "published": "2026-09-30T16:02:04Z",
      "abstract_url": "http://arxiv.org/abs/2609.40027v1",
      "pdf_url": "https://arxiv.org/pdf/2609.40027v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "TACTIC: Temporal and Context-Aware LLM Tactical Planning for Roadside LiDAR Attacks",
      "authors": [
        "Yiming Gao",
        "Shaocheng Luo"
      ],
      "abstract": "Physical LiDAR attacks are often evaluated using fixed primitives and manually selected parameters, despite their strong dependence on surrounding traffic. We present TACTIC, a scene-aware framework that uses a multimodal large language model (MLLM) to coordinate state-adaptive roadside LiDAR attacks. Under a gray-box threat model, TACTIC relies only on an attacker-operated roadside perception stack, without accessing the victim LiDAR's native point clouds or internal processing. Local perception provides metric vehicle states, while the MLLM combines these measurements with roadside imagery to infer relational traffic context and construct a semantic scene graph. Based on this representation, TACTIC selects and configures two complementary primitives: \\emph{push-away}, which shifts the perceived range of a lead vehicle, and \\emph{phantom-obstacle braking}, which triggers emergency braking through obstacle injection. Measured traffic states and empirically calibrated constraints ground the generated tactics in physically feasible operating regions. To accommodate MLLM latency, TACTIC overlaps reasoning and execution asynchronously while high-rate local perception detects scene changes and triggers replanning. Across 280 randomized CARLA trials, the full policy achieves a 100% collision rate, compared with 35% for a fixed rule, 60% for random selection, and 75% for a restricted LLM using mode selection with default parameters. Joint physical-and-image input achieves 100% success, versus 65% with physical measurements alone and 75% with imagery alone, while asynchronous $Δ$ refresh reduces scene-mutation response from 7.4 s to 2.0 s. These results show that scene-dependent tactical planning can expose context-sensitive LiDAR failure modes that fixed attack policies may miss.",
      "published": "2026-09-30T15:33:46Z",
      "abstract_url": "http://arxiv.org/abs/2609.39969v1",
      "pdf_url": "https://arxiv.org/pdf/2609.39969v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling",
      "authors": [
        "Yuliana Shakhvalieva",
        "Dmitrii Kharchev",
        "Viacheslav Bezrukov",
        "Inessa Fedorova",
        "Dmitry Bocharov",
        "Ivan Oseledets",
        "Valerii Ternovskii"
      ],
      "abstract": "Recursive reasoning models apply a small shared Transformer block many times to refine a latent state. This gives them large effective depth with few parameters and makes them strong on algorithmic tasks. Such compact solvers are natural candidates for tools that an LLM can call on narrow algorithmic subproblems. However, existing models such as HRM, TRM and URM differ in architecture, gradient propagation and training procedure simultaneously. This makes it hard to tell what drives their performance, and their optimization is still poorly understood and often unstable. In this work we address both of these gaps. First, we study these questions under a unified experimental pipeline spanning six algorithmic domains. Individual controlled ablations are performed on representative domains, while the resulting recipe is evaluated across the full suite. The study reveals a surprisingly simple recipe for stable and generalizable recursive reasoning: an intermediate gradient horizon, large physical batches and controlled updates of the recurrent state. An explicit hierarchical architecture is not needed. Second, we combine these findings into a stable 13.6M-parameter model that achieves the strongest overall performance among the evaluated recursive baselines, with particularly large gains on out-of-distribution generalization. It raises Arithmetic OOD accuracy to 71.2%, from 36.2% for the strongest baseline, while reaching 98.41% on Sudoku and 59.5% pass@2 on ARC-AGI-1. Our results show that, within the recursive architectures studied here, performance depends strongly on how recurrence is optimized and stabilized. More broadly, it shows how AI systems can be improved by optimizing their components one at a time.",
      "published": "2026-09-30T15:32:55Z",
      "abstract_url": "http://arxiv.org/abs/2609.39967v1",
      "pdf_url": "https://arxiv.org/pdf/2609.39967v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning When and How to Intervene: A Hindsight-Distilled Sentinel for Coding Agents",
      "authors": [
        "Jiangrui Zhao",
        "Chenglong Li",
        "Meng Zhang",
        "Xiaoting Du"
      ],
      "abstract": "Coding agents solve repository-level tasks through sequences of actions, where a single erroneous action can misdirect subsequent decisions and increase recovery costs. Existing approaches use execution feedback for recovery or specialized checks to block errors, but deciding before execution whether intervention will benefit eventual task completion remains challenging. To address this challenge, we propose HiSentinel, a hindsight-distillation framework that trains lightweight 0.6B and 1.7B sentinels to select pre-execution interventions aimed at improving task completion rather than correcting every imperfect action. A privileged teacher uses recorded execution outcomes as evidence for intervention judgments, which are distilled into a causal student that receives only the pre-action context and proposed action. Beyond identifying whether and when to intervene, the sentinel must also provide actionable feedback that helps the coding agent recover or obtain necessary human input. To support these capabilities, we introduce SWE-Intervene, an action-level dataset constructed from software-engineering trajectories that annotates whether an action should be allowed, autonomously redirected, or paused for human assistance, together with corresponding intervention feedback. Across SWE-bench Verified Mini and Ask or Assume, HiSentinel consistently improves task completion across Sentinel scales and coding-agent families, with gains of up to 14% and 10%, respectively, while maintaining competitive token consumption. These results demonstrate that lightweight pre-execution intervention can effectively prevent error propagation and improve the reliability of autonomous coding agents.",
      "published": "2026-09-30T15:26:16Z",
      "abstract_url": "http://arxiv.org/abs/2609.39957v1",
      "pdf_url": "https://arxiv.org/pdf/2609.39957v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
