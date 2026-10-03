const PAPERS_DATA = {
  "last_updated": "2026-10-03 04:59:24 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "One Basis to Animate Them All: Gaussian Blendshape Distillation for Real-Time Avatars",
      "authors": [
        "Ramazan Fazylov",
        "Stamatis Lefkimmiatis",
        "Ivan Laptev"
      ],
      "abstract": "3D Gaussian avatars support fast rendering, however, their real-time animation is often challenged by the costly neural inference. We address this bottleneck and show that the animation of pretrained avatar models can be closely approximated by a linear combination of identity-independent blendshapes. Building on this finding, we introduce GALA (Gaussian Animation via Linear Approximation), a distillation method that replaces per-frame heavy neural decoding with a shallow coefficient predictor and a linear blend. To improve fidelity and reduce memory requirements, we propose to construct the basis using block-local PCA under a rendering-aware metric and a memory budget. Our method learns a shallow MLP network to predict blendshape coefficients and applies to various animation architectures without retraining original models. We validate GALA by accelerating the inference of three distinct avatar models for 3D animation of facial expressions and full-bodies with clothing dynamics. Across these models, our distillation generalizes to held-out identities and reduces CPU animation cost by up to three orders of magnitude while preserving most of the rendering quality. Excellent results of our method confirm the shared linear structure of learned avatar representations and enable highly efficient and accurate animation at frame rates reaching up to 60fps on mobile devices. Project page: https://ramazan793.github.io/gala/",
      "published": "2026-10-01T17:59:58Z",
      "abstract_url": "http://arxiv.org/abs/2610.02207v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02207v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "SILSA: Sliding-Window Slice Latents for Topology-Preserving High-Resolution 3D Generation",
      "authors": [
        "Tianjiao Yu",
        "Xinzhuo Li",
        "Yifan Shen",
        "Ying Shen",
        "Kiet A. Nguyen",
        "Adheesh Sunil Juvekar",
        "Ismini Lourentzou"
      ],
      "abstract": "High-resolution 3D generation increasingly relies on voxel latents and multi-stage pipelines that first predict active structure and then synthesize local geometry. While effective, this design fragments continuous surfaces into many local tokens, inflates generation cost, and often weakens topological consistency for thin or highly connected shapes. We introduce SILSA, a topology-aware 3D generation framework that represents shapes with compact sliding-window slice latents. Instead of generating expensive voxel tokens, SILSA uses a fixed set of overlapping slices along the three canonical axes, where each token summarizes a local depth window to preserve cross-sectional continuity and support single-stage rectified-flow generation. A Slice VAE encodes oriented surface samples into multi-axis slice latents and reconstructs them with a sparse volumetric decoder, while a Volumetric Anchor Lattice coordinates directional slice streams through a shared 3D workspace. To preserve structural correctness, we introduce slice-level topology supervision that matches persistence diagrams and aligns Betti transitions across neighboring slices. Experiments show that SILSA improves structural fidelity while substantially reducing generation cost. SILSA improves PSNR by $8.7\\%$, coverage by $5.96$ absolute points, and Betti error by $9.2\\%$ over the strongest baseline, while using $70.0\\%$ fewer tokens than the next-most compact baseline and over $98\\%$ fewer tokens than sparse or hierarchical tokenizers, effectively reducing training memory by $40.4\\%$ and inference time by $58.5\\%$. Qualitative results further show improved preservation of thin structures, repeated components, and long-range connectivity.",
      "published": "2026-10-01T17:59:46Z",
      "abstract_url": "http://arxiv.org/abs/2610.02201v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02201v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "FERPO: Forward Entropy-Regularized Policy Optimization",
      "authors": [
        "Sebastian Sanokowski",
        "Alireza Sarmadi",
        "Majid Khadiv"
      ],
      "abstract": "Several state-of-the-art methods for online reinforcement learning in continuous control improve policies using action gradients of a learned critic. However, critics are typically trained to predict returns, and accurate value predictions do not necessarily yield accurate action derivatives, potentially leading to unreliable policy updates. We propose Forward Entropy-Regularized Policy Optimization (FERPO), an on-policy maximum entropy reinforcement learning algorithm that performs policy improvement using critic values without differentiating the critic with respect to actions. FERPO derives an optimal target action distribution from a policy-improvement objective regularized by entropy and Kullback-Leibler (KL) divergence. We then fit the actor to this target by minimizing a forward-KL objective, estimated using self-normalized importance sampling (SNIS) with actions drawn from the rollout policy. By limiting the target distribution's deviation from the rollout policy, the KL regularization helps keep these importance weights well behaved. In contrast to reverse-KL objectives, which can favor a subset of the target distribution's modes, the forward-KL objective encourages coverage of multiple high-value modes and thereby promotes exploration. Experiments and ablations on MuJoCo Playground and ManiSkill show competitive performance and sample-efficiency gains. Computational benchmarks also demonstrate faster actor updates than Relative Entropy Pathwise Policy Optimization (REPPO).",
      "published": "2026-10-01T17:59:41Z",
      "abstract_url": "http://arxiv.org/abs/2610.02198v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02198v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO",
        "stat.ML"
      ]
    },
    {
      "title": "Hierarchical Continuous Diffusion Language Models",
      "authors": [
        "Hui Ren",
        "Zihan Li",
        "Chang Liu",
        "Huidong Liu",
        "Alexander Schwing"
      ],
      "abstract": "Discrete diffusion language models offer a compelling alternative to autoregressive generation for tasks demanding bidirectional reasoning and global constraint satisfaction. Yet they share a structural bottleneck: when decoding in parallel, each token is sampled independently from its marginal, severing the statistical dependencies among the tokens decoded together. Continuous diffusion language models avoid this by denoising a shared continuous state, but their denoiser sees only that state, so nothing ties it to a valid token configuration until it is finally decoded. To address this, we propose Hierarchical Continuous Diffusion Language Models (HC-DLM), which couple discrete token generation with a continuous latent trajectory in a single, principled denoising process, whose training objective is derived from a variational bound on the token likelihood. In contrast to recent methods that attach continuous context to a self-contained discrete chain, HC-DLM makes the latent the only persistent generative state: tokens are read out from it at every step and feed back as a scaffold for the next latent update. On structured reasoning (Sudoku), mathematical planning (Countdown) and language modeling (LM1B), HC-DLM improves over discrete and continuous diffusion baselines at matched model size, in puzzle accuracy on Sudoku and Countdown and in generative perplexity on LM1B. Project page: https://hc-dlm.github.io/.",
      "published": "2026-10-01T17:59:39Z",
      "abstract_url": "http://arxiv.org/abs/2610.02193v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02193v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Higher-Order Molecular Grammars for Generative and Foundation Models in Chemistry",
      "authors": [
        "Yiming Huang",
        "Yujie Zeng",
        "Vijay Prakash Dwivedi",
        "Simone Foti",
        "Jianmin Wang",
        "Jure Leskovec",
        "Tolga Birdal"
      ],
      "abstract": "Molecular learning models are strongly shaped by their underlying representations. Yet standard sequential and graph formalisms struggle to explicitly encode higher-order topology, such as ring systems and recurring motifs. Existing higher-order representations can capture these structures directly, but they are often computationally demanding and difficult to decode into valid molecules. Here, we introduce Higher-order Grammar Representation (HGR), a principled, topology-aware framework that lifts molecules to combinatorial complexes and parses each complex into a compact sequence of production rules under a context-free higher-order grammar. By serialising higher-order topology into rule sequences, HGR makes these structures directly compatible with standard sequence models, avoiding the computational overhead of explicit higher-order encodings while preserving topological expressiveness. To reduce benchmark bias towards simple ring systems, we construct RingDiv, a ring-enriched benchmark containing 1.18 million molecules, including the curated RingDiv300k subset, and introduce the ring diversity index (RDI) to quantify ring-system coverage. In molecular generation, HGR-based models uniquely combine 100% validity by construction with leading distributional alignment, ranking first in FCD on all five generation benchmarks. In representation learning, HGR-FM achieves the highest mean AUC across seven MoleculeNet benchmarks under both transfer protocols, improving on the strongest baseline by 8.3 and 3.3 AUC points under probing and full fine-tuning, respectively. Collectively, these results establish HGR as an efficient higher-order representation for molecular generation and transferable representation learning.",
      "published": "2026-10-01T17:58:45Z",
      "abstract_url": "http://arxiv.org/abs/2610.02186v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02186v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SoftServe: A Scalable Quasi-Newton Method for Deep Learning",
      "authors": [
        "Joohwan Ko",
        "Tetiana Parshakova",
        "Diana Cai",
        "Robert M. Gower"
      ],
      "abstract": "Quasi-Newton (QN) methods have long been among the most effective methods for large-scale unconstrained convex optimization. Two obstacles have limited their use in deep learning: non-convexity and enormous parameter sizes. We introduce SoftServe, a family of QN methods designed to overcome these obstacles without line searches or ad hoc curvature corrections. SoftServe derives positivedefinite curvature estimates from the variational objective of Berglund et al. (2025), even in the presence of negative curvature. We develop diagonal and Kroneckerfactored variants that preserve positive definiteness by construction and scale to massive neural networks. Finally, SoftServe relies on the stable coupled Newton-Schulz iteration for the required matrix operations, replacing costly matrix decompositions with GPU-friendly matrix multiplications. SoftServe excels on problems that are severely ill-conditioned, including tasks such as recurrent networks, deep autoencoders, physics-informed neural networks, and a 136M-parameter physics-informed diffusion model, often achieving lower losses than established baselines including Adam, Muon, and SOAP.",
      "published": "2026-10-01T17:58:24Z",
      "abstract_url": "http://arxiv.org/abs/2610.02182v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02182v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "From Knowledge Access to Source Learning: Developing Source-Specific Competence",
      "authors": [
        "Lucheng Fu",
        "Kejing Xia",
        "Yiyang Wang",
        "Yiqiao Jin",
        "Jinjin He",
        "Xiyuan Yang",
        "Haoxin Liu",
        "Ye Yu",
        "Haibo Jin",
        "Yijia Xiao",
        "Wenke Lee",
        "B. Aditya Prakash",
        "Haohan Wang"
      ],
      "abstract": "Large language model (LLM) agents increasingly rely on persistent external sources to solve sequences of knowledge-intensive tasks. Existing methods improve how source content is accessed and organized, while agent-memory systems preserve reusable knowledge from prior interactions, but repeated use of the same source is still largely treated as repeated access rather than an opportunity to progressively improve understanding of that source. We study source learning: developing reusable source-specific competence over a persistent authoritative source. We represent this competence with a persistent source model that captures reusable understanding of the source, including how its knowledge is structured, interpreted, and applied. To construct and progressively refine such models, we propose SourceLearn, which combines two complementary learning mechanisms. Self-Directed Source Learning identifies what remains incompletely understood and adaptively revisits the source, while Task-Guided Source Learning uses downstream experience to reveal local representational gaps and recurring needs in how source knowledge should be organized. In both cases, learning signals determine what should be reconsidered, while persistent updates are reconstructed from the authoritative source. Across five benchmarks and three LLM backends, SourceLearn achieves the best performance in 13 of 15 settings, with gains of up to 22.6 points over Hybrid RAG and substantial overall improvements over static source representations and experience-based memory baselines.",
      "published": "2026-10-01T17:50:16Z",
      "abstract_url": "http://arxiv.org/abs/2610.02150v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02150v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Finetuning with Sampling: SFT Learns Better Than You Think",
      "authors": [
        "Aayush Karan",
        "Sitan Chen",
        "Yilun Du"
      ],
      "abstract": "Introducing new capabilities to frontier models has long been the goal of posttraining, which predominantly employs supervised finetuning (SFT) and reinforcement learning (RL) to this end. Conventional wisdom dictates that RL enables strong generalization on new tasks without losing existing capabilities, while SFT is prone to weak generalization and catastrophic forgetting. At the same time, SFT can learn from off-policy expert data, whereas RL must rely on a model's ability to find successful trajectories with repeated sampling. In our work, we seek to leverage the strength of on-policy learning while utilizing the privileged information contained in off-policy data. However, rather than modifying the learning objective to accommodate this data, we instead tailor the data distribution to better suit the learner. We introduce a Markov chain Monte Carlo (MCMC) sampling algorithm that progressively transforms off-policy traces to be more on-policy given a reference model for finetuning. Across tasks like scientific skill acquisition, mathematical reasoning, and open-ended expertise, our sampling algorithm enables SFT to rival prevailing posttraining techniques, often generalizing better and forgetting less than strong on-policy baselines. In addition, the resulting finetuned models exhibit strong distributional performance and are capable of learning beyond sharpening the base model distribution. At a higher level, our approach presents sampling as a model-native operator that shapes data for learnability, offering broader utility as a general-purpose primitive throughout the posttraining stack.",
      "published": "2026-10-01T17:45:07Z",
      "abstract_url": "http://arxiv.org/abs/2610.02140v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02140v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Local Support Learning",
      "authors": [
        "Assaf Ben-Kish",
        "Akarsh Kumar",
        "James Glass",
        "Raja Giryes"
      ],
      "abstract": "We explore catastrophic forgetting in the context of large pre-trained models. By considering forgetting as a geometric problem in the input space of each weight matrix, we uncover a natural retention objective under which updates produced by gradient-based optimizers are suboptimal. Following this observation, we propose Local Support Learning (LSL), a general-purpose framework that augments gradient-based training for retention of prior capabilities without access to prior data. During a new learning phase, LSL pairs two components with distinct roles: a standard weight adapter, trained as usual to minimize the loss, and a gating function that enables the adapter only on input activations from its own training distribution, making the update local to that distribution. The key challenge is that this gate must route data from all learning phases while training only on data from the current one. We address this with a gate based on a Gaussian Mixture Model (GMM), whose likelihood decays rapidly away from its training data, giving it a natural tendency to stay closed on data from prior phases. We show that this post-training approach can resolve forgetting in LLMs of up to 7 billion parameters, retaining both pretrained and finetuned capabilities across multiple training phases, while being efficient in memory and compute, robust to hyperparameter choice, and showing scaling potential.",
      "published": "2026-10-01T17:39:03Z",
      "abstract_url": "http://arxiv.org/abs/2610.02126v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02126v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Where-OPD: Spatially Guided On-Policy Self-Distillation of MLLMs with Synthetic Scenes",
      "authors": [
        "Sophia Sirko-Galouchenko",
        "Monika Wysoczanska",
        "Andrei Bursuc",
        "Nicolas Thome",
        "Spyros Gidaris"
      ],
      "abstract": "On-policy self-distillation has recently emerged as an effective approach for improving language-model reasoning by supervising students with a frozen or EMA version of themselves that receives privileged information. Its application to multimodal large language models (MLLMs), however, remains largely unexplored. Recent approaches use privileged visual information, such as image crops corresponding to a question, to improve fine-grained perception, but their gains are confined to tasks that benefit from such visual zooming and require either human-annotated grounding data or external teacher models. We introduce a different form of on-policy self-distillation for MLLMs that provides the teacher with textual, spatially grounded guidance identifying the visual elements relevant to a query. We use procedurally generated scenes with automatically available object identities and spatial coordinates, enabling scalable and annotation-free post-training. The teacher uses this spatial guidance to locate and integrate evidence from multiple relevant image regions, while the student learns to reproduce the resulting behavior from the image and question alone. Our approach consistently improves performance on counting, document and chart understanding benchmarks across multiple models. Importantly, although post-training uses only synthetic scenes, the resulting improvements transfer to real-world perception benchmarks, yielding a 3.23-point gain in average performance across CVBench, V*, ZoomBench, BLINK, HR-Bench, and MME-RealWorld. These results show that spatially grounded privileged information can induce broader perceptual capabilities through on-policy self-distillation, enabling substantial synthetic-to-real transfer beyond the task and data distribution used for post-training. Project page: https://github.com/sirkosophia/Where-OPD",
      "published": "2026-10-01T17:34:10Z",
      "abstract_url": "http://arxiv.org/abs/2610.02117v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02117v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Scalable, Transferable Meta-network for Data Selection Requires a Different Loss (and Why the Obvious Choice is Problematic)",
      "authors": [
        "Zilin Du",
        "Bowen Yang",
        "Boyang Albert Li"
      ],
      "abstract": "Data selection is critical for training large language models on massive and heterogeneous corpora. Meta-learning for Training-data Selection offers a principled alternative to heuristic scoring by learning data weights from a target validation objective, but existing methods face a trade-off between fine-grained valuation and transferability to unseen data. A natural solution is to replace per-sample weights with a selection network. However, we find that directly incorporating such a network into existing MTS objectives leads to unstable optimization and poor generalization, caused by weight suppression and persistent reliance on easy-to-learn features. To address these issues, we propose Transferable Example Scoring and Selection (TESS), a scalable data-selection framework built on a Pointwise Value Matching objective (PVM). Experiments on LLM safety and targeted instruction tuning demonstrate strong transfer across datasets, from subsets to full corpora, and from smaller to larger models.",
      "published": "2026-10-01T17:23:29Z",
      "abstract_url": "http://arxiv.org/abs/2610.02092v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02092v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Homomorphic Advantage Operator: Stabilizing Reinforcement Learning Under Fully Homomorphic Encryption Constraints",
      "authors": [
        "Abid Mohamed Nadhir",
        "Ahmad Al Hanbali",
        "Beggas Mounir"
      ],
      "abstract": "Privacy-preserving machine learning presents significant deployment challenges on the cloud for intelligent systems with confidential data. Fully Homomorphic Encryption (FHE) offers a compelling solution for secure computation, preserving data confidentiality of cloud computations. However, applying FHE to reinforcement learning (RL) requires replacing non-linear operations with polynomial approximations, which diverge catastrophically due to a unique recursive error phenomenon known as the Bellman drift. This article introduces the Homomorphic Advantage Operator (HAO), a stabilization framework designed to prevent polynomial approximation divergence in FHE-based deep RL. HAO adapts the zero-mean centering projection from advantage-based value estimation directly to temporal-difference (TD) targets. This linear projection annihilates the uniform state-value baseline that drives the Bellman drift, maintaining per-state action rankings while requiring zero additional non-linear multiplicative depth and avoiding expensive ciphertext bootstrapping. The proposed HAO framework was evaluated using a three-tier experimental methodology, including a tabular Markov Decision Process (MDP), an encrypted CartPole environment using real CKKS cryptographic operations, and a 20-node logistics routing benchmark with dense continuous features. The results demonstrate that the proposed HAO strictly bounds network pre-activations within the safe polynomial approximation domain. The proposed HAO RL agents achieved 0% boundary breaches across all random seeds used, whereas regularization alone (L2 weight decay and gradient clipping) breached the bound on 3 of 5 seeds and the unstabilized baseline did so in 83.8% of episodes. Finally, HAO agents improve optimal policy accuracy by 18.0 percentage points in tabular domains and remain stable when DP-SGD-style Gaussian noise is added to the clipped gradients.",
      "published": "2026-10-01T17:14:32Z",
      "abstract_url": "http://arxiv.org/abs/2610.02074v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02074v1",
      "categories": [
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Causal Memory Policy: Making Memory Utility Identifiable by Intervening on Retrieval",
      "authors": [
        "Arman Behnam",
        "Binghui Wang"
      ],
      "abstract": "Memory-augmented large language models must decide which memories to retain, and recent systems do so by estimating each memory's effect on task performance. However, these estimates rely entirely on retrieved memories. When a memory is never retrieved, store-level interventions produce identical outcomes, leaving its utility unidentified. This is a retrieval-level positivity violation, invisible to diagnostics that examine only memory operations. We introduce Causal Memory Policy (CMP), a causal framework that restores identification by intervening on retrieval itself, reserving a fixed number of context slots for memories sampled with known propensities. CMP estimates memory utility by self-normalized inverse propensity weighting under a balanced assignment design. We prove the causal factorization of memory utility through retrieval, the unbiasedness and exact variance of the estimator, and the optimal decision rule under irreversible operations. Empirically, identification fails for 54% of required memories on LongMemEval and 67% on LoCoMo, and the failure persists in a deployed memory system. CMP improves discrimination between required and non-required memories from 0.54 to 0.66 AUC. Finally, we show that identified memory utility alone is insufficient for retention decisions: per-query utility reaches 0.78 AUC on the query for which it is estimated, yet no aggregation available to a retention policy predicts a memory's value on unseen queries. Code is available at: https://anonymous.4open.science/r/cmp-release-D0C3/.",
      "published": "2026-10-01T17:11:25Z",
      "abstract_url": "http://arxiv.org/abs/2610.02070v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02070v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "External Observers May See More Clearly: Cross-Model Span-Level Hallucination Detection in Large Language Models via Hidden State Probing",
      "authors": [
        "Kingshuk Gupta",
        "Davide Buscaldi"
      ],
      "abstract": "As Large Language Models (LLMs) increasingly serve as foundational reasoning engines, their tendency to hallucinate remains a critical vulnerability. While recent internal state probes offer a promising alternative to slow external retrieval systems, they largely reduce hallucination detection to a token-wise binary classification task, failing to capture the structured, sequential boundaries of semantic drift. Here, we introduce an internal hidden state framework for fine-grained, span-level hallucination detection. By inspecting layer-wise activation patterns, we attempt to detect the exact hallucination onset and continuation tokens in an LLM generation. Our experiments show that this approach successfully isolates hallucination onsets, achieving substantial improvements in Precision-Recall AUC over random baselines despite extreme class imbalance. Ultimately, we propose a novel cross-model detection framework in which one model observes the internal representations elicited by another model's generation. We find that an external observer can match or exceed a generator's self-detection of its own hallucination onsets, including when the observer is the smaller model, suggesting that self-detection is not the ceiling for onset localisation.",
      "published": "2026-10-01T17:06:53Z",
      "abstract_url": "http://arxiv.org/abs/2610.02066v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02066v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "HydroJEV: A one-second, training-free screen for cyber-attack and fault attribution in water distribution networks",
      "authors": [
        "Tianwei Mu",
        "Shengyan Jiang",
        "Mingzhe Yuan",
        "Qing Luo",
        "Min Xiao",
        "Wenhong Wang",
        "Jun Li",
        "Manhong Huang"
      ],
      "abstract": "When a SCADA alarm is raised in a water distribution network, operators must decide quickly whether it reflects a cyberattack, a physical fault, a normal transient or a faulty sensor. Supervised classifiers need labelled incidents that utilities rarely have, and frontier large language models (LLMs) take tens of seconds per decision. We tested whether Jev, a training-free model that returns class probabilities in about one second, can serve as the first tier of this triage. On a four-class cause-attribution benchmark built on the C-Town network in EPANET, Jev was compared with a hand-written rule tree, a supervised classifier and seven cloud LLMs on identical evidence in four sealed, pre-registered rounds. With only a label-free prior correction, Jev matched the rule tree (macro-F1 0.62-0.64 against 0.56-0.61 in distribution) and exceeded the supervised classifier by 0.36-0.42 on event subtypes absent from its labels, in all four rounds, and it outperformed the classifier whenever fewer than about four labelled events per class were available. Jev also decided 20-40 times faster than frontier LLMs. Accepting only benign Jev verdicts confirmed by the rule tree spared an LLM reviewer 35-38% of windows on fresh sealed sets without loss of macro-F1. Transferred unchanged to two further networks, this gated cascade stayed within the non-inferiority margin of its reviewer on all four sets. A fast, training-free screen can therefore take over about a third of the review load in SCADA anomaly triage while preserving the accuracy of deliberate review.",
      "published": "2026-10-01T16:57:33Z",
      "abstract_url": "http://arxiv.org/abs/2610.02048v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02048v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Distributionally Robust Schrödinger Bridge",
      "authors": [
        "Jinhwan Sul",
        "Panagiotis Theodoropoulos",
        "Vincent Pacelli",
        "Jaemoo Choi",
        "Evangelos Theodorou"
      ],
      "abstract": "Schrödinger bridge (SB) learns stochastic transport between prescribed initial and target distributions. When the initial distribution shifts at test time, the learned dynamics can fail to recover the target distribution. We introduce the Distributionally Robust Schrödinger Bridge (DRSB), which learns a single controller that accounts for uncertainty in the initial distribution. The DRSB objective consists of control energy and a KL penalty between the resulting terminal distribution and the target distribution. DRSB seeks a single controller that minimizes the worst-case value of this objective as the initial distribution varies within an ambiguity set around the nominal distribution. We derive an exact variational formulation of this objective and connect its fixed-terminal-cost subproblem to stochastic optimal control and distributionally robust optimization. This formulation motivates an alternating algorithm that updates the adversarial initial distribution, estimates the terminal log-density ratio, and trains the controller. We develop Wasserstein and Sinkhorn variants using stochastic control optimality conditions to approximate the gradients required for adversarial updates. Experiments on two-dimensional transport tasks and image-to-image translation show improved robustness to input perturbations relative to standard SB, with a tradeoff in nominal performance. On Gaussian mixture transport, Sinkhorn DRSB also achieves lower mean sliced Wasserstein distance than fixed-level noise augmentation at both tested unseen noise levels.",
      "published": "2026-10-01T16:54:34Z",
      "abstract_url": "http://arxiv.org/abs/2610.02043v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02043v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CARM: Cancellation-Aware Response Masking for LLM Reinforcement Learning",
      "authors": [
        "Yafei Zhang",
        "Songshuo Lu",
        "Sicong Liao",
        "Zhi Chen",
        "Yaohua Tang"
      ],
      "abstract": "Recent years have witnessed the rapid adoption of reinforcement learning (RL) in large language model (LLM) post-training, with substantial gains in mathematical reasoning and code generation. In practical systems, however, policy updates and differences between rollout and training engines can make sampled responses off-policy. Sequence-level masking addresses this mismatch by deciding whether an entire response should contribute to optimization. A common masking rule uses the length-normalized geometric mean of sampled token probability ratios. Its signed log-ratios can cancel across positions, concealing substantial bidirectional policy drift. We propose \\emph{Cancellation-Aware Response Masking} (CARM), a sequence-level mask that takes the absolute value of each token log-ratio before averaging, preventing opposing probability changes from canceling. We prove that accepted responses satisfy a joint bound on the fraction of sampled-token ratios outside a prescribed band and their mean log-distance beyond its boundaries. Experiments on mathematical reasoning and code generation show that CARM improves mean@16 averaged over AIME 2024/2025/2026 and BeyondAIME by up to $3.13$ percentage points over geometric-mean masking, and increases average pass@1 across four code benchmarks by $2.88$ points over the strongest evaluated baseline. These findings support CARM as a theoretically grounded and effective method for response-level off-policy control in LLM reinforcement learning.",
      "published": "2026-10-01T16:51:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.02039v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02039v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Mimir: Physics-Grounded LLM Agents for Long-Horizon Irrigation Control",
      "authors": [
        "Yimeng Liu",
        "Mi Zhang",
        "Younsuk Dong",
        "Zhichao Cao"
      ],
      "abstract": "Large language model (LLM) agents increasingly combine reasoning, tool use, and action, but most evidence comes from episodic tasks with relatively immediate feedback and reset failures. Long-running physical control operates in a different regime: actions alter future states, errors compound across decisions, and an agent must improve from experience without being allowed to rewrite the physical rules that make execution safe. We study this regime through irrigation, where daily decisions interact with soil-water dynamics over entire growing seasons. We present Mimir, a physics-grounded LLM agent organized around two repair timescales. At the fast timescale, a structured physical interface and deterministic simulator turn an LLM output into a proposal that we numerically check, revise, and subject to bounded deterministic action selection before execution. At the slow timescale, recurrent failure patterns are consolidated into persistent contextual principles that condition future proposals, while the physical model, evaluator, and execution constraints remain immutable. Under a common retrospective evaluator across multiple sites, crops, and years, Mimir attains the lowest reported aggregate control cost among the evaluated references and uses about 51% less irrigation than the historical schedule replay. The ablation study show higher control cost when forward simulation, verified revision, or persistent context is removed; model-scale and model-family studies show no monotonic gain from increasing LLM size. The resulting lesson show that persistent physical agents can combine semantic reasoning with bounded, evidence-driven self-improvement while reserving physical truth and actuator authority for explicit numerical mechanisms.",
      "published": "2026-10-01T16:50:52Z",
      "abstract_url": "http://arxiv.org/abs/2610.02038v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02038v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SPHERE: Adaptive VR Indoor Scene Generation via LLM-Enhanced Spatial Preference Learning and Human-in-the-Loop RL",
      "authors": [
        "Hyeonmin Lee",
        "Zheng Wei",
        "Kyungmin Kwon",
        "Jumin Seo",
        "Jiwon Park",
        "Hayoung Oh"
      ],
      "abstract": "While Large Language Models (LLMs) advance 3D indoor scene synthesis, current pipelines fail to retain user-specific preferences across sessions, making immersive authoring a repetitive and physically fatiguing process. We present SPHERE, an adaptive VR generation framework that transforms isolated synthesis into continuous human-AI co-creation. SPHERE extracts persistent spatial preferences from natural multimodal interactions (speech and controller edits). To ensure geometric resilience against spatial distortions, it abstracts these raw edits into hierarchical constraints modeling both local functional and global topological contexts. Furthermore, a human-in-the-loop reinforcement learning mechanism dynamically updates retrieval policies based on the user's final edited scenes. A mixed-design user study ($N=42$) and an offline ablation demonstrate that SPHERE significantly reduces corrective edits and physical demand, preventing bias toward shallow object-level traits to yield geometrically resilient, profile-aligned layouts. Ultimately, SPHERE demonstrates how capturing demonstrated spatial logic enables controlled spatial adaptation, establishing a reliable, governed human-AI collaboration framework for immersive authoring. Project page and source code will be available at: https://github.com/hyeonmin11/SPHERE",
      "published": "2026-10-01T16:43:27Z",
      "abstract_url": "http://arxiv.org/abs/2610.02023v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02023v1",
      "categories": [
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "On Language Drift during RLVR Post-Training",
      "authors": [
        "Michael Sullivan",
        "Alexander Koller"
      ],
      "abstract": "Recent advances in LLM reasoning models---driven primarily by the paradigm of post-training via reinforcement learning with verifiable reward (RLVR)---have enabled them to accomplish impressively complex tasks. However, in parallel with their rising capabilities, LLMs have increasingly displayed signs of language drift in their chains of thought (CoTs): unusual, non-standard, and seemingly nonsensical language use. Although it is well-documented---and can potentially impair CoT monitorability---the causes of language drift are thus far poorly understood. In this paper, we identify the conditions under which language drift occurs: we prove theoretically that RLVR optimization pressure permits unbounded language drift, while supervised fine-tuning does not. We then show empirically that language drift specifically arises during RLVR on novel reasoning tasks---i.e. when the target behavior cannot be drawn out of the base model. Finally, we prove that it is not possible to constrain language drift without constraining expected reward, suggesting that CoT monitorability cannot be improved without harming performance during RLVR post-training at the frontier.",
      "published": "2026-10-01T16:39:48Z",
      "abstract_url": "http://arxiv.org/abs/2610.02015v1",
      "pdf_url": "https://arxiv.org/pdf/2610.02015v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
