const PAPERS_DATA = {
  "last_updated": "2026-07-07 04:05:22 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model",
      "authors": [
        "Wenhao Li",
        "Xueying Jiang",
        "Quanhao Qian",
        "Deli Zhao",
        "Shijian Lu",
        "Gongjie Zhang",
        "Ran Xu"
      ],
      "abstract": "Real-world robot deployment rarely maintains the training-stage camera setup, where cameras often experience repositioning or remounting depending on actual scenarios. Existing view-robust Vision-Language-Action (VLA) policies tolerate such camera variations only when the camera extrinsics are explicitly provided, making them fragile and hard to use especially when view robustness is critical. We argue that the policy should not be told where the camera is, but rather figure it out by itself. To this end, we introduce Camera-Centric VLA (CamVLA), a new VLA model that decouples manipulation controls from camera geometry by predicting (i) a camera-centric end-effector action expressed in the local camera frame, and (ii) a 6-DoF hand-eye matrix relating cameras to the robot base. A deterministic geometric transformation composes the two predictions into a robot base-frame action. This disentangles how I should move in pose-independent camera-centric action generation from where I am looking from in camera-perspective geometric grounding. The resulting policy is calibration-free, depth-free, and single-view, requiring only a single monocular RGB image as the visual observation and task instruction at deployment. Evaluations in both simulation and real-world robot data show that CamVLA consistently improves success rates across diverse unseen viewpoints. Project page: https://alibaba-damo-academy.github.io/CamVLA/.",
      "published": "2026-07-06T17:59:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.05396v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05396v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Weak-to-Strong Generalization via Direct On-Policy Distillation",
      "authors": [
        "Shiyuan Feng",
        "Huan-ang Gao",
        "Haohan Chi",
        "Hanlin Wu",
        "Zhilong Zhang",
        "Zheng Jiang",
        "Bingxiang He",
        "Wei-Ying Ma",
        "Ya-Qin Zhang",
        "Hao Zhou"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) is a powerful recipe for improving language-model reasoning, but it is expensive to repeat on every new strong model because the target model must generate many rollouts during training. As models scale, post-training itself becomes a bottleneck. We study a weak-to-strong alternative: run RL on a smaller model where rollouts are cheaper, then reuse what that RL run learned to improve a stronger target model. Directly distilling the post-RL weak teacher is not enough, because the teacher's final policy mixes useful RL gains with the limitations of the smaller model. We propose Direct On-Policy Distillation (Direct-OPD), which transfers the teacher's RL-induced policy shift instead. Direct-OPD compares the post-RL teacher with its own pre-RL reference and treats their log-ratio as a dense implicit reward for the student. In plain terms, the checkpoint pair tells us which actions RL made the weak model more or less likely to take, and Direct-OPD applies that signal on the stronger student's own on-policy states. This directly reuses the weak model's RL supervision signal without training an explicit reward model or running sparse-reward RL on the target model. Empirically, Direct-OPD consistently leverages weaker teachers to improve stronger target models; notably, it boosts Qwen3-1.7B from 48.3% to 62.4% on AIME 2024 in just 4 hours on 8 A100 GPUs. It outperforms step-matched direct RL and enables the sequential composition of multiple policy shifts. Our results show that RL outcomes can be reused across model scales as implicit reward signals, not merely as final models to imitate.",
      "published": "2026-07-06T17:59:58Z",
      "abstract_url": "http://arxiv.org/abs/2607.05394v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05394v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Interpretable Human-Label-Free Deep Learning for Real-Bogus Classification with Uncertainty Quantification",
      "authors": [
        "Raphaël Bonnet-Guerrini",
        "Bruno Sanchez",
        "Dominique Fouchez",
        "Benjamin Racine",
        "Maya Guy",
        "Mariam Sabalbal",
        "Manal Yassine",
        "Vincenzo Piuri"
      ],
      "abstract": "Time-domain surveys generate many transient candidates, making Real-Bogus classification a critical step in automated discovery pipelines. Reliable labels are costly, while community labels can be noisy and survey-dependent. We aim to develop a Real-Bogus classification framework that can be trained without human-labeled data using injected transients and bogus-dominated survey data, remains robust under strong class contamination, and provides calibrated uncertainty quantification. We combine simulated transient injections with a contaminated survey class and train a dual-network model using asymmetric co-teaching for classes with different label-noise levels. We evaluate performance on a benchmark subset and analyze the learned representation with latent-space visualization tools. For uncertainty quantification (UQ), we compare MC dropout and deep ensembles and propose a low-cost hybrid strategy that exploits the dual-network setting to improve calibration. We extend the evaluation to the light-curve domain to assess recovery of light-curve classes. The method achieves strong Real-Bogus performance on the labeled subset and remains stable under severe class contamination. It recovers transient light-curve classes with high fidelity, while single-source identification is limited by ambiguity in light-curve-derived labels. Our hybrid UQ approach achieves competitive calibration relative to more expensive ensemble baselines. Latent-space analyses indicate that uncertainty aligns with the decision boundary and reveal subclasses within the bogus population. Our results show that injection-driven, weakly supervised training can enable scalable and consistent Real-Bogus classification without human-labeled training data while providing calibrated uncertainties. The method is suited for transfer to forthcoming surveys by re-running the injection-based training pipeline.",
      "published": "2026-07-06T17:59:58Z",
      "abstract_url": "http://arxiv.org/abs/2607.05393v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05393v1",
      "categories": [
        "astro-ph.IM",
        "astro-ph.GA",
        "astro-ph.HE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "LLM-as-a-Verifier: A General-Purpose Verification Framework",
      "authors": [
        "Jacky Kwok",
        "Shulu Li",
        "Pranav Atreya",
        "Yuejiang Liu",
        "Yixing Jiang",
        "Chelsea Finn",
        "Marco Pavone",
        "Ion Stoica",
        "Azalia Mirhoseini"
      ],
      "abstract": "Scaling pre-training, post-training, and test-time compute have become the central paradigms for improving the capabilities of LLMs. In this work, we identify verification, the ability to determine the correctness of a solution, as a new scaling axis. To unlock this and demonstrate its effectiveness, we introduce LLM-as-a-Verifier, a general-purpose verification framework that provides fine-grained feedback for agentic tasks without requiring additional training. Unlike standard LM judges that prompt LLMs to produce discrete scores for candidate solutions, LLM-as-a-Verifier computes the expectation over the distribution of scoring token logits to generate continuous scores. This probabilistic formulation enables verification to scale along multiple dimensions: (1) score granularity, (2) repeated evaluation, and (3) criteria decomposition. In particular, we show that scaling the scoring granularity leads to better separation between positive and negative solutions, resulting in more calibrated comparisons. Moreover, scaling repeated evaluation and criteria decomposition consistently lead to additional gains in verification accuracy through variance and complexity reduction. We further introduce a cost-efficient ranking algorithm for selecting the best solution among candidates using the verifier's continuous scores. LLM-as-a-Verifier achieves state-of-the-art performance on Terminal-Bench V2 (86.5%), SWE-Bench Verified (78.2%), RoboRewardBench (87.4%), and MedAgentBench (73.3%). Beyond verification, the fine-grained signals from LLM-as-a-Verifier can also serve as a proxy for estimating task progress. We build an extension for Claude Code, enabling developers to monitor and improve their own agentic systems. Finally, we show that LLM-as-a-Verifier can provide dense feedback for RL, improving the sample efficiency of SAC and GRPO on robotics and mathematical reasoning benchmarks.",
      "published": "2026-07-06T17:59:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.05391v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05391v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA",
        "cs.RO"
      ]
    },
    {
      "title": "What Does a Discrete Diffusion Model Learn?",
      "authors": [
        "Rodrigo Casado Noguerales",
        "Bernhard Schölkopf",
        "Thomas Hofmann",
        "Aran Raoufi"
      ],
      "abstract": "What does a discrete diffusion model learn: a denoiser, a score ratio, or a bridge plug-in predictor? At the level of jump rates, these are one object in different coordinates, and reading a neural network in the wrong coordinate changes the process being trained and sampled. Starting with a rigorous derivation of the continuous-time Markov chain (CTMC) ELBO for any noising process, boundary terms included, we prove the \\emph{Oracle Distance} theorem: the negative ELBO is exactly equal to the data entropy plus the path KL from the oracle reverse process to the learned one, not merely a bound. Its unique optimizer is therefore the conditional expectation of the true reverse jump rate given the current noisy state, and its irreducible cost is the rate at which the forward process $Z_t$ destroys information about the clean data $Z_0$, $-\\tfrac{d}{dt}I(Z_0; Z_t)$, so every noising process shares the same best achievable negative ELBO: the data entropy. For sequences with token-factorizing noise, the oracle projection yields three exact coordinates for the optimizer: denoiser, cavity (bridge plug-in), and score, with closed-form conversions among them. This framework identifies which law each loss in the literature actually optimizes, recovering MDM, UDM, SEDD, and GIDD as special cases; explains why denoiser and cavity coincide for masked diffusion but not for uniform diffusion; proves that a denoiser parameterization makes the uniform ELBO diverge at initialization while the bridge plug-in stays finite; and calibrates ELBO implementations exactly at initialization. Every identity is verified numerically, without approximation, on an exactly solvable model.",
      "published": "2026-07-06T17:56:11Z",
      "abstract_url": "http://arxiv.org/abs/2607.05381v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05381v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.IT",
        "stat.ML"
      ]
    },
    {
      "title": "GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks",
      "authors": [
        "Kaiyuan Chen",
        "Shuangyu Xie",
        "Letian Fu",
        "Justin Yu",
        "William Pacini",
        "Sandeep Bajamahal",
        "Hudson Kim",
        "Jaimyn Drake",
        "Daehwa Kim",
        "Haoru Xue",
        "Jonathan Francis",
        "Christian Juette",
        "Peter Schaldenbrand",
        "Muhammet Yunus Seker",
        "Ruwan Wickramarachchi",
        "Uksang Yoo",
        "Guanzhi Wang",
        "Adithyavairavan Murali",
        "Balakumar Sundaralingam",
        "S. Shankar Sastry",
        "Spencer Huang",
        "Yuke Zhu",
        "Linxi \"Jim\" Fan",
        "Ken Goldberg"
      ],
      "abstract": "For robots to work reliably in commercial and industrial applications, can recent advances in agentic coding systems combine interpretable robot programming with the open-world adaptability of model-free policies? We focus on \"Variational Automation\" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP then generates an internal simulation environment to rehearse task instances with different graphs in parallel to iteratively refine the graph structure and parameters to improve success rates and throughput. Evaluation with 8 new open VA task benchmarks, 4 in-simulation and 4 in real-world, suggests that GaP can achieve success rates that significantly outperform baselines. Details, code, and data can be found online: https://graph-robots.github.io/gap",
      "published": "2026-07-06T17:47:31Z",
      "abstract_url": "http://arxiv.org/abs/2607.05369v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05369v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Selective Disclosure Watermarking for Large Language Models",
      "authors": [
        "Xuyang Chen",
        "Xiang Li",
        "Yangxinyu Xie",
        "Qi Long"
      ],
      "abstract": "Watermarking methods embed imperceptible and verifiable signals into text generated by large language models (LLMs). Existing approaches include zero-bit schemes for distinguishing synthetic text from human writing and multi-bit schemes for embedding metadata. However, current multi-bit watermarking methods do not allow selective disclosure: verifying any part of the watermark requires revealing the entire embedded message. This lack of control leads to unnecessary information exposure and raises privacy concerns. We propose Hierarchical Vocabulary Routing (HeRo), a watermarking framework that enables selective disclosure of embedded metadata. The method recursively partitions the vocabulary and distributes watermark information across hierarchical layers, so that different verifiers can decode only the portions of the payload corresponding to their access level. We show that the proposed scheme preserves the unbiasedness of the underlying sampling process and thus maintains text quality. Experiments demonstrate that our framework supports fine-grained access control while achieving high detection accuracy and low latency. Code is available at https://github.com/xuyangc03/hero-watermark.",
      "published": "2026-07-06T17:32:01Z",
      "abstract_url": "http://arxiv.org/abs/2607.05353v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05353v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Multiplayer Interactive World Models with Representation Autoencoders",
      "authors": [
        "Anthony Hu",
        "Václav Volhejn",
        "Adrien Ramanana Rahary",
        "Chris Mulder",
        "Aditya Makkar",
        "Amélie Royer",
        "Manu Orsini",
        "Alyx Liao",
        "Adam Jelley",
        "Eloi Alonso",
        "Florian Laurent",
        "Fredrik Norén",
        "James Swingos",
        "Jan Hünermann",
        "Kent Rollins",
        "Lucas Hosseini",
        "Matthieu Le Cauchois",
        "Maxim Peter",
        "Pim de Witte",
        "Tim Brown",
        "Vincent Micheli",
        "Moritz Böhle",
        "Gabriel de Marmiesse",
        "Viktoriia Sharmanska",
        "Lucia Specia",
        "Michael Black",
        "Patrick Pérez"
      ],
      "abstract": "We introduce the first multiplayer world model for highly dynamic environments governed by complex physical interactions. Whereas single-player world models treat the other agents as part of the environment, ours conditions on the action streams of multiple agents, learning to attribute changes in the scene to the correct player and to stay coherent under arbitrary combinations of their actions. We study this problem in the game of Rocket League, where players compete and cooperate under fast, tightly coupled dynamics. Trained on 10,000 hours of gameplay collected with publicly available bots, our 5-billion-parameter latent diffusion model generates four-player matches in real time, producing 20 frames per second on a single Nvidia B200 GPU. Although trained only on short clips, its rollouts stay stable far beyond the training horizon: distributional quality holds steady out to five minutes, the longest horizon we measure, and in practice we observe rollouts continuing for hours with no sign of collapse. We systematically investigate the central design choices: the video codec, the generative objective, and the multiplayer conditioning scheme. In addition, we characterize how behavior changes with model and data scale, including the capabilities that emerge and the failure modes that persist. We further develop targeted evaluations that probe the model's physical understanding rather than visual appearance alone. To support continued research on multiplayer world models, we release our dataset, our full training and inference codebase, and a live demo.",
      "published": "2026-07-06T17:31:52Z",
      "abstract_url": "http://arxiv.org/abs/2607.05352v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05352v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "TREK: Distill to Explore, Reinforce to Refine",
      "authors": [
        "Yuanda Xu",
        "Zhengze Zhou",
        "Kayhan Behdin",
        "Jelena Markovic-Voronov",
        "Hejian Sang",
        "Xiaomin Li",
        "Wenhui Zhu",
        "Xinchen Du",
        "Aida Rahmattalabi",
        "Ran He",
        "Sen Na",
        "Zhipeng Wang",
        "Alborz Geramifard"
      ],
      "abstract": "Group Relative Policy Optimization (GRPO) is effective when the current policy already samples useful reasoning trajectories, but it stalls on hard prompts whose correct solution modes lie outside the student's on-policy support. We propose TREK (Teacher-Routed Exploration via Forward KL), a simple staged procedure that uses distillation not for imitation but for exploration support expansion. A key advantage of TREK is its generality: because it only consumes verified output trajectories, it can use an external black-box teacher, a white-box teacher, or the same model given additional inference-time context, and it can efficiently identify which hard-prompt samples are most worth consolidating even when teacher internals are unavailable. TREK first identifies prompts where the unaided student has very low pass rate, queries a proposal source to produce verified candidate solutions, keeps the top-$r$ proposals ranked by current student likelihood, applies a short forward-KL phase to pull those verified modes into the student's support, and then returns to standard on-policy GRPO refinement. On mathematical reasoning, TREK with DeepSeek-V4 proposals improves Qwen3 models across all tested scales on AIME 2024 and AIME 2025; for Qwen3-8B, it improves AIME 2025 from 36.9 to 40.3 and AIME 2024 from 47.9 to 51.1 (avg@16), while the self-context variant reaches 38.5 and 49.6 without an external teacher. On agentic tasks, TREK raises ALFWorld success rate from 75.8 to 82.8 and ScienceWorld success rate from 12.5 to 26.7; notably, on the hardest task types, TREK achieves high success rates early in training while unaided GRPO requires substantially more optimization steps to reach comparable levels.",
      "published": "2026-07-06T17:21:16Z",
      "abstract_url": "http://arxiv.org/abs/2607.05339v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05339v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Topological Shape Representation for Aneurysm -- Bifurcation Detection",
      "authors": [
        "Akshay Gokhale",
        "Mansi Dhamne"
      ],
      "abstract": "Automated detection of intracranial aneurysms (IAs) from CT angiography (CTA) is severely hindered by high false-positive rates. Convolutional neural networks (CNNs) rely on local pixel intensities, causing systematic confusion between saccular aneurysms and vascular bifurcations -- a problem especially acute for small lesions (<3 mm), where detection sensitivity falls below 60%. We propose a plug-and-play, topology-aware false-positive reduction framework evaluating the Smooth Euler Characteristic Transform (SECT) -- a directional representation encoding global 3D vascular geometry independently of intensity -- against persistence-based summaries (Persistence Images and Landscapes), tested on a stratified subset of the RSNA 2025 dataset. SECT achieves an AUC of 0.943, substantially outperforming direction-agnostic methods (AUC ~0.68), and exhibits a clinical performance inversion: it excels on the sub-3 mm cohort, maintaining 0.943 AUC and 78.5% sensitivity at 95% specificity. The representation is also scanner-agnostic, achieving 0.927 mean AUC under leave-one-scanner-out (LOGO) validation across four manufacturers. By capturing asymmetric geometric invariants rather than intensity profiles, SECT reliably resolves the primary structural confounder in IA detection, positioning it as a robust downstream filter for hybrid deep-learning diagnostic pipelines.",
      "published": "2026-07-06T16:56:17Z",
      "abstract_url": "http://arxiv.org/abs/2607.05317v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05317v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Air Quality Downscaling with Station-Guided Pseudo-Supervision",
      "authors": [
        "Guorun Wang",
        "Simone Foti",
        "Andreas D. Demou",
        "Leonidas Kotoulas",
        "Theodoros Christoudias",
        "Alexandros Koliousis",
        "Mihalis Nicolaou",
        "Stefanos Zafeiriou"
      ],
      "abstract": "Super-resolving coarse atmospheric fields to local PM$_{2.5}$ variations is uniquely challenged by a mismatch in spatial support: while pixels represent regional averages, ground-truth observations are discrete, unaligned samples of a continuous spatial signal. To bridge this gap, we present a station-guided framework for high-resolution PM$_{2.5}$ downscaling over Europe. Taking coarse CAMS atmospheric composition fields alongside heterogeneous side information (i.e., human activity, land cover, elevation, satellite aerosol observations, and wind fields) our framework jointly super-resolves ($\\times 40$, $\\approx$ 1 km) and bias-corrects CAMS rasters, without relying on temporal sequence modelling. To address the challenge of densely supervising our multi-scale transformer network with sparse in-situ data, we introduce a time-agnostic propagation strategy that utilises spatial Gaussian blending of interpolated OpenAQ observations. Extensive qualitative and station-level evaluations across Europe demonstrate that our model recovers fine-grained spatial structures and effectively mitigates localised CAMS biases.",
      "published": "2026-07-06T16:32:25Z",
      "abstract_url": "http://arxiv.org/abs/2607.05292v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05292v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Wavelet Scattering Transform for Interpretable Schizophrenia Biomarker Discovery and Classification from Resting-State EEG",
      "authors": [
        "Md. Taksimul Ahsan Tawhid",
        "Nasif Ahmed Rafe",
        "Alif Tahmid Priyom",
        "K. M. Mustafizur Rahman"
      ],
      "abstract": "Schizophrenia is a debilitating neuropsychiatric disorder characterized by profound cortical network dysregulation, for which objective, clinically translatable EEG based biomarkers remain underdeveloped. Existing automated classification pipelines rely predominantly on static power spectral density features inherently blind to amplitude modulation dynamics and cross-frequency coupling, phenomena central to schizophrenia pathophysiology, while adopting epoch level cross validation strategies that introduce temporal data leakage, artificially inflate reported performance. This study introduces a mathematically principled diagnostic framework integrating the multi-order Wavelet Scattering Transform(WST), strict Leave One Subject Out (LOSO) cross-validation, and SHAP explainability for simultaneous EEG classification and biomarker discovery. Hierarchical WST coefficients capturing multi-scale amplitude modulation structure were extracted from resting state multichannel EEG. Subject-level ANOVA with Benjamini Hochberg false discovery rate correction identified significant biomarkers, with Random Forest and SVM classifiers evaluated under strict LOSO cross validation and subject-level majority voting. Second-order scattering coefficients encoding cross frequency coupling dominated the discriminative biomarker set, with gamma-band features most prevalent, demonstrating that temporal amplitude modulation constitutes the primary electrophysiological signature of schizophrenia. Electrode P3 was identified as the single most discriminative site. Under rigorous subject independent evaluation, the Random Forest achieved 90.48% accuracy (AUC = 0.9339; sensitivity = 95.56%). The proposed WST framework establishes a rigorous, interpretable standard for EEG-driven psychiatric biomarker discovery that can also be applicable in the detection of schizophrenia subtypes in the future.",
      "published": "2026-07-06T16:27:26Z",
      "abstract_url": "http://arxiv.org/abs/2607.05282v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05282v1",
      "categories": [
        "eess.SP",
        "cs.AI",
        "cs.LG",
        "stat.ME"
      ]
    },
    {
      "title": "Adaptive Inference Batching using Policy Gradients",
      "authors": [
        "Ruslan Sharifullin"
      ],
      "abstract": "Inference serving systems must balance throughput and latency under bursty, heterogeneous workloads, yet the industry standard remains static batching policies that require manual tuning and cannot adapt to shifting traffic. We investigate whether reinforcement learning (RL) can learn adaptive batching and routing policies that outperform these heuristics, training REINFORCE and PPO agents on a discrete-event simulator validated against queuing theory and production traces (Azure Functions, BurstGPT). We formulate the problem as an MDP over queue state, request type and GPU availability, evaluating across standard Poisson traffic, extreme bursts, real-world traces and heterogeneous multi-GPU routing. Our central finding is a clear boundary condition for RL's value in systems problems. In single-GPU settings, a well-tuned static batching policy is already near-optimal under Poisson-like arrivals and RL offers only marginal gains (+0.1% to +1.0%). In multi-GPU heterogeneous routing, however, where fast and slow requests compete for shared resources, the agent discovers a workload-segregation policy that eliminates Head-of-Line blocking, yielding a 3.5x (348%) improvement over Round-Robin and a 48% improvement over the strongest heuristic baseline (Shortest-Queue), with 60% higher throughput and 25% lower latency while respecting SLA constraints. The policy generalizes to unseen bursty and real-world traffic despite training only on synthetic Poisson arrivals and an attention-augmented policy network converges roughly 20% faster than an MLP baseline. These results suggest RL's advantage over engineered heuristics concentrates in combinatorial, multi-resource decisions rather than single-resource temporal scheduling, a practical distinction for deciding where learned policies justify their engineering cost in production inference infrastructure.",
      "published": "2026-07-06T16:20:19Z",
      "abstract_url": "http://arxiv.org/abs/2607.05272v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05272v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC",
        "cs.PF"
      ]
    },
    {
      "title": "Privacy-Preserving Robustness Verification for Neural Networks",
      "authors": [
        "Nianyun Song",
        "Xiaokun Luan",
        "Yu Guo",
        "Rongfang Bie",
        "Meng Sun",
        "Xiyue Zhang"
      ],
      "abstract": "Neural network verification and data privacy are inherently in tension: verification demands full access to model parameters and input data, yet both are increasingly restricted by privacy regulations and intellectual property constraints. This tension has left robustness verification impractical in privacy-sensitive domains. In this work, we address this gap with SecureCROWN, the first framework for privacy-preserving neural network robustness verification. Built upon secure two-party computation (2PC), our framework enables a model owner and a data owner to jointly compute certified robustness bounds -- revealing only the final result while provably protecting both parties' private data under the semi-honest security model. A key challenge is securely computing the conditional operations in Linear Bound Propagation, where the data-dependent branching is incompatible with standard secure computation protocols. We eliminate branching by formulating conditional logic as continuous arithmetic operations. Additionally, we introduce a Newton--Raphson refinement method to improve numerical stability. Extensive analysis and experiments show that SecureCROWN strictly matches plaintext verification results, while completing in 0.1--200s across varied model sizes and communication settings (LAN/WAN), demonstrating the feasibility of privacy-preserving neural network verification.",
      "published": "2026-07-06T15:59:17Z",
      "abstract_url": "http://arxiv.org/abs/2607.05251v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05251v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.LG",
        "cs.LO"
      ]
    },
    {
      "title": "CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling",
      "authors": [
        "Zuwang He",
        "Shihao Shu",
        "Yuli Qu",
        "Hanyu Gao",
        "Ziliang Zhang",
        "Diwei Chen",
        "Xiangda Yan",
        "Buyu Gao",
        "Tanchao Zhu",
        "Yumeng Li",
        "Junxiong Zhu"
      ],
      "abstract": "Personalized incentive allocation is vital for e-commerce, where uplift modeling is the standard for estimating Individual Treatment Effects (ITE). However, traditional models often fail in complex multi-seller environments with violations of the Stable Unit Treatment Value Assumption (SUTVA). We identify two critical challenges: Seller-level Cannibalization, where incentives shift expenditure between shops without growing the platform, and Incentive-level Cannibalization, where organic conversions or alternative rewards introduce significant noise into incrementality estimation. In this paper, we propose CanniUplift, a unified framework to mitigate these dual-source cannibalization effects. Specifically, we design Platform-level Global Alignment (PGA) to capture cross-shop substitution through global GMV consistency constraints. To tackle incentive-driven noise, we introduce Redemption-based Decomposition Denoising (RDD), which uses redemption behavior to decompose treated outcomes and reduce attribution noise within an entire-space framework. Furthermore, a Treat-Attention mechanism is designed to model intricate interactions between users' historical behaviors and current treatment options. Extensive experiments on both synthetic and large-scale industrial datasets demonstrate that CanniUplift significantly outperforms state-of-the-art baselines. Ablation studies confirm that the integration of PGA and RDD consistently improves wAUUC and wQINI. Successfully deployed online, our framework achieved a 4.08% relative increase in platform-wide incremental GMV (Delta GMV) over the production baseline and improved ROI in online A/B tests, proving effective in driving global platform growth.",
      "published": "2026-07-06T15:52:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.05242v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05242v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.IR"
      ]
    },
    {
      "title": "Optimizing ML Workload Partitioning between CPUs and CIM Accelerators for Heterogeneous Computing",
      "authors": [
        "Joel Klein",
        "Rebecca Pelke",
        "Roberto Laudani",
        "Jan Moritz Joseph",
        "Rainer Leupers"
      ],
      "abstract": "Computing-in-Memory (CIM) accelerators execute Matrix-Vector Multiplications (MVMs) in memory, making them a compelling solution for Machine Learning (ML) workloads. However, existing ML workload partitioning approaches for CIM accelerators do not fully account for Resistive Random Access Memory (RRAM) constraints such as limited memory, high write latency, and limited endurance. They also neglect parallelism, low-level architectural effects, or the Central Processing Unit (CPU) as a complementary compute resource. To address these limitations, we propose an Integer Linear Programming (ILP)-based workload partitioning framework for heterogeneous CPU-CIM systems. It minimizes end-to-end inference latency under RRAM constraints, captures parallelism, and combines empirical profiling with analytical models. Using our framework, heterogeneous CPU-CIM execution achieves speedups of up to 30.9x over CPU-only execution on an edge CPU and 7.3x over a high-performance CPU. A Design Space Exploration (DSE) yields further design insights for future CIM accelerators.",
      "published": "2026-07-06T15:50:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.05240v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05240v1",
      "categories": [
        "cs.ET",
        "cs.AI",
        "cs.AR",
        "cs.DC",
        "cs.LG"
      ]
    },
    {
      "title": "Noisy-Channel Minimum Bayes Risk Decoding",
      "authors": [
        "Yusuke Sakai",
        "Hidetaka Kamigaito",
        "Taro Watanabe"
      ],
      "abstract": "Minimum Bayes Risk (MBR) decoding yields more robust and higher-quality text generation than maximum a posteriori (MAP) decoding by selecting hypotheses that maximize expected utility over sampled pseudo-references. However, there exists a discrepancy in the design: hypothesis selection calculates expected utility scores conditioned on given pseudo-references, while commonly used evaluation metrics, e.g., BLEU and COMET, are asymmetric. Therefore, it is important to consider both hypothesis-to-reference and reference-to-hypothesis directional effects. In this study, we introduce a noisy channel decomposition of MBR decoding that naturally incorporates bidirectional effects to account for these asymmetries. We decompose MBR decoding into four interacting components: hypothesis-to-reference likelihood, reference-to-hypothesis likelihood, hypothesis prior, and reference prior. This decomposition provides a unified interpretation of existing MBR variants and enables metric- and task-specific interpretability by isolating the contribution of each channel. Our comprehensive analysis reveals that channel-wise contributions exhibit distinct characteristics across metrics while remaining consistent across tasks, and suggests that appropriate channel weighting may lead to improvements over original MBR decoding.",
      "published": "2026-07-06T15:14:13Z",
      "abstract_url": "http://arxiv.org/abs/2607.05198v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05198v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Unified Audio Intelligence Without Regressing on Text Intelligence",
      "authors": [
        "Zhifeng Kong",
        "Sang-gil Lee",
        "Jaehyeon Kim",
        "Boxin Wang",
        "Zihan Liu",
        "Sungwon Kim",
        "Yang Chen",
        "Arushi Goel",
        "Rajarshi Roy",
        "Wenliang Dai",
        "Zhuolin Yang",
        "Yangyi Chen",
        "Dongfu Jiang",
        "Sreyan Ghosh",
        "Tuomas Rintamaki",
        "Andrew Tao",
        "Jonathan Raiman",
        "Mohammad Shoeybi",
        "Bryan Catanzaro",
        "Wei Ping"
      ],
      "abstract": "Audio intelligence involves understanding, reasoning about, and generating both audio and speech. In this work, we introduce Nemotron-Labs-Audex-30B-A3B (Audex), a unified audio-text LLM built on Nemotron-Cascade-2-30B-A3B, a strong text-only MoE LLM. Audex adopts a simple unified design with a single Transformer decoder: audio inputs are encoded and projected into the text embedding space, while text tokens and quantized audio output tokens are treated uniformly during generation. This architecture enables strong audio-text fusion, seamless multimodal generation, and compatibility with standard LLM training and inference infrastructure. For training, we meticulously curate audio-text datasets comprising 157.4B audio tokens and 320.5B text tokens. We apply multi-stage supervised training on these datasets, followed by text-only Cascade RL and multi-domain on-policy distillation. Audex delivers state-of-the-art audio understanding, speech recognition and translation, text-to-speech, audio generation, and speech-to-speech generation, while preserving very compelling reasoning, alignment, knowledge, long-context, and agentic capabilities of its text-only LLM backbone with marginal or no regression. We release the model checkpoints to facilitate open research.",
      "published": "2026-07-06T15:11:57Z",
      "abstract_url": "http://arxiv.org/abs/2607.05196v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05196v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG",
        "cs.SD",
        "eess.AS"
      ]
    },
    {
      "title": "Rethinking On-Policy Self-Distillation for Thinking Models",
      "authors": [
        "Simran Kaur",
        "Narutatsu Ri",
        "Yinghui He",
        "Liam Fowl",
        "Sanjeev Arora"
      ],
      "abstract": "Self-distillation is a promising recipe for self-improvement in language models. In this setting, a model can serve as its own teacher when given privileged information, such as a solution to a math problem. This seems especially appealing for thinking models, which can use test-time reasoning to absorb the privileged information. Surprisingly, we show that privileged self-distillation degrades thinking models on long reasoning traces: across five Qwen3 and OLMo thinking models evaluated on AIME24, AIME25, and HMMT25, privileged-context distillation causes a relative drop of up to 17% in avg@16 accuracy. The degradation scales with the amount of privileged context withheld from the student and is most pronounced at long rollout budgets, where thinking models otherwise obtain their largest gains. This failure mode is not specific to self-distillation: on-policy distillation (OPD) improves thinking models, but privileged OPD reverses these gains. Our diagnostics link this failure mode to how privileged teacher context reshapes learning at high-entropy forking positions, where multiple continuations remain plausible and may lead to different reasoning paths. Privileged context lowers fork rates in thinking-model rollouts but not in instruction-model rollouts. This leads to an interesting dichotomy, where privileged context can help instruction-tuned models but hurts stronger thinking models. The effect is visible when the student begins a self-correction branch, where privileged OPD penalizes sampled reconsideration tokens that vanilla OPD supports. Thinking models trained with a privileged teacher produce fewer verification, backtracking, and hedging markers, even after length normalization. These findings indicate that self-distillation for strong thinking models requires attention to token-level signal, especially around correction and reasoning steps.",
      "published": "2026-07-06T15:01:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.05184v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05184v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Relational Multi-Agent Reinforcement Learning for Dynamic Pricing in High-Speed Railway Markets",
      "authors": [
        "Enrique Adrian Villarrubia-Martin",
        "David Muñoz-Valero",
        "Luis Rodriguez-Benitez",
        "Giovanni Montana",
        "Luis Jimenez-Linares"
      ],
      "abstract": "In liberalised railway systems, operators must set prices dynamically in an environment with partial observability, as they retain private information about their objectives and performance, where regulatory constraints prohibit communication or direct information exchange between competitors to prevent explicit collusion. Consequently, agents must learn to infer strategic interactions only from observable market data which presents a significant challenge for multi-agent reinforcement learning, where standard approaches typically treat observations as unstructured vectors, ignoring the underlying market topology that governs strategic interactions. To address this, an entity graph modelling approach is proposed, which represents the environment as a graph of operational units, rather than decision-making agents or static infrastructure, encoding competition, coordination, and connectivity relations between entities. Then, an extension of the multi-agent twin delayed deep deterministic policy gradient algorithm with graph-based representation learning processes the features of the entities through a multi-layer relational graph convolutional network and aggregates them via a learnt attention mechanism. Experimental results in a rail pricing reinforcement learning environment show that this novel framework achieves higher revenue and stability in two different settings of increasing market complexity compared to a representative selection of relational and non-relational baselines. The code is publicly available at: https://github.com/Kinrre/RelationalRailPricing-RL",
      "published": "2026-07-06T14:58:26Z",
      "abstract_url": "http://arxiv.org/abs/2607.05179v1",
      "pdf_url": "https://arxiv.org/pdf/2607.05179v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.MA"
      ]
    }
  ]
};
