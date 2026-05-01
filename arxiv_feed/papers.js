const PAPERS_DATA = {
  "last_updated": "2026-05-01 04:03:01 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Computing Equilibrium beyond Unilateral Deviation",
      "authors": [
        "Mingyang Liu",
        "Gabriele Farina",
        "Asuman Ozdaglar"
      ],
      "abstract": "Most familiar equilibrium concepts, such as Nash and correlated equilibrium, guarantee only that no single player can improve their utility by deviating unilaterally. They offer no guarantees against profitable coordinated deviations by coalitions. Although the literature proposes solution concepts that provide stability against multilateral deviations (\\emph{e.g.}, strong Nash and coalition-proof equilibrium), these generally fail to exist. In this paper, we study an alternative solution concept that minimizes coalitional deviation incentives, rather than requiring them to vanish, and is therefore guaranteed to exist. Specifically, we focus on minimizing the average gain of a deviating coalition, and extend the framework to weighted-average and maximum-within-coalition gains. In contrast, the minimum-gain analogue is shown to be computationally intractable. For the average-gain and maximum-gain objectives, we prove a lower bound on the complexity of computing such an equilibrium and present an algorithm that matches this bound. Finally, we use our framework to solve the \\emph{Exploitability Welfare Frontier} (EWF), the maximum attainable social welfare subject to a given exploitability (the maximum gain over all unilateral deviations).",
      "published": "2026-04-30T17:59:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.28186v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28186v1",
      "categories": [
        "cs.GT",
        "cs.AI",
        "cs.CC",
        "cs.LG",
        "econ.TH"
      ]
    },
    {
      "title": "Synthetic Computers at Scale for Long-Horizon Productivity Simulation",
      "authors": [
        "Tao Ge",
        "Baolin Peng",
        "Hao Cheng",
        "Jianfeng Gao"
      ],
      "abstract": "Realistic long-horizon productivity work is strongly conditioned on user-specific computer environments, where much of the work context is stored and organized through directory structures and content-rich artifacts. To scale synthetic data creation for such productivity scenarios, we introduce Synthetic Computers at Scale, a scalable methodology for creating such environments with realistic folder hierarchies and content-rich artifacts (e.g., documents, spreadsheets, and presentations). Conditioned on each synthetic computer, we run long-horizon simulations: one agent creates productivity objectives that are specific to the computer's user and require multiple professional deliverables and about a month of human work; another agent then acts as that user and keeps working across the computer -- for example, navigating the filesystem for grounding, coordinating with simulated collaborators, and producing professional artifacts -- until these objectives are completed. In preliminary experiments, we create 1,000 synthetic computers and run long-horizon simulations on them; each run requires over 8 hours of agent runtime and spans more than 2,000 turns on average. These simulations produce rich experiential learning signals, whose effectiveness is validated by significant improvements in agent performance on both in-domain and out-of-domain productivity evaluations. Given that personas are abundant at billion scale, this methodology can in principle scale to millions or even billions of synthetic user worlds with sufficient compute, enabling broader coverage of diverse professions, roles, contexts, environments, and productivity needs. We argue that scalable synthetic computer creation, together with at-scale simulations, is highly promising as a foundational substrate for agent self-improvement and agentic reinforcement learning in long-horizon productivity scenarios.",
      "published": "2026-04-30T17:58:02Z",
      "abstract_url": "http://arxiv.org/abs/2604.28181v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28181v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "LLM as Clinical Graph Structure Refiner: Enhancing Representation Learning in EEG Seizure Diagnosis",
      "authors": [
        "Lincan Li",
        "Zheng Chen",
        "Yushun Dong"
      ],
      "abstract": "Electroencephalogram (EEG) signals are vital for automated seizure detection, but their inherent noise makes robust representation learning challenging. Existing graph construction methods, whether correlation-based or learning-based, often generate redundant or irrelevant edges due to the noisy nature of EEG data. This significantly impairs the quality of graph representation and limits downstream task performance. Motivated by the remarkable reasoning and contextual understanding capabilities of large language models (LLMs), we explore the idea of using LLMs as graph edge refiners. Specifically, we propose a two-stage framework: we first verify that LLM-based edge refinement can effectively identify and remove redundant connections, leading to significant improvements in seizure detection accuracy and more meaningful graph structures. Building on this insight, we further develop a robust solution where the initial graph is constructed using a Transformer-based edge predictor and multilayer perceptron, assigning probability scores to potential edges and applying a threshold to determine their existence. The LLM then acts as an edge set refiner, making informed decisions based on both textual and statistical features of node pairs to validate the remaining connections. Extensive experiments on TUSZ dataset demonstrate that our LLM-refined graph learning framework not only enhances task performance but also yields cleaner and more interpretable graph representations.",
      "published": "2026-04-30T17:57:12Z",
      "abstract_url": "http://arxiv.org/abs/2604.28178v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28178v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "PhyCo: Learning Controllable Physical Priors for Generative Motion",
      "authors": [
        "Sriram Narayanan",
        "Ziyu Jiang",
        "Srinivasa Narasimhan",
        "Manmohan Chandraker"
      ],
      "abstract": "Modern video diffusion models excel at appearance synthesis but still struggle with physical consistency: objects drift, collisions lack realistic rebound, and material responses seldom match their underlying properties. We present PhyCo, a framework that introduces continuous, interpretable, and physically grounded control into video generation. Our approach integrates three key components: (i) a large-scale dataset of over 100K photorealistic simulation videos where friction, restitution, deformation, and force are systematically varied across diverse scenarios; (ii) physics-supervised fine-tuning of a pretrained diffusion model using a ControlNet conditioned on pixel-aligned physical property maps; and (iii) VLM-guided reward optimization, where a fine-tuned vision-language model evaluates generated videos with targeted physics queries and provides differentiable feedback. This combination enables a generative model to produce physically consistent and controllable outputs through variations in physical attributes-without any simulator or geometry reconstruction at inference. On the Physics-IQ benchmark, PhyCo significantly improves physical realism over strong baselines, and human studies confirm clearer and more faithful control over physical attributes. Our results demonstrate a scalable path toward physically consistent, controllable generative video models that generalize beyond synthetic training environments.",
      "published": "2026-04-30T17:53:03Z",
      "abstract_url": "http://arxiv.org/abs/2604.28169v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28169v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "FlexiTac: A Low-Cost, Open-Source, Scalable Tactile Sensing Solution for Robotic Systems",
      "authors": [
        "Binghao Huang",
        "Yunzhu Li"
      ],
      "abstract": "We present FlexiTac, a low-cost, open-source, and scalable piezoresistive tactile sensing solution designed for robotic end-effectors. FlexiTac is a practical \"plug-in\" module consisting of (i) thin, flexible tactile sensor pads that provide dense tactile signals and (ii) a compact multi-channel readout board that streams synchronized measurements for real-time control and large-scale data collection. FlexiTac pads adopt a sealed three-layer laminate stack (FPC-Velostat-FPC) with electrode patterns directly integrated into flexible printed circuits, substantially improving fabrication throughput and repeatability while maintaining mechanical compliance for deployment on both rigid and soft grippers. The readout electronics use widely available, low-cost components and stream tactile signals to a host computer at 100 Hz via serial communication. Across multiple configurations, including fingertip pads and larger tactile mats, FlexiTac can be mounted on diverse platforms without major mechanical redesign. We further show that FlexiTac supports modern tactile learning pipelines, including 3D visuo-tactile fusion for contact-aware decision making, cross-embodiment skill transfer, and real-to-sim-to-real fine-tuning with GPU-parallel tactile simulation. Our project page is available at https://flexitac.github.io/.",
      "published": "2026-04-30T17:43:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.28156v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28156v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Do Sparse Autoencoders Capture Concept Manifolds?",
      "authors": [
        "Usha Bhalla",
        "Thomas Fel",
        "Can Rager",
        "Sheridan Feucht",
        "Tal Haklay",
        "Daniel Wurgaft",
        "Siddharth Boppana",
        "Matthew Kowal",
        "Vasudev Shyam",
        "Jack Merullo",
        "Atticus Geiger",
        "Ekdeep Singh Lubana"
      ],
      "abstract": "Sparse autoencoders (SAEs) are widely used to extract interpretable features from neural network representations, often under the implicit assumption that concepts correspond to independent linear directions. However, a growing body of evidence suggests that many concepts are instead organized along low-dimensional manifolds encoding continuous geometric relationships. This raises three basic questions: what does it mean for an SAE to capture a manifold, when do existing SAE architectures do so, and how? We develop a theoretical framework that answers these questions and show that SAEs can capture manifolds in two fundamentally different ways: globally, by allocating a compact group of atoms whose linear span contains the entire manifold, or locally, by distributing it across features that each selectively tile a restricted region of the underlying geometry. Empirically, we find that SAEs suboptimally recover continuous structures, mixing the global subspace and local tiling solutions in a fragmented regime we call dilution. This explains why manifold structure is rarely visible at the level of individual concepts and motivates post-hoc unsupervised discovery methods that search for coherent groups of atoms rather than isolated directions. More broadly, our results suggest that future representation learning methods should treat geometric objects, not just individual directions, as the basic units of interpretability.",
      "published": "2026-04-30T17:08:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.28119v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28119v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "DEFault++: Automated Fault Detection, Categorization, and Diagnosis for Transformer Architectures",
      "authors": [
        "Sigma Jahan",
        "Saurabh Singh Rajput",
        "Tushar Sharma",
        "Mohammad Masudur Rahman"
      ],
      "abstract": "Transformer models are widely deployed in critical AI applications, yet faults in their attention mechanisms, projections, and other internal components often degrade behavior silently without raising runtime errors. Existing fault diagnosis techniques often target generic deep neural networks and cannot identify which transformer component is responsible for an observed symptom. In this article, we present DEFault++, a hierarchical learning-based diagnostic technique that operates at three level of abstraction: it detects whether a fault is present, classifies it into one of 12 transformer-specific fault categories (covering both attention-internal mechanisms and surrounding architectural components), and identifies the underlying root cause from up to 45 mechanisms. To facilitate both training and evaluation, we construct DEFault-bench, a benchmark of 3,739 labeled instances obtained through systematic mutation testing. These instances are created across seven transformer models and nine downstream tasks using DEForm, a transformer-specific mutation technique we developed for this purpose. DEFault++ measures runtime behavior at the level of individual transformer components. It organizes these measurements through a Fault Propagation Graph (FPG) derived from the transformer architecture. It then produces an interpretable diagnosis using prototype matching combined with supervised contrastive learning. On DEFault-bench, DEFault++ exceeds an AUROC of 0.96 for detection and a Macro-F1 of 0.85 for both categorization and root-cause diagnosis on encoder and decoder architectures. In a developer study with 21 practitioners, the accuracy of choosing correct repair actions increased from 57.1% without support to 83.3% when using DEFault++.",
      "published": "2026-04-30T17:07:11Z",
      "abstract_url": "http://arxiv.org/abs/2604.28118v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28118v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "What Makes a Good Terminal-Agent Benchmark Task: A Guideline for Adversarial, Difficult, and Legible Evaluation Design",
      "authors": [
        "Ivan Bercovich"
      ],
      "abstract": "Terminal-agent benchmarks have become a primary signal for measuring the coding and system-administration capabilities of large language models. As the market for evaluation environments grows, so does the pressure to ship tasks quickly, often without thorough adversarial review of the verification logic. This paper is a guideline for writing good benchmark tasks, drawn from over a year of contributing to and reviewing tasks for Terminal Bench. Most people write benchmark tasks the way they write prompts. They shouldn't. A prompt is designed to help the agent succeed; a benchmark is designed to find out if it can. We argue that good tasks are adversarial, difficult, and legible, and that a large class of common failure modes -- AI-generated instructions, over-prescriptive specifications, clerical difficulty, oracle solutions that assume hidden knowledge, tests that validate the wrong things, and reward-hackable environments -- are predictable consequences of treating task authoring as prompt authoring. We catalog these failure modes, argue that real difficulty is conceptual rather than environmental, and discuss recent empirical evidence that over 15% of tasks in popular terminal-agent benchmarks are reward-hackable. We hope this serves as a useful reference for benchmark maintainers, task contributors, and researchers using benchmark scores as evidence.",
      "published": "2026-04-30T16:37:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.28093v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28093v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Towards Neuro-symbolic Causal Rule Synthesis, Verification, and Evaluation Grounded in Legal and Safety Principles",
      "authors": [
        "Zainab Rehan",
        "Christian Medeiros Adriano",
        "Sona Ghahremani",
        "Holger Giese"
      ],
      "abstract": "Rule-based systems remain central in safety-critical domains but often struggle with scalability, brittleness, and goal misspecification. These limitations can lead to reward hacking and failures in formal verification, as AI systems tend to optimize for narrow objectives. In previous research, we developed a neuro-symbolic causal framework that integrates first-order logic abduction trees, structural causal models, and deep reinforcement learning within a MAPE-K loop to provide explainable adaptations under distribution shifts. In this paper, we extend that framework by introducing a meta-level layer designed to mitigate goal misspecification and support scalable rule maintenance. This layer consists of a Goal/Rule Synthesizer and a Rule Verification Engine, which iteratively refine a formal rule theory from high-level natural-language goals and principles provided by human experts. The synthesis pipeline employs large language models (LLMs) to: (1) decompose goals into candidate causes, (2) consolidate semantics to remove redundancies, (3) translate them into candidate first-order rules, and (4) compose necessary and sufficient causal sets. The verification pipeline then performs (1) syntax and schema validation, (2) logical consistency analysis, and (3) safety and invariant checks before integrating verified rules into the knowledge base. We evaluated our approach with a proof-of-concept implementation in two autonomous driving scenarios. Results indicate that, given human-specified goals and principles, the pipeline can successfully derive minimal necessary and sufficient rule sets and formalize them as logical constraints. These findings suggest that the pipeline supports incremental, modular, and traceable rule synthesis grounded in established legal and safety principles.",
      "published": "2026-04-30T16:34:02Z",
      "abstract_url": "http://arxiv.org/abs/2604.28087v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28087v1",
      "categories": [
        "cs.LO",
        "cs.AI"
      ]
    },
    {
      "title": "Characterizing the Consistency of the Emergent Misalignment Persona",
      "authors": [
        "Anietta Weckauff",
        "Yuchen Zhang",
        "Maksym Andriushchenko"
      ],
      "abstract": "Fine-tuning large language models (LLMs) on narrowly misaligned data generalizes to broadly misaligned behavior, a phenomenon termed emergent misalignment (EM). While prior work has found a correlation between harmful behavior and self-assessment in emergently misaligned models, it remains unclear how consistent this correspondence is across tasks and whether it varies across fine-tuning domains. We characterize the consistency of the EM persona by fine-tuning Qwen 2.5 32B Instruct on six narrowly misaligned domains (e.g., insecure code, risky financial advice, bad medical advice) and administering experiments including harmfulness evaluation, self-assessment, choosing between two descriptions of AI systems, output recognition, and score prediction. Our results reveal two distinct patterns: coherent-persona models, in which harmful behavior and self-reported misalignment are coupled, and inverted-persona models, which produce harmful outputs while identifying as aligned AI systems. These findings reveal a more fine-grained picture of the effects of emergent misalignment, calling into question the consistency of the EM persona.",
      "published": "2026-04-30T16:26:53Z",
      "abstract_url": "http://arxiv.org/abs/2604.28082v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28082v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "TopBench: A Benchmark for Implicit Prediction and Reasoning over Tabular Question Answering",
      "authors": [
        "An-Yang Ji",
        "Jun-Peng Jiang",
        "De-Chuan Zhan",
        "Han-Jia Ye"
      ],
      "abstract": "Large Language Models (LLMs) have advanced Table Question Answering, where most queries can be answered by extracting information or simple aggregation. However, a common class of real-world queries is implicitly predictive, requiring the inference of unobserved answers from historical patterns rather than mere retrieval. These queries introduce two challenges: recognizing latent intent and reliable predictive reasoning over massive tables. To assess LLMs in such Tabular questiOn answering with implicit Prediction tasks, we introduce TopBench, a benchmark consisting of 779 samples across four sub-tasks, ranging from single-point prediction to decision making, treatment effect analysis, and complex filtering, requiring models to generate outputs spanning reasoning text and structured tables. We evaluate diverse models under both text-based and agentic workflows. Experiments reveal that current models often struggle with intent recognition, defaulting to just lookups. Deeper analysis identifies that accurate intent disambiguation serves as the prerequisite for leading these predictive behaviors. Furthermore, elevating the upper bound of prediction precision requires the integration of more sophisticated modeling or reasoning capabilities.",
      "published": "2026-04-30T16:22:51Z",
      "abstract_url": "http://arxiv.org/abs/2604.28076v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28076v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "RHyVE: Competence-Aware Verification and Phase-Aware Deployment for LLM-Generated Reward Hypotheses",
      "authors": [
        "Feiyu Wu",
        "Xu Zheng",
        "Zhuocheng Wang",
        "Yi ming Dai",
        "Hui Li"
      ],
      "abstract": "Large language models (LLMs) make reward design in reinforcement learning substantially more scalable, but generated rewards are not automatically reliable training objectives. Existing work has focused primarily on generating, evolving, or selecting reward candidates, while paying less attention to when such candidates can be verified and deployed during policy optimization. We study this deployment-time problem by treating generated rewards as reward hypotheses whose utility depends on the competence of the current policy and the phase of training. We propose \\textsc{RHyVE}, a competence-aware verification and phase-aware deployment protocol that compares small sets of reward hypotheses from shared policy checkpoints using short-horizon fork verification. Our experiments show that reward rankings are unreliable at low competence but become informative after task-dependent thresholds. On a sparse manipulation task, phase-aware deployment improves peak and retained performance under a locked protocol. Updated LLM-generated reward-candidate experiments show candidate-family-dependent behavior: generated pools can exhibit phase-dependent winner changes, but no fixed warm-up schedule is universally optimal. Held-out schedule selection, conservative selector baselines, compute-matched controls, and scale controls further show that \\textsc{RHyVE} is best understood as a verification-informed deployment protocol rather than a universal scheduler. Dense and all-failure boundary experiments delimit the scope of the method. Together, these results suggest that reward generation and reward deployment should be studied as coupled problems: generated rewards must be verified and deployed under changing policy competence.",
      "published": "2026-04-30T16:01:51Z",
      "abstract_url": "http://arxiv.org/abs/2604.28056v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28056v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "PROMISE-AD: Progression-aware Multi-horizon Survival Estimation for Alzheimer's Disease Progression and Dynamic Tracking",
      "authors": [
        "Qing Lyu",
        "Jeremy Hudson",
        "Mohammad Kawas",
        "Yuming Jiang",
        "Chenyu You",
        "Christopher T Whitlow"
      ],
      "abstract": "Individualized Alzheimer's disease (AD) progression prediction requires models that use irregular visits, account for censoring, avoid diagnostic leakage, and provide calibrated horizon risks. We propose PROgression-aware MultI-horizon Survival Estimation for Alzheimer's Disease (PROMISE-AD), a leakage-safe survival framework for predicting conversion from cognitively normal (CN) to mild cognitive impairment (MCI) and from MCI to AD dementia using ADNI/TADPOLE tabular histories. PROMISE-AD converts pre-index visits into tokens with standardized measurements, missingness masks, longitudinal changes, time-normalized slopes, visit timing, and non-diagnostic categorical attributes. A temporal Transformer fuses global, attention-pooled, and latest-visit representations to estimate a progression score and latent discrete-time mixture hazards. Training combines survival likelihood, horizon-specific focal risk loss, progression ranking, hazard smoothness, and mixture-balance regularization, followed by validation-set isotonic calibration for 1-, 2-, 3-, and 5-year risks. In held-out testing across three seeds, PROMISE-AD achieved an integrated Brier score (IBS) of 0.085 $\\pm$ 0.012, C-index of 0.808 $\\pm$ 0.015, and mean time-dependent AUC of 0.840 $\\pm$ 0.081 for CN-to-MCI conversion, yielding the lowest IBS among compared methods. For MCI-to-AD conversion, PROMISE-AD achieved the highest C-index (0.894 $\\pm$ 0.018) and near-ceiling 5-year discrimination (AUROC 0.997 $\\pm$ 0.003; AUPRC 0.999 $\\pm$ 0.001), although some baselines had lower IBS. Ablations and interpretability supported longitudinal change features, fused temporal representations, mixture hazards, cognitive and functional measures, APOE4 status, and recent conversion-proximal visits. These findings suggest that progression-aware survival modeling can provide interpretable multi-horizon AD conversion risk estimates.",
      "published": "2026-04-30T16:01:30Z",
      "abstract_url": "http://arxiv.org/abs/2604.28055v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28055v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.IV"
      ]
    },
    {
      "title": "Collaborative Agent Reasoning Engineering (CARE): A Three-Party Design Methodology for Systematically Engineering AI Agents with Subject Matter Experts, Developers, and Helper Agents",
      "authors": [
        "Rahul Ramachandran",
        "Nidhi Jha",
        "Muthukumaran Ramasubramanian"
      ],
      "abstract": "We present Collaborative Agent Reasoning Engineering (CARE), a disciplined methodology for engineering Large Language Model (LLM) agents in scientific domains. Unlike ad-hoc trial-and-error approaches, CARE specifies behavior, grounding, tool orchestration, and verification through reusable artifacts and systematic, stage-gated phases. The methodology employs a three-party workflow involving Subject-Matter Experts (SMEs), developers, and LLM-based helper agents. These helper agents function as facilitation infrastructure, transforming informal domain intent into structured, reviewable specifications for human approval at defined gates. CARE addresses the \"jagged technological frontier\", characterized by uneven LLM performance, by bridging the gap between novice and expert analysts regarding domain constraints and verification practices. By generating concrete artifacts, including interaction requirements, reasoning policies, and evaluation criteria, CARE ensures agent behavior is specifiable, testable, and maintainable. Evaluation results from a scientific use case demonstrate that this stage-gated, artifact-driven methodology yields measurable improvements in development efficiency and complex-query performance.",
      "published": "2026-04-30T15:54:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.28043v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28043v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SpecVQA: A Benchmark for Spectral Understanding and Visual Question Answering in Scientific Images",
      "authors": [
        "Jialu Shen",
        "Han Lyu",
        "Suyang Zhong",
        "Hanzheng Li",
        "Haoyi Tao",
        "Nan Wang",
        "Changhong Chen",
        "Xi Fang"
      ],
      "abstract": "Spectra are a prevalent yet highly information-dense form of scientific imagery, presenting substantial challenges to multimodal large language models (MLLMs) due to their unstructured and domain-specific characteristics. Here we introduce SpecVQA, a professional scientific-image benchmark for evaluating multimodal models on scientific spectral understanding, covering 7 representative spectrum types with expert-annotated question-answer pairs. The aim comprises two aspects: spectra scientific QA evaluation and corresponding underlying task evaluation. SpecVQA contains 620 figures and 3100 QA pairs curated from peer-reviewed literature, targeting both direct information extraction and domain-specific reasoning. To effectively reduce token length while preserving essential curve characteristics, we propose a spectral data sampling and interpolation reconstruction approach. Ablation studies further confirm that the approach achieves substantial performance improvements on the proposed benchmark. We test the capability of prominent MLLMs in scientific spectral understanding on our benchmark and present a leaderboard. This work represents an essential step toward enhancing spectral understanding in multimodal large models and suggests promising directions for extending visual-language models to broader scientific research and data analysis.",
      "published": "2026-04-30T15:51:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.28039v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28039v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "MIFair: A Mutual-Information Framework for Intersectionality and Multiclass Fairness",
      "authors": [
        "Jeanne Monnier",
        "Thomas George",
        "Frédéric Guyard",
        "Christèle Tarnec",
        "Marios Kountouris"
      ],
      "abstract": "Fairness in machine learning remains challenging due to its ethical complexity, the absence of a universal definition, and the need for context-specific bias metrics. Existing methods still struggle with intersectionality, multiclass settings, and limited flexibility and generality. To address these gaps, we introduce MIFair, a unified framework for bias assessment and mitigation based on mutual information. MIFair provides a flexible metric template and an in-processing mitigation method inspired by the Prejudice Remover, defining group fairness as statistical independence between prediction-derived variables and sensitive attributes. We further strengthen its information-theoretic foundation by establishing equivalences with widely used fairness notions such as independence and separation. MIFair naturally supports intersectionality, complex subgroup structures, and multiclass classification and employs regularization-based training to reduce bias according to the selected metric. Its key advantage is its versatility: it consolidates diverse fairness requirements into a single coherent framework, enabling consistent benchmarking and simplifying practical use. Experiments on real-world tabular and image datasets show that MIFair effectively reduces bias, including previously unaddressed multi-attribute scenarios, while maintaining strong predictive performance across the evaluated settings.",
      "published": "2026-04-30T15:46:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.28030v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28030v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CY",
        "cs.IT"
      ]
    },
    {
      "title": "Reliable Answers for Recurring Questions: Boosting Text-to-SQL Accuracy with Template Constrained Decoding",
      "authors": [
        "Smit Jivani",
        "Sarvam Maheshwari",
        "Sunita Sarawagi"
      ],
      "abstract": "Large language models (LLMs) have revolutionized Text-to-SQL generation, allowing users to query structured data using natural language with growing ease. Yet, real-world deployment remains challenging, especially in complex or unseen schemas, due to inconsistent accuracy and the risk of generating invalid SQL. We introduce Template Constrained Decoding (TeCoD), a system that addresses these limitations by harnessing the recurrence of query patterns in labeled workloads. TeCoD converts historical NL-SQL pairs into reusable templates and introduces a robust template selection module that uses a fine-tuned natural language inference model to match or reject queries efficiently. Once the template is selected, TeCoD enforces it during SQL generation through grammar-constrained decoding, implemented via a novel partitioned strategy that ensures both syntactic validity and efficiency. Together, these components yield up to 36% higher execution accuracy than in-context learning (ICL) and 2.2x lower latency on matched queries.",
      "published": "2026-04-30T15:44:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.28028v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28028v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.DB",
        "cs.IR"
      ]
    },
    {
      "title": "Design Structure Matrix Modularization with Large Language Models",
      "authors": [
        "Shuo Jiang",
        "Jianxi Luo"
      ],
      "abstract": "Design Structure Matrix (DSM) modularization, the task of partitioning system elements into cohesive modules, is a fundamental combinatorial challenge in engineering design. Traditional methods treat modularization as a pure graph optimization, without access to the engineering context embedded in the system. Building on prior work on LLM-based combinatorial optimization for DSM sequencing, this paper extends the method to modularization across five cases and three backbone LLMs. Our method achieves near-reference quality within 30 iterations without requiring specialized optimization code. Counterintuitively, domain knowledge, beneficial in sequencing, consistently impairs performance on more complex DSMs. We attribute this to semantic misalignment between the LLM's functional priors and the purely structural optimization objective, and propose the semantic-alignment hypothesis as a testable condition governing knowledge effectiveness with LLMs. Ablation studies identify the most effective input representation, objective formulation, and solution pool design for practical deployment. These findings offer practical guidance for deploying LLMs in engineering design optimization.",
      "published": "2026-04-30T15:38:38Z",
      "abstract_url": "http://arxiv.org/abs/2604.28018v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28018v1",
      "categories": [
        "cs.CE",
        "cs.AI"
      ]
    },
    {
      "title": "Learning from Disagreement: Clinician Overrides as Implicit Preference Signals for Clinical AI in Value-Based Care",
      "authors": [
        "Prabhjot Singh",
        "Abhishek Gupta",
        "Chris Betz",
        "Abe Flansburg",
        "Brett Ives",
        "Sudeep Lama",
        "Jung Hoon Son"
      ],
      "abstract": "We reframe clinician overrides of clinical AI recommendations as implicit preference data - the same signal structure exploited by reinforcement learning from human feedback (RLHF), but richer: the annotator is a domain expert, the alternatives carry real consequences, and downstream outcomes are observable. We present a formal framework extending standard preference learning with three contributions: a five-category override taxonomy mapping override types to distinct model update targets; a preference formulation conditioned on patient state s, organizational context c, and clinician capability kappa, where kappa decomposes into execution capability kappa-exec and alignment capability kappa-align; and a dual learning architecture that jointly trains a reward model and a capability model via alternating optimization, preventing a failure mode we term suppression bias-the systematic suppression of correct-but-difficult recommendations when clinician capability falls below the execution threshold. We argue that chronic disease management under outcome-based payment contracts produces override data with uniquely favorable properties-longitudinal density, concentrated decision space, outcome labels, and natural capability variation-and that training environments combining longitudinal outcome measurement with aligned financial incentives are a necessary condition for learning a reward model aligned with patient trajectory rather than with encounter economics. This framework emerged from operational work to improve clinician capability in a live value-based care deployment.",
      "published": "2026-04-30T15:30:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.28010v1",
      "pdf_url": "https://arxiv.org/pdf/2604.28010v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Exploring Interaction Paradigms for LLM Agents in Scientific Visualization",
      "authors": [
        "Jackson Vonderhorst",
        "Kuangshi Ai",
        "Haichao Miao",
        "Shusen Liu",
        "Chaoli Wang"
      ],
      "abstract": "This paper examines how different types of large language model (LLM) agents perform on scientific visualization (SciVis) tasks, where users generate visualization workflows from natural-language instructions. We compare three primary interaction paradigms, including domain-specific agents with structured tool use, computer-use agents, and general-purpose coding agents, by evaluating eight representative agents across 15 benchmark tasks and measuring visualization quality, efficiency, robustness, and computational cost. We further analyze interaction modalities, including code scripts and model context protocol (MCP) or API calls for structured tool use, as well as command-line interfaces (CLI) and graphical user interfaces (GUI) for more general interaction, while additionally studying the effect of persistent memory in selected agents. The results reveal clear tradeoffs across paradigms and modalities. General-purpose coding agents achieve the highest task success rates but are computationally expensive, while domain-specific agents are more efficient and stable but less flexible. Computer-use agents perform well on individual steps but struggle with longer multi-step workflows, indicating that long-horizon planning is their primary limitation. Across both CLI- and GUI-based settings, persistent memory improves performance over repeated trials, although its benefits depend on the underlying interaction mode and the quality of feedback. These findings suggest that no single approach is sufficient, and future SciVis systems should combine structured tool use, interactive capabilities, and adaptive memory mechanisms to balance performance, robustness, and flexibility.",
      "published": "2026-04-30T15:22:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.27996v1",
      "pdf_url": "https://arxiv.org/pdf/2604.27996v1",
      "categories": [
        "cs.AI",
        "cs.GR",
        "cs.HC"
      ]
    }
  ]
};
