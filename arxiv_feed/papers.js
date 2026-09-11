const PAPERS_DATA = {
  "last_updated": "2026-09-11 04:16:37 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "General Quantification of Covariate and Concept Shifts",
      "authors": [
        "Hongbo Chen",
        "Li Charlie Xia"
      ],
      "abstract": "Generalization under distribution shift remains a core challenge in modern machine learning, yet existing learning bound theory is limited to narrow, idealized settings and is non-estimable from samples. In this paper, we bridge the gap between theory and practical applications. We first show that existing definition of concept shift breaks when the source and target supports mismatch. Leveraging entropic optimal transport, we propose a key notion: $γ^{*}\\!$-concept shifts, and derive a general error bound unifying covariate and $γ^{*}\\!$-concept shifts, which applies to broad loss functions, label spaces, and stochastic labeling. We further develop estimators for these shifts with concentration guarantees, and the DataShifts algorithm, which can quantify distribution shifts and estimate the error bound in most applications - a rigorous and general tool for analyzing learning error under distribution shift.",
      "published": "2026-09-10T17:57:52Z",
      "abstract_url": "http://arxiv.org/abs/2609.11918v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11918v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Generative Marketing Mix Modeling: A Causal Inference Framework Linking GEO and GEM to Business Impact",
      "authors": [
        "Masahiro Kato",
        "Daiki Honma",
        "Taka Kato"
      ],
      "abstract": "Generative artificial intelligence changes how firms reach customers, but standard marketing data do not record how often users see and notice a firm's name in generated answers. We develop Generative Marketing Mix Modeling (GMMM) to estimate the causal effects of Generative Engine Optimization (GEO) and Generative Engine Marketing (GEM). For GEO, GMMM combines repeated generated answers with question counts, shares of use across generative systems, and notice probabilities. For GEM, it combines records of sponsored placements with notice probabilities. GMMM compares expected business responses under alternative treatment sequences and establishes sufficient conditions for identifying the resulting effects. We investigate the empirical performance of the proposed method using simulated answers to product recommendation in English and Japanese.",
      "published": "2026-09-10T17:57:28Z",
      "abstract_url": "http://arxiv.org/abs/2609.11915v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11915v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG",
        "econ.EM",
        "stat.ME"
      ]
    },
    {
      "title": "Domain-Specific Hallucination Detection in Large Language Models",
      "authors": [
        "Varun Teja Chundru",
        "Debasmita Biswas"
      ],
      "abstract": "Large language models generate fluent text that can contain unfaithful claims -- a phenomenon known as hallucination. We present a multi-signal detection pipeline combining fine-tuned DeBERTa-v3 classification, Monte Carlo (MC) Dropout uncertainty quantification, and temperature-scaled calibration for response-level hallucination detection. Evaluated on the HaluEval benchmark, our pipeline achieves F1=0.915 and AUROC=0.977 on general-domain tasks, with per-task F1 scores of 0.97 (QA), 0.96 (Summarization), and 0.82 (Dialogue). MC Dropout inference further improves accuracy to 93.2%. A context ablation study confirms the model performs genuine entailment reasoning rather than exploiting surface patterns, with summarization F1 dropping 24% when knowledge context is removed. Learning curve analysis reveals that 25% of training data captures 77% of full-data performance. Beyond detection, we apply Direct Preference Optimization (DPO) to a Qwen2.5-0.5B generator, reducing its hallucination rate from 85.5% to 37.7% (55.9% relative reduction) as measured by our detector. Cross-domain evaluation on the SciFact biomedical benchmark shows that general-domain training transfers poorly (F1=0.52), motivating domain-specific fine-tuning. PubMedBERT fine-tuned on SciFact achieves F1=0.63 and AUROC=0.81, demonstrating that domain-matched pre-training is the strongest adaptation strategy. Code and models are available at https://github.com/varunteja99/hallucination-detection-nlp",
      "published": "2026-09-10T17:45:36Z",
      "abstract_url": "http://arxiv.org/abs/2609.11878v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11878v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement",
      "authors": [
        "Yi Duan",
        "Ying Liu",
        "Zirui Tang",
        "Haodong Chen",
        "Jun Zhou",
        "Yumou Liu",
        "Bangrui Xu",
        "Yukai Wu",
        "Sidi Chen",
        "Yuhan Zhou",
        "Haoyu Wang",
        "Xiaoyou Yu",
        "Shaokun Han",
        "Xuzhou Zhu",
        "Le Zhou",
        "Bolin Lu",
        "Wei Zhou",
        "Jiachen Liu",
        "Nuozhou Fang",
        "Jiaxin Tian",
        "Ruoyu Chen",
        "Yuxuan Li",
        "Kai Zuo",
        "Kaiyan Zhang",
        "Jiantao Qiu",
        "Conghui He",
        "Guoliang Li",
        "Bowen Zhou",
        "Zhiyuan Liu",
        "Zhoufutu Wen",
        "Jihua Kang",
        "Xuanhe Zhou",
        "Fan Wu"
      ],
      "abstract": "Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.",
      "published": "2026-09-10T17:44:23Z",
      "abstract_url": "http://arxiv.org/abs/2609.11873v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11873v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "RetroThinker: Enabling Retrospective Thinking in Speech LLMs",
      "authors": [
        "Yi-Jen Shih",
        "Puyuan Peng",
        "Abdelrahman Mohamed",
        "David Harwath"
      ],
      "abstract": "Speech large language models (SpeechLLMs) offer reduced latency and retain paralinguistic nuances that are typically lost in cascaded automatic speech recognition (ASR) and text-based LM architectures. However, they continue to lag behind text-only LLMs on complex reasoning tasks, while real-time spoken interaction imposes strict latency constraints. Although prior works employ Chain-of-Thought (CoT) and concurrent reasoning to enhance reasoning capabilities without inducing prohibitive delays, an inherent accuracy-latency trade-off persists. In this paper, we investigate whether a streaming SpeechLLM can dynamically revise its reasoning traces on the fly. We introduce RetroThinker, a multi-stage post-training framework that equips the Moshi model to self-verify and forward-correct CoT steps during inference. RetroThinker combines supervised fine-tuning (SFT) on curated retrospective thinking data with length-based direct preference optimization (DPO) to optimize retrospective during early reasoning (i.e., reasoning concurrently while the user speaks). Evaluated on the GSM8K benchmark, RetroThinker significantly improves the accuracy-latency trade-off over non-retrospective baselines, achieving an 11% absolute accuracy gain at a comparable latency.",
      "published": "2026-09-10T17:41:53Z",
      "abstract_url": "http://arxiv.org/abs/2609.11864v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11864v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Explainability Assistant: A Conversational XAI Interface for Interpreting Energy Consumption Models",
      "authors": [
        "Rodion Krjutškov",
        "Eduard Barbu",
        "Nikos Sakkas",
        "Sofia Yfanti"
      ],
      "abstract": "Energy consumption forecasting relies on increasingly complex machine learning (ML) models, such as Genetic Programming-based symbolic regressors, whose predictions can be difficult for facility managers and building operators to interpret. Explainable Artificial Intelligence (XAI) techniques address this opacity, but traditional XAI dashboards require substantial technical expertise and provide limited flexibility for dynamic, context-aware inquiry. Conversational XAI systems offer a promising alternative; however, previous approaches, such as TalkToModel, were constrained by rigid custom grammars and achieved only 76.8% intent-parsing accuracy. This paper introduces the Explainability Assistant, an open-source conversational XAI system that leverages the function-calling capabilities of modern Large Language Models (LLMs) to overcome these limitations. The system achieves 94% intent-parsing accuracy, supports flexible natural language interaction, and adapts to different ML problem types without task-specific fine-tuning. We present the system's architecture and report results from a comparative evaluation conducted with energy domain specialists, contrasting the Explainability Assistant with a traditional XAI dashboard. The evaluation suggests improved usability and consistent task accuracy, with all experts unanimously preferring the conversational interface for practical use.",
      "published": "2026-09-10T17:40:11Z",
      "abstract_url": "http://arxiv.org/abs/2609.11860v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11860v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Model-Aware Schedules Improve Generation via Fiberwise Optimal Transport",
      "authors": [
        "Luyi Jia",
        "Boyan Zhang",
        "Yilun Liu",
        "Steffen Rulands"
      ],
      "abstract": "Diffusion and flow-matching schedules control the signal and noise coefficients that mix data and noise along affine probability paths. Minimizing a kinetic action defined on coefficient paths, motivated by optimal transport, helps explain strong baselines but remains model-agnostic and ignores prediction error. Here we introduce a model-aware schedule construction based on fiberwise optimal transport. At a fixed time and state on the probability path, compatible signal/noise decompositions form an affine fiber. We define a fiberwise prediction risk by averaging optimal-transport costs between the true and predictor-induced decompositions within these fibers. On a fixed coefficient curve, combining this risk with coefficient-path kinetic action yields a closed-form optimal time allocation. This construction extends to general linear prediction targets, and the risk profile can be estimated from an early baseline checkpoint. We evaluate DDPMs and flow matching across prediction targets, training configurations, risk-estimation checkpoints, datasets, and architectures. Our model-aware schedules consistently outperform strong baselines, including a 38.6% relative FID reduction for flow matching on CIFAR-10 at 16 function evaluations. Each model-agnostic kinetic baseline determines its own kinetic reference coordinate. In these coordinates, fiberwise-risk profiles from independently trained models in different settings align closely after normalization to unit area. The resulting schedule deformations used in training also align, suggesting empirical universality across the evaluated models and settings. Pretrained-checkpoint diagnostics extend this normalized-risk agreement to larger conditional latent diffusion and 2-RF models. A frozen analytic allocation template retains most of the model-aware improvement without further risk estimation or model-specific fitting.",
      "published": "2026-09-10T17:30:44Z",
      "abstract_url": "http://arxiv.org/abs/2609.11842v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11842v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Logit Refiner: Improving Visual Autoregressive Models via Intra-Scale Dependency Modeling",
      "authors": [
        "Meimingwei Li",
        "Stefan Andreas Baumann",
        "Felix Krause",
        "Björn Ommer"
      ],
      "abstract": "Visual Autoregressive Models (VAR) generate images through next-scale prediction, producing all tokens within each scale in parallel. We show that this parallel decoding constitutes a mean-field-style approximation that discards spatial dependencies among same-scale tokens, causing locally incoherent samples regardless of backbone capacity -- a limitation of the decoding rule. Addressing this limitation, we introduce the Logit Refiner, a lightweight autoregressive module that restores intra-scale dependencies by sequentially sampling tokens conditioned on frozen backbone features. Adding only ~10% parameters and less than 5% of the base model's training compute, it plugs into any pretrained VAR checkpoint without retraining. Controlled ablations isolate joint intra-scale sampling -- rather than additional capacity or training -- as the critical ingredient. Across backbones from 310M to 2B parameters on class-conditional ImageNet 256x256, the refiner consistently improves generation quality, enabling a 1.1B-parameter model to surpass one twice its size. The approach further generalizes to text-to-image generation, confirming that the mean-field bottleneck persists across VAR variants and is effectively alleviated by our method. Project page: https://compvis.github.io/logit-refiner/",
      "published": "2026-09-10T16:57:05Z",
      "abstract_url": "http://arxiv.org/abs/2609.11804v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11804v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Thinking with Looped Flows",
      "authors": [
        "Ayhan Suleymanzade",
        "Chanhyuk Lee",
        "Floor Eijkelboom",
        "Nicholas M. Boffi",
        "İsmail İlkan Ceylan",
        "Jinwoo Kim"
      ],
      "abstract": "Humans and machines often solve harder problems by spending more time on computation. In deep learning, looped models implement this idea during inference by recurrently updating a hidden state. In practice, however, their training backpropagates through only one or a few updates, making it hard to train early updates to support future ones. We propose looped flows, an approach that sidesteps this issue by training the recurrence with local denoising objectives. By imposing temporal association across denoising objectives through progressively decreasing noise levels and shared noise, the model is incentivized to learn recurrent states that transfer useful computation over time, even when gradients cover only a few updates. We then formulate inference as integrating the velocity of a probability flow parameterized by the learned denoiser, coupled with recurrent states. This allows solving harder problems by spending more computation through a finer temporal grid and enables multiple valid predictions from different initial noise samples. Across six reasoning benchmarks including two multi-solution benchmarks, looped flows outperform prior state-of-the-art looped models overall, achieving 58.8% test accuracy on ARC-AGI-1 and 12.2% on ARC-AGI-2.",
      "published": "2026-09-10T16:52:54Z",
      "abstract_url": "http://arxiv.org/abs/2609.11801v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11801v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Recognizing Is Not Reversing: A Controlled Inversion Test of Fact-Preserving News Framing",
      "authors": [
        "Yi Liu"
      ],
      "abstract": "Large language models (LLMs) are increasingly used to analyze and rewrite news, yet current framing studies mainly evaluate generation, detection, or whether rewritten text appears more neutral. They do not directly show whether a model can undo a known framing transformation while keeping the facts fixed. We introduce a controlled inversion test over three established textual realizations of framing: evaluative lexis, agency realization, and information salience. Across 60 news articles and three intervention strengths, this yields 540 paired variants with preserved atomic facts and recorded edits. Across Qwen, DeepSeek, and Kimi, factual preservation remains near 0.84, whereas intervention reversal is 0.044--0.068. Even when both framing type and direction are recognized correctly, pooled reversal reaches 0.071. These results reveal a clear separation between factual fidelity, framing recognition, and framing inversion: recognizing how an article is framed does not imply that the framing can be undone.",
      "published": "2026-09-10T16:23:51Z",
      "abstract_url": "http://arxiv.org/abs/2609.11769v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11769v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Unified Per-Token Gating Family for On-Policy Distillation: FKL/RKL Mixing with Multi-Channel and Bias Coefficients",
      "authors": [
        "Suwan Wu",
        "Yumeng Lin",
        "Pengcheng Yuan",
        "Xiaolong Jiang"
      ],
      "abstract": "Per-token gating of forward/reverse KL losses has become a standard technique for on-policy knowledge distillation (OPD), but existing methods such as EOPD (Jin et al., 2026) and ToDi (Jung et al., 2025) each fix a single gating signal and a single gating direction, and the two have never been compared directly. We introduce a four-coefficient parameterization lambda_t = sigma(a * h_t + b * u(x) + c + d * gap_t) in which direction-aligned proxies of EOPD and ToDi appear as one-dimensional (1D) restrictions, and which adds multi-channel composition and an explicit bias as further degrees of freedom. On TweetEval (Barbieri et al., 2020) emotion and hate, with a Qwen3-32B teacher and a Qwen3-4B student, configurations in the full family reach higher accuracy than the matched-magnitude single-channel (entropy-only / gap-only) 1D restrictions in 33 of 36 comparable cells, and a 26-cell mean-match isolation experiment places dynamic gating ahead of effective-KL-matched static baselines in 19 of 26 cells. Because cells share training data, models, and parameter substructure, we report both counts as exploratory aggregate directional evidence rather than as independent hypothesis tests. Targeted three-seed paired replications of the nine headline comparisons singled out by that sweep -- including a third task, offensive -- are directionally consistent, but individually smaller than the single-seed estimates and not significant at n=3. We therefore present the parameterization primarily as a shared coordinate system for comparing per-token gating designs in short-output classification OPD.",
      "published": "2026-09-10T16:22:35Z",
      "abstract_url": "http://arxiv.org/abs/2609.11768v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11768v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "SIRF: A Spec-Internalized Risk Foundation Model for Industrial Content Risk Control",
      "authors": [
        "Suwan Wu",
        "Yumeng Lin",
        "Pengcheng Yuan",
        "Xiaolong Jiang"
      ],
      "abstract": "For industrial content risk control, the real deployment constraint is not average accuracy but how much risk can be auto-handled under high precision and second-level latency. We present SIRF (Spec-Internalized Risk Foundation Model), which internalizes a platform's complex policies, synthesized without additional human annotation via EntiGraph, MAGA rewriting and account-level chain-of-thought (CoT), into the weights via continued pretraining (CPT), so rules are applied at high precision under an ultra-low-latency, verdict-only deployment. A controlled same-source comparison (Qwen3-8B-SFT vs. SIRF-8B-SFT, identical policy injection and verdict-only output form, differing only in policy-grounded CPT) attributes the gain to internalization: SIRF-8B-SFT reaches 71.3% Black Recall@P95, +15.1pp over the baseline, using only ~70M CPT tokens without harming general ability, and among included, logprob-available models under this interface it matches or exceeds far larger systems. SIRF is deployed as a tree-model adjudication layer (20% more mis-penalized samples recovered) and transfers to a freezing scenario at low cost (~70% relative mis-penalization reduction).",
      "published": "2026-09-10T16:08:54Z",
      "abstract_url": "http://arxiv.org/abs/2609.11752v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11752v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation",
      "authors": [
        "Dongfang Zhao"
      ],
      "abstract": "Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of post-training updates affects generation length: low-rank subspaces alter sequence length without modifying the alignment loss. We present LOCUS, a method that selects a task-aware low-rank adaptation subspace to minimize output-token cost subject to a utility constraint. Within this subspace, post-training retains the native preference objective with a frozen backbone. Across Anthropic HH-RLHF dialogue preferences, we evaluate two $\\sim$3B decoder backbones, Pythia-2.8B and Qwen2.5-3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. LOCUS reduces continuation length by up to 39.84\\% on Pythia-2.8B and by 14.87--17.58\\% on Qwen2.5-3B while updating only 0.24--0.28\\% of model parameters, with no material change in the internal preference diagnostic.",
      "published": "2026-09-10T15:53:25Z",
      "abstract_url": "http://arxiv.org/abs/2609.11739v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11739v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ORCH: Organizational Principles Enable Collective Intelligence in Embodied AI",
      "authors": [
        "Zhengran Ji",
        "Jonathan Hyun",
        "Boyuan Chen"
      ],
      "abstract": "Collective intelligence depends not only on the capabilities of individual members, but also on how those members are organized. Yet artificial multi-agent systems are typically assembled using fixed organizational structures, even when the physical tasks they perform impose fundamentally different coordination requirements. Here we show that principles from human organization theory can be operationalized to organize large, heterogeneous collectives of embodied artificial agents. We introduce ORCH (Organizing Roles and Coordination Hierarchies), which constructs task-specific hierarchical organizations by combining pooled interdependence for work that can proceed concurrently with sequential interdependence for work governed by prerequisite relationships. Across 25 wildfire-response missions spanning reconnaissance, rescue, transportation, resource management, containment and suppression, we evaluated teams of up to 50 heterogeneous agents using eight large language models. Organizations constructed using these principles consistently outperformed four representative embodied multi-agent approaches across mission outcome, execution efficiency, exploration and computational resource use. Human-designed ORCH organizations improved final score by 63.97% and execution efficiency by 74.29% on average relative to the four prior frameworks. Organizations generated automatically by language models improved these measures by 43.63% and 52.53%, respectively. These advantages persisted across missions and underlying language models. Notably, collective performance was not monotonically determined by model scale. Analysis of long-horizon missions showed that hierarchical organization enabled teams to preserve concurrent activity within specialized groups while coordinating ordered transitions between mission phases.",
      "published": "2026-09-10T15:52:35Z",
      "abstract_url": "http://arxiv.org/abs/2609.11737v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11737v1",
      "categories": [
        "cs.MA",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Language-Augmented Semantic Priors for B-Spline Surface Fitting",
      "authors": [
        "Yunzhong Lou",
        "Yusheng Luo",
        "Jiahao Li",
        "Yu Song",
        "Xiangdong Zhou"
      ],
      "abstract": "The use of B-splines and Non-Uniform Rational B-Splines surfaces constitutes the mathematical foundation of contemporary computer-aided design (CAD) systems. Despite long-term progress, geometric kernels in traditional CAD still rely heavily on predetermined heuristic initialization for surface fitting and parameterization. Meanwhile, the procedural semantics and design intent encoded in modeling histories are largely ignored during geometry generation. This disconnect creates a gap between high-level design intent and solver-executable geometric configuration, often leading to suboptimal and semantically inconsistent fitting results. To bridge this gap, we introduce LASP, a Language-Augmented Semantic Priors framework that leverages large language models (LLMs) to infer structured, solver-usable B-spline priors from procedural modeling histories. Rather than modifying the geometric kernel itself, LASP operates as a semantic reasoning layer above existing solvers. It first translates modeling histories into rich textual descriptions that capture design intent, geometric context, and functional relationships, and then uses a fine-tuned LLM to predict structured B-spline prior parameters. LASP is trained through a two-stage scheme that combines local geometric regularities with long-range contextual dependencies, producing priors that are both interpretable and semantically coherent. This approach furnishes inductive signals that direct the conventional B-spline fitting process toward solutions that more accurately encapsulate the intended design objectives and demonstrate heightened semantic coherence. Compared to traditional machine learning schemes, the experiments demonstrate that language-driven reasoning can serve as a powerful inductive bias for geometric solving, establishing a new paradigm of language-guided geometric optimization in modern CAD systems.",
      "published": "2026-09-10T15:27:31Z",
      "abstract_url": "http://arxiv.org/abs/2609.11708v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11708v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "COBRA-Skills: Contextual Bandit-Guided Evolution for Agent Skill Optimization",
      "authors": [
        "Pingchen Lu",
        "Xiangyi Wang",
        "Xiang Li",
        "Jie Mao",
        "Zikun Qu",
        "Junfeng Luo",
        "Yao Shu",
        "Bryan Kian Hsiang Low",
        "Zhongxiang Dai"
      ],
      "abstract": "Large language model (LLM) agents can benefit from reusable skills distilled from prior task experience, yet existing skill optimization methods often rely on costly execution-based evaluation and substantial task data. We introduce \\textbf{COBRA-Skills}, an efficient framework that formulates skill optimization as budgeted sequential optimization over a dynamically evolving candidate space. COBRA-Skills couples contextual-bandit-guided prioritization with evidence-grounded skill evolution, selectively allocating evaluations to promising or informative candidates while continually refining the skill population from execution feedback. Across six heterogeneous agent benchmarks and three target models, COBRA-Skills consistently achieves the strongest average performance among compared methods, while reducing optimization cost by 55--58\\% relative to SkillOpt and using only 50 unique optimization examples per benchmark. Further analyses show that COBRA-Skills remains robust to changes in the agent harness and performs effectively when the target model itself is used for skill generation and refinement.",
      "published": "2026-09-10T15:12:24Z",
      "abstract_url": "http://arxiv.org/abs/2609.11682v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11682v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Ecdysis: Efficient and Effective Training of Runtime Harnesses for LLM Agents",
      "authors": [
        "Ruiqing Yue",
        "Yu Cui",
        "Zhuoyu Sun",
        "Sicheng Pan",
        "Xianhong Xue",
        "Tingyu Li",
        "Ting Li",
        "Wenzhuo Zhu",
        "Yi Chen",
        "Yifei Liu",
        "Baohan Huang",
        "Zhe Cui",
        "Haibin Zhang",
        "Cong Zuo"
      ],
      "abstract": "Self-evolving runtime harnesses can substantially improve the capabilities of large language model (LLM) agents and provide a promising paradigm for optimizing agent execution. Existing harness evolution methods typically rely on iterative search, repeatedly evaluating and revising candidate harnesses based on execution feedback from task instances. While this paradigm enables continuous harness optimization, it incurs substantial time overhead due to repeated agent executions and code modifications, and may overfit to observed tasks and specific failure patterns, resulting in degraded generalization to unseen tasks. We identify the lack of principled failure diagnosis as a key bottleneck in harness evolution: an observed failure can reflect either model-specific deficiencies or systematic harness deficiencies, and directly optimizing against individual failures can lead to unnecessary model-specific accommodation. We therefore propose Ecdysis, an efficient and effective framework that distinguishes model-specific accommodation from harness-level repair and biases adaptation toward systematic harness deficiencies by identifying recurring cross-task failure patterns. Ecdysis adopts a batch-level cross-instance failure aggregation paradigm to jointly analyze failure evidence from multiple task instances and further introduces Failure-Driven Collaborative Refinement to diagnose failure causes and iteratively refine harness modification specifications. By combining cross-instance failure analysis with multi-role diagnosis, Ecdysis enables more effective harness evolution with lower training time. Experiments show that Ecdysis achieves up to a 1.84x speedup in harness training compared with existing harness evolution methods, while improving the reasoning accuracy of the resulting harnesses by 18.56%.",
      "published": "2026-09-10T15:09:08Z",
      "abstract_url": "http://arxiv.org/abs/2609.11677v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11677v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "ZipCodec: Ultra-Low-Frame-Rate Streaming Speech Coding",
      "authors": [
        "Luca Della Libera",
        "Cem Subakan",
        "Mirco Ravanelli"
      ],
      "abstract": "Neural audio codecs are a fundamental component of modern speech generation systems. While recent codecs achieve increasingly low bitrates, reducing frame rate remains challenging, as each token must preserve more information while maintaining reconstruction quality. We present ZipCodec, a streaming neural speech codec operating at 6.25 Hz and 0.80 kbps with a theoretical latency of 160 ms. Our approach combines large-scale WavLM distillation with a redesigned transformer-based architecture, a scalar spherical quantizer, and a latency-aware streaming decoder. Experiments show that ZipCodec substantially outperforms existing streaming codecs at comparable bitrates in both reconstruction and downstream tasks, while operating at a significantly lower frame rate. Despite its 842M parameters, ZipCodec achieves real-time single-stream inference on a consumer-grade CPU. Demo samples, code and checkpoints are available at https://lucadellalib.github.io/zipcodec-web/.",
      "published": "2026-09-10T14:49:54Z",
      "abstract_url": "http://arxiv.org/abs/2609.11642v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11642v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "LoaDiff: Conditional Generation of Electricity Consumption Time Series for Energy Analytics",
      "authors": [
        "Mariia Baranova",
        "Adrien Petralia",
        "Etienne Le Naour",
        "Nathan Etourneau",
        "Guillaume Hofmann",
        "Themis Palpanas"
      ],
      "abstract": "The energy transition is reshaping residential electricity consumption through the increasing adoption of distributed generation, electrified appliances, and demand-response programs. Understanding these evolving behaviors requires access to granular smart-meter data for applications such as load forecasting, appliance detection, and demand-side flexibility analysis. However, such data are subject to strict access restrictions and data-protection regulations. Thus, realistic synthetic alternatives are necessary. In this paper, we introduce LoaDiff, a diffusion-based generative model for year-long, sub-hourly smart-meter load curves. LoaDiff supports flexible conditioning on static household attributes, such as appliance ownership, and dynamic contextual variables, including calendar information and outdoor temperature. We evaluate the model against multiple generative baselines on three residential electricity-consumption datasets. Our experiments assess four complementary dimensions: fidelity and diversity, training-record memorization risk, downstream utility for load forecasting and appliance detection, and conditional controllability under alternative temperature conditions. The results show that LoaDiff generates realistic and diverse load profiles, achieves a favorable trade-off between generation quality and limited evidence of memorization, preserves information useful for downstream energy applications, and responds coherently to changes in conditioning variables.",
      "published": "2026-09-10T14:46:30Z",
      "abstract_url": "http://arxiv.org/abs/2609.11639v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11639v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SP"
      ]
    },
    {
      "title": "Distributed Optimization of Modular Production Systems using Model-based Reinforcement Learning with Inverse Models",
      "authors": [
        "Andreas Schwung",
        "Steve Yuwono",
        "Sofiene Lassoued",
        "Dorothea Schwung"
      ],
      "abstract": "This paper presents a novel approach for data-driven self-learning control of highly flexible, modular manufacturing systems. Specifically, we employ a novel framework for model-based reinforcement learning which introduces approximate inverse process models within the training of reinforcement policies. This approach disentangles the learning of actuation dynamics and the dynamics in state space, resulting in RL-based training solely within the task space. We propose a lightweight feedforward architecture for approximate inverse models and integrate them within the policy network of standard RL algorithms. We apply the approach to a laboratory modular production testbed with heterogeneous production modules. The results underline the efficiency improvements for modular manufacturing units in terms of both performance and training speed, particularly for off-policy algorithms.",
      "published": "2026-09-10T14:30:52Z",
      "abstract_url": "http://arxiv.org/abs/2609.11615v1",
      "pdf_url": "https://arxiv.org/pdf/2609.11615v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    }
  ]
};
