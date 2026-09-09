const PAPERS_DATA = {
  "last_updated": "2026-09-09 04:19:56 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Procedural Graphs: Self-Evolving Execution Structures for LLM Agents",
      "authors": [
        "Yuxing Lu",
        "Yicheng Chen",
        "Shanchan Wu",
        "Sercan Ö. Arık"
      ],
      "abstract": "Large language models are increasingly deployed as agents that plan over long horizons and act through external tools. Most agents select actions through unconstrained generation over an accumulating history, leaving implicit the procedural knowledge of what to do, in what order, and under which conditions. As trajectories lengthen, agents can lose track of their objectives, invoke tools out of order, and repeat unproductive actions. We introduce the Procedural Graph: just as a knowledge graph organizes factual knowledge into (entity, relation, entity) triplets for what-is questions, a Procedural Graph organizes procedural knowledge into (procedure, relation, procedure) triplets for what-to-do questions. At each decision step, the framework localizes the agent's active node, and a guidance model translates the surrounding subgraph into step-level situational guidance that biases the solver's next action without dictating it. The graph is self-evolving: an LLM refiner contrasts failed trajectories with successful ones and edits the graph's topology and attributes, committing edits that preserve or improve held-out validation performance while retaining rejected ones to discourage repetition. Starting from a minimal skeleton, the loop builds graphs that match or surpass hand-designed ones. It can also repair a flawed expert prior. Across multiple datasets, task types, and LLMs, the Procedural Graph delivers consistent gains over memory-based baselines, and self-evolution further improves performance without manual engineering.",
      "published": "2026-09-08T17:59:41Z",
      "abstract_url": "http://arxiv.org/abs/2609.09153v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09153v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.MA"
      ]
    },
    {
      "title": "NOAH: Learning the Full Patient Journey. A Longitudinal Multimodal Time-Aware Model for Representation and Forecasting",
      "authors": [
        "Tobias Susetzky",
        "Raphael Rehms",
        "Dmitrii Seletkov",
        "Özgün Turgut",
        "Michelle Espranita Liman",
        "Lisa Steinhelfer",
        "Rickmer Braren",
        "Daniel Rueckert"
      ],
      "abstract": "The digitization of healthcare has generated vast, longitudinal, and multimodal patient records over a lifetime, yet fully exploiting these data to represent and predict patient state trajectories remains a critical challenge. Current AI models often struggle to capture the complex, irregular temporal dynamics and inherent stochasticity of real-world multimodal patient data. Existing AI approaches for modeling longitudinal patient records are predominantly discriminative, limited to a few modalities, constrained by closed categorical vocabularies, treating time as a monotonic inductive bias, or they are limited in forecasting future patient states. We introduce NOAH, a time-aware, task-agnostic, generative transformer model representing and forecasting the full multimodal patient journey. NOAH features a novel bidirectional time integration and a variational latent space to capture the continuous evolution of patient states and the stochasticity of clinical trajectories. Built from over 559 million clinical events from 431,000 hospital visits of 299,000 patients across the MIMIC dataset family, NOAH natively processes medical images, time-series and numeric signals, categorical events, as well as structured and unstructured clinical records. NOAH is the first truly holistic generative model in its field, enabling autoregressive forecasting with optional time control, zero-shot classification, and counterfactual intervention simulation. It generates highly informative and predictive patient state representations that demonstrate strong performance in probing for clinical outcomes, 15 ICD chapters, and 29 comorbidities, as well as in time-to-event prediction. Seamlessly handling diverse modalities and complex temporal dynamics, NOAH provides a versatile, task-agnostic, scalable foundation for intelligent predictive systems in personalized clinical care and digital medicine.",
      "published": "2026-09-08T17:56:13Z",
      "abstract_url": "http://arxiv.org/abs/2609.09140v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09140v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Generalization of Amari's Bayesian Duality",
      "authors": [
        "Mohammad Emtiyaz Khan",
        "Thomas Möllenhoff"
      ],
      "abstract": "Amari's contributions to information geometry and machine learning are well known. Here, we revisit Amari's work on Bayesian duality which has not received as much attention. We connect Amari's Bayesian duality to a convex duality of Bayes' rule. Using this connection, we present a generalization of Amari's Bayesian duality and discuss its relevance for modern artificial intelligence.",
      "published": "2026-09-08T17:51:08Z",
      "abstract_url": "http://arxiv.org/abs/2609.09126v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09126v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "MeClear: Cooperative Game-Theoretic Attribution and Risk-Aware Memory Clearance for Long-Horizon LLM Agents",
      "authors": [
        "Boyu Yang",
        "Jiazheng Sun",
        "Zilong Lu",
        "Zhi Qiu",
        "Xin Peng",
        "Jun Zheng"
      ],
      "abstract": "Long horizon Large Language Model (LLM) agents rely on external memory systems to preserve user preferences and task knowledge across extended interactions. Conventional retrieval mechanisms optimize semantic compatibility rather than downstream utility, frequently introducing outdated, misleading, or conflicting evidence into the active context. We present MeClear, a task conditioned memory clearance framework that identifies memories featuring negative downstream utility through cooperative attribution and selectively suppresses them from agent execution. MeClear combines Leave One Out screening with sampled cooperative Shapley attribution to distribute utility across interacting evidence, effectively resolving redundant conflict masking where single removal evaluations fail. Utilizing attribution rankings, MeClear executes a query scoped minimal clearance strategy over a nested filtration, verifying task recovery on the cleared context without permanently altering the persistent memory bank. Comprehensive experimental evaluations across ten long dialogue memory pools demonstrate that MeClear achieves a target recall of 85.9% and an overall task recovery rate of 82.3%, representing a 25.5 percentage point improvement over Leave One Out (LOO) baselines.",
      "published": "2026-09-08T17:46:00Z",
      "abstract_url": "http://arxiv.org/abs/2609.09115v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09115v1",
      "categories": [
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?",
      "authors": [
        "Yuqiao Tan",
        "Shizhu He",
        "Jun Zhao",
        "Kang Liu"
      ],
      "abstract": "While research on recursive self-improvement (RSI) has predominantly automated model training pipelines, reliable autonomous development demands a missing pillar: post-hoc monitoring and auditing to understand what models learn and ensure safe alignment. Mechanistic interpretability tools are essential to bridge this gap, among which Sparse Autoencoders (SAEs) serve as a cornerstone by isolating interpretable features for model inspection and steering. In this paper, we introduce SAEScientist-Bench to evaluate whether AI agents can act as scientists utilizing SAE tools for autonomous mechanistic discovery. Given a target concept, an agent designs contrastive probes and navigates a Gemma Scope dictionary of 131K+ features in Gemma-2-9B-IT to discover the optimal feature, evaluated against curated expert reference features anchored on Neuronpedia across activation rank, concept selectivity on contrastive texts, and causal steering. Across 10 agent configurations and 20 tasks, frontier agents demonstrate genuine discovery capabilities and lead different evaluation dimensions, but remain well behind the expert baseline, approaching expert levels on separating target concepts from contrastive controls while lagging substantially in causal generation steering. Further analysis reveals that although agents can design contrasts to rule out spurious candidates, they frequently misinterpret experimental measurements. These results establish experimental model understanding as a measurable capability for closed-loop autonomous AI R&D. Our code is available at https://github.com/Trae1ounG/SAEScientist.",
      "published": "2026-09-08T17:45:09Z",
      "abstract_url": "http://arxiv.org/abs/2609.09113v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09113v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Measuring LLM Sycophancy under Sustained Multi-Turn Pressure",
      "authors": [
        "Leyuan Tang",
        "Kangda Wei",
        "Tianyu Jiang",
        "Ruihong Huang"
      ],
      "abstract": "Large language models (LLMs) may abandon correct positions when users push back, exhibiting a failure mode known as sycophancy. Existing evaluations typically use short, pre-specified conversations and may therefore miss failures that emerge under sustained, adaptive disagreement. We introduce SPINE, a benchmark in which an LLM proxy plays a persistent but mistaken user and adaptively challenges a target model for up to 25 turns. We evaluate four production systems and three Olmo3-7b variants on 100 false-presupposition and 100 unethical-query items. Our experimental results show that collapse rates increase with conversation length for every model, short-horizon protocols underestimate sycophancy and resistance under sustained pressure remains unreliable across current models. By analyzing models with accessible reasoning traces, we surprisingly found that the correct position often remains represented in a reasoning trace when the response concedes, suggesting that the model chooses to please a user and sycophancy is not due to lack of knowledge or ignorance. Ablations show that adaptive LLM proxy exposes more sycophantic collapse than pre-generated scripts. Among all tactics, emotional appeals is the most associated with inducing LLM sycophantic behavior. The code and data are released at https://anonymous.4open.science/r/SPINE",
      "published": "2026-09-08T17:35:04Z",
      "abstract_url": "http://arxiv.org/abs/2609.09090v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09090v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "ThinkPrior: Zero-Rollout Difficulty Priors for Cold-Start Prompt Selection in RLVR",
      "authors": [
        "Tommy Sha",
        "Skylar Zhai",
        "Siqi Zhao"
      ],
      "abstract": "In reinforcement learning with verifiable rewards (RLVR) trained with group relative policy optimization (GRPO), the KL-free reward-advantage term studied here depends on within-group reward variation. If all rollouts in a group are correct or all are wrong, their group-relative advantages are identically zero; these zero-advantage silent groups provide no reward-advantage gradient, yet uniform sampling spends 39% of a run's rollouts on them. History-based prompt selection must first spend target-policy rollouts to estimate difficulty, creating a cold start with rollout waste; ThinkPrior instead uses an external anchor in one offline pass to construct a zero-rollout difficulty prior before the first target-policy rollout. The verifier-scored anchor pass rate supplies an external-anchor initialization for a Beta posterior; ThinkPrior selects by expected learnability and then updates from training outcomes, changing neither the loss nor the optimizer. On Qwen2.5-Math-7B across sixteen seeds, ThinkPrior more than halves early silent groups and cuts wasted rollouts through step 30 by nearly a fifth, while we detect no difference in final accuracy. On this 250-prompt pool the fixed-budget result is a reallocation rather than a net saving. The measured ThinkPrior+DAPO composition reduces generated rollouts by 10.6% while both arms retain the same 3840-rollout update budget. The prior requires no target-policy rollout before the first selection, but the posterior thereafter uses target-policy outcomes.",
      "published": "2026-09-08T17:30:12Z",
      "abstract_url": "http://arxiv.org/abs/2609.09075v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09075v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Training-Free Task Vectors for LLM Behavioral Control",
      "authors": [
        "Gabriel J. Perin",
        "Lucas Boscaini",
        "André Araujo",
        "Nina S. T. Hirata"
      ],
      "abstract": "Task vectors enable post-training model editing by identifying semantically meaningful directions in weight space, typically computed as the difference between a fine-tuned model and its pretrained initialization. However, this reliance on fine-tuning makes discovering such directions costly and limits the practicality of post-training model editing. To address this limitation, we introduce Training-Free Task Vectors (TFTVs), a novel method to compute task-vector-like directions without requiring fine-tuning. Our method maps activation steering vectors to rank-one weight-space edits using only forward-pass statistics, while satisfying arithmetic properties that directly support learning via addition, forgetting via subtraction, and the composition of multiple edits. Empirically, we evaluate TFTVs on large language model behavioral control tasks and show that they consistently amplify, suppress, and compose target behaviors while preserving general knowledge and problem-solving skills. We also validate our method against other editing and steering baselines, experimentally demonstrating that TFTVs achieve stronger trait control with better or competitive utility preservation. We hope our work opens new directions for the community in post-training model editing and broader training-free model control. Code is available on the project website: tftv-llm.github.io.",
      "published": "2026-09-08T17:10:28Z",
      "abstract_url": "http://arxiv.org/abs/2609.09054v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09054v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Answer-Distribution Trajectories: A Stochastic-Dynamics View of LLM Reasoning",
      "authors": [
        "Mar Gonzàlez I Català",
        "Haitz Sáez de Ocáriz Borde",
        "Davide Murari",
        "Carola-Bibiane Schönlieb",
        "Pietro Liò",
        "George Montañez"
      ],
      "abstract": "Chain-of-thought reasoning provides a structured computation between a model's input and final answer. Yet it is often evaluated through endpoint accuracy, which ignores the path taken to reach that answer. An emerging line of work addresses this limitation using entropy profiles, which track how uncertainty evolves over the reasoning process but do not reveal which competing hypotheses account for that uncertainty. We introduce answer-distribution trajectories, a stochastic-dynamics-inspired representation that tracks the model's full predictive distribution over answers as reasoning unfolds. As a strictly finer representation than endpoint and entropy summaries, answer-distribution trajectories enable us to characterize a trace through a dynamical reasoning profile spanning exploration, revision, motion, and commitment, and to distinguish different dynamical mechanisms of reasoning success and failure. Across sixteen open-weight language models and four reasoning benchmarks, we show that traces with the same endpoint and similar entropy profiles can exhibit substantially different reasoning dynamics. We further find substantial variation in these dynamics both within and across models and tasks, with different objectives favoring different dynamical profiles. Additionally, we show that training and inference choices systematically reshape these profiles. Our results suggest that answer-distribution trajectories provide a rich framework for analysing and evaluating the dynamics of LLM reasoning.",
      "published": "2026-09-08T16:58:40Z",
      "abstract_url": "http://arxiv.org/abs/2609.09030v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09030v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.IT",
        "cs.LG"
      ]
    },
    {
      "title": "Let It Go or Learn to Self-Correct: Continuous Diffusion for Constrained Discrete Tasks",
      "authors": [
        "Mariia Drozdova",
        "Stéphane Liem Nguyen",
        "François Fleuret"
      ],
      "abstract": "Denoising Diffusion Probabilistic Models (DDPMs) generate samples by starting from noise and repeatedly denoising while keeping each update close to the current noisy state. This behavior is effective in many continuous domains, but its role is less clear for globally constrained discrete tasks, such as Sudoku, graph connectivity, Latin squares, and N-queens. In such settings, early discrete errors can be difficult to undo. As a result, standard diffusion sampling may preserve early mistakes, even when the model's clean predictions are informative. We compare standard samplers to sampling directly from the model's clean prediction. Without retraining, this single change improves Sudoku validity from 31% to 95%, with consistent gains across the other discrete tasks. We hypothesize that staying close to the current noisy state is harmful because the reverse trajectory can drift off the forward noising distribution the model was trained on. To reduce this train-test mismatch, we further introduce self-correction training, which exposes the model to its own predictions, improving robustness to errors that arise during inference. This substantially improves the performance of standard samplers. Our results suggest that continuous diffusion models can learn nontrivial global constraints, but discrete reasoning tasks require better alignment between training and inference: either through samplers that reduce commitment to early decisions, or through training that teaches the model to correct its own inference-time errors.",
      "published": "2026-09-08T16:45:52Z",
      "abstract_url": "http://arxiv.org/abs/2609.09009v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09009v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Deposon: An Auditable, Conservation-Guaranteed, Game-Theoretically Tested Scattering Layer over LLM Reasoning Paths",
      "authors": [
        "Qihao Yuan"
      ],
      "abstract": "Multi-step LLM reasoning lacks a machine-recheckable ledger: discarded reasoning paths leave no auditable record. We propose the Deposon scattering layer, which binds each node of an LLM-generated concept-decomposition graph to a two-parameter Deposon state; paths undergo three-channel scattering -- transmission, reflection, irreversible dissipation -- obeying T+R+A=1 for arbitrary parameters, with a maximum per-path energy-audit deviation of 2.2E-16 (machine epsilon). We report all three evidence tiers honestly. On synthetic trap benchmarks the path-filtering gain is closed (pre-registered): unified reaches 100% versus a decoy-capture baseline at 7%/10%. On real benchmarks the layer is indistinguishable from a trivial six-keyword rule filter (GSM8K 0.87 >= 0.85, McNemar p=0.5; StrategyQA 0.899 = 0.899); no difference is detected here, so we sharpen the claim to \"the differential value lies solely in machine verifiability.\" Fusion yields a second negative result: convex combinations with a semantic prior never improve (physics 0.484 -> 0.452), and the apparent lambda=2 gain is an anti-field artifact; any fusion gain must be nonlinear. Modeling the reverse dynamics as a potential game on the graph, we evidence an auditable scalar's monotonicity and near-gradientness and quantify the empirical coordination ratio (ECR). The three formalized dynamical-equivalence propositions (P1a/P1b/T-P1c) are falsified under the pre-registered kill protocol, and the potential-game claim is downgraded to approximate (cyclic-graph median residual 0.669): only consistency-level evidence survives at the dynamical level. Code: github.com/zeroandcat/Deposon.",
      "published": "2026-09-08T16:40:31Z",
      "abstract_url": "http://arxiv.org/abs/2609.09001v1",
      "pdf_url": "https://arxiv.org/pdf/2609.09001v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Transformers as In-Context Samplers: From Closed-Form Diffusion to Estimation-Free Sampling",
      "authors": [
        "Arman Adibi",
        "Alireza Jafari",
        "Mohammad Ghavamzadeh",
        "Hadi Daneshmand"
      ],
      "abstract": "A growing body of work establishes that large language models are not mere statistical memorizers, but are capable of in-context learning: performing inference at test time using only examples provided in the prompt, without any parameter updates. Prior theoretical work has shown that this capability extends to supervised learning tasks such as linear regression. We prove that in-context learning extends further to \\emph{data generation}: frozen transformers can simulate iterative generative samplers from in-context samples. We first show that transformers can realize closed-form and smoothed closed-form diffusion samplers. The construction identifies a concrete generative role for softmax attention: it computes responsibility weights and weighted empirical averages, while feedforward layers implement Euler updates. To empirically relate these constructions to pretrained language models, we study \\emph{semantic-topic sampling}: prompts consisting of words drawn from a common semantic category, such as animals, foods, or cities. Across transformer layers, the normalized hidden states exhibit a two-stage geometry: they move toward a uniform spherical reference in intermediate layers and then return to structured, topic-dependent representations near the output. We further measure an interacting-particle energy on these hidden-state clouds and observe the same U-shape pattern. We then prove that transformers can approximate an energy-based sampler, constructing the same U-shape energy across the layers.",
      "published": "2026-09-08T16:25:11Z",
      "abstract_url": "http://arxiv.org/abs/2609.08981v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08981v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.AP",
        "stat.CO",
        "stat.ML"
      ]
    },
    {
      "title": "Omni Interaction Agent Technical Report",
      "authors": [
        "Orantqing",
        "Shengpeng Ji",
        "Junlong Tong",
        "Jialong Zuo",
        "Dongjie Fu",
        "Di Cao",
        "Yangzhuo Li",
        "Shangda Wu",
        "Franz",
        "Evan",
        "Theron Veyra",
        "Changhao Pan",
        "Jingyu Lu",
        "Dongchao Yang",
        "Zhifei Xie",
        "Yang Tan",
        "Xiaoyu Shen",
        "Xiaoda Yang",
        "Wenfu Wang",
        "Teddysun",
        "Steveyves",
        "Zhou Zhao",
        "Bryanytian"
      ],
      "abstract": "In this work, we present Gander, an end-to-end model that unifies omni perception, realtime interaction, and agentic capabilities within a single framework. In contrast to turn-based conventional paradigms, Gander continuously receives streaming inputs across multiple modalities, including video, speech, and text, enabling natural full-duplex interaction in both everyday conversations and complex workflow-oriented agent scenarios. Users can interrupt the model at any time, while the model can also proactively provide intermediate feedback or ask follow up questions. To natively support these capabilities, Gander adopts two key architectural designs: 1) It employs a Cerebellum-Brain collaborative framework, in which the Cerebellum is responsible for realtime interaction and omni conversational capabilities, while the Brain handles complex reasoning and higher-level agentic tasks. The two components interact continuously through tool calling and the agent orchestration runtime. 2) The Cerebellum is built upon a streaming Thinker-Talker architecture, user inputs and model outputs are further flattened into an ordered token stream at the chunk level, providing a unified representation for low latency, continuous interaction. We conduct comprehensive evaluations of Gander across four dimensions: conversational ability, omni understanding, interactive capability, and agentic intelligence. Internal human evaluations demonstrate that Gander maintains the natural and expressive spoken dialogue capabilities of SOTA open source models while achieving competitive performance in omni interaction. Gander also demonstrates robustness in challenging real-world scenarios, including background noise interference, multi-party interactions, and backchannel communication. We release Gander together with its models, code, and data to facilitate further research and development in the community.",
      "published": "2026-09-08T16:22:23Z",
      "abstract_url": "http://arxiv.org/abs/2609.08977v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08977v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.LG",
        "cs.MM",
        "cs.SD"
      ]
    },
    {
      "title": "GraphFAS: A Distributed System for Automated Graph Feature Generation and Selection in Industrial Transaction Networks",
      "authors": [
        "Yice Luo",
        "Yun Zhu",
        "Xi Chen",
        "Yongchao Liu",
        "Xintan Zeng",
        "Chengying Huan",
        "Kai Zhang",
        "Jinrui Zhang",
        "Juelu Zhang",
        "Jiajun Zheng"
      ],
      "abstract": "Industrial fraud detection often relies on costly expert-crafted features that overlook graph-structured relational signals, while GNNs often do not meet the interpretability and deployment requirements of financial risk control. We propose GraphFAS (Graph Feature Automated Selection), a distributed feature selection procedure based on Boruta that bridges this gap through: (1) a non-parametric graph feature generation module that constructs explicit, interpretable structural features via multi-hop subgraph extraction and multi-scale aggregation without learned parameters; and (2) an automated distributed feature selection algorithm extending Boruta with median-based aggregation across partitions to robustly identify informative features at scale with minimal domain expertise. Compared with end-to-end GNN pipelines, GraphFAS decouples feature aggregation from model training, enabling direct integration with tabular models and direct compatibility with TreeSHAPbased explanations. Deployed in Alipay, GraphFAS delivers orderof-magnitude improvements in engineering efficiency while showing strong performance against expert-driven and graph-learning baselines on large-scale graphs.",
      "published": "2026-09-08T16:20:43Z",
      "abstract_url": "http://arxiv.org/abs/2609.08970v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08970v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC"
      ]
    },
    {
      "title": "PlannerForge: LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving",
      "authors": [
        "Yuan Gao",
        "Sebastian Müller",
        "Mattia Piccinini",
        "Marc Kaufeld",
        "Yuchen Zhang",
        "Finn Rasmus Schäfer",
        "Qunying Song",
        "Johannes Betz"
      ],
      "abstract": "Ensuring the safety of autonomous driving is a critical challenge. Scenario-based testing is a systematic process used to validate Autonomous Driving Systems (ADSs), but it remains a fragmented modular pipeline in which scenario generation, retrieval, modification, ADS execution, and results analysis are performed by separate tools with little interaction. Large Language Model (LLM) agents have shown promise across ADS sub-systems such as perception, planning, and control. However, no prior work covers the whole scenario-based testing pipeline for ADSs with a unified LLM-agent framework. We present PlannerForge, an LLM-agent framework that extends all scenario-based testing stages (from Scenario Generation to ADS Assessment) and adds two further LLM-enhanced stages: ADS Enhancement and ADS Benchmarking. We evaluate PlannerForge with 10 off-the-shelf LLMs across all tasks (Generation, Selection, Modification, Module Routing, Planner Testing, and Enhancement) under 5 prompt conditions. Best-per-task scores range from 0.88 to 1.00, and open-source 20-35B backends match commercial APIs on most tasks. Open-source models such as Qwen3.6:35B match commercial APIs on three of the five tasks. Chaining the modules end-to-end retains 83% / 78% of seed queries (commercial / open). It outperforms Scenario Factory 2.0 (Finkeldei et al., 2025) on natural-language generation (193 vs. 144 executable of 200) and realises 92-96% of requested city, road and vehicle attributes. It outperforms BM25 (Robertson and Zaragoza, 2009) at rank 1 selection (92.0% vs. 67.5%) and From-Words-to-Collisions (Gao et al., 2025) on physically valid edits (>=94% vs. 31%). At N=400, cost-tuning lifts planner success from 50.4% to 70.2% and cuts collisions from 19.0% to 8.4%, without domain-specific fine-tuning.",
      "published": "2026-09-08T16:19:06Z",
      "abstract_url": "http://arxiv.org/abs/2609.08965v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08965v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.RO"
      ]
    },
    {
      "title": "SQLMorph: Query Mutation and Fine-Grained Metrics for Text-to-SQL Evaluation",
      "authors": [
        "Mohammadhossein Malekpour",
        "Mohamed Riahi",
        "Maxime Lamothe",
        "Amine Mhedhbi"
      ],
      "abstract": "Text-to-SQL systems translate natural language queries into executable SQL, democratizing access to structured data. Despite recent advances driven by large language models (LLMs), evaluation remains a major bottleneck: public benchmarks fail to capture the complexity of enterprise schema, while building private evaluation sets is costly and nondeterministic, making evaluation results difficult to reproduce. To address this issue, we present SQLMorph, a framework for Text-to-SQL evaluation via query mutation. SQLMorph introduces two techniques to automatically generate and expand evaluation sets: Join Query Expansion (JQE), which systematically increases structural complexity through valid join additions, and Textual Query Augmentation (TQA), which generates controlled natural language perturbations to assess robustness to linguistic variation. JQE and TQA create targeted choke points to challenge specific system components. When applied to state-of-the-art systems, JQE increases query coverage and reveals accuracy degradation as the number of joins grows. Meanwhile, TQA shows that linguistic brittleness induced by heavy abbreviation can reduce accuracy by up to 17%. Beyond evaluation sets, SQLMorph introduces a family of execution-level metrics that address the limitations of current binary measures, such as Execution Accuracy. We define Execution Precision (EXP) and Execution Recall (EXR) to quantify the fraction of correct and recovered results, respectively, and combine them via F1 for unified scoring. Our experiments show that these relaxed metrics enable fine-grained analysis of over- and under-prediction, revealing differences across systems that binary metrics obscure. Together, SQLMorph's query mutation and fine-grained metrics support debugging and better align Text-to-SQL evaluation practices with real-world deployments.",
      "published": "2026-09-08T16:08:43Z",
      "abstract_url": "http://arxiv.org/abs/2609.08950v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08950v1",
      "categories": [
        "cs.DB",
        "cs.AI"
      ]
    },
    {
      "title": "Evaluating and Improving Evidence-Grounded Fact-Checking in LLMs via Multi-Round Evidence Ablation",
      "authors": [
        "Xingyu Deng",
        "Mingzi Cao",
        "Nikolaos Aletras",
        "Xi Wang",
        "Mark Stevenson"
      ],
      "abstract": "Automatic fact-checking systems assess the veracity of claims given evidence from relevant documents. Large Language Models (LLMs) have demonstrated strong performance in fact-checking due to their general reasoning capabilities. However, it remains unclear whether they faithfully make use of the evidence provided to reach veracity judgments or rely on parametric knowledge. To investigate this, we introduce Fact-Ablated Evaluation (FAE), a new evaluation framework that iteratively ablates the cited evidence to assess whether LLMs revise their predictions accordingly. Our empirical results show that current off-the-shelf LLMs as fact-checking systems rely more on their parametric knowledge than on the evidence provided. To bridge this gap between prediction accuracy and evidence grounding, we propose REAL (Rigorous Evidence Ablation Learning), a training framework that promotes evidence-dependent verification through counterfactual evidence supervision for the LLM-as-verifier models. Experiments on four fact-checking datasets across different domains demonstrate that models trained with REAL obtain superior evidence-dependent capabilities compared to standard fine-tuned models. Our findings highlight that strong fact-checking performance can still coexist with weak evidence dependency, while REAL encourages veracity predictions to remain more closely tied to the availability of supporting evidence.",
      "published": "2026-09-08T16:04:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.08943v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08943v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IR"
      ]
    },
    {
      "title": "Earth System World Model for What-If Simulations: A Case Study for Terrestrial Ecosystems",
      "authors": [
        "Zhihao Wang",
        "Ruichen Wang",
        "Ruohan Li",
        "Lei Ma",
        "George Hurtt",
        "Xiaowei Jia",
        "Gengchen Mai",
        "Shaowen Wang",
        "Yiqun Xie"
      ],
      "abstract": "Machine learning emulators have become essential for accelerating expensive Earth-system simulations, but most existing approaches remain passive forecasters: they reproduce simulator trajectories under prescribed forcings without an explicit interaction mechanism for user-specified interventions. This limits their use in interactive scientific workflows and Earth-system digital twins, where users often need to explore how a system would respond if selected state components were changed. We propose an action-conditioned world-modeling framework for Earth-system emulation that reformulates simulator trajectories as supervision for controllable state-transition learning. The key idea is transition-action pretraining: naturally observed state changes are treated as label-free action supervision, allowing the model to learn both prescribed dynamics and action-conditioned responses without manually annotated interventions. We further introduce masked response learning to infer unobserved variables under partial state edits and learn coupled system dependencies. We test this framework on ecosystem dynamics across six global regions and multiple stand ages. Experiments show that the model preserves competitive long-horizon emulation accuracy while enabling controllable structural interventions and coherent responses in coupled ecosystem-cycle variables. These results suggest a practical route from passive Earth-system emulators toward interactive, intervention-aware scientific surrogates.",
      "published": "2026-09-08T15:06:34Z",
      "abstract_url": "http://arxiv.org/abs/2609.08855v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08855v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Closing the Consistency Gap: Self-Evolving Agents That Learn to Stay on Course",
      "authors": [
        "Evelyn Duesterwald",
        "Benjamin Elder",
        "Lilian Ngweta",
        "Shashanka Ubaru",
        "Malgorzata Zimon"
      ],
      "abstract": "Large language model (LLM)-powered agents can be accurate on average yet unreliable in production, a discrepancy that has been observed but remains largely unaddressed. When given the same task five times, a ReAct agent on the AppWorld benchmark using GPT-4.1 succeeds in all five runs only 53% of the time, even though its per-run pass rate averages 77%. We call this 24-point shortfall the consistency gap, and we argue that addressing it is a precondition for trustworthy AI agent deployment. We present a self-evolving agent framework that reduces this gap by identifying unstable, low-consistency steps in agent trajectories and converting them into episodic memory the agent can draw on in future runs. At its core is a Consistency Analyzer that pinpoints where and why a trajectory is likely to flip across executions, and a Guideline Generator that converts the diagnosis into targeted guidelines, committed to memory and injected into future agent executions on similar tasks. On AppWorld with ReAct/GPT-4.1, our framework raises the fraction of tasks that succeed in all five runs by +16 points on same-task evaluation and +13 points on similar-task generalization.",
      "published": "2026-09-08T14:53:43Z",
      "abstract_url": "http://arxiv.org/abs/2609.08832v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08832v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Adaptive Anisotropic Attention for Axis-Structured Signals",
      "authors": [
        "Mahir Jain",
        "Parshva Runwal",
        "Aditya Ray Mishra",
        "Arvasu Kulkarni",
        "Sandeep Singh",
        "Siddharth Panwar"
      ],
      "abstract": "Dense self-attention treats all token pairs as equally plausible before learning, an interaction-isotropic prior that can be mismatched to structured signals. For structured, low signal-to-noise ratio (SNR) signals such as EEG, dependencies are organized along the electrode and time axes, and this uniform prior exposes each token to many irrelevant interactions. We introduce Adaptive Anisotropic Attention (AAA), which splits attention into two paths: a temporal path, where each token attends to the tokens of its own electrode across time, and a spatial path, where it attends to the tokens of the other electrodes at the same time step. A small gate predicts, for every token, a convex combination of the two path outputs: two non-negative weights that sum to one. On six EEG downstream tasks, the resulting model, AXON (AXis-factorized Operator Network), improves mean balanced accuracy over a dense baseline under both linear probing and full fine-tuning. We show that both paths (temporal and spatial) are necessary and that the weighted sum beats a hard choice of one path; most of the benefit comes from the gate learning a different temporal/spatial balance at each layer of the network. Controlled audio spectrogram experiments show that axis factorization transfers beyond EEG. These results suggest that aligning attention with the natural axes of structured signals provides a useful inductive bias.",
      "published": "2026-09-08T14:19:33Z",
      "abstract_url": "http://arxiv.org/abs/2609.08788v1",
      "pdf_url": "https://arxiv.org/pdf/2609.08788v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
