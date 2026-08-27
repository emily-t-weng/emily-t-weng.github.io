const PAPERS_DATA = {
  "last_updated": "2026-08-27 08:51:36 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning",
      "authors": [
        "Junxiang Xu",
        "Ruisi Wang",
        "Fanyi Pu",
        "Maijunxian Wang",
        "Ran Ji",
        "Tongxi Zhou",
        "Chenyang Gu",
        "Jing Zuo",
        "Hongcan Xiao",
        "Yimeng Geng",
        "Wanqi Yin",
        "Wei Chen",
        "Oscar Qian",
        "Zhengan Yan",
        "Ziqi Huang",
        "Haiwen Diao",
        "Liang Pan",
        "Bo Li",
        "Xiangyu Fan",
        "Dezhi Luo",
        "Fengyuan Yu",
        "Zehong Zhao",
        "Qingying Gao",
        "Tinghui Zhu",
        "Yilan Zhang",
        "Jingqi Tong",
        "Pinyuan Feng",
        "Zhengze Jiang",
        "Letian Wang",
        "Ziyu Guo",
        "Renrui Zhang",
        "Jieneng Chen",
        "Sonia Joseph",
        "Constantin Venhoff",
        "Saman Motamed",
        "Mengyue Yang",
        "Chandra Sripada",
        "Alan Yuille",
        "Philip Torr",
        "Lvmin Zhang",
        "Vikash Kumar",
        "Daniel Khashabi",
        "Nikolaus Kriegeskorte",
        "Raphaël Millière",
        "Vincent C. Müller",
        "Anyi Rao",
        "Quan Wang",
        "Ziwei Liu",
        "Dahua Lin",
        "Lei Yang",
        "Hokin Deng",
        "Zhongang Cai"
      ],
      "abstract": "Native visual reasoning treats visual generation as the medium of reasoning itself: visual states (i.e. images and videos) are not merely inputs to be understood or outputs to be rendered, but first-class substrates for problem solving beyond language. Yet progress remains bottlenecked by the lack of scalable training tasks, reliable feedback, and controlled comparisons across generative substrates. In this work, we introduce VBVR-Pro, a closed-loop testbed that makes native visual reasoning through generation trainable, verifiable, optimizable, and experimentally controllable. 1) Task scaling. VBVR-Pro turns visual reasoning into a controlled task space of 300 procedurally generated tasks. Models trained on VBVR-Pro show strong transfer beyond the proposed suite across seven external visual reasoning benchmarks such as RISE-Video, MME-CoF-Pro, and BabyVision. 2) Verifiable rewards. VBVR-Pro provides verifiable reward scorers for task-grounded evaluation. Through a systematic study of leading MLLMs as judges, we identify recurring failure modes of the prevalent VLM-as-a-judge paradigm. In contrast, the proposed scorers are grounded in deterministic, task-specific rules, achieve fine-grained alignment with human judgments. Importantly, they serve as reliable reward signals for large-scale multi-task reinforcement learning and demonstrate stronger post-RL performance across visual reasoning tasks. 3) Mechanism study. VBVR-Pro enables controlled modality studies across more than 30 image, video, and interleaved generators. Our analysis shows that video generation remains strongest for tasks requiring persistent spatiotemporal state tracking, while interleaved generation provides a compute-efficient alternative. Critically, ablations and probing suggest the presence of vision-native trajectories that are crucial to visual reasoning. We release all data, models, scorers, and code.",
      "published": "2026-08-26T17:59:51Z",
      "abstract_url": "http://arxiv.org/abs/2608.26105v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26105v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.MM",
        "cs.RO"
      ]
    },
    {
      "title": "MyoMechanix: Biomechanically-Grounded Compositional Skilled Activity Understanding and Coaching",
      "authors": [
        "Hao Yin",
        "Paritosh Parmar",
        "Lijun Gu",
        "Lin Xu",
        "Tianxiao Guo",
        "Xiujin Liu",
        "Tianyou Zheng",
        "Yang Zhang",
        "Weiwei Fu"
      ],
      "abstract": "Existing action quality assessment (AQA) datasets and methods rely primarily on visual inputs such as RGB and pose, overlooking physiological dynamics such as muscle mechanics and often modeling actions as monolithic patterns. These limitations hinder fine-grained, biomechanically grounded feedback. We introduce MyoMechanix, a multimodal ecosystem for weight-loaded actions that aligns motion with muscle activity. Expert-annotated, it contains 7,500+ samples of 20 actions from 38 subjects, with synchronized multiview RGB video, 3D pose, sEMG, and additional physiological signals, forming the largest multimodal AQA benchmark to date. We further construct the Fitness Knowledge Graph (FKG), which organizes expert annotations into structured relationships among actions, phases, key steps, errors, and corrective feedback, enabling compositional scoring and interpretable assessment. Building on these representations, we develop CUBIST (Compositional Ontological Reasoning Engine), which performs decomposition-analysis-recomposition for fine-grained error attribution and feedback generation. We also establish MyoMechanix-AQA, MyoMechanix-VideoQA, and a novel MyoMechanix-Video2EMG task. Experiments show that multimodal sensing and structured representations improve performance, interpretability, and error attribution, with CUBIST achieving state-of-the-art results; VideoQA enhances language-grounded action understanding; and Video2EMG suggests video-based alternatives to costly EMG sensing. MyoMechanix advances skilled activity understanding toward biomechanically grounded, multimodal, and compositional reasoning for Physical AI applications in fitness, rehabilitation, healthcare, and machine learning. Project page: https://haoyin116.github.io/MyoMechanix/",
      "published": "2026-08-26T17:56:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.26094v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26094v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.ET",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "Finding and using interpretable latents in a neutrino foundation model with sparse autoencoders",
      "authors": [
        "Raphaël Bonnet-Guerrini",
        "Johann Ioannou-Nikolaides",
        "Inar Timiryasov",
        "Vincenzo Piuri"
      ],
      "abstract": "We present a first application of sparse-autoencoder-based mechanistic interpretability to particle physics. Studying a neutrino foundation model pretrained on IceCube data and fine-tuned for direction reconstruction, we identify a validated atlas of physical concepts in the model representation, using a strict validation protocol consisting of held-out tests, matched nuisance controls, and replication across independent dictionary trainings. Causal interventions show that the direction head barely draws on this atlas. Motivated by this underused information, we train an uncertainty head on the same event-level representation to predict the model's angular reconstruction error. Unlike the direction head, it depends causally on quality and brightness features from the atlas. At $20\\%$ selection efficiency, this interpretable estimator improves the median angular resolution from $20.2^\\circ$ to $3.2^\\circ$. These results suggest that mechanistic interpretability can reveal learned latent physics encoded within a model's internal representation and help design downstream tasks that exploit it.",
      "published": "2026-08-26T17:53:00Z",
      "abstract_url": "http://arxiv.org/abs/2608.26090v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26090v1",
      "categories": [
        "astro-ph.HE",
        "cs.AI",
        "cs.LG",
        "hep-ex"
      ]
    },
    {
      "title": "Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings",
      "authors": [
        "Evelyn Ma",
        "Rama Kumar Pasumarthi",
        "Kishwar Shafin",
        "Mandar Sharma",
        "Mimi Sun",
        "Hamed Sadeghi",
        "Dav M. Ebengo",
        "Mbulayi Onesime",
        "Rouslan Solomakhin",
        "John Wamburu",
        "William Ogallo",
        "Aisha Walcott-Bryant",
        "Sanxing Chen",
        "Arbaaz Muslim",
        "Yael Mayer",
        "Ronald Ho",
        "Roy Lee",
        "Ruth Alcantara",
        "Abdoulaye Diack",
        "Monica Bharel",
        "Lambert Rosique",
        "Jeremy Amez-Droz",
        "Christopher Haire",
        "James Manyika",
        "Yossi Matias",
        "Niv Efron",
        "Gautam Prasad",
        "Shravya Shetty"
      ],
      "abstract": "Addressing critical global challenges, from food security and disaster risk to disease outbreaks and socio-economic vulnerability, demands high-fidelity geospatial modeling. However, building predictive planetary models remains bottlenecked by a fragmented data ecosystem, requiring manual data retrieval, multimodal data curation and fusion along with iterative model selection. We present the Planetary Prediction Engine (PPE), an autonomous AI system that executes this end-to-end workflow directly from natural-language queries. PPE synthesizes multimodal datasets on the fly, retrieving spatiotemporally relevant covariates across open-web and Earth observation platforms (Data Commons, Google Earth Engine) and fusing them with geospatial foundation model embeddings (PDFM, AlphaEarth). Simultaneously, it searches over task-tailored model architecture families with automated overfitting guards. Across diverse tasks, geographies, and scientific domains, PPE consistently outperforms state-of-the-art or manually tuned expert baselines. For US spatial regression, PPE improves mean $R^2$ across 21 CDC health indicators (76.8% vs. 60.0%), FEMA national risk indices (64.9% vs. 60.0%), and the Social Vulnerability Index (66.2% vs. 58.6%). For spatial downscaling in data-scarce settings, PPE integrates localized proxies to double baseline accuracy in Nigerian food security indicators ($R^2$ of 66.1% vs. 31.5%). For epidemiological nowcasting of the 2026 DRC Bundibugyo Ebola outbreak, PPE achieves a Recall@10 of 83.3% (identifying 15 of 18 newly invaded health zones across five weekly forecasts), a +10.3 percentage-point improvement over the public state-of-the-art modeling (~73%). By combining autonomous multimodal planetary data discovery with targeted model optimization, PPE lowers the technical barrier to planetary-scale analytics, enabling rapid, customized, expert-level deployment.",
      "published": "2026-08-26T17:50:52Z",
      "abstract_url": "http://arxiv.org/abs/2608.26088v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26088v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "TraceML: An Empirical Analysis of Human-Agent Planning in Machine Learning Development",
      "authors": [
        "Jiarui Yan",
        "Weiwei Sun",
        "Sijie Li",
        "Wenhan Li",
        "Yiming Yang"
      ],
      "abstract": "Large language models write correct code for isolated problems but remain far weaker at autonomous machine-learning development, where an agent must revise data pipelines, models, and validation over hours of feedback, and on most competitions still finishes below strong human competitors. Outcome-based benchmarks record this gap but not its cause, because they grade the final submission and discard the development process behind it. We introduce TraceML, which pairs human and agent work on the same competitions under one version-level schema: 4,465 human Kaggle trajectories across 134 competitions, seven of which are also worked by two agent scaffolds, giving 430 paired human and 207 agent trajectories. Every code version carries its score, its timestamp, and labels for the action taken, its intent, the edit size, and the score effect. Read this way, the gap becomes concrete. Experts alternate data work, validation, model changes, and ensembling, and return to approaches they had set aside. Each agent scaffold instead collapses into a narrow loop: Codex spends its steps re-weighting ensembles and tuning submissions, MLEvolve mutates its model in place, and neither pivots at the human rate nor reopens abandoned work. A short planning prompt distilled from human practice moves the behaviors it names toward the human profile and lifts scores, but the effort profile stays agent-shaped: instruction closes only the part of the gap that reduces to instructions. We release the corpus, the schema, the labelers, and the extraction pipeline at https://huggingface.co/datasets/jerryyan/TraceML.",
      "published": "2026-08-26T17:50:13Z",
      "abstract_url": "http://arxiv.org/abs/2608.26086v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26086v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing",
      "authors": [
        "Roshan Prakash Rane",
        "Marco Simnacher",
        "Manuel Pfeuffer",
        "Marc-Andre Schulz",
        "Nys Tjade Siegel",
        "Maximilian Dreyer",
        "Frederik Pahde",
        "Wojciech Samek",
        "Sonja Greven",
        "Kerstin Ritter"
      ],
      "abstract": "Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validate by retraining and out-of-distribution testing.",
      "published": "2026-08-26T17:47:49Z",
      "abstract_url": "http://arxiv.org/abs/2608.26083v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26083v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "stat.ML"
      ]
    },
    {
      "title": "Prefix Sliding for efficient test-time scaling",
      "authors": [
        "Niklas Muennighoff",
        "Zhengyang Wang",
        "Zeyi Chen",
        "Weijia Shi",
        "Binyuan Hui",
        "John Yang",
        "Dapeng Jiang",
        "Mika Senghaas",
        "Fares Obeid",
        "Johannes Hagemann",
        "Sami Jaghouar",
        "Ludwig Schmidt",
        "Percy Liang",
        "Jason Wei",
        "Andrew Y. Ng",
        "Luke Zettlemoyer",
        "Yejin Choi",
        "Mike Lewis"
      ],
      "abstract": "Test-time scaling uses extra test-time compute to improve performance, such as letting language models reason longer when solving a problem. As models keep the entire reasoning trace in memory via full attention, hard tasks that need long thinking can be prohibitively expensive. However, we find most intermediate reasoning tokens lose importance as the model continues reasoning. This calls into question whether retaining them is worth the cost. Based on this insight, we propose Prefix Sliding, which discards tokens during reasoning that are not part of the prefix or the window of the last few thousand tokens. The prefix has key instructions and tools available to the model, while the most recent tokens are the current reasoning the model is working on. This caps the total memory requirement regardless of how long the model reasons, allowing for efficient long-horizon test-time scaling. Without training, Prefix Sliding can make existing models 3x faster while maintaining performance. Training with Prefix Sliding using reinforcement learning can achieve better performance by enabling scaling to reasoning traces beyond a hundred thousand tokens. Ablations show Prefix Sliding outperforms summarizing intermediate tokens or vanilla sliding window. Our code is at https://github.com/Muennighoff/prefix-sliding",
      "published": "2026-08-26T17:37:15Z",
      "abstract_url": "http://arxiv.org/abs/2608.26070v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26070v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "$R^3$: Training Robots to Reason in Natural Language via Reinforcement Learning",
      "authors": [
        "Lehong Wu",
        "Yuxiao Qu",
        "Zheyuan Hu",
        "Ivan Zhang",
        "Limin Wei",
        "Zackory Erickson",
        "Aviral Kumar"
      ],
      "abstract": "Reasoning in language allows foundation models to spend more test-time compute on hard problems, such as those requiring decomposition, constraint tracking, and prediction of future consequences. Whether this mechanism can improve robotic manipulation remains unclear, where long-horizon tasks require tracking partial progress, reasoning about object relations, recovering from mistakes, and steering noisy low-level policies. In this paper, we study whether VLMs can be trained to reason directly in natural language to guide low-level manipulation policies. We introduce $R^3$, a simple post-training recipe that turns off-the-shelf VLMs into robotic reasoners: it first mid-trains a VLM on expert-generated reasoning traces to initialize the desired reasoning style, then improves the reasoner with single-step rubric-based RL from offline action data. Unlike prior robotic reasoning methods that mostly use structured traces as auxiliary supervision, $R^3$ trains free-form language reasoning to produce test-time guidance for action. We instantiate $R^3$ on Language Table and simulated bimanual grocery packing, two controlled testbeds for studying robotic reasoning and long-horizon manipulation. $R^3$ improves exploration and generalization across unseen tasks and significantly outperforms instruction-only imitation learning baselines on both benchmarks. Our analyses suggest that free-form language reasoning can function as a test-time compute mechanism for steering low-level policies. Our project page is available at https://robotic-reasoner.github.io/.",
      "published": "2026-08-26T17:25:10Z",
      "abstract_url": "http://arxiv.org/abs/2608.26053v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26053v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "How Much Rank Does LoRA Need? Rank-Error Bounds for Transformer Attention",
      "authors": [
        "Gerard Conangla Planes"
      ],
      "abstract": "Choosing the rank of a low-rank adaptation (LoRA) update is usually an empirical task. In this paper, we provide a task-dependent theory of the approximation error achievable at each LoRA rank for Transformer attention. We fix a pretrained attention head, a target attention function, and a distribution over inputs from the downstream task, and bound the smallest expected Kullback--Leibler (KL) error achievable by a rank-$r$ query LoRA update. When target attention probabilities are bounded away from zero, we prove a lower bound of the error proportional to $ψ(\\|d\\|_2)$, where $d$ is the difference between candidate and target attention scores and $ψ(t)=\\min\\{t^2,t\\}$. We also prove an unconditional upper bound $\\min\\{\\|d\\|_2^2/4,\\sqrt2\\|d\\|_2\\}$. Under explicit realizability, geometry, and moment conditions, we then bound the best rank-$r$ error between an explicit multiple of $ψ(\\sqrt{T_r})$ and $\\min\\{T_r/4,\\sqrt{2T_r}\\}$, where $T_r$ is the downstream-weighted tail energy of the target update. We also provide target-Fisher bounds when candidate scores remain within a fixed range of the target scores, and an unrestricted lower bound when a subset of tokens carries most of the probability mass. These spectral bounds describe finite-score approximation. We then construct explicit families in which softmax saturation makes the rank required to match the attention function strictly smaller than the rank required to match the finite logits. Finally, we extend the analysis to fused multi-head LoRA and joint query/key updates, exposing the effects of rank sharing and query/key factorization constraints.",
      "published": "2026-08-26T17:25:03Z",
      "abstract_url": "http://arxiv.org/abs/2608.26052v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26052v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "DualOPSD: Adaptive Privileged Teachers for On-Policy Self-Distillation",
      "authors": [
        "Yutong Chen",
        "Guangfu Guo",
        "Zhichao Xu",
        "Kunpeng Liu"
      ],
      "abstract": "On-policy self-distillation (OPSD) uses a privileged copy of the student model to provide dense supervision without an external teacher. OPSD keeps this privileged teacher fixed, even though the student distribution and output style change during training. We propose DualOPSD, an asymmetric alternating framework that adapts both policies. The student first learns from the privileged teacher. The teacher then moves toward the updated student distribution on the same student trajectory. This update makes later supervision responsive to the learner and does not require another rollout. On Qwen3-8B in non-thinking mode, DualOPSD improves avg@12 over OPSD by 23.61, 13.89, and 10.00 points on AIME 2024, AIME 2025, and HMMT 2025. Results at 1.7B and 4B show that the accuracy gain depends on model scale. Across all three scales, DualOPSD reduces truncation. The 4B diagnostic also shows lower KL in both directions between the teacher and student.",
      "published": "2026-08-26T17:01:21Z",
      "abstract_url": "http://arxiv.org/abs/2608.26019v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26019v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Imitation Learning for Connection-Tableau Construction",
      "authors": [
        "Fredrik Rømming",
        "Mantas Bakšys",
        "Martin S. Fixman",
        "Sean B. Holden"
      ],
      "abstract": "An automated theorem prover builds a proof step by step, choosing at each point what to add and what to remove. We cast this construction as a policy acting in a transition system induced by a formal calculus, which fixes which steps are sound: for clausal connection tableaux, leanCoP-style search and plCoP/rlCoP-style planning then become stateful policies over one interface, and policy-learning methods apply directly. We equip such policies with a graph neural network that scores proof edits from structure that transfers across problems, train it by imitation learning from found proofs, and measure how performance holds as we remove search scaffolding, from full symbolic backtracking to a policy the network drives alone. Within a fixed step budget on M2k, MPTP2078-bushy, and TPTP v9.2.1, learned policies solve up to 46% more problems than leanCoP, and reach proofs in an order of magnitude fewer steps.",
      "published": "2026-08-26T16:53:25Z",
      "abstract_url": "http://arxiv.org/abs/2608.26009v1",
      "pdf_url": "https://arxiv.org/pdf/2608.26009v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.LO"
      ]
    },
    {
      "title": "ProgRouter: Online Progress-Guided Orchestration for Multi-Agent LLM Workflows under Quality-Cost Tradeoffs",
      "authors": [
        "Somgyuan Li",
        "Ahmed M. Abdelmoniem",
        "Shiqiang Wang"
      ],
      "abstract": "Multi-agent large language model (LLM) workflows have emerged as a powerful paradigm for solving complex, open-ended tasks through collaborative reasoning among specialized LLM agents, but they incur substantial operating costs due to repeated LLM invocations and long-horizon context accumulation. Existing cascade routing methods make one-shot, query-level decisions and cannot adapt to the dynamic, state-dependent nature of multi-step workflows, in which the right LLM at each step depends on evolving task progress, remaining task difficulty, and cost-efficiency requirements. We present ProgRouter, an online progress-guided routing framework that adaptively selects LLM agents across workflow steps to preserve task-solving quality while adhering to time and cost budgets. ProgRouter introduces a multi-view task progress scorer that combines coarse workflow outcome regimes with fine-grained signals on subtask completion, progress trends, and workflow state quality. Then, a dual-path task progress predictor and an adaptive meta-gating mechanism estimate the progress gain for each candidate routed LLM. ProgRouter makes online step-wise routing decisions that balance progress gain, task time budgets, and long-term operating cost efficiency. Experiments on HumanEval Plus, MBPP, MATH-500, and ASQA, spanning agentic code generation, mathematical reasoning, and retrieval-augmented long-form question answering, demonstrate that ProgRouter reduces the operating cost relative to key baselines while maintaining strong task-solving performance.",
      "published": "2026-08-26T16:42:02Z",
      "abstract_url": "http://arxiv.org/abs/2608.25992v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25992v1",
      "categories": [
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "Multi-Granularity Context-Enhanced RAG over Multimodal Knowledge Graphs",
      "authors": [
        "Zongyu Wu",
        "Yilong Wang",
        "Xiaochen Wang",
        "Minhua Lin",
        "Zhichao Xu",
        "Fenglong Ma",
        "Xiang Zhang",
        "Suhang Wang"
      ],
      "abstract": "Retrieval-augmented generation (RAG) is widely used to mitigate hallucination issues in large language models (LLMs) and multimodal large language models (MLLMs). In particular, knowledge graph (KG)-based RAG leverages structured knowledge to provide (M)LLMs with high-quality external information. Building on these works, recent studies have explored multimodal knowledge graphs (MMKGs) as knowledge bases for GraphRAG. This enables Graph RAG to integrate knowledge across multiple modalities, thereby further enhancing its performance. However, existing MMKG-based RAG methods generally follow a common pipeline in which different modalities are largely processed independently before being fusion. As a result, textual context is only used to a limited extent during visual information extraction and subsequent multimodal knowledge fusion. This brings a semantic gap between images and text which limits the multimodal GraphRAG performance. To address this issue, we propose a novel framework for constructing a Context-Enhanced MMKG (CEMMKG) to better support multimodal GraphRAG. The proposed CEMMKG enriches each image with complementary textual context at both local and global scopes. Local context goes beyond the surrounding text by incorporating sentences that are semantically related to the image, while global context provides a summary of the entire passage. We further introduce a multi-granularity design for the local context, allowing it to capture semantically relevant information at different levels of detail. Extensive experiments on the selected vision-centric dataset validate that CEMMKG is effective in leveraging contextual information to improve MMKG-based RAG performance. Moreover, its effectiveness across different MMKG-based RAG methods demonstrates its broad applicability.",
      "published": "2026-08-26T16:38:02Z",
      "abstract_url": "http://arxiv.org/abs/2608.25986v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25986v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "FRAME: separating sampling variation from representational cause in medical imaging fairness",
      "authors": [
        "Mahshad Lotfinia",
        "Daniel Truhn",
        "Andreas Maier",
        "Soroosh Tayebi Arasteh"
      ],
      "abstract": "Subgroup performance differences are the standard evidence for fairness bias in medical imaging, and the usual response removes the demographic information that a model encodes. Here we introduce Fair-model Reference And Mechanism Evaluation (FRAME), a two-step framework for auditing such a claim. The first step derives a fair-model reference, the distribution of the difference under exact fairness at the observed subgroup sizes. In the second step, we test the remainder with two operators in representation space. One operator cannot change a within-group ranking by construction. Across 702,206 images and 36 encoders, the reference accounts for a median 41% of the reported race difference and 22% of the age difference. Injecting demographic decodability leaves the remainder unchanged, while entangling the group with the disease direction raises the race difference from 0.077 to 0.118. No intervention we tested changes the remainder more than a change of random seed does. Those interventions reduce a difference at the operating point and leave the within-group ranking difference at a median of 0.000. Applied to 89 differences in 9 published studies across 6 medical imaging modalities, the reference accounts for a median 25% of a rate difference and 70% of a difference in the area under the receiver operating characteristic curve. Image-text pretraining instead raises worst-group performance by about 0.05. Applying FRAME before choosing an intervention could distinguish differences that need a mechanistic explanation from differences compatible with sampling variation at the current cohort sizes.",
      "published": "2026-08-26T16:34:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.25981v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25981v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SciMIF: Understanding Multimodal Instruction Following in Scientific Domains",
      "authors": [
        "Ye Shen",
        "Yuting Zheng",
        "Dun Pei",
        "Zijian Chen",
        "Wenlong Zhang",
        "Qi Jia",
        "Guangtao Zhai"
      ],
      "abstract": "Understanding instruction-following capabilities in scientific domains is essential for effectively leveraging Multimodal Large Language Models (MLLMs) to advance the development of scientific fields. In this work, we introduce SciMIF, a novel benchmark designed to evaluate the capability of MLLMs in following complex scientific instructions. Specifically, based on an extensive analysis of 22 distinct tasks across 5 representative scientific disciplines, we propose a comprehensive taxonomy comprising 10 constraint groups that captures both general functional requirements and discipline-specific characteristics. Guided by this taxonomy, we develop a high-fidelity instruction injection pipeline to systematically augment existing scientific datasets. We conduct comprehensive experiments on multiple state-of-the-art closed-source and open-source MLLMs. Our findings reveal significant performance disparities across different scientific disciplines, with chemistry posing greater challenges for current MLLMs. Furthermore, we observe that increasing the model scale does not yield corresponding improvements in constraint adherence, and current models still struggle severely with fine-grained constraints and instructions requiring the deep application of disciplinary knowledge. SciMIF fills the current void in evaluating multimodal instruction adherence within scientific domains, laying a crucial foundation for future enhancements of MLLMs in rigorous scientific applications. Data and code will be released at https://github.com/shenye7436/SciMIF .",
      "published": "2026-08-26T16:30:20Z",
      "abstract_url": "http://arxiv.org/abs/2608.25973v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25973v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "One Symptom, Three Levers: A Critical Review of On-Policy Self-Distillation",
      "authors": [
        "Justin Robert",
        "Raheel Qader"
      ],
      "abstract": "On-policy distillation trains a language model on its own generations while a teacher scores them token by token. It combines the dense supervision of imitation learning with the on-policy sampling of reinforcement learning. But it requires a second, larger model to act as teacher. On-Policy Self-Distillation (OPSD) removes that cost. The teacher is the model itself, conditioned on privileged information the student will not have at test time, such as a reference solution, a plan, or environment feedback. The teacher is no stronger than the student, only better informed. Early results were promising, with accuracy comparable to reinforcement learning at a fraction of the generated tokens. But the same asymmetry that produces the signal also biases it. One failure mode now dominates the field: collapse, the progressive narrowing of the set of reasoning paths the model can produce. Collapse is not specific to OPSD, though privileged information aggravates it. This review treats collapse as a symptom governed by three levers: (i) where the signal is applied, that is, how tokens are weighted; (ii) what the teacher is shown, that is, the nature of the privileged information; and (iii) when the signal changes, that is, the teacher's dynamics and the decay of guidance. We restrict our scope to mathematical reasoning, where the method originated and where its failure modes are best documented. We report no new experiments. The contribution is structural: a shared vocabulary for phenomena named differently across papers, and a clear line between what is settled and what is still disputed.",
      "published": "2026-08-26T15:52:19Z",
      "abstract_url": "http://arxiv.org/abs/2608.25936v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25936v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "How Robust Are Automated Fact-Checking Systems? A Cross-Benchmark Evaluation",
      "authors": [
        "Aida Usmanova",
        "Zangir Iklassov",
        "Markus Leippold",
        "Ricardo Usbeck"
      ],
      "abstract": "Automated fact-checking (AFC) systems retrieve evidence and predict claim veracity, yet evaluations omit simple baselines, systems are developed for a single benchmark and cannot be trusted to generalise across domains. No prior work cross-evaluates the full two-stage retrieve-then-verify pipeline across diverse datasets, complementing retrieval-only studies (Thakur et al., 2021) and single-stage benchmarking studies (Calamai et al., 2025). We benchmark nine models, ranging from random and sparse baselines to fine-tuned transformers, zero-shot LLMs, and the two highest-ranked systems from the AVeriTeC 2025 shared task, across four datasets spanning scientific, open-web, and climate domains. Three findings stand out: (1) on ClimateCheck claim-only and fine-tuned models outperform zero-shot LLM and top-performing AVeriTeC 2025 systems, highlighting that noisy evidence can degrade veracity prediction; (2) system rankings are strongly domain- and metric-dependent: the best model on SciFact (macro-F1 0.70) drops to 0.31 on ClimateCheck, while the AVeriTeC 2025 winner and runner-up swap rankings based on evaluation metrics and datasets; (3) replacing retrieved evidence with gold annotations improves veracity accuracy by 14-22 points across models, confirming retrieval remains primary bottleneck. We release code, pre-processed datasets, and all results to support reproducible AFC research.",
      "published": "2026-08-26T15:50:10Z",
      "abstract_url": "http://arxiv.org/abs/2608.25934v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25934v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Repair or Resample? Rethinking Failure Debugging in LLM Multi-Agent Systems",
      "authors": [
        "Zhongwen Luan",
        "Xiaoyu Zhang",
        "Ming Hu",
        "Yue Yang",
        "Jiongchi Yu",
        "Xiaohong Chen"
      ],
      "abstract": "As large language model (LLM)-based multi-agent systems (MASs) are increasingly applied to long-horizon complex tasks, their reliability has emerged as the core bottleneck hindering their real-world deployment. Existing MAS debugging and repair methods typically rely on rerunning and resampling the entire execution trajectory. However, a fundamental question remains to be answered: do these methods causally repair MAS failures or merely stochastically repair by leveraging the randomness of LLM sampling? To evaluate the effectiveness of MAS repair methods, we introduce SymTrace, a controlled evaluation framework that records the MAS execution trajectory and establishes intervention anchors. During replay, it effectively reconstructs the execution before the anchor using recorded logs and only regenerates the downstream trajectory, thereby enabling the reliable reproduction of MAS failures. We further construct the dataset SymFail, comprising 536 human-annotated failure trajectories with graph-linked locations, categories, and trace evidence. Based on these foundations, we conduct a large-scale empirical study across three mainstream MAS frameworks. Our findings reveal that existing unguided rerun methods are highly unreliable, exhibiting low failure reproduction and repair rates (only 67.97% and 6.90%, respectively). Building upon these findings, we further explore the effectiveness of a symptom-driven intervention method, which successfully repairs 20.15% of the failed cases (a 191.89% improvement to state-of-the-art repair methods). This study aims to provide actionable insights for MAS debugging and repair research, paving the way for the robust deployment of multi-agent systems.",
      "published": "2026-08-26T15:33:47Z",
      "abstract_url": "http://arxiv.org/abs/2608.25920v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25920v1",
      "categories": [
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "Towards A Unified Information Bottleneck Framework for Time Series Explanations",
      "authors": [
        "Xu Zheng",
        "Zichuan Liu",
        "Zhuomin Chen",
        "Mayur Akewar",
        "Janki Bhimani",
        "Jason Liu",
        "Mo Sha",
        "Jingchao Ni",
        "Wei Cheng",
        "Dongsheng Luo"
      ],
      "abstract": "Explaining deep learning models operating on time series data is crucial in various applications that require transparent and interpretable insights into model behavior. {Existing explanation methods generally fall into two categories: attribution-based explanations, which identify the temporal regions most responsible for a prediction, and counterfactual explanations, which reveal how an input should be modified to alter the model's decision.} {Despite valuable insights, these two fields are largely studied independently. This disconnect leaves attribution methods lacking causal validation, while counterfactual methods suffer from severe instability, producing adversarial-like noise instead of meaningful explanations.} In this work, we revisit time-series explainability from an information-theoretic perspective and show that existing explainers are vulnerable to trivial solutions and distributional shifts. To address these limitations, we propose a unified objective function for explainable time series learning that bridges attribution and counterfactual reasoning within a single framework. Building upon the Information Bottleneck principle, our formulation explicitly prevents trivial explanations and out-of-distribution counterfactuals. {Based on this objective function, we introduce {\\modelname}, a novel explanation framework that learns a parametric transformation network to construct explanation-embedded instances, where preserved information yields attribution explanations and controlled information removal produces stable counterfactual explanations.} We evaluate {\\modelname} on synthetic and real-world benchmarks against state-of-the-art baselines. Extensive quantitative and qualitative results show that {\\modelname} consistently outperforms competing methods, yielding faithful attributions and stable counterfactual explanations.",
      "published": "2026-08-26T15:14:52Z",
      "abstract_url": "http://arxiv.org/abs/2608.25897v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25897v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Why ML-based cough models do not generalize: a systematic cross-dataset evaluation for tuberculosis screening",
      "authors": [
        "Wensi Zhang",
        "Tomas Teijeiro",
        "Jérôme Thevenot",
        "David Atienza"
      ],
      "abstract": "Cough acoustics are promising for non-invasive tuberculosis (TB) screening, yet whether machine learning (ML) models capture disease-related acoustics or artifacts of data collection remains unresolved. We evaluated the cross-dataset generalizability of classical ML and deep learning (DL) cough-based TB classifiers across three independent datasets. Despite moderate within-dataset performance (ROC-AUC up to $0.755 \\pm 0.056$), both pipelines fail to generalize, with external performance frequently below 0.6, indicating a possible limitation of the data. We further observed audio representations are organized by recording device and dataset rather than TB status, predicted TB probability tracks country-level prevalence in CODA, and device mismatch degrades transfer while device-diverse training improves it. Additionally, a clinical-variable baseline generalizes more consistently (ROC-AUC $0.655 - 0.711$), indicating acquisition-specific variability is a stronger driver of poor generalizability than population shift. High within-dataset performance is not enough. External validation is essential before cough-based TB models are clinically ready.",
      "published": "2026-08-26T14:22:32Z",
      "abstract_url": "http://arxiv.org/abs/2608.25846v1",
      "pdf_url": "https://arxiv.org/pdf/2608.25846v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.LG",
        "cs.SD",
        "eess.SP"
      ]
    }
  ]
};
