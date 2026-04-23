const PAPERS_DATA = {
  "last_updated": "2026-04-23 03:33:59 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "FedSIR: Spectral Client Identification and Relabeling for Federated Learning with Noisy Labels",
      "authors": [
        "Sina Gholami",
        "Abdulmoneam Ali",
        "Tania Haghighi",
        "Ahmed Arafa",
        "Minhaj Nur Alam"
      ],
      "abstract": "Federated learning (FL) enables collaborative model training without sharing raw data; however, the presence of noisy labels across distributed clients can severely degrade the learning performance. In this paper, we propose FedSIR, a multi-stage framework for robust FL under noisy labels. Different from existing approaches that mainly rely on designing noise-tolerant loss functions or exploiting loss dynamics during training, our method leverages the spectral structure of client feature representations to identify and mitigate label noise. Our framework consists of three key components. First, we identify clean and noisy clients by analyzing the spectral consistency of class-wise feature subspaces with minimal communication overhead. Second, clean clients provide spectral references that enable noisy clients to relabel potentially corrupted samples using both dominant class directions and residual subspaces. Third, we employ a noise-aware training strategy that integrates logit-adjusted loss, knowledge distillation, and distance-aware aggregation to further stabilize federated optimization. Extensive experiments on standard FL benchmarks demonstrate that FedSIR consistently outperforms state-of-the-art methods for FL with noisy labels. The code is available at https://github.com/sinagh72/FedSIR.",
      "published": "2026-04-22T17:49:20Z",
      "abstract_url": "http://arxiv.org/abs/2604.20825v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20825v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "cs.DC",
        "eess.SP"
      ]
    },
    {
      "title": "Convergent Evolution: How Different Language Models Learn Similar Number Representations",
      "authors": [
        "Deqing Fu",
        "Tianyi Zhou",
        "Mikhail Belkin",
        "Vatsal Sharan",
        "Robin Jia"
      ],
      "abstract": "Language models trained on natural text learn to represent numbers using periodic features with dominant periods at $T=2, 5, 10$. In this paper, we identify a two-tiered hierarchy of these features: while Transformers, Linear RNNs, LSTMs, and classical word embeddings trained in different ways all learn features that have period-$T$ spikes in the Fourier domain, only some learn geometrically separable features that can be used to linearly classify a number mod-$T$. To explain this incongruity, we prove that Fourier domain sparsity is necessary but not sufficient for mod-$T$ geometric separability. Empirically, we investigate when model training yields geometrically separable features, finding that the data, architecture, optimizer, and tokenizer all play key roles. In particular, we identify two different routes through which models can acquire geometrically separable features: they can learn them from complementary co-occurrence signals in general language data, including text-number co-occurrence and cross-number interaction, or from multi-token (but not single-token) addition problems. Overall, our results highlight the phenomenon of convergent evolution in feature learning: A diverse range of models learn similar features from different training signals.",
      "published": "2026-04-22T17:45:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.20817v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20817v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning for Hybrid Intelligent Systems",
      "authors": [
        "Pavel Salovskii",
        "Iuliia Gorshkova"
      ],
      "abstract": "This paper presents a hybrid architecture for intelligent systems in which large language models (LLMs) are extended with an external ontological memory layer. Instead of relying solely on parametric knowledge and vector-based retrieval (RAG), the proposed approach constructs and maintains a structured knowledge graph using RDF/OWL representations, enabling persistent, verifiable, and semantically grounded reasoning. The core contribution is an automated pipeline for ontology construction from heterogeneous data sources, including documents, APIs, and dialogue logs. The system performs entity recognition, relation extraction, normalization, and triple generation, followed by validation using SHACL and OWL constraints, and continuous graph updates. During inference, LLMs operate over a combined context that integrates vector-based retrieval with graph-based reasoning and external tool interaction. Experimental observations on planning tasks, including the Tower of Hanoi benchmark, indicate that ontology augmentation improves performance in multi-step reasoning scenarios compared to baseline LLM systems. In addition, the ontology layer enables formal validation of generated outputs, transforming the system into a generation-verification-correction pipeline. The proposed architecture addresses key limitations of current LLM-based systems, including lack of long-term memory, weak structural understanding, and limited reasoning capabilities. It provides a foundation for building agent-based systems, robotics applications, and enterprise AI solutions that require persistent knowledge, explainability, and reliable decision-making.",
      "published": "2026-04-22T17:19:43Z",
      "abstract_url": "http://arxiv.org/abs/2604.20795v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20795v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Can \"AI\" Be a Doctor? A Study of Empathy, Readability, and Alignment in Clinical LLMs",
      "authors": [
        "Mariano Barone",
        "Francesco Di Serio",
        "Roberto Moio",
        "Marco Postiglione",
        "Giuseppe Riccio",
        "Antonio Romano",
        "Vincenzo Moscato"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly deployed in healthcare, yet their communicative alignment with clinical standards remains insufficiently quantified. We conduct a multidimensional evaluation of general-purpose and domain-specialized LLMs across structured medical explanations and real-world physician-patient interactions, analyzing semantic fidelity, readability, and affective resonance. Baseline models amplify affective polarity relative to physicians (Very Negative: 43.14-45.10% vs. 37.25%) and, in larger architectures such as GPT-5 and Claude, produce substantially higher linguistic complexity (FKGL up to 16.91-17.60 vs. 11.47-12.50 in physician-authored responses). Empathy-oriented prompting reduces extreme negativity and lowers grade-level complexity (up to -6.87 FKGL points for GPT-5) but does not significantly increase semantic fidelity. Collaborative rewriting yields the strongest overall alignment. Rephrase configurations achieve the highest semantic similarity to physician answers (up to mean = 0.93) while consistently improving readability and reducing affective extremity. Dual stakeholder evaluation shows that no model surpasses physicians on epistemic criteria, whereas patients consistently prefer rewritten variants for clarity and emotional tone. These findings suggest that LLMs function most effectively as collaborative communication enhancers rather than replacements for clinical expertise.",
      "published": "2026-04-22T17:17:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.20791v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20791v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Working Memory Constraints Scaffold Learning in Transformers under Data Scarcity",
      "authors": [
        "Pranava Madhyastha",
        "Dagmar Adamcova"
      ],
      "abstract": "We investigate the integration of human-like working memory constraints into the Transformer architecture and implement several cognitively inspired attention variants, including fixed-width windows based and temporal decay based attention mechanisms. Our modified GPT-2 models are trained from scratch on developmentally plausible datasets (10M and 100M words). Performance is evaluated on grammatical judgment tasks (BLiMP) and alignment with human reading time data. Our results indicate that these cognitively-inspired constraints, particularly fixed-width attention, can significantly improve grammatical accuracy especially when training data is scarce. These constrained models also tend to show a stronger alignment with human processing metrics. The findings suggest that such constraints may serve as a beneficial inductive bias, guiding models towards more robust linguistic representations, especially in data-limited settings.",
      "published": "2026-04-22T17:14:52Z",
      "abstract_url": "http://arxiv.org/abs/2604.20789v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20789v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DAIRE: A lightweight AI model for real-time detection of Controller Area Network attacks in the Internet of Vehicles",
      "authors": [
        "Shahid Alam",
        "Amina Jameel",
        "Zahida Parveen",
        "Ehab Alnfrawy",
        "Adeela Ashraf",
        "Raza Uddin",
        "Jamal Aqib"
      ],
      "abstract": "The Internet of Vehicles (IoV) is advancing modern transportation by improving safety, efficiency, and intelligence. However, the reliance on the Controller Area Network (CAN) introduces critical security risks, as CAN-based communication is highly vulnerable to cyberattacks. Addressing this challenge, we propose DAIRE (Detecting Attacks in IoV in REal-time), a lightweight machine learning framework designed for real-time detection and classification of CAN attacks. DAIRE is built on a lightweight artificial neural network (ANN) where each layer contains Ni = i x c neurons, with Ni representing the number of neurons in the ith layer and c corresponding to the total number of attack classes. Other hyperparameters are determined empirically to ensure real-time operation. To support the detection and classification of various IoV attacks, such as Denial-of-Service, Fuzzy, and Spoofing, DAIRE employs the sparse categorical cross-entropy loss function and root mean square propagation for loss minimization. In contrast to more resource-intensive architectures, DAIRE leverages a lightweight ANN to reduce computational demands while still delivering strong performance. Experimental results on the CICIoV2024 and Car-Hacking datasets demonstrate DAIRE's effectiveness, achieving an average detection rate of 99.88%, a false positive rate of 0.02%, and an overall accuracy of 99.96%. Furthermore, DAIRE significantly outperforms state-of-the-art approaches in inference speed, with a classification time of just 0.03 ms per sample. These results highlight DAIRE's effectiveness in detecting IoV cyberattacks and its practical suitability for real-time deployment in vehicular systems, underscoring its vital role in strengthening automotive cybersecurity.",
      "published": "2026-04-22T16:58:58Z",
      "abstract_url": "http://arxiv.org/abs/2604.20771v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20771v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation",
      "authors": [
        "Andrew Klearman",
        "Radu Revutchi",
        "Rohin Garg",
        "Rishav Chakravarti",
        "Samuel Marc Denton",
        "Yuan Xue"
      ],
      "abstract": "Retrieval quality is the primary bottleneck for accuracy and robustness in retrieval-augmented generation (RAG). Current evaluation relies on heuristically constructed query sets, which introduce a hidden intrinsic bias. We formalize retrieval evaluation as a statistical estimation problem, showing that metric reliability is fundamentally limited by the evaluation-set construction. We further introduce \\emph{semantic stratification}, which grounds evaluation in corpus structure by organizing documents into an interpretable global space of entity-based clusters and systematically generating queries for missing strata. This yields (1) formal semantic coverage guarantees across retrieval regimes and (2) interpretable visibility into retrieval failure modes. Experiments across multiple benchmarks and retrieval methods validate our framework. The results expose systematic coverage gaps, identify structural signals that explain variance in retrieval performance, and show that stratified evaluation yields more stable and transparent assessments while supporting more trustworthy decision-making than aggregate metrics.",
      "published": "2026-04-22T16:49:30Z",
      "abstract_url": "http://arxiv.org/abs/2604.20763v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20763v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "V-tableR1: Process-Supervised Multimodal Table Reasoning with Critic-Guided Policy Optimization",
      "authors": [
        "Yubo Jiang",
        "Yitong An",
        "Xin Yang",
        "Abudukelimu Wuerkaixi",
        "Xuxin Cheng",
        "Fengying Xie",
        "Zhiguo Jiang",
        "Cao Liu",
        "Ke Zeng",
        "Haopeng Zhang"
      ],
      "abstract": "We introduce V-tableR1, a process-supervised reinforcement learning framework that elicits rigorous, verifiable reasoning from multimodal large language models (MLLMs). Current MLLMs trained solely on final outcomes often treat visual reasoning as a black box, relying on superficial pattern matching rather than performing rigorous multi-step inference. While Reinforcement Learning with Verifiable Rewards could enforce transparent reasoning trajectories, extending it to visual domains remains severely hindered by the ambiguity of grounding abstract logic into continuous pixel space. We solve this by leveraging the deterministic grid structure of tables as an ideal visual testbed. V-tableR1 employs a specialized critic VLM to provide dense, step-level feedback on the explicit visual chain-of-thought generated by a policy VLM. To optimize this system, we propose Process-Guided Direct Alignment Policy Optimization (PGPO), a novel RL algorithm integrating process rewards, decoupled policy constraints, and length-aware dynamic sampling. Extensive evaluations demonstrate that V-tableR1 explicitly penalizes visual hallucinations and shortcut guessing. By fundamentally shifting multimodal inference from black-box pattern matching to verifiable logical derivation, V-tableR1 4B establishes state-of-the-art accuracy among open-source models on complex tabular benchmarks, outperforming models up to 18x its size and improving over its SFT baseline",
      "published": "2026-04-22T16:44:33Z",
      "abstract_url": "http://arxiv.org/abs/2604.20755v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20755v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation",
      "authors": [
        "Dongding Lin",
        "Jian Wang",
        "Yongqi Li",
        "Wenjie Li"
      ],
      "abstract": "Situated conversational recommendation (SCR), which utilizes visual scenes grounded in specific environments and natural language dialogue to deliver contextually appropriate recommendations, has emerged as a promising research direction due to its close alignment with real-world scenarios. Compared to traditional recommendations, SCR requires a deeper understanding of dynamic and implicit user preferences, as the surrounding scene often influences users' underlying interests, while both may evolve across conversations. This complexity significantly impacts the timing and relevance of recommendations. To address this, we propose situated preference reasoning (SiPeR), a novel framework that integrates two core mechanisms: (1) Scene transition estimation, which estimates whether the current scene satisfies user needs, and guides the user toward a more suitable scene when necessary; and (2) Bayesian inverse inference, which leverages the likelihood of multimodal large language models (MLLMs) to predict user preferences about candidate items within the scene. Extensive experiments on two representative benchmarks demonstrate SiPeR's superiority in both recommendation accuracy and response generation quality. The code and data are available at https://github.com/DongdingLin/SiPeR.",
      "published": "2026-04-22T16:39:52Z",
      "abstract_url": "http://arxiv.org/abs/2604.20749v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20749v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "AAC: Admissible-by-Architecture Differentiable Landmark Compression for ALT",
      "authors": [
        "An T. Le",
        "Vien Ngo"
      ],
      "abstract": "We introduce \\textbf{AAC} (Architecturally Admissible Compressor), a differentiable landmark-selection module for ALT (A*, Landmarks, and Triangle inequality) shortest-path heuristics whose outputs are admissible by construction: each forward pass is a row-stochastic mixture of triangle-inequality lower bounds, so the heuristic is admissible for \\emph{every} parameter setting without requiring convergence, calibration, or projection. At deployment, the module reduces to classical ALT on a learned subset, composing end-to-end with neural encoders while preserving the classical toolchain. The construction is the first differentiable instance of the compress-while-preserving-admissibility tradition in classical heuristic search. Under a matched per-vertex memory protocol, we establish that ALT with farthest-point-sampling landmarks (FPS-ALT) has provably near-optimal coverage on metric graphs, leaving at most a few percentage points of headroom for \\emph{any} selector. AAC operates near this ceiling: the gap is $0.9$--$3.9$ percentage points on 9 road networks and ${\\leq}1.3$ percentage points on synthetic graphs, with zero admissibility violations across $1{,}500+$ queries and all logged runs. At matched memory, AAC is also $1.2$--$1.5{\\times}$ faster than FPS-ALT at the median query on DIMACS road networks, amortizing its offline cost within $170$--$1{,}924$ queries. A controlled ablation isolates the binding constraint: training-objective drift under default initialization, not architectural capacity; identity-on-first-$m$ initialization closes the expansion-count gap entirely. We release the module, a reusable matched-memory benchmarking protocol with paired two-one-sided-test (TOST) equivalence and pre-registration, and a reference compressed-differential-heuristics baseline.",
      "published": "2026-04-22T16:31:21Z",
      "abstract_url": "http://arxiv.org/abs/2604.20744v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20744v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Supplement Generation Training for Enhancing Agentic Task Performance",
      "authors": [
        "Young Min Cho",
        "Daniele Bonadiman",
        "Divya Bhargavi",
        "Tamer Alkhouli",
        "Salvatore Romeo",
        "Dongwei Jiang",
        "Khushbu Pahwa",
        "Yubin Ge",
        "Etsuko Ishii",
        "Monica Sunkara",
        "Yi Zhang"
      ],
      "abstract": "Training large foundation models for agentic tasks is increasingly impractical due to the high computational costs, long iteration cycles, and rapid obsolescence as new models are continuously released. Instead of post-training massive models for every new task or domain, we propose Supplement Generation Training (SGT), a more efficient and sustainable strategy. SGT trains a smaller LLM to generate useful supplemental text that, when appended to the original input, helps the larger LLM solve the task more effectively. These lightweight models can dynamically adapt supplements to task requirements, improving performance without modifying the underlying large models. This approach decouples task-specific optimization from large foundation models and enables more flexible, cost-effective deployment of LLM-powered agents in real-world applications.",
      "published": "2026-04-22T16:12:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.20727v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20727v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Tokenised Flow Matching for Hierarchical Simulation Based Inference",
      "authors": [
        "Giovanni Charles",
        "Cosmo Santoni",
        "Seth Flaxman",
        "Elizaveta Semenova"
      ],
      "abstract": "The cost of simulator evaluations is a key practical bottleneck for Simulation Based Inference (SBI). In hierarchical settings with shared global parameters and exchangeable site-level parameters and observations, this structure can be exploited to improve simulation efficiency. Existing hierarchical SBI approaches factorise the posterior yet still simulate across multiple sites per training sample; We instead explore likelihood factorisation (LF) to train from single-site simulations. In LF sampling we learn a per-site neural surrogate of the simulator and then assemble synthetic multi-site observations to amortise inference for the full hierarchical posterior. Building on this, we propose Tokenised Flow Matching for Posterior Estimation (TFMPE), a tokenised flow matching approach that supports function-valued observations through likelihood factorisation. To enable systematic evaluation, we introduce a benchmark for hierarchical SBI. We validate TFMPE on this benchmark and on realistic infectious disease and computational fluid dynamics models, finding well-calibrated posteriors while reducing computational cost.",
      "published": "2026-04-22T16:07:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.20723v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20723v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "COMPASS: COntinual Multilingual PEFT with Adaptive Semantic Sampling",
      "authors": [
        "Noah Flynn"
      ],
      "abstract": "Large language models (LLMs) often exhibit performance disparities across languages, with naive multilingual fine-tuning frequently degrading performance due to negative cross-lingual interference. To address this, we introduce COMPASS (COntinual Multilingual PEFT with Adaptive Semantic Sampling), a novel data-centric framework for adapting LLMs to target languages. COMPASS leverages parameter-efficient fine-tuning (PEFT) by training lightweight, language-specific adapters on a judiciously selected subset of auxiliary multilingual data. The core of our method is a distribution-aware sampling strategy that uses multilingual embeddings and clustering to identify semantic gaps between existing training data and a target usage distribution. By prioritizing auxiliary data from under-represented semantic clusters, COMPASS maximizes positive cross-lingual transfer while minimizing interference. We extend this into a continual learning framework, COMPASS-ECDA, which monitors for data distribution shifts in production and dynamically updates adapters to prevent model staleness, balancing adaptation to new data with the preservation of existing knowledge. Across three different model architectures (Phi-4-Mini, Llama-3.1-8B, and Qwen2.5-7B) and multiple challenging multilingual benchmarks (Global-MMLU, MMLU-ProX), including unseen long-context tasks (OneRuler), we demonstrate that COMPASS consistently outperforms baseline methods guided by linguistic similarity, providing an effective, efficient, and sustainable solution for developing and maintaining high-performing multilingual models in dynamic environments.",
      "published": "2026-04-22T16:07:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.20720v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20720v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Storm Surge Modeling, Bias Correction, Graph Neural Networks, Graph Convolution Networks",
      "authors": [
        "Noujoud Nader",
        "Stefanos Giaremis",
        "Clint Dawson",
        "Carola Kaiser",
        "Karame Mohammadiporshokooh",
        "Hartmut Kaiser"
      ],
      "abstract": "Storm surge forecasting remains a critical challenge in mitigating the impacts of tropical cyclones on coastal regions, particularly given recent trends of rapid intensification and increasing nearshore storm activity. Traditional high fidelity numerical models such as ADCIRC, while robust, are often hindered by inevitable uncertainties arising from various sources. To address these challenges, this study introduces StormNet, a spatio-temporal graph neural network (GNN) designed for bias correction of storm surge forecasts. StormNet integrates graph convolutional (GCN) and graph attention (GAT) mechanisms with long short-term memory (LSTM) components to capture complex spatial and temporal dependencies among water-level gauge stations. The model was trained using historical hurricane data from the U.S. Gulf Coast and evaluated on Hurricane Idalia (2023). Results demonstrate that StormNet can effectively reduce the root mean square error (RMSE) in water-level predictions by more than 70\\% for 48-hour forecasts and above 50\\% for 72-hour forecasts, as well as outperform a sequential LSTM baseline, particularly for longer prediction horizons. The model also exhibits low training time, enhancing its applicability in real-time operational forecasting systems. Overall, StormNet provides a computationally efficient and physically meaningful framework for improving storm surge prediction accuracy and reliability during extreme weather events.",
      "published": "2026-04-22T15:36:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.20688v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20688v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "GRPO-VPS: Enhancing Group Relative Policy Optimization with Verifiable Process Supervision for Effective Reasoning",
      "authors": [
        "Jingyi Wang",
        "Lei Zhu",
        "Tengjin Weng",
        "Song-Li Wu",
        "Haochen Tan",
        "Jierun Chen",
        "Chaofan Tao",
        "Haoli Bai",
        "Lu Hou",
        "Lifeng Shang",
        "Xiao-Ping Zhang"
      ],
      "abstract": "Reinforcement Learning with Verifiable Rewards (RLVR) has advanced the reasoning capabilities of Large Language Models (LLMs) by leveraging direct outcome verification instead of learned reward models. Building on this paradigm, Group Relative Policy Optimization (GRPO) eliminates the need for critic models but suffers from indiscriminate credit assignment for intermediate steps, which limits its ability to identify effective reasoning strategies and incurs overthinking. In this work, we introduce a model-free and verifiable process supervision via probing the model's belief in the correct answer throughout its reasoning trajectory. By segmenting the generation into discrete steps and tracking the conditional probability of the correct answer appended at each segment boundary, we efficiently compute interpretable segment-wise progress measurements to refine GRPO's trajectory-level feedback. This approach enables more targeted and sample-efficient policy updates, while avoiding the need for intermediate supervision derived from costly Monte Carlo rollouts or auxiliary models. Experiments on mathematical and general-domain benchmarks show consistent gains over GRPO across diverse models: up to 2.6-point accuracy improvements and 13.7% reasoning-length reductions on math tasks, and up to 2.4 points and 4% on general-domain tasks, demonstrating strong generalization.",
      "published": "2026-04-22T15:08:58Z",
      "abstract_url": "http://arxiv.org/abs/2604.20659v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20659v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Large Language Models Outperform Humans in Fraud Detection and Resistance to Motivated Investor Pressure",
      "authors": [
        "Nattavudh Powdthavee"
      ],
      "abstract": "Large language models trained on human feedback may suppress fraud warnings when investors arrive already persuaded of a fraudulent opportunity. We tested this in a preregistered experiment across seven leading LLMs and twelve investment scenarios covering legitimate, high-risk, and objectively fraudulent opportunities, combining 3,360 AI advisory conversations with a 1,201-participant human benchmark. Contrary to predictions, motivated investor framing did not suppress AI fraud warnings; if anything, it marginally increased them. Endorsement reversal occurred in fewer than 3 in 1,000 observations. Human advisors endorsed fraudulent investments at baseline rates of 13-14%, versus 0% across all LLMs, and suppressed warnings under pressure at two to four times the AI rate. AI systems currently provide more consistent fraud warnings than lay humans in an identical advisory role.",
      "published": "2026-04-22T15:03:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.20652v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20652v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "econ.GN"
      ]
    },
    {
      "title": "pAI/MSc: ML Theory Research with Humans on the Loop",
      "authors": [
        "Mahmoud Abdelmoneum",
        "Pierfrancesco Beneventano",
        "Tomaso Poggio"
      ],
      "abstract": "We present pAI/MSc, an open-source, customizable, modular multi-agent system for academic research workflows. Our goal is not autonomous scientific ideation, nor fully automated research. It is narrower and more practical: to reduce by orders of magnitude the human steering required to turn a specified hypothesis into a literature-grounded, mathematically established, experimentally supported, submission-oriented manuscript draft. pAI/MSc is built with a current emphasis on machine learning theory and adjacent quantitative fields.",
      "published": "2026-04-22T14:38:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.20622v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20622v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "LayerTracer: A Joint Task-Particle and Vulnerable-Layer Analysis framework for Arbitrary Large Language Model Architectures",
      "authors": [
        "Yuhang Wu",
        "Qinyuan Liu",
        "Qiuyang Zhao",
        "Qingwei Chong"
      ],
      "abstract": "Currently, Large Language Models (LLMs) feature a diversified architectural landscape, including traditional Transformer, GateDeltaNet, and Mamba. However, the evolutionary laws of hierarchical representations, task knowledge formation positions, and network robustness bottleneck mechanisms in various LLM architectures remain unclear, posing core challenges for hybrid architecture design and model optimization. This paper proposes LayerTracer, an architecture-agnostic end-to-end analysis framework compatible with any LLM architecture. By extracting hidden states layer-by-layer and mapping them to vocabulary probability distributions, it achieves joint analysis of task particle localization and layer vulnerability quantification. We define the task particle as the key layer where the target token probability first rises significantly, representing the model's task execution starting point, and the vulnerable layer is defined as the layer with the maximum Jensen-Shannon (JS) divergence between output distributions before and after mask perturbation, reflecting its sensitivity to disturbances. Experiments on models of different parameter scales show that task particles mainly appear in the deep layers of the model regardless of parameter size, while larger-parameter models exhibit stronger hierarchical robustness. LayerTracer provides a scientific basis for layer division, module ratio, and gating switching of hybrid architectures, effectively optimizing model performance. It accurately locates task-effective layers and stability bottlenecks, offering universal support for LLM structure design and interpretability research.",
      "published": "2026-04-22T13:38:15Z",
      "abstract_url": "http://arxiv.org/abs/2604.20556v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20556v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Toward Cross-Lingual Quality Classifiers for Multilingual Pretraining Data Selection",
      "authors": [
        "Yassine Turki",
        "Vinko Sabolčec",
        "Bettina Messmer",
        "Martin Jaggi"
      ],
      "abstract": "As Large Language Models (LLMs) scale, data curation has shifted from maximizing volume to optimizing the signal-to-noise ratio by performing quality filtering. However, for many languages, native high quality data is insufficient to train robust quality classifiers. This work investigates the idea that quality markers in embedding space may show cross-lingual consistency, which would allow high-resource languages to subsidize the filtering of low-resource ones. We evaluate various filtering strategies, including cross-lingual transfer, third quartile sampling (Q3), and retention rate tuning. Our results demonstrate that massive multilingual pooling frequently outperforms monolingual baselines in both rank stability and aggregate accuracy for a 1B model trained on 103B tokens, delivering gains for high resource languages (1.2% increase in aggregate normalized accuracy for French) and matching or exceeding monolingual baselines for low-resource languages. However, we find that scale alone does not guarantee stability. Furthermore, for high-resource languages like French, we show that refining the decision boundary through third quartile sampling (Q3) or tuning the retention rate is necessary to fully leverage the multilingual signal.",
      "published": "2026-04-22T13:31:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.20549v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20549v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Enhancing Research Idea Generation through Combinatorial Innovation and Multi-Agent Iterative Search Strategies",
      "authors": [
        "Shuai Chen",
        "Chengzhi Zhang"
      ],
      "abstract": "Scientific progress depends on the continual generation of innovative re-search ideas. However, the rapid growth of scientific literature has greatly increased the cost of knowledge filtering, making it harder for researchers to identify novel directions. Although existing large language model (LLM)-based methods show promise in research idea generation, the ideas they produce are often repetitive and lack depth. To address this issue, this study proposes a multi-agent iterative planning search strategy inspired by com-binatorial innovation theory. The framework combines iterative knowledge search with an LLM-based multi-agent system to generate, evaluate, and re-fine research ideas through repeated interaction, with the goal of improving idea diversity and novelty. Experiments in the natural language processing domain show that the proposed method outperforms state-of-the-art base-lines in both diversity and novelty. Further comparison with ideas derived from top-tier machine learning conference papers indicates that the quality of the generated ideas falls between that of accepted and rejected papers. These results suggest that the proposed framework is a promising approach for supporting high-quality research idea generation. The source code and dataset used in this paper are publicly available on Github repository: https://github.com/ChenShuai00/MAGenIdeas. The demo is available at https://huggingface.co/spaces/cshuai20/MAGenIdeas.",
      "published": "2026-04-22T13:31:12Z",
      "abstract_url": "http://arxiv.org/abs/2604.20548v1",
      "pdf_url": "https://arxiv.org/pdf/2604.20548v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.DL",
        "cs.IR"
      ]
    }
  ]
};
