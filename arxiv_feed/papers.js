const PAPERS_DATA = {
  "last_updated": "2026-03-02 02:45:40 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems",
      "authors": [
        "Jialiang Fan",
        "Weizhe Xu",
        "Mengyu Liu",
        "Oleg Sokolsky",
        "Insup Lee",
        "Fangxin Kong"
      ],
      "abstract": "Safety-critical task planning in robotic systems remains challenging: classical planners suffer from poor scalability, Reinforcement Learning (RL)-based methods generalize poorly, and base Large Language Models (LLMs) cannot guarantee safety. To address this gap, we propose safety-generalizable large language models, named SafeGen-LLM. SafeGen-LLM can not only enhance the safety satisfaction of task plans but also generalize well to novel safety properties in various domains. We first construct a multi-domain Planning Domain Definition Language 3 (PDDL3) benchmark with explicit safety constraints. Then, we introduce a two-stage post-training framework: Supervised Fine-Tuning (SFT) on a constraint-compliant planning dataset to learn planning syntax and semantics, and Group Relative Policy Optimization (GRPO) guided by fine-grained reward machines derived from formal verification to enforce safety alignment and by curriculum learning to better handle complex tasks. Extensive experiments show that SafeGen-LLM achieves strong safety generalization and outperforms frontier proprietary baselines across multi-domain planning tasks and multiple input formats (e.g., PDDLs and natural language).",
      "published": "2026-02-27T18:06:10Z",
      "abstract_url": "http://arxiv.org/abs/2602.24235v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24235v1",
      "categories": [
        "cs.RO",
        "cs.AI"
      ]
    },
    {
      "title": "An Efficient Unsupervised Federated Learning Approach for Anomaly Detection in Heterogeneous IoT Networks",
      "authors": [
        "Mohsen Tajgardan",
        "Atena Shiranzaei",
        "Mahdi Rabbani",
        "Reza Khoshkangini",
        "Mahtab Jamali"
      ],
      "abstract": "Federated learning (FL) is an effective paradigm for distributed environments such as the Internet of Things (IoT), where data from diverse devices with varying functionalities remains localized while contributing to a shared global model. By eliminating the need to transmit raw data, FL inherently preserves privacy. However, the heterogeneous nature of IoT data, stemming from differences in device capabilities, data formats, and communication constraints, poses significant challenges to maintaining both global model performance and privacy. In the context of IoT-based anomaly detection, unsupervised FL offers a promising means to identify abnormal behavior without centralized data aggregation. Nevertheless, feature heterogeneity across devices complicates model training and optimization, hindering effective implementation. In this study we propose an efficient unsupervised FL framework that enhances anomaly detection by leveraging shared features from two distinct IoT datasets: one focused on anomaly detection and the other on device identification, while preserving dataset-specific features. To improve transparency and interpretability, we employ explainable AI techniques, such as SHAP, to identify key features influencing local model decisions. Experiments conducted on real-world IoT datasets demonstrate that the proposed method significantly outperforms conventional FL approaches in anomaly detection accuracy. This work underscores the potential of using shared features from complementary datasets to optimize unsupervised federated learning and achieve superior anomaly detection results in decentralized IoT environments.",
      "published": "2026-02-27T17:39:04Z",
      "abstract_url": "http://arxiv.org/abs/2602.24209v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24209v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume",
      "authors": [
        "Gregory Kang Ruey Lau",
        "Hieu Dao",
        "Nicole Kan Hui Lin",
        "Bryan Kian Hsiang Low"
      ],
      "abstract": "Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. UMPIRE computes the incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, effectively capturing both the global semantic diversity of samples and the local incoherence of responses based on internal model confidence. We propose uncertainty desiderata for MLLMs and provide theoretical analysis motivating UMPIRE's design. Extensive experiments show that UMPIRE consistently outperforms baseline metrics in error detection and uncertainty calibration across image, audio, and video-text benchmarks, including adversarial and out-of-distribution settings. We also demonstrate UMPIRE's generalization to non-text output tasks, including image and audio generation.",
      "published": "2026-02-27T17:18:42Z",
      "abstract_url": "http://arxiv.org/abs/2602.24195v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24195v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Task-Centric Acceleration of Small-Language Models",
      "authors": [
        "Dor Tsur",
        "Sharon Adar",
        "Ran Levy"
      ],
      "abstract": "Small language models (SLMs) have emerged as efficient alternatives to large language models for task-specific applications. However, they are often employed in high-volume, low-latency settings, where efficiency is crucial. We propose TASC, Task-Adaptive Sequence Compression, a framework for SLM acceleration comprising two use-cases: When performing SLM fine-tuning, we propose TASC-ft, which iteratively enriches the tokenizer vocabulary with high-frequency output n-grams and then fine-tunes the model to utilize the expanded vocabulary. Next, we propose an inference-time method, termed TASC-spec. TASC-spec is a lightweight, training-free speculative decoding method that constructs an n-gram draft model from the task's output corpus, mixing task and context n-gram information.TASC-spec avoids any additional training, while bypassing draft-target vocabulary alignment constraints. We demonstrate the effectiveness of both methods across multiple low output-variability generation tasks. Our methods show consistent improvements in inference efficiency while maintaining task performance.",
      "published": "2026-02-27T16:55:22Z",
      "abstract_url": "http://arxiv.org/abs/2602.24174v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24174v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IT"
      ]
    },
    {
      "title": "LemmaBench: A Live, Research-Level Benchmark to Evaluate LLM Capabilities in Mathematics",
      "authors": [
        "Antoine Peyronnet",
        "Fabian Gloeckle",
        "Amaury Hayat"
      ],
      "abstract": "We present a new approach for benchmarking Large Language Model (LLM) capabilities on research-level mathematics. Existing benchmarks largely rely on static, hand-curated sets of contest or textbook-style problems as proxies for mathematical research. Instead, we establish an updatable benchmark evaluating models directly on the latest research results in mathematics. This consists of an automatic pipeline that extracts lemmas from arXiv and rewrites them into self-contained statements by making all assumptions and required definitions explicit. It results in a benchmark that can be updated regularly with new problems taken directly from human mathematical research, while previous instances can be used for training without compromising future evaluations. We benchmark current state-of-the-art LLMs, which obtain around 10-15$\\%$ accuracy in theorem proving (pass@1) depending on the model, showing that there is currently a large margin of progression for LLMs to reach human-level proving capabilities in a research context.",
      "published": "2026-02-27T16:52:52Z",
      "abstract_url": "http://arxiv.org/abs/2602.24173v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24173v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ArgLLM-App: An Interactive System for Argumentative Reasoning with Large Language Models",
      "authors": [
        "Adam Dejl",
        "Deniz Gorur",
        "Francesca Toni"
      ],
      "abstract": "Argumentative LLMs (ArgLLMs) are an existing approach leveraging Large Language Models (LLMs) and computational argumentation for decision-making, with the aim of making the resulting decisions faithfully explainable to and contestable by humans. Here we propose a web-based system implementing ArgLLM-empowered agents for binary tasks. ArgLLM-App supports visualisation of the produced explanations and interaction with human users, allowing them to identify and contest any mistakes in the system's reasoning. It is highly modular and enables drawing information from trusted external sources. ArgLLM-App is publicly available at https://argllm.app, with a video demonstration at https://youtu.be/vzwlGOr0sPM.",
      "published": "2026-02-27T16:52:27Z",
      "abstract_url": "http://arxiv.org/abs/2602.24172v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24172v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Terminology Rarity Predicts Catastrophic Failure in LLM Translation of Low-Resource Ancient Languages: Evidence from Ancient Greek",
      "authors": [
        "James L. Zainaldin",
        "Cameron Pattison",
        "Manuela Marai",
        "Jacob Wu",
        "Mark J. Schiefsky"
      ],
      "abstract": "This study presents the first systematic, reference-free human evaluation of large language model (LLM) machine translation (MT) for Ancient Greek (AG) technical prose. We evaluate translations by three commercial LLMs (Claude, Gemini, ChatGPT) of twenty paragraph-length passages from two works by the Greek physician Galen of Pergamum (ca. 129-216 CE): On Mixtures, which has two published English translations, and On the Composition of Drugs according to Kinds, which has never been fully translated into English. We assess translation quality using both standard automated evaluation metrics (BLEU, chrF++, METEOR, ROUGE-L, BERTScore, COMET, BLEURT) and expert human evaluation via a modified Multidimensional Quality Metrics (MQM) framework applied to all 60 translations by a team of domain specialists. On the previously translated expository text, LLMs achieved high translation quality (mean MQM score 95.2/100), with performance approaching expert level. On the untranslated pharmacological text, aggregate quality was lower (79.9/100) but with high variance driven by two passages presenting extreme terminological density; excluding these, scores converged to within 4 points of the translated text. Terminology rarity, operationalized via corpus frequency in the literary Diorisis Ancient Greek Corpus, emerged as a strong predictor of translation failure (r = -.97 for passage-level quality on the untranslated text). Automated metrics showed moderate correlation with human judgment overall on the text with a wide quality spread (Composition), but no metric discriminated among high-quality translations. We discuss implications for the use of LLMs in Classical scholarship and for the design of automated evaluation pipelines for low-resource ancient languages.",
      "published": "2026-02-27T15:57:15Z",
      "abstract_url": "http://arxiv.org/abs/2602.24119v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24119v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "ARGUS: Seeing the Influence of Narrative Features on Persuasion in Argumentative Texts",
      "authors": [
        "Sara Nabhani",
        "Federico Pianzola",
        "Khalid Al-Khatib",
        "Malvina Nissim"
      ],
      "abstract": "Can narratives make arguments more persuasive? And to this end, which narrative features matter most? Although stories are often seen as powerful tools for persuasion, their specific role in online, unstructured argumentation remains underexplored. To address this gap, we present ARGUS, a framework for studying the impact of narration on persuasion in argumentative discourse. ARGUS introduces a new ChangeMyView corpus annotated for story presence and six key narrative features, integrating insights from two established theoretical frameworks that capture both textual narrative features and their effects on recipients. Leveraging both encoder-based classifiers and zero-shot large language models (LLMs), ARGUS identifies stories and narrative features and applies them at scale to examine how different narrative dimensions influence persuasion success in online argumentation.",
      "published": "2026-02-27T15:47:57Z",
      "abstract_url": "http://arxiv.org/abs/2602.24109v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24109v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Artificial Agency Program: Curiosity, compression, and communication in agents",
      "authors": [
        "Richard Csaky"
      ],
      "abstract": "This paper presents the Artificial Agency Program (AAP), a position and research agenda for building AI systems as reality embedded, resource-bounded agents whose development is driven by curiosity-as-learning-progress under physical and computational constraints. The central thesis is that AI is most useful when treated as part of an extended human--tool system that increases sensing, understanding, and actuation capability while reducing friction at the interface between people, tools, and environments. The agenda unifies predictive compression, intrinsic motivation, empowerment and control, interface quality (unification), and language/self-communication as selective information bottlenecks. We formulate these ideas as a falsifiable program with explicit costs, staged experiments, and a concrete multimodal tokenized testbed in which an agent allocates limited budget among observation, action, and deliberation. The aim is to provide a conceptual and experimental framework that connects intrinsic motivation, information theory, thermodynamics, bounded rationality, and modern reasoning systems",
      "published": "2026-02-27T15:40:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.24100v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24100v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer",
      "authors": [
        "Yuxuan Zhang",
        "Katarína Tóthová",
        "Zian Wang",
        "Kangxue Yin",
        "Haithem Turki",
        "Riccardo de Lutio",
        "Yen-Yu Chang",
        "Or Litany",
        "Sanja Fidler",
        "Zan Gojcic"
      ],
      "abstract": "Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.",
      "published": "2026-02-27T15:35:30Z",
      "abstract_url": "http://arxiv.org/abs/2602.24096v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24096v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Preference Packing: Efficient Preference Optimization for Large Language Models",
      "authors": [
        "Jaekyung Cho"
      ],
      "abstract": "Resource-efficient training optimization techniques are becoming increasingly important as the size of large language models (LLMs) continues to grow. In particular, batch packing is commonly used in pre-training and supervised fine-tuning to achieve resource-efficient training. We propose preference packing, a method to enhance resource efficiency in training techniques that use data with different responses for the same input prompt, such as reward models or Direct Preference Optimization (DPO). Preference packing improves resource efficiency by reducing the attention operations for duplicate input prompts and decreasing KV cache memory usage. We conducted experiments on text-only datasets and image-included datasets and achieved at least 37% reduction in training time. Notably, this method can be applied alongside existing optimization techniques such as batch sorting, resulting in a 3.22x speedup.",
      "published": "2026-02-27T15:19:57Z",
      "abstract_url": "http://arxiv.org/abs/2602.24082v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24082v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Adaptive Correlation-Weighted Intrinsic Rewards for Reinforcement Learning",
      "authors": [
        "Viet Bac Nguyen",
        "Phuong Thai Nguyen"
      ],
      "abstract": "We propose ACWI (Adaptive Correlation Weighted Intrinsic), an adaptive intrinsic reward scaling framework designed to dynamically balance intrinsic and extrinsic rewards for improved exploration in sparse reward reinforcement learning. Unlike conventional approaches that rely on manually tuned scalar coefficients, which often result in unstable or suboptimal performance across tasks, ACWI learns a state dependent scaling coefficient online. Specifically, ACWI introduces a lightweight Beta Network that predicts the intrinsic reward weight directly from the agent state through an encoder based architecture. The scaling mechanism is optimized using a correlation based objective that encourages alignment between the weighted intrinsic rewards and discounted future extrinsic returns. This formulation enables task adaptive exploration incentives while preserving computational efficiency and training stability. We evaluate ACWI on a suite of sparse reward environments in MiniGrid. Experimental results demonstrate that ACWI consistently improves sample efficiency and learning stability compared to fixed intrinsic reward baselines, achieving superior performance with minimal computational overhead.",
      "published": "2026-02-27T15:16:53Z",
      "abstract_url": "http://arxiv.org/abs/2602.24081v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24081v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Task Complexity Matters: An Empirical Study of Reasoning in LLMs for Sentiment Analysis",
      "authors": [
        "Donghao Huang",
        "Zhaoxia Wang"
      ],
      "abstract": "Large language models (LLMs) with reasoning capabilities have fueled a compelling narrative that reasoning universally improves performance across language tasks. We test this claim through a comprehensive evaluation of 504 configurations across seven model families--including adaptive, conditional, and reinforcement learning-based reasoning architectures--on sentiment analysis datasets of varying granularity (binary, five-class, and 27-class emotion). Our findings reveal that reasoning effectiveness is strongly task-dependent, challenging prevailing assumptions: (1) Reasoning shows task-complexity dependence--binary classification degrades up to -19.9 F1 percentage points (pp), while 27-class emotion recognition gains up to +16.0pp; (2) Distilled reasoning variants underperform base models by 3-18 pp on simpler tasks, though few-shot prompting enables partial recovery; (3) Few-shot learning improves over zero-shot in most cases regardless of model type, with gains varying by architecture and task complexity; (4) Pareto frontier analysis shows base models dominate efficiency-performance trade-offs, with reasoning justified only for complex emotion recognition despite 2.1x-54x computational overhead. We complement these quantitative findings with qualitative error analysis revealing that reasoning degrades simpler tasks through systematic over-deliberation, offering mechanistic insight beyond the high-level overthinking hypothesis.",
      "published": "2026-02-27T14:49:05Z",
      "abstract_url": "http://arxiv.org/abs/2602.24060v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24060v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Data Driven Optimization of GPU efficiency for Distributed LLM Adapter Serving",
      "authors": [
        "Ferran Agullo",
        "Joan Oliveras",
        "Chen Wang",
        "Alberto Gutierrez-Torre",
        "Olivier Tardieu",
        "Alaa Youssef",
        "Jordi Torres",
        "Josep Ll. Berral"
      ],
      "abstract": "Large Language Model (LLM) adapters enable low-cost model specialization, but introduce complex caching and scheduling challenges in distributed serving systems where hundreds of adapters must be hosted concurrently. While prior work has largely focused on latency minimization, resource efficiency through throughput maximization remains underexplored. This paper presents a data-driven pipeline that, for a given workload, computes an adapter placement that serves the workload with the minimum number of GPUs while avoiding request starvation and GPU memory errors. To that end, the approach identifies the maximum feasible throughput attainable on each GPU by leveraging accurate performance predictions learned from real serving behavior. The proposed pipeline integrates three components: (i) a Digital Twin (DT) tailored to LLM-adapter serving, (ii) a distilled machine learning (ML) model trained on DT-generated data, and (iii) a greedy placement algorithm that exploits ML-based performance estimates to maximize GPU efficiency. The DT emulates real system dynamics with high fidelity, achieving below 5% throughput estimation error while executing up to 90 times faster than full LLM benchmarking across both predictable and unpredictable workloads. The learned ML models further accelerate performance estimation with marginal accuracy degradation, enabling scalable optimization. Experimental results demonstrate that the pipeline substantially improves GPU efficiency by reducing the number of GPUs required to sustain target workloads. Beyond GPU efficiency, the pipeline can be adapted to alternative objectives, such as latency minimization, highlighting its versatility for future large-scale LLM serving infrastructures.",
      "published": "2026-02-27T14:22:51Z",
      "abstract_url": "http://arxiv.org/abs/2602.24044v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24044v1",
      "categories": [
        "cs.DC",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "RewardUQ: A Unified Framework for Uncertainty-Aware Reward Models",
      "authors": [
        "Daniel Yang",
        "Samuel Stante",
        "Florian Redhardt",
        "Lena Libon",
        "Parnian Kassraie",
        "Ido Hakimi",
        "Barna Pásztor",
        "Andreas Krause"
      ],
      "abstract": "Reward models are central to aligning large language models (LLMs) with human preferences. Yet most approaches rely on pointwise reward estimates that overlook the epistemic uncertainty in reward models arising from limited human feedback. Recent work suggests that quantifying this uncertainty can reduce the costs of human annotation via uncertainty-guided active learning and mitigate reward overoptimization in LLM post-training. However, uncertainty-aware reward models have so far been adopted without thorough comparison, leaving them poorly understood. This work introduces a unified framework, RewardUQ, to systematically evaluate uncertainty quantification for reward models. We compare common methods along standard metrics measuring accuracy and calibration, and we propose a new ranking strategy incorporating both dimensions for a simplified comparison. Our experimental results suggest that model size and initialization have the most meaningful impact on performance, and most prior work could have benefited from alternative design choices. To foster the development and evaluation of new methods and aid the deployment in downstream applications, we release our open-source framework as a Python package. Our code is available at https://github.com/lasgroup/rewarduq.",
      "published": "2026-02-27T14:15:57Z",
      "abstract_url": "http://arxiv.org/abs/2602.24040v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24040v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking",
      "authors": [
        "Zhicheng Fang",
        "Jingjie Zheng",
        "Chenxu Fu",
        "Wei Xu"
      ],
      "abstract": "Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce JAILBREAK FOUNDRY (JBF), a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) JBF-LIB for shared contracts and reusable utilities; (ii) JBF-FORGE for the multi-agent paper-to-module translation; and (iii) JBF-EVAL for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced-reported) attack success rate (ASR) deviation of +0.26 percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape.",
      "published": "2026-02-27T13:32:53Z",
      "abstract_url": "http://arxiv.org/abs/2602.24009v1",
      "pdf_url": "https://arxiv.org/pdf/2602.24009v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Foundation World Models for Agents that Learn, Verify, and Adapt Reliably Beyond Static Environments",
      "authors": [
        "Florent Delgrange"
      ],
      "abstract": "The next generation of autonomous agents must not only learn efficiently but also act reliably and adapt their behavior in open worlds. Standard approaches typically assume fixed tasks and environments with little or no novelty, which limits world models' ability to support agents that must evolve their policies as conditions change. This paper outlines a vision for foundation world models: persistent, compositional representations that unify reinforcement learning, reactive/program synthesis, and abstraction mechanisms. We propose an agenda built around four components: (i) learnable reward models from specifications to support optimization with clear objectives; (ii) adaptive formal verification integrated throughout learning; (iii) online abstraction calibration to quantify the reliability of the model's predictions; and (iv) test-time synthesis and world-model generation guided by verifiers. Together, these components enable agents to synthesize verifiable programs, derive new policies from a small number of interactions, and maintain correctness while adapting to novelty. The resulting framework positions foundation world models as a substrate for learning, reasoning, and adaptation, laying the groundwork for agents that not only act well but can explain and justify the behavior they adopt.",
      "published": "2026-02-27T13:20:46Z",
      "abstract_url": "http://arxiv.org/abs/2602.23997v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23997v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MINT: Multimodal Imaging-to-Speech Knowledge Transfer for Early Alzheimer's Screening",
      "authors": [
        "Vrushank Ahire",
        "Yogesh Kumar",
        "Anouck Girard",
        "M. A. Ganaie"
      ],
      "abstract": "Alzheimer's disease is a progressive neurodegenerative disorder in which mild cognitive impairment (MCI) marks a critical transition between aging and dementia. Neuroimaging modalities, such as structural MRI, provide biomarkers of this transition; however, their high costs and infrastructure needs limit their deployment at a population scale. Speech analysis offers a non-invasive alternative, but speech-only classifiers are developed independently of neuroimaging, leaving decision boundaries biologically ungrounded and limiting reliability on the subtle CN-versus-MCI distinction. We propose MINT (Multimodal Imaging-to-Speech Knowledge Transfer), a three-stage cross-modal framework that transfers biomarker structure from MRI into a speech encoder at training time. An MRI teacher, trained on 1,228 subjects, defines a compact neuroimaging embedding space for CN-versus-MCI classification. A residual projection head aligns speech representations to this frozen imaging manifold via a combined geometric loss, adapting speech to the learned biomarker space while preserving imaging encoder fidelity. The frozen MRI classifier, which is never exposed to speech, is applied to aligned embeddings at inference and requires no scanner. Evaluation on ADNI-4 shows aligned speech achieves performance comparable to speech-only baselines (AUC 0.720 vs 0.711) while requiring no imaging at inference, demonstrating that MRI-derived decision boundaries can ground speech representations. Multimodal fusion improves over MRI alone (0.973 vs 0.958). Ablation studies identify dropout regularization and self-supervised pretraining as critical design decisions. To our knowledge, this is the first demonstration of MRI-to-speech knowledge transfer for early Alzheimer's screening, establishing a biologically grounded pathway for population-level cognitive triage without neuroimaging at inference.",
      "published": "2026-02-27T13:14:54Z",
      "abstract_url": "http://arxiv.org/abs/2602.23994v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23994v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Intrinsic Lorentz Neural Network",
      "authors": [
        "Xianglong Shi",
        "Ziheng Chen",
        "Yunhan Jiang",
        "Nicu Sebe"
      ],
      "abstract": "Real-world data frequently exhibit latent hierarchical structures, which can be naturally represented by hyperbolic geometry. Although recent hyperbolic neural networks have demonstrated promising results, many existing architectures remain partially intrinsic, mixing Euclidean operations with hyperbolic ones or relying on extrinsic parameterizations. To address it, we propose the \\emph{Intrinsic Lorentz Neural Network} (ILNN), a fully intrinsic hyperbolic architecture that conducts all computations within the Lorentz model. At its core, the network introduces a novel \\emph{point-to-hyperplane} fully connected layer (FC), replacing traditional Euclidean affine logits with closed-form hyperbolic distances from features to learned Lorentz hyperplanes, thereby ensuring that the resulting geometric decision functions respect the inherent curvature. Around this fundamental layer, we design intrinsic modules: GyroLBN, a Lorentz batch normalization that couples gyro-centering with gyro-scaling, consistently outperforming both LBN and GyroBN while reducing training time. We additionally proposed a gyro-additive bias for the FC output, a Lorentz patch-concatenation operator that aligns the expected log-radius across feature blocks via a digamma-based scale, and a Lorentz dropout layer. Extensive experiments conducted on CIFAR-10/100 and two genomic benchmarks (TEB and GUE) illustrate that ILNN achieves state-of-the-art performance and computational cost among hyperbolic models and consistently surpasses strong Euclidean baselines. The code is available at \\href{https://github.com/Longchentong/ILNN}{\\textcolor{magenta}{this url}}.",
      "published": "2026-02-27T12:48:05Z",
      "abstract_url": "http://arxiv.org/abs/2602.23981v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23981v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Ask don't tell: Reducing sycophancy in large language models",
      "authors": [
        "Magda Dubois",
        "Cozmin Ududec",
        "Christopher Summerfield",
        "Lennart Luettgau"
      ],
      "abstract": "Sycophancy, the tendency of large language models to favour user-affirming responses over critical engagement, has been identified as an alignment failure, particularly in high-stakes advisory and social contexts. While prior work has documented conversational features correlated with sycophancy, we lack a systematic understanding of what provokes or prevents AI sycophancy. Here, we present a set of controlled experimental studies where we first isolate how input framing influences sycophancy, and second, leverage these findings to develop mitigation strategies. In a nested factorial design, we compare questions to various non-questions where we vary three orthogonal factors: epistemic certainty (statement, belief, conviction), perspective (I- vs user-perspective), and affirmation vs negation. We show that (1) sycophancy is substantially higher in response to non-questions compared to questions. Additionally, we find that (2) sycophancy increases monotonically with epistemic certainty conveyed by the user, and (3) is amplified by I-perspective framing. Building on this, we show that asking a model to convert non-questions into questions before answering significantly reduces sycophancy. Importantly, this effect is stronger than a simple baseline prompt asking models \"not to be sycophantic\". Our work offers a practical and effective input-level mitigation that both developers and users can easily adopt.",
      "published": "2026-02-27T12:27:04Z",
      "abstract_url": "http://arxiv.org/abs/2602.23971v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23971v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    }
  ]
};
