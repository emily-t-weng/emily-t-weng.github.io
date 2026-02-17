const PAPERS_DATA = {
  "last_updated": "2026-02-16 20:14:50 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Semantic Chunking and the Entropy of Natural Language",
      "authors": [
        "Weishun Zhong",
        "Doron Sivan",
        "Tankut Can",
        "Mikhail Katkov",
        "Misha Tsodyks"
      ],
      "abstract": "The entropy rate of printed English is famously estimated to be about one bit per character, a benchmark that modern large language models (LLMs) have only recently approached. This entropy rate implies that English contains nearly 80 percent redundancy relative to the five bits per character expected for random text. We introduce a statistical model that attempts to capture the intricate multi-scale structure of natural language, providing a first-principles account of this redundancy level. Our model describes a procedure of self-similarly segmenting text into semantically coherent chunks down to the single-word level. The semantic structure of the text can then be hierarchically decomposed, allowing for analytical treatment. Numerical experiments with modern LLMs and open datasets suggest that our model quantitatively captures the structure of real texts at different levels of the semantic hierarchy. The entropy rate predicted by our model agrees with the estimated entropy rate of printed English. Moreover, our theory further reveals that the entropy rate of natural language is not fixed but should increase systematically with the semantic complexity of corpora, which are captured by the only free parameter in our model.",
      "published": "2026-02-13T18:58:10Z",
      "abstract_url": "http://arxiv.org/abs/2602.13194v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13194v1",
      "categories": [
        "cs.CL",
        "cond-mat.dis-nn",
        "cond-mat.stat-mech",
        "cs.AI"
      ]
    },
    {
      "title": "Asynchronous Verified Semantic Caching for Tiered LLM Architectures",
      "authors": [
        "Asmit Kumar Singh",
        "Haozhe Wang",
        "Laxmi Naga Santosh Attaluri",
        "Tak Chiam",
        "Weihua Zhu"
      ],
      "abstract": "Large language models (LLMs) now sit in the critical path of search, assistance, and agentic workflows, making semantic caching essential for reducing inference cost and latency. Production deployments typically use a tiered static-dynamic design: a static cache of curated, offline vetted responses mined from logs, backed by a dynamic cache populated online. In practice, both tiers are commonly governed by a single embedding similarity threshold, which induces a hard tradeoff: conservative thresholds miss safe reuse opportunities, while aggressive thresholds risk serving semantically incorrect responses. We introduce \\textbf{Krites}, an asynchronous, LLM-judged caching policy that expands static coverage without changing serving decisions. On the critical path, Krites behaves exactly like a standard static threshold policy. When the nearest static neighbor of the prompt falls just below the static threshold, Krites asynchronously invokes an LLM judge to verify whether the static response is acceptable for the new prompt. Approved matches are promoted into the dynamic cache, allowing future repeats and paraphrases to reuse curated static answers and expanding static reach over time. In trace-driven simulations on conversational and search workloads, Krites increases the fraction of requests served with curated static answers (direct static hits plus verified promotions) by up to $\\textbf{3.9}$ times for conversational traffic and search-style queries relative to tuned baselines, with unchanged critical path latency.",
      "published": "2026-02-13T18:25:00Z",
      "abstract_url": "http://arxiv.org/abs/2602.13165v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13165v1",
      "categories": [
        "cs.IR",
        "cs.AI"
      ]
    },
    {
      "title": "In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach",
      "authors": [
        "Yiran Gao",
        "Kim Hammar",
        "Tao Li"
      ],
      "abstract": "Rapidly evolving cyberattacks demand incident response systems that can autonomously learn and adapt to changing threats. Prior work has extensively explored the reinforcement learning approach, which involves learning response strategies through extensive simulation of the incident. While this approach can be effective, it requires handcrafted modeling of the simulator and suppresses useful semantics from raw system logs and alerts. To address these limitations, we propose to leverage large language models' (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM (14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action). By comparing LLM-simulated outcomes with actual observations, the LLM agent repeatedly refines its attack conjecture and corresponding response, thereby demonstrating in-context adaptation. Our agentic approach is free of modeling and can run on commodity hardware. When evaluated on incident logs reported in the literature, our agent achieves recovery up to 23% faster than those of frontier LLMs.",
      "published": "2026-02-13T18:09:30Z",
      "abstract_url": "http://arxiv.org/abs/2602.13156v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13156v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "SCOPE: Selective Conformal Optimized Pairwise LLM Judging",
      "authors": [
        "Sher Badshah",
        "Ali Emami",
        "Hassan Sajjad"
      ],
      "abstract": "Large language models (LLMs) are increasingly used as judges to replace costly human preference labels in pairwise evaluation. Despite their practicality, LLM judges remain prone to miscalibration and systematic biases. This paper proposes SCOPE (Selective Conformal Optimized Pairwise Evaluation), a framework for selective pairwise judging with finite-sample statistical guarantees. Under exchangeability, SCOPE calibrates an acceptance threshold such that the error rate among non-abstained judgments is at most a user-specified level $α$. To provide SCOPE with a bias-neutral uncertainty signal, we introduce Bidirectional Preference Entropy (BPE), which queries the judge under both response positions, aggregates the implied preference probabilities to enforce invariance to response order, and converts the aggregated probability into an entropy-based uncertainty score. Across MT-Bench, RewardBench, and Chatbot Arena, BPE improves uncertainty quality over standard confidence proxies, providing a stronger selection signal that enables SCOPE to consistently meet the target risk level while retaining good coverage across judge scales. In particular, at $α= 0.10$, \\textsc{Scope} consistently satisfies the risk bound across all benchmarks and judge scales (empirical risk $\\approx 0.097$ to $0.099$), while retaining substantial coverage, reaching $0.89$ on RewardBench with Qwen-14B and $0.98$ on RewardBench with Qwen-32B. Compared to naïve baselines, \\textsc{Scope} accepts up to $2.4\\times$ more judgments on MT-Bench with Qwen-7B under the same target risk constraint, demonstrating that BPE enables reliable and high-coverage LLM-based evaluation.",
      "published": "2026-02-13T17:10:43Z",
      "abstract_url": "http://arxiv.org/abs/2602.13110v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13110v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Which Algorithms Can Graph Neural Networks Learn?",
      "authors": [
        "Solveig Wittig",
        "Antonis Vasileiou",
        "Robert R. Nerem",
        "Timo Stoll",
        "Floris Geerts",
        "Yusu Wang",
        "Christopher Morris"
      ],
      "abstract": "In recent years, there has been growing interest in understanding neural architectures' ability to learn to execute discrete algorithms, a line of work often referred to as neural algorithmic reasoning. The goal is to integrate algorithmic reasoning capabilities into larger neural pipelines. Many such architectures are based on (message-passing) graph neural networks (MPNNs), owing to their permutation equivariance and ability to deal with sparsity and variable-sized inputs. However, existing work is either largely empirical and lacks formal guarantees or it focuses solely on expressivity, leaving open the question of when and how such architectures generalize beyond a finite training set. In this work, we propose a general theoretical framework that characterizes the sufficient conditions under which MPNNs can learn an algorithm from a training set of small instances and provably approximate its behavior on inputs of arbitrary size. Our framework applies to a broad class of algorithms, including single-source shortest paths, minimum spanning trees, and general dynamic programming problems, such as the $0$-$1$ knapsack problem. In addition, we establish impossibility results for a wide range of algorithmic tasks, showing that standard MPNNs cannot learn them, and we derive more expressive MPNN-like architectures that overcome these limitations. Finally, we refine our analysis for the Bellman-Ford algorithm, yielding a substantially smaller required training set and significantly extending the recent work of Nerem et al. [2025] by allowing for a differentiable regularization loss. Empirical results largely support our theoretical findings.",
      "published": "2026-02-13T17:09:50Z",
      "abstract_url": "http://arxiv.org/abs/2602.13106v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13106v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DS",
        "cs.NE"
      ]
    },
    {
      "title": "EXCODER: EXplainable Classification Of DiscretE time series Representations",
      "authors": [
        "Yannik Hahn",
        "Antonin Königsfeld",
        "Hasan Tercan",
        "Tobias Meisen"
      ],
      "abstract": "Deep learning has significantly improved time series classification, yet the lack of explainability in these models remains a major challenge. While Explainable AI (XAI) techniques aim to make model decisions more transparent, their effectiveness is often hindered by the high dimensionality and noise present in raw time series data. In this work, we investigate whether transforming time series into discrete latent representations-using methods such as Vector Quantized Variational Autoencoders (VQ-VAE) and Discrete Variational Autoencoders (DVAE)-not only preserves but enhances explainability by reducing redundancy and focusing on the most informative patterns. We show that applying XAI methods to these compressed representations leads to concise and structured explanations that maintain faithfulness without sacrificing classification performance. Additionally, we propose Similar Subsequence Accuracy (SSA), a novel metric that quantitatively assesses the alignment between XAI-identified salient subsequences and the label distribution in the training data. SSA provides a systematic way to validate whether the features highlighted by XAI methods are truly representative of the learned classification patterns. Our findings demonstrate that discrete latent representations not only retain the essential characteristics needed for classification but also offer a pathway to more compact, interpretable, and computationally efficient explanations in time series analysis.",
      "published": "2026-02-13T16:47:45Z",
      "abstract_url": "http://arxiv.org/abs/2602.13087v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13087v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Bus-Conditioned Zero-Shot Trajectory Generation via Task Arithmetic",
      "authors": [
        "Shuai Liu",
        "Ning Cao",
        "Yile Chen",
        "Yue Jiang",
        "Gao Cong"
      ],
      "abstract": "Mobility trajectory data provide essential support for smart city applications. However, such data are often difficult to obtain. Meanwhile, most existing trajectory generation methods implicitly assume that at least a subset of real mobility data from target city is available, which limits their applicability in data-inaccessible scenarios. In this work, we propose a new problem setting, called bus-conditioned zero-shot trajectory generation, where no mobility trajectories from a target city are accessible. The generation process relies solely on source city mobility data and publicly available bus timetables from both cities. Under this setting, we propose MobTA, the first approach to introduce task arithmetic into trajectory generation. MobTA models the parameter shift from bus-timetable-based trajectory generation to mobility trajectory generation in source city, and applies this shift to target city through arithmetic operations on task vectors. This enables trajectory generation that reflects target-city mobility patterns without requiring any real mobility data from it. Furthermore, we theoretically analyze MobTA's stability across base and instruction-tuned LLMs. Extensive experiments show that MobTA significantly outperforms existing methods, and achieves performance close to models finetuned using target city mobility trajectories.",
      "published": "2026-02-13T16:30:21Z",
      "abstract_url": "http://arxiv.org/abs/2602.13071v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13071v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Diverging Flows: Detecting Extrapolations in Conditional Generation",
      "authors": [
        "Constantinos Tsakonas",
        "Serena Ivaldi",
        "Jean-Baptiste Mouret"
      ],
      "abstract": "The ability of Flow Matching (FM) to model complex conditional distributions has established it as the state-of-the-art for prediction tasks (e.g., robotics, weather forecasting). However, deployment in safety-critical settings is hindered by a critical extrapolation hazard: driven by smoothness biases, flow models yield plausible outputs even for off-manifold conditions, resulting in silent failures indistinguishable from valid predictions. In this work, we introduce Diverging Flows, a novel approach that enables a single model to simultaneously perform conditional generation and native extrapolation detection by structurally enforcing inefficient transport for off-manifold inputs. We evaluate our method on synthetic manifolds, cross-domain style transfer, and weather temperature forecasting, demonstrating that it achieves effective detection of extrapolations without compromising predictive fidelity or inference latency. These results establish Diverging Flows as a robust solution for trustworthy flow models, paving the way for reliable deployment in domains such as medicine, robotics, and climate science.",
      "published": "2026-02-13T16:15:58Z",
      "abstract_url": "http://arxiv.org/abs/2602.13061v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13061v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Curriculum-DPO++: Direct Preference Optimization via Data and Model Curricula for Text-to-Image Generation",
      "authors": [
        "Florinel-Alin Croitoru",
        "Vlad Hondru",
        "Radu Tudor Ionescu",
        "Nicu Sebe",
        "Mubarak Shah"
      ],
      "abstract": "Direct Preference Optimization (DPO) has been proposed as an effective and efficient alternative to reinforcement learning from human feedback (RLHF). However, neither RLHF nor DPO take into account the fact that learning certain preferences is more difficult than learning other preferences, rendering the optimization process suboptimal. To address this gap in text-to-image generation, we recently proposed Curriculum-DPO, a method that organizes image pairs by difficulty. In this paper, we introduce Curriculum-DPO++, an enhanced method that combines the original data-level curriculum with a novel model-level curriculum. More precisely, we propose to dynamically increase the learning capacity of the denoising network as training advances. We implement this capacity increase via two mechanisms. First, we initialize the model with only a subset of the trainable layers used in the original Curriculum-DPO. As training progresses, we sequentially unfreeze layers until the configuration matches the full baseline architecture. Second, as the fine-tuning is based on Low-Rank Adaptation (LoRA), we implement a progressive schedule for the dimension of the low-rank matrices. Instead of maintaining a fixed capacity, we initialize the low-rank matrices with a dimension significantly smaller than that of the baseline. As training proceeds, we incrementally increase their rank, allowing the capacity to grow until it converges to the same rank value as in Curriculum-DPO. Furthermore, we propose an alternative ranking strategy to the one employed by Curriculum-DPO. Finally, we compare Curriculum-DPO++ against Curriculum-DPO and other state-of-the-art preference optimization approaches on nine benchmarks, outperforming the competing methods in terms of text alignment, aesthetics and human preference. Our code is available at https://github.com/CroitoruAlin/Curriculum-DPO.",
      "published": "2026-02-13T16:09:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.13055v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13055v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Geometric Manifold Rectification for Imbalanced Learning",
      "authors": [
        "Xubin Wang",
        "Qing Li",
        "Weijia Jia"
      ],
      "abstract": "Imbalanced classification presents a formidable challenge in machine learning, particularly when tabular datasets are plagued by noise and overlapping class boundaries. From a geometric perspective, the core difficulty lies in the topological intrusion of the majority class into the minority manifold, which obscures the true decision boundary. Traditional undersampling techniques, such as Edited Nearest Neighbours (ENN), typically employ symmetric cleaning rules and uniform voting, failing to capture the local manifold structure and often inadvertently removing informative minority samples. In this paper, we propose GMR (Geometric Manifold Rectification), a novel framework designed to robustly handle imbalanced structured data by exploiting local geometric priors. GMR makes two contributions: (1) Geometric confidence estimation that uses inverse-distance weighted kNN voting with an adaptive distance metric to capture local reliability; and (2) asymmetric cleaning that is strict on majority samples while conservatively protecting minority samples via a safe-guarding cap on minority removal. Extensive experiments on multiple benchmark datasets show that GMR is competitive with strong sampling baselines.",
      "published": "2026-02-13T15:59:32Z",
      "abstract_url": "http://arxiv.org/abs/2602.13045v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13045v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL",
      "authors": [
        "Yixiao Zhou",
        "Yang Li",
        "Dongzhou Cheng",
        "Hehe Fan",
        "Yu Cheng"
      ],
      "abstract": "Reinforcement Learning from Verifiable Rewards (RLVR) trains large language models (LLMs) from sampled trajectories, making decoding strategy a core component of learning rather than a purely inference-time choice. Sampling temperature directly controls the exploration--exploitation trade-off by modulating policy entropy, yet existing methods rely on static values or heuristic adaptations that are decoupled from task-level rewards. We propose Introspective LLM, a hierarchical reinforcement learning framework that learns to control sampling temperature during generation. At each decoding step, the model selects a temperature based on its hidden state and samples the next token from the resulting distribution. Temperature and token policies are jointly optimized from downstream rewards using a coordinate ascent scheme. Experiments on mathematical reasoning benchmarks show that learned temperature policies outperform fixed and heuristic baselines, while exhibiting interpretable exploration behaviors aligned with reasoning uncertainty.",
      "published": "2026-02-13T15:42:59Z",
      "abstract_url": "http://arxiv.org/abs/2602.13035v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13035v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Buy versus Build an LLM: A Decision Framework for Governments",
      "authors": [
        "Jiahao Lu",
        "Ziwei Xu",
        "William Tjhi",
        "Junnan Li",
        "Antoine Bosselut",
        "Pang Wei Koh",
        "Mohan Kankanhalli"
      ],
      "abstract": "Large Language Models (LLMs) represent a new frontier of digital infrastructure that can support a wide range of public-sector applications, from general purpose citizen services to specialized and sensitive state functions. When expanding AI access, governments face a set of strategic choices over whether to buy existing services, build domestic capabilities, or adopt hybrid approaches across different domains and use cases. These are critical decisions especially when leading model providers are often foreign corporations, and LLM outputs are increasingly treated as trusted inputs to public decision-making and public discourse. In practice, these decisions are not intended to mandate a single approach across all domains; instead, national AI strategies are typically pluralistic, with sovereign, commercial and open-source models coexisting to serve different purposes. Governments may rely on commercial models for non-sensitive or commodity tasks, while pursuing greater control for critical, high-risk or strategically important applications. This paper provides a strategic framework for making this decision by evaluating these options across dimensions including sovereignty, safety, cost, resource capability, cultural fit, and sustainability. Importantly, \"building\" does not imply that governments must act alone: domestic capabilities may be developed through public research institutions, universities, state-owned enterprises, joint ventures, or broader national ecosystems. By detailing the technical requirements and practical challenges of each pathway, this work aims to serve as a reference for policy-makers to determine whether a buy or build approach best aligns with their specific national needs and societal goals.",
      "published": "2026-02-13T15:39:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.13033v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13033v1",
      "categories": [
        "cs.CY",
        "cs.AI",
        "cs.CE",
        "cs.CL",
        "cs.SI"
      ]
    },
    {
      "title": "Prior-Guided Symbolic Regression: Towards Scientific Consistency in Equation Discovery",
      "authors": [
        "Jing Xiao",
        "Xinhai Chen",
        "Jiaming Peng",
        "Qinglin Wang",
        "Menghan Jia",
        "Zhiquan Lai",
        "Guangping Yu",
        "Dongsheng Li",
        "Tiejun Li",
        "Jie Liu"
      ],
      "abstract": "Symbolic Regression (SR) aims to discover interpretable equations from observational data, with the potential to reveal underlying principles behind natural phenomena. However, existing approaches often fall into the Pseudo-Equation Trap: producing equations that fit observations well but remain inconsistent with fundamental scientific principles. A key reason is that these approaches are dominated by empirical risk minimization, lacking explicit constraints to ensure scientific consistency. To bridge this gap, we propose PG-SR, a prior-guided SR framework built upon a three-stage pipeline consisting of warm-up, evolution, and refinement. Throughout the pipeline, PG-SR introduces a prior constraint checker that explicitly encodes domain priors as executable constraint programs, and employs a Prior Annealing Constrained Evaluation (PACE) mechanism during the evolution stage to progressively steer discovery toward scientifically consistent regions. Theoretically, we prove that PG-SR reduces the Rademacher complexity of the hypothesis space, yielding tighter generalization bounds and establishing a guarantee against pseudo-equations. Experimentally, PG-SR outperforms state-of-the-art baselines across diverse domains, maintaining robustness to varying prior quality, noisy data, and data scarcity.",
      "published": "2026-02-13T15:26:21Z",
      "abstract_url": "http://arxiv.org/abs/2602.13021v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13021v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Synaptic Activation and Dual Liquid Dynamics for Interpretable Bio-Inspired Models",
      "authors": [
        "Mónika Farsang",
        "Radu Grosu"
      ],
      "abstract": "In this paper, we present a unified framework for various bio-inspired models to better understand their structural and functional differences. We show that liquid-capacitance-extended models lead to interpretable behavior even in dense, all-to-all recurrent neural network (RNN) policies. We further demonstrate that incorporating chemical synapses improves interpretability and that combining chemical synapses with synaptic activation yields the most accurate and interpretable RNN models. To assess the accuracy and interpretability of these RNN policies, we consider the challenging lane-keeping control task and evaluate performance across multiple metrics, including turn-weighted validation loss, neural activity during driving, absolute correlation between neural activity and road trajectory, saliency maps of the networks' attention, and the robustness of their saliency maps measured by the structural similarity index.",
      "published": "2026-02-13T15:23:37Z",
      "abstract_url": "http://arxiv.org/abs/2602.13017v1",
      "pdf_url": "https://arxiv.org/pdf/2602.13017v1",
      "categories": [
        "cs.NE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models",
      "authors": [
        "Hao Chen",
        "Ye He",
        "Yuchun Fan",
        "Yukun Yan",
        "Zhenghao Liu",
        "Qingfu Zhu",
        "Maosong Sun",
        "Wanxiang Che"
      ],
      "abstract": "Knowledge augmentation has significantly enhanced the performance of Large Language Models (LLMs) in knowledge-intensive tasks. However, existing methods typically operate on the simplistic premise that model performance equates with internal knowledge, overlooking the knowledge-confidence gaps that lead to overconfident errors or uncertain truths. To bridge this gap, we propose a novel meta-cognitive framework for reliable knowledge augmentation via differentiated intervention and alignment. Our approach leverages internal cognitive signals to partition the knowledge space into mastered, confused, and missing regions, guiding targeted knowledge expansion. Furthermore, we introduce a cognitive consistency mechanism to synchronize subjective certainty with objective accuracy, ensuring calibrated knowledge boundaries. Extensive experiments demonstrate the our framework consistently outperforms strong baselines, validating its rationality in not only enhancing knowledge capabilities but also fostering cognitive behaviors that better distinguish knowns from unknowns.",
      "published": "2026-02-13T15:07:35Z",
      "abstract_url": "http://arxiv.org/abs/2602.12996v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12996v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Drift-Aware Variational Autoencoder-based Anomaly Detection with Two-level Ensembling",
      "authors": [
        "Jin Li",
        "Kleanthis Malialis",
        "Christos G. Panayiotou",
        "Marios M. Polycarpou"
      ],
      "abstract": "In today's digital world, the generation of vast amounts of streaming data in various domains has become ubiquitous. However, many of these data are unlabeled, making it challenging to identify events, particularly anomalies. This task becomes even more formidable in nonstationary environments where model performance can deteriorate over time due to concept drift. To address these challenges, this paper presents a novel method, VAE++ESDD, which employs incremental learning and two-level ensembling: an ensemble of Variational AutoEncoder(VAEs) for anomaly prediction, along with an ensemble of concept drift detectors. Each drift detector utilizes a statistical-based concept drift mechanism. To evaluate the effectiveness of VAE++ESDD, we conduct a comprehensive experimental study using real-world and synthetic datasets characterized by severely or extremely low anomalous rates and various drift characteristics. Our study reveals that the proposed method significantly outperforms both strong baselines and state-of-the-art methods.",
      "published": "2026-02-13T14:53:56Z",
      "abstract_url": "http://arxiv.org/abs/2602.12976v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12976v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Extending confidence calibration to generalised measures of variation",
      "authors": [
        "Andrew Thompson",
        "Vivek Desai"
      ],
      "abstract": "We propose the Variation Calibration Error (VCE) metric for assessing the calibration of machine learning classifiers. The metric can be viewed as an extension of the well-known Expected Calibration Error (ECE) which assesses the calibration of the maximum probability or confidence. Other ways of measuring the variation of a probability distribution exist which have the advantage of taking into account the full probability distribution, for example the Shannon entropy. We show how the ECE approach can be extended from assessing confidence calibration to assessing the calibration of any metric of variation. We present numerical examples upon synthetic predictions which are perfectly calibrated by design, demonstrating that, in this scenario, the VCE has the desired property of approaching zero as the number of data samples increases, in contrast to another entropy-based calibration metric (the UCE) which has been proposed in the literature.",
      "published": "2026-02-13T14:49:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.12975v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12975v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TriGen: NPU Architecture for End-to-End Acceleration of Large Language Models based on SW-HW Co-Design",
      "authors": [
        "Jonghun Lee",
        "Junghoon Lee",
        "Hyeonjin Kim",
        "Seoho Jeon",
        "Jisup Yoon",
        "Hyunbin Park",
        "Meejeong Park",
        "Heonjae Ha"
      ],
      "abstract": "Recent studies have extensively explored NPU architectures for accelerating AI inference in on-device environments, which are inherently resource-constrained. Meanwhile, transformer-based large language models (LLMs) have become dominant, with rapidly increasing model sizes but low degree of parameter reuse compared to conventional CNNs, making end-to-end execution on resource-limited devices extremely challenging. To address these challenges, we propose TriGen, a novel NPU architecture tailored for resource-constrained environments through software-hardware co-design. Firstly, TriGen adopts low-precision computation using microscaling (MX) to enable additional optimization opportunities while preserving accuracy, and resolves the issues that arise by employing such precision. Secondly, to jointly optimize both nonlinear and linear operations, TriGen eliminates the need for specialized hardware for essential nonlinear operations by using fast and accurate LUT, thereby maximizing performance gains and reducing hardware-cost in on-device environments, and finally, by taking practical hardware constraints into account, further employs scheduling techniques to maximize computational utilization even under limited on-chip memory capacity. We evaluate the performance of TriGen on various LLMs and show that TriGen achieves an average 2.73x performance speedup and 52% less memory transfer over the baseline NPU design with negligible accuracy loss.",
      "published": "2026-02-13T14:28:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.12962v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12962v1",
      "categories": [
        "cs.AR",
        "cs.AI"
      ]
    },
    {
      "title": "Transporting Task Vectors across Different Architectures without Training",
      "authors": [
        "Filippo Rinaldi",
        "Aniello Panariello",
        "Giacomo Salici",
        "Angelo Porrello",
        "Simone Calderara"
      ],
      "abstract": "Adapting large pre-trained models to downstream tasks often produces task-specific parameter updates that are expensive to relearn for every model variant. While recent work has shown that such updates can be transferred between models with identical architectures, transferring them across models of different widths remains largely unexplored. In this work, we introduce Theseus, a training-free method for transporting task-specific updates across heterogeneous models. Rather than matching parameters directly, we characterize a task update by the functional effect it induces on intermediate representations. We formalize task-vector transport as a functional matching problem on observed activations and show that, after aligning representation spaces via orthogonal Procrustes analysis, it admits a stable closed-form solution that preserves the geometry of the update. We evaluate Theseus on vision and language models across different widths, showing consistent improvements over strong baselines without additional training or backpropagation. Our results show that task updates can be meaningfully transferred across architectures when task identity is defined functionally rather than parametrically.",
      "published": "2026-02-13T14:16:34Z",
      "abstract_url": "http://arxiv.org/abs/2602.12952v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12952v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions",
      "authors": [
        "Fox Pettersen",
        "Hong Zhu"
      ],
      "abstract": "As self-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients (AFFC) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena (i.e., decline in robustness) if overtrained.",
      "published": "2026-02-13T13:02:44Z",
      "abstract_url": "http://arxiv.org/abs/2602.12902v1",
      "pdf_url": "https://arxiv.org/pdf/2602.12902v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    }
  ]
};
