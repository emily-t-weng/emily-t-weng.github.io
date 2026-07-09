const PAPERS_DATA = {
  "last_updated": "2026-07-09 03:59:37 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Accurate, Interdisciplinary and Transparent Structure-property Understanding with Deep Native Structural Reasoning",
      "authors": [
        "Chen Tang",
        "Yizhou Wang",
        "Jianyu Wu",
        "Lintao Wang",
        "Shixiang Tang",
        "Pengze Li",
        "Encheng Su",
        "Jun Yao",
        "Jiabei Xiao",
        "Yuqi Shi",
        "Jielan Li",
        "Hongxia Hao",
        "Zhangyang Gao",
        "Fang Wu",
        "Ben Fei",
        "Xiangyu Yue",
        "Pan Tan",
        "Bozitao Zhong",
        "Jinouwen Zhang",
        "Aoran Wang",
        "Yan Lu",
        "Jiaheng Liu",
        "Xinzhu Ma",
        "Liang Hong",
        "Mingyue Zheng",
        "Phil Torr",
        "Bowen Zhou",
        "Wanli Ouyang",
        "Lei Bai"
      ],
      "abstract": "Structure-property relationships are foundational to biology, chemistry and materials science, where function, reactivity and physical response emerge from spatial, chemical and periodic organization. Mechanistically explaining these relationships requires interpreting structural evidence through scientific principles and physical constraints, from stereochemistry and bonding to symmetry, energetics and periodic order. However, applying artificial intelligence to this process presents a joint challenge of representation and reasoning: models must preserve domain-native structural information while showing how specific evidence supports predictions under these constraints. Here we introduce SciReasoner, a multimodal scientific foundation model for native structural reasoning across proteins, small molecules and inorganic crystals. SciReasoner discretizes coordinates, topologies and periodic connectivities into a unified structure-aware vocabulary, treating structural tokens as addressable evidence units during reasoning. In homology-controlled Gene Ontology prediction, SciReasoner improves Cellular Component annotation for low-homology and orphan-like proteins, increasing $F_{\\max}$ from 0.42 to 0.55. In chemistry, it raises single-step retrosynthesis accuracy from 0.63 to 0.72 while generating fragment-level disconnection and precursor-verification traces. In materials science, its representations separate elemental and compound phases and resolve high- and low-band-gap regimes. Across 86 benchmarks, SciReasoner achieves state-of-the-art performance on 67 tasks. Double-blind expert evaluation rates its reasoning traces as preferred or at least comparable to those of a frontier large language model in 98% of cases. By making structure an inspectable substrate for reasoning under scientific constraints, SciReasoner connects accurate prediction with interpretable scientific inference.",
      "published": "2026-07-08T17:59:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.07708v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07708v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CE",
        "cs.LG"
      ]
    },
    {
      "title": "Co-LMLM: Continuous-Query Limited Memory Language Models",
      "authors": [
        "Yair Feldman",
        "Linxi Zhao",
        "Nathan Godey",
        "Dongyoung Go",
        "Yilun Hua",
        "Kilian Q. Weinberger",
        "Jennifer J. Sun",
        "Yoav Artzi"
      ],
      "abstract": "Limited memory language models (LMLMs) externalize factual knowledge during pretraining to a knowledge base (KB), rather than memorizing it in their weights. During generation, the model then fetches knowledge from the KB as needed. This recently introduced paradigm provides multiple advantages, including knowledge control capabilities that remain beyond conventional LLMs. We propose continuous-query LMLM (CO-LMLM), where the KB pairs continuous keys with textual knowledge values, a significant departure from prior reliance on relational KB and queries. CO-LMLM generates flexible vector queries at minimal cost, while still integrating human-readable and attributable retrieved knowledge into its generation. We pair this design with an annotation pipeline that tags free-form factual spans in arbitrary text, removing prior work's restriction to Wikipedia. Across pretraining on Wikipedia and FineWeb-Edu and at multiple model scales, CO-LMLM outperforms prior LMLMs and vanilla LLMs in both perplexity and factual precision. At 360M scale, this includes lower perplexity than models pretrained on 40x more data, and SimpleQA-verified performance that is in line with gpt-4o-mini and higher than Claude Sonnet 4.5.",
      "published": "2026-07-08T17:59:45Z",
      "abstract_url": "http://arxiv.org/abs/2607.07707v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07707v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Breaking Database Lock-in: Agentic Regeneration of High Performance Storage Readers for Database Bypass",
      "authors": [
        "Victor Giannakouris",
        "Immanuel Trummer"
      ],
      "abstract": "Analytical workloads operating on data stored in external database systems face a fundamental bottleneck: data access is guarded entirely by the database driver, like JDBC or ODBC, forcing all reads through query execution and other driver layers that are not designed for bulk columnar analytics. We present Jailbreak, an approach that bypasses the database engine entirely by reading storage files directly and materializing data as in-memory columnar buffers. Jailbreak's key insight is that database file formats, while complex, are fully specified by their source code and documentation, artifacts that Large Language Models (LLMs) can ingest to regenerate operator-specific table reading components without human-engineered parsing logic. Jailbreak leverages LLM-assisted code synthesis for database storage decoding, turning a traditionally opaque format into a directly queryable artifact. We evaluate Jailbreak on PostgreSQL and MySQL storage files, targeting analytical snapshot scenarios common in read replicas and offline processing pipelines. The generated reader produces Apache Arrow buffers consumable directly by most of the widely known query engines, including DuckDB, Apache Spark, and GPU-accelerated frameworks such as cuDF and Spark RAPIDS. We validate correctness against JDBC/ODBC-based baselines using the TPC-H benchmark across all query results, and demonstrate significant performance improvements in end-to-end analytical throughput, achieving up to 27x speedups. Our results showcase that LLM-assisted storage reader synthesis is a viable and generalizable methodology for breaking data lock-in across database systems, with applications beyond PostgreSQL and MySQL for any system whose file format is available to the LLM from documentation or source code.",
      "published": "2026-07-08T17:55:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.07696v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07696v1",
      "categories": [
        "cs.DB",
        "cs.AI"
      ]
    },
    {
      "title": "Selective Timestep Weighting and Advantage-Based Replay for Sample-Efficient Diffusion RLHF",
      "authors": [
        "Eric Zhu",
        "Abhinav Shrivastava",
        "Soumik Mukhopadhyay"
      ],
      "abstract": "Reinforcement learning from human feedback (RLHF) has emerged as a powerful paradigm for aligning generative models with human preferences. However, applying RLHF to diffusion models remains highly feedback inefficient, as existing approaches typically require large amounts of human or reward model evaluations. This limitation reduces the practicality of diffusion RLHF in realworld settings where feedback is the primary bottleneck. In this paper, we propose two complementary strategies that substantially improve the feedback efficiency of diffusion RLHF while preserving generalization to unseen prompts. Our key observation is that reward information in diffusion trajectories is unevenly distributed: not all denoising timesteps or trajectories contribute equally to learning from a reward signal. By emphasizing informative timesteps and trajectories during optimization, we obtain more effective gradient updates. First, we introduce a per-timestep weighting scheme that reweights denoising steps during policy optimization. We theoretically connect this weighting to the optimal convergence properties of proximal policy optimization (PPO) and approximate the resulting weighting trend empirically. Second, we introduce a replay mechanism that prioritizes informative trajectories, enabling the model to reuse past samples instead of repeatedly querying new rewards. Together, these strategies significantly improve the feedback efficiency of diffusion RLHF. Under identical hyperparameter settings, our approach achieves up to a 6$\\times$ improvement in sample efficiency compared to widely used diffusion RLHF baselines.",
      "published": "2026-07-08T17:49:49Z",
      "abstract_url": "http://arxiv.org/abs/2607.07693v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07693v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Agon: Competitive Cross-Model RL with Implicit Rival Grading of Reasoning",
      "authors": [
        "Vladislav Beliaev"
      ],
      "abstract": "Reinforcement learning from verifiable rewards (e.g. GRPO) is the engine behind today's reasoning models, yet it grades only the final answer. On hard problems this trains models to write more rather than to think better, since the trace itself is never graded and no label for good thinking exists. We introduce Agon, which makes two competing models each other's graders. Both attempt the same problem; in alternating roles, one drafts a solution and the other reads it while solving, and each is rewarded for out-solving the other. To win, a model must out-reason a rival that has seen its work, so reasoning is judged implicitly during training, with no process labels and no reward model. Because both models are optimized, each faces a progressively stronger rival, which single-model RL cannot provide. The two need only be comparably strong and behaviorally different. At inference the pair deploys as it trains, a two-stage cascade in which one model drafts and the other answers after reading the draft. On the hard split of DeepMath with Qwen3, this doubles GRPO's pass@1, roughly eight times the gain of an untrained Mixture-of-Agents pass over the same base. The ordering replicates on competitive-programming code and across model families (Qwen3.5, Gemma 4). For now the models talk in text; the next step is to let them reason together in latent space.",
      "published": "2026-07-08T17:49:14Z",
      "abstract_url": "http://arxiv.org/abs/2607.07690v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07690v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "DiaLLM: An Investigation into the Robustness-Generation Gap in English Dialect Adaptation",
      "authors": [
        "Jordan Painter",
        "Dipankar Srirag",
        "Adarsh Kappiyath",
        "Diptesh Kanojia",
        "Aditya Joshi",
        "Lu Yin"
      ],
      "abstract": "Large language models increasingly \\emph{understand} dialectal English, yet still \\emph{produce} only standard, US-leaning English, leaving dialectal generation, the harder half of the problem, largely unaddressed. We introduce \\textbf{DiaLLM}, which continually pretrains three open-weight language model families on the International Corpus of English and applies implicit and explicit post-training paradigms, each combined with three model alignment strategies, giving the first controlled comparison of these components across Australian, Indian, and Northern British English. Our results reveal that dialectal robustness and generation are \\emph{dissociated}: benchmarks are shaped by continual pretraining and SFT, while alignment visibly reshapes generation in ways benchmarks do not capture. Explicit variety-targeted adaptation produces output reliably recognised as dialectal and preferred over broad alignment, yet the method that most aggressively optimises the dialectal reward is not preferred by human evaluators. Independent linguistic analysis corroborates this reward-quality gap, most clearly on two of the three families. No single alignment method dominates, and closing the gap will require richer reward designs and continued investment in dialectal resources. We release all code, checkpoints, and preference datasets.",
      "published": "2026-07-08T17:24:27Z",
      "abstract_url": "http://arxiv.org/abs/2607.07669v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07669v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "ALER-TI: Aligned Latent Embedding Retrieval for Time Series Imputation",
      "authors": [
        "Xuan-Thong Truong",
        "Trung-Kien Le",
        "Tung Kieu",
        "Thi-Thu Nguyen",
        "Nhat-Hai Nguyen"
      ],
      "abstract": "Deep learning has significantly advanced time series imputation, yet most existing architectures primarily rely on localized temporal context within the corrupted input sequence. This reliance can be limiting in real-world scenarios, where time series often exhibit non-stationary dynamics, weak temporal correlations, and infrequent patterns that are difficult to reconstruct from nearby observations alone. In this paper, we propose ALER-TI, Aligned Latent Embedding Retrieval for Time Series Imputation, a retrieval-augmented framework that explicitly leverages historical patterns to supplement degraded local context for more reliable missing-value reconstruction. The core of ALER-TI is Latent Embedding Alignment (LEA), which mitigates the representation mismatch between corrupted queries and complete historical candidates. By applying post-hoc masking in the latent space, LEA aligns candidates with the query's missingness pattern while allowing historical embeddings to be pre-computed and cached for efficient retrieval. ALER-TI is model-agnostic and can be integrated with various imputation backbones through a lightweight adaptation module. Extensive experiments on six real-world datasets under different missing rates demonstrate that ALER-TI consistently improves strong baseline models and enhances robustness across diverse imputation settings.",
      "published": "2026-07-08T16:59:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.07640v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07640v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Future Confidence Distillation in Large Language Models",
      "authors": [
        "Sahil Kale"
      ],
      "abstract": "Reliable confidence estimation is essential for deploying large language models (LLMs) in confidence-aware systems, where downstream decisions such as retrieval, tool use, and adaptive computation depend on accurately estimating answer reliability. Existing approaches, however, largely treat confidence as a property of completed responses, overlooking how confidence-related information evolves throughout the answering process. In this work, we investigate confidence from a temporal perspective by comparing pre-solution Feeling-of-Knowing (FOK) and post-solution Judgement-of-Learning (JOL) confidence estimates across frontier and open-source LLMs. We show that post-solution confidence is consistently better calibrated and more discriminative than pre-solution confidence, while linear probes trained on hidden representations recover substantially richer confidence-related information than models explicitly verbalise. Building on this observation, we introduce future confidence distillation, which trains predictors operating on pre-solution hidden representations using teacher confidence estimates produced by post-solution correctness probes. Despite requiring only pre-solution representations for inference, distilled predictors recover much of the calibration improvement achieved by post-solution confidence, remain highly sample efficient, and transfer across datasets within the same domain. Together, our findings demonstrate that confidence-related information evolves throughout the answering process and can be anticipated before answer generation is complete, enabling significantly more reliable yet low-cost confidence estimation.",
      "published": "2026-07-08T16:43:11Z",
      "abstract_url": "http://arxiv.org/abs/2607.07626v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07626v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Collaborative Synthetic Data Generation for Knowledge Transfer in Federated Learning",
      "authors": [
        "Maximilian Andreas Hoefler",
        "Karsten Mueller",
        "Wojciech Samek"
      ],
      "abstract": "One-shot federated learning (OSFL) addresses the communication overhead of federated learning by limiting training to a single round, but doing so without sacrificing model quality is non-trivial, particularly when client data distributions diverge. Recent work has addressed this challenge by aggregating client knowledge on the server through the construction of transferable synthetic datasets or distillates. However, most of these methods lack formal privacy guarantees, leaving a gap in jointly achieving low communication, robustness to heterogeneity, and rigorous privacy. We propose FedKT-CSD (Federated Knowledge Transfer via Collaborative Synthetic Data), a framework inspired by neural image compression that closes this gap by leveraging publicly pretrained autoencoders as a shared latent space. Each client encodes its private data in a single forward pass, computes class-conditional latent statistics, and transmits these to the server. The server aggregates these statistics via secure aggregation, adds calibrated differential privacy noise, and decodes a synthetic dataset for training a global model and further downstream tasks. This design provides formal $(\\varepsilon,δ)$-differential privacy by construction, while keeping client-side computation and communication lightweight. Despite operating under privacy constraints, FedKT-CSD is competitive with and even outperforms non-private baselines across diverse datasets and heterogeneity settings, and scales to a large number of clients. Our code is available at: https://github.com/an7123/FedKT-CSD",
      "published": "2026-07-08T15:56:58Z",
      "abstract_url": "http://arxiv.org/abs/2607.07565v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07565v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Stability of Flow Models for Graph Signals",
      "authors": [
        "Martin Schmidt",
        "Gonzalo Mateos"
      ],
      "abstract": "Generating signals on graphs requires permutation-equivariant models that exhibit stability with respect to relative structural perturbations. While favorable stability properties of Graph Neural Networks (GNNs) have been well documented, it is unclear how structural errors propagate through the dynamics of continuous generative flow models that are gaining traction for graph signal generation. In this paper, we analyze continuous normalized flow models parameterized by GNNs and show that permutation equivariance is preserved for both the resulting continuous-time ordinary differential equations and their discrete numerical approximations used as graph signal samplers. Our primary contribution is to derive explicit stability bounds on the generated probability distributions, which quantify how relative graph perturbations affect the final sampled signals. Motivated by these theoretical bounds, we introduce a stability-promoting regularized flow matching strategy that actively penalizes the spatial Lipschitz constant of the vector field during model training. Experiments using synthetic smooth signals on stochastic block model graphs and real-world fMRI signals on brain connectomes demonstrate that this bound-oriented approach yields generative models that are more robust to structural noise, without sacrificing output quality.",
      "published": "2026-07-08T15:04:39Z",
      "abstract_url": "http://arxiv.org/abs/2607.07510v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07510v1",
      "categories": [
        "eess.SP",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning",
      "authors": [
        "Zhenyu Hou",
        "Yujiang Li",
        "Jie Tang",
        "Yuxiao Dong"
      ],
      "abstract": "Reinforcement learning (RL) is becoming increasingly important for post-training large language models (LLMs). Previous RL pipelines for LLMs were mostly synchronous and batch-interleaved, which is inefficient for long-horizon agentic tasks. Recently, asynchronous RL has emerged as a more efficient alternative by updating the model as rollouts arrive. However, existing asynchronous RL systems often emphasize throughput, while leaving training stability and task effectiveness largely underexplored. For example, a key challenge is that group-wise sampling in the widely-used GRPO framework does not naturally fit asynchronous agentic training. In this paper, we present Single-rollout Asynchronous Optimization (SAO) to address the stability and off-policy challenges in asynchronous RL. To reduce off-policy effects and improve generalization, we replace group-wise sampling with single-rollout sampling, that is, using one rollout per prompt. We further improve this single-rollout strategy with practical value-model training designs. To improve optimization stability, we introduce a strict double-side token-level clipping strategy. SAO is able to train stably for one thousand steps and consistently outperform GRPO and its variants on agentic coding and reasoning benchmarks, such as SWE-Bench Verified, BeyondAIME, and IMOAnswerBench. We also demonstrate that single-rollout RL is particularly effective in a simulated online learning setting, where the model must adapt to changing evolving environments. To this end, SAO is successfully deployed in the agentic RL pipeline for training the open GLM-5.2 model (750B-A40B).",
      "published": "2026-07-08T15:02:19Z",
      "abstract_url": "http://arxiv.org/abs/2607.07508v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07508v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TimEE: End-to-end Time Series Classification via In-Context Learning",
      "authors": [
        "Jaris Küken",
        "Shi Bin Hoo",
        "Martin Mráz",
        "Frank Hutter",
        "Lennart Purucker"
      ],
      "abstract": "Time series classification (TSC) is dominated by a two-stage paradigm: train a feature encoder -- either from scratch on the target dataset or via pretraining on large corpora -- and then fit a task-specific classifier on top. While effective, this decoupling optimizes representation learning independently of the classification objective, requires per-dataset training, and prevents the model from exploiting label information during inference. We introduce TimEE, a 4.5M-parameter foundation model for end-to-end TSC via in-context learning. Given a labeled support set and a query time series, TimEE directly outputs a predicted class distribution in a single forward pass with no per-dataset training required. Following the prior-data fitted network (PFN) framework, TimEE is meta-trained exclusively on synthetic TSC tasks, where each task contains time series with distinct class identities arising from structured distributional shifts in the generative process. Despite seeing no real time series during pre-training, TimEE ranks first in ROC AUC (and third on accuracy) on the UCR benchmark among all compared methods, which include both foundation models and supervised deep learning baselines. To our knowledge, TimEE is the first purely synthetic-pretrained model to reach state-of-the-art performance on the UCR benchmark. These results establish end-to-end ICL with synthetic priors as a compelling, largely unexplored direction for TSC, with scaling, prior design, and richer generation mechanisms as natural avenues for improvement. Code is publicly available at http://github.com/automl/timee.",
      "published": "2026-07-08T14:58:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.07500v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07500v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Reward-Adaptive Iterative Discovery: A Case Study on Automated Game Testing for NHL26",
      "authors": [
        "Florian Fuchs",
        "Jessy Gosselin-Grant",
        "Boris Skuin",
        "Michele Petteni",
        "Alessandro Sestini",
        "Joakim Bergdahl",
        "Amir Baghi",
        "Linus Gisslén"
      ],
      "abstract": "Testing is a major effort for the gaming industry, requiring a significant part of development budget and people power. We present a case study on a development version of the ice hockey game EA SPORTS NHL 26, for which human playtesters test the goalie AI for behavioral exploits. To reduce the effort of re-testing the goalie AI after every game or behavior modification in the development phase, we propose Reward-Adaptive Iterative Discovery (RAID), a novel approach to automatically find exploits using an iterative Reinforcement Learning (RL) approach that trains a population of goal scoring agents. While previous approaches can already successfully find exploits, RL algorithms tend to overfit to a single solution. We introduce a simple extension on top of existing RL algorithms, such that they find multiple diverse high-quality solutions. For our first deployment of this approach, within a single experiment we were able to find six hockey scoring exploit strategies that were qualitatively similar to those that playtesters had found in hours-long manual testing sessions.",
      "published": "2026-07-08T14:57:39Z",
      "abstract_url": "http://arxiv.org/abs/2607.07498v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07498v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data",
      "authors": [
        "Vinícius Gabriel Angelozzi",
        "Héber H. Arcolezi"
      ],
      "abstract": "Machine learning models are increasingly deployed in high-stakes domains, raising concerns about both privacy and fairness. Differential Privacy (DP) has become a gold standard for privacy-preserving data analysis, while fairness-aware mechanisms aim to mitigate discrimination against underrepresented groups. However, these objectives can conflict: DP often amplifies disparities across demographic groups, and little is known about whether established fairness interventions remain effective under DP constraints. In this work, we present, to our knowledge, the first systematic evaluation of fairness interventions on differentially private synthetic tabular data. Our benchmark centers on the Adaptive Iterative Mechanism (AIM), identified as the state-of-the-art marginal-based DP synthesizer (Cormode et al. 2025). We thus evaluate fairness interventions across four datasets, multiple group fairness metrics, and three categories of mitigation strategies (pre-processing, in-processing, and post-processing) under a wide range of privacy budgets. We compare four pipeline configurations: (Baseline) training on original data; (DP-only) training on DP synthetic data; (Fair-only) applying fairness mechanisms on original data; and (DP+Fair) combining fairness mechanisms with DP synthetic data. Our results demonstrate that while DP alone can degrade both utility and fairness, applying fairness interventions can partially restore equitable outcomes. Among them, post-processing methods tend to provide more stable fairness-utility trade-offs across privacy budgets and synthesizers, achieving strong fairness improvements while preserving competitive utility relative to other intervention stages. We release all code, data, and experimental artifacts in an open-source repository to ensure full reproducibility and to support future research on the privacy-fairness-utility trade-off.",
      "published": "2026-07-08T14:33:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.07471v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07471v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation",
      "authors": [
        "Andrea Scarinci",
        "Virginia Negri",
        "Brayan Impata",
        "Suleiman Khan",
        "Victor Martinez",
        "Marcello Federico"
      ],
      "abstract": "Fine-tuning large language models (LLMs) for e-commerce attribute extraction requires labeled data representative across thousands of product types, attributes, and multiple languages. This combinatorial scale translates to millions of annotations, rendering human labeling prohibitively costly. While recent work has demonstrated synthetic label generation using LLMs, deploying such approaches at industrial scale requires integrated quality control mechanisms. We present SynthAVE, a large-scale human-validated benchmark for attribute value extraction spanning 12,726 products across 229 product types, 792 attributes, and 4 languages (Spanish, French, Italian, German). To validate synthetic labels at scale, we introduce a multi-LLM arena framework where samples are independently evaluated by 21 judge configurations (7 model families $\\times$ 3 prompts), with final labels determined via majority voting. The majority vote ensemble agrees with human experts at Cohen's $κ= 0.92$ (95.2% agreement), while individual judges show substantial inter-model agreement (Fleiss' $κ= 0.76$). This demonstrates that diverse models with varying individual judgments aggregate into highly reliable predictions, enabling cost-effective validation at scale while maintaining quality parity with human review.",
      "published": "2026-07-08T14:32:28Z",
      "abstract_url": "http://arxiv.org/abs/2607.07469v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07469v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "SpaCellAgent: A Self-Evolving LLM-Based Multi-Agent Framework for Trajectory Analysis",
      "authors": [
        "Songhan Wang",
        "Haoang Chi",
        "He Li",
        "Zhiheng Zhang",
        "Jiayan Yuan",
        "Cheems Wang",
        "Hao Peng",
        "Xinwang Liu",
        "Wenjing Yang"
      ],
      "abstract": "Spatial and Single-cell transcriptomics are transformative in deciphering cellular dynamics. As the fundamental paradigm for reconstructing cell developmental paths, trajectory inference (TI) is critical. However, existing methods require extensive manual intervention and proficiency in heterogeneous tools, posing a significant barrier to efficient TI analysis. To bridge this gap, we propose SpaCellAgent, an autonomous large language model (LLM) multi-agent framework that automates end-to-end spatiotemporal analysis and narrative generation. SpaCellAgent utilizes a multi-agent architecture for strategic workflow planning, a dynamic tool-orchestration engine for adaptive algorithm selection, and a self-evolution module that iteratively refines performance through feedback. We evaluate SpaCellAgent on six heterogeneous datasets encompassing complex temporal developmental trajectories, diverse sequencing platforms, and spatially-resolved tissue architectures. SpaCellAgent consistently demonstrates over 40\\% improvement in analytical efficiency while maintaining expert-aligned performance. By converting natural language specifications into optimized analytical workflows and fully automating the pipeline, SpaCellAgent democratizes advanced spatiotemporal modeling and establishes a scalable, agent-driven paradigm for computational biology. The code and materials are available at https://github.com/LittleXH-shw/SpaCellAgent.",
      "published": "2026-07-08T14:31:46Z",
      "abstract_url": "http://arxiv.org/abs/2607.07467v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07467v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "RLVP: Penalize the Path, Reward the Outcome",
      "authors": [
        "Bojie Li",
        "Noah Shi"
      ],
      "abstract": "Agents acting on our behalf in the real world (e.g. placing phone calls) must learn online from costly, often irreversible interactions rather than cheap simulator steps. Two things follow. First, deployability depends on the path, not only the outcome. An agent must respect outcome-neutral constraints such as not repeatedly calling an unresponsive user, respecting business hours, or completing required authentication constraints that outcome-based rewards cannot express, since violating them frequently improves apparent success. Second, because each interaction is expensive, the agent must learn efficiently from very few examples. Reinforcement learning from verifiable rewards (RLVR) is blind to both challenges: it optimizes solely on the outcome and wastes expensive rollouts on all-fail groups where group-relative advantage collapses to zero. Attempts to densify supervision by rewarding progress target the hard-to-verify direction. In contrast, real agentic environments can cheaply detect bad moves. Since group-relative advantage is equivalent to within-group variance, a dense signal helps only when it supplies variance the outcome lacks. A verifiable penalty on the path meets this condition reliably, while a progress potential helps only where partial progress is reachable. The resulting recipe \"penalize the path, reward the outcome\" achieves high task success with near-zero violations, where outcome-only training violates constraints on nearly every episode. We provide four design rules for effective penalties, including avoidance of the inaction trap that arises when a penalty is used in isolation.",
      "published": "2026-07-08T14:06:14Z",
      "abstract_url": "http://arxiv.org/abs/2607.07435v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07435v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Heterogeneity-Adaptive Diffusion Schrodinger Bridge for PET-Guided Whole-Body MRI Translation",
      "authors": [
        "Chengbo Wang",
        "Jiacheng Yu",
        "Linjie Bian",
        "Ming Qi",
        "Xiaosheng Liu",
        "Tongtong Che",
        "Jichang Zhang",
        "Shuyu Li",
        "Shaoli Song",
        "Xiuying Wang"
      ],
      "abstract": "While whole-body multimodal medical imaging scanners have been increasingly recognized for more effective medical applications, the excessive long acquisition time in PET-MR scanning is a major obstacle in more efficient clinical practice. Deep learning-based MRI translation provides a potential solution to reduce scan duration. However, current models often focus on specific anatomical regions and face challenges for whole-body scans that consists of highly heterogeneous feature distributions mainly due to (1) different anatomical regions across whole-body, and (2) lesions or pathological tissues. This paper tackles the challenges through a novel Heterogeneity-Adaptive Diffusion Schrodinger Bridge (HA-DSB) framework. By explicitly modeling translation as stochastic transport between source and target distributions, HA-DSB incorporates region context embeddings derived from a vision-language model (VLM) to enable region-specific modeling. To enhance fidelity of the pathological tissue, lesion-aware metabolic prior from PET is integrated directly into the bridge dynamics through a dual-stage guidance mechanism. Specifically, a PET-guided noise modulation module adaptively scales spatial diffusion perturbations during the forward process, while PET features are leveraged during the reverse process to selectively amplify lesion-relevant structures via an attention mechanism. Experiments demonstrate the superiority of our method across different body regions in whole-body MRI translation and show improved translation quality in lesion areas under PET guidance. Our code is available at Github.",
      "published": "2026-07-08T13:35:22Z",
      "abstract_url": "http://arxiv.org/abs/2607.07401v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07401v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When Prompts Ignore Structure: Graph-Based Attribute Reasoning for Calibrated VLMs",
      "authors": [
        "Tanay Sodha",
        "Aditya Sharma",
        "Ramya Hebbalaguppe",
        "Vinti Agarwal",
        "Pranav Murthy Yeluripaty"
      ],
      "abstract": "Reliable confidence estimation remains a key limitation of test-time adaptation in vision-language models (VLMs), where prompt tuning improves zero-shot accuracy but often degrades calibration due to entropy-driven overconfidence. Prior approaches mitigate this using LLM-derived class attributes and contrastive regularization, yet treat attributes independently, ignoring their relational structure. We propose ARGTCA, which represents (class, attribute) pairs as nodes in a Symbolic Attribute Graph and trains a Graph Attention Network (GAT) using contrastive objectives to produce structurally informed embeddings that capture inter-attribute dependencies. We introduce two attribute selection strategies: ARGTCA-DIV for intra-class diversity and ARGTCA-DISC for inter-class discrimination. Experiments across nine benchmarks show that ARGTCA-DIV reduces average Expected Calibration Error (ECE) by approximately ~37% over baselines, while ARGTCA-DISC consistently performs as the second-best variant, reducing average ECE by approximately ~17% over baselines. These results suggest that modeling symbolic attribute interactions provides a principled approach for reliable test-time adaptation in VLMs.",
      "published": "2026-07-08T13:31:17Z",
      "abstract_url": "http://arxiv.org/abs/2607.07395v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07395v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Physics-Audited Agentic Discovery in Scientific Machine Learning",
      "authors": [
        "Diab W. Abueidda",
        "Bilal Ahmed",
        "Panos Pantidis",
        "Mostafa E. Mobasher"
      ],
      "abstract": "In agentic scientific machine learning (SciML), large language model (LLM) agents can discover surrogate models and select one by an automated score, typically an error metric. A low error, however, does not establish that the predicted fields satisfy the physics that matter for mechanics, such as boundary conditions, superposition, stiffness scaling, or causality. We introduce Physics-Audited Agentic SciML (PA-SciML), a verification-first workflow for agentic SciML discovery. The workflow fixes a scoring evaluator before search, derives reviewable machine-checkable physics requirements, checks each trained candidate on its outputs, and separately searches prescribed input ranges or measured load-history spans for high-violation cases without reference solution fields. A surrogate is reported as verified only under the stated checks. When enabled, the workflow also adds advisory numerical probes before training and tests one modeling change at a time to record which isolated edits are associated with score gains before reuse. In the reported computational-solid-mechanics numerical examples, the static elasticity run selects a surrogate with lower validation error than the error-only baseline while both selected models pass the common linear-elastic checks. In the transient elastodynamics run, an error-only baseline with similar mean error fails a stricter causality check by responding to future parts of the loading history, while the selected surrogate passes the stated checks. The main distinction is per-candidate physics evidence on predicted fields, not a richer aggregate score.",
      "published": "2026-07-08T13:10:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.07379v1",
      "pdf_url": "https://arxiv.org/pdf/2607.07379v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
