const PAPERS_DATA = {
  "last_updated": "2026-09-07 04:12:52 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "RegionFed: Federated Learning for Personalized Query Understanding in Heterogeneous Retail Environments",
      "authors": [
        "Quoc H. Nguyen",
        "Ali Lafzi",
        "Abhijeet Phatak",
        "Siddharth Pratap Singh",
        "Rohit Upadhyay",
        "Yogananda Domlur Seetharama",
        "Chittaranjan Tripathy"
      ],
      "abstract": "Retail search systems serve diverse geographic regions with distinct query patterns, vocabularies, and product preferences, creating significant data heterogeneity that challenges both privacy-preserving training and model personalization. Federated learning offers a natural solution for privacy, but standard FL methods produce global models that sacrifice regional performance, while existing personalized FL approaches operate at the parameter level and catastrophically collapse on modern transformers (below 10\\% accuracy on T5) due to tied embeddings and LayerNorm interactions. We introduce RegionFed, an \\textit{architecture-robust} federated learning framework that sidesteps this failure by operating entirely at the gradient level. RegionFed uses the $\\ell_2$ conflict between regional and global gradients as a unified signal that (i) diagnoses heterogeneity, (ii) routes each region to the cheapest sufficient personalization strategy, and (iii) adaptively controls personalization strength. Because it treats models as differentiable black boxes, RegionFed deploys on T5-Small, T5-3B, RoBERTa, and CNN with zero code changes, providing large gains on transformers (where parameter-level methods collapse) and consistent improvements on CNNs. Across three public datasets (Amazon ESCI, Amazon Reviews, LEAF-FEMNIST) and four architectures, RegionFed-Meta achieves 92.27\\%, closing the gap to the privacy-violating centralized upper bound (Centralized + Regional Weighting: 92.04\\%, $Δ$=0.23pp, within 1$σ$) while providing $(ε{\\approx}0.60)$-differential privacy and $\\mathcal{O}(1/\\sqrt{T})$ convergence.",
      "published": "2026-09-04T17:50:10Z",
      "abstract_url": "http://arxiv.org/abs/2609.05403v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05403v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Molecular Déjà Vu: Digit-Level Retrieval of Published Values in Frontier Language Models",
      "authors": [
        "Matthias Busch",
        "Marius Tacke",
        "Sviatlana V. Lamaka",
        "Mikhail L. Zheludkevich",
        "Christian J. Cyron",
        "Roland C. Aydin",
        "Christian Feiler"
      ],
      "abstract": "Large language models (LLMs) are increasingly evaluated on molecular property benchmarks, but accuracy cannot distinguish a model that predicts a property from one that retrieves a published number. We audit 22 frontier models on 12 regression benchmarks for verbatim retrieval and find that it is widespread but relatively benchmark-specific: on five datasets more than $50\\%$ of the LLMs show verbatim retrieval, while on the remaining datasets it appears only in isolated cells. We run our experiments at two reasoning levels and find that reasoning changes retrieval. The same experiments, on the same molecules and with the same prompt, are flagged $89\\%$ more often at the higher reasoning level than at the lowest one. Finally, we test a way to interrupt retrieval in our most contaminated cases, and find that the strongest models in some cases still recognise a combination of transformed SMILES strings and original labels. Furthermore, suppressing retrieval moves the prediction errors of the different models closer together in relative terms, while their differing use of verbatim retrieval spreads them apart. This indicates that the general predictive capability of an LLM is not determined solely by the amount of memorised values. This work provides an overview of the amount and depth of verbatim retrieval in molecular regression benchmarks using LLMs.",
      "published": "2026-09-04T17:32:48Z",
      "abstract_url": "http://arxiv.org/abs/2609.05381v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05381v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Design Docs Are All You Need: An AI-native Machine-Learning Performance Tool",
      "authors": [
        "Samuel Kushnir",
        "Kimia Noorbakhsh",
        "Kavya Sreedhar",
        "Liqun Cheng",
        "Ming Liu",
        "Parthasarathy Ranganathan",
        "Mohammad Alizadeh",
        "Fred Kjolstad",
        "Suvinay Subramanian"
      ],
      "abstract": "Machine-learning performance modeling is a uniquely hostile terrain for long-lived software: the assumptions baked into today's abstractions are invalidated by tomorrow's models and systems, forcing perpetual refactoring of performance-modeling frameworks. Meanwhile, AI coding agents have become fast and capable enough that regenerating an entire library is cheaper than paying down the tech debt of incrementally patching it. We describe SMART, a rigorous symbolic performance-modeling library for ML systems whose main branch contains almost no code: the repository is a DAG of self-contained natural-language design docs, coding sub-agents regenerate the implementation from only the docs on new version updates, and every human change is a natural-language edit to a doc--self-documenting by construction. Two ingredients make regeneration reliable: (i) a design-doc style built around step-by-step worked examples that act as in-context demonstrations for the generating agents, and (ii) a minimal, recursively defined operator IR with symbolic (SymPy) cost expressions, a fast analytical roll-up mode for large sweeps, and a slow modulo-scheduling mode for fine-grained schedule studies. Regenerated implementations reproduce hand-audited reference models--including DeepSeek-V3 serving on a TPU pod slice--to round-off precision, suggesting that design docs--not code--can be the durable artifact for ML-systems co-design tools.",
      "published": "2026-09-04T17:08:51Z",
      "abstract_url": "http://arxiv.org/abs/2609.05364v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05364v1",
      "categories": [
        "cs.PL",
        "cs.AI"
      ]
    },
    {
      "title": "Lightweight Vision Transformer Compression for On-Device Plant Disease Detection in Resource-Constrained Agricultural Field Conditions",
      "authors": [
        "Mahadev Sunil Kumar",
        "Bhavika Gondi",
        "Desaisetty Venkata Satya Sai Swapnith",
        "Gangireddy Rahul Jogi",
        "Sudheesh Manalil",
        "Arnab Raha",
        "Amitava Mukherjee",
        "Parthasarathy Seethapathy",
        "G. Gopakumar"
      ],
      "abstract": "Chilli (Capsicum annuum) is one of India's most economically significant crops, yet its productivity is persistently threatened by diseases that are difficult to identify without expert intervention. While Vision Transformers (ViTs) have achieved high classification accuracy, their large computational footprint makes deployment on resource constrained devices challenging. Existing compression approaches typically address pruning, quantization, and knowledge distillation in isolation, leaving the potential benefits and interactions of their combined application insufficiently explored. We propose a unified Vision Transformer compression framework that combines Hessian-Balanced Adaptive Block Pruning (H-BAC), guided by second-order sensitivity estimation, with quantization and attention-based knowledge distillation. To systematically identify the most effective configuration within each compression family, each technique is first evaluated independently through controlled ablation studies, after which the best-performing components are integrated into a sequential deployment pipeline tailored to real-world agricultural constraints. On a chilli 3-class village-split dataset with a genuine cross-village, cross-device out-of-distribution test split, the resulting compressed models match or exceed the 95.13% FP32 baseline's accuracy, alongside 74-98% model size reduction, and the fully integrated compression pipeline achieves a 54.5x size reduction (327.42 MB to 6.01 MB) at 95.13 +/- 2.32% accuracy across four tested configurations. A direct comparison further reveals that, on this dataset, a directly-trained student of the same final size, without pruning or distillation, reaches comparable accuracy of 94.87%, at the same 6.01 MB INT8 size, indicating where H-BAC and knowledge distillation are, and are not yet shown to be, worth their computational cost.",
      "published": "2026-09-04T16:40:06Z",
      "abstract_url": "http://arxiv.org/abs/2609.05334v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05334v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness",
      "authors": [
        "Alexander Neubauer",
        "Tianzhen Hong",
        "Han Li",
        "Mengbo Yu",
        "Amin Darbandi",
        "Yannick Fürst",
        "Martin Kriegel"
      ],
      "abstract": "Building automation systems generate rich sensor data yet remain insight-poor because heterogeneous point naming, missing metadata, and fragmented documentation obstruct their operational use. This systematic review analyses and codes 66 peer-reviewed studies on large language models (LLMs) for HVAC operations published between 2023 and March 2026. Each study is classified across five application families and three LLM method families and assessed for evidence realism, deployment readiness, and the responsibility boundary between the LLM and physical HVAC decisions. The corpus is concentrated in building energy modelling (BEM, 32 of 66 papers), while load forecasting remains too sparse for subfield-level conclusions. Only four studies reach pilot-level evidence, and none reports sustained operational deployment. No study was classified as ready-now for industry adoption; three were near-term and 63 research-only. Nevertheless, several bounded, human-in-the-loop uses merit near-term trials, including point-name normalisation, document-grounded operator support, BEM workflow assistance, and advisory interfaces around physics-based controllers. Conventional machine learning (ML), model predictive control (MPC), reinforcement learning (RL) and ontology-based tools remain more adopted for high-frequency control, short-horizon numerical forecasting, and well-posed ontology mapping, while autonomous agentic operation and unvalidated occupant proxies remain research-stage. Current evidence therefore supports LLMs primarily as semantic and workflow layers rather than autonomous HVAC controllers. Future work should prioritise field-validated benchmarks, orchestration evaluation under operational constraints, and LLM-MPC/RL architectures with bounded latency and verifiable safety properties.",
      "published": "2026-09-04T16:08:05Z",
      "abstract_url": "http://arxiv.org/abs/2609.05314v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05314v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "eess.SY"
      ]
    },
    {
      "title": "How Does mHC Use Its Residual Streams? Selective Routing and Near-Identity Mixing",
      "authors": [
        "Pengxiang Zhao",
        "Xing Li",
        "Xianzhi Yu",
        "Wei Guo",
        "Zhenhua Dong"
      ],
      "abstract": "Hyper-Connections and their manifold-constrained variant mHC widen a residual pathway from one stream to n, yet how trained models use this capacity remains unclear: how broadly blocks read and write, how strongly the residual pathway mixes streams, and whether the streams carry distinct representations. We examine these properties in the four-stream residual pathway of DeepSeek-V4-Flash using effective stream counts, cross-stream residual weights, and inter-stream cosine similarity. Read/write routing is concentrated but varies across depth: a typical attention or FFN site effectively uses about two streams, while the dominant stream changes across layers and the representations remain directionally distinct. Residual mixing is modest and occurs primarily in early layers; in layers 22-42, the pathway mostly carries each stream forward separately. Targeted interventions establish the functional significance of these patterns. Replacing the late mixers by identity increases C4 perplexity by only 1.9% and preserves the six-task average score, whereas replacing the early mixers increases perplexity by 41%. Fixing each early mixer to its C4 diagnostic mean increases perplexity by only 0.2% and reduces the average score by 0.25 percentage points, showing that its site-specific structure matters more than its token-wise variation on the evaluated metrics. Likewise, retaining the three largest routing weights per token at every site increases perplexity by at most 2.7% and changes the average score by at most 0.4 points. Thus, the studied model realizes only part of the flexibility afforded by four-stream mHC: individual blocks rarely require all four streams, and late residual mixing provides little measured benefit.",
      "published": "2026-09-04T16:02:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.05309v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05309v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "GUT: Quantifying and Optimizing the Reasoning Uncertainty of LLMs via Graph Complexity",
      "authors": [
        "Shuang Liang",
        "Xin-Yu Hu",
        "Xiang-Jun Ou",
        "Shao-Qun Zhang"
      ],
      "abstract": "Recent years have witnessed great advances in the reasoning ability of Large Language Models (LLMs). However, the reasoning processes of LLMs often exhibit uncertainty, where LLMs often produce a proliferation of divergent branches at each reasoning step even when fed the same prompting inputs, and certain branches exhibit evidently incredible, even nonsensical, reasoning chains and results. In this paper, we propose the Graph-complexity-based UncerTainty (GUT) method for investigating the reasoning uncertainty of LLMs. The key idea of GUT is to characterize the potential branches of each reasoning chain with a directed acyclic graph, thereby ensuring that all potential branches are comprehensively covered within the graph space. Building upon this recognition, we further build two modules of GUT, that is, a Quantification (GUT-Q) module and an Optimization (GUT-O) module, for quantifying and reducing the reasoning uncertainty of LLMs, respectively. GUT-Q measures LLM reasoning uncertainty by approximating the reasoning space complexity with graph complexity. GUT-O implements uncertainty optimization by treating negative uncertainty as the reward function in reinforcement learning. Experimental results conducted on four LLMs and five datasets validate the effectiveness of GUT.",
      "published": "2026-09-04T15:40:12Z",
      "abstract_url": "http://arxiv.org/abs/2609.05284v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05284v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Don't Drop Dropout: Optimizing Layer Sparsity for Efficient LLM Training and Inference",
      "authors": [
        "Mostafa Elhoushi",
        "Alex Pretko",
        "Nolan Dey",
        "Bin Claire Zhang",
        "Gavia Gray",
        "Gurpreet Gosal",
        "Abdulrahman Mahmoud",
        "Shane Bergsma",
        "Joel Hestness"
      ],
      "abstract": "Layer dropout (a.k.a. stochastic depth) has been shown to enable faster training, higher accuracy, and robustness to zero-shot layer pruning in both language and vision transformers. However, as models and datasets have scaled, dropout - particularly layer dropout - has largely disappeared from large language models (LLMs) pre-training recipes. While some prior work has reported that dropout can degrade accuracy, no comprehensive study has quantified, let alone mitigated, this effect. In this study, we show that layer dropout should be used in state-of-the-art LLM training, establishing best practices and scaling analysis for both training and post-training benefits. Concretely, with optimal layer distribution, time schedule, and optimizer hyperparameters, we observe that at the same training FLOPs layer dropout leads to lower loss. For a given number of training steps, LLMs can achieve lower or similar validation loss while saving upto 25% of training FLOPs. Moreover, layer dropout enables significant post-training optimizations, such as early exit, intermediate-layer skipping, and self-speculative decoding, yielding up to 1.5x inference speedup with negligible accuracy loss. Across more than 2400 training experiments, spanning models from 271M to 8.2B parameters and datasets up to 160B tokens, we demonstrate that these findings extend reliably to large-scale training regimes. All pre-training experiments were run on Cerebras CS-3 systems.",
      "published": "2026-09-04T15:30:47Z",
      "abstract_url": "http://arxiv.org/abs/2609.05275v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05275v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Trace2Tower: Transition-Aware EigenTrace Induction of Multi-Level Skills for LLM Agents",
      "authors": [
        "Jiazheng Sun",
        "Boyu Yang",
        "Binhao Yuan",
        "Mingxuan Li",
        "Xin Peng"
      ],
      "abstract": "Large language model agents increasingly rely on execution traces to master complex interactive tasks. However, current paradigms are bottlenecked by shallow trajectory retrieval and flat skill summarization, fundamentally ignoring the temporal dependencies and outcome-conditioned topology of agent behavior. We introduce Trace2Tower, a transition-aware EigenTrace framework that distills raw trajectories into a robust skill hierarchy. Trace2Tower abstracts step-level interactions into canonical events, constructing a unified graph governed by semantic compatibility, transition dynamics, and outcome evidence. Through a novel contrastive spectral decomposition, it isolates stable, success-aligned behavioral modes while rigorously suppressing failure-prone shortcuts. These modes organically populate a dynamic skill tower of action templates, procedural routines, and overarching task strategies, continuously refined via verifier-guided feedback. On ALFWorld, Trace2Tower achieves 87.31% success requiring only 10.35 steps and 0.26 invalid actions; on WebShop, it reaches 50.67% exact success. Across both benchmarks, Trace2Tower significantly outperforms existing baselines in task mastery and context-efficient experience reuse.",
      "published": "2026-09-04T15:22:39Z",
      "abstract_url": "http://arxiv.org/abs/2609.05261v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05261v1",
      "categories": [
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "Ask Before You Optimize: Dynamic Pre-Formulation Clarification for Interactive Optimization",
      "authors": [
        "Sihan Ge",
        "Yichen Lin",
        "Chenyu Zhou",
        "Jianghao Lin",
        "Tao Yao",
        "Dongdong Ge"
      ],
      "abstract": "Large language models (LLMs) are increasingly used to formulate optimization models from natural-language problem descriptions, yet realistic operations research (OR) requests are often incomplete: missing objectives, constraints, or business rules can change the resulting mathematical program. Existing evaluations largely assume a complete specification and therefore overlook whether an agent knows when clarification is needed before modeling. We introduce OR-Clarify, a benchmark for pre-formulation clarification. Each task presents a partial public problem description, withholds structured hidden slots, and evaluates agents through bounded interaction with a simulated user. The benchmark supports both openended and choice-based clarification, and measures slot recovery, stopping behavior, silent assumptions, and interaction cost. We further propose Interactive Optimization (InterOPT), a two-stage framework that identifies unresolved formulation-critical gaps and uses them to guide whether to ask the next question or to stop. In our choice-based experiments, InterOPT substantially outperforms all baselines in exact slot recovery; in the open-ended setting, it remains competitive with strong prior methods. Together, OR-Clarify and InterOPT reframe OR assistance as a selective completeness decision: clarify when needed, stop when ready, and quantify what remains missing.",
      "published": "2026-09-04T15:19:38Z",
      "abstract_url": "http://arxiv.org/abs/2609.05258v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05258v1",
      "categories": [
        "math.OC",
        "cs.AI"
      ]
    },
    {
      "title": "Commonsense Reasoning in Computer Vision: Foundations, Recent Advancements, and Future Directions",
      "authors": [
        "Bahar Uddin Mahmud",
        "Sumit Barua",
        "Guan Yue Hong",
        "Ajay Gupta",
        "Hexu Liu"
      ],
      "abstract": "Commonsense reasoning in computer vision encompasses integrating visual data and contextual knowledge, crucial for enhancing AI's understanding of everyday scenarios. This understanding not only improves machine learning models but also enhances their ability to interact meaningfully with humans and the environment. Unlike CNN-based conventional vision models, which are designed to identify objects within a specific image, incorporating commonsense knowledge enables models to interpret scenes in a more holistic manner, thereby improving their spatial ability to reason about relationships among objects and actions. This integration not only enhances object recognition but also facilitates a deeper understanding of the contextual factors, ultimately leading to more precise predictions and interactions in real-world applications. This paper presents a comprehensive survey of recent developments that integrate commonsense knowledge into computer vision tasks. We systematically review approaches based on knowledge graphs, scene graphs, neuro-symbolic models, and commonsense-augmented transformers. We also outline current limitations related to dataset bias, knowledge incompleteness, and integration challenges. Finally, we highlight prospective research trajectories in cross-modal reasoning, scalable commonsense knowledge injection, and neuro-symbolic hybrid architectures to develop truly intelligent visual systems.",
      "published": "2026-09-04T15:19:37Z",
      "abstract_url": "http://arxiv.org/abs/2609.05257v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05257v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Unified Physics-Aware Quantum Machine Learning Framework across Power GaN HEMTs and Logic Nanowire FETs: Predicting Unseen Process Splits and Held-Out Geometry Combinations with Lower Error and Tighter Split-to-Split Variability",
      "authors": [
        "Rushat Rai",
        "Yun-Yuan Wang",
        "Autsada Kakaen",
        "Pei-Jie Chang",
        "Doan Viet Nguyen",
        "Yuan-Chieh Chiu",
        "Doldet Tantraviwat",
        "Niall Tumilty",
        "Simon See",
        "Wen-Jay Lee",
        "Tai-Yue Li",
        "Nan-Yow Chen",
        "Tian-Li Wu"
      ],
      "abstract": "We present a unified reinforcement-learning (RL) framework that discovers compact parametrized quantum circuits (PQCs) for data-scarce device modeling. A graph neural network (GNN) policy optimized by proximal policy optimization (PPO) searches circuit architectures using leave-one-group-out cross-validation (LOGOCV) error on held-out process or geometry groups as the reward. The framework achieves the lowest mean absolute error (MAE) on all 11 targets versus six classical baselines, with 59% lower error (Ioff) and 81% tighter fold variability (VTH) for HEMTs and 84% lower error (VTH, SS, Ioff) and 82% tighter fold variability (Ioff) for NWFETs. These results demonstrate the potential of RL-selected, classically simulated PQCs as compact surrogates with low OOD error and improved physical consistency, despite imposing no explicit physical constraints, penalty terms, or device-specific equations, on the two evaluated device datasets.",
      "published": "2026-09-04T15:15:40Z",
      "abstract_url": "http://arxiv.org/abs/2609.05251v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05251v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Uncensored Open-weight Models: Redistribution as the Persistence Layer",
      "authors": [
        "10a Labs",
        ":",
        "Juliette Garcia",
        "Hailey May",
        "Bobby McKenzie",
        "David Pham",
        "Matthew Swain",
        "Joshua Valdez",
        "Corie Wieland",
        "Zachary Yahn"
      ],
      "abstract": "A rapidly expanding ecosystem of actors is removing built-in safety guardrails from open-weight AI models. We profile this ecosystem by identifying key producers, downstream reproductions, and emerging applications. Between January 2024 and March 2026, we identified 3,471 original uncensored models on HuggingFace, each repackaged an average of 2.4 times; three actors account for 52% of all 8,164 compressed redistributions. Once quantized and mirrored across separate accounts, formats, and registries such as Ollama, these models persist regardless of upstream removal and become easier to deploy downstream. Of the 1,643 identified GitHub applications integrating uncensored large language models (ULLMs), 25% were classified as explicitly malicious.",
      "published": "2026-09-04T15:06:39Z",
      "abstract_url": "http://arxiv.org/abs/2609.05241v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05241v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "PRICE: A Systematic Study of LLM Adaptation Choices for Bitcoin Price Forecasting",
      "authors": [
        "Maryam Fakhari",
        "Mehran Safayani"
      ],
      "abstract": "Cryptocurrency markets exhibit extreme volatility and non-stationary dynamics that challenge conventional forecasting methods. Although Large Language Models (LLMs) have shown promise for time series forecasting, the combined effects of adaptation choices remain largely unexplored in financial settings. This study introduces PRICE, a structured approach for adapting LLMs to short-term Bitcoin price forecasting. Built on a 4-bit quantized LLaMA-3 8B model, PRICE investigates how fine-tuning, numerical representation, prompting, inference, and decoding jointly influence forecasting performance. PRICE integrates Parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA), Recursive multi-step inference, Integer-rounded numerical representation, Context-Task-Format (CTF) prompting, and Exact zero-temperature decoding. Ablation studies show that each component contributes to forecasting accuracy and reliability. LoRA enables efficient training on limited hardware, recursive inference improves accuracy, integer-rounded values reduce errors, CTF prompting outperforms Chain-of-Thought, Implicit Chain-of-Thought (iCoT), and few-shot prompting, and zero-temperature decoding improves stability during recursive forecasting. Comparative evaluation against eight transformer-based and time-series foundation models shows that PRICE achieves the lowest forecasting errors on both validation and test sets while maintaining robust performance across evaluation periods. Despite being based on a model primarily pretrained on text rather than time-series data, PRICE achieves competitive or superior performance relative to specialized foundation models. These findings demonstrate that adaptation choices critically determine the accuracy and robustness of LLMs for numerical time-series forecasting.",
      "published": "2026-09-04T15:01:38Z",
      "abstract_url": "http://arxiv.org/abs/2609.05235v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05235v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.NA"
      ]
    },
    {
      "title": "ACE: Adaptive Calibration-Free Expert Skipping for MoE-based LLMs",
      "authors": [
        "Zukang Xu",
        "Zhixiong Zhao",
        "Xing Hu",
        "Jiangyong Yu",
        "Houji Wen",
        "Jun Li",
        "Zhe Jiang",
        "Dawei Yang"
      ],
      "abstract": "Mixture-of-Experts (MoE) architectures provide an efficient paradigm for scaling large language models (LLMs), yet fixed top-k routing activates the same number of expert slots for every token, causing substantial redundant computation. Existing expert-skipping methods often rely on router confidence, calibration data, or additional training, and therefore cannot reliably estimate the actual contribution of routed experts. To this end, we propose ACE, a training-free, calibration-free, and checkpoint-preserving framework for token-adaptive expert skipping in MoE-based LLMs. ACE contains two complementary components: 1) Global Spectral Proxy (GSP), which estimates global transformation capacity from the coupled gate, up, and down projections together with RMSNorm scaling; and 2) Router-Conditioned Refinement (RCR), which constructs expert-specific direction prototypes from centered router weights and evaluates expert responses along routing-preferred directions. During inference, ACE combines both estimates with runtime router gates and skips an expert slot only when both views identify it as low-contribution, while always retaining the top-1 expert. All expert statistics are computed offline, leaving only table lookups and lightweight scalar operations online. Extensive experiments across three MoE-based LLMs and eight benchmarks demonstrate that ACE consistently outperforms existing static and dynamic baselines, with increasingly pronounced advantages under aggressive expert skipping. For instance, at a 50% skipping ratio on Qwen3.6-35B-A3B, ACE reduces WikiText-2 perplexity by 7.96% and improves average downstream accuracy by 4.15 percentage points over the strongest competing method.",
      "published": "2026-09-04T14:56:00Z",
      "abstract_url": "http://arxiv.org/abs/2609.05228v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05228v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Verifier-Guided Explainable Reasoning Framework with Gold-Anchored QLoRA, Task-Aware Mixture-of-Experts, and Group-Relative RLVR",
      "authors": [
        "Thi Kim Trang Vo",
        "Nam Tien Le",
        "Thi Kim Nguyet Vo",
        "Minh Khang Tran",
        "Duy Phuong Tran"
      ],
      "abstract": "Large language models (LLMs) show strong reasoning ability, but their explanations can remain inconsistent, weakly grounded, or difficult to verify. We propose a verifier-guided explainable reasoning framework for transparent educational question answering that combines gold-anchored QLoRA, task-aware symbolic routing, and group-relative RLVR. Qwen2.5-3B-Instruct is first adapted with field-weighted QLoRA supervision anchored to authoritative answers. A lightweight router then assigns logic problems to a FOL/Z3 verifier and physics problems to a formula- and unit aware symbolic solver. Verifier feedback is further used to support candidate evaluation, self-revision, and reward construction during RLVR. Candidate responses are evaluated along three complementary dimensions: P1 for answer correctness, P2 for evidence or unit consistency, and P3 for reasoning depth and explainability. At inference, gold-free self-consistency aggregates multiple candidate responses before an optional question-only physics verifier performs conservative system-level correction. On 438 held-out examples, RLVR increases P3 from 50.68% to 72.20%, while hybrid P1 remains approximately stable at 55.94%. Self-consistency improves model only P1 from 48.86% to 50.23%, with symbolic verification providing the remaining hybrid gain. These results indicate that RLVR primarily strengthens explicit reasoning structure, while symbolic verification complements the neural policy by improving answer reliability at the system level.",
      "published": "2026-09-04T14:52:06Z",
      "abstract_url": "http://arxiv.org/abs/2609.05221v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05221v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "What Matters in On-Policy Distillation? A Perspective on Data Efficiency and Data Selection",
      "authors": [
        "Zhinan Hou",
        "Jiaqi Zhang",
        "Xunliang Cai",
        "Keyou You"
      ],
      "abstract": "On-Policy Distillation (OPD) has emerged as a widely adopted post-training paradigm for enhancing large language models in reasoning domains. However, the data-centric mechanisms in OPD remain relatively underexplored. This paper presents a empirical study of data efficiency and data selection in OPD. We begin by investigating an extreme setting: training OPD on only one example, namely 1-shot OPD. Surprisingly, we find that 1-shot OPD is consistently effective across all sampled training examples and harder examples often yield superior performance gain. We next investigate what actually drives the student model's improvement in the training data. Our analysis reveals that the improvement is not driven by high token entropy, but the longer CoT paths which hard problems naturally generate. Training on longer CoT can help maintain closer alignment with the teacher over a long reasoning horizon, and learn critical thinking patterns usually missing in short CoTs, such as reflection (e.g., ``Alternatively''). Based on these insights, we propose a simple data selection method that selects only hard examples for training, where even ``unsolvable'' examples that completely exceed the teacher's capability can be successfully used. Our experiments conducted on four models ranging from 1.5B to 7B show that training the student model on only 8 selected hard examples matches the performance of the 17K dataset baseline.",
      "published": "2026-09-04T14:32:08Z",
      "abstract_url": "http://arxiv.org/abs/2609.05198v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05198v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Phase Transition Frequency as a Training Time Predictor of Test Accuracy in ResNets",
      "authors": [
        "Arunan J"
      ],
      "abstract": "The number of discrete class-separability jumps observed during ResNet finetuning is examined empirically as a predictor of final test accuracy. Across 75 experiments spanning four benchmarks (CIFAR-10, CIFAR-100, TinyImageNet, and CIFAR-10-C) and three architectures (ResNet-18, ResNet-50, and ResNet-101), with five to ten seeds per configuration, a strong within-dataset negative correlation is obtained on standard i.i.d. classification benchmarks: \\(r = -0.84\\) on CIFAR-10 (\\(p < 10^{-8}\\), \\(n = 30\\)) and \\(r = -0.87\\) on CIFAR-100 (\\(p < 10^{-5}\\), \\(n = 15\\)). Under distributional stress, the relationship attenuates: TinyImageNet yields \\(r = -0.45\\), and the CIFAR-10-C corruption benchmark yields \\(r = -0.19\\). Two additional analyses discipline the empirical claim. A partial correlation controlling for architecture depth, treated as a linear covariate, shows that on CIFAR-100 the transition count retains statistically significant predictive power (\\(r_{\\mathrm{partial}} = -0.69\\), \\(p = 0.007\\)); the corresponding result under the stricter categorical conditioning is not established at \\(n = 15\\). A comparison against six alternative training-curve signals shows that transition count achieved the strongest correlation among the evaluated signals on CIFAR-100 and one of the strongest on CIFAR-10, but is dominated by other signals on the two stressed benchmarks. The comparison is restricted to training-curve-level signals; comparisons against effective rank, Hessian sharpness, Fisher information, margin, and neural-collapse measures, which are the strongest competitors in the current literature, are not part of the present study and remain open. The observation is presented as an in-distribution training-quality probe among a family of candidate probes, and an inexpensive detection procedure suitable for logging alongside a standard training loop is provided.",
      "published": "2026-09-04T14:28:57Z",
      "abstract_url": "http://arxiv.org/abs/2609.05194v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05194v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "AxQM: A Textbook-Scale Benchmark for Formal Proof Synthesis in a Library of Finite-Dimensional Quantum Mechanics",
      "authors": [
        "Weichen Winston Yin",
        "Jacob M. Taylor",
        "Dirk R. Englund",
        "Frank H. L. Koppens"
      ],
      "abstract": "Formalizing mathematics in a proof assistant, where a machine checks every definition, statement and proof, has set a new standard of rigor. Large language models are now capable of formalizing autonomously, even at the scale of whole textbooks. We bring this standard of rigor to physics, where theoretical arguments carry idealizations that are rarely stated fully, and any logical gaps could have a cascading effect on interdependent results. Recognizing the need to evaluate autoformalization systems for physics, we release AxQM, 1,019 kernel-checkable proof-synthesis tasks over 479 items drawn from the textbook Quantum Computation and Quantum Information by Nielsen and Chuang. The tasks are stated in a custom Lean library of finite-dimensional quantum mechanics. By task count, it is the largest proof-synthesis benchmark in physics by a factor of four. AxQM is derived from a near-complete formalization of the formal portions of the textbook, so every task is guaranteed a solution, which we keep private. Grading of the benchmark is done deterministically by the Lean kernel, which checks that the proof compiles, that no sorry appears in it or in any declaration it depends on, and that it introduces no new axioms.",
      "published": "2026-09-04T13:58:19Z",
      "abstract_url": "http://arxiv.org/abs/2609.05157v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05157v1",
      "categories": [
        "quant-ph",
        "cs.AI",
        "cs.LO"
      ]
    },
    {
      "title": "Beyond Stationarity in Time Series: Discovering Causal Structures and Latent Regimes via Markov Blankets",
      "authors": [
        "Lei Zan",
        "Charles K. Assaad",
        "Emilie Devijver",
        "Eric Gaussier"
      ],
      "abstract": "This paper introduces Regime-aware Constraint-Based and Noise-Based causal discovery with Markov Blankets (RCBNB-MB), a novel causal discovery algorithm for time series that relaxes the common assumption of a single, time-consistent causal structure. Time series are typically observed at discrete time points and often exhibit regime changes that challenge the assumption of a static causal structure, a limitation in many real-world dynamic systems. To address this challenge, RCBNB-MB identifies latent causal regimes, defined as subsets of time points within which a stable causal structure holds. The algorithm follows an iterative strategy that segments the time series into regimes and discovers the causal graph within each regime. By leveraging the Markov blanket rather than direct parents, RCBNB-MB gains robustness to errors in causal discovery and preserves predictive information. We provide theoretical guarantees for RCBNB-MB's ability to recover both regime transitions and causal graphs under reasonable assumptions. Furthermore, we validate its effectiveness through extensive experiments on simulated datasets with known ground truth and real-world IT monitoring data, where taking into account regime shifts is critical. Empirical results show that RCBNB-MB systematically outperforms baseline approaches in accurately detecting regime changes and their associated causal graphs, positioning it as a robust and versatile framework for non-stationary time series analysis.",
      "published": "2026-09-04T13:52:51Z",
      "abstract_url": "http://arxiv.org/abs/2609.05150v1",
      "pdf_url": "https://arxiv.org/pdf/2609.05150v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
