const PAPERS_DATA = {
  "last_updated": "2026-04-20 03:39:46 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Using Large Language Models and Knowledge Graphs to Improve the Interpretability of Machine Learning Models in Manufacturing",
      "authors": [
        "Thomas Bayer",
        "Alexander Lohr",
        "Sarah Weiß",
        "Bernd Michelberger",
        "Wolfram Höpken"
      ],
      "abstract": "Explaining Machine Learning (ML) results in a transparent and user-friendly manner remains a challenging task of Explainable Artificial Intelligence (XAI). In this paper, we present a method to enhance the interpretability of ML models by using a Knowledge Graph (KG). We store domain-specific data along with ML results and their corresponding explanations, establishing a structured connection between domain knowledge and ML insights. To make these insights accessible to users, we designed a selective retrieval method in which relevant triplets are extracted from the KG and processed by a Large Language Model (LLM) to generate user-friendly explanations of ML results. We evaluated our method in a manufacturing environment using the XAI Question Bank. Beyond standard questions, we introduce more complex, tailored questions that highlight the strengths of our approach. We evaluated 33 questions, analyzing responses using quantitative metrics such as accuracy and consistency, as well as qualitative ones such as clarity and usefulness. Our contribution is both theoretical and practical: from a theoretical perspective, we present a novel approach for effectively enabling LLMs to dynamically access a KG in order to improve the explainability of ML results. From a practical perspective, we provide empirical evidence showing that such explanations can be successfully applied in real-world manufacturing environments, supporting better decision-making in manufacturing processes.",
      "published": "2026-04-17T17:41:17Z",
      "abstract_url": "http://arxiv.org/abs/2604.16280v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16280v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Learning to Reason with Insight for Informal Theorem Proving",
      "authors": [
        "Yunhe Li",
        "Hao Shi",
        "Bowen Deng",
        "Wei Wang",
        "Mengzhe Ruan",
        "Hanxu Hou",
        "Zhongxiang Dai",
        "Siyang Gao",
        "Chao Wang",
        "Shuang Qiu",
        "Linqi Song"
      ],
      "abstract": "Although most of the automated theorem-proving approaches depend on formal proof systems, informal theorem proving can align better with large language models' (LLMs) strength in natural language processing. In this work, we identify a primary bottleneck in informal theorem proving as a lack of insight, namely the difficulty of recognizing the core techniques required to solve complex problems. To address this, we propose a novel framework designed to cultivate this essential reasoning skill and enable LLMs to perform insightful reasoning. We propose $\\mathtt{DeepInsightTheorem}$, a hierarchical dataset that structures informal proofs by explicitly extracting core techniques and proof sketches alongside the final proof. To fully exploit this dataset, we design a Progressive Multi-Stage SFT strategy that mimics the human learning process, guiding the model from basic proof writing to insightful thinking. Our experiments on challenging mathematical benchmarks demonstrate that this insight-aware generation strategy significantly outperforms baselines. These results demonstrate that teaching models to identify and apply core techniques can substantially improve their mathematical reasoning.",
      "published": "2026-04-17T17:36:21Z",
      "abstract_url": "http://arxiv.org/abs/2604.16278v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16278v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "From Benchmarking to Reasoning: A Dual-Aspect, Large-Scale Evaluation of LLMs on Vietnamese Legal Text",
      "authors": [
        "Van-Truong Le"
      ],
      "abstract": "The complexity of Vietnam's legal texts presents a significant barrier to public access to justice. While Large Language Models offer a promising solution for legal text simplification, evaluating their true capabilities requires a multifaceted approach that goes beyond surface-level metrics. This paper introduces a comprehensive dual-aspect evaluation framework to address this need. First, we establish a performance benchmark for four state-of-the-art large language models (GPT-4o, Claude 3 Opus, Gemini 1.5 Pro, and Grok-1) across three key dimensions: Accuracy, Readability, and Consistency. Second, to understand the \"why\" behind these performance scores, we conduct a large-scale error analysis on a curated dataset of 60 complex Vietnamese legal articles, using a novel, expert-validated error typology. Our results reveal a crucial trade-off: models like Grok-1 excel in Readability and Consistency but compromise on fine-grained legal Accuracy, while models like Claude 3 Opus achieve high Accuracy scores that mask a significant number of subtle but critical reasoning errors. The error analysis pinpoints \\textit{Incorrect Example} and \\textit{Misinterpretation} as the most prevalent failures, confirming that the primary challenge for current LLMs is not summarization but controlled, accurate legal reasoning. By integrating a quantitative benchmark with a qualitative deep dive, our work provides a holistic and actionable assessment of LLMs for legal applications.",
      "published": "2026-04-17T17:28:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.16270v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16270v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Distribution Sharpening: The Importance of Task Rewards",
      "authors": [
        "Sarthak Mittal",
        "Leo Gagnon",
        "Guillaume Lajoie"
      ],
      "abstract": "Frontier models have demonstrated exceptional capabilities following the integration of task-reward-based reinforcement learning (RL) into their training pipelines, enabling systems to evolve from pure reasoning models into sophisticated agents. However, debate persists regarding whether RL genuinely instills new skills within a base model or merely sharpens its existing distribution to elicit latent capabilities. To address this dichotomy, we present an explicit comparison between distribution sharpening and task-reward-based learning, utilizing RL as a tool to implement both paradigms. Our analysis reveals the inherent limitations of distribution sharpening, demonstrating from first principles how and why the optima can be unfavorable and the approach fundamentally unstable. Furthermore, our experiments using Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct and Qwen3-4B-Instruct-2507 on math datasets confirm that sharpening yields limited gains, whereas incorporating task-based reward signal can greatly help achieve robust performance improvements and stable learning.",
      "published": "2026-04-17T17:17:55Z",
      "abstract_url": "http://arxiv.org/abs/2604.16259v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16259v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Joint-Centric Dual Contrastive Alignment with Structure-Preserving and Information-Balanced Regularization",
      "authors": [
        "Habibeh Naderi",
        "Behrouz Haji Soleimani",
        "Stan Matwin"
      ],
      "abstract": "We propose HILBERT (HIerarchical Long-sequence Balanced Embedding with Reciprocal contrastive Training), a cross-attentive multimodal framework for learning document-level audio-text representations from long, segmented sequences in low-resource data settings. HILBERT leverages frozen pre-trained speech and language encoders to extract segment-level features, which are aggregated via cross-modal attention and self-attentive pooling to form modality-specific document representations and a joint cross-attentive embedding. To align modalities while preserving modality-specific structure under severe audio-text dimensional imbalance, we introduce a reciprocal dual contrastive objective that simultaneously aligns audio-to-joint and text-to-joint representations, rather than directly contrasting audio and text alone. Two auxiliary regularizers further stabilize long-sequence fusion: a Centered Kernel Alignment (CKA) loss that preserves structural consistency between each modality and the joint embedding, and a mutual information balancing loss that prevents dominance of a single modality by equalizing information flow from audio and text into the joint space. For downstream prediction, HILBERT employs a Mixture-of-Experts (MoE) classifier over concatenated audio, text, and joint representations to accommodate heterogeneous label regimes. Extensive evaluation across multiple audio-text backbone combinations demonstrates that HILBERT learns semantically meaningful long-sequence representations and achieves superior performance on highly imbalanced multi-class settings.",
      "published": "2026-04-17T17:07:35Z",
      "abstract_url": "http://arxiv.org/abs/2604.16247v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16247v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "BAGEL: Benchmarking Animal Knowledge Expertise in Language Models",
      "authors": [
        "Jiacheng Shen",
        "Masato Hagiwara",
        "Milad Alizadeh",
        "Ellen Gilsenan-McMahon",
        "Marius Miron",
        "David Robinson",
        "Emmanuel Chemla",
        "Sara Keen",
        "Gagan Narula",
        "Mathieu Laurière",
        "Matthieu Geist",
        "Olivier Pietquin"
      ],
      "abstract": "Large language models have shown strong performance on broad-domain knowledge and reasoning benchmarks, but it remains unclear how well language models handle specialized animal-related knowledge under a unified closed-book evaluation protocol. We introduce BAGEL, a benchmark for evaluating animal knowledge expertise in language models. BAGEL is constructed from diverse scientific and reference sources, including bioRxiv, Global Biotic Interactions, Xeno-canto, and Wikipedia, using a combination of curated examples and automatically generated closed-book question-answer pairs. The benchmark covers multiple aspects of animal knowledge, including taxonomy, morphology, habitat, behavior, vocalization, geographic distribution, and species interactions. By focusing on closed-book evaluation, BAGEL measures animal-related knowledge of models without external retrieval at inference time. BAGEL further supports fine-grained analysis across source domains, taxonomic groups, and knowledge categories, enabling a more precise characterization of model strengths and systematic failure modes. Our benchmark provides a new testbed for studying domain-specific knowledge generalization in language models and for improving their reliability in biodiversity-related applications.",
      "published": "2026-04-17T17:00:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.16241v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16241v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Neuro-Symbolic ODE Discovery with Latent Grammar Flow",
      "authors": [
        "Karin Yu",
        "Eleni Chatzi",
        "Georgios Kissas"
      ],
      "abstract": "Understanding natural and engineered systems often relies on symbolic formulations, such as differential equations, which provide interpretability and transferability beyond black-box models. We introduce Latent Grammar Flow (LGF), a neuro-symbolic generative framework for discovering ordinary differential equations from data. LGF embeds equations as grammar-based representations into a discrete latent space and forces semantically similar equations to be positioned closer together with a behavioural loss. Then, a discrete flow model guides the sampling process to recursively generate candidate equations that best fit the observed data. Domain knowledge and constraints, such as stability, can be either embedded into the rules or used as conditional predictors.",
      "published": "2026-04-17T16:46:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.16232v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16232v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CE",
        "cs.SC"
      ]
    },
    {
      "title": "Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations",
      "authors": [
        "Yanli Wang",
        "Peng Kuang",
        "Xiaoyu Han",
        "Kaidi Xu",
        "Haohan Wang"
      ],
      "abstract": "Large language models are increasingly deployed in settings where reliability matters, yet output-level uncertainty signals such as token probabilities, entropy, and self-consistency can become brittle under calibration--deployment mismatch. Conformal prediction provides finite-sample validity under exchangeability, but its practical usefulness depends on the quality of the nonconformity score. We propose a conformal framework for LLM question answering that uses internal representations rather than output-facing statistics: specifically, we introduce Layer-Wise Information (LI) scores, which measure how conditioning on the input reshapes predictive entropy across model depth, and use them as nonconformity scores within a standard split conformal pipeline. Across closed-ended and open-domain QA benchmarks, with the clearest gains under cross-domain shift, our method achieves a better validity--efficiency trade-off than strong text-level baselines while maintaining competitive in-domain reliability at the same nominal risk level. These results suggest that internal representations can provide more informative conformal scores when surface-level uncertainty is unstable under distribution shift.",
      "published": "2026-04-17T16:28:31Z",
      "abstract_url": "http://arxiv.org/abs/2604.16217v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16217v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "ChemGraph-XANES: An Agentic Framework for XANES Simulation and Analysis",
      "authors": [
        "Vitor F. Grizzi",
        "Thang Duc Pham",
        "Luke N. Pretzie",
        "Jiayi Xu",
        "Murat Keceli",
        "Cong Liu"
      ],
      "abstract": "Computational X-ray absorption near-edge structure (XANES) is widely used to probe local coordination environments, oxidation states, and electronic structure in chemically complex systems. However, the use of computational XANES at scale is constrained more by workflow complexity than by the underlying simulation method itself. To address this challenge, we present ChemGraph-XANES, an agentic framework for automated XANES simulation and analysis that unifies natural-language task specification, structure acquisition, FDMNES input generation, task-parallel execution, spectral normalization, and provenance-aware data curation. Built on ASE, FDMNES, Parsl, and a LangGraph/LangChain-based tool interface, the framework exposes XANES workflow operations as typed Python tools that can be orchestrated by large language model (LLM) agents. In multi-agent mode, a retrieval-augmented expert agent consults the FDMNES manual to ground parameter selection, while executor agents translate user requests into structured tool calls. We demonstrate documentation-grounded parameter retrieval and show that the same workflow supports both explicit structure-file inputs and chemistry-level natural-language requests. Because independent XANES calculations are naturally task-parallel, the framework is well suited for high-throughput deployment on high-performance computing (HPC) systems, enabling scalable XANES database generation for downstream analysis and machine-learning applications. ChemGraph-XANES thus provides a reproducible and extensible workflow layer for physics-based XANES simulation, spectral curation, and agent-compatible computational spectroscopy.",
      "published": "2026-04-17T16:15:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.16205v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16205v1",
      "categories": [
        "cond-mat.mtrl-sci",
        "cs.AI",
        "physics.chem-ph"
      ]
    },
    {
      "title": "Synthetic data in cryptocurrencies using generative models",
      "authors": [
        "André Saimon S. Sousa",
        "Otto Pires",
        "Frank Acasiete",
        "Oscar M. Granados",
        "Valéria Loureiro da Silva",
        "Hugo Saba"
      ],
      "abstract": "Data plays a fundamental role in consolidating markets, services, and products in the digital financial ecosystem. However, the use of real data, especially in the financial context, can lead to privacy risks and access restrictions, affecting institutions, research, and modeling processes. Although not all financial datasets present such limitations, this work proposes the use of deep learning techniques for generating synthetic data applied to cryptocurrency price time series. The approach is based on Conditional Generative Adversarial Networks (CGANs), combining an LSTM-type recurrent generator and an MLP discriminator to produce statistically consistent synthetic data. The experiments consider different crypto-assets and demonstrate that the model is capable of reproducing relevant temporal patterns, preserving market trends and dynamics. The generation of synthetic series through GANs is an efficient alternative for simulating financial data, showing potential for applications such as market behavior analysis and anomaly detection, with lower computational cost compared to more complex generative approaches.",
      "published": "2026-04-17T15:48:05Z",
      "abstract_url": "http://arxiv.org/abs/2604.16182v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16182v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models",
      "authors": [
        "Alexandra Dragomir",
        "Ioana Pintilie",
        "Antonio Barbalau",
        "Marius Dragoi",
        "Florin Brad",
        "Cristian Daniel Paduraru",
        "Alexandru Tifrea",
        "Elena Burceanu",
        "Radu Tudor Ionescu"
      ],
      "abstract": "Adapter-based methods have become a cost-effective approach to continual learning (CL) for Large Language Models (LLMs), by sequentially learning a low-rank update matrix for each task. To mitigate catastrophic forgetting, state-of-the-art approaches impose constraints on new adapters with respect to the previous ones, by targeting either subspace or coordinate-wise interference. In this paper, we propose JumpLoRA, a novel framework to adaptively induce sparsity in the Low-Rank Adaptation (LoRA) blocks through the use of JumpReLU gating. The method achieves dynamic parameter isolation, which helps prevent task interference. We demonstrate that our method is highly modular and compatible with LoRA-based CL approaches. Specifically, it significantly boosts the performance of IncLoRA and outperforms the leading state-of-the-art CL method, ELLA.",
      "published": "2026-04-17T15:38:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.16171v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16171v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency",
      "authors": [
        "Max Henning Höth",
        "Kristian Kersting",
        "Björn Deiseroth",
        "Letitia Parcalabescu"
      ],
      "abstract": "Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks. Yet ensuring that the reasoning trace both contributes to and faithfully reflects the processes underlying the model's final answer, rather than merely accompanying it, remains challenging. We introduce AtManRL, a method that leverages differentiable attention manipulation to learn more faithful reasoning through reinforcement learning. By training an additive attention mask that identifies tokens in the CoT crucial for producing correct answers, we derive a saliency reward signal that encourages the model to generate reasoning traces that genuinely influence its final predictions. We integrate this saliency reward with outcome-based rewards within the GRPO framework to jointly optimize for correctness and interpretability. Experiments on GSM8K and MMLU with Llama-3.2-3B-Instruct demonstrate that our approach can identify influential reasoning tokens and enable training more transparent reasoning models.",
      "published": "2026-04-17T15:27:35Z",
      "abstract_url": "http://arxiv.org/abs/2604.16158v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16158v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Training Time Prediction for Mixed Precision-based Distributed Training",
      "authors": [
        "Minchul Kang",
        "Changyong Shin",
        "Jinwoo Jeong",
        "Hyunho Lee",
        "Younghun Go",
        "Gyeongmin Kim",
        "Gyeongsik Yang",
        "Chuck Yoo"
      ],
      "abstract": "Accurate prediction of training time in distributed deep learning is crucial for resource allocation, cost estimation, and job scheduling. We observe that the floating-point precision setting is a key determinant of training time, leading to training time variations of ~2.4x over its minimum. However, existing studies on distributed training time prediction rely on static model computation graphs that do not capture precision variations, including mixed precision. According to our experiments, training time prediction without considering precision results in significant prediction errors - reaching up to 147.85% in mean absolute percentage error (MAPE). To address this issue, we propose a precision-aware distributed training time predictor that achieves robust accuracy across diverse precision settings, including mixed precision, with 9.8% MAPE.",
      "published": "2026-04-17T15:18:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.16145v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16145v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC",
        "cs.PF"
      ]
    },
    {
      "title": "Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs Coding the Interviews of Firearm Violence Survivors",
      "authors": [
        "Jessica H. Zhu",
        "Shayla Stringfield",
        "Vahe Zaprosyan",
        "Michael Wagner",
        "Michel Cukier",
        "Joseph B. Richardson"
      ],
      "abstract": "Firearm violence is a pressing public health issue, yet research into survivors' lived experiences remains underfunded and difficult to scale. Qualitative research, including in-depth interviews, is a valuable tool for understanding the personal and societal consequences of community firearm violence and designing effective interventions. However, manually analyzing these narratives through thematic analysis and inductive coding is time-consuming and labor-intensive. Recent advancements in large language models (LLMs) have opened the door to automating this process, though concerns remain about whether these models can accurately and ethically capture the experiences of vulnerable populations. In this study, we assess the use of open-source LLMs to inductively code interviews with 21 Black men who have survived community firearm violence. Our results demonstrate that while some configurations of LLMs can identify important codes, overall relevance remains low and is highly sensitive to data processing. Furthermore, LLM guardrails lead to substantial narrative erasure. These findings highlight both the potential and limitations of LLM-assisted qualitative coding and underscore the ethical challenges of applying AI in research involving marginalized communities.",
      "published": "2026-04-17T15:07:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.16132v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16132v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "SCRIPT: Implementing an Intelligent Tutoring System for Programming in a German University Context",
      "authors": [
        "Alina Deriyeva",
        "Jesper Dannath",
        "Benjamin Paassen"
      ],
      "abstract": "Practice and extensive exercises are essential in programming education. Intelligent tutoring systems (ITSs) are a viable option to provide individualized hints and advice to programming students even when human tutors are not available. However, prior ITS for programming rarely support the Python programming language, mostly focus on introductory programming, and rarely take recent developments in generative models into account. We aim to establish a novel ITS for Python programming that is highly adaptable, serves both as a teaching and research platform, provides interfaces to plug in hint mechanisms (e.g.\\ via large language models), and works inside the particularly challenging regulatory environment of Germany, that is, conforming to the European data protection regulation, the European AI act, and ethical framework of the German Research Foundation. In this paper, we present the description of the current state of the ITS along with future development directions, as well as discuss the challenges and opportunities for improving the system.",
      "published": "2026-04-17T14:53:38Z",
      "abstract_url": "http://arxiv.org/abs/2604.16117v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16117v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "The Relic Condition: When Published Scholarship Becomes Material for Its Own Replacement",
      "authors": [
        "Lin Deng",
        "Chang-bo Liu"
      ],
      "abstract": "We extracted the scholarly reasoning systems of two internationally prominent humanities and social science scholars from their published corpora alone, converted those systems into structured inference-time constraints for a large language model, and tested whether the resulting scholar-bots could perform core academic functions at expert-assessed quality. The distillation pipeline used an eight-layer extraction method and a nine-module skill architecture grounded in local, closed-corpus analysis. The scholar-bots were then deployed across doctoral supervision, peer review, lecturing and panel-style academic exchange. Expert assessment involved three senior academics producing reports and appointment-level syntheses. Across the preserved expert record, all review and supervision reports judged the outputs benchmark-attaining, appointment-level recommendations placed both bots at or above Senior Lecturer level in the Australian university system, and recovered panel scores placed Scholar A between 7.9 and 8.9/10 and Scholar B between 8.5 and 8.9/10 under multi-turn debate conditions. A research-degree-student survey showed high performance ratings across information reliability, theoretical depth and logical rigor, with pronounced ceiling effects on a 7-point scale, despite all participants already being frontier-model users. We term this the Relic condition: when publication systems make stable reasoning architectures legible, extractable and cheaply deployable, the public record of intellectual labor becomes raw material for its own functional replacement. Because the technical threshold for this transition is already crossed at modest engineering effort, we argue that the window for protective frameworks covering disclosure, consent, compensation and deployment restriction is the present, while deployment remains optional rather than infrastructural.",
      "published": "2026-04-17T14:52:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.16116v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16116v1",
      "categories": [
        "cs.ET",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Stylistic-STORM (ST-STORM) : Perceiving the Semantic Nature of Appearance",
      "authors": [
        "Hamed Ouattara",
        "Pierre Duthon",
        "Pascal Houssam Salmane",
        "Frédéric Bernardin",
        "Omar Ait Aider"
      ],
      "abstract": "One of the dominant paradigms in self-supervised learning (SSL), illustrated by MoCo or DINO, aims to produce robust representations by capturing features that are insensitive to certain image transformations such as illumination, or geometric changes. This strategy is appropriate when the objective is to recognize objects independently of their appearance. However, it becomes counterproductive as soon as appearance itself constitutes the discriminative signal. In weather analysis, for example, rain streaks, snow granularity, atmospheric scattering, as well as reflections and halos, are not noise: they carry the essential information. In critical applications such as autonomous driving, ignoring these cues is risky, since grip and visibility depend directly on ground conditions and atmospheric conditions. We introduce ST-STORM, a hybrid SSL framework that treats appearance (style) as a semantic modality to be disentangled from content. Our architecture explicitly separates two latent streams, regulated by gating mechanisms. The Content branch aims at a stable semantic representation through a JEPA scheme coupled with a contrastive objective, promoting invariance to appearance variations. In parallel, the Style branch is constrained to capture appearance signatures (textures, contrasts, scattering) through feature prediction and reconstruction under an adversarial constraint. We evaluate ST-STORM on several tasks, including object classification (ImageNet-1K), fine-grained weather characterization, and melanoma detection (ISIC 2024 Challenge). The results show that the Style branch effectively isolates complex appearance phenomena (F1=97% on Multi-Weather and F1=94% on ISIC 2024 with 10% labeled data), without degrading the semantic performance (F1=80% on ImageNet-1K) of the Content branch, and improves the preservation of critical appearance",
      "published": "2026-04-17T14:15:12Z",
      "abstract_url": "http://arxiv.org/abs/2604.16086v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16086v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "Unveiling Stochasticity: Universal Multi-modal Probabilistic Modeling for Traffic Forecasting",
      "authors": [
        "Weijiang Xiong",
        "Robert Fonod",
        "Nikolas Geroliminis"
      ],
      "abstract": "Traffic forecasting is a challenging spatio-temporal modeling task and a critical component of urban transportation management. Current studies mainly focus on deterministic predictions, with limited considerations on the uncertainty and stochasticity in traffic dynamics. Therefore, this paper proposes an elegant yet universal approach that transforms existing models into probabilistic predictors by replacing only the final output layer with a novel Gaussian Mixture Model (GMM) layer. The modified model requires no changes to the training pipeline and can be trained using only the Negative Log-Likelihood (NLL) loss, without any auxiliary or regularization terms. Experiments on multiple traffic datasets show that our approach generalizes from classic to modern model architectures while preserving deterministic performance. Furthermore, we propose a systematic evaluation procedure based on cumulative distributions and confidence intervals, and demonstrate that our approach is considerably more accurate and informative than unimodal or deterministic baselines. Finally, a more detailed study on a real-world dense urban traffic network is presented to examine the impact of data quality on uncertainty quantification and to show the robustness of our approach under imperfect data conditions. Code available at https://github.com/Weijiang-Xiong/OpenSkyTraffic",
      "published": "2026-04-17T14:10:16Z",
      "abstract_url": "http://arxiv.org/abs/2604.16084v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16084v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Early Detection of Acute Myeloid Leukemia (AML) Using YOLOv12 Deep Learning Model",
      "authors": [
        "Enas E. Ahmed",
        "Salah A. Aly",
        "Mayar Moner"
      ],
      "abstract": "Acute Myeloid Leukemia (AML) is one of the most life-threatening type of blood cancers, and its accurate classification is considered and remains a challenging task due to the visual similarity between various cell types. This study addresses the classification of the multiclasses of AML cells Utilizing YOLOv12 deep learning model. We applied two segmentation approaches based on cell and nucleus features, using Hue channel and Otsu thresholding techniques to preprocess the images prior to classification. Our experiments demonstrate that YOLOv12 with Otsu thresholding on cell-based segmentation achieved the highest level of validation and test accuracy, both reaching 99.3%.",
      "published": "2026-04-17T14:07:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.16082v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16082v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Prototype-Grounded Concept Models for Verifiable Concept Alignment",
      "authors": [
        "Stefano Colamonaco",
        "David Debot",
        "Pietro Barbiero",
        "Giuseppe Marra"
      ],
      "abstract": "Concept Bottleneck Models (CBMs) aim to improve interpretability in Deep Learning by structuring predictions through human-understandable concepts, but they provide no way to verify whether learned concepts align with the human's intended meaning, hurting interpretability. We introduce Prototype-Grounded Concept Models (PGCMs), which ground concepts in learned visual prototypes: image parts that serve as explicit evidence for the concepts. This grounding enables direct inspection of concept semantics and supports targeted human intervention at the prototype level to correct misalignments. Empirically, PGCMs match the predictive performance of state-of-the-art CBMs while substantially improving transparency, interpretability, and intervenability.",
      "published": "2026-04-17T14:04:14Z",
      "abstract_url": "http://arxiv.org/abs/2604.16076v1",
      "pdf_url": "https://arxiv.org/pdf/2604.16076v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.NE"
      ]
    }
  ]
};
