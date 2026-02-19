const PAPERS_DATA = {
  "last_updated": "2026-02-19 02:49:52 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Measuring Mid-2025 LLM-Assistance on Novice Performance in Biology",
      "authors": [
        "Shen Zhou Hong",
        "Alex Kleinman",
        "Alyssa Mathiowetz",
        "Adam Howes",
        "Julian Cohen",
        "Suveer Ganta",
        "Alex Letizia",
        "Dora Liao",
        "Deepika Pahari",
        "Xavier Roberts-Gaal",
        "Luca Righetti",
        "Joe Torres"
      ],
      "abstract": "Large language models (LLMs) perform strongly on biological benchmarks, raising concerns that they may help novice actors acquire dual-use laboratory skills. Yet, whether this translates to improved human performance in the physical laboratory remains unclear. To address this, we conducted a pre-registered, investigator-blinded, randomized controlled trial (June-August 2025; n = 153) evaluating whether LLMs improve novice performance in tasks that collectively model a viral reverse genetics workflow. We observed no significant difference in the primary endpoint of workflow completion (5.2% LLM vs. 6.6% Internet; P = 0.759), nor in the success rate of individual tasks. However, the LLM arm had numerically higher success rates in four of the five tasks, most notably for the cell culture task (68.8% LLM vs. 55.3% Internet; P = 0.059). Post-hoc Bayesian modeling of pooled data estimates an approximate 1.4-fold increase (95% CrI 0.74-2.62) in success for a \"typical\" reverse genetics task under LLM assistance. Ordinal regression modelling suggests that participants in the LLM arm were more likely to progress through intermediate steps across all tasks (posterior probability of a positive effect: 81%-96%). Overall, mid-2025 LLMs did not substantially increase novice completion of complex laboratory procedures but were associated with a modest performance benefit. These results reveal a gap between in silico benchmarks and real-world utility, underscoring the need for physical-world validation of AI biosecurity assessments as model capabilities and user proficiency evolve.",
      "published": "2026-02-18T18:51:28Z",
      "abstract_url": "http://arxiv.org/abs/2602.16703v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16703v1",
      "categories": [
        "cs.CY",
        "cs.AI"
      ]
    },
    {
      "title": "SPARC: Scenario Planning and Reasoning for Automated C Unit Test Generation",
      "authors": [
        "Jaid Monwar Chowdhury",
        "Chi-An Fu",
        "Reyhaneh Jabbarvand"
      ],
      "abstract": "Automated unit test generation for C remains a formidable challenge due to the semantic gap between high-level program intent and the rigid syntactic constraints of pointer arithmetic and manual memory management. While Large Language Models (LLMs) exhibit strong generative capabilities, direct intent-to-code synthesis frequently suffers from the leap-to-code failure mode, where models prematurely emit code without grounding in program structure, constraints, and semantics. This will result in non-compilable tests, hallucinated function signatures, low branch coverage, and semantically irrelevant assertions that cannot properly capture bugs. We introduce SPARC, a neuro-symbolic, scenario-based framework that bridges this gap through four stages: (1) Control Flow Graph (CFG) analysis, (2) an Operation Map that grounds LLM reasoning in validated utility helpers, (3) Path-targeted test synthesis, and (4) an iterative, self-correction validation loop using compiler and runtime feedback. We evaluate SPARC on 59 real-world and algorithmic subjects, where it outperforms the vanilla prompt generation baseline by 31.36% in line coverage, 26.01% in branch coverage, and 20.78% in mutation score, matching or exceeding the symbolic execution tool KLEE on complex subjects. SPARC retains 94.3% of tests through iterative repair and produces code with significantly higher developer-rated readability and maintainability. By aligning LLM reasoning with program structure, SPARC provides a scalable path for industrial-grade testing of legacy C codebases.",
      "published": "2026-02-18T18:09:03Z",
      "abstract_url": "http://arxiv.org/abs/2602.16671v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16671v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Towards a Science of AI Agent Reliability",
      "authors": [
        "Stephan Rabanser",
        "Sayash Kapoor",
        "Peter Kirgis",
        "Kangheng Liu",
        "Saiteja Utpala",
        "Arvind Narayanan"
      ],
      "abstract": "AI agents are increasingly deployed to execute important tasks. While rising accuracy scores on standard benchmarks suggest rapid progress, many agents still continue to fail in practice. This discrepancy highlights a fundamental limitation of current evaluations: compressing agent behavior into a single success metric obscures critical operational flaws. Notably, it ignores whether agents behave consistently across runs, withstand perturbations, fail predictably, or have bounded error severity. Grounded in safety-critical engineering, we provide a holistic performance profile by proposing twelve concrete metrics that decompose agent reliability along four key dimensions: consistency, robustness, predictability, and safety. Evaluating 14 agentic models across two complementary benchmarks, we find that recent capability gains have only yielded small improvements in reliability. By exposing these persistent limitations, our metrics complement traditional evaluations while offering tools for reasoning about how agents perform, degrade, and fail.",
      "published": "2026-02-18T18:05:44Z",
      "abstract_url": "http://arxiv.org/abs/2602.16666v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16666v1",
      "categories": [
        "cs.AI",
        "cs.CY",
        "cs.LG"
      ]
    },
    {
      "title": "Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment",
      "authors": [
        "Yuyan Bu",
        "Xiaohao Liu",
        "ZhaoXing Ren",
        "Yaodong Yang",
        "Juntao Dai"
      ],
      "abstract": "The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability. In this work, we propose a resource-efficient method for improving multilingual safety alignment. We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision.",
      "published": "2026-02-18T18:01:23Z",
      "abstract_url": "http://arxiv.org/abs/2602.16660v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16660v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Retrieval Augmented Generation of Literature-derived Polymer Knowledge: The Example of a Biodegradable Polymer Expert System",
      "authors": [
        "Sonakshi Gupta",
        "Akhlak Mahmood",
        "Wei Xiong",
        "Rampi Ramprasad"
      ],
      "abstract": "Polymer literature contains a large and growing body of experimental knowledge, yet much of it is buried in unstructured text and inconsistent terminology, making systematic retrieval and reasoning difficult. Existing tools typically extract narrow, study-specific facts in isolation, failing to preserve the cross-study context required to answer broader scientific questions. Retrieval-augmented generation (RAG) offers a promising way to overcome this limitation by combining large language models (LLMs) with external retrieval, but its effectiveness depends strongly on how domain knowledge is represented. In this work, we develop two retrieval pipelines: a dense semantic vector-based approach (VectorRAG) and a graph-based approach (GraphRAG). Using over 1,000 polyhydroxyalkanoate (PHA) papers, we construct context-preserving paragraph embeddings and a canonicalized structured knowledge graph supporting entity disambiguation and multi-hop reasoning. We evaluate these pipelines through standard retrieval metrics, comparisons with general state-of-the-art systems such as GPT and Gemini, and qualitative validation by a domain chemist. The results show that GraphRAG achieves higher precision and interpretability, while VectorRAG provides broader recall, highlighting complementary trade-offs. Expert validation further confirms that the tailored pipelines, particularly GraphRAG, produce well-grounded, citation-reliable responses with strong domain relevance. By grounding every statement in evidence, these systems enable researchers to navigate the literature, compare findings across studies, and uncover patterns that are difficult to extract manually. More broadly, this work establishes a practical framework for building materials science assistants using curated corpora and retrieval design, reducing reliance on proprietary models while enabling trustworthy literature analysis at scale.",
      "published": "2026-02-18T17:46:09Z",
      "abstract_url": "http://arxiv.org/abs/2602.16650v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16650v1",
      "categories": [
        "cs.CE",
        "cs.AI"
      ]
    },
    {
      "title": "Enhanced Diffusion Sampling: Efficient Rare Event Sampling and Free Energy Calculation with Diffusion Models",
      "authors": [
        "Yu Xie",
        "Ludwig Winkler",
        "Lixin Sun",
        "Sarah Lewis",
        "Adam E. Foster",
        "José Jiménez Luna",
        "Tim Hempel",
        "Michael Gastegger",
        "Yaoyi Chen",
        "Iryna Zaporozhets",
        "Cecilia Clementi",
        "Christopher M. Bishop",
        "Frank Noé"
      ],
      "abstract": "The rare-event sampling problem has long been the central limiting factor in molecular dynamics (MD), especially in biomolecular simulation. Recently, diffusion models such as BioEmu have emerged as powerful equilibrium samplers that generate independent samples from complex molecular distributions, eliminating the cost of sampling rare transition events. However, a sampling problem remains when computing observables that rely on states which are rare in equilibrium, for example folding free energies. Here, we introduce enhanced diffusion sampling, enabling efficient exploration of rare-event regions while preserving unbiased thermodynamic estimators. The key idea is to perform quantitatively accurate steering protocols to generate biased ensembles and subsequently recover equilibrium statistics via exact reweighting. We instantiate our framework in three algorithms: UmbrellaDiff (umbrella sampling with diffusion models), $Δ$G-Diff (free-energy differences via tilted ensembles), and MetaDiff (a batchwise analogue for metadynamics). Across toy systems, protein folding landscapes and folding free energies, our methods achieve fast, accurate, and scalable estimation of equilibrium properties within GPU-minutes to hours per system -- closing the rare-event sampling gap that remained after the advent of diffusion-model equilibrium samplers.",
      "published": "2026-02-18T17:26:15Z",
      "abstract_url": "http://arxiv.org/abs/2602.16634v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16634v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG",
        "physics.bio-ph",
        "physics.chem-ph"
      ]
    },
    {
      "title": "Almost Sure Convergence of Differential Temporal Difference Learning for Average Reward Markov Decision Processes",
      "authors": [
        "Ethan Blaser",
        "Jiuqi Wang",
        "Shangtong Zhang"
      ],
      "abstract": "The average reward is a fundamental performance metric in reinforcement learning (RL) focusing on the long-run performance of an agent. Differential temporal difference (TD) learning algorithms are a major advance for average reward RL as they provide an efficient online method to learn the value functions associated with the average reward in both on-policy and off-policy settings. However, existing convergence guarantees require a local clock in learning rates tied to state visit counts, which practitioners do not use and does not extend beyond tabular settings. We address this limitation by proving the almost sure convergence of on-policy $n$-step differential TD for any $n$ using standard diminishing learning rates without a local clock. We then derive three sufficient conditions under which off-policy $n$-step differential TD also converges without a local clock. These results strengthen the theoretical foundations of differential TD and bring its convergence analysis closer to practical implementations.",
      "published": "2026-02-18T17:24:27Z",
      "abstract_url": "http://arxiv.org/abs/2602.16629v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16629v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Systematic Evaluation of Sample-Level Tokenization Strategies for MEG Foundation Models",
      "authors": [
        "SungJun Cho",
        "Chetan Gohil",
        "Rukuang Huang",
        "Oiwi Parker Jones",
        "Mark W. Woolrich"
      ],
      "abstract": "Recent success in natural language processing has motivated growing interest in large-scale foundation models for neuroimaging data. Such models often require discretization of continuous neural time series data, a process referred to as 'tokenization'. However, the impact of different tokenization strategies for neural data is currently poorly understood. In this work, we present a systematic evaluation of sample-level tokenization strategies for transformer-based large neuroimaging models (LNMs) applied to magnetoencephalography (MEG) data. We compare learnable and non-learnable tokenizers by examining their signal reconstruction fidelity and their impact on subsequent foundation modeling performance (token prediction, biological plausibility of generated data, preservation of subject-specific information, and performance on downstream tasks). For the learnable tokenizer, we introduce a novel approach based on an autoencoder. Experiments were conducted on three publicly available MEG datasets spanning different acquisition sites, scanners, and experimental paradigms. Our results show that both learnable and non-learnable discretization schemes achieve high reconstruction accuracy and broadly comparable performance across most evaluation criteria, suggesting that simple fixed sample-level tokenization strategies can be used in the development of neural foundation models. The code is available at https://github.com/OHBA-analysis/Cho2026_Tokenizer.",
      "published": "2026-02-18T17:21:02Z",
      "abstract_url": "http://arxiv.org/abs/2602.16626v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16626v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.NC"
      ]
    },
    {
      "title": "Who can we trust? LLM-as-a-jury for Comparative Assessment",
      "authors": [
        "Mengjie Qian",
        "Guangzhi Sun",
        "Mark J. F. Gales",
        "Kate M. Knill"
      ],
      "abstract": "Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-as-a-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminator strongly correlates with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability.",
      "published": "2026-02-18T17:04:02Z",
      "abstract_url": "http://arxiv.org/abs/2602.16610v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16610v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models",
      "authors": [
        "Melkamu Abay Mersha",
        "Jugal Kalita"
      ],
      "abstract": "Transformer models achieve state-of-the-art performance across domains and tasks, yet their deeply layered representations make their predictions difficult to interpret. Existing explainability methods rely on final-layer attributions, capture either local token-level attributions or global attention patterns without unification, and lack context-awareness of inter-token dependencies and structural components. They also fail to capture how relevance evolves across layers and how structural components shape decision-making. To address these limitations, we proposed the \\textbf{Context-Aware Layer-wise Integrated Gradients (CA-LIG) Framework}, a unified hierarchical attribution framework that computes layer-wise Integrated Gradients within each Transformer block and fuses these token-level attributions with class-specific attention gradients. This integration yields signed, context-sensitive attribution maps that capture supportive and opposing evidence while tracing the hierarchical flow of relevance through the Transformer layers. We evaluate the CA-LIG Framework across diverse tasks, domains, and transformer model families, including sentiment analysis and long and multi-class document classification with BERT, hate speech detection in a low-resource language setting with XLM-R and AfroLM, and image classification with Masked Autoencoder vision Transformer model. Across all tasks and architectures, CA-LIG provides more faithful attributions, shows stronger sensitivity to contextual dependencies, and produces clearer, more semantically coherent visualizations than established explainability methods. These results indicate that CA-LIG provides a more comprehensive, context-aware, and reliable explanation of Transformer decision-making, advancing both the practical interpretability and conceptual understanding of deep neural models.",
      "published": "2026-02-18T17:03:10Z",
      "abstract_url": "http://arxiv.org/abs/2602.16608v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16608v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving",
      "authors": [
        "Chia-chi Hsieh",
        "Zan Zong",
        "Xinyang Chen",
        "Jianjiang Li",
        "Jidong Zhai",
        "Lijie Wen"
      ],
      "abstract": "The growing demand for large language models (LLMs) requires serving systems to handle many concurrent requests with diverse service level objectives (SLOs). This exacerbates head-of-line (HoL) blocking during the compute-intensive prefill phase, where long-running requests monopolize resources and delay higher-priority ones, leading to widespread time-to-first-token (TTFT) SLO violations. While chunked prefill enables interruptibility, it introduces an inherent trade-off between responsiveness and throughput: reducing chunk size improves response latency but degrades computational efficiency, whereas increasing chunk size maximizes throughput but exacerbates blocking. This necessitates an adaptive preemption mechanism. However, dynamically balancing execution granularity against scheduling overheads remains a key challenge. In this paper, we propose FlowPrefill, a TTFT-goodput-optimized serving system that resolves this conflict by decoupling preemption granularity from scheduling frequency. To achieve adaptive prefill scheduling, FlowPrefill introduces two key innovations: 1) Operator-Level Preemption, which leverages operator boundaries to enable fine-grained execution interruption without the efficiency loss associated with fixed small chunking; and 2) Event-Driven Scheduling, which triggers scheduling decisions only upon request arrival or completion events, thereby supporting efficient preemption responsiveness while minimizing control-plane overhead. Evaluation on real-world production traces shows that FlowPrefill improves maximum goodput by up to 5.6$\\times$ compared to state-of-the-art systems while satisfying heterogeneous SLOs.",
      "published": "2026-02-18T16:57:45Z",
      "abstract_url": "http://arxiv.org/abs/2602.16603v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16603v1",
      "categories": [
        "cs.DC",
        "cs.AI"
      ]
    },
    {
      "title": "A Contrastive Learning Framework Empowered by Attention-based Feature Adaptation for Street-View Image Classification",
      "authors": [
        "Qi You",
        "Yitai Cheng",
        "Zichao Zeng",
        "James Haworth"
      ],
      "abstract": "Street-view image attribute classification is a vital downstream task of image classification, enabling applications such as autonomous driving, urban analytics, and high-definition map construction. It remains computationally demanding whether training from scratch, initialising from pre-trained weights, or fine-tuning large models. Although pre-trained vision-language models such as CLIP offer rich image representations, existing adaptation or fine-tuning methods often rely on their global image embeddings, limiting their ability to capture fine-grained, localised attributes essential in complex, cluttered street scenes. To address this, we propose CLIP-MHAdapter, a variant of the current lightweight CLIP adaptation paradigm that appends a bottleneck MLP equipped with multi-head self-attention operating on patch tokens to model inter-patch dependencies. With approximately 1.4 million trainable parameters, CLIP-MHAdapter achieves superior or competitive accuracy across eight attribute classification tasks on the Global StreetScapes dataset, attaining new state-of-the-art results while maintaining low computational cost. The code is available at https://github.com/SpaceTimeLab/CLIP-MHAdapter.",
      "published": "2026-02-18T16:41:32Z",
      "abstract_url": "http://arxiv.org/abs/2602.16590v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16590v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "AIFL: A Global Daily Streamflow Forecasting Model Using Deterministic LSTM Pre-trained on ERA5-Land and Fine-tuned on IFS",
      "authors": [
        "Maria Luisa Taccari",
        "Kenza Tazi",
        "Oisín M. Morrison",
        "Andreas Grafberger",
        "Juan Colonese",
        "Corentin Carton de Wiart",
        "Christel Prudhomme",
        "Cinzia Mazzetti",
        "Matthew Chantry",
        "Florian Pappenberger"
      ],
      "abstract": "Reliable global streamflow forecasting is essential for flood preparedness and water resource management, yet data-driven models often suffer from a performance gap when transitioning from historical reanalysis to operational forecast products. This paper introduces AIFL (Artificial Intelligence for Floods), a deterministic LSTM-based model designed for global daily streamflow forecasting. Trained on 18,588 basins curated from the CARAVAN dataset, AIFL utilises a novel two-stage training strategy to bridge the reanalysis-to-forecast domain shift. The model is first pre-trained on 40 years of ERA5-Land reanalysis (1980-2019) to capture robust hydrological processes, then fine-tuned on operational Integrated Forecasting System (IFS) control forecasts (2016-2019) to adapt to the specific error structures and biases of operational numerical weather prediction. To our knowledge, this is the first global model trained end-to-end within the CARAVAN ecosystem. On an independent temporal test set (2021-2024), AIFL achieves high predictive skill with a median modified Kling-Gupta Efficiency (KGE') of 0.66 and a median Nash-Sutcliffe Efficiency (NSE) of 0.53. Benchmarking results show that AIFL is highly competitive with current state-of-the-art global systems, achieving comparable accuracy while maintaining a transparent and reproducible forcing pipeline. The model demonstrates exceptional reliability in extreme-event detection, providing a streamlined and operationally robust baseline for the global hydrological community.",
      "published": "2026-02-18T16:26:36Z",
      "abstract_url": "http://arxiv.org/abs/2602.16579v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16579v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "physics.app-ph"
      ]
    },
    {
      "title": "Creating a digital poet",
      "authors": [
        "Vered Tohar",
        "Tsahi Hayat",
        "Amir Leshem"
      ],
      "abstract": "Can a machine write good poetry? Any positive answer raises fundamental questions about the nature and value of art. We report a seven-month poetry workshop in which a large language model was shaped into a digital poet through iterative in-context expert feedback, without retraining. Across sessions, the model developed a distinctive style and a coherent corpus, supported by quantitative and qualitative analyses, and it produced a pen name and author image. In a blinded authorship test with 50 humanities students and graduates (three AI poems and three poems by well-known poets each), judgments were at chance: human poems were labeled human 54% of the time and AI poems 52%, with 95% confidence intervals including 50%. After the workshop, a commercial publisher released a poetry collection authored by the model. These results show that workshop-style prompting can support long-horizon creative shaping and renew debates on creativity and authorship.",
      "published": "2026-02-18T16:25:10Z",
      "abstract_url": "http://arxiv.org/abs/2602.16578v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16578v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents",
      "authors": [
        "Doron Shavit"
      ],
      "abstract": "Jailbreak prompts are a practical and evolving threat to large language models (LLMs), particularly in agentic systems that execute tools over untrusted content. Many attacks exploit long-context hiding, semantic camouflage, and lightweight obfuscations that can evade single-pass guardrails. We present RLM-JB, an end-to-end jailbreak detection framework built on Recursive Language Models (RLMs), in which a root model orchestrates a bounded analysis program that transforms the input, queries worker models over covered segments, and aggregates evidence into an auditable decision. RLM-JB treats detection as a procedure rather than a one-shot classification: it normalizes and de-obfuscates suspicious inputs, chunks text to reduce context dilution and guarantee coverage, performs parallel chunk screening, and composes cross-chunk signals to recover split-payload attacks. On AutoDAN-style adversarial inputs, RLM-JB achieves high detection effectiveness across three LLM backends (ASR/Recall 92.5-98.0%) while maintaining very high precision (98.99-100%) and low false positive rates (0.0-2.0%), highlighting a practical sensitivity-specificity trade-off as the screening backend changes.",
      "published": "2026-02-18T15:07:09Z",
      "abstract_url": "http://arxiv.org/abs/2602.16520v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16520v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Framework of Thoughts: A Foundation Framework for Dynamic and Optimized Reasoning based on Chains, Trees, and Graphs",
      "authors": [
        "Felix Fricke",
        "Simon Malberg",
        "Georg Groh"
      ],
      "abstract": "Prompting schemes such as Chain of Thought, Tree of Thoughts, and Graph of Thoughts can significantly enhance the reasoning capabilities of large language models. However, most existing schemes require users to define static, problem-specific reasoning structures that lack adaptability to dynamic or unseen problem types. Additionally, these schemes are often under-optimized in terms of hyperparameters, prompts, runtime, and prompting cost. To address these limitations, we introduce Framework of Thoughts (FoT)--a general-purpose foundation framework for building and optimizing dynamic reasoning schemes. FoT comes with built-in features for hyperparameter tuning, prompt optimization, parallel execution, and intelligent caching, unlocking the latent performance potential of reasoning schemes. We demonstrate FoT's capabilities by implementing three popular schemes--Tree of Thoughts, Graph of Thoughts, and ProbTree--within FoT. We empirically show that FoT enables significantly faster execution, reduces costs, and achieves better task scores through optimization. We release our codebase to facilitate the development of future dynamic and efficient reasoning schemes.",
      "published": "2026-02-18T14:58:25Z",
      "abstract_url": "http://arxiv.org/abs/2602.16512v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16512v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Interpretability-by-Design with Accurate Locally Additive Models and Conditional Feature Effects",
      "authors": [
        "Vasilis Gkolemis",
        "Loukas Kavouras",
        "Dimitrios Kyriakopoulos",
        "Konstantinos Tsopelas",
        "Dimitrios Rontogiannis",
        "Giuseppe Casalicchio",
        "Theodore Dalamagas",
        "Christos Diou"
      ],
      "abstract": "Generalized additive models (GAMs) offer interpretability through independent univariate feature effects but underfit when interactions are present in data. GA$^2$Ms add selected pairwise interactions which improves accuracy, but sacrifices interpretability and limits model auditing. We propose \\emph{Conditionally Additive Local Models} (CALMs), a new model class, that balances the interpretability of GAMs with the accuracy of GA$^2$Ms. CALMs allow multiple univariate shape functions per feature, each active in different regions of the input space. These regions are defined independently for each feature as simple logical conditions (thresholds) on the features it interacts with. As a result, effects remain locally additive while varying across subregions to capture interactions. We further propose a principled distillation-based training pipeline that identifies homogeneous regions with limited interactions and fits interpretable shape functions via region-aware backfitting. Experiments on diverse classification and regression tasks show that CALMs consistently outperform GAMs and achieve accuracy comparable with GA$^2$Ms. Overall, CALMs offer a compelling trade-off between predictive accuracy and interpretability.",
      "published": "2026-02-18T14:45:33Z",
      "abstract_url": "http://arxiv.org/abs/2602.16503v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16503v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Fast and Scalable Analytical Diffusion",
      "authors": [
        "Xinyi Shang",
        "Peng Sun",
        "Jingyu Lin",
        "Zhiqiang Shen"
      ],
      "abstract": "Analytical diffusion models offer a mathematically transparent path to generative modeling by formulating the denoising score as an empirical-Bayes posterior mean. However, this interpretability comes at a prohibitive cost: the standard formulation necessitates a full-dataset scan at every timestep, scaling linearly with dataset size. In this work, we present the first systematic study addressing this scalability bottleneck. We challenge the prevailing assumption that the entire training data is necessary, uncovering the phenomenon of Posterior Progressive Concentration: the effective golden support of the denoising score is not static but shrinks asymptotically from the global manifold to a local neighborhood as the signal-to-noise ratio increases. Capitalizing on this, we propose Dynamic Time-Aware Golden Subset Diffusion (GoldDiff), a training-free framework that decouples inference complexity from dataset size. Instead of static retrieval, GoldDiff uses a coarse-to-fine mechanism to dynamically pinpoint the ''Golden Subset'' for inference. Theoretically, we derive rigorous bounds guaranteeing that our sparse approximation converges to the exact score. Empirically, GoldDiff achieves a $\\bf 71 \\times$ speedup on AFHQ while matching or achieving even better performance than full-scan baselines. Most notably, we demonstrate the first successful scaling of analytical diffusion to ImageNet-1K, unlocking a scalable, training-free paradigm for large-scale generative modeling.",
      "published": "2026-02-18T14:41:09Z",
      "abstract_url": "http://arxiv.org/abs/2602.16498v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16498v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "From Growing to Looping: A Unified View of Iterative Computation in LLMs",
      "authors": [
        "Ferdinand Kapl",
        "Emmanouil Angelis",
        "Kaitlin Maile",
        "Johannes von Oswald",
        "Stefan Bauer"
      ],
      "abstract": "Looping, reusing a block of layers across depth, and depth growing, training shallow-to-deep models by duplicating middle layers, have both been linked to stronger reasoning, but their relationship remains unclear. We provide a mechanistic unification: looped and depth-grown models exhibit convergent depth-wise signatures, including increased reliance on late layers and recurring patterns aligned with the looped or grown block. These shared signatures support the view that their gains stem from a common form of iterative computation. Building on this connection, we show that the two techniques are adaptable and composable: applying inference-time looping to the middle blocks of a depth-grown model improves accuracy on some reasoning primitives by up to $2\\times$, despite the model never being trained to loop. Both approaches also adapt better than the baseline when given more in-context examples or additional supervised fine-tuning data. Additionally, depth-grown models achieve the largest reasoning gains when using higher-quality, math-heavy cooldown mixtures, which can be further boosted by adapting a middle block to loop. Overall, our results position depth growth and looping as complementary, practical methods for inducing and scaling iterative computation to improve reasoning.",
      "published": "2026-02-18T14:25:16Z",
      "abstract_url": "http://arxiv.org/abs/2602.16490v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16490v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Learn from Language Feedback with Social Meta-Learning",
      "authors": [
        "Jonathan Cook",
        "Diego Antognini",
        "Martin Klissarov",
        "Claudiu Musat",
        "Edward Grefenstette"
      ],
      "abstract": "Large language models (LLMs) often struggle to learn from corrective feedback within a conversational context. They are rarely proactive in soliciting this feedback, even when faced with ambiguity, which can make their dialogues feel static, one-sided, and lacking the adaptive qualities of human conversation. To address these limitations, we draw inspiration from social meta-learning (SML) in humans - the process of learning how to learn from others. We formulate SML as a finetuning methodology, training LLMs to solicit and learn from language feedback in simulated pedagogical dialogues, where static tasks are converted into interactive social learning problems. SML effectively teaches models to use conversation to solve problems they are unable to solve in a single turn. This capability generalises across domains; SML on math problems produces models that better use feedback to solve coding problems and vice versa. Furthermore, despite being trained only on fully-specified problems, these models are better able to solve underspecified tasks where critical information is revealed over multiple turns. When faced with this ambiguity, SML-trained models make fewer premature answer attempts and are more likely to ask for the information they need. This work presents a scalable approach to developing AI systems that effectively learn from language feedback.",
      "published": "2026-02-18T14:22:13Z",
      "abstract_url": "http://arxiv.org/abs/2602.16488v1",
      "pdf_url": "https://arxiv.org/pdf/2602.16488v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    }
  ]
};
