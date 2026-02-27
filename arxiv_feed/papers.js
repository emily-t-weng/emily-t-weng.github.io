const PAPERS_DATA = {
  "last_updated": "2026-02-27 02:43:13 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Mitigating Legibility Tax with Decoupled Prover-Verifier Games",
      "authors": [
        "Yegon Kim",
        "Juho Lee"
      ],
      "abstract": "As large language models become increasingly capable, it is critical that their outputs can be easily checked by less capable systems. Prover-verifier games can be used to improve checkability of model outputs, but display a degradation in accuracy compared to a baseline trained only to maximize correctness -- a phenonemon named legibility tax. We propose a solution by decoupling the correctness from the checkability condition and instead training a \"translator\" model that turns a fixed solver model's solution into a checkable form. This allows us to first train the solver to maximize correctness, and then train the translator to translate the solver into a checkable form while retaining the solver's answer. To accommodate this new objective of translation, we formulate a decoupled prover-verifier game where the equilibria correspond to faithful and checkable translators.",
      "published": "2026-02-26T17:25:22Z",
      "abstract_url": "http://arxiv.org/abs/2602.23248v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23248v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Agency and Architectural Limits: Why Optimization-Based Systems Cannot Be Norm-Responsive",
      "authors": [
        "Radha Sarma"
      ],
      "abstract": "AI systems are increasingly deployed in high-stakes contexts -- medical diagnosis, legal research, financial analysis -- under the assumption they can be governed by norms. This paper demonstrates that assumption is formally invalid for optimization-based systems, specifically Large Language Models trained via Reinforcement Learning from Human Feedback (RLHF). We establish that genuine agency requires two necessary and jointly sufficient architectural conditions: the capacity to maintain certain boundaries as non-negotiable constraints rather than tradeable weights (Incommensurability), and a non-inferential mechanism capable of suspending processing when those boundaries are threatened (Apophatic Responsiveness). These conditions apply across all normative domains. RLHF-based systems are constitutively incompatible with both conditions. The operations that make optimization powerful -- unifying all values on a scalar metric and always selecting the highest-scoring output -- are precisely the operations that preclude normative governance. This incompatibility is not a correctable training bug awaiting a technical fix; it is a formal constraint inherent to what optimization is. Consequently, documented failure modes - sycophancy, hallucination, and unfaithful reasoning - are not accidents but structural manifestations. Misaligned deployment triggers a second-order risk we term the Convergence Crisis: when humans are forced to verify AI outputs under metric pressure, they degrade from genuine agents into criteria-checking optimizers, eliminating the only component in the system capable of normative accountability. Beyond the incompatibility proof, the paper's primary positive contribution is a substrate-neutral architectural specification defining what any system -- biological, artificial, or institutional -- must satisfy to qualify as an agent rather than a sophisticated instrument.",
      "published": "2026-02-26T17:16:17Z",
      "abstract_url": "http://arxiv.org/abs/2602.23239v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23239v1",
      "categories": [
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Scaling Search Relevance: Augmenting App Store Ranking with LLM-Generated Judgments",
      "authors": [
        "Evangelia Christakopoulou",
        "Vivekkumar Patel",
        "Hemanth Velaga",
        "Sandip Gaikwad"
      ],
      "abstract": "Large-scale commercial search systems optimize for relevance to drive successful sessions that help users find what they are looking for. To maximize relevance, we leverage two complementary objectives: behavioral relevance (results users tend to click or download) and textual relevance (a result's semantic fit to the query). A persistent challenge is the scarcity of expert-provided textual relevance labels relative to abundant behavioral relevance labels. We first address this by systematically evaluating LLM configurations, finding that a specialized, fine-tuned model significantly outperforms a much larger pre-trained one in providing highly relevant labels. Using this optimal model as a force multiplier, we generate millions of textual relevance labels to overcome the data scarcity. We show that augmenting our production ranker with these textual relevance labels leads to a significant outward shift of the Pareto frontier: offline NDCG improves for behavioral relevance while simultaneously increasing for textual relevance. These offline gains were validated by a worldwide A/B test on the App Store ranker, which demonstrated a statistically significant +0.24% increase in conversion rate, with the most substantial performance gains occurring in tail queries, where the new textual relevance labels provide a robust signal in the absence of reliable behavioral relevance labels.",
      "published": "2026-02-26T17:11:26Z",
      "abstract_url": "http://arxiv.org/abs/2602.23234v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23234v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation",
      "authors": [
        "Jiahao Zhao",
        "Feng Jiang",
        "Shaowei Qin",
        "Zhonghui Zhang",
        "Junhao Liu",
        "Guibing Guo",
        "Hamid Alinejad-Rokny",
        "Min Yang"
      ],
      "abstract": "Large language models (LLMs) are increasingly applied in scientific research, offering new capabilities for knowledge discovery and reasoning. In single-cell biology, however, evaluation practices for both general and specialized LLMs remain inadequate: existing benchmarks are fragmented across tasks, adopt formats such as multiple-choice classification that diverge from real-world usage, and rely on metrics lacking interpretability and biological grounding. We present SC-ARENA, a natural language evaluation framework tailored to single-cell foundation models. SC-ARENA formalizes a virtual cell abstraction that unifies evaluation targets by representing both intrinsic attributes and gene-level interactions. Within this paradigm, we define five natural language tasks (cell type annotation, captioning, generation, perturbation prediction, and scientific QA) that probe core reasoning capabilities in cellular biology. To overcome the limitations of brittle string-matching metrics, we introduce knowledge-augmented evaluation, which incorporates external ontologies, marker databases, and scientific literature to support biologically faithful and interpretable judgments. Experiments and analysis across both general-purpose and domain-specialized LLMs demonstrate that (i) under the Virtual Cell unified evaluation paradigm, current models achieve uneven performance on biologically complex tasks, particularly those demanding mechanistic or causal understanding; and (ii) our knowledge-augmented evaluation framework ensures biological correctness, provides interpretable, evidence-grounded rationales, and achieves high discriminative capacity, overcoming the brittleness and opacity of conventional metrics. SC-Arena thus provides a unified and interpretable framework for assessing LLMs in single-cell biology, pointing toward the development of biology-aligned, generalizable foundation models.",
      "published": "2026-02-26T16:50:28Z",
      "abstract_url": "http://arxiv.org/abs/2602.23199v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23199v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering",
      "authors": [
        "Elzo Brito dos Santos Filho"
      ],
      "abstract": "Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (agent.result or issue.report); a deterministic orchestrator validates, persists events in an append-only log (activity.jsonl), applies file-writing effects, and projects a verifiable materialized view (roadmap.json). The proposal incorporates boundary contracts (AGENT_CONTRACT.yaml), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with run.status=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.",
      "published": "2026-02-26T16:45:59Z",
      "abstract_url": "http://arxiv.org/abs/2602.23193v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23193v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring",
      "authors": [
        "Usman Anwar",
        "Julianna Piskorz",
        "David D. Baek",
        "David Africa",
        "Jim Weatherall",
        "Max Tegmark",
        "Christian Schroeder de Witt",
        "Mihaela van der Schaar",
        "David Krueger"
      ],
      "abstract": "Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \\textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \\textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.",
      "published": "2026-02-26T16:27:24Z",
      "abstract_url": "http://arxiv.org/abs/2602.23163v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23163v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CR",
        "cs.IT",
        "cs.MA"
      ]
    },
    {
      "title": "Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs",
      "authors": [
        "Jayadev Billa"
      ],
      "abstract": "Multimodal LLMs can process speech and images, but they cannot hear a speaker's voice or see an object's texture. We show this is not a failure of encoding: speaker identity, emotion, and visual attributes survive through every LLM layer (3--55$\\times$ above chance in linear probes), yet removing 64--71% of modality-specific variance improves decoder loss. The decoder has no learned use for these directions; their presence is noise. We formalize this as a mismatched decoder problem: a decoder trained on text can only extract information along text-aligned directions. Accessible information is bounded by the Generalized Mutual Information (GMI), with degradation scaling with distributional distance and decoder sensitivity. The bound is a property of the decoder's scoring rule, not of any particular architecture; it applies whether non-text inputs arrive through a learned projection, a discrete codebook, or no explicit adapter at all. We validate this across five models spanning speech and vision. A controlled experiment (two Prismatic VLMs differing only in encoder text-alignment) confirms the bottleneck is the decoder's scoring rule, not the encoder or projection. A LoRA intervention demonstrates the fix: training with an emotion objective improves emotion accessibility ($+$7.5%) without affecting other attributes, confirming that the training objective determines what becomes accessible.",
      "published": "2026-02-26T15:52:48Z",
      "abstract_url": "http://arxiv.org/abs/2602.23136v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23136v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DyGnROLE: Modeling Asymmetry in Dynamic Graphs with Node-Role-Oriented Latent Encoding",
      "authors": [
        "Tyler Bonnet",
        "Marek Rei"
      ],
      "abstract": "Real-world dynamic graphs are often directed, with source and destination nodes exhibiting asymmetrical behavioral patterns and temporal dynamics. However, existing dynamic graph architectures largely rely on shared parameters for processing source and destination nodes, with limited or no systematic role-aware modeling. We propose DyGnROLE (Dynamic Graph Node-Role-Oriented Latent Encoding), a transformer-based architecture that explicitly disentangles source and destination representations. By using separate embedding vocabularies and role-semantic positional encodings, the model captures the distinct structural and temporal contexts unique to each role. Critical to the effectiveness of these specialized embeddings in low-label regimes is a self-supervised pretraining objective we introduce: Temporal Contrastive Link Prediction (TCLP). The pretraining uses the full unlabeled interaction history to encode informative structural biases, enabling the model to learn role-specific representations without requiring annotated data. Evaluation on future edge classification demonstrates that DyGnROLE substantially outperforms a diverse set of state-of-the-art baselines, establishing role-aware modeling as an effective strategy for dynamic graph learning.",
      "published": "2026-02-26T15:51:51Z",
      "abstract_url": "http://arxiv.org/abs/2602.23135v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23135v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.SI"
      ]
    },
    {
      "title": "Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection",
      "authors": [
        "Keito Inoshita"
      ],
      "abstract": "In the attention economy, sensational content exposes consumers to excessive emotional stimulation, hindering calm decision-making. This study proposes Multi-Agent LLM-based Emotional deToxification (MALLET), a multi-agent information sanitization system consisting of four agents: Emotion Analysis, Emotion Adjustment, Balance Monitoring, and Personal Guide. The Emotion Analysis Agent quantifies stimulus intensity using a 6-emotion BERT classifier, and the Emotion Adjustment Agent rewrites texts into two presentation modes, BALANCED (neutralized text) and COOL (neutralized text + supplementary text), using an LLM. The Balance Monitoring Agent aggregates weekly information consumption patterns and generates personalized advice, while the Personal Guide Agent recommends a presentation mode according to consumer sensitivity. Experiments on 800 AG News articles demonstrated significant stimulus score reduction (up to 19.3%) and improved emotion balance while maintaining semantic preservation. Near-zero correlation between stimulus reduction and semantic preservation confirmed that the two are independently controllable. Category-level analysis revealed substantial reduction (17.8-33.8%) in Sports, Business, and Sci/Tech, whereas the effect was limited in the World category, where facts themselves are inherently high-stimulus. The proposed system provides a framework for supporting calm information reception of consumers without restricting access to the original text.",
      "published": "2026-02-26T15:37:03Z",
      "abstract_url": "http://arxiv.org/abs/2602.23123v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23123v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design",
      "authors": [
        "Zhuoliang Xie",
        "Fei Liu",
        "Zhenkun Wang",
        "Qingfu Zhang"
      ],
      "abstract": "The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational challenges, particularly for large-scale instances. This study presents AILS-AHD (Adaptive Iterated Local Search with Automatic Heuristic Design), a novel approach that leverages Large Language Models (LLMs) to revolutionize CVRP solving. Our methodology integrates an evolutionary search framework with LLMs to dynamically generate and optimize ruin heuristics within the AILS method. Additionally, we introduce an LLM-based acceleration mechanism to enhance computational efficiency. Comprehensive experimental evaluations against state-of-the-art solvers, including AILS-II and HGS, demonstrate the superior performance of AILS-AHD across both moderate and large-scale instances. Notably, our approach establishes new best-known solutions for 8 out of 10 instances in the CVRPLib large-scale benchmark, underscoring the potential of LLM-driven heuristic design in advancing the field of vehicle routing optimization.",
      "published": "2026-02-26T15:12:23Z",
      "abstract_url": "http://arxiv.org/abs/2602.23092v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23092v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "MoDora: Tree-Based Semi-Structured Document Analysis System",
      "authors": [
        "Bangrui Xu",
        "Qihang Yao",
        "Zirui Tang",
        "Xuanhe Zhou",
        "Yeye He",
        "Shihan Yu",
        "Qianqian Xu",
        "Bin Wang",
        "Guoliang Li",
        "Conghui He",
        "Fan Wu"
      ],
      "abstract": "Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document. To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at https://github.com/weAIDB/MoDora.",
      "published": "2026-02-26T14:48:49Z",
      "abstract_url": "http://arxiv.org/abs/2602.23061v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23061v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CL",
        "cs.DB",
        "cs.LG"
      ]
    },
    {
      "title": "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure",
      "authors": [
        "Jaehong Cho",
        "Hyunmin Choi",
        "Guseul Heo",
        "Jongse Park"
      ],
      "abstract": "Large language model (LLM) serving infrastructures are undergoing a shift toward heterogeneity and disaggregation. Modern deployments increasingly integrate diverse accelerators and near-memory processing technologies, introducing significant hardware heterogeneity, while system software increasingly separates computation, memory, and model components across distributed resources to improve scalability and efficiency. As a result, LLM serving performance is no longer determined by hardware or software choices in isolation, but by their runtime interaction through scheduling, data movement, and interconnect behavior. However, understanding these interactions remains challenging, as existing simulators lack the ability to jointly model heterogeneous hardware and disaggregated serving techniques within a unified, runtime-driven framework. This paper presents LLMServingSim 2.0, a unified system-level simulator designed to make runtime-driven hardware-software interactions in heterogeneous and disaggregated LLM serving infrastructures explicit and analyzable. LLMServingSim 2.0 embeds serving decisions and hardware behavior into a single runtime loop, enabling interaction-aware modeling of batching, routing, offloading, memory, and power. The simulator supports extensible integration of emerging accelerators and memory systems through profile-based modeling, while capturing dynamic serving behavior and system-level effects. We validate LLMServingSim 2.0 against real deployments, showing that it reproduces key performance, memory, and power metrics with an average error of 0.97%, while maintaining simulation times of around 10 minutes even for complex configurations. These results demonstrate that LLMServingSim 2.0 provides a practical bridge between hardware innovation and serving-system design, enabling systematic exploration and co-design for next-generation LLM serving infrastructures.",
      "published": "2026-02-26T14:22:17Z",
      "abstract_url": "http://arxiv.org/abs/2602.23036v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23036v1",
      "categories": [
        "cs.DC",
        "cs.AI"
      ]
    },
    {
      "title": "Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization",
      "authors": [
        "Zeyuan Liu",
        "Jeonghye Kim",
        "Xufang Luo",
        "Dongsheng Li",
        "Yuqing Yang"
      ],
      "abstract": "Exploration remains the key bottleneck for large language model agents trained with reinforcement learning. While prior methods exploit pretrained knowledge, they fail in environments requiring the discovery of novel states. We propose Exploratory Memory-Augmented On- and Off-Policy Optimization (EMPO$^2$), a hybrid RL framework that leverages memory for exploration and combines on- and off-policy updates to make LLMs perform well with memory while also ensuring robustness without it. On ScienceWorld and WebShop, EMPO$^2$ achieves 128.6% and 11.3% improvements over GRPO, respectively. Moreover, in out-of-distribution tests, EMPO$^2$ demonstrates superior adaptability to new tasks, requiring only a few trials with memory and no parameter updates. These results highlight EMPO$^2$ as a promising framework for building more exploratory and generalizable LLM-based agents.",
      "published": "2026-02-26T13:50:57Z",
      "abstract_url": "http://arxiv.org/abs/2602.23008v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23008v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Residual Koopman Spectral Profiling for Predicting and Preventing Transformer Training Instability",
      "authors": [
        "Bum Jun Kim",
        "Shohei Taniguchi",
        "Makoto Kawano",
        "Yusuke Iwasawa",
        "Yutaka Matsuo"
      ],
      "abstract": "Training divergence in transformers wastes compute, yet practitioners discover instability only after expensive runs begin. They therefore need an expected probability of failure for a transformer before training starts. Our study of Residual Koopman Spectral Profiling (RKSP) provides such an estimate. From a single forward pass at initialization, RKSP extracts Koopman spectral features by applying whitened dynamic mode decomposition to layer-wise residual snapshots. Our central diagnostic, the near-unit spectral mass, quantifies the fraction of modes concentrated near the unit circle, which captures instability risk. For predicting divergence across extensive configurations, this estimator achieves an AUROC of 0.995, outperforming the best gradient baseline. We further make this diagnostic actionable through Koopman Spectral Shaping (KSS), which reshapes spectra during training. We empirically validate that our method works in practice: RKSP predicts divergence at initialization, and when RKSP flags high risk, turning on KSS successfully prevents divergence. In the challenging high learning rate regime without normalization layers, KSS reduces the divergence rate from 66.7% to 12.5% and enables learning rates that are 50% to 150% higher. These findings generalize to WikiText-103 language modeling, vision transformers on CIFAR-10, and pretrained language models, including GPT-2 and LLaMA-2 up to 7B, as well as emerging architectures such as MoE, Mamba-style SSMs, and KAN.",
      "published": "2026-02-26T13:33:25Z",
      "abstract_url": "http://arxiv.org/abs/2602.22988v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22988v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search",
      "authors": [
        "Xun Huang",
        "Simeng Qin",
        "Xiaoshuang Jia",
        "Ranjie Duan",
        "Huanqian Yan",
        "Zhitao Zeng",
        "Fei Yang",
        "Yang Liu",
        "Xiaojun Jia"
      ],
      "abstract": "As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in jailbreak attacks. Owing to its conciseness and obscurity, classical Chinese can partially bypass existing safety constraints, exposing notable vulnerabilities in LLMs. Based on this observation, this paper proposes a framework, CC-BOS, for the automatic generation of classical Chinese adversarial prompts based on multi-dimensional fruit fly optimization, facilitating efficient and automated jailbreak attacks in black-box settings. Prompts are encoded into eight policy dimensions-covering role, behavior, mechanism, metaphor, expression, knowledge, trigger pattern and context; and iteratively refined via smell search, visual search, and cauchy mutation. This design enables efficient exploration of the search space, thereby enhancing the effectiveness of black-box jailbreak attacks. To enhance readability and evaluation accuracy, we further design a classical Chinese to English translation module. Extensive experiments demonstrate that effectiveness of the proposed CC-BOS, consistently outperforming state-of-the-art jailbreak attack methods.",
      "published": "2026-02-26T13:25:35Z",
      "abstract_url": "http://arxiv.org/abs/2602.22983v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22983v1",
      "categories": [
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Modeling Expert AI Diagnostic Alignment via Immutable Inference Snapshots",
      "authors": [
        "Dimitrios P. Panagoulias",
        "Evangelia-Aikaterini Tsichrintzi",
        "Georgios Savvidis",
        "Evridiki Tsoureli-Nikita"
      ],
      "abstract": "Human-in-the-loop validation is essential in safety-critical clinical AI, yet the transition between initial model inference and expert correction is rarely analyzed as a structured signal. We introduce a diagnostic alignment framework in which the AI-generated image based report is preserved as an immutable inference state and systematically compared with the physician-validated outcome. The inference pipeline integrates a vision-enabled large language model, BERT- based medical entity extraction, and a Sequential Language Model Inference (SLMI) step to enforce domain-consistent refinement prior to expert review. Evaluation on 21 dermatological cases (21 complete AI physician pairs) em- ployed a four-level concordance framework comprising exact primary match rate (PMR), semantic similarity-adjusted rate (AMR), cross-category alignment, and Comprehensive Concordance Rate (CCR). Exact agreement reached 71.4% and remained unchanged under semantic similarity (t = 0.60), while structured cross-category and differential overlap analysis yielded 100% comprehensive concordance (95% CI: [83.9%, 100%]). No cases demonstrated complete diagnostic divergence. These findings show that binary lexical evaluation substantially un- derestimates clinically meaningful alignment. Modeling expert validation as a structured transformation enables signal-aware quantification of correction dynamics and supports traceable, human aligned evaluation of image based clinical decision support systems.",
      "published": "2026-02-26T13:11:58Z",
      "abstract_url": "http://arxiv.org/abs/2602.22973v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22973v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SPM-Bench: Benchmarking Large Language Models for Scanning Probe Microscopy",
      "authors": [
        "Peiyao Xiao",
        "Xiaogang Li",
        "Chengliang Xu",
        "Jiayi Wang",
        "Ben Wang",
        "Zichao Chen",
        "Zeyu Wang",
        "Kejun Yu",
        "Yueqian Chen",
        "Xulin Liu",
        "Wende Xiao",
        "Bing Zhao",
        "Hu Wei"
      ],
      "abstract": "As LLMs achieved breakthroughs in general reasoning, their proficiency in specialized scientific domains reveals pronounced gaps in existing benchmarks due to data contamination, insufficient complexity, and prohibitive human labor costs. Here we present SPM-Bench, an original, PhD-level multimodal benchmark specifically designed for scanning probe microscopy (SPM). We propose a fully automated data synthesis pipeline that ensures both high authority and low-cost. By employing Anchor-Gated Sieve (AGS) technology, we efficiently extract high-value image-text pairs from arXiv and journal papers published between 2023 and 2025. Through a hybrid cloud-local architecture where VLMs return only spatial coordinates \"llbox\" for local high-fidelity cropping, our pipeline achieves extreme token savings while maintaining high dataset purity. To accurately and objectively evaluate the performance of the LLMs, we introduce the Strict Imperfection Penalty F1 (SIP-F1) score. This metric not only establishes a rigorous capability hierarchy but also, for the first time, quantifies model \"personalities\" (Conservative, Aggressive, Gambler, or Wise). By correlating these results with model-reported confidence and perceived difficulty, we expose the true reasoning boundaries of current AI in complex physical scenarios. These insights establish SPM-Bench as a generalizable paradigm for automated scientific data synthesis.",
      "published": "2026-02-26T13:08:56Z",
      "abstract_url": "http://arxiv.org/abs/2602.22971v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22971v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Discovery of Interpretable Physical Laws in Materials via Language-Model-Guided Symbolic Regression",
      "authors": [
        "Yifeng Guan",
        "Chuyi Liu",
        "Dongzhan Zhou",
        "Lei Bai",
        "Wan-jian Yin",
        "Jingyuan Li",
        "Mao Su"
      ],
      "abstract": "Discovering interpretable physical laws from high-dimensional data is a fundamental challenge in scientific research. Traditional methods, such as symbolic regression, often produce complex, unphysical formulas when searching a vast space of possible forms. We introduce a framework that guides the search process by leveraging the embedded scientific knowledge of large language models, enabling efficient identification of physical laws in the data. We validate our approach by modeling key properties of perovskite materials. Our method mitigates the combinatorial explosion commonly encountered in traditional symbolic regression, reducing the effective search space by a factor of approximately $10^5$. A set of novel formulas for bulk modulus, band gap, and oxygen evolution reaction activity are identified, which not only provide meaningful physical insights but also outperform previous formulas in accuracy and simplicity.",
      "published": "2026-02-26T13:05:05Z",
      "abstract_url": "http://arxiv.org/abs/2602.22967v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22967v1",
      "categories": [
        "physics.comp-ph",
        "cs.AI"
      ]
    },
    {
      "title": "FactGuard: Agentic Video Misinformation Detection via Reinforcement Learning",
      "authors": [
        "Zehao Li",
        "Hongwei Yu",
        "Hao Jiang",
        "Qiang Sheng",
        "Yilong Xu",
        "Baolong Bi",
        "Yang Li",
        "Zhenlong Yuan",
        "Yujun Cai",
        "Zhaoqi Wang"
      ],
      "abstract": "Multimodal large language models (MLLMs) have substantially advanced video misinformation detection through unified multimodal reasoning, but they often rely on fixed-depth inference and place excessive trust in internally generated assumptions, particularly in scenarios where critical evidence is sparse, fragmented, or requires external verification. To address these limitations, we propose FactGuard, an agentic framework for video misinformation detection that formulates verification as an iterative reasoning process built upon MLLMs. FactGuard explicitly assesses task ambiguity and selectively invokes external tools to acquire critical evidence, enabling progressive refinement of reasoning trajectories. To further strengthen this capability, we introduce a two-stage training strategy that combines domain-specific agentic supervised fine-tuning with decision-aware reinforcement learning to optimize tool usage and calibrate risk-sensitive decision making. Extensive experiments on FakeSV, FakeTT, and FakeVV demonstrate FactGuard's state-of-the-art performance and validate its excellent robustness and generalization capacity.",
      "published": "2026-02-26T13:00:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.22963v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22963v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "pMoE: Prompting Diverse Experts Together Wins More in Visual Adaptation",
      "authors": [
        "Shentong Mo",
        "Xufang Luo",
        "Dongsheng Li"
      ],
      "abstract": "Parameter-efficient fine-tuning has demonstrated promising results across various visual adaptation tasks, such as classification and segmentation. Typically, prompt tuning techniques have harnessed knowledge from a single pre-trained model, whether from a general or a specialized medical domain. However, this approach typically overlooks the potential synergies that could arise from integrating diverse domain knowledge within the same tuning process. In this work, we propose a novel Mixture-of-Experts prompt tuning method called pMoE, which leverages the strengths of multiple expert domains through expert-specialized prompt tokens and the learnable dispatcher, effectively combining their expertise in a unified model framework. Our pMoE introduces expert-specific prompt tokens and utilizes a dynamic token dispatching mechanism at various prompt layers to optimize the contribution of each domain expert during the adaptation phase. By incorporating both domain knowledge from diverse experts, the proposed pMoE significantly enhances the model's versatility and applicability to a broad spectrum of tasks. We conduct extensive experiments across 47 adaptation tasks, including both classification and segmentation in general and medical domains. The results demonstrate that our pMoE not only achieves superior performance with a large margin of improvements but also offers an optimal trade-off between computational efficiency and adaptation effectiveness compared to existing methods.",
      "published": "2026-02-26T12:27:06Z",
      "abstract_url": "http://arxiv.org/abs/2602.22938v1",
      "pdf_url": "https://arxiv.org/pdf/2602.22938v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
