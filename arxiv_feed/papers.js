const PAPERS_DATA = {
  "last_updated": "2026-10-21 04:21:53 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Joints, Camera, Action! Co-Generating 3D Motion and Video for Arbitrary Rigs",
      "authors": [
        "Noam Rotstein",
        "Yoad Tewel",
        "Chen Tessler",
        "Haggai Maron",
        "Gal Chechik",
        "Or Litany",
        "Yoni Kasten",
        "Yuval Atzmon"
      ],
      "abstract": "Video generation models can now depict remarkably realistic motion. Yet it is unclear how to use this motion beyond pixels: to animate a specific 3D asset and replay the animation in its 3D scene. For instance, a video model can show a monkey jumping over a crossbar, but it cannot return that jump as replayable 3D motion placed in the 3D scene. Here we present a model that directly co-generates the video and 3D motion for any supplied rig, along with a camera that registers this motion to the scene. Prior video-motion models either handle only human skeletons, often adding billions of parameters, or triangulate 3D motion from generated 2D pose maps. We instead keep 3D motion in a separate model that supports arbitrary skeletons, and connect it to a pretrained video model through a small set of cross-attention layers. This separation lets the motion model first learn how skeletons move from motion-only data, which is easier to obtain and augment than paired video. Additionally, our model learns to locate each joint in the video frames, which lets us recover the camera and place the 3D motion in the scene. To train across many skeletons, we use a generative video editor to place animated 3D assets in scenes suited to their actions, curating the edits so their known motion remains valid. Together with human and robot video, this yields nearly 200K training pairs. Our evaluations demonstrate that our model animates diverse rigs from text more faithfully than video-first pipelines, and its motion responds to the surrounding scene. It also turns video into rig motion, rig motion into video, and enables keyframe-based control of the 3D motion. Videos and additional results are available on our project page: https://research.nvidia.com/labs/par/projects/jca/.",
      "published": "2026-10-20T17:59:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.28632v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28632v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "GBClaw: An agentic AI framework for atomistic grain boundary research and mechanistic discovery",
      "authors": [
        "Yazhuo Liu",
        "Kunqing Ding",
        "Xiaoguo Gong",
        "Zifeng Wang",
        "Yichen Yang",
        "Yifeng Che",
        "Wen Chen",
        "Josh Kacher",
        "Olivier Pierron",
        "Shigenobu Ogata",
        "Ting Zhu"
      ],
      "abstract": "Agentic artificial intelligence (AI) systems hold immense potential for automating complex workflows in materials discovery and design, but extending them to domain-intensive mechanistic investigations remains challenging. Here, we present GBClaw, an agentic AI framework for investigating the structure and deformation of general grain boundaries (GBs). Although general GBs constitute the vast majority of real-world interfaces in polycrystalline materials, they remain understudied because their structural complexity and interfacial mismatch complicate experimental and modeling research. GBClaw integrates large language models (LLMs) into a modular architecture comprising an orchestrator, specialized agents, domain-specific tools, simulation interfaces, and hierarchical memory. Through dynamic replanning and iterative workflow refinement, the framework effectively connects systematic simulations with targeted mechanistic analysis. Applied to 14 general asymmetric tilt GBs in nickel, GBClaw characterized interfacial defect structures and identified diverse deformation modes across these incommensurate boundaries. Comparing boundary sliding with and without secondary GB misfit dislocations revealed substantial differences in shear strength, underscoring the critical role of defect hierarchies in sliding resistance. Furthermore, researcher-guided follow-up calculations traced an anomalously high shear resistance to metastable boundary structures. This work demonstrates that combining agentic AI with domain expertise bridges high-throughput computational screening with deep physical understanding of complex material interfaces.",
      "published": "2026-10-20T17:59:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.28629v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28629v1",
      "categories": [
        "cond-mat.mtrl-sci",
        "cond-mat.mes-hall",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "Adaptive Multi-Token Prediction: How far should we go?",
      "authors": [
        "Emile Anand",
        "Mohammed Rehan Parwani",
        "Xinyuan Cao"
      ],
      "abstract": "Multi-token prediction (MTP) can accelerate language-model decoding by drafting several tokens and verifying them in a single target-model pass. However, the best speculation horizon changes with context: longer drafts offer greater amortization when reliable, but can waste compute on suffixes discarded after a rejection. Unlike autoregressive drafting, where a stopping rule can decide token by token, MTP must fix the horizon before drafting, as all heads fire in the same forward pass. We therefore formulate horizon selection as an online control problem, using verification feedback to learn how far the draft can be trusted in each context, and introduce Adaptive Multi-Token Prediction (AMTP): a lightweight controller that requires no retraining and leaves the exact verifier unchanged. To guide horizon selection, we characterize the expected number of tokens committed at each horizon via acceptance survival, the probability that a draft prefix is fully accepted. Before each round, AMTP predicts how far the draft is likely to survive and picks the horizon with the highest expected tokens per second, learning freely from every verification pass. Theoretically, we show that any online learner's prediction error translates into throughput regret against the best horizon for each context, and in turn into a bound on wall-clock completion time. With online gradient descent, this regret is sublinear. Across five draft-target pairings built on Llama-3.1-8B-Instruct with Medusa heads, Meta's MTP models, and Qwen3-4B-Inst-2507-MTP, evaluated on AIME 2025 and GSM8K, AMTP decodes up to 4.1x faster than standard decoding and matches or outperforms the fixed window in 8 of 10 settings, with throughput gains of up to 10%.",
      "published": "2026-10-20T17:56:32Z",
      "abstract_url": "http://arxiv.org/abs/2610.28619v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28619v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "Semantic Observability for Autonomous Systems Diagnostics",
      "authors": [
        "Jitendra Solanki",
        "Anand Todkar",
        "Ziran Min"
      ],
      "abstract": "Modern distributed microservice deployments generate massive volumes of heterogeneous, multi-format observability logs across physically and logically partitioned infrastructure. Isolating root causes during cascading failures remains an arduous task for Site Reliability Engineers (SREs), bottlenecked by manual keyword filtering and disjointed temporal correlation across monitoring dashboards. We introduce Semantic Observability, an autonomous multi-agent framework that unifies disparate log streams into a dynamic, queryable Property Graph Database. By modeling physical infrastructure topologies, logical software dependencies, and chronological event sequences as attributed nodes and directed relations, it bridges the semantic gap inherent in flat, unstructured log repositories. Building on this unified substrate, we deploy a stateful multi-agent system orchestrated through Directed Cyclic Graphs (DCGs). The system autonomously navigates the topology, applying causal graph algorithms and temporal back-traversal to isolate the precise blast radius of anomalies. The pruned subgraph is then interpreted by a Large Language Model (LLM) reasoning engine to generate contextualized, code-level Root Cause Analyses (RCA) and actionable remediations. This paper establishes the foundational system architecture, formal relational ontology, and multi-agent control loops of Semantic Observability. We show conceptually how the framework eliminates ambient telemetry noise, reduces Mean Time to Resolution (MTTR), compresses LLM token consumption, and prevents structural hallucinations. We also specify an open-source validation framework on standard benchmarks (LogHub and Train-Ticket), laying the groundwork for a forthcoming companion paper on exhaustive quantitative evaluation and production benchmarking.",
      "published": "2026-10-20T17:53:32Z",
      "abstract_url": "http://arxiv.org/abs/2610.28611v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28611v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Benchmarking Arabic Dialects for Math Problems",
      "authors": [
        "Noor Abo Mokh",
        "Teresa Lynn",
        "Nizar Habash",
        "Ahmed Elshabrawy",
        "Sara Alhajeri",
        "Alaa Elsetohy",
        "Preslav Nakov",
        "Alham Fikri Aji"
      ],
      "abstract": "Mathematical reasoning remains underexplored for low-resource languages, including Dialectal Arabic. To bridge this gap, we introduce the DAR-MGSM (Dialectal Arabic Multilingual Grade School Math) Benchmark, the first manually curated benchmark for evaluating mathematical reasoning in Dialectal Arabic. DAR-MGSM consists of math word problems (MWPs) requiring multi-step reasoning and covers five major Arabic dialects alongside Modern Standard Arabic (MSA). The benchmark comprises two parallel versions: translated-only and translated & culturally localized. The latter version adapts entities and references to regional varieties. We evaluate eleven Arabic-centric and multilingual large language models on DAR-MGSM. The results show that the performance generally declines on the localized version, highlighting the challenges introduced by culturally grounded adaptations. However, the magnitude of this effect varies across models, with Arabic-centric models exhibiting different levels of sensitivity to localization. A mechanistic interpretability analysis shows that translated and localized variants of the same MWPs often induce distinct internal reasoning trajectories, even when aggregate accuracy remains similar. Overall, our results show that mathematical reasoning performance depends on both linguistic variety and cultural context.",
      "published": "2026-10-20T17:50:13Z",
      "abstract_url": "http://arxiv.org/abs/2610.28607v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28607v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Turning TS2Vec into a Foundation Model for Time Series Classification",
      "authors": [
        "Vasilii Feofanov",
        "Zhenghan Tai",
        "Andrei Ermakov",
        "Irina Rish"
      ],
      "abstract": "TS2Vec is one of the strongest self-supervised methods for time series classification, introduced in 2022. It is designed to be trained anew on every dataset: self-supervised pre-training followed by linear probing. Since then, the field has shifted towards foundation models that reuse one encoder across datasets. Yet TS2Vec offers a compact convolutional architecture that operates directly on temporal samples and accepts variable-length inputs without resampling. We ask whether these properties can be retained when the encoder is pre-trained only once, and show that they can: TS2Vec can essentially be turned into a foundation model. Three changes suffice: pre-train on a synthetic corpus, feed the encoder a raw amplitude view and an instance-normalized shape view of the input through two small stems, and distill one specialist encoder per view into a single encoder. The result, TS2Vec-FM, is a 1.6M-parameter dilated convolutional network that, kept frozen, reaches 83.3% on UCR-128 and 75.3% on UEA-27 under linear probing, within 0.3 pp of MantisV2, the current state of the art, and above TS2Vec trained separately on each target dataset. What we find most interesting is how different TS2Vec-FM is from other foundation models in design: near-parity is reached without attention, without a patch tokenizer, and without intermediate-layer extraction, using an \"old-school\" convolutional encoder and a two-layer MLP stem.",
      "published": "2026-10-20T17:47:51Z",
      "abstract_url": "http://arxiv.org/abs/2610.28602v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28602v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Evolving Semantic Indexes for Generative Retrieval",
      "authors": [
        "Jinming Nian",
        "Jyun-Yu Jiang",
        "Henry Tsai",
        "Yi Fang",
        "Zhirong Liang"
      ],
      "abstract": "Generative retrieval (GR) casts search and recommendation as autoregressive generation of discrete semantic identifiers (SIDs). SID design largely determines GR quality, but its discrete, interdependent components (SID construction, collision handling, decoding, and training) cannot be optimized end-to-end with gradients and remain largely hand-designed. We introduce a framework that improves these components through LLM-driven evolutionary program search, to our knowledge the first application of this approach to GR. A locally served, open-weight 31B model mutates executable GR components, so no proprietary LLM API is used, and a 1M-parameter proxy decoder provides fast feedback in place of repeatedly training the 1B target generator. Across 21 SID programs on NQ320K, proxy and target Recall@100 show strong rank correlation (Spearman $ρ= 0.90$). Guided by this proxy, evolution improves the seed GR on NQ320K, MSMARCO300K, and a multilingual e-commerce dataset under a fixed training recipe. On NQ320K, jointly evolving SID construction and collision handling improves Recall@1 from 55.8% to 59.3% and Recall@100 from 85.8% to 90.6%. A component study shows that proxy reliability depends on what is evolved: improvements transfer reliably for the semantic index but not for the generator's training objective or the full pipeline. LLM-driven evolutionary search is thus a practical tool for GR design, provided proxy reliability is established for the chosen search space.",
      "published": "2026-10-20T17:27:46Z",
      "abstract_url": "http://arxiv.org/abs/2610.28568v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28568v1",
      "categories": [
        "cs.AI",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "OptScientist: Automated Discovery of Optimization Laws",
      "authors": [
        "Ma ZhenYao",
        "Yue Liang",
        "Jian-Qiao Zhu"
      ],
      "abstract": "Many scientific phenomena are best understood as outcomes of underlying optimization problems, yet automated scientific discovery has largely focused on recovering equations that describe observed input-output patterns. We introduce OptScientist, a novel automated framework for discovering compact symbolic optimization laws directly from data. OptScientist comprises (i) a model that infers symbolic optimization problems by first learning to represent data as an optimization problem and then extracting its symbolic form, and (ii) an LLM agentic system embedding the model in a discovery process that integrates theoretical priors from scientific literature, discovers candidate laws, abstracts their underlying mechanisms, and suggests new data to collect. We first evaluate OptScientist on OptFeynman, a novel benchmark of 99 physics-derived optimization problems, where OptScientist recovers ground-truth laws for over 80% of the problems and remains robust to noisy contextual information, while substantially outperforming machine-learning, symbolic, and LLM-based model-discovery baselines. On human risky choice and central-bank monetary policy, OptScientist discovers compact symbolic optimization laws that achieve held-out predictive performance competitive with leading baselines, while using few parameters and substantially fewer tokens than LLM baselines. Overall, OptScientist extends automated scientific discovery to uncovering optimization laws that generate observed phenomena, enabling automated theory building across the physical, biological, behavioral, and social sciences.",
      "published": "2026-10-20T17:23:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.28562v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28562v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SC",
        "econ.EM",
        "math.OC"
      ]
    },
    {
      "title": "When Do LLMs Help with Feature Selection? A Systematic Empirical Study",
      "authors": [
        "Fateme Memar",
        "Dongjie Wang",
        "Arvin Agah"
      ],
      "abstract": "Large language models (LLMs) can leverage semantic feature names alongside summary statistics for tabular feature selection, yet it remains unclear when such information improves subset selection relative to established statistical methods. We investigate this question within an 18-dataset framework comprising 10 datasets for Group Relative Policy Optimization (GRPO) training, 2 for checkpoint validation, and 6 held-out evaluation datasets. We compare five classical baselines, frozen Qwen2.5-7B and Llama-3.3-70B selectors, and two GRPO-adapted Qwen2.5-7B models, reporting classification ROC-AUC and regression R^2 separately and using dataset-wise ranks for cross-task comparison. Across five seeded evaluation partitions, classical selectors generally achieve higher predictive performance than the tested 7B LLM/RL selectors, including under a pre-specified ~33% feature budget with strict train-only classical fitting, while the 7B selectors retain compact subsets. A cardinality-matched semantic sensitivity experiment shows that, for Zero-shot Qwen-7B, informative feature names improve regression performance over anonymized and shuffled alternatives (+0.1964 and +0.1606 mean R^2), whereas aggregate classification differences are smaller (+0.0159 and +0.0084 ROC-AUC) and dataset-dependent. These results characterize when semantic information benefits LLM-based feature selection and show that classical selectors remain strong baselines for predictive tabular feature selection.",
      "published": "2026-10-20T17:18:55Z",
      "abstract_url": "http://arxiv.org/abs/2610.28556v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28556v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning Computational Structure for Sample-Efficient Prediction",
      "authors": [
        "Hangyu Zhou",
        "Constantine Dovrolis",
        "Stephen Mussmann"
      ],
      "abstract": "Statistical learning in high dimensions suffers from the curse of dimensionality. Fortunately, many functions of practical interest are compositionally sparse, meaning they admit a composition of local functions, each depending on only a few variables, allowing sample complexity to depend on local arity rather than ambient dimension. Yet realizing this benefit requires exploiting hidden compositional structure, which theory does not establish for practical gradient-based training. We study this gap with CSFRBench, a controlled benchmark that hides the target's compositional computation during training but retains it for diagnosis, and find that dense networks and sparsity-based baselines leave much of that benefit unrealized. Inspired by the spatial organization of biological neural circuits, we introduce Learned Wiring (LeWi): each neuron has a learned low-dimensional embedding, only nearby neurons connect, and embeddings and weights are trained jointly. LeWi improves sample efficiency on targets with hierarchical structure, intermediate reuse, or independent modules. Across 32 additional targets generated by independently sampled computation graphs, LeWi is the most sample-efficient method not given the target graph, with 27% lower mean learning-curve area than dense networks. Dense networks perform better on targets with a single high-arity function. LeWi's representations and connectivity align with the target computation. On 17 world-model prediction tasks spanning control, navigation, and manipulation, LeWi beats the dense baseline on 16 and wins 99/119 baseline comparisons. These results show that jointly learning connectivity and weights can exploit computational structure to improve sample efficiency.",
      "published": "2026-10-20T17:18:35Z",
      "abstract_url": "http://arxiv.org/abs/2610.28555v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28555v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks",
      "authors": [
        "Rodrigo Casado Noguerales",
        "Dimitri von Rütte",
        "Bernhard Schölkopf",
        "Thomas Hofmann",
        "Aran Raoufi"
      ],
      "abstract": "Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and graph coloring, the symmetric formulation of binary diffusion consistently outperforms strong, well-tuned masked and uniform baselines under matched architectures and controlled training, sampling, and tuning budgets, with its largest gains on the hardest instances, while remaining competitive on language modeling (Text8, TinyStories, OpenWebText). These results support candidate sets as an effective intermediate representation for iterative reasoning and show that the corruption kernel can materially change the computation learned by the reverse model.",
      "published": "2026-10-20T17:11:48Z",
      "abstract_url": "http://arxiv.org/abs/2610.28543v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28543v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "stat.ML"
      ]
    },
    {
      "title": "ECG for laboratory values:A comprehensive physiological and technical analysis",
      "authors": [
        "Walentin Palanchuk"
      ],
      "abstract": "Laboratory measurements are essential for clinical decision making but typically require invasive blood sampling. Recent research suggests that ECG recordings may contain information beyond cardiac rhythm and provide information related to laboratory values. This thesis investigates NT-proBNP prediction from ECG recordings and patient metadata using a multimodal deep learning framework, while considering the physiological plausibility of ECG-based prediction. It also compares a 1-lead configuration using lead II with the corresponding 12-lead configuration. Additional experiments on glucose, hemoglobin, and potassium are presented in the appendix. Regression, binary classification, and three-class classification experiments were conducted using MIMIC-IV, with the three-class task based on clinically established thresholds. The 12-lead configuration consistently achieved better performance. Regression reached a mean absolute error of 534.98 pg/mL for 1-lead and 514.27 pg/mL for 12-lead. Binary classification achieved AUROCs of 0.849 and 0.876, respectively, while three-class classification achieved macro-averaged AUROCs of 0.738 and 0.752. The results indicate that precise NT-proBNP regression remains challenging, whereas broader concentration ranges can be classified more effectively. The findings are consistent with the physiological hypothesis that ECG characteristics and NT-proBNP concentrations reflect shared cardiovascular processes. ECG-based NT-proBNP prediction may therefore provide complementary information, particularly for classifying broader concentration ranges. However, laboratory measurements remain necessary, and validation on independent datasets is required.",
      "published": "2026-10-20T16:46:26Z",
      "abstract_url": "http://arxiv.org/abs/2610.28507v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28507v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models",
      "authors": [
        "Ning Zhang",
        "Shihao Wang",
        "Jinrui Zhang",
        "Chaodong Xiao",
        "Lei Zhang"
      ],
      "abstract": "For efficient deployment of large language models (LLMs), methods of ultra-low-bit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key-value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm-system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidence-only decoding, speculative-only decoding, and hybrid decoding provide different quality-throughput operating points without using separate draft and verifier models. Accepted-prefix alignment keeps the committed prefix, materialized causal KV boundary, and CUDA Graph replay state consistent for continued generation. BitMTP uses this contract to select width-specialized low-bit execution and reuse captured graphs over shared KV storage. On NVIDIA H20, BitMTP with hybrid decoding achieves five-task mean speedups of 2.2x on BitNet b1.58-2B and 2.1-2.4x on 1.7B-4B Bonsai models at comparable task quality. On RTX 3090 and Jetson Orin, it further achieves up to 2.0x in five-task mean throughput and 3.3x on MBPP+, respectively. These results demonstrate that BitMTP accelerates ultra-low-bit LLM generation across backbones, model sizes, and GPU types. The source code is released at https://github.com/PolyU-VCLab/BitMTP",
      "published": "2026-10-20T16:42:17Z",
      "abstract_url": "http://arxiv.org/abs/2610.28500v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28500v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.DC",
        "cs.LG"
      ]
    },
    {
      "title": "Model- and Layer-Level Closed-Loop Forecasting with SSM-Based Spiking Neural Networks",
      "authors": [
        "Erik Stolt",
        "Sanja Karilanova",
        "Ayca Ozcelikkale"
      ],
      "abstract": "Deep State-Space Models (Deep SSMs) have shown promising performance and computational benefits in processing of time-series data with complex temporal dependencies. Spiking neural networks (SNNs) constitute a biologically inspired class of deep SSMs, where computing blocks communicate through spikes, discrete single-bit events rather than continuous activation functions. A particularly important task in time-series processing is forecasting, i.e. predicting future values of a sequence from past observations alone. However, how to best perform forecasting with SNNs is an open question, where the standard autoregressive nature of the data processing with SNNs requires values that are not available during the forecasting phase. In this article, we address this challenge. We explore two closed-loop forecasting strategies, in which the network's own generated values are fed back into the network as if they were observed past values: (i) model-level and (ii) layer-level closed-loop forecasting. We evaluate these approaches on the ETT-small datasets, a standard benchmark for time-series forecasting, under both spike-based and continuous activation functions. We compare the proposed closed-loop approaches against state-of-the-art methods such as SpaceTime, as well as the zero-masking baseline in S4. Both of the proposed approaches perform comparably to or better than zero-masking with S4, and in most cases match SpaceTime. These findings highlight the trade-off between continuous and low-bit quantized activations, and demonstrate that the proposed Deep SSM architectures provide a competitive framework for forecasting.",
      "published": "2026-10-20T16:39:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.28495v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28495v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Claims of emergence in AI should be grounded in information decomposition",
      "authors": [
        "Christoph Riedl",
        "Fernando E. Rosas"
      ],
      "abstract": "When a language model develops capabilities not traceable to its individual components, when a multi-agent AI system outperforms each agent on its own, or when a human-AI pair succeeds where either fails in isolation, AI researchers describe it as a case of ``synergy'' or ``emergence''. However, these are usually informal assessments not supported by a theoretical framework, making these claims impossible to compare or even meaningfully dispute. Here we argue that \\emph{information decomposition can serve as a unifying formal framework to operationalize assessments about synergy in AI systems}. Information decomposition provides a way to distinguish qualitatively different \\emph{types} of information, including unique information available from individual components, redundant information shared across multiple components, and synergistic information accessible only from parts taken together. This gives claims of emergence a precise, falsifiable criterion: a system's behavior is emergent only if it carries synergistic information about task-relevant outcomes. We apply this lens to attention heads, autonomous agents, and human-AI teams, showing how it organizes existing knowledge across scales and substrates, while also generating cross-level conjectures that siloed fields would miss.",
      "published": "2026-10-20T16:29:28Z",
      "abstract_url": "http://arxiv.org/abs/2610.28479v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28479v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "cs.IT",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Stratified Sparse Attention: Reducing the Sparse-to-Dense Gap at Extreme Sparsity",
      "authors": [
        "Sahil Joshi",
        "Prithvi Dixit",
        "Aryan Katiyar",
        "Agniva Chowdhury",
        "Joseph E. Gonzalez",
        "Anshumali Shrivastava",
        "Aditya Desai",
        "Kumar Krishna Agrawal"
      ],
      "abstract": "Large Language Model inference over long contexts is increasingly bottlenecked by the cost of attending to a growing Key-Value (KV) cache. Sparse attention reduces this cost by accessing only a fraction of the cache, but existing methods face a trade-off between deterministic retrieval, which captures high-scoring tokens but discards residual attention mass, and sampling, which improves coverage at the cost of higher variance. Hybrid methods combine top-$k$ retrieval with uniform sampling, yet still exhibit a substantial gap from dense attention at aggressive sparsity. We introduce \\emph{Stratified Sparse Attention (SSA)}, a unified framework combining deterministic retrieval with score-guided sampling. SSA deterministically attends to high-scoring tokens while sampling the residual using proxy attention scores and correcting for their inclusion probabilities, thereby recovering attention mass beyond the retrieval set. SSA encompasses top-$k$, top-$p$, and hybrid top-$k$+sampling as special cases. Across RULER, LongBench, LOFT, and AIME, and five models ranging from 3B to 27B parameters, SSA consistently improves over the baselines at matched sparsity. On RULER-HARD with Ministral-3-8B, SSA improves accuracy over the strongest practical baseline from 81.8\\% to 84.2\\% at 32K and from 63.8\\% to 69.3\\% at 128K under $50\\times$ attention sparsity, recovering 94\\% and 85\\% of dense-attention performance, respectively. With optimized CUDA kernels, SSA achieves up to $2.96\\times$ decode speedup over FlashInfer at 1M context, demonstrating that its accuracy gains translate into practical long-context inference acceleration. We open-source our code at https://github.com/skylight-org/SSA-Paper-Code.",
      "published": "2026-10-20T16:02:12Z",
      "abstract_url": "http://arxiv.org/abs/2610.28441v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28441v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting",
      "authors": [
        "Mohammad Reza Modarres",
        "Armin Tourajmehr",
        "Artin Tavasoli",
        "Yadollah Yaghoobzadeh",
        "Mohammad Taher Pilehvar"
      ],
      "abstract": "Large language models leave model-specific traces in their generated text, but it remains unclear how finely these fingerprints can distinguish related models and which signals remain informative as obvious cues are reduced. We study this through fingerprint resolution, considering both model resolution, from different families to distilled and adapted variants, and signal resolution across behavioral, stylometric, lexical, and semantic evidence. We progressively suppress surface cues, paraphrase outputs, vary the amount of available text, and test generalization across prompt distributions and languages. Our results show that fingerprints persist beyond broad family differences, remaining detectable in distilled models and even LoRA adaptations of the same base model. Although attribution weakens as surface and lexical cues are altered, substantial model-specific signal remains. These signals transfer across prompt sources and languages, with semantic representations proving most robust across languages. These findings show that model identity is recoverable at multiple resolutions, but that recoverability depends on model similarity and the evidence available in the text.",
      "published": "2026-10-20T16:00:34Z",
      "abstract_url": "http://arxiv.org/abs/2610.28434v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28434v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Digital Twin to Improve Cyber-Social Impact",
      "authors": [
        "Vita Santa Barletta",
        "Danilo Caivano",
        "Gabriel Cellammare",
        "Andrea Dimauro",
        "Pasquale Mesagna",
        "Massimiliano Morga"
      ],
      "abstract": "The increasing complexity of Industrial Control Systems (ICS) and their integration with IT infrastructures has significantly expanded the cyber threat landscape. Traditional cybersecurity models, primarily based on the Confidentiality, Integrity, and Availability (CIA triad), fail to capture the broader societal and operational impacts of cyberattacks. This paper proposes a novel framework that integrates Digital Twin technology with a Cyber-Social Security model to simulate, analyze, and quantify cyber threats in critical infrastructures. Virtualized ICS environments built with Mininet and MiniCPS are used to model attack scenarios and assess their consequences across technical and social dimensions. At the core of the framework lies the Cyber-Social Vulnerability Scoring System (CSVSS), a quantitative metric that extends CVSS with a Social Impact dimension grounded in the Entity and Place pillars, producing a single, reproducible severity score that reflects real-world consequences. Crucially, the framework does not automate risk assessment but keeps the human analyst at its center: a Large Language Model (LLM) performs a first-pass analysis, mapping vulnerabilities and drafting risk reports, while the analyst, supported by explainable AI feedback, validates the findings and retains decision authority over critical cases. Experimental results demonstrate the framework's ability to model and score cyber-social impact, while the human-in-the-loop design is presented as a conceptual principle to preserve oversight.",
      "published": "2026-10-20T15:58:33Z",
      "abstract_url": "http://arxiv.org/abs/2610.28430v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28430v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "DISTRACT: Degradation-Indexed Study of Textual Reference Across Competing Targets",
      "authors": [
        "Dan Zlotnikov",
        "Ohad Ben-Shahar"
      ],
      "abstract": "Vision-language trackers (VLTs) accept descriptions intended to assist tracking, but accepting text does not establish that its content changes tracking decisions, particularly as visual evidence deteriorates. We introduce DISTRACT, a matched-intervention benchmark that holds video, target, and distractor fixed while crossing four text priors with four image degradations along a controlled severity axis. We evaluate five VLTs on four datasets using behavioral contrasts, layerwise representation probes, and activation patching. Molmo2 and JointNLT exhibit directed text use. Some effects contract under degradation, whereas others show no significant endpoint change after multiplicity correction. UVLTrack, MMTrack, and DUTrack respond only weakly to changes in text semantics, with no directed behavioral effect surviving FDR correction in the tested settings. Model-internal analyses reveal that at decision-near states, correct-versus-conflicting text is decodable in Molmo2 and JointNLT but not in the other three systems. In the responsive models, decodability peaks at late states where replacement produces little behavioral change, whereas replacing selected earlier states redirects conflict-associated trajectories toward the target. Because matched-correct donors do not significantly outperform unrelated donors, these interventions identify earlier intervention-sensitive states without establishing target-specific semantic rescue. DISTRACT thus exposes when and where text affects tracking under degraded visual evidence.",
      "published": "2026-10-20T15:53:23Z",
      "abstract_url": "http://arxiv.org/abs/2610.28419v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28419v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Code2Video-Compass: A Comprehensive Benchmark for General-Purpose Code-Based Video Generation across T2V, I2V, and V2V",
      "authors": [
        "Zhenghan Chen",
        "Zekai Shao",
        "Xin Lin",
        "Leting Cheng",
        "Siming Chen"
      ],
      "abstract": "Code-based video generation offers fine-grained control, targeted editing, and component reuse compared with direct video generation. This approach requires multimodal large language models (MLLMs) to translate visual and temporal requirements into executable programs that produce complete videos. However, existing code-based benchmarks primarily target specialized tasks, leaving complete video creation across diverse creative categories and input modalities insufficiently evaluated. We introduce Code2Video-Compass, a general-purpose benchmark for code-based video generation across text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) settings. Code2Video-Compass comprises 2,104 tasks spanning 10 categories, 24 subtypes, and three complexity dimensions covering construction, dynamic animation, and state dependency. It combines tasks curated from textual instructions and visual references with corresponding T2V, I2V, and V2V tasks derived from complete professional and user-created videos. We propose a dual-level evaluation framework combining 12 objective metrics and five subjective metrics. Objective evaluation covers operational performance, video quality, text-video alignment, and reference fidelity, while subjective evaluation uses an MLLM judge to evaluate instruction following and holistic video quality. Extensive evaluation of 15 recent proprietary and open-weight MLLMs from nine families reveals that even leading models face challenges in reliably generating high-quality complete videos aligned with textual and visual requirements, with performance generally declining as construction, dynamic animation, and state-dependency complexity increases. We anticipate that Code2Video-Compass will advance reliable, coherent, and controllable code-based video generation and inspire broader progress in programmatic visual creation.",
      "published": "2026-10-20T15:51:41Z",
      "abstract_url": "http://arxiv.org/abs/2610.28416v1",
      "pdf_url": "https://arxiv.org/pdf/2610.28416v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    }
  ]
};
