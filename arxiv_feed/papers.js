const PAPERS_DATA = {
  "last_updated": "2026-04-13 03:39:54 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism",
      "authors": [
        "Hadas Orgad",
        "Boyi Wei",
        "Kaden Zheng",
        "Martin Wattenberg",
        "Peter Henderson",
        "Seraphina Goldfarb-Tarrant",
        "Yonatan Belinkov"
      ],
      "abstract": "Large language models (LLMs) undergo alignment training to avoid harmful behaviors, yet the resulting safeguards remain brittle: jailbreaks routinely bypass them, and fine-tuning on narrow domains can induce ``emergent misalignment'' that generalizes broadly. Whether this brittleness reflects a fundamental lack of coherent internal organization for harmfulness remains unclear. Here we use targeted weight pruning as a causal intervention to probe the internal organization of harmfulness in LLMs. We find that harmful content generation depends on a compact set of weights that are general across harm types and distinct from benign capabilities. Aligned models exhibit a greater compression of harm generation weights than unaligned counterparts, indicating that alignment reshapes harmful representations internally--despite the brittleness of safety guardrails at the surface level. This compression explains emergent misalignment: if weights of harmful capabilities are compressed, fine-tuning that engages these weights in one domain can trigger broad misalignment. Consistent with this, pruning harm generation weights in a narrow domain substantially reduces emergent misalignment. Notably, LLMs harmful generation capability is dissociated from how they recognize and explain such content. Together, these results reveal a coherent internal structure for harmfulness in LLMs that may serve as a foundation for more principled approaches to safety.",
      "published": "2026-04-10T17:58:31Z",
      "abstract_url": "http://arxiv.org/abs/2604.09544v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09544v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Case-Grounded Evidence Verification: A Framework for Constructing Evidence-Sensitive Supervision",
      "authors": [
        "Soroosh Tayebi Arasteh",
        "Mehdi Joodaki",
        "Mahshad Lotfinia",
        "Sven Nebelung",
        "Daniel Truhn"
      ],
      "abstract": "Evidence-grounded reasoning requires more than attaching retrieved text to a prediction: a model should make decisions that depend on whether the provided evidence supports the target claim. In practice, this often fails because supervision is weak, evidence is only loosely tied to the claim, and evaluation does not test evidence dependence directly. We introduce case-grounded evidence verification, a general framework in which a model receives a local case context, external evidence, and a structured claim, and must decide whether the evidence supports the claim for that case. Our key contribution is a supervision construction procedure that generates explicit support examples together with semantically controlled non-support examples, including counterfactual wrong-state and topic-related negatives, without manual evidence annotation. We instantiate the framework in radiology and train a standard verifier on the resulting support task. The learned verifier substantially outperforms both case-only and evidence-only baselines, remains strong under correct evidence, and collapses when evidence is removed or swapped, indicating genuine evidence dependence. This behavior transfers across unseen evidence articles and an external case distribution, though performance degrades under evidence-source shift and remains sensitive to backbone choice. Overall, the results suggest that a major bottleneck in evidence grounding is not only model capacity, but the lack of supervision that encodes the causal role of evidence.",
      "published": "2026-04-10T17:55:38Z",
      "abstract_url": "http://arxiv.org/abs/2604.09537v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09537v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "VisionFoundry: Teaching VLMs Visual Perception with Synthetic Images",
      "authors": [
        "Guanyu Zhou",
        "Yida Yin",
        "Wenhao Chai",
        "Shengbang Tong",
        "Xingyu Fu",
        "Zhuang Liu"
      ],
      "abstract": "Vision-language models (VLMs) still struggle with visual perception tasks such as spatial understanding and viewpoint recognition. One plausible contributing factor is that natural image datasets provide limited supervision for low-level visual skills. This motivates a practical question: can targeted synthetic supervision, generated from only a task keyword such as Depth Order, address these weaknesses? To investigate this question, we introduce VisionFoundry, a task-aware synthetic data generation pipeline that takes only the task name as input and uses large language models (LLMs) to generate questions, answers, and text-to-image (T2I) prompts, then synthesizes images with T2I models and verifies consistency with a proprietary VLM, requiring no reference images or human annotation. Using VisionFoundry, we construct VisionFoundry-10K, a synthetic visual question answering (VQA) dataset containing 10k image-question-answer triples spanning 10 tasks. Models trained on VisionFoundry-10K achieve substantial improvements on visual perception benchmarks: +7% on MMVP and +10% on CV-Bench-3D, while preserving broader capabilities and showing favorable scaling behavior as data size increases. Our results suggest that limited task-targeted supervision is an important contributor to this bottleneck and that synthetic supervision is a promising path toward more systematic training for VLMs.",
      "published": "2026-04-10T17:48:51Z",
      "abstract_url": "http://arxiv.org/abs/2604.09531v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09531v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Envisioning the Future, One Step at a Time",
      "authors": [
        "Stefan Andreas Baumann",
        "Jannik Wiese",
        "Tommaso Martorella",
        "Mahdi M. Kalayeh",
        "Björn Ommer"
      ],
      "abstract": "Accurately anticipating how complex, diverse scenes will evolve requires models that represent uncertainty, simulate along extended interaction chains, and efficiently explore many plausible futures. Yet most existing approaches rely on dense video or latent-space prediction, expending substantial capacity on dense appearance rather than on the underlying sparse trajectories of points in the scene. This makes large-scale exploration of future hypotheses costly and limits performance when long-horizon, multi-modal motion is essential. We address this by formulating the prediction of open-set future scene dynamics as step-wise inference over sparse point trajectories. Our autoregressive diffusion model advances these trajectories through short, locally predictable transitions, explicitly modeling the growth of uncertainty over time. This dynamics-centric representation enables fast rollout of thousands of diverse futures from a single image, optionally guided by initial constraints on motion, while maintaining physical plausibility and long-range coherence. We further introduce OWM, a benchmark for open-set motion prediction based on diverse in-the-wild videos, to evaluate accuracy and variability of predicted trajectory distributions under real-world uncertainty. Our method matches or surpasses dense simulators in predictive accuracy while achieving orders-of-magnitude higher sampling speed, making open-set future prediction both scalable and practical. Project page: http://compvis.github.io/myriad.",
      "published": "2026-04-10T17:46:05Z",
      "abstract_url": "http://arxiv.org/abs/2604.09527v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09527v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Strategic Algorithmic Monoculture:Experimental Evidence from Coordination Games",
      "authors": [
        "Gonzalo Ballestero",
        "Hadi Hosseini",
        "Samarth Khanna",
        "Ran I. Shorrer"
      ],
      "abstract": "AI agents increasingly operate in multi-agent environments where outcomes depend on coordination. We distinguish primary algorithmic monoculture -- baseline action similarity -- from strategic algorithmic monoculture, whereby agents adjust similarity in response to incentives. We implement a simple experimental design that cleanly separates these forces, and deploy it on human and large language model (LLM) subjects. LLMs exhibit high levels of baseline similarity (primary monoculture) and, like humans, they regulate it in response to coordination incentives (strategic monoculture). While LLMs coordinate extremely well on similar actions, they lag behind humans in sustaining heterogeneity when divergence is rewarded.",
      "published": "2026-04-10T17:14:46Z",
      "abstract_url": "http://arxiv.org/abs/2604.09502v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09502v1",
      "categories": [
        "cs.AI",
        "cs.GT",
        "cs.MA",
        "econ.TH"
      ]
    },
    {
      "title": "BERT-as-a-Judge: A Robust Alternative to Lexical Methods for Efficient Reference-Based LLM Evaluation",
      "authors": [
        "Hippolyte Gisserot-Boukhlef",
        "Nicolas Boizard",
        "Emmanuel Malherbe",
        "Céline Hudelot",
        "Pierre Colombo"
      ],
      "abstract": "Accurate evaluation is central to the large language model (LLM) ecosystem, guiding model selection and downstream adoption across diverse use cases. In practice, however, evaluating generative outputs typically relies on rigid lexical methods to extract and assess answers, which can conflate a model's true problem-solving ability with its compliance with predefined formatting guidelines. While recent LLM-as-a-Judge approaches mitigate this issue by assessing semantic correctness rather than strict structural conformity, they also introduce substantial computational overhead, making evaluation costly. In this work, we first systematically investigate the limitations of lexical evaluation through a large-scale empirical study spanning 36 models and 15 downstream tasks, demonstrating that such methods correlate poorly with human judgments. To address this limitation, we introduce BERT-as-a-Judge, an encoder-driven approach for assessing answer correctness in reference-based generative settings, robust to variations in output phrasing, and requiring only lightweight training on synthetically annotated question-candidate-reference triplets. We show that it consistently outperforms the lexical baseline while matching the performance of much larger LLM judges, providing a compelling tradeoff between the two and enabling reliable, scalable evaluation. Finally, through extensive experimentation, we provide detailed insights into BERT-as-a-Judge's performance to offer practical guidance for practitioners, and release all project artifacts to foster downstream adoption.",
      "published": "2026-04-10T17:08:40Z",
      "abstract_url": "http://arxiv.org/abs/2604.09497v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09497v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "RecaLLM: Addressing the Lost-in-Thought Phenomenon with Explicit In-Context Retrieval",
      "authors": [
        "Kyle Whitecross",
        "Negin Rahimi"
      ],
      "abstract": "We propose RecaLLM, a set of reasoning language models post-trained to make effective use of long-context information. In-context retrieval, which identifies relevant evidence from context, and reasoning are deeply intertwined: retrieval supports reasoning, while reasoning often determines what must be retrieved. However, their interaction remains largely underexplored. In preliminary experiments on several open-source LLMs, we observe that in-context retrieval performance substantially degrades even after a short reasoning span, revealing a key bottleneck for test-time scaling that we refer to as lost-in-thought: reasoning steps that improve performance also make subsequent in-context retrieval more challenging. To address this limitation, RecaLLM interleaves reasoning with explicit in-context retrieval, alternating between reasoning and retrieving context information needed to solve intermediate subproblems. We introduce a negligible-overhead constrained decoding mechanism that enables verbatim copying of evidence spans, improving the grounding of subsequent generation. Trained on diverse lexical and semantic retrieval tasks, RecaLLM achieves strong performance on two long-context benchmarks, RULER and HELMET, significantly outperforming baselines. Notably, we observe consistent gains at context windows of up to 128K tokens using training samples of at most 10K tokens, far shorter than those used by existing long-context approaches, highlighting a promising path toward improving long-context performance without expensive long-context training data.",
      "published": "2026-04-10T17:04:32Z",
      "abstract_url": "http://arxiv.org/abs/2604.09494v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09494v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "XFED: Non-Collusive Model Poisoning Attack Against Byzantine-Robust Federated Classifiers",
      "authors": [
        "Israt Jahan Mouri",
        "Muhammad Ridowan",
        "Muhammad Abdullah Adnan"
      ],
      "abstract": "Model poisoning attacks pose a significant security threat to Federated Learning (FL). Most existing model poisoning attacks rely on collusion, requiring adversarial clients to coordinate by exchanging local benign models and synchronizing the generation of their poisoned updates. However, sustaining such coordination is increasingly impractical in real-world FL deployments, as it effectively requires botnet-like control over many devices. This approach is costly to maintain and highly vulnerable to detection. This context raises a fundamental question: Can model poisoning attacks remain effective without any communication between attackers? To address this challenge, we introduce and formalize the \\textbf{non-collusive attack model}, in which all compromised clients share a common adversarial objective but operate independently. Under this model, each attacker generates its malicious update without communicating with other adversaries, accessing other clients' updates, or relying on any knowledge of server-side defenses. To demonstrate the feasibility of this threat model, we propose \\textbf{XFED}, the first aggregation-agnostic, non-collusive model poisoning attack. Our empirical evaluation across six benchmark datasets shows that XFED bypasses eight state-of-the-art defenses and outperforms six existing model poisoning attacks. These findings indicate that FL systems are substantially less secure than previously believed and underscore the urgent need for more robust and practical defense mechanisms.",
      "published": "2026-04-10T16:54:29Z",
      "abstract_url": "http://arxiv.org/abs/2604.09489v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09489v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.DC",
        "cs.LG"
      ]
    },
    {
      "title": "E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning",
      "authors": [
        "Weiyang Guo",
        "Zesheng Shi",
        "Liye Zhao",
        "Jiayuan Ma",
        "Zeen Zhu",
        "Junxian He",
        "Min Zhang",
        "Jing Li"
      ],
      "abstract": "While Large Language Models (LLMs) have demonstrated significant potential in Tool-Integrated Reasoning (TIR), existing training paradigms face significant limitations: Zero-RL suffers from inefficient exploration and mode degradation due to a lack of prior guidance, while SFT-then-RL is limited by high data costs and capability plateaus caused by low-entropy collapse. To address these challenges, we propose E3-TIR (Enhanced Experience Exploitation), a warm-up paradigm for the early stages of agent training. Specifically, we formulate training as the dynamic integration of three experience types: Expert Prefixes, Expert Guided, and Self-Exploration. By executing diverse branching exploration around expert \"anchors\" and employing a mix policy optimization mechanism, we effectively mitigate distribution shifts and resolve optimization conflicts arising from shared prefixes. Our method dynamically adapts the model's knowledge boundaries, effectively balancing exploration diversity with training efficiency.Experimental results demonstrate that E3-TIR achieves a 6 performance improvement over traditional paradigms on tool-use tasks, while requiring less than 10 of the synthetic data. Furthermore, in terms of ROI, a comprehensive metric integrating performance, data cost, and training efficiency we achieve a 1.46x gain compared to baselines. Code is available at https://github.com/yuki-younai/E3-TIR.",
      "published": "2026-04-10T16:14:48Z",
      "abstract_url": "http://arxiv.org/abs/2604.09455v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09455v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SafeAdapt: Provably Safe Policy Updates in Deep Reinforcement Learning",
      "authors": [
        "Maksim Anisimov",
        "Francesco Belardinelli",
        "Matthew Wicker"
      ],
      "abstract": "Safety guarantees are a prerequisite to the deployment of reinforcement learning (RL) agents in safety-critical tasks. Often, deployment environments exhibit non-stationary dynamics or are subject to changing performance goals, requiring updates to the learned policy. This leads to a fundamental challenge: how to update an RL policy while preserving its safety properties on previously encountered tasks? The majority of current approaches either do not provide formal guarantees or verify policy safety only a posteriori. We propose a novel a priori approach to safe policy updates in continual RL by introducing the Rashomon set: a region in policy parameter space certified to meet safety constraints within the demonstration data distribution. We then show that one can provide formal, provable guarantees for arbitrary RL algorithms used to update a policy by projecting their updates onto the Rashomon set. Empirically, we validate this approach across grid-world navigation environments (Frozen Lake and Poisoned Apple) where we guarantee an a priori provably deterministic safety on the source task during downstream adaptation. In contrast, we observe that regularisation-based baselines experience catastrophic forgetting of safety constraints while our approach enables strong adaptation with provable guarantees that safety is preserved.",
      "published": "2026-04-10T16:09:39Z",
      "abstract_url": "http://arxiv.org/abs/2604.09452v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09452v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ECHO: Efficient Chest X-ray Report Generation with One-step Block Diffusion",
      "authors": [
        "Lifeng Chen",
        "Tianqi You",
        "Hao Liu",
        "Zhimin Bao",
        "Jile Jiao",
        "Xiao Han",
        "Zhicai Ou",
        "Tao Sun",
        "Xiaofeng Mou",
        "Xiaojie Jin",
        "Yi Xu"
      ],
      "abstract": "Chest X-ray report generation (CXR-RG) has the potential to substantially alleviate radiologists' workload. However, conventional autoregressive vision--language models (VLMs) suffer from high inference latency due to sequential token decoding. Diffusion-based models offer a promising alternative through parallel generation, but they still require multiple denoising iterations. Compressing multi-step denoising to a single step could further reduce latency, but often degrades textual coherence due to the mean-field bias introduced by token-factorized denoisers. To address this challenge, we propose \\textbf{ECHO}, an efficient diffusion-based VLM (dVLM) for chest X-ray report generation. ECHO enables stable one-step-per-block inference via a novel Direct Conditional Distillation (DCD) framework, which mitigates the mean-field limitation by constructing unfactorized supervision from on-policy diffusion trajectories to encode joint token dependencies. In addition, we introduce a Response-Asymmetric Diffusion (RAD) training strategy that further improves training efficiency while maintaining model effectiveness. Extensive experiments demonstrate that ECHO surpasses state-of-the-art autoregressive methods, improving RaTE and SemScore by \\textbf{64.33\\%} and \\textbf{60.58\\%} respectively, while achieving an \\textbf{$8\\times$} inference speedup without compromising clinical accuracy.",
      "published": "2026-04-10T16:07:14Z",
      "abstract_url": "http://arxiv.org/abs/2604.09450v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09450v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.IV"
      ]
    },
    {
      "title": "Many-Tier Instruction Hierarchy in LLM Agents",
      "authors": [
        "Jingyu Zhang",
        "Tianjian Li",
        "William Jurayj",
        "Hongyuan Zhan",
        "Benjamin Van Durme",
        "Daniel Khashabi"
      ],
      "abstract": "Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, and more-each carrying different levels of trust and authority. When these instructions conflict, models must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.",
      "published": "2026-04-10T16:00:04Z",
      "abstract_url": "http://arxiv.org/abs/2604.09443v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09443v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "On the Representational Limits of Quantum-Inspired 1024-D Document Embeddings: An Experimental Evaluation Framework",
      "authors": [
        "Dario Maio"
      ],
      "abstract": "Text embeddings are central to modern information retrieval and Retrieval-Augmented Generation (RAG). While dense models derived from Large Language Models (LLMs) dominate current practice, recent work has explored quantum-inspired alternatives motivated by the geometric properties of Hilbert-like spaces and their potential to encode richer semantic structure. This paper presents an experimental framework for constructing quantum-inspired 1024-dimensional document embeddings based on overlapping windows and multi-scale aggregation. The pipeline combines semantic projections (e.g., EigAngle), circuit-inspired feature mappings, and optional teacher-student distillation, together with a fingerprinting mechanism for reproducibility and controlled evaluation. We introduce a set of diagnostic tools for hybrid retrieval, including static and dynamic interpolation between BM25 and embedding-based scores, candidate union strategies, and a conceptual alpha-oracle that provides an upper bound for score-level fusion. Experiments on controlled corpora of Italian and English documents across technical, narrative, and legal domains, using synthetic queries, show that BM25 remains a strong baseline, teacher embeddings provide stable semantic structure, and standalone quantum-inspired embeddings exhibit weak and unstable ranking signals. Distillation yields mixed effects, improving alignment in some cases but not consistently enhancing retrieval performance, while hybrid retrieval can recover competitive results when lexical and embedding-based signals are combined. Overall, the results highlight structural limitations in the geometry of quantum-inspired embeddings, including distance compression and ranking instability, and clarify their role as auxiliary components rather than standalone retrieval representations.",
      "published": "2026-04-10T15:48:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.09430v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09430v1",
      "categories": [
        "cs.IR",
        "cs.AI"
      ]
    },
    {
      "title": "Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories",
      "authors": [
        "Wonbong Jang",
        "Shikun Liu",
        "Soubhik Sanyal",
        "Juan Camilo Perez",
        "Kam Woh Ng",
        "Sanskar Agrawal",
        "Juan-Manuel Perez-Rua",
        "Yiannis Douratsos",
        "Tao Xiang"
      ],
      "abstract": "Recovering camera parameters from images and rendering scenes from novel viewpoints have long been treated as separate tasks in computer vision and graphics. This separation breaks down when image coverage is sparse or poses are ambiguous, since each task needs what the other produces. We propose Rays as Pixels, a Video Diffusion Model (VDM) that learns a joint distribution over videos and camera trajectories. We represent each camera as dense ray pixels (raxels) and denoise them jointly with video frames through Decoupled Self-Cross Attention mechanism. A single trained model handles three tasks: predicting camera trajectories from video, jointly generating video and camera trajectory from input images, and generating video from input images along a target camera trajectory. Because the model can both predict trajectories from a video and generate views conditioned on its own predictions, we evaluate it through a closed-loop self-consistency test, demonstrating that its forward and inverse predictions agree. Notably, trajectory prediction requires far fewer denoising steps than video generation, even a few denoising steps suffice for self-consistency. We report results on pose estimation and camera-controlled video generation.",
      "published": "2026-04-10T15:47:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.09429v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09429v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "PhysInOne: Visual Physics Learning and Reasoning in One Suite",
      "authors": [
        "Siyuan Zhou",
        "Hejun Wang",
        "Hu Cheng",
        "Jinxi Li",
        "Dongsheng Wang",
        "Junwei Jiang",
        "Yixiao Jin",
        "Jiayue Huang",
        "Shiwei Mao",
        "Shangjia Liu",
        "Yafei Yang",
        "Hongkang Song",
        "Shenxing Wei",
        "Zihui Zhang",
        "Peng Huang",
        "Shijie Liu",
        "Zhengli Hao",
        "Hao Li",
        "Yitian Li",
        "Wenqi Zhou",
        "Zhihan Zhao",
        "Zongqi He",
        "Hongtao Wen",
        "Shouwang Huang",
        "Peng Yun",
        "Bowen Cheng",
        "Pok Kazaf Fu",
        "Wai Kit Lai",
        "Jiahao Chen",
        "Kaiyuan Wang",
        "Zhixuan Sun",
        "Ziqi Li",
        "Haochen Hu",
        "Di Zhang",
        "Chun Ho Yuen",
        "Bing Wang",
        "Zhihua Wang",
        "Chuhang Zou",
        "Bo Yang"
      ],
      "abstract": "We present PhysInOne, a large-scale synthetic dataset addressing the critical scarcity of physically-grounded training data for AI systems. Unlike existing datasets limited to merely hundreds or thousands of examples, PhysInOne provides 2 million videos across 153,810 dynamic 3D scenes, covering 71 basic physical phenomena in mechanics, optics, fluid dynamics, and magnetism. Distinct from previous works, our scenes feature multiobject interactions against complex backgrounds, with comprehensive ground-truth annotations including 3D geometry, semantics, dynamic motion, physical properties, and text descriptions. We demonstrate PhysInOne's efficacy across four emerging applications: physics-aware video generation, long-/short-term future frame prediction, physical property estimation, and motion transfer. Experiments show that fine-tuning foundation models on PhysInOne significantly enhances physical plausibility, while also exposing critical gaps in modeling complex physical dynamics and estimating intrinsic properties. As the largest dataset of its kind, orders of magnitude beyond prior works, PhysInOne establishes a new benchmark for advancing physics-grounded world models in generation, simulation, and embodied AI.",
      "published": "2026-04-10T15:27:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.09415v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09415v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "LLM-Rosetta: A Hub-and-Spoke Intermediate Representation for Cross-Provider LLM API Translation",
      "authors": [
        "Peng Ding"
      ],
      "abstract": "The rapid proliferation of Large Language Model (LLM) providers--each exposing proprietary API formats--has created a fragmented ecosystem where applications become tightly coupled to individual vendors. Switching or bridging providers requires $O(N^2)$ bilateral adapters, impeding portability and multi-provider architectures. We observe that despite substantial syntactic divergence, the major LLM APIs share a common semantic core: the practical challenge is the combinatorial surface of syntactic variations, not deep semantic incompatibility. Based on this finding, we present LLM-Rosetta, an open-source translation framework built on a hub-and-spoke Intermediate Representation (IR) that captures the shared semantic core--messages, content parts, tool calls, reasoning traces, and generation controls--in a 9-type content model and 10-type stream event schema. A modular Ops-composition converter architecture enables each API standard to be added independently. LLM-Rosetta supports bidirectional conversion (provider-to-IR-to-provider) for both request and response payloads, including chunk-level streaming with stateful context management. We implement converters for four API standards (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Google GenAI), covering the vast majority of commercial providers. Empirical evaluation demonstrates lossless round-trip fidelity, correct streaming behavior, and sub-100 microsecond conversion overhead--competitive with LiteLLM's single-pass approach while providing bidirectionality and provider neutrality. LLM-Rosetta passes the Open Responses compliance suite and is deployed in production at Argonne National Laboratory. Code is available at https://github.com/Oaklight/llm-rosetta.",
      "published": "2026-04-10T14:31:32Z",
      "abstract_url": "http://arxiv.org/abs/2604.09360v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09360v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Constraint-Aware Corrective Memory for Language-Based Drug Discovery Agents",
      "authors": [
        "Maochen Sun",
        "Youzhi Zhang",
        "Gaofeng Meng"
      ],
      "abstract": "Large language models are making autonomous drug discovery agents increasingly feasible, but reliable success in this setting is not determined by any single action or molecule. It is determined by whether the final returned set jointly satisfies protocol-level requirements such as set size, diversity, binding quality, and developability. This creates a fundamental control problem: the agent plans step by step, while task validity is decided at the level of the whole candidate set. Existing language-based drug discovery systems therefore tend to rely on long raw history and under-specified self-reflection, making failure localization imprecise and planner-facing agent states increasingly noisy. We present CACM (Constraint-Aware Corrective Memory), a language-based drug discovery framework built around precise set-level diagnosis and a concise memory write-back mechanism. CACM introduces protocol auditing and a grounded diagnostician, which jointly analyze multimodal evidence spanning task requirements, pocket context, and candidate-set evidence to localize protocol violations, generate actionable remediation hints, and bias the next action toward the most relevant correction. To keep planning context compact, CACM organizes memory into static, dynamic, and corrective channels and compresses them before write-back, thereby preserving persistent task information while exposing only the most decision-relevant failures. Our experimental results show that CACM improves the target-level success rate by 36.4% over the state-of-the-art baseline. The results show that reliable language-based drug discovery benefits not only from more powerful molecular tools, but also from more precise diagnosis and more economical agent states.",
      "published": "2026-04-10T13:16:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.09308v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09308v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SAGE: A Service Agent Graph-guided Evaluation Benchmark",
      "authors": [
        "Ling Shi",
        "Yuqin Dai",
        "Ziyin Wang",
        "Ning Gao",
        "Wei Zhang",
        "Chaozheng Wang",
        "Yujie Wang",
        "Wei He",
        "Jinpeng Wang",
        "Deiyi Xiong"
      ],
      "abstract": "The development of Large Language Models (LLMs) has catalyzed automation in customer service, yet benchmarking their performance remains challenging. Existing benchmarks predominantly rely on static paradigms and single-dimensional metrics, failing to account for diverse user behaviors or the strict adherence to structured Standard Operating Procedures (SOPs) required in real-world deployments. To bridge this gap, we propose SAGE (Service Agent Graph-guided Evaluation), a universal multi-agent benchmark for automated, dual-axis assessment. SAGE formalizes unstructured SOPs into Dynamic Dialogue Graphs, enabling precise verification of logical compliance and comprehensive path coverage. We introduce an Adversarial Intent Taxonomy and a modular Extension Mechanism, enabling low-cost deployment across domains and facilitating automated dialogue data synthesis. Evaluation is conducted via a framework where Judge Agents and a Rule Engine analyze interactions between User and Service Agents to generate deterministic ground truth. Extensive experiments on 27 LLMs across 6 industrial scenarios reveal a significant ``Execution Gap'' where models accurately classify intents but fail to derive correct subsequent actions. We also observe ``Empathy Resilience'', a phenomenon where models maintain polite conversational facades despite underlying logical failures under high adversarial intensity. Code and resources are available at https://anonymous.4open.science/r/SAGE-Bench-4CD3/.",
      "published": "2026-04-10T12:55:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.09285v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09285v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Statistical Properties of the King Wen Sequence: An Anti-Habituation Structure That Does Not Improve Neural Network Training",
      "authors": [
        "Augustin Chan"
      ],
      "abstract": "The King Wen sequence of the I-Ching (c. 1000 BC) orders 64 hexagrams -- states of a six-dimensional binary space -- in a pattern that has puzzled scholars for three millennia. We present a rigorous statistical characterization of this ordering using Monte Carlo permutation analysis against 100,000 random baselines. We find that the sequence has four statistically significant properties: higher-than-random transition distance (98.2nd percentile), negative lag-1 autocorrelation (p=0.037), yang-balanced groups of four (p=0.002), and asymmetric within-pair vs. between-pair distances (99.2nd percentile). These properties superficially resemble principles from curriculum learning and curiosity-driven exploration, motivating the hypothesis that they might benefit neural network training. We test this hypothesis through three experiments: learning rate schedule modulation, curriculum ordering, and seed sensitivity analysis, conducted across two hardware platforms (NVIDIA RTX 2060 with PyTorch and Apple Silicon with MLX). The results are uniformly negative. King Wen LR modulation degrades performance at all tested amplitudes. As curriculum ordering, King Wen is the worst non-sequential ordering on one platform and within noise on the other. A 30-seed sweep confirms that only King Wen's degradation exceeds natural seed variance. We explain why: the sequence's high variance -- the very property that makes it statistically distinctive -- destabilizes gradient-based optimization. Anti-habituation in a fixed combinatorial sequence is not the same as effective training dynamics.",
      "published": "2026-04-10T11:44:09Z",
      "abstract_url": "http://arxiv.org/abs/2604.09234v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09234v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.NE"
      ]
    },
    {
      "title": "GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking",
      "authors": [
        "Yunqiang Wang",
        "Hengyuan Na",
        "Di Wu",
        "Miao Hu",
        "Guocong Quan"
      ],
      "abstract": "Audio large language models (ALLMs) enable rich speech-text interaction, but they also introduce jailbreak vulnerabilities in the audio modality. Existing audio jailbreak methods mainly optimize jailbreak success while overlooking utility preservation, as reflected in transcription quality and question answering performance. In practice, stronger attacks often come at the cost of degraded utility. To study this trade-off, we revisit existing attacks by varying their perturbation coverage in the frequency domain, from partial-band to full-band, and find that broader frequency coverage does not necessarily improve jailbreak performance, while utility consistently deteriorates. This suggests that concentrating perturbation on a subset of bands can yield a better attack-utility trade-off than indiscriminate full-band coverage. Based on this insight, we propose GRM, a utility-aware frequency-selective jailbreak framework. It ranks Mel bands by their attack contribution relative to utility sensitivity, perturbs only a selected subset of bands, and learns a reusable universal perturbation under a semantic-preservation objective. Experiments on four representative ALLMs show that GRM achieves an average Jailbreak Success Rate (JSR) of 88.46% while providing a better attack-utility trade-off than representative baselines. These results highlight the potential of frequency-selective perturbation for better balancing attack effectiveness and utility preservation in audio jailbreak. Content Warning: This paper includes harmful query examples and unsafe model responses.",
      "published": "2026-04-10T11:27:25Z",
      "abstract_url": "http://arxiv.org/abs/2604.09222v1",
      "pdf_url": "https://arxiv.org/pdf/2604.09222v1",
      "categories": [
        "cs.SD",
        "cs.AI"
      ]
    }
  ]
};
