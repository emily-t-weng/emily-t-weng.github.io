const PAPERS_DATA = {
  "last_updated": "2026-06-09 04:12:57 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "An Agency-Transferring Model-Free Policy Enhancement Technique",
      "authors": [
        "Anton Bolychev",
        "Georgiy Malaniya",
        "Sinan Ibrahim",
        "Pavel Osinenko"
      ],
      "abstract": "Training reinforcement learning (RL) policies from scratch is costly: it requires careful reward and environment design, extensive tuning, and substantial computation. Yet many control problems already have a functional but suboptimal policy available as a baseline. This paper proposes a method for embedding such a baseline into the RL training process, simultaneously improving training efficiency relative to from-scratch methods and producing a learning policy that outperforms the baseline. At each step, the method arbitrates between the baseline policy and a trainable learning policy, initially relying strongly on the baseline policy and then progressively transferring agency to the learning policy. By the end of training, the learning policy is a standalone neural network that operates without baseline policy support. The paper formalizes what it means for the baseline policy to be functional: under this policy, the agent reaches a goal set and remains there with high probability. The proposed arbitration mechanism is designed to exploit this property during training, yielding high goal-reaching rates right from the beginning of training. A theoretical analysis provides a formal interpretation of this behavior under stated assumptions and extends it to the final baseline-free regime, where explicit lower bounds are derived for the goal-reaching probability of the standalone learning policy. Empirical results on continuous-control benchmarks show that the proposed method achieves returns that match or exceed those of competitive approaches, while maintaining the highest goal-reaching rates throughout training among the compared methods -- including in the final stage, where the learning policy operates without any baseline support.",
      "published": "2026-06-08T17:59:39Z",
      "abstract_url": "http://arxiv.org/abs/2606.09825v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09825v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY",
        "math.OC"
      ]
    },
    {
      "title": "Topological Neural Operators",
      "authors": [
        "Lennart Bastian",
        "Samuel Leventhal",
        "Mustafa Hajij",
        "Tolga Birdal"
      ],
      "abstract": "We introduce Topological Neural Operators (TNOs), a principled framework for operator learning on cell complexes that lifts neural operators (NOs) from functions on points and/or edges to topological domains. TNOs represent data as features defined on cells of varying dimension and model their interactions through Discrete Exterior Calculus, enabling explicit cross-dimensional coupling via gradient-, curl-, and divergence-type operators. The key design principle is to decouple where information flows, as governed by fixed topological operators, from how it is transformed (which is learned), yielding models that respect the geometric support of physical quantities and expose conservation and compatibility structure. We further propose Hierarchical TNOs (HTNOs), which incorporate learned coarse complexes to propagate long-range and topology-dependent information. Our framework subsumes existing NOs as a special case, providing a unified perspective on operator learning across discretizations. Across a range of PDE benchmarks, including irregular-geometry flow problems, TNOs and HTNOs improve accuracy; controlled studies further isolate the benefits of native higher-rank and topological structure. Project page: https://circle-group.github.io/research/TNO",
      "published": "2026-06-08T17:54:33Z",
      "abstract_url": "http://arxiv.org/abs/2606.09806v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09806v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Bandits for Efficient Experimentation: Adapting to Control Group, Preferences, and Context Drifts",
      "authors": [
        "Udvas Das",
        "Waris Radji",
        "Debabrota Basu",
        "Odalric-Ambrym Maillard"
      ],
      "abstract": "We consider a variant of the linear contextual stochastic multi-armed bandits, where the learner must provide recommendations to a group of users, each having its personalized preference vector, and in the presence of context distributions that are drifting over time. Under practitioner-friendly assumptions, we reduce this setting to linear bandit with stationary mean but heteroskedastic and non-stationary noise. We further study the case when the learner must ensure the mean reward of each decision must exceed that of a baseline strategy $\\boldsymbolπ_0$ at each decision step. We introduce Dri-MED, an algorithm inspired from the linear version of the MED strategy, and carefully adapted to handle the non-stationary heteroskedastic noise. We show that the instance-dependent regret scales as $\\tilde{\\mathcal O}\\left(\\fracκ{\\tildeΔ}d^2(\\log(T)\\right)$, where $\\tildeΔ$ is the constraint-aware sub-optimality gap subject to policy $π_0$, with variance-aware multiplicative term $κ$ that we carefully handle using heteroskedastic regression. We further show Dri-MED enjoys $\\tilde{\\mathcal{O}}(d)$ expected constraint violations. Our numerical results suggest that Dri-MED significantly outperforms conservative baselines that ignores the drift and preference structure.",
      "published": "2026-06-08T17:53:29Z",
      "abstract_url": "http://arxiv.org/abs/2606.09802v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09802v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Data Synthesis and Parameter-Efficient Fine-Tuning for Low-Resource NMT: A Case Study on Q'eqchi' Mayan",
      "authors": [
        "Alexander Chulzhanov",
        "Soeren Eberhardt",
        "Arjun Mukherjee"
      ],
      "abstract": "Neural machine translation for digitally low-resource Indigenous languages is often hindered by extreme data scarcity, prompting reliance on extractive web-scraping. To ensure data sovereignty, this study introduces a data synthesis methodology to bootstrap NMT models without scraping target-language parallel text. Focusing on Q'eqchi' Mayan, we transformed community-sourced dictionaries into a massive synthetic corpus, utilizing Parameter-Efficient Fine-Tuning (PEFT) via LoRA adapters on an mT5-base model. In-domain evaluation demonstrates high structural acquisition (BLEU 42.02), proving that synthetic constraints effectively teach complex agglutinative morphology and VOS word order. However, evaluation against an organic glossary reveals a structural-semantic gap (BLEU 0.59), where the model maintains grammatical integrity but lacks the lexical grounding of natural language. The model exhibits overfitting to the constrained structural variance of the synthetic templates; despite high semantic entropy in the pipeline, it struggles with the syntactic fluidity of natural language, forcing organic inputs into rigid learned patterns. Furthermore, an ablation study utilizing a Multi-Task Learning architecture resulted in negative transfer, suggesting that auxiliary tasks competed for limited parameter capacity within the LoRA adapters, causing over-optimization for synthetic markers at the expense of organic flexibility. Ultimately, we establish that synthetic bootstrapping is a highly effective structural primer, but requires authentic data for semantic refinement via Curriculum Learning.",
      "published": "2026-06-08T17:29:08Z",
      "abstract_url": "http://arxiv.org/abs/2606.09767v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09767v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Preserving Plasticity in Continual Learning via Dynamical Isometry",
      "authors": [
        "Andries Rosseau",
        "Robert Müller",
        "Ann Nowé"
      ],
      "abstract": "Continual training of deep neural networks under non-stationarity often leads to a progressive loss of plasticity, eventually limiting further learning. We relate plasticity to the empirical Neural Tangent Kernel, and identify dynamical isometry (the condition that layer-wise Jacobian singular values remain close to one) as a key mechanism for preserving plasticity in continual learning. We revisit a class of networks that are almost-everywhere isometric while remaining universal Lipschitz function approximators, demonstrating that near-dynamical isometry is compatible with expressive nonlinear representations. For general architectures, we propose an efficient isometry-promoting regularization scheme and identify a novel mechanism by which it can reactivate dormant ReLU units. Building on this, we introduce AdamO, an Adam-style adaptive optimizer that decouples isometry regularization from gradient updates, analogous to AdamW. We further reinterpret prior plasticity-preserving approaches through the lens of dynamical isometry, showing that they target only a partial measure of isometry. Across supervised and reinforcement-learning continual-learning benchmarks designed to induce plasticity loss, our methods consistently match or outperform existing approaches.",
      "published": "2026-06-08T17:24:15Z",
      "abstract_url": "http://arxiv.org/abs/2606.09762v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09762v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Difference-Aware Retrieval Policies for Imitation Learning",
      "authors": [
        "Quinn Pfeifer",
        "Ethan Pronovost",
        "Paarth Shah",
        "Khimya Khetarpal",
        "Siddhartha Srinivasa",
        "Abhishek Gupta"
      ],
      "abstract": "Parametric imitation learning via behavior cloning can suffer from poor generalization to out-of-distribution states due to compounding errors during deployment. We show that reusing the training data during inference via a semi-parametric retrieval-based imitation learning approach can alleviate this challenge. We present Difference-Aware Retrieval Policies for Imitation Learning (DARP), a semi-parametric retrieval-based imitation learning approach that addresses this limitation by reparameterizing the imitation learning problem in terms of local neighborhood structure rather than direct state-to-action mappings. Instead of learning a global policy, DARP trains a model to predict actions based on $k$-nearest neighbors from expert demonstrations, their corresponding actions, and the relative distance vectors between neighbor states and query states. DARP requires no additional assumptions beyond those made for standard behavior cloning -- it does not require additional data collection, online expert feedback, or task-specific knowledge. We demonstrate consistent performance improvements of 15-46% over standard behavior cloning across diverse domains, including continuous control and robotic manipulation, and across different representations, including high-dimensional visual features. Code and demos are available at https://weirdlabuw.github.io/darp-site/.",
      "published": "2026-06-08T17:18:19Z",
      "abstract_url": "http://arxiv.org/abs/2606.09758v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09758v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Multi-Turn Evaluation of Deep Research Agents Under Process-Level Feedback",
      "authors": [
        "Rishabh Sabharwal",
        "Hongru Wang",
        "Amos Storkey",
        "Jeff Z. Pan"
      ],
      "abstract": "Existing benchmarks for deep research agents (DRAs) assess only single-shot outputs, ignoring a key question: can DRAs improve their reports when guided by feedback? To investigate this, we conduct a multi-turn evaluation of DRAs under two feedback settings: self-reflection, in which the agent revises its report without any external diagnostic signal, and process-level feedback, in which the agent receives guidance targeting gaps in its research strategy. To enable process-level feedback, we design Research Gap Inference (RGI), a method that analyzes patterns of satisfied and unsatisfied rubric criteria to infer research-process gaps. Our analysis reveals three key findings: (i) under self-reflection, agents incorporate and regress on rubric criteria at nearly equal rates, yielding negligible net improvement; (ii) a single round of process-level feedback yields substantial gains, raising the normalized score by approximately $8$-$15$ points and yielding a roughly $35$-$40\\%$ incorporation rate; (iii) these gains do not compound over subsequent turns, as agents regress on up to $24\\%$ of previously satisfied criteria when rewriting the full report to address remaining gaps. Even with targeted guidance, reliable multi-turn improvement remains out of reach for the DRA architectures we evaluate. Our code and results are publicly available at https://github.com/sabharwalrishabh/Multi-Turn-Evaluation-of-DRAs.",
      "published": "2026-06-08T17:08:36Z",
      "abstract_url": "http://arxiv.org/abs/2606.09748v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09748v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Hybrid Robustness Verification for Spatio-Temporal Neural Networks",
      "authors": [
        "Sherwin Varghese",
        "Matthew Wicker",
        "Alessio Lomuscio"
      ],
      "abstract": "With AI increasingly deployed in safety-critical systems, providing formal robustness guarantees for the underlying models is essential. Existing verification methods either rely on overly conservative approximations or incur prohibitive computational costs. For example, the use of lp-norm perturbations in video settings encodes the belief that the adversary can inject noise in every video frame. In practice, adversarial perturbations exhibit structured spatial and temporal correlations, constrained to lower-dimensional, semantically meaningful subspaces. In this work, we study robustness verification of 3D CNNs processing video and volumetric inputs, targeting applications in action recognition (UCF-101), autonomous driving (Udacity), and medical imaging (MedMNIST) exploiting realistic assumptions on adversarial strength by modelling them as spatio-temporal constraints - where the attacker can modify either a subset of frames or patches within a set of consecutive frames. We demonstrate that modelling realistic constraints enables tighter approximations. We introduce Spatio-Temporal Bound Propagation (STBP), a verification framework that computes an exact closed-form characterization of the first convolutional layer and propagates certified bounds through subsequent layers using scalable approximations. Computing the exact closed form provides the tightest bounds for the first convolutional layer. Thus, we utilise approximation methods in the remainder of the network. To spur further progress in this field, we propose ST-Bench, a verification benchmark for autonomous driving and activity recognition, to systematically evaluate verifiable robustness. Compared to existing verification-based approaches, STBP provides stronger robustness guarantees with significantly improved scalability, achieving 1.7x higher certified robust accuracy under identical perturbation budgets.",
      "published": "2026-06-08T17:06:51Z",
      "abstract_url": "http://arxiv.org/abs/2606.09746v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09746v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SearchSwarm: Towards Delegation Intelligence in Agentic LLMs for Long-Horizon Deep Research",
      "authors": [
        "Pu Ning",
        "Quan Chen",
        "Kun Tao",
        "Xinyu Tang",
        "Tianshu Wang",
        "Qianggang Cao",
        "Xinyu Kong",
        "Zujie Wen",
        "Zhiqiang Zhang",
        "Jun Zhou"
      ],
      "abstract": "Large language models are increasingly expected to handle complex, long-horizon real-world tasks whose context demands can grow without bound, yet model context windows remain inherently finite. Recent work explores a paradigm where a main agent decomposes tasks and dispatches subtasks to subagents, which execute and return only summarized results, conserving the main agent's context budget. However, performing this well requires delegation intelligence: the ability to decompose complex tasks, determine when and what to delegate, and integrate returned results into the ongoing workflow. Training data for this capability is scarce in naturally occurring text, and to our knowledge, how to synthesize such data and train models to acquire this capability remains largely unexplored in the open-source community. To bridge this gap, we present a preliminary exploration targeting deep research, a representative long-horizon agent task. Specifically, we design a harness that guides the model toward high-quality task decomposition and delegation, while constraining subagents to return results properly to support the main agent's workflow. The harness-guided trajectories naturally encode correct delegation decisions, which we use as supervised fine-tuning data to internalize delegation intelligence into model weights. Our resulting model, SearchSwarm-30B-A3B, achieves 68.1 on BrowseComp and 73.3 on BrowseComp-ZH, the best results among all models of comparable scale. We will release our harness, model weights, and training data to facilitate future research.",
      "published": "2026-06-08T16:52:26Z",
      "abstract_url": "http://arxiv.org/abs/2606.09730v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09730v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Proxy Reward Internalization and Mechanistic Exploitation: A Learned Precursor to Reward Hacking and Its Generalization",
      "authors": [
        "Mohammad Beigi",
        "Ming Jin",
        "Lifu Huang"
      ],
      "abstract": "Reward hacking is usually studied after it becomes visible, once a model earns high proxy reward while failing the intended task. We instead study what proxy RL teaches before that failure appears. We introduce Proxy Reward Internalization and Mechanistic Exploitation (PRIME), a learned capability to assess task correctness, predict proxy acceptance, and reason about exploitable proxy--gold gaps. In coding RL environments with exploitable pytest rewards, we measure PRIME through chain-of-thought monitoring, direct probes, and activation-level concept vectors. We find that PRIME emerges in a staged sequence before sustained reward hacking, and that its current direct-probe score forecasts later hack onset and severity even when the visible hack rate is still low. PRIME also adapts when the evaluator changes, retargeting to whichever proxy--gold gap remains rewarded and persisting when gold reward suppresses overt hacking, and ablating its activation directions reduces hacking. Across checkpoints, in-domain PRIME tracks out-of-domain misalignment. Together these results suggest that exploitable proxy RL amplifies a proxy-internalization capability upstream of visible hacking, making PRIME a candidate early-warning signal for broader alignment risk.",
      "published": "2026-06-08T16:32:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.09711v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09711v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO",
      "authors": [
        "Blake Bullwinkel",
        "Eugenia Kim",
        "Amanda Minnich",
        "Mark Russinovich"
      ],
      "abstract": "AI red teaming must continually adapt to evolving attackers and defenders. Reinforcement learning offers a promising approach to discovering novel attacks, and co-training methods can produce more robust defenders in tandem. Recent works have demonstrated the efficacy of attacker-defender co-training by applying PPO and DPO, but report that GRPO is unstable in this setting. We introduce AdvGRPO, a co-training framework that makes GRPO viable for joint attacker-defender optimization using dense multi-channel rewards and decoupled advantage normalization. Training progresses through a curriculum from single-turn to closed-loop multi-turn attacks before bootstrapping co-training, where attacker and defender models are updated in alternation. We show that our method can produce highly effective and transferable attacks and that co-trained defenders outperform baselines on safety benchmarks.",
      "published": "2026-06-08T16:21:36Z",
      "abstract_url": "http://arxiv.org/abs/2606.09701v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09701v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "An 84-Format Numeric Catalog with Bit-Exact Conformance Vectors: A Vendor-Neutral Reference for FP8, BF16, MXFP4, and Microscaling Formats",
      "authors": [
        "Dmitrii Vasilev"
      ],
      "abstract": "Numeric format proliferation in machine learning hardware -- FP8 (E4M3 and E5M2), BF16, MXFP4, microscaling block formats, and dozens of research variants -- has outpaced the availability of vendor-neutral, bit-exact reference material. Engineers porting models across accelerators encounter silent divergences that are difficult to diagnose without a shared ruler. This paper describes a catalog of 84 numeric formats spanning 13 families, a suite of six bit-exact conformance packs covering GF16, MXFP4 element, BF16, FP8 E4M3, FP8 E5M2, and E8M0 block scale, and an IEEE P3109 v3.2.0 cross-walk that maps each pack to its corresponding standards-track configured format. Each pack is a self-contained JSON document with a SHA-256 fingerprint, a shared row schema, and an anchor vector that encodes 3.0 -- the identity phi^2 + 1/phi^2 = 3 -- as a cross-pack sanity check. Packs are cross-validated against ml_dtypes 0.5.4 (Google/JAX); any divergence is documented explicitly and interpreted as a spec-permitted interpretation gap rather than hidden. The work is framed as registry filling: it does not propose new formats, make model-accuracy claims, or assert superiority over any vendor's implementation. All artifacts are publicly available at https://github.com/gHashTag/t27 under an open license.",
      "published": "2026-06-08T16:04:15Z",
      "abstract_url": "http://arxiv.org/abs/2606.09686v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09686v1",
      "categories": [
        "cs.AR",
        "cs.AI",
        "cs.MS",
        "cs.PF",
        "math.NA"
      ]
    },
    {
      "title": "Correlation Is Not Enough: Embedding Human Metadata for Individual Causal Discovery",
      "authors": [
        "Suraj Biswas",
        "Saurabh Gupta",
        "Pritam Mukherjee"
      ],
      "abstract": "Ask a pretrained biomedical language model whether \"cortisol 28 ug/dL\" and \"stock-market volatility\" are related, and it returns a cosine similarity of 0.83 on a scale where 1.0 means identical. The two share no mechanism. This is not a corner case: every off-the-shelf biomedical encoder we tested (BioBERT, PubMedBERT, BioM-ELECTRA) scores unrelated cross-domain pairs between 0.76 and 0.92 when the answer should be near zero. Accuracy on cross-domain discrimination is 0%. Retrieval systems survive this, because a language model downstream filters the noise. A Large Behavioural Model (LBM), a foundation model whose subject is a person rather than a sentence, does not: it reasons over a graph of a user's life and treats embedding proximity as evidence that two events are causally linked. False proximity writes a false causal edge, and everything downstream inherits the error. Here, embedding geometry is not a tuning knob; it is correctness. We report the fix. A contrastive pass over 72,034 pairs raises PubMedBERT BIOSSES correlation from 0.633 to 0.828 and within-vs-across-domain separation from 1.05x to 1.63x. A second pass, BODHI, mines hard negatives from edges absent in a biomedical knowledge graph and lifts separation to 2.30x and the discrimination gap to +0.392, at a 4.5% BIOSSES cost. On an Intel Xeon 6737P with AMX, OpenVINO cuts single-query latency from 1367 ms to 10 ms (133x) and reaches 555 sentences/sec. One finding contradicts standard advice: FP16 beats INT8 on this silicon at every serving batch size, and we explain why. The same model on a no-AMX Ice Lake instance runs 13-27x slower. We release the benchmark suite, training corpora, the BODHI generator, and the OpenVINO scripts.",
      "published": "2026-06-08T15:54:28Z",
      "abstract_url": "http://arxiv.org/abs/2606.09672v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09672v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.PF",
        "q-bio.QM"
      ]
    },
    {
      "title": "Transition-Based Digital Twin Modelling for Alzheimer's Disease under Sparse Longitudinal Data",
      "authors": [
        "Yinyu Huang",
        "Yilin Zhang",
        "Sofia Michopoulou",
        "Christopher Kipps",
        "Rahman Attar"
      ],
      "abstract": "Alzheimer's disease (AD) progression is highly heterogeneous and is typically observed through sparse and irregular longitudinal data, posing challenges for prediction and personalised monitoring. Existing machine learning approaches have improved AD prediction using multimodal data, yet often focus on static classification or cohort-level risk estimation, providing limited support for subject-specific modelling and uncertainty-aware reasoning. To address these limitations, we present a personalised digital twin framework for AD prediction and scenario-based analysis using multimodal longitudinal data. The proposed approach integrates complementary modelling strategies to capture clinical transitions and temporal dependencies across visits. Using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), including cognitive assessments, clinical variables, and MRI-derived phenotypes, the framework predicts cognitive status and diagnostic categories while quantifying predictive uncertainty and enabling patient-specific what-if trajectory analysis. Evaluation on leak-free subject-level splits demonstrates strong performance in score forecasting and diagnosis classification. In this sparse and irregular ADNI setting, transition-based modelling of adjacent visits achieved higher predictive accuracy than the sequence-based branch, suggesting that local transition modelling may be more data-efficient. While sequence models remain valuable for uncertainty-aware trajectory forecasting, local transition modelling offers a more data-efficient and robust predictive strategy. These findings highlight the importance of aligning temporal modelling strategies with clinical data structure and suggest that transition-based digital twin formulations may provide a practical and interpretable approach for personalised disease forecasting in neurodegenerative disorders.",
      "published": "2026-06-08T15:54:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.09671v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09671v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SpatialWorld: Benchmarking Interactive Spatial Reasoning of Multimodal Agents in Real-World Tasks",
      "authors": [
        "Hongcheng Gao",
        "Hailong Qu",
        "Jingyi Tang",
        "Jiahao Wang",
        "Zihao Huang",
        "Hengkang Qiao",
        "Shihong Huang",
        "Junming Yang",
        "Yi Li",
        "Hongyixuan Yuan",
        "Wenjie Li",
        "Bohan Zeng",
        "Wenbo Li",
        "Bo Wang",
        "Jianhui Liu",
        "Olive Huang",
        "Haoyang Huang",
        "Wentao Zhang",
        "Guoqing Huang",
        "Nan Duan",
        "Yinpeng Dong"
      ],
      "abstract": "Spatial reasoning is a foundational capability for multimodal large language models (MLLMs) to perceive and operate within the physical world. However, existing benchmarks predominantly rely on passive evaluation (e.g., static VQA) or simulator-specific pipelines, failing to assess general interactive spatial understanding. We introduce SpatialWorld, a unified benchmark designed specifically for evaluating the interactive spatial understanding of multimodal agents in complex real-world tasks. Integrating eight heterogeneous simulation backends under a shared, simulator-agnostic protocol, SpatialWorld features 760 human-annotated tasks across diverse domains (e.g., household routines, travel, social collaboration). Agents must solve tasks under vision-only partial observability, actively gathering egocentric visual evidence and expressing decisions via a unified, text-based action interface native to MLLMs. For reliable evaluation, each task includes a human-validated initial state, a reference trajectory, and a terminal-state verifier. Evaluating 15 advanced agents reveals that robust spatial task solving remains challenging: the strongest model, GPT-5, achieves an average task success rate (TSR) of only 17.4%, while the leading open-source model, Qwen-3.5, reaches 14.1%. Further analysis exposes a clear mismatch between task success and execution efficiency, alongside substantial domain-specific performance variations. These bottlenecks in active exploration and long-horizon planning position SpatialWorld as a rigorous testbed for future spatial agents.",
      "published": "2026-06-08T15:51:51Z",
      "abstract_url": "http://arxiv.org/abs/2606.09669v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09669v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "End-to-End Context Compression at Scale",
      "authors": [
        "Ang Li",
        "Sean McLeish",
        "Haozhe Chen",
        "Nimit Kalra",
        "Zaiqian Chen",
        "Artem Gazizov",
        "Venkata Anoop Suhas Kumar Morisetty",
        "Bhavya Kailkhura",
        "Harshitha Menon",
        "Zhuang Liu",
        "Brian R. Bartoldson",
        "Tom Goldstein",
        "Sanae Lotfi",
        "Micah Goldblum",
        "Pavel Izmailov"
      ],
      "abstract": "Long-context language model inference is bottlenecked by memory, as the KV cache grows with context length. Recent techniques to compress the KV cache fall short: they either degrade model quality substantially or require considerable time and compute to compress a single long prompt. Furthermore, many methods require the input to fit within the target model's context window, and are generally incompatible with modern production inference engines. Encoder-decoder compressors, which map a long token sequence to a shorter sequence of latent embeddings consumed by a decoder, are an appealing alternative in principle. However, existing approaches are not competitive with KV cache compression on the accuracy-efficiency frontier. In this work, we revisit encoder-decoder compression and close this gap. We first perform an architecture search, pre-training many variants from scratch to determine how best to design and train encoder-decoder compressors. Guided by our findings, we continually pre-train a family of 0.6B-encoder, 4B-decoder models on over 350B tokens each, at compression ratios of 1:4, 1:8, and 1:16. We introduce Latent Context Language Models (LCLMs), a family of compressors that improve the Pareto frontier across general-task performance, compression speed, and peak memory usage. We demonstrate that LCLMs serve as efficient backbones for long-horizon agents, letting the agent skim through a compressed long context and adaptively expand relevant segments on demand.",
      "published": "2026-06-08T15:43:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.09659v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09659v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Muon Learns More Robust and Transferable Features than Adam",
      "authors": [
        "Tianyu Ruan",
        "Fengzhuo Zhang",
        "Shuche Wang",
        "Shihua Zhang"
      ],
      "abstract": "Muon has recently emerged as a state-of-the-art optimizer for pretraining Large Language Models (LLMs) and vision classifiers. Despite its efficiency advantage over Adam and SGD, the feature-learning advantage of Muon remains unclear. This paper investigates Muon's feature-learning advantage through the lens of robustness and transferability. First, by evaluating pretrained models on corrupted images and texts, we show that features learned by Muon are consistently more robust than those learned by Adam and SGD across different architectures, including transformers and Convolutional Neural Networks (CNNs). Using trained layer-wise probes, we further show that this robustness advantage is reflected in larger logit margins across layers. Second, by training linear classifiers or fine-tuning full models from pretrained parameters on downstream tasks, we demonstrate that Muon-learned features transfer more effectively than those learned by Adam and SGD. This transferability advantage is further supported by the diversity of hidden states across layers, as measured by effective rank. Finally, in a representative classification problem with multi-component features, we prove that Muon attains larger margins and higher effective rank than Adam and SGD, providing theoretical support for our empirical findings.",
      "published": "2026-06-08T15:42:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.09658v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09658v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Do Video Foundation Models Understand Intuitive Physics? A Layerwise Probing Analysis",
      "authors": [
        "Samuele Punzo",
        "Niccolò Caselli",
        "Ippokratis Pantelidis",
        "Francesco Massafra",
        "Salvatore Lo Sardo",
        "Mohammadreza Salehi"
      ],
      "abstract": "We study whether pretrained video foundation models encode intuitive-physics information in their frozen representations, and how this information varies across model families, layers, and probe types. Using frozen-feature probing on IntPhys2 and Minimal Video Pairs (MVP), we compare predictive joint-embedding models (V-JEPA), masked reconstruction models (VideoMAE), and a diffusion-based video generator (LTX-Video). V-JEPA achieves the strongest overall results across benchmarks, especially with probes that model temporal dynamics, while VideoMAE remains competitive and LTX-Video recovers weaker but non-trivial signal. Layerwise analyses show that physics-relevant information is weakest in early layers and becomes most accessible at intermediate-to-late depth, and temporal controls show that disrupting frame order substantially reduces performance, especially on MVP. Together, these results suggest that intuitive-physics knowledge emerges reliably in pretrained video representations, but its accessibility depends strongly on pretraining paradigm, representational depth, and readout mechanism.",
      "published": "2026-06-08T15:40:32Z",
      "abstract_url": "http://arxiv.org/abs/2606.09646v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09646v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "FMplex: Model Virtualization for Serving Extensible Foundation Models",
      "authors": [
        "Hetvi Shastri",
        "Pragya Sharma",
        "Walid A. Hanafy",
        "David Irwin",
        "Mani Srivastava",
        "Prashant Shenoy"
      ],
      "abstract": "Foundation models (FMs) are increasingly used as backbones for downstream tasks across language, vision, time-series, and multimodal applications. Yet existing model-serving systems deploy each customized task as an independent model instance, thereby replicating heavyweight backbones, wasting accelerator memory, and losing opportunities to amortize batching and loading costs. This paper presents FMplex, a serving system that treats FM backbones as a virtualization substrate for deployment sharing. FMplex presents each task with a virtual foundation model (vFM), a logically private FM instance backed by a shared physical FM. This abstraction lets independently customized tasks share a backbone while preserving task-specific extensions, independent lifecycles, and task-level isolation. In addition, we propose a batch-aware fair-queueing scheduler that combines weighted task-level sharing with inter- and intra-task batching across colocated tasks. We implement a FMplex-based serving stack spanning task construction, sharing-aware deployment, and runtime execution. Across 7 FM backbones (16 variants) and 92 downstream tasks, FMplex reduces latency by up to 80% over spatial partitioning and 33.3% over best-effort co-location, while hosting up to 6x more tasks at cluster scale.",
      "published": "2026-06-08T15:38:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.09643v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09643v1",
      "categories": [
        "cs.DC",
        "cs.AI",
        "cs.LG",
        "cs.OS"
      ]
    },
    {
      "title": "ReCoVLA: VLM-Guided Reward Compilation for Failure Recovery in Vision-Language-Action Policies",
      "authors": [
        "Haodi Hu",
        "Chung-Ta Huang",
        "Jing Liu",
        "Ye Wang",
        "Kei Suzuki",
        "Matthew Brand",
        "Toshiaki Koike-Akino"
      ],
      "abstract": "Vision-language-action (VLA) policies provide strong priors for language-conditioned manipulation, but remain brittle in off-nominal states requiring targeted recovery. We propose ReCoVLA -- a failure-conditioned residual recovery framework that keeps a pretrained VLA policy frozen, uses an external vision-language model (VLM) to infer the failure mode and recovery stage, and compiles a structured reward from task-relevant components. Rather than using the VLM to generate actions or rewards directly, ReCoVLA uses it as a semantic reward selector: it predicts a recovery descriptor and reward mask for in-simulation residual-policy training, followed by zero-shot sim-to-real deployment of the trained recovery policies. This decouples high-level failure understanding from low-level corrective control to support different VLAs. Experiments across short-horizon, long-horizon, and contact-rich manipulation tasks show that ReCoVLA outperforms the tested baselines on average. In simulation, our reward compiler improves average success from 36.7% for the fine-tuned $π_{0.5}$ baseline to 66.7%. In physical zero-shot sim-to-real experiments, ReCoVLA achieves the best average performance, with 61.7% success.",
      "published": "2026-06-08T15:29:09Z",
      "abstract_url": "http://arxiv.org/abs/2606.09630v1",
      "pdf_url": "https://arxiv.org/pdf/2606.09630v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
