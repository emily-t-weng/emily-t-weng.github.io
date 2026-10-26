const PAPERS_DATA = {
  "last_updated": "2026-10-26 04:22:44 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms",
      "authors": [
        "Xi Chen",
        "Fengqi Hu",
        "Zeyu Shen",
        "Jiaying Xu",
        "Shangding Gu",
        "Pan Xu",
        "Anqi Liu",
        "Laixi Shi"
      ],
      "abstract": "Robust reinforcement learning (RL) is critical for reliable real-world deployment, where policies may fail under diverse shifts and disruptions. Existing robust RL algorithms offer a rich set of mechanisms and principles, yet two major barriers limit their broader potential: limited algorithm reusability and an unclear empirical capability boundary. We introduce RobustRLlib, an algorithm-centric library and benchmark that addresses both challenges. It integrates 17 representative robust online, offline, and safe RL algorithms under a unified interface for substantially improved reusability, together with an enhanced evaluation platform spanning diverse real-world shifts and advanced humanoid and vision-language-action tasks. Systematic evaluation over 273 perturbation configurations provides an up-to-date capability map of current robust RL algorithms, revealing where they remain effective and where substantial robustness gaps persist. As an open-source, user-friendly platform, RobustRLlib offers a reproducible and extensible codebase for applying existing methods to new problems and developing new robust RL algorithms.",
      "published": "2026-10-23T17:57:22Z",
      "abstract_url": "http://arxiv.org/abs/2610.32971v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32971v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection",
      "authors": [
        "Amber Xie",
        "Jiawei Fu",
        "Xiaofeng Guo",
        "Carmelo Sferrazza",
        "Guanya Shi",
        "Haozhi Qi"
      ],
      "abstract": "Collecting high-quality demonstrations remains a central bottleneck for learning dexterous manipulation. Teleoperation records executable robot actions, but it requires a complete robot setup and places the burden of coordinating many finger joints on the operator. Human demonstrations are easier to collect, but their motions must be estimated and retargeted to the robot, and they do not directly provide robot actions. In this paper, we introduce DexPort, a portable data-collection interface for high-DoF robot hands. It uses the same hand hardware for collection and deployment, avoiding a hand embodiment gap, and records both commanded joint targets and measured joint states. Our key observation is that a small set of skills can support a wide range of dexterous tasks. During collection, the operator moves the hand-held interface to specify the robot wrist motion and selects or adjusts skills that coordinate the finger motion. This removes the need to control each joint independently while retaining behaviors such as precise grasping and in-hand manipulation. We collect demonstrations across several real-world tasks and in unstructured environments without a robot arm. We then use behavior cloning to train visuomotor policies directly on the recorded wrist motions and hand commands, without skill labels. We evaluate across 4 tasks: pick-and-place with diverse objects, screwing with in-hand tool use, drawer stowage which requires simultaneous grasp and pulling, and a bimanual scanning task. We collect over 2,500 demonstrations with our portable device, in both laboratory and in-the-wild settings. Across our tasks, we see an improvement from 11% to 64% when using recorded robot commands instead of states, and an improvement from 36% to 78% for task success when using in-the-wild data.",
      "published": "2026-10-23T17:56:28Z",
      "abstract_url": "http://arxiv.org/abs/2610.32969v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32969v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Monetizing Intelligence: Advertising and Subscription Design on LLM Platforms",
      "authors": [
        "Yuhang Ji",
        "Fasheng Xu",
        "Tianjun Feng"
      ],
      "abstract": "Large language model (LLM) platforms introducing advertising must decide how much capability ad revenue should fund and what consumers should pay to avoid ads. We study a monopolistic platform that jointly chooses costly response quality, advertising intensity, and subscription prices when advertising reduces user value but better service raises advertising returns. Three findings follow. First, when partly ad-supported and ad-free plans coexist, advertising can finance useful service for consumers who receive none under pure subscription. Lower effective quality in middle plans weakens the alternatives available to premium users, allowing higher prices for unchanged ad-free service. Some fully ad-supported users gain utility, while partly ad-supported and ad-free users lose: broader access can coexist with utility polarization. Second, an intermediate advertising cap can maximize consumer surplus by preserving some revenue that finances service while limiting degradation. Third, cheaper compute can raise quality and reduce ad intensity, yet sufficiently large cost reductions can increase aggregate advertising revenue above its initial level because better service raises the value of the remaining advertised interactions. An extension shows how advertising-funded upgrades can bring plans to the same maximum capability and advertising exposure, eliminating the basis for different prices. These results make menu design central to evaluating LLM advertising: the consequences extend beyond users who see ads to the alternatives and prices faced by those who do not.",
      "published": "2026-10-23T17:39:42Z",
      "abstract_url": "http://arxiv.org/abs/2610.32945v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32945v1",
      "categories": [
        "econ.TH",
        "cs.AI"
      ]
    },
    {
      "title": "SemantEq: Faithful Bidirectional Equivalence of Formal Statements",
      "authors": [
        "Xinrui Li",
        "Pawan Sasanka Ammanamanchi",
        "Siddharth Bhat",
        "Rianna Jamdar",
        "Yongxi Lin",
        "Stella Biderman"
      ],
      "abstract": "Statement autoformalization translates mathematics from natural language into a proof assistant such as Lean 4, and models now do it at scale. As provers get stronger, the burden of correctness shifts from the proof to the statement, that is, to whether the formal statement faithfully translates its source. Given a trusted reference formalization, the natural test is logical equivalence, and it proves too much. $P \\leftrightarrow Q$ holds trivially for any two statements $P$ and $Q$ that the library proves, because a prover can prove each separately and pass through $P \\leftrightarrow \\mathrm{True} \\leftrightarrow Q$. The stronger the prover, the more unrelated pairs pass. Earlier methods restrict the prover and check that each proof names the other statement. We introduce SemantEq, a more principled relation that instead requires each statement to turn into the other only through a chain of allowed steps, such as logical rearrangement and rewriting with the library's own equations. Our programs judge a proof by the proof term Lean checks, not by its tactic script. SemantEq-Eval, a search without a model, reaches 100% precision and 65.6% recall on the ProofNetVerif test split under our adjudicated labels, against 95.8% and 61.8% for BEq+, and an 8.9-fold increase in its CPU time adds at most three pairs. A language model finds more proofs, but under two of four prompts it proves both directions of about two in five of the non-equivalent pairs. SemantEq-ProofAudit audits each model-written proof under a policy derived from the relation and accepts none of the 104 non-equivalent pairs, and it raises recall on 407 equivalent pairs from 65.4% to 81.1% under the best prompt. Separating the generation of equivalence proofs from their acceptance lets a stronger prover raise recall without changing what an acceptance means. We release code and data at: https://github.com/hlxr89/SemantEq",
      "published": "2026-10-23T17:20:12Z",
      "abstract_url": "http://arxiv.org/abs/2610.32919v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32919v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations",
      "authors": [
        "Sadra Emami",
        "Benjamin Weigell",
        "Mohammad Reza Mousavi",
        "Gunel Jahangirova",
        "Hugo Araujo",
        "Bernhard Bauer"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly used to optimize code for performance. This practice often assumes that improvements in execution speed naturally translate into overall lower energy consumption. We challenge this assumption empirically by measuring whether performance-optimized code produced by LLM self-refinement pipelines is also energy-efficient relative to canonical human reference solutions, and whether the energy cost of the optimization process can be offset by subsequent execution-time savings. We evaluate 11 open-source models across four decoding configurations on all 1,000 problems of EffiBench, a state-of-the-art performance benchmark, while collecting per-process, hardware-level energy measurements throughout. Inference energy spans 0.79 to 8.68 MJ across model-variant pairs, which is roughly equivalent to the energy of 250 to 2800 GPT-4.1 queries. Limiting the maximum number of output tokens has the largest impact, reducing energy use by up to 69% compared to the default configuration. On the execution side, 42 of 44 configurations produce code that consumes more energy per problem than the canonical human reference, with overheads ranging from +4% to +886%. The two exceptions achieve only marginal gains, with break-even points on the order of 10^6 to 10^7 executions. Mixture-of-Experts architectures consume substantially less inference energy than equivalently-sized dense models without sacrificing correctness, making architectural choice a more consequential sustainability lever than decoding parameters. Performance optimization and energy efficiency are not always aligned goals: LLM self-refinement consistently produces faster-seeking but energy-heavier code relative to what a human engineer would write. All artifacts are available at https://doi.org/10.5281/zenodo.23598170.",
      "published": "2026-10-23T16:58:01Z",
      "abstract_url": "http://arxiv.org/abs/2610.32891v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32891v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding",
      "authors": [
        "Nilesh Prasad Pandey",
        "Jangseon Park",
        "Jason Kong",
        "Onat Gungor",
        "Tajana Rosing"
      ],
      "abstract": "Reinforcement learning (RL) post-training improves the reasoning capabilities of large language models, but its cost is dominated by autoregressive rollout generation, which can exceed 70% of training time. Existing speculative-decoding methods are poorly suited to this setting: model-based drafters consume scarce GPU memory and may become misaligned as the policy evolves, while existing model-free methods do not explicitly exploit repetition across training epochs. We observe that recent rollouts of the same prompt provide a high-quality, self-updating draft source. Based on this insight, we propose NgramRL, a per-prompt n-gram drafter constructed from bounded recent rollout history. NgramRL requires no auxiliary model, drafter training, or persistent GPU-resident drafter state, while target-model verification preserves the rollout distribution. To hide CPU cache lookup and scheduling, we further introduce an adaptive fan-out prefetcher that predicts likely verification outcomes (j, b), consisting of the accepted draft length j and target-emitted continuation token b. A causal budget allocator concentrates prefetches on high-probability outcomes, while uncovered outcomes fall back to sequential lookup without changing committed tokens. Across two model families and three reasoning and structured-generation workloads, NgramRL commits 3.60-7.77 tokens per target-model forward in the warm-cache regime, up to 1.99x more than SuffixDecoding and 7.0x more than prompt-lookup decoding. On Spider, the prefetcher covers 96.6% of Qwen3-8B outcomes at a candidate budget of 16 and improves measured rollout throughput by 1.38x over synchronous NgramRL, approaching its 1.40x overlap ceiling.",
      "published": "2026-10-23T16:47:29Z",
      "abstract_url": "http://arxiv.org/abs/2610.32881v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32881v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery",
      "authors": [
        "Marek Masiak",
        "Andrzej Szablewski",
        "Nicola Cancedda",
        "Constantin Venhoff"
      ],
      "abstract": "Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and ablate a circuit that is both necessary and sufficient to implement the backdoor. Furthermore, our new fine-tuning method produces sparse circuits that are up to an order of magnitude smaller than a dense LoRA baseline, allowing surgical ablation of the malicious capability without sacrificing instruction-following performance. Finally, we propose a technique to construct model organisms with localised circuits using TopKLoRA latents and gradient-routing, which we use to validate the findings of our circuit search algorithm. We consider these findings as a step toward interpretable-by-design architectures that provide faithful units of analysis without compromising model performance. We share our code: https://github.com/interpretable-finetuning/TopKLoRA.",
      "published": "2026-10-23T16:37:29Z",
      "abstract_url": "http://arxiv.org/abs/2610.32866v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32866v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets",
      "authors": [
        "Joao P. A. Dantas",
        "Andre N. Costa",
        "Diego Geraldo",
        "Marcos R. O. A. Maximo",
        "Takashi Yoneyama"
      ],
      "abstract": "This work introduces PoKER, a novel probabilistic model engineered to optimize missile launch effectiveness in air-to-air scenarios, specifically within Beyond Visual Range (BVR) air combat. Unlike conventional Weapon Engagement Zone (WEZ) models that delineate zones based on static distances such as maximum, minimum, and no-escape ranges, PoKER applies machine learning to predict kill probabilities more accurately by integrating the stochastic behaviors of targets and missile miss distances. This model dynamically evaluates target behavior, greatly expanding the predictive capabilities of engagement analysis. By factoring in elements such as warhead lethality, target and shooter orientations, and the specific conditions of engagement, PoKER provides important insights into engagement dynamics and quantifies success probabilities. Consequently, it can potentially be an important tool for BVR air combat pilots, improving operational decision-making within this specialized combat domain.",
      "published": "2026-10-23T16:34:57Z",
      "abstract_url": "http://arxiv.org/abs/2610.32864v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32864v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules",
      "authors": [
        "M. Harper Langston",
        "Pierre-David Letourneau",
        "Dalton Jones",
        "Roland Memisevic",
        "Peter Scott",
        "Mingu Lee",
        "Richard A. Lethin",
        "Christopher Lott",
        "Fatih Porikli",
        "Harris Teague"
      ],
      "abstract": "We learn schedules for a continuous penalty relaxation of MAX-CUT by backpropagating through an unrolled classical solver. A state-blind schedule generalizes to instances substantially larger than those used for training. A state-adaptive bilinear schedule also generalizes out of distribution, outperforming its untrained baseline across 15 seeds ($t=4.1$). By contrast, an additive ablation with identical inputs and output heads is statistically indistinguishable from the bilinear model in distribution but performs worse than its own baseline out of distribution ($t=-11.7$). This separation appears without regularization and persists across a $30$ round weight-decay sweep: the bilinear advantage remains significant at every tested value, whereas the additive model never recovers. The results in this setting isolate how solver state enters the schedule, rather than state adaptivity alone, as the architectural distinction associated with out-of-distribution transfer.",
      "published": "2026-10-23T15:40:23Z",
      "abstract_url": "http://arxiv.org/abs/2610.32801v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32801v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "math.NA",
        "math.OC"
      ]
    },
    {
      "title": "AI Alignment for Digital Democracy: Fairer Voters Representation against LLM Biases",
      "authors": [
        "Yi Qi",
        "Srijoni Majumdar",
        "Evangelos Pournaras"
      ],
      "abstract": "AI representation is emerging in digital democracy, assisting in voter emulation and decision-making. Large language models (LLMs) behave differently depending on the model, the prompt context, and the voter information they receive, representing some groups more accurately than others and potentially making the emulation unfair to certain groups. We investigate demographic biases in LLM-emulated electorates: how accurately LLMs represent individual votes, whether some voter groups are represented more accurately than others, and whether AI alignment can reduce these disparities without sacrificing overall accuracy. Using three U.S. presidential elections, we compare a closed-source and an open-source LLM across voters' demographic features and test AI alignment strategies toward fairer voter representation. Removing gender from the voter profile reduces the largest group error of the models by 22-42%, and removing education reduces it by 64%, whereas removing self-reported race increases the mean group error by over 40%, all without lowering accuracy. Common alignment strategies, such as repeated queries or fairness instructions, do not reduce these errors and can double them.",
      "published": "2026-10-23T15:27:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.32780v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32780v1",
      "categories": [
        "cs.CY",
        "cs.AI"
      ]
    },
    {
      "title": "On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models",
      "authors": [
        "Leon Sick",
        "Lukas Hoyer",
        "Yannick Strümpler",
        "Christian Reisswig",
        "Timo Ropinski",
        "Federico Tombari"
      ],
      "abstract": "On-policy self-distillation enables post-training foundation models without relying on stronger external teachers. However, existing approaches rely on ground-truth question-answer pairs, human-curated problem sets, or other supervised signals. In this work, we reformulate on-policy self-distillation as a self-training framework, driven by self-generated question-answer pairs and privileged reasoning demonstrations. We push supervision to the limit: guided solely by coarse, high-level topic descriptions, the model synthesizes its own curriculum of queries and privileged reasoning traces without access to any human-curated prompts or answers. We find that self-distillation on fully synthetic privileged information enables a model to improve across multiple domains while retaining its capabilities on domains it is not self-trained on. We demonstrate the generality of this framework across text and multimodal domains, achieving consistent self-improvement on standard language benchmarks and Visual Question Answering. Our findings demonstrate that foundation models possess rich latent capabilities that can be amplified without external task supervision or curated problem distributions.",
      "published": "2026-10-23T15:12:03Z",
      "abstract_url": "http://arxiv.org/abs/2610.32770v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32770v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks",
      "authors": [
        "Yalin E. Sagduyu",
        "Tugba Erpek",
        "Nathan Stephenson",
        "Mengran Xue"
      ],
      "abstract": "This paper studies interactions between clean-label poisoning attack and defense in vehicular networks using federated learning (FL). Basic Safety Message (BSM) classification for anomaly detection is modeled as a distributed learning problem over a vehicular network with dynamic topology. As a critical threat to FL, a feature-space clean-label attack is introduced, wherein benign samples are perturbed toward malicious feature extremes, degrading class separability while preserving labels. To counter this, a feature-consistency relabeling defense is proposed to detect and relabel suspicious samples. A game-theoretic framework is developed to capture attack-defense interactions and characterize equilibrium strategies based on attack effectiveness and cost parameters. Results show that the attack scales effectively with compromised nodes, while the proposed defense keeps attack success low without degrading clean accuracy and shifts equilibrium regions toward attack-averse regimes.",
      "published": "2026-10-23T15:06:14Z",
      "abstract_url": "http://arxiv.org/abs/2610.32765v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32765v1",
      "categories": [
        "cs.NI",
        "cs.AI",
        "cs.DC",
        "cs.LG",
        "eess.SP"
      ]
    },
    {
      "title": "Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors",
      "authors": [
        "Jialiang Wang",
        "Qing Ma",
        "Xiong Zhou",
        "Gangfeng Hu",
        "Haoliang Li",
        "Xianming Liu"
      ],
      "abstract": "Contrastive learning has emerged as a popular paradigm for training representation models. While prior studies pursue more practical and less restrictive assumptions to enhance theoretical understanding of contrastive learning, they essentially describe the interconnections among representations within the data or induced by the function class. To encapsulate these assumptions, we propose the measure of internal discrepancy that quantifies the difference between the mean representation of positive samples and latent cluster centers. We then prove an internal discrepancy-aware generalization bound for downstream supervised tasks following contrastive learning, without requiring additional assumptions. This generalization bound not only serves as a unifying framework connecting various assumptions but also motivates us to further reduce the internal discrepancy, thereby introducing the relaxed k-means loss as an approximation. Moreover, we investigate the downstream behaviors of contrastive learning from the perspective of contrastive learning-incurred supervised losses. In the presence of class imbalance, we shed light on why contrastive learning can facilitate imbalanced learning. Finally, we investigate the optimality of these incurred supervised losses, proving several intriguing properties.",
      "published": "2026-10-23T14:48:35Z",
      "abstract_url": "http://arxiv.org/abs/2610.32729v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32729v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences",
      "authors": [
        "Tobias Schneider",
        "Lorenz Sparrenberg",
        "Priya Tomar",
        "Shahzeb Qamar",
        "Tobias Deußer",
        "Rafet Sifa"
      ],
      "abstract": "Raw player input sequences in video games contain fine-grained behavior, but are high-dimensional and noisy, which makes large-scale analytics challenging. To address this, we introduce PlayerEmbed, a framework that learns compact and discriminative player embeddings using deep metric learning. Our approach uses a hybrid CNN-LSTM encoder trained on raw input sequences from 1,000 players. We evaluate the model on held-out sequences from these players and on sequences from 1,000 players entirely unseen during training to assess generalization to unseen identities. For player identification, a nearest-centroid classifier built from reference embeddings achieves strong performance, reaching balanced accuracies of 0.950 for players seen during training and 0.879 for players unseen during training. Systematically increasing the number of training players substantially improves generalization to unseen players, indicating a transition from simple fingerprinting to learning transferable behavioral patterns.",
      "published": "2026-10-23T14:48:02Z",
      "abstract_url": "http://arxiv.org/abs/2610.32727v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32727v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Do Language Models Trust Their Own Justifications? A Study on Functional Consistency",
      "authors": [
        "Alisson Rosa Pereira",
        "Bruno Iochins Grisci"
      ],
      "abstract": "Large Language Models (LLMs) have been widely adopted in text classification tasks, where they not only output class predictions but also generate explanations that highlight the tokens deemed most relevant for reaching the predicted label. Yet it remains unclear whether these highlighted elements are behaviorally consistent with the model's own predictions under targeted interventions, which we refer to as functional auto-consistency. While much of the literature evaluates the textual plausibility of such explanations, few studies assess their functional consistency with the model's actual behavior. In this work, we propose an experimental framework based on the principle of auto-consistency: if a model identifies certain tokens as decisive, then isolating, removing, or semantically inverting them should produce systematic and interpretable changes in its predictions. We operationalize this evaluation through sufficiency, comprehensiveness, and counterfactuality metrics, and conduct experiments on IMDB and Steam reviews across both closed-source (GPT-4o-mini) and open-source LLMs (Gemma3, Granite8B, DeepSeek). Results show that GPT-4o-mini follows the expected progression across all metrics, Gemma3 and Granite8B maintain coherence under sufficiency but lose consistency under more demanding interventions, while DeepSeek variants display structural deviations, either failing to preserve sufficiency or overreacting under comprehensiveness and counterfactuality. These findings show that explanation reliability varies across LLM families and scales, with smaller models displaying contradictions and larger ones exhibiting over-sensitivity. By combining sufficiency, comprehensiveness, and counterfactuality, our approach provides a systematic methodology for assessing the functional consistency of LLM self-explanations.",
      "published": "2026-10-23T14:08:32Z",
      "abstract_url": "http://arxiv.org/abs/2610.32665v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32665v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "LGMT: A Tool for Testing the Logical Reasoning Reliability of LLMs",
      "authors": [
        "Zenghui Zhou",
        "Xiaoke Fang",
        "Zheng Zheng"
      ],
      "abstract": "Large Language Models (LLMs) increasingly serve as reasoning components in modern software systems, yet a model that answers a logical query correctly can contradict itself when the same problem is restated in an equivalent form. Static benchmarks evaluate queries in isolation and therefore fail to expose such inconsistencies. To address this limitation, we present LGMT, an open-source tool that detects reasoning inconsistencies through metamorphic relations grounded in first-order logic. Given a source test case in premise-conclusion form, LGMT generates logically equivalent follow-up cases and checks whether the model's entailment predictions agree across the pair, without ground-truth labels. The tool provides 20 built-in metamorphic relations, supports custom ones verified by the Z3 theorem prover, and runs from a CLI or a web studio, including an offline mock mode requiring no API key. Across six widely used LLMs on a bundled benchmark of 3,091 metamorphic groups, LGMT reveals inconsistency rates ranging from 20.84% to 29.44%. LGMT is publicly available on GitHub at https://github.com/MaxwelsDonc/lgmt-tool, with a demonstration video at https://youtu.be/gou4opptu1U.",
      "published": "2026-10-23T13:41:43Z",
      "abstract_url": "http://arxiv.org/abs/2610.32627v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32627v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling",
      "authors": [
        "Zijian Jia",
        "Peixuan Zhang",
        "Genying Li",
        "Di Wu",
        "Yujie Fu",
        "Si Li",
        "Boxin Shi",
        "Shuchen Weng"
      ],
      "abstract": "Continuous diffusion models have emerged as a promising non-autoregressive paradigm for text generation over continuous embedding spaces. Along the continuous trajectory, the residual token identity directly shapes the denoising objective. However, the geometric trajectory misaligns with temporal progression, triggering severe temporal compression. In this paper, we show that embedding norm decay causes Euclidean distance to yield spurious overtaking, indicating that token competition must be evaluated directionally. In this space, a candidate overtakes the source token only when the effective noise magnitude surpasses their semantic separation. We formalize these pairwise dynamics as the Competition Threshold Theory (CTT). Crucially, high-dimensional overtaking requires extreme noise levels, causing standard trajectories to concentrate identity decay into a narrow window near the noisy endpoint. To resolve this, we propose Semantic Retention Scheduling (SRS) to align time allocation with the progression of identity decay, and demonstrate that existing token identity metrics are fundamentally statistical aggregations of these pairwise crossings. Building on these theoretical insights, we introduce LLaCO, a generative framework powered by competition-aware semantic scheduling. LLaCO adapts a pretrained backbone to continuous representations via an embedding directional adapter and timestep-conditioned modulation. To operate within the directional space, LLaCO employs a spherical noising process formulated as a geodesic interpolation toward a mask-biased endpoint, eliminating norm decay while preserving accurate token competition. Guided by SRS, LLaCO reallocates the denoising trajectory proportionally to the logarithmic rank decay. Instantiated as an 8B model, LLaCO outperforms previous continuous diffusion baselines and rivals discrete diffusion models of comparable scale.",
      "published": "2026-10-23T13:28:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.32604v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32604v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations",
      "authors": [
        "Sarah Sulollari",
        "Vanja Karan",
        "Benjamin Roth"
      ],
      "abstract": "This paper examines speech characteristics of the Austrian National Council members (CMs). We introduce analysis methodology based on a fine-tuned BERT-Model which assigns to every speech a probability distribution over Austrian parties. These are then aggregated at the speaker and party level. We perform an extensive evaluation to ensure the quality of the generated distributions prior to further analysis. Instead of considering mis-assignments of predicted parties as a performance issue, we leverage them to derive insights about the CMs and political developments. Our findings show that parties that govern together become more similar in their speech characteristics, while, analogously, parties that are in opposition move further away from governing parties. Moreover, CMs who switch party usually adapt their speech characteristics only after the switch occurs. Finally, we publicly release a data set containing party affiliation predictions for the speeches of the Austrian National Council from 1996 to 2018.",
      "published": "2026-10-23T11:38:35Z",
      "abstract_url": "http://arxiv.org/abs/2610.32515v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32515v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY",
        "cs.LG"
      ]
    },
    {
      "title": "Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)",
      "authors": [
        "Khaled Waleed Dawoud",
        "Muhammad Haris Khan"
      ],
      "abstract": "Deploying Vision-Language Models for video anomaly detection confronts a fundamental constraint. Their semantic reasoning demands computational budgets incompatible with the scale of real-world video streams. We present Signal-Aware Adaptive Fusion (SAAF), a framework that achieves efficient zero-shot anomaly detection through three key innovations. First, we introduce a dual-signal adaptive sampler operating in semantic embedding space: a delta signal captures sudden events via frame-to-frame cosine distance, while a drift signal detects static and gradual anomalies through deviation from rolling temporal context. Their OR-gated combination with adaptive EMA-based thresholding provides comprehensive anomaly coverage without training data. Second, we propose a threshold-gated cascaded architecture where lightweight frame-level scoring precedes expensive video-level temporal analysis, invoking it only when confidence warrants additional context. Third, we design a signal-aware weighted fusion mechanism that dynamically adapts score combination based on which signal triggered frame selection. Across four benchmarks, SAAF delivers competitive accuracy at substantially lower cost. Among training-free zero-shot methods, SAAF sets a new state of the art on XD-Violence (92.07%), MSAD (86.70%), and ShanghaiTech (80.54%) AUC-ROC, with XD-Violence and MSAD also surpassing supervised baselines. SAAF sustains 44.95, 48.61, and 41.06 FPS on XD-Violence, UCF-Crime, and MSAD, a 2.43$\\times$, 2.63$\\times$, and 2.22$\\times$ throughput improvement over the prior best zero-shot method. Code: https://github.com/kdawoud91/SAAF",
      "published": "2026-10-23T11:02:16Z",
      "abstract_url": "http://arxiv.org/abs/2610.32486v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32486v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models",
      "authors": [
        "Siyu Wu",
        "Zeyu Liang",
        "Linghao Wang",
        "Qiyun Zheng",
        "Yi Yao",
        "Ling Lo",
        "Hongxia Xie"
      ],
      "abstract": "Multimodal large language models (MLLMs) have substantially advanced Multimodal Emotion Recognition (MER) by enabling fine-grained emotion understanding through reasoning over multimodal evidence. However, they still exhibit systematic confusion between semantically adjacent emotions with highly overlapping multimodal cues. Existing approaches typically address these errors through global parameter adaptation, making confusion correction costly and inflexible. We instead observe that emotion confusion is highly localized and therefore better addressed through reusable pair-specific expertise. Building on this insight, we propose Confusion-Aware Skill Evolution (CASE), the first agent-skill framework for MER. CASE maintains a sparse, self-evolving library of pair-specific EmotionSkills that externalize affective expertise from model parameters, enabling the learned expertise to be continuously refined and readily transferred across diverse frozen MLLM backbones. During inference, CASE dynamically retrieves the appropriate EmotionSkill to reinforce boundary-aware reasoning while keeping the MLLM backbone frozen. Extensive experiments on CA-MER, DFEW, and EMER demonstrate the effectiveness and transferability of CASE across both general-purpose and emotion-specialized MLLMs. Notably, CASE improves the average accuracy of Qwen3-VL-8B on the CA-MER benchmark from 46.80% to 59.13%, while achieving consistent gains across MLLMs.",
      "published": "2026-10-23T10:50:03Z",
      "abstract_url": "http://arxiv.org/abs/2610.32481v1",
      "pdf_url": "https://arxiv.org/pdf/2610.32481v1",
      "categories": [
        "cs.AI"
      ]
    }
  ]
};
