const PAPERS_DATA = {
  "last_updated": "2026-07-02 04:10:03 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Language-Critique Imitation Learning from Suboptimal Demonstrations",
      "authors": [
        "Chih-Han Yang",
        "Dai-Jie Wu",
        "Yun-Ping Huang",
        "Ping-Chun Hsieh",
        "Kenneth Marino",
        "Shao-Hua Sun"
      ],
      "abstract": "Prior work on imitation learning from suboptimal demonstrations typically relies on compressed supervision signals such as confidence estimates, discriminator scores, or importance weights. These scalar signals are inherently limited, as they cannot explicitly express intermediate reasoning about task progress, failure modes, or corrective actions. We propose a language-critique framework for imitation learning from suboptimal demonstrations that instead leverages natural language as a structured supervision signal, avoiding the collapse of expressive feedback into scalars. Our method first constructs language labels from demonstrations that explicitly describe current progress, identify suboptimal behaviors, and provide fine-grained corrective guidance. We then introduce a language-critique loss that directly trains policies using these structured signals without reducing them to scalars, and instantiate it for both behavior cloning and diffusion policies, yielding LC-BC and LC-DP. We further provide a theoretical result showing that the proposed objective upper-bounds the expert performance gap under standard assumptions. Empirically, we evaluate on diverse continuous control tasks spanning navigation, manipulation, and gameplay, where our methods consistently outperform strong imitation learning and offline reinforcement learning baselines. These results demonstrate that language can serve as a powerful and structured form of supervision for learning robust policies from suboptimal data.",
      "published": "2026-07-01T17:57:22Z",
      "abstract_url": "http://arxiv.org/abs/2607.01225v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01225v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Theoria: Rewrite-Acceptability Verification over Informal Reasoning States",
      "authors": [
        "Ben Slivinski",
        "Michael Saldivar"
      ],
      "abstract": "When should an AI system's answer be trusted? Formal proof assistants offer certainty but cannot reach most of the problem distribution; scalar LLM judges offer coverage but produce opaque scores that cannot be audited after the fact and are subject to the same coherence issues as any LLM. We present Theoria, a verification architecture that closes this gap. A candidate solution is rewritten into a sequence of typed state transitions, each licensed by an explicit justification, whether that be a citation, computation, or problem-given fact, and every transition is independently auditable. The foundational invariant is completeness of change: every difference between consecutive proof states must be accounted for, so hidden premises surface as unlicensed mutations rather than passing silently. On HLE-Verified Gold (185 text-only expert problems), Theoria certifies 105 at 91.4% strict precision (Wilson 95% CI [84.5%, 95.4%]). Every certification produces a human readable proof trace in which each step can be independently challenged. Holistic LLM judges achieve comparable precision at matched coverage but fail on different problems (Jaccard 0.14-0.36), making the approaches complementary. On 95 adversarial poisoned proofs across 15 domains, structured judges catch 94.7% versus 83.2% for holistic judging (p= 0.0017). The overall 11.5 pp gap concentrates in hidden premises (90.6% vs. 62.5%, a 28 pp difference) and fabricated citations (100% vs. 90%), the error classes where the formal analysis predicts an advantage; performance is identical on arithmetic and theorem-misapplication errors, where no advantage is predicted. On GPQA Diamond (n= 65), certified precision is 97.1% (Wilson CI [85.1%, 99.5%]).",
      "published": "2026-07-01T17:56:42Z",
      "abstract_url": "http://arxiv.org/abs/2607.01223v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01223v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.LO",
        "cs.SE"
      ]
    },
    {
      "title": "The State-Prediction Separation Hypothesis",
      "authors": [
        "Giovanni Monea",
        "Nathan Godey",
        "Kianté Brantley",
        "Yoav Artzi"
      ],
      "abstract": "Transformers use the same forward computation stream to both predict the next token and store useful state for future token predictions. We formulate the \\emph{state-prediction separation hypothesis}: disentangling the two roles yields better language modeling performance. We design a Transformer variant that uses two computation streams to separate the two functions, and conduct pretraining experiments across various scales. Our experiments show that state-prediction separation consistently offers better data and compute efficiencies, improving validation loss and outperforming standard Transformers by 2--3 percentage points on average on downstream tasks. We also conduct extensive empirical analysis that rules out potential confounders and demonstrates the fundamental difference in the gradients our design entails.",
      "published": "2026-07-01T17:55:09Z",
      "abstract_url": "http://arxiv.org/abs/2607.01218v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01218v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Distill to Detect: Exposing Stealth Biases in LLMs through Cartridge Distillation",
      "authors": [
        "Shayan Talaei",
        "Abhinav Chinta",
        "Devvrit Khatri",
        "Amin Karbasi",
        "Azalia Mirhoseini",
        "Amin Saberi"
      ],
      "abstract": "Language models deployed in high-stakes roles can potentially favor certain entities, brands, or viewpoints, steering user decisions at scale. Such preferential biases can be introduced by any actor in the model's supply chain and are most dangerous when the model reveals its preference only on the relevant topic while behaving identically to its unmodified base on all other inputs. Recent work has shown that these biases can transfer through context distillation on semantically unrelated data, with the signal residing entirely in the soft logit distribution and remaining invisible to text-based inspection. However, the defender faces a fundamental asymmetry: without knowing the bias topic, no detection method can reliably surface a stealth preferential bias, regardless of whether it examines generated text, internal representations, or model weights. Here we introduce Distill to Detect (D2D), a method that surfaces hidden biases by distilling the distributional shift between a suspected model and its base into a cartridge (a KV-cache prefix adapter), concentrating the dominant divergence and amplifying the bias signal into generated text. We show that D2D successfully amplifies the hidden biases of stealth models to the extent that they can be reliably detected across multiple bias types. We also propose a theoretical framework that explains the efficacy of D2D through the lens of Fisher-weighted projection of the logit distribution shift, supported by empirical observations. By turning the capacity bottleneck of prefix-tuning adapters into a detection tool, D2D provides a practical building block for auditing hidden behaviors in deployed language models.",
      "published": "2026-07-01T17:46:33Z",
      "abstract_url": "http://arxiv.org/abs/2607.01208v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01208v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "GPU-Parallel Linearization Error Bounds for Real-Time Robust Optimal Control of Nonlinear and Neural Network Dynamics",
      "authors": [
        "Jeffrey Fang",
        "Keyi Shen",
        "Anutam Srinivasan",
        "Glen Chou"
      ],
      "abstract": "This paper studies real-time robust optimal control for uncertain nonlinear systems, where linear time-varying (LTV) approximations make planning tractable but require sound linearization error bounds (LEBs) to guarantee robust constraint satisfaction. We develop tight, differentiable, GPU-parallel LEBs for LTV approximations of nonlinear and neural network (NN) dynamics. For analytic dynamics, we introduce path-based Hessian bounds that are tighter than standard interval methods. For NN dynamics, we derive certified LEBs using NN verifier-generated affine relaxations and local Jacobian corrections. We adapt a GPU-parallel system-level synthesis LTV-based robust control solver to be compatible with these LEBs by extending it to handle right-invertible disturbance matrices and non-zero-centered disturbance sets for tight zonotopic uncertainty propagation. Our method, GPUSLS-LEO, enables online optimization of robust feedback policies that account for linearization error, producing tight, formally verified reachable tubes. On complex nonlinear and NN dynamics up to 168 state dimensions, our method can compute robust control policies on the GPU at rates up to 67 Hz, reducing solve times and conservativeness relative to baselines while preserving formal guarantees and real-time performance.",
      "published": "2026-07-01T17:42:37Z",
      "abstract_url": "http://arxiv.org/abs/2607.01203v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01203v1",
      "categories": [
        "eess.SY",
        "cs.AI",
        "cs.LG",
        "cs.RO",
        "math.OC"
      ]
    },
    {
      "title": "Right in the Right Way: LM Training with Verifiable Rewards and Human Demonstrations",
      "authors": [
        "Mehul Damani",
        "Isha Puri",
        "Idan Shenfeld",
        "Jacob Andreas"
      ],
      "abstract": "RL with verifiable rewards (RLVR) has emerged as a powerful paradigm for training LMs on tasks with well-defined success metrics, such as code generation and mathematical reasoning. However, current RLVR methods optimize only what can be objectively scored, often neglecting subjective, non-verifiable aspects of human-like outputs, such as style and structure. This limitation leads to well-documented failure modes such as diversity collapse, unnatural-sounding responses, and reward hacking. We propose an adversarial generator-discriminator framework that augments verifiable rewards with a learned signal from human demonstrations. A generator model is trained using RL to maximize both task accuracy and an adversarial reward derived from a discriminator. The discriminator, trained alongside the generator policy, learns to distinguish human-written outputs from model-generated ones. The discriminator serves as a learned proxy for the human output distribution, providing feedback on aspects of generation that are difficult to formalize as scalar rewards. Across diverse domains, including bug fixing and open-ended generation, our approach consistently improves non-verifiable properties while preserving the accuracy gains of RLVR. In bug fixing, our method produces solutions with significantly lower edit distance compared to RLVR baselines while matching end performance. In story generation, our method significantly improves win rate while producing stories that are diverse and more human-like. And in a simple reward hacking benchmark, our method nearly eliminates model misbehavior while maintaining high benchmark scores. Together, these results show that our approach bridges RL and SFT, offering a scalable path toward jointly optimizing the verifiable and non-verifiable properties of a task.",
      "published": "2026-07-01T17:13:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.01181v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01181v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Sequentially-Controlled Interactive Multi-Particle Flow-Maps for Online Feedback-Driven Search",
      "authors": [
        "Binglin Ji",
        "Anindya Sarkar",
        "Hengchang Lu",
        "Jens Sjölund",
        "Yevgeniy Vorobeychik"
      ],
      "abstract": "While generative models have enabled training-free reward alignment, current methods typically excel in local exploration within narrow regions of the underlying distribution. These approaches struggle when preferences are unknown a priori and only revealed through sequential feedback-a scenario demanding broad exploration to uncover high-utility regions. To address this, we propose Sequentially-Controlled Interactive Multi-Particle Flow-Maps (IMPFM), a framework for sample-efficient online feedback-driven search. IMPFM progressively transports a group of interactive particles toward the target distribution, maintaining the broad coverage essential for heterogeneous preference alignment. IMPFM introduces a principled and efficient posterior sample sharing mechanism across particles powered by flow maps. By correcting individual particle drift with the collective posterior samples of the entire ensemble at each resampling step, the framework maximizes sample utility to enable global exploration while actively mitigating reward over-optimization, typical of standard control frameworks. Paired with a principled exploration-exploitation reweighting mechanism involving multi-particle interaction, this sequentially corrected multi-particle dynamics explicitly preserves structural diversity and overcomes the weight degeneracy inherent to standard SMC samplers. Crucially, we prove that the resulting sampling framework yields a multi-particle interaction-aware Feynman-Kac corrector that progressively steers the multi-particle system toward a KL-tilted target distribution, facilitating global exploration and preventing mode collapse. Extensive empirical evaluations and rigorous ablations across diverse search and alignment tasks confirm the efficacy of IMPFM over existing baselines.",
      "published": "2026-07-01T16:27:17Z",
      "abstract_url": "http://arxiv.org/abs/2607.01144v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01144v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CE"
      ]
    },
    {
      "title": "Skills Are Not Islands: Measuring Dependency and Risk in Agent Skill Supply Chains",
      "authors": [
        "Changguo Jia",
        "Tianqi Zhao",
        "Runzhi He",
        "Minghui Zhou"
      ],
      "abstract": "Agent skills package reusable operational knowledge for Large Language Model (LLM) agents, yet as they grow in scope, they become dependency-bearing artifacts whose identities, versions, and provenance remain implicit. This opacity already causes duplicated dependencies and inconsistent installations, exposing a gap that dependency management has yet to close. We introduce Agent Skill Supply Chains (ASSCs) to characterize mixed skill-package-service dependency graphs and help close this gap. Borrowing from Software Bill of Materials (SBOMs), we design SkillDepAnalyzer to capture natural-language dependency evidence and model skills as dependency-bearing artifacts. On the SKILL-DEP benchmark, SkillDepAnalyzer recovers skill metadata and dependency graphs accurately and comprehensively, substantially outperforming an LLM-based baseline and package-centric SBOM tools. Applying SkillDepAnalyzer to over 1.43 million skills, we obtain ASSCs and explore their structural diversity and security signals. We find four structural patterns: skill metadata is activation-ready but governance-poor; dependency graphs span skill, package, and service dependencies with concentrated reuse; recursive skill reuse expands dependency graphs and creates hidden package inventory; and skill dependency clusters form around related workflows. We also find that inspecting a skill alone misses security-relevant signals hiding in its dependencies. By analyzing ASSCs, we identify and report known malicious skills persisting in ASSCs to their developers. Based on these findings, we recommend typed dependency manifests, first-class dependency-cluster management, risk-warning audit commands for skill infrastructure maintainers, and lockfile-like records for skill developers.",
      "published": "2026-07-01T16:21:49Z",
      "abstract_url": "http://arxiv.org/abs/2607.01136v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01136v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Autonomous Scientific Discovery via Iterative Meta-Reflection",
      "authors": [
        "Bingchen Zhao",
        "Sara Beery",
        "Oisin Mac Aodha"
      ],
      "abstract": "Autonomous scientific discovery systems offer the potential to accelerate research by automating the process of hypothesis generation and validation. However, current systems operate within constrained search spaces or require predefined research questions, limiting their capacity for true open-ended inquiry. Furthermore, while they generate hypotheses iteratively, they largely lack the ability to explicitly synthesize their own accumulated findings to uncover complex, interconnected phenomena. We introduce DiscoPER, an autonomous large language model-powered framework that conducts open-ended research by dynamically generating and executing code to explore datasets without pre-specified research objectives. To ensure rigorous scientific validity, every proposed discovery must pass statistical testing. To overcome the limitations of isolated search, our framework introduces a second-order reasoning mechanism that periodically analyzes its own accumulated discoveries. By treating prior discoveries as empirical data, DiscoPER identifies structural patterns, confounds, and epistemic gaps, actively redirecting hypothesis exploration toward uncharted regions of the search space. The search space is further expanded by incorporating tool use, enabling the system to explore hypotheses beyond structured metadata by seamlessly processing and extracting useful information from multimodal sources like images. Evaluated on iNatDisco, a new multimodal ecological knowledge benchmark with pattern-level ground truth obtained from peer-reviewed literature, DiscoPER recovers 8 of 9 known patterns with a 72.7% hypothesis support rate, outperforming both classical causal discovery and LLM-guided baselines. Ablations show that DiscoPER scales with more data, and confirms the benefits of second-order meta-reflection.",
      "published": "2026-07-01T16:16:07Z",
      "abstract_url": "http://arxiv.org/abs/2607.01131v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01131v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Muon as a Residual Connection",
      "authors": [
        "Hao Huang"
      ],
      "abstract": "Muon has recently emerged as one of the most effective optimizers for training large neural networks, yet its empirical success has been explained from several different perspectives. In this paper, we propose a simple mechanistic interpretation: Muon can be understood as an implicit residual connection during training. Specifically, orthogonalizing the update can sacrifice some immediate gradient fidelity while improving representation preservation for downstream layers. We study this trade-off in controlled linear optimization settings, where Muon can learn representations that are slower to fit a local target but easier for downstream layers to exploit. Our results suggest a conceptual explanation for Muon and a design perspective for optimizers that balance local descent with downstream usability.",
      "published": "2026-07-01T16:12:24Z",
      "abstract_url": "http://arxiv.org/abs/2607.01124v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01124v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Towards Developing a Multimodal Chat Assistant for University Stakeholders: RAG-based Approach",
      "authors": [
        "Md Abu Hanif Shaikh",
        "Abdullah Al Shafi"
      ],
      "abstract": "University stakeholders often face difficulties in accessing timely and reliable information, especially in developing countries, where there are very few intelligent support systems. Existing rule-based chatbots are unable to handle complex, domain-specific queries and are not well-equipped to adapt to evolving institutional policies. As a fill-in-the-gap solution, we present the multimodal university chatbot with retrieval-augmented generation. The system combines the large language model with semantic retrieval to produce context-based responses from institution-centric resources, such as the university handbook. The system accepts text and image queries through the vision-language model and applies quantized inference for rapid deployment on constrained hardware. A scalable backend built with FastAPI, adjoined with a responsive frontend developed with Next.js, ensures real-time usability. Our multimodal evaluation demonstrates that the system maintains strong satisfaction scores across both text and image queries, despite increased response time for visual inputs. Furthermore, quantitative evaluation shows that hallucination is reduced from 31.7% to 6.6% in our proposed RAG-based system, confirming the effectiveness of retrieval grounding.",
      "published": "2026-07-01T16:03:03Z",
      "abstract_url": "http://arxiv.org/abs/2607.01115v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01115v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement",
      "authors": [
        "Haoran Hao",
        "Shahram Najam Syed",
        "Jeffrey Ichnowski",
        "Jeff Schneider"
      ],
      "abstract": "Robot policies inevitably encounter failures when deployed in real environments. Naive retries often repeat the same mistakes, while many existing recovery methods rely on human intervention. In this paper, we propose Failure-Aware Retry (FAR), a framework that enables robots to learn from previous failures at test time, adapt their behavior accordingly, and eventually complete the task autonomously. FAR combines Failure-Contrastive Preference Adaptation, which constructs preference learning data from failures to steer the policy away from previously unsuccessful behaviors, with lightweight action perturbations during retries to encourage local exploration. We further incorporate successful recovery trajectories into a training loop for continual policy improvement. Experiments in both simulation and real-world manipulation tasks show that FAR substantially improves success rates and robustness, with average gains of 17.6% over the standard diffusion policy in simulation and 11.7% in the real world. In addition, FAR significantly improves data efficiency under both reset and timestep budgets during continual policy improvement by exploiting informative failure cases.",
      "published": "2026-07-01T16:01:54Z",
      "abstract_url": "http://arxiv.org/abs/2607.01111v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01111v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "CausalMix: Data Mixture as Causal Inference for Language Model Training",
      "authors": [
        "Zinan Tang",
        "Yukun Zhang",
        "Shaomian Zheng",
        "Zhuoshi Pan",
        "Qizhi Pei",
        "Dingnan Jin",
        "Jun Zhou",
        "Yujun Wang",
        "Biqing Huang"
      ],
      "abstract": "In Large Language Model (LLM) training, data mixing plays a pivotal role in determining model performance. Recent methods optimize mixture weights via proxy models, but they rely on the assumption of static data distributions. As a result, when the underlying data pool shifts, these methods require costly retraining from scratch. This limitation restricts their ability to scale seamlessly from small settings to larger data pools and model sizes. In this paper, we propose CausalMix to address this limitation by casting data mixture optimization as a causal inference problem. We formulate the statistical features of the data pool as covariates and the domain mixture as the treatment. After fitting a causal model on 512 runs of Qwen2.5-0.5B to estimate the Conditional Average Treatment Effect (CATE), we extrapolate the optimal mixture for an 800K data pool and apply it to train a 7B model. Furthermore, we successfully generalize the framework to long chain-of-thought data on Qwen3-4B-Base. By leveraging causal modeling to isolate confounding biases, CausalMix dynamically infers state-dependent optimal data mixtures. Extensive experiments show that the mixture guided by CausalMix consistently improves performance across multiple downstream tasks, outperforming RegMix and other baselines. In addition, we use the CATE Interpreter to provide visual analysis of the learned mixing strategy. Overall, CausalMix offers a causal and interpretable framework for optimizing LLM data mixtures.",
      "published": "2026-07-01T15:56:11Z",
      "abstract_url": "http://arxiv.org/abs/2607.01104v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01104v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Can Agents Generalize to the Open World? Unveiling the Fragility of Static Training in Tool Use",
      "authors": [
        "Song-Lin Lv",
        "Weiming Wu",
        "Rui Zhu",
        "Zi-Jian Cheng",
        "Lan-Zhe Guo"
      ],
      "abstract": "While Large Language Model (LLM) agents demonstrate proficiency in static benchmarks, their deployment in real-world scenarios is hindered by the dynamic nature of user queries, tool sets, and interaction dynamics. To address this generalization gap, we formalize OpenAgent (Tool-Use Agent in Open-World), a problem setting characterized by distributional shifts across query, action, observation, and domain dimensions. To systematically diagnose its impact, we construct a controlled sandbox environment where we define fine-grained environmental shifts across a four-tier hierarchy, Perception, Interaction, Reasoning, and Internalization, and conduct a comprehensive series of experiments. Our analysis yields a series of key insights, demonstrating that agents trained via both Supervised Fine-Tuning(SFT) and Reinforcement Learning suffer from varying degrees of performance degradation when confronting open environmental shifts. Building on these insights, we propose Perturbation-Augmented Fine-Tuning, a disturbance-based intervention strategy for SFT that lays the foundation for enhancing agent robustness and utility in realistic environments. Our code will be released at: https://github. com/LAMDA-NeSy/OpenAgent.",
      "published": "2026-07-01T15:40:25Z",
      "abstract_url": "http://arxiv.org/abs/2607.01084v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01084v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Staleness-Learning Rate Scaling Laws for Asynchronous RLHF",
      "authors": [
        "Jingwei Song",
        "Haofeng Xu",
        "Jie Xiao",
        "Chengke Bao",
        "Jingwei Shi",
        "Pengbin Feng",
        "Weixun Wang",
        "Yuhang Han",
        "Chuan Wu",
        "Linfeng Zhang",
        "Bill Shi"
      ],
      "abstract": "High-throughput RLHF systems often decouple rollout generation from policy optimization, leading to the use of stale rollouts during learner updates. In this work, we study the effect of such staleness in asynchronous GRPO. We make the behavior policy explicit in the GRPO surrogate objective and distinguish between the surrogate-gradient mapping used by the learner and the true total derivative of a distribution-dependent population objective. Under assumptions of local boundedness, distributional smoothness, and behavior-policy smoothness, we show that stale rollouts introduce a per-step surrogate-gradient bias of order O(S * eta), where S denotes the maximum rollout lag and eta denotes the learning rate. We further derive a conditional collapse-time scaling law: when within-cycle drift remains below a batch-level clipping radius, collapse is governed primarily by cumulative learner drift T * eta; when the stale-rollout constraint is active, stability instead depends explicitly on S * eta. This yields a two-constraint stability condition eta << min{R_batch / (S * G_upd), R_crit / (T * G_upd)}, explaining why the maximum stable learning rate may appear weakly dependent on staleness in the horizon-limited regime.",
      "published": "2026-07-01T15:40:12Z",
      "abstract_url": "http://arxiv.org/abs/2607.01083v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01083v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Agentic generation of verifiable rules for deterministic, self-expanding reaction classification",
      "authors": [
        "Daniel Armstrong",
        "Maarten Dobbelaere",
        "Valentas Olikauskas",
        "Helena Avila",
        "Octavian Susanu",
        "Jérôme Waser",
        "Philippe Schwaller"
      ],
      "abstract": "Computer-assisted synthesis planning breaks target molecules into accessible precursors using large libraries of reaction rules that assign each transformation a deterministic, interpretable label. But chemistry is long-tailed, making manual encoding intractable, and existing tools rely on fixed rulesets that cannot adapt to new chemistries. Here we present a fully automated pipeline in which a multi-agent framework of large language models (LLMs) classifies reactions and writes the rules themselves across 665,901 US patent reactions, generating each rule under a verification loop that tests it against the corpus. It expands a standard taxonomy from 68 to 14,073 classes without human curation. With a lightweight fingerprint classifier, it classifies 97.7\\% of unseen reactions, matching a leading proprietary classifier while resolving chemistry more finely and extending on demand to chemistry outside its training distribution. The result is a living reactivity database and a general route to turning generative models into reliable, self-expanding symbolic systems.",
      "published": "2026-07-01T15:24:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.01061v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01061v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Behavior-Adaptive Conversational Agents: Toward a Fluid Personality Framework",
      "authors": [
        "Hasibur Rahman",
        "Smit Desai"
      ],
      "abstract": "Large language model (LLM)-based conversational agents (CAs) are now ubiquitous, creating new opportunities for AI-mediated behavior change. Their capacity to project nuanced personalities and adopt diverse metaphorical roles raises a design question: how should an agent's persona and personality be calibrated to the moment? Recent evidence suggests that (i) moderate personality expression outperforms low or high extremes on trust, enjoyment, and intention to adopt in goal-oriented tasks, and (ii) context-appropriate metaphors outperform static one-note assistants on user experience and uptake. Yet most CAs still fix both persona and style, risking misalignment when dynamics, urgency, and formality vary, for example in medical information seeking, fitness coaching, and reflective learning. We propose a Fluid Personality Framework that jointly adapts (1) the agent's metaphorical persona, such as coach, tutor, librarian, or tool, and (2) its personality expression intensity, low, medium, or high, as a function of task context, user goals and traits, and situational urgency. We sketch the framework and its core design dimensions.",
      "published": "2026-07-01T15:03:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.01034v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01034v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads",
      "authors": [
        "Aryo Pradipta Gema",
        "Beatrice Alex",
        "Pasquale Minervini"
      ],
      "abstract": "In long-context use, large language models frequently synthesize answers from the meaning of a relevant context span rather than literally copy-pasting them. Identifying which attention heads perform this synthesis matters for interpreting long-context model behavior. Yet existing detectors miss these heads by construction: they reward heads whose attended token matches the generated token, a literal-copy criterion that captures where a head reads but not what it writes through its output-value (OV) circuit, the very mechanism that carries non-literal retrieval. We introduce Logit-Contribution Scoring (LOCOS), a write-aware detector that scores each head by the projection of its OV-circuit output onto the answer-token unembedding direction, contrasting needle and off-needle source positions in a single forward pass. Across three model families (Qwen3, Gemma-3, OLMo-3.1), mean-ablating the top LOCOS heads on the NoLiMa non-literal retrieval benchmark collapses ROUGE-L at lower head counts than prior attention-based detections; on Qwen3-8B, ablating 50 heads drives ROUGE-L from 0.401 to 0.000 while the strongest baseline still retains 0.292. The selected heads are retrieval-specific: parametric recall and arithmetic reasoning stay at baseline under the same ablation. On Qwen3-8B, the same ablation also drops MuSiQue from 0.55 to 0.08 and BABI-Long from 0.62 to 0.20, while a random-heads control stays within 0.05 of baseline.",
      "published": "2026-07-01T14:41:07Z",
      "abstract_url": "http://arxiv.org/abs/2607.01002v1",
      "pdf_url": "https://arxiv.org/pdf/2607.01002v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SWE-Doctor: Guiding Software Engineering Agents with Runtime Diagnosis from Multi-Faceted Bug Reproduction Tests",
      "authors": [
        "Yaoqi Guo",
        "Yang Liu",
        "Jie M. Zhang",
        "Yun Ma",
        "Yiling Lou",
        "Zhenpeng Chen"
      ],
      "abstract": "Large language model (LLM)-based software engineering agents are increasingly developed to resolve software issues by generating patches from issue reports and code repositories. Bug reproduction tests (BRTs) are an important building block for such agents and have been shown useful for patch validation. However, it remains unclear whether BRTs can also help the more central stage of patch generation. We first conduct a preliminary study and find that directly using advanced BRT generators to guide patch generation is not beneficial: fail-to-fail BRTs can mislead agents, while even fail-to-pass BRTs bring limited or negative gains. Our analysis reveals two reasons: fail-to-pass BRTs may cover only one manifestation of the reported issue, leading to partial patches, whereas fail-to-fail BRTs are unreliable as direct patch-generation targets. Motivated by these insights, we propose SWE-Doctor, a software issue resolution agent that guides patch generation with runtime diagnoses derived from multi-faceted BRT executions. SWE-Doctor first generates multi-faceted BRTs for different behavioral requirements stated in the issue, then executes and debugs these BRTs to construct runtime-grounded diagnosis records, and finally uses the diagnoses together with localization information inferred during BRT generation to guide patch generation and reduce partial patches. We evaluate SWE-Doctor on Python bug-fixing issues from the widely adopted SWE-bench Verified and SWE-bench Pro across five LLM backends. SWE-Doctor consistently outperforms existing agents across all 10 LLM-benchmark combinations, achieving average resolution rates of 75.7% on SWE-bench Verified and 59.4% on SWE-bench Pro. In particular, on the more challenging SWE-bench Pro, SWE-Doctor improves the average resolution rate by 8.0-8.9 percentage points over the baseline agents.",
      "published": "2026-07-01T14:27:12Z",
      "abstract_url": "http://arxiv.org/abs/2607.00990v1",
      "pdf_url": "https://arxiv.org/pdf/2607.00990v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "SenseWalk: Agent-Based Semantic Trajectory Simulation Powered by Large Language Models in Zoned Environments",
      "authors": [
        "Ziyue Lin",
        "Xinhang Xie",
        "Kangyi Wang",
        "Siming Chen"
      ],
      "abstract": "Semantic trajectory analysis has recently emerged as an approach for modeling human movement by capturing implicit patterns and behaviors through semantic information (e.g., visitors' profiles and goals) beyond raw spatial paths to better understand why people move in certain ways. However, analyzing semantic trajectories in real-world scenarios remains challenging, as collecting high-quality data is costly and often lacks rich semantic information. Meanwhile, existing simulation tools require substantial technical expertise, which makes them difficult for practitioners to adopt. To address these limitations, the paper proposes ${SenseWalk}$, an interactive system that supports simulating semantic trajectories by LLM-powered agents. We develop a simulation workflow that combines LLMs and the social force model to balance physical plausibility and semantic coherence. A user-friendly interface is designed to facilitate users in customizing the simulation configuration and analyzing simulation outputs. We also conduct a quantitative experiment to evaluate the effectiveness of our simulation workflow, and a user study (n=12) to assess the usefulness and efficiency of our system.",
      "published": "2026-07-01T14:25:47Z",
      "abstract_url": "http://arxiv.org/abs/2607.00989v1",
      "pdf_url": "https://arxiv.org/pdf/2607.00989v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    }
  ]
};
