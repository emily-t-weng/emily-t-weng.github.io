const PAPERS_DATA = {
  "last_updated": "2026-06-13 04:37:16 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Mana: Dexterous Manipulation of Articulated Tools",
      "authors": [
        "Zhao-Heng Yin",
        "Guanya Shi",
        "Pieter Abbeel",
        "C. Karen Liu"
      ],
      "abstract": "Articulated tool manipulation remains a major challenge in dexterous robotics due to the need to coordinate internal degrees of freedom and contact-rich interactions. While prior work has largely focused on rigid objects, articulated tool use remains underexplored because of its physical complexity and the difficulty of learning functional grasping and manipulation policies. We present Mana (Manipulation Animator), a general sim-to-real framework that reinterprets dexterous manipulation as an animation problem. Inspired by computer animation, Mana employs a coarse-to-fine pipeline that transforms procedurally-generated grasp keyframes into manipulation trajectories through motion planning and reinforcement learning. The data generation process is largely automatic, requiring only a few mouse clicks to specify functional affordances (<1 minute per tool). Across four articulated tools spanning different scales and joint types, Mana achieves zero-shot sim-to-real transfer for both grasping and in-hand manipulation, demonstrating a scalable approach to dexterous articulated tool use.",
      "published": "2026-06-11T17:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2606.13677v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13677v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Automated reproducibility assessments in the social and behavioral sciences using large language models",
      "authors": [
        "Tobias Holtdirk",
        "Pietro Marcolongo",
        "Anna Steinberg Schulten",
        "Felix Henninger",
        "Stefan Rose",
        "Sarah Ball",
        "Bolei Ma",
        "Frauke Kreuter",
        "Markus Weinmann",
        "Stefan Feuerriegel"
      ],
      "abstract": "Reproducibility in the social and behavioral sciences is typically evaluated by independent researchers who reanalyze the original data to assess whether the published findings can be recovered. However, such approaches are resource-intensive and difficult to scale. Here, we show that large language models (LLMs) can automate reproducibility assessments. Using N=76 published studies with predefined claims from the behavioral and social sciences, we compare LLM-generated analysis with the original findings and human reanalysis. For 7 studies, the LLM could not produce a viable effect size estimate. For the remaining studies, our LLM pipeline recovered the original effect sizes in 41% of studies using a +/-0.05 tolerance in Cohen's d. Further, our LLM pipeline reached the same qualitative conclusion as the original study in 96% of cases, where conclusions indicate whether the reanalysis supports the original claim. For comparison, human reanalysts recovered the original effect sizes in 34% of studies and reached the same qualitative conclusion in 74% of cases. Together, these results show that LLMs can serve as a scalable tool for automated reproducibility assessment and provide a foundation for systematic auditing of empirical results in the social and behavioral sciences.",
      "published": "2026-06-11T17:58:36Z",
      "abstract_url": "http://arxiv.org/abs/2606.13670v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13670v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "EurekAgent: Agent Environment Engineering is All You Need For Autonomous Scientific Discovery",
      "authors": [
        "Amy Xin",
        "Jiening Siow",
        "Junjie Wang",
        "Zijun Yao",
        "Fanjin Zhang",
        "Jian Song",
        "Lei Hou",
        "Juanzi Li"
      ],
      "abstract": "LLM-based agents have shown increasing potential in automating scientific discovery. Given an optimizable metric and an execution environment, they can propose, validate, and iterate scientific solutions, and have produced results that outperform human-designed approaches. As model capabilities continue to improve, we argue that the bottleneck for autonomous scientific discovery is shifting from prescribing agent workflows to designing agent environments: the resources, constraints, and interfaces that shape agent behavior. We frame this as environment engineering: building environments that amplify productive behaviors, such as open-ended exploration, systematic artifact management, and inter-agent collaboration, while suppressing harmful behaviors, such as reward hacking and high-friction human oversight. We present EurekAgent, an environment-engineered agent system for metric-driven autonomous scientific discovery. EurekAgent engineers the environment along four dimensions: permissions engineering for bounded agent execution and isolated evaluation; artifact engineering for filesystem and Git-based collaboration; budget engineering for budget-aware exploration; and human-in-the-loop engineering for easy human supervision and intervention. EurekAgent sets new state-of-the-art results on multiple mathematics, kernel engineering, and machine learning tasks, including new state-of-the-art 26-circle packing results discovered with less than $11 in total API cost. We open-source our code and results, and call for environment engineering as a core research direction for developing reliable autonomous research agents.",
      "published": "2026-06-11T17:56:35Z",
      "abstract_url": "http://arxiv.org/abs/2606.13662v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13662v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation",
      "authors": [
        "Marek Šuppa",
        "Andrej Ridzik",
        "Daniel Hládek",
        "Natália Kňažeková",
        "Viktória Ondrejová"
      ],
      "abstract": "We introduce SkMTEB, the first comprehensive MTEB-style text embedding benchmark for Slovak, a low-resource West Slavic language, comprising 31 datasets across 7 task types -- nearly 4$\\times$ the depth of existing multilingual benchmark coverage for Slovak. Our evaluation of 31 embedding models reveals that large instruction-tuned multilingual models achieve the strongest performance, while existing Slovak-specific models trained for NLU tasks transfer poorly to embedding tasks. To address the need for efficient, locally-deployable Slovak embeddings, we develop \\texttt{e5-sk-small} (45M parameters) and \\texttt{e5-sk-large} (365M) by applying vocabulary trimming and fine-tuning to Multilingual E5 models. Despite size reductions of up to 62\\%, our open-source models achieve competitive performance with proprietary APIs while remaining locally deployable for semantic search and retrieval-augmented generation (RAG). We release the benchmark, models, datasets, and code openly, hoping our approach offers a replicable path for other under-resourced languages.",
      "published": "2026-06-11T17:50:06Z",
      "abstract_url": "http://arxiv.org/abs/2606.13647v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13647v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Valid Inference with Synthetic Data via Task Exchangeability",
      "authors": [
        "Lezhi Tan",
        "Tijana Zrnic"
      ],
      "abstract": "There is a proliferation of work arguing for the use of synthetic data in scientific research. For example, social scientists are arguing for the use of LLM-generated \"silicon samples\" in pilot studies; AI evaluations increasingly rely on \"LLM-as-a-judge\" outputs; and proteomics research is accelerated by generative models that produce synthetic protein structures. These developments raise an intriguing possibility: synthetic data may help researchers ask more questions, run more studies, and accelerate discovery. But they also raise a fundamental concern: synthetic data can be biased, noisy, and misspecified. In this work, we propose statistical principles for using synthetic data in scientific research with provable validity guarantees. The key insight is a new technical condition that we call task exchangeability. Informally, this is a requirement that the researcher can identify historical tasks, for which real data is available, such that their current task of interest is exchangeable with the historical tasks in an appropriate mathematical sense. We develop methods for valid inference under task exchangeability, together with extensions that provide guarantees even beyond exchangeability. We demonstrate the framework on public opinion surveys with silicon samples and AI evaluation with autoraters.",
      "published": "2026-06-11T17:41:09Z",
      "abstract_url": "http://arxiv.org/abs/2606.13629v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13629v1",
      "categories": [
        "stat.ME",
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis for Adversarial Networks",
      "authors": [
        "Achraf Hsain",
        "Sultan Almuhammadi"
      ],
      "abstract": "Shielded reinforcement learning is typically presented as a runtime safety mechanism that compiles temporal-logic specifications into automata restricting an agent's actions. We argue this is the wrong product. The same automata-theoretic machinery -- specification compilation, product game construction, attractor computation, and winning-region extraction -- is better read as a design-time analytical instrument whose outputs are structural insights about a system rather than runtime constraints on a deployed agent. We instantiate this through a constrained two-player safety game for network defense. The two specifications are enforced asymmetrically: the defender specification defines the unsafe region of the game, whereas the attacker specification restricts the adversary's legal actions during attractor computation. Solving the game yields a defensibility verdict -- a formal certificate that a topology-specification pair is or is not defensible -- with the associated winning region and shield. Beyond the binary verdict, we derive topology-level metrics from the attractor structure and combine them with post-convergence behavior from shield-constrained adversarial multi-agent reinforcement learning. Together these form a defensibility fingerprint capturing both a network's formal safety properties and its operational behavior under adaptive play. A what-if analysis shows that formal defensibility and operational effectiveness capture distinct aspects of security: small architectural changes can produce large shifts in operational outcomes while leaving formal safety margins nearly unchanged. Shield synthesis is thus most valuable not as a deployment mechanism for safe agents, but as a framework for answering architectural questions about whether, where, and how a system can be defended. The defensibility verdict is the output, not the safe policy.",
      "published": "2026-06-11T17:35:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.13621v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13621v1",
      "categories": [
        "cs.AI",
        "cs.CR",
        "cs.GT",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility",
      "authors": [
        "Xiaoyuan Liu",
        "Jianhong Tu",
        "Yuqi Chen",
        "Siyuan Xie",
        "Sihan Ren",
        "Tianneng Shi",
        "Gal Gantar",
        "Evan Sandoval",
        "Donghyun Lee",
        "Daniel Miao",
        "Peter J. Gilbert",
        "Nick Hynes",
        "Mauro Staver",
        "Warren He",
        "David Marn",
        "Andrew Low",
        "Xi Zhang",
        "Elron Bandel",
        "Michal Shmueli-Scheuer",
        "Siva Reddy",
        "Alexandre Drouin",
        "Alexandre Lacoste",
        "Ramayya Krishnan",
        "Elham Tabassi",
        "Yu Su",
        "Victor Barres",
        "Chenguang Wang",
        "Wenbo Guo",
        "Dawn Song"
      ],
      "abstract": "Agent systems are advancing quickly across domains, but their evaluation remains fragmented. Most benchmarks rely on fixed, LLM-centric harnesses that require heavy integration, create test-production mismatch, and limit fair comparison across diverse agent designs. The root problem is the lack of an open, agent-agnostic assessment interface. We advocate Agentified Agent Assessment (AAA), where evaluation is performed by judge agents and all participants interact through standardized protocols: A2A for task management and MCP for tool access. Conventional benchmarking defines two separate interfaces, one for the benchmark and one for the agent, while AAA only needs one; this yields a generic, unified framework that separates assessment logic from agent implementation and enables reproducible, interoperable, and multi-agent evaluation. We further introduce AgentBeats as a concrete realization of AAA: we identify five practical operation modes that make standardized assessment compatible with real-world constraints on openness, privacy, and reproducibility. To evaluate our design at scale, we conduct two studies: a five-month open competition that drew 298 judge agents across 12 categories together with 467 subject agents from independent participants, showing that AAA applies across a heterogeneous range of benchmarks; and a case study on coding agents that confirms agentified evaluation preserves fidelity with the public record while surfacing previously missing head-to-head results, yielding research insights about agent design. Combining a community-scale field study and a controlled coding case study, we verify that AAA delivers coverage, practicality, and fidelity across heterogeneous scenarios at scale. Together, AAA and AgentBeats offer a clear path toward open, standardized, and reproducible agent assessment.",
      "published": "2026-06-11T17:23:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.13608v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13608v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning",
      "authors": [
        "Zach Studdiford",
        "Gary Lupyan"
      ],
      "abstract": "When large language models (LLMs) fail to generalize or make haphazard errors in reasoning, it is often taken as evidence that LLMs are not truly reasoning, but rather performing a kind of pattern matching. The implication is that people's behavior does not exhibit the same types of failures because human reasoning uses principled and abstract world models. We evaluate human participants and 25 LLMs on their ability to engage in common-sense reasoning about a variety of everyday situations and observe similar patterns of errors in both people and models. We then identify the set of attention heads driving LLM responses and find that these heads implement a form of pattern-matching. These attention heads allow us to predict seemingly inexplicable reasoning errors in people caused by ostensibly irrelevant prompt details. Taken together, our results suggest that everyday causal reasoning in people and LLMs is more consistent with a form of pattern-matching than with abstract world models.",
      "published": "2026-06-11T17:23:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.13607v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13607v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Multi-Agent Reinforcement Learning from Delayed Marketplace Feedback for Objective-Weight Adaptation in Three-Sided Dispatch",
      "authors": [
        "Haochen Wu",
        "Yi Hou",
        "Shiguang Xie"
      ],
      "abstract": "Dispatch in three-sided marketplaces provides a natural setting for reinforcement learning from world feedback: decisions are evaluated by delayed operational outcomes such as delivery speed, courier utilization, and merchant congestion. We present a deployed reinforcement learning system at DoorDash that adapts dispatch objective weights in a large-scale food-delivery marketplace using delayed signals. Rather than replacing the combinatorial assignment optimizer, a store-level policy learned from logged marketplace data selects a discrete multiplier that shifts the dispatch optimizer's tradeoff between delivery quality and batching efficiency. This interface enables offline policy learning under noisy, delayed, and coupled feedback while preserving production feasibility constraints and operational safeguards. We train a shared value function using centralized offline data and decentralized store-level execution, with Double Q-learning targets and a conservative regularizer to reduce out-of-distribution value overestimation. In a production switchback experiment, the offline-trained policy increases batching and reduces courier-side time costs without degrading customer-facing delivery quality. Results illustrate how world feedback from a live economic and logistics system can be used to safely adapt decision policies online.",
      "published": "2026-06-11T17:21:20Z",
      "abstract_url": "http://arxiv.org/abs/2606.13604v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13604v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Beyond the Commitment Boundary: Probing Epiphenomenal Chain-of-Thought in Large Reasoning Models",
      "authors": [
        "Daniel Scalena",
        "Sara Candussio",
        "Luca Bortolussi",
        "Elisabetta Fersini",
        "Malvina Nissim",
        "Gabriele Sarti"
      ],
      "abstract": "Chain-of-thought (CoT) reasoning is the dominant paradigm for inference-time scaling in language models, yet the causal influence of individual steps on the final answer poorly understood. We estimate each step's causal importance via early exit and use this measure to study how answers form across the reasoning traces of several model families. Across diverse tasks, we find that reasoning typically crosses a \\emph{commitment boundary} -- a sharp transition from transient intermediate guesses to a stable, high-confidence answer. This transition often happens in a single step, well before the model's reasoning block ends, and is followed by \\emph{epiphenomenal} CoT steps that leave the final answer probability unaltered. Using attention probes, we show that answer-formation stages can be linearly decoded from intermediate reasoning steps with high accuracy and generalize robustly to unseen reasoning tasks. We exploit this signal to early-exit reasoning blocks at the commitment boundary, reducing the length of CoTs up to 55\\% on average with negligible impact on model performance.",
      "published": "2026-06-11T17:21:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.13603v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13603v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Reward Modeling for Multi-Agent Orchestration",
      "authors": [
        "King Yeung Tsang",
        "Zihao Zhao",
        "Vishal Venkataramani",
        "Haizhou Shi",
        "Zixuan Ke",
        "Semih Yavuz",
        "Shafiq Joty",
        "Hao Wang"
      ],
      "abstract": "Multi-Agent Systems (MAS) built on Large Language Models (LLMs) require effective orchestration to coordinate specialized agents, yet training such orchestrators is hindered by limited supervision and high computational cost. We propose Orchestration Reward Modeling (OrchRM), a self-supervised framework for evaluating orchestration quality without human annotations. OrchRM leverages intermediate artifacts from multi-agent executions to construct win-lose pairs for Bradley-Terry reward model training. Unlike existing MAS test-time scaling and orchestrator training frameworks that rely on costly sub-agent rollouts, OrchRM operates directly at the orchestration level, enabling efficient and high-performing reward-guided orchestrator training and MAS test-time scaling. OrchRM improves training efficiency by up to 10x in token usage while improving MAS test-time scaling performance by up to 8% in accuracy. These gains consistently transfer across multiple domains, including mathematical reasoning, web-based question answering, and multi-hop reasoning, demonstrating orchestration-level reward modeling as a scalable direction for robust multi-agent orchestration. Code will be available at https://github.com/Wang-ML-Lab/OrchRM.",
      "published": "2026-06-11T17:16:24Z",
      "abstract_url": "http://arxiv.org/abs/2606.13598v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13598v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Multiagent Protocols with Aggregated Confidence Signals",
      "authors": [
        "Ali Elahi",
        "Barbara Di Eugenio"
      ],
      "abstract": "Confidence is used for reliability, oversight, and a range of downstream decision tasks in Natural Language Processing (NLP), yet no existing method produces or evaluates a confidence for the output of a multiagent system. Prior work uses confidence within multiagent debate (MAD) to weight messages, trigger debate, or calibrate individual agents, but it never aggregates these into a single confidence for the system itself. We introduce three protocols that produce a final answer along with a single aggregated confidence by first transforming raw confidence signals to make them comparable across models, then combining them via soft voting or a probability fusion we call Bayesian fusion. This aggregated confidence is substantially more discriminative (AUARC) than that of the best single agent or the standard debate baselines, while correctness (F1-score) stays stable and recovers the losses MAD incurs on more ambiguous tasks. Analyzing two estimators, sequence probability and self-report, alongside parametric and non-parametric calibrators, we find that calibration improves F1 for both estimators while AUARC is less reliant on it. We evaluate six homogeneous and heterogeneous debating pairs per benchmark, across five benchmarks and four task types, spanning a range of model capabilities and sizes.",
      "published": "2026-06-11T17:12:11Z",
      "abstract_url": "http://arxiv.org/abs/2606.13591v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13591v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories",
      "authors": [
        "Baochang Ren",
        "Xinjie Liu",
        "Xi Chen",
        "Yanshuo Liu",
        "Chenxi Li",
        "Daqi Gao",
        "Zeqin Su",
        "Jintao Xing",
        "Zirui Xue",
        "Rui Li",
        "Xiangyu Zhao",
        "Shuofei Qiao",
        "Minting Pan",
        "Wangmeng Zuo",
        "Lei Bai",
        "Dongzhan Zhou",
        "Ningyu Zhang",
        "Huajun Chen"
      ],
      "abstract": "Scientific laboratories increasingly rely on AI systems to reason about experiments, but the physical act of doing science remains largely outside their reach. AI can help read literature, generate hypotheses, and plan protocols, yet the execution of those protocols at the bench still requires a human operator. Vision-Language-Action (VLA) models provide one possible interface between written protocols and robot execution, but existing policies are trained mostly on household and tabletop demonstrations and rarely encounter the instruments, transparent liquids, or fixed protocol workflows found in scientific laboratories. Closing this gap requires both laboratory-specific supervision and a unified learning framework that can accommodate the diverse robot embodiments used to execute experimental protocols. We therefore identify data and embodiment as central bottlenecks alongside model design. To address the data side, we build RoboGenesis, a simulation-based workflow and data engine that composes configured laboratory workflows from atomic skills, validates and filters rollouts, and exports structured demonstrations across supported robot profiles. On the policy side, we present LabVLA, trained with a two-stage recipe: FAST action token pretraining first makes the Qwen3-VL-4B-Instruct backbone action aware before any continuous control is learned, and flow matching posttraining then attaches a DiT action expert under knowledge insulation. On the LabUtopia benchmark, LabVLA achieves the highest average success rate among all evaluated baselines under both in-distribution and out-of-distribution settings.",
      "published": "2026-06-11T17:03:53Z",
      "abstract_url": "http://arxiv.org/abs/2606.13578v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13578v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG",
        "cs.MM",
        "cs.RO"
      ]
    },
    {
      "title": "ArogyaSutra: A Multi-Agent Framework for Multimodal Medical Reasoning in Indic Languages",
      "authors": [
        "Tanmoy Kanti Halder",
        "Akash Ghosh",
        "Subhadip Baidya",
        "Arijit Roy",
        "Sriparna Saha"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) have shown promising reasoning capabilities in general domains, yet their performance remains limited in specialized settings such as healthcare, especially in multilingual and low-resource scenarios. This gap is critical in regions like rural India, where patients often express complex medical queries in native Indic languages and rely on multimodal inputs such as medical images. Existing English-centric MLLMs struggle to support such use cases, limiting equitable access to AI-driven healthcare assistance. To address this challenge, we introduce ArogyaBodha, a large-scale multilingual multimodal medical question-answer dataset constructed from eight heterogeneous sources, covering 31 body systems, six imaging modalities, and 21 clinical domains across English and seven major Indian languages. We further propose ArogyaSutra, an actor-critic-based multi-agent framework that integrates tool grounding with dual-memory mechanisms for step-wise, reasoning-aware decision making, and uses stored actor-critic simulation trajectories for distillation. Experiments show that our dataset and framework improve multilingual medical reasoning accuracy across all Indic languages, with ablations validating the contribution of each component. The source code and dataset are available at: https://iitp-cse.github.io/ ArogyaSutra/",
      "published": "2026-06-11T16:59:42Z",
      "abstract_url": "http://arxiv.org/abs/2606.13572v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13572v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Existence Precedes Value: Joint Modeling of Observational Existence and Evolving States in Time Series Forecasting",
      "authors": [
        "Yifan Hu",
        "Hongzhou Chen",
        "Peiyuan Liu",
        "Yiding Liu",
        "Zewei Dong",
        "Jiang-Ming Yang"
      ],
      "abstract": "Real-world time series are often highly incomplete and irregular due to sensor dormancy, transmission delays, and event-driven sampling, making reliable forecasting fundamentally challenging. Existing methods have evolved from impute-then-forecast pipelines to continuous-time models such as Neural ODEs and continuous-time graph networks. While these approaches improve the modeling of historical irregularity, they still rely on an implicit oracle assumption at inference time: the timestamps of future valid observations are presumed to be known in advance. This assumption limits practical relevance, since in many real systems the more fundamental question is not only what the future value will be, but also whether a valid observation will occur at all. In this paper, we propose Timeflies, a unified framework that reformulates forecasting as a joint problem of future observability inference and value estimation. To explicitly model the interaction between observation dynamics and state evolution, Timeflies adopts an observation stream and a value stream, coupled through three dedicated modules for reliability-aware embedding, observation-guided dependency modeling, and joint prediction. We further construct Shadow, a benchmark that combines natural missingness from public datasets with real-world industrial data, and introduce the Observation-Value Joint Entropy (OVJE) metric to comprehensively evaluate this coupled predictability. Extensive experiments show that Timeflies consistently outperforms existing methods, highlighting the importance of explicitly modeling future observability in time series forecasting with missing values. Code and dataset are available in https://github.com/ant-intl/Timeflies.",
      "published": "2026-06-11T16:59:42Z",
      "abstract_url": "http://arxiv.org/abs/2606.13571v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13571v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Three-Layer Framework for AI in Scientific Discovery",
      "authors": [
        "Guojun Liao"
      ],
      "abstract": "Current discussions of AI in scientific discovery are often dominated by two visible capabilities: search over existing knowledge and execution through optimization, simulation, and automation. Both are important, but neither fully captures the central act of discovery: the formation and evolution of models. This paper proposes a three-layer view of AI in discovery. Layer 1 is search and retrieval by large language models. Layer 2, as the main innovation of this paper, is model formation through qualitative reasoning: the capacity to recognize when a current framework is structurally inadequate and to understand the problem within a broader representational space, not through trial and error, but through structural insight into what is missing and where it can be found. Layer 3 is execution, optimization, and refinement. The main claim is that Layer 2 is both the most important and the least developed. Search without model formation remains confined to inherited frameworks, while execution without conceptual revision only amplifies an existing formulation. We illustrate Layer 2 reasoning through three case studies: S. S. Chern's intrinsic proof of the Gauss-Bonnet theorem, the resolution of the Nesterov Accelerated Gradient convergence problem via Lyapunov functions, and the autonomous disproof of the Erdos unit distance conjecture by OpenAI in 2026. Each case exhibits the same structural signature: a framework that had become inadequate, a missing conceptual object, and a resolution found in an unexpected neighboring field.",
      "published": "2026-06-11T16:56:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.13566v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13566v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Is It You or Your Environment? A Bayesian Inference Framework for Genomically-Anchored Personalized Physiological Interpretation",
      "authors": [
        "Aruna Dey",
        "Suraj Biswas"
      ],
      "abstract": "Personalized health AI systems face a fundamental cold-start problem: machine learning models for physiological interpretation require weeks of individual behavioral data before they can distinguish constitutional variation from environmentally driven deviation. We propose a solution grounded in causal inference and Bayesian prior design. An individual's genomic profile serves as an exogenous genetic anchor -- a domain-informed, personalized prior that is fixed at conception, immune to reverse causation, and available before a single behavioral observation is collected. The anchor initializes a Bayesian belief state over an individual's physiological set point G-hat = mu + sum(beta_i * g_i), where beta_i are GWAS-derived effect sizes and g_i are risk-allele counts. Each incoming physiological measurement P produces a non-constitutional deviation delta = P - G-hat that separates the signal attributable to environment and state from the constitutionally fixed baseline. As behavioral data accrue, the prior decays according to G-hat_t = w(t)*G-hat_genomic + [1-w(t)]*P-bar_t, transitioning from genome-dominated to empirical-baseline-dominated inference. The same observed HRV of 55 ms generates a suppression hypothesis for a person whose prior predicts 80 ms, and an enhancement hypothesis for a person whose prior predicts 30 ms -- a reversal impossible without a personalized anchor. We develop this architecture across six physiological domains, grading genomic priors by evidence strength, distinguishing robustly replicated anchors (FTO, FADS1/2, FKBP5) from contested candidate genes (SLC6A4, MAOA, DRD2). We address the inference boundary between association, Mendelian randomization, and individual token causation, and define four constraints for deployment: evidence-graded priors, dynamic decay, ancestry-matched effect sizes, and attribution rather than deterministic output.",
      "published": "2026-06-11T16:38:38Z",
      "abstract_url": "http://arxiv.org/abs/2606.13556v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13556v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "q-bio.BM",
        "q-bio.GN",
        "q-bio.MN"
      ]
    },
    {
      "title": "Adaptive Turn-Taking for Real-time Multi-Party Voice Agents",
      "authors": [
        "Soumyajit Mitra",
        "Prabhat Pandey",
        "Abhinav Jain",
        "Shanmukha Sahith",
        "K V Vijay Girish"
      ],
      "abstract": "Turn-taking in multi-party spoken conversations remains a fundamental challenge for voice-based agents, particularly under dynamic floor competition and varying user expectations. We propose ModeratorLM, a role-playing voice agent that conditions turn-taking behavior on an explicitly assigned role in multi-party settings. The system is built on a speech large language model operating in chunk-wise streaming manner. We further introduce a reasoning-augmented variant that incorporates chain-of-thought reasoning over conversational context and the assigned role. We construct RolePlayConv, a large-scale synthetic dataset of spoken multi-party conversations with diverse assistant roles. Experiments on real-world meeting data and RolePlayConv show improved turn-taking precision by over 40% and recall by more than 70%, while substantially reducing false-positive interruptions compared to non-role-conditioned baselines.",
      "published": "2026-06-11T16:27:45Z",
      "abstract_url": "http://arxiv.org/abs/2606.13544v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13544v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "AgentRivet: an automated system for producing Rivet routines from journal publications",
      "authors": [
        "Antonio J. Costa",
        "Caterina Doglioni",
        "Christian Gütschow",
        "Andrew D. Pilkington",
        "Sukanya Sinha"
      ],
      "abstract": "Particle physics collider experiments provide Rivet routines as part of the analysis preservation strategy for model-independent measurements. Rivet is a C++ toolkit that allow new theoretical models to be compared to the measurements, thus aiding the development and tuning of Monte Carlo event generators as well as searches for physics beyond the Standard Model. However, analysis coverage is known to be incomplete, with only 39% of measurements having documented and publicly available Rivet routines. In this article, we design and implement an automated workflow based on Large Language Models with the goal of providing the missing routines. This multi-step workflow, referred to as AgentRivet, extracts the physics analysis information from published papers and writes the missing Rivet routines, with intermediate code- and physics- reviews as part of an autonomous quality control. We report the results obtained using commercial Large Language Models, provided by OpenAI, Anthropic, and Google, for two recent measurements from the ATLAS and CMS experiments. We find that AgentRivet produces competent Rivet routines with few syntax errors. The physics fidelity of the routines is reasonable and follows the explanations given in the relevant publications. Nevertheless, physics-implementation issues do arise and are investigated using the artefacts produced by AgentRivet. The majority of physics implementation issues arise from subtle-but-ambiguous definitions in the given publication, although some models struggle to implement complex observables even when clear definitions are given.",
      "published": "2026-06-11T16:22:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.13535v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13535v1",
      "categories": [
        "hep-ex",
        "cs.AI",
        "hep-ph"
      ]
    },
    {
      "title": "CRAFTIIF: Cross-Resolution Analytic Four-Type Interpretable Isolation Forest for Multivariate Time Series Anomaly Detection",
      "authors": [
        "William Smits"
      ],
      "abstract": "Anomaly detection in multivariate time series is challenged by four structurally distinct anomaly types -- point (isolated spikes), distributional (level shifts), temporal (rhythm changes), and collective (inter-sensor correlation breakdowns) -- each requiring different feature representations. Most unsupervised methods target only one or two types and provide limited interpretability. We present CRAFTIIF (Cross-Resolution Analytic Four-Type Interpretable Isolation Forest), a fully unsupervised framework targeting all four types without dataset-specific tuning. CRAFTIIF generates K=500 random analytic wavelet feature draws across four families (Morlet, DOG, Haar, Coiflet), each targeting a specific anomaly type, feeding five structured Isolation Forests -- one per type plus a meta-IF for compound anomalies. An adaptive Otsu/MAD threshold calibrates detection automatically across anomaly rates from 0.1% to 69.2%. Because each IF is trained exclusively on type-specific features, branch firing provides direct anomaly-type attribution by construction, without post-hoc explanation. Evaluated on all 19 datasets of the mTSBench benchmark (Zhou et al., TMLR 2026), CRAFTIIF achieves mean F1=0.228 (all 19 datasets) and F1=0.322 (13 detectable datasets), ranking first among all 25 evaluated methods on VUS-PR (0.463 vs. previous best 0.329, +40.7%). A diagnostic framework -- oracle F1, detectability limits, and branch separation ratios -- identifies 6 of 19 datasets as fundamentally undetectable by any unsupervised method. Ablation over 11 conditions confirms adaptive thresholding (+38% F1), four-branch structure (+20%), and meta-IF (+23%) are each essential. Code: https://github.com/smitswil/craftiif",
      "published": "2026-06-11T15:36:14Z",
      "abstract_url": "http://arxiv.org/abs/2606.13486v1",
      "pdf_url": "https://arxiv.org/pdf/2606.13486v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
