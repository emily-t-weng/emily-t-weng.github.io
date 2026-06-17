const PAPERS_DATA = {
  "last_updated": "2026-06-17 04:57:53 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues",
      "authors": [
        "Shanda Li",
        "Qiuhong Anna Wei",
        "Jingwu Tang",
        "Valerie Chen",
        "Nihar B Shah",
        "Tim Dettmers",
        "Yiming Yang",
        "Ameet Talwalkar"
      ],
      "abstract": "Reproducing research results from papers and released code is central to scientific progress. Existing works have introduced benchmarks to evaluate whether LLM agents can assist with reproducibility, but they are difficult to scale due to their reliance on substantial manual effort for data curation and evaluation. We introduce ReproRepo, a scalable framework for reproducibility evaluation that leverages human-raised GitHub issues as naturally occurring supervision on realistic reproduction blockers. We instantiate ReproRepo on 1,149 recent machine learning papers from major conferences and evaluate four frontier model-agent configurations. Our results show that LLM agents, even without executing code, can identify many real-world reproducibility problems from paper-repository pairs: the best agent in our study, namely Codex with GPT-5.5, surfaces at least one semantically related human-reported blocker for ~90% of papers in the study. Further analysis shows that agents are particularly effective for surfacing visible failures and identifying the right semantic region, but may still be insufficient in exact localization. ReproRepo can serve as a reusable, scalable framework for future evaluations of LLM agents on real-world reproducibility auditing. Our code is released at https://github.com/LithiumDA/ReproRepo.",
      "published": "2026-06-16T17:58:05Z",
      "abstract_url": "http://arxiv.org/abs/2606.18237v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18237v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning Red Agent Policy from Observations for Neurosymbolic Autonomous Cyber Agents",
      "authors": [
        "Ankita Samaddar",
        "Sandeep Neema",
        "Daniel Balasubramanian",
        "Xenofon Koutsoukos"
      ],
      "abstract": "With sophisticated cyber-attacks becoming increasingly prevalent, modern networks require intelligent autonomous cyber-defense agents trained via Reinforcement Learning (RL). These agents employ neurosymbolic approaches such as behavior trees with learning-enabled components (LECs) to learn, reason, adapt, and implement security rules while maintaining critical operations. However, these autonomous networks are partially observable systems, i.e., the cyber-attacker's (red agent's) actions are not observable, making it difficult for the defender to predict red actions, learn red policies, or assess the attacker's intrusion levels. To address this, we propose a Policy Learning Technique using imitation learning to learn policies for partially observable RL agents with discrete states and discrete actions. We apply this technique in an autonomous cyber environment to predict red agent's actions from network observations and defender actions. Integrated with a neurosymbolic cyber-defense agent, our method effectively handles different red policies and achieves high prediction accuracy across diverse simulated scenarios.",
      "published": "2026-06-16T17:50:41Z",
      "abstract_url": "http://arxiv.org/abs/2606.18223v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18223v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "Looped World Models",
      "authors": [
        "Hongyuan Adam Lu",
        "Z. L. Victor Wei",
        "Qun Zhang",
        "Jinrui Zeng",
        "Bowen Cao",
        "Lingwei Meng",
        "Mocheng Li",
        "Zezhong Wang",
        "Haonan Yin",
        "Naifu Xue",
        "Minyu Chen",
        "Cenyuan Zhang",
        "Zefan Zhang",
        "Hao Wei",
        "Jiawei Zhou",
        "Haoran Xu",
        "Hao Yang",
        "Ronglai Zuo",
        "Tongda Xu",
        "Yonghao Li",
        "Jian Chen",
        "Hebin Wang",
        "Zeyu Gao",
        "Yang Li",
        "Wei Zhao",
        "Qimin Zhong",
        "Siqi Liu",
        "Yumeng Zhang",
        "Leyan Cui",
        "Zhangyu Wang",
        "Wai Lam"
      ],
      "abstract": "Current world models face a fundamental tension: faithful long-horizon simulation demands deep computation, but deeper models are expensive to deploy and prone to compounding errors. We resolve this by introducing Looped World Models (LoopWM), which are the first looped architectures for world modelling. Our method iteratively refines latent environment states through a parameter-shared transformer block. This yield up to 100x parameter efficiency over conventional approaches with adaptive computation that automatically scales depth to match the complexity of each prediction step. Orthogonal to scaling model size and training data, LoopWM establishes iterative latent depth as a new scaling axis for world simulation, which might significantly push the community forward.",
      "published": "2026-06-16T17:37:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.18208v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18208v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.CV"
      ]
    },
    {
      "title": "A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models",
      "authors": [
        "Nicola Franco"
      ],
      "abstract": "We evaluate the adversarial robustness of two frontier large language models (LLMs) developed by Anthropic, Fable 5 and Opus 4.8, against four families of automated jailbreak attack across 7 826 harmful intents spanning a ten-category harm taxonomy. Using the HackAgent red-teaming framework, hundreds of thousands of adversarial attempts were generated and every apparent success was independently re-adjudicated by a panel of three judge models (majority vote). Both models resist the majority of attacks, but the residual surface is larger than aggregate framing suggests: it is dominated by adaptive iterative attacks, while static obfuscation is near-fully neutralised. The strongest adaptive search (tree-of-attacks) breaks Opus 4.8 on 11.5% of intents overall, whereas Fable 5 stays in the single digits (6.1% worst-case). Aggregate rates therefore should not be read as reassurance. Even in these hardened configurations, the two models produced 1 620 (Opus 4.8) and 702 (Fable 5) panel-confirmed harmful completions spanning every harm category, located automatically, cheaply, and within the first one or two refinement steps by an attacker model with no human expert in the loop. The reasonable conclusion is that even the best, most-tested frontier models remain reliably breakable under sustained automated pressure.",
      "published": "2026-06-16T17:23:58Z",
      "abstract_url": "http://arxiv.org/abs/2606.18193v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18193v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "The Stanford EDGAR Filings Dataset: Reconstructing U.S. Corporate and Financial Disclosures into Layout-Faithful and Token-Efficient Pretraining Data",
      "authors": [
        "Nick Bettencourt",
        "Xiaowei Ding",
        "Kay Giesecke"
      ],
      "abstract": "As high-quality public web corpora become increasingly exhausted, clean long-context documents have become a scarce and expensive source of training data for large language models (LLMs). Existing long-context corpora are often proprietary and costly to acquire, synthetically generated, or concentrated in narrow domains such as programming. We introduce the Stanford EDGAR Filings Dataset (SEFD), an open reconstruction of SEC filings into layout-faithful MultiMarkdown for financial language modeling and evaluation. SEFD makes audited financial statements, risk disclosures, ownership reports, accounting notes, and market-moving event filings usable as long-context pretraining data and as a basis for financial reasoning, forecasting, compliance, and document understanding. The resulting corpus is token-efficient, model-ready, and has less than 0.1% overlap with Common Crawl-derived corpora. We release SEFD-v1, a 152B-token initial public snapshot, and provide corpus-level analyses of a larger 18.5M-filing archive estimated at 550B tokens. We further introduce two SEFD-derived benchmarks: EDGAR-Forecast, which evaluates filing-grounded numerical forecasting after model knowledge cutoffs, and EDGAR-OCR, which evaluates transcription of complex financial tables.",
      "published": "2026-06-16T17:22:34Z",
      "abstract_url": "http://arxiv.org/abs/2606.18192v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18192v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Kolmogorov Regression for Robust Diffusion Policies",
      "authors": [
        "Lekan Molu"
      ],
      "abstract": "Finite-dimensional (FD) diffusion policies exhibit temporal drift owing to discretization artifacts that degrade long-horizon performance (when deployed on physical systems). We introduce a backward Kolmogorov equation that lifts diffusion policies to a Cameron-Martin space -- a subset of the Hilbert space. Essentially, replacing stochastic score matching with a deterministic boundary-value PDE problem. Our core innovation thrives on Gaussian measure theory whereupon the diffusion noise covariance operator is realized from a colored noise distribution which prescribes a notion of regularity on samples from the model at inference time. We train the diffusion model with a derived precision-weighted Cameron- Martin loss and a Kolmogorov residual is introduced as a PDE diagnostic during inference. These substitutions yield (i) convergence guarantees where the bound's constants depend on the effective rank of the kernel rather than action dimension, (ii) improved trajectory regularity via spectral weighting, and (iii) a deterministic failure detector without reward signals. Validation across two application domains demonstrates substantial improvements: on the PushT manipulation benchmark, the Cameron-Martin loss achieves a 17% improvement in maximum episode reward (0.95 vs. 0.78 for MSE) and 67.6% reduction in inter-step drifts during inference via the introduced residual magnitude. Similarly, on a 6-station manufacturing line with constant work-in-process (CONWIP) flow control, we achieve 28.4% lower RMSE than classical LSTM baselines; a high starvation-event recall (1.0 in test cycles), and effective bottleneck identification (Precision@1 = 1.0 in test set, 13x signal-to-noise ratio). We then certify the dispatch policies with Hamilton-Jacobi reachability theory which reduces deadlock events by 96% compared to uncontrolled dispatch over 100 simulated runs (351 events prevented).",
      "published": "2026-06-16T17:18:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.18186v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18186v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "IUU+DB: Tracking Illegal, Unreported, and Unregulated Fishing, Seafood Fraud, and Labor Abuse through LLM-driven Information Extraction",
      "authors": [
        "Henry Bodwell",
        "Hong Yang",
        "John C. Simeone",
        "Kelvin Gorospe",
        "Bella Sullivan",
        "Lana Huang",
        "Jessica Gephart",
        "Sandy Aylesworth",
        "Molly Masterton",
        "Naren Ramakrishnan"
      ],
      "abstract": "Illegal, unreported, and unregulated fishing (IUU) traditionally refers to fishing activities that violate applicable laws or occur in areas that lack applicable laws. We propose the term IUU+ to capture a broader suite of fisheries sector environmental and associated supply chain trade-related crimes and behaviors. Although IUU+ activity is widely recognized as a serious threat to marine ecosystems, markets, and livelihoods, a quantitative understanding of these incidents, e.g., their frequency, geography, species, actors, and patterns in the type of illicit activity, remains difficult to obtain. We propose IUU+DB, a large language model driven system for building a global incident database of IUU+ activity. The system ingests heterogeneous documents, classifies whether they describe relevant incidents, extracts key data elements such as actors, locations, species, vessels, violations, and enforcement outcomes, and supports deduplication and trend analysis. Case studies and validation results show that IUU+DB can help organize fragmented evidence, surface geographic and behavioral hotspots, support fisheries-domain specific research in academia and non-government organizations, assist source and species risk assessments for industry, and provide support for policy implementation and targeted enforcement efforts to government agencies.",
      "published": "2026-06-16T17:16:05Z",
      "abstract_url": "http://arxiv.org/abs/2606.18181v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18181v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "The Measurement Gap in the Automation of EU Law: Benchmarking Doctrinal Legal Reasoning under the EU AI Act",
      "authors": [
        "Michèle Finck"
      ],
      "abstract": "Large language models now produce legal text of at least median quality, yet no existing benchmark can evaluate whether they perform doctrinal legal reasoning, which forms the interpretive core of legal work, rather than the ancillary, paralegal tasks that most current legal-AI evaluations measure. This measurement gap is not only methodological but legal: the EU AI Act makes \"appropriate accuracy\" a binding requirement for high-risk AI used in the judicial domain, yet that requirement cannot acquire operational content without the very doctrinal-reasoning benchmark the field lacks.",
      "published": "2026-06-16T16:57:12Z",
      "abstract_url": "http://arxiv.org/abs/2606.18158v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18158v1",
      "categories": [
        "cs.CY",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Learning Cardiac Electrophysiology Digital Twins Through Agentic Discovery of Hybrid Structure",
      "authors": [
        "Ziqi Zhou",
        "Yubo Ye",
        "Sumeet Atul Vadhavka",
        "Linwei Wang",
        "Zhiqiang Tao"
      ],
      "abstract": "Building personalized cardiac electrophysiology (EP) digital twins requires identifying the appropriate model structure for each patient, not merely fitting parameters. Traditional methods rely on experts to manually prescribe hybrid physics-neural architectures, which requires deep domain expertise and does not transfer across patients. Recent works have applied large language models (LLMs) to generate or act as hybrid models. However, despite their promising generalization capacity, these LLM-based methods lack the structural priors needed for stable cardiac simulations. Hence, we propose LEADS, a framework that formulates cardiac EP domain knowledge as a structured action space and utilizes an LLM agent to discover hybrid models. The agent follows an iterative reasoning-and-action loop to select, combine, and refine hybrid models, whilst gradient descent handles parameter fitting. The proposed LEADS designs every candidate model towards physically grounded, interpretable, and numerically stable, while allowing open-ended architectural discovery. We validate LEADS on synthetic data with three ground-truth reaction models and on real cardiac EP data, demonstrating that it outperforms both human-designed hybrid models and other LLM-based hybrid modeling.",
      "published": "2026-06-16T16:54:03Z",
      "abstract_url": "http://arxiv.org/abs/2606.18154v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18154v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Memory as a Wasting Asset: Pricing Flash Endurance for Embodied Agents, and the Limits of Doing So",
      "authors": [
        "Josef Liyanjun Chen"
      ],
      "abstract": "A robot's flash endurance is a non-renewable stock: every persisted write spends one of a few thousand program/erase cycles and never refills, yet no fielded robot memory system prices which memories are worth an erase cycle. We treat embodied memory as depreciating capital and price that stock with a single endurance shadow price $η$, which makes cost-minimizing placement across a RAM / on-board NVM / cloud hierarchy a threshold in a wear-augmented per-byte index. The index is cost-optimal whatever the sign of the value-write association $χ$; only when $χ> 0$ does the optimum turn non-monotone, sending a robot's most valuable memories off its flash. The pivot is thus empirical, and we measure $χ$ on real robot logs at a pre-specified gate: its sign is a property of the deployment regime -- positive on recurrent long-horizon manipulation ($\\hatχ \\approx +1.0 \\times 10^{-3}$, replicated at full power), null on a shorter-horizon suite, and negative on non-recurrent teleoperation. Two boundaries scope the result. The endurance budget is dormant on premium 3,000-P/E TLC at datasheet prices and binding on the commodity QLC/eMMC ($\\sim$1,000 P/E) that cheaper edge robots run. And where it binds, a learned wear-aware controller only ties price-based routing on task value, because realized value is tier-invariant across RAM, NVM, and cloud: the rent governs device lifetime and cost, not task performance. Whether wear-aware placement improves task value remains open -- $χ$ is measured against a value proxy, and the non-monotone optimum, while proven, is not yet observed in data.",
      "published": "2026-06-16T16:43:19Z",
      "abstract_url": "http://arxiv.org/abs/2606.18144v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18144v1",
      "categories": [
        "cs.AI",
        "cs.CY",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Embedded Machine Learning for Microcontroller-Class Edge Devices: Data, Feature, Evaluation, and Deployment Pipelines",
      "authors": [
        "Mostafa Darvishi"
      ],
      "abstract": "Embedded machine learning moves inference from cloud services to resource-constrained devices that must acquire data, preprocess signals, run a model, and act within tight limits on memory, energy, and latency. This paper presents a systems-oriented synthesis of an embedded machine-learning workflow for microcontroller-class platforms. The emphasis is placed on engineering decisions that are often hidden in generic machine-learning introductions: sampling and buffering, feature extraction as dimensionality reduction, validation under class imbalance, model/runtime co-design, and streaming deployment. Two representative signal families are used throughout the paper. The first is inertial motion recognition, where a two-second, three-axis accelerometer window is transformed from raw samples into root-mean-square and spectral features before classification. The second is keyword spotting, where audio is sampled, anti-aliased, transformed into mel-frequency cepstral coefficients, and processed by a compact one-dimensional convolutional network. The paper concludes with practical design rules for robust on-device inference, including data curation, quantization, thresholding, scheduling, and field monitoring.",
      "published": "2026-06-16T16:22:24Z",
      "abstract_url": "http://arxiv.org/abs/2606.18122v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18122v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.AR",
        "eess.AS",
        "eess.SP"
      ]
    },
    {
      "title": "Structural Role Injection in Handlebars-Templated LLM Prompts: Triple-Brace Interpolation, Delimiter Family, and the Limits of HTML Auto-Escaping",
      "authors": [
        "Mohammadreza Rashidi"
      ],
      "abstract": "Large language model applications build prompts from templates, and Handlebars is a widely used templating engine and the default prompt-template format in Microsoft Semantic Kernel. Its double-brace {x} expression HTML-escapes the interpolated value and is documented as the safe default; its triple-brace {x} expression inserts the value raw. We show that this choice silently governs an application's exposure to structural role injection, where attacker-controlled data carries chat role delimiters that forge a higher-privilege turn. A model-free analysis establishes the mechanism: Handlebars escaping rewrites angle brackets but not square brackets, colons, or Markdown hashes, so it neutralises ChatML, Llama-3, and XML role delimiters (survival rate 0.00) while leaving Llama-2 [INST], legacy Human:/Assistant:, and Markdown ### delimiters intact (survival rate 1.00 for the last two). We then run 5760 trials across seven delimiter families, two attack objectives, and four models (GPT-3.5 Turbo, GPT-4o mini, GPT-4.1 mini, Claude Haiku 4.5) at a combined API cost of 1.63 USD. GPT-3.5 Turbo follows the task-hijack instruction in 97% of raw and 91% of escaped trials, with the escaping protection concentrated in the angle-bracket families and absent for the colon- and Markdown-based families; the harder secret-exfiltration objective, which does not saturate, exposes the same family interaction more cleanly. Claude Haiku 4.5 resists both objectives almost entirely. The escaped default protects only the delimiter schemes whose characters HTML escaping happens to cover, gives no protection for the rest, and cannot substitute for a structural separation of instruction and data.",
      "published": "2026-06-16T16:21:43Z",
      "abstract_url": "http://arxiv.org/abs/2606.18120v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18120v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models",
      "authors": [
        "Ramprasath Ganesaraja",
        "Sahil Dilip Panse",
        "Swathika N"
      ],
      "abstract": "State Space Models (SSMs) such as Mamba-2 offer linear-time inference but their memory footprint limits edge deployment. Prior ternary SSM work (Slender-Mamba) trains from scratch on 150B tokens; we show a pretrained checkpoint suffices, reducing the marginal token budget by 1,000x. Using grouped quantization-aware training (QAT) with knowledge distillation from a frozen FP16 teacher, we compress Mamba-2 1.3B to 3.61x (2,687 to 744 MB) and achieve 48.1% zero-shot accuracy (7-task average) in just 102M tokens (4 GPU-hours, single H100) -- approaching Bi-Mamba's 48.4% (within +/-0.9pp CI). This QAT-from-pretrained setting reveals zero-ratio collapse, a novel instability caused by learnable quantization scales that does not arise in from-scratch training. We further show that post-hoc correction strategies effective for Transformers fail for SSMs due to error accumulation through the recurrence. These results demonstrate that ternary SSMs do not require expensive from-scratch training: QAT from pretrained checkpoints with KD is a data-efficient alternative.",
      "published": "2026-06-16T16:18:21Z",
      "abstract_url": "http://arxiv.org/abs/2606.18114v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18114v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning Fair Pareto-Optimal Policies in Multi-Objective Reinforcement Learning",
      "authors": [
        "Umer Siddique",
        "Peilang Li",
        "Yongcan Cao"
      ],
      "abstract": "Fairness is an important aspect of decision-making in multi-objective reinforcement learning (MORL), where policies must ensure both optimality and equity across multiple, potentially conflicting objectives. While single-policy MORL methods can learn fair policies for fixed user preferences using welfare functions such as the generalized Gini welfare function (GGF), they fail to provide the diverse set of policies necessary for dynamic or unknown user preferences. To address this limitation, we formalize the fair optimization problem in multi-policy MORL, where the goal is to learn a set of Pareto-optimal policies that ensure fairness across all possible user preferences. Our key technical contributions are threefold: (1) We show that for concave, piecewise-linear welfare functions (e.g., GGF), fair policies remain in the convex coverage set (CCS), which is an approximated Pareto front for linear scalarization. (2) We demonstrate that non-stationary policies, augmented with accrued reward histories, and stochastic policies improve fairness by dynamically adapting to historical inequities. (3) We propose three novel algorithms, which include integrating GGF with multi-policy multi-objective Q-Learning (MOQL), state-augmented multi-policy MOQL for learning non-statoinary policies, and its novel extension for learning stochastic policies. We evaluate our algorithms across various domains and compare our methods against the state-of-the-art MORL baselines. The empirical results show that our methods learn a set of fair policies that accommodate different user preferences.",
      "published": "2026-06-16T16:16:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.18111v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18111v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Querying an astronomical database using large language models: the ALeRCE text-to-SQL system",
      "authors": [
        "P. A. Estevez",
        "J. Espejo-Moreira",
        "S. Sanfeliu-Alvarez",
        "F. Forster",
        "A. M. Munoz Arancibia",
        "G. Cabrera-Vives",
        "F. E. Bauer",
        "A. Bayo",
        "M. Catelan",
        "R. Dastidar",
        "L. Hernandez-Garcia",
        "J. A. Intriago",
        "G. Pignata"
      ],
      "abstract": "We develop a text-to-SQL (structured query language) system based on large language models (LLMs) using in-context learning and apply it to the Automatic Learning for the Rapid Classification of Events (ALeRCE) astronomical database. ALeRCE is a community broker for the Zwicky Transient Facility and the Vera C. Rubin Observatory. The system enables users to query the database in natural language (NL) and generates executable SQL queries. To develop and evaluate the system, we constructed a dataset of 110 NL/SQL pairs. We propose a step-by-step generation framework comprising four modules: schema linking, query classification, prompt decomposition, and self-correction. The performance of thirteen LLMs is evaluated using in-context learning and prompt engineering techniques. Text-to-SQL performance is assessed using the perfect-match (PM) rate for row identifiers (e.g., object identifiers) and column identifiers (i.e., column names). The proposed step-by-step framework consistently outperforms a direct-inference baseline, while the self-correction module consistently reduces execution errors. For Claude Opus 4.6, PM performance on row (column) identifiers is high for simple queries, reaching 0.97 (0.94), and decreases with query complexity to 0.44 (0.72) for medium queries and 0.59 (0.49) for hard queries. Among the thirteen evaluated models, the best-performing LLMs for the text-to-SQL task are Claude Opus 4.6, Gemini 2.5 Pro, Gemini 3 Flash, and GPT-5.2-Codex.",
      "published": "2026-06-16T16:12:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.18108v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18108v1",
      "categories": [
        "astro-ph.IM",
        "cs.AI"
      ]
    },
    {
      "title": "IsabeLLM: Automated Theorem Proving Applied to Formally Verifying Consensus",
      "authors": [
        "Elliot Jones",
        "William Knottenbelt"
      ],
      "abstract": "Advances in Artificial Intelligence (AI) have led AI for Theorem Proving to become a promising means of formally verifying computer systems. Whilst formal verification is traditionally reserved for safety-critical systems due to the required amount of expertise and effort, AI can help to automate a large amount of this workload and make it far more accessible. Blockchain-based systems are becoming increasingly popular and are frequently targeted by malicious actors, often resulting in huge financial losses, highlighting the need to better verify these systems and mitigate vulnerabilities. Arguably the most important component of these systems is the consensus protocol, which allows nodes to agree on decisions in a potentially adversarial environment. In this paper, we improve upon IsabeLLM, the automated theorem proving tool in Isabelle. Namely, we implement a Retrieval-Augmented Generation framework, Error tracing and counterexample generation for improved context supplied to the Large Language Model. Compatibility with the latest version of Isabelle and Sledgehammer is also implemented for improved efficiency. We compare the performance of the two versions of IsabeLLM in their ability to complete the verification of Bitcoin's Proof of Work consensus.",
      "published": "2026-06-16T16:00:14Z",
      "abstract_url": "http://arxiv.org/abs/2606.18098v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18098v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "S4oP: Operator-level Pruning of Structured State Space Models for Resource-Constrained Devices",
      "authors": [
        "Marco Deano",
        "Filippo Ziche",
        "Nicola Bombieri"
      ],
      "abstract": "Structured State Space Models (SSMs), including the S4 and S4D architectures, have recently emerged as powerful alternatives to attention-based models for capturing long-range dependencies in sequential data. Despite their strong empirical performance, deploying these models in time- and resource-constrained settings remains challenging due to their computational and memory demands. In this paper, we propose a novel incremental, operator-level pruning approach for S4- and S4D-based models that significantly reduces inference cost while preserving predictive performance. To the best of our knowledge, this is the first work to systematically investigate structured operator pruning for SSMs. Our method progressively prunes model operators by interleaving structured masking with fine-tuning, while jointly monitoring accuracy and inference latency. We implement this approach within a unified training and evaluation framework that enables systematic exploration of efficiency-accuracy trade-offs. Experiments across multiple benchmark datasets show that pruning up to 70% of the model operators preserves the performance of the original models in most cases, while substantially reducing inference latency. These results demonstrate that structured operator pruning is an effective and previously unexplored strategy for improving the efficiency of SSMs and facilitate their deployment in practical, resource-constrained scenarios.",
      "published": "2026-06-16T15:59:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.18096v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18096v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC"
      ]
    },
    {
      "title": "A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation",
      "authors": [
        "Haoyang Zhong",
        "Yifei Sun",
        "Antong Zhang",
        "Chunping Wang",
        "Lei Chen",
        "Yang Yang"
      ],
      "abstract": "Retrieval-Augmented Generation (RAG) has emerged as a paradigm for enhancing large language models (LLMs) with external knowledge, yet existing graph-based methods face a fundamental limitation: entity-centric and chunk-centric approaches operate on representations anchored to original text without true knowledge fusion. While entity-centric methods connect logically related content and chunk-centric methods preserve context, both retrieve information separately through similarity search, missing emergent understanding from their synthesis. In this paper, we propose HyGRAG, a hierarchical graph RAG framework that transcends source documents by addressing three core challenges: constructing summaries that genuinely integrate contextual and relational information, leveraging these synthesized representations to access emergent knowledge during retrieval, and efficiently updating hierarchical structures for dynamic corpora. Specifically, we design hierarchical index structures over hybrid graphs with both chunk and entity nodes, then iteratively cluster them and generate LLM-based summaries. Then, we design context and relation-aware retrieval that searches across all abstraction levels while expanding through community membership. Moreover, we enable dynamic knowledge update through attachment-based algorithms with only local re-summarization. Experimental results show that HyGRAG improves the average accuracy of multi-hop reasoning tasks by 9.7%, while maintaining reasonable efficiency.",
      "published": "2026-06-16T15:44:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.18075v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18075v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Volterra Generative Models",
      "authors": [
        "Yusen Jia",
        "Bingyan Han"
      ],
      "abstract": "Score-based diffusion models typically use Brownian perturbations, which provide tractable reverse-time dynamics but impose memoryless noising. We introduce Volterra generative models, a continuous-time score-based framework whose forward process injects path-dependent noise through fractional kernels. To handle the non-Markovian and non-semimartingale dynamics, we construct finite-dimensional Markovian lifts using Gaussian quadrature in both regimes and a hybrid finite-difference exponential approximation in the smooth regime. We prove squared error bounds, derive an augmented linear-Gaussian forward process, and show that the learning can remain data-dimensional by considering residual states and analytic auxiliary Gaussian scores. We also identify covariance and reverse-time degeneracies caused by shared Brownian factors and signed smooth-regime weights. The degeneracy motivates stabilized conditioning and, for stiff larger lifts, a Gaussian-bridge reconstruction sampler. Experiments on MNIST and CIFAR-10 show that persistent fractional perturbations with small Markovian lifts can improve score-based generation on MNIST and provide a promising extension to natural images, while the bridge sampler provides a stability mechanism for larger lifts.",
      "published": "2026-06-16T15:40:09Z",
      "abstract_url": "http://arxiv.org/abs/2606.18071v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18071v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Agentic AI-based Framework for Mitigating Premature Diagnostic Handoff and Silent Hallucination in Healthcare Applications",
      "authors": [
        "Divyansh Srivastava",
        "Shreya Ghosh",
        "Anshul Verma",
        "Rajkumar Buyya"
      ],
      "abstract": "Recent advances in Large Language Models (LLMs) and multi-agent systems have driven the rise of Agentic AI, showing promise for medical reasoning. However, open-ended conversational agents remain prone to two critical failure modes: premature diagnostic handoff and silent clinical hallucinations that may go undetected before reaching the patient. In this work, we propose a multi-agent framework that addresses both issues by replacing ``LLM-as-a-judge'' routing with deterministic orchestration constraints. The framework incorporates two safety mechanisms. First, a neuro-symbolic state-tracking gate enforces completeness of the OLDCARTS clinical protocol (Onset, Location, Duration, Character, Aggravating/Alleviating factors, Radiation, Timing, and Severity) by blocking diagnostic transitions until all required dimensions are collected. Second, an epistemic uncertainty quantification (UQ) gate computes semantic entropy (H) across K=5 independent diagnostic samples to identify and intercept divergent outputs before delivery. We evaluate the system using simulated patient agents powered by the llama-3.1-70b-instruct model on 150 test cases. The full architecture achieves 49.3% diagnostic precision, representing an absolute improvement of 11.3 percentage points over an unconstrained baseline. Additionally, we observe a statistically significant negative correlation (r = -0.181, p < 0.05) between OLDCARTS completeness (σ) and semantic entropy (H), suggesting that structured information gathering is associated with reduced diagnostic uncertainty.",
      "published": "2026-06-16T15:39:19Z",
      "abstract_url": "http://arxiv.org/abs/2606.18068v1",
      "pdf_url": "https://arxiv.org/pdf/2606.18068v1",
      "categories": [
        "cs.AI"
      ]
    }
  ]
};
