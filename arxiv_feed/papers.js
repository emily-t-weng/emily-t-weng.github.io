const PAPERS_DATA = {
  "last_updated": "2026-08-06 03:21:16 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "OctoLong: Mid-Training On Cross-Repository Code Contexts Enhances Long-Context Modeling",
      "authors": [
        "Indraneil Paul",
        "Falko Helm",
        "Goran Glavaš",
        "Iryna Gurevych"
      ],
      "abstract": "Context lengths of language models (LMs) have dramatically increased, driven by the demands for in-context learning, self-improvement, and long-horizon agentic workflows. Existing long-context corpora, however, are dominated by books, academic articles, and code repositories, which are finite resources and often scarce in long-distance dependencies. In this work, we introduce OctoLong, a context engineering pipeline that instruments an AST parser, a language server backend, and a package manager to facilitate the recursive retrieval of code references, enabling the curation of dependency-rich code contexts of millions of tokens in length. We then train OctoLong-Instruct, a suite of capable long-context open LMs, derived from base models ranging in size from 600M to 14B parameters, via context-extension mid-training on a ~50B-token mixture containing ~6.2B tokens of OctoLong code contexts, followed by ~10B tokens of instruction tuning. Our training ablations and experimental evaluations against 18 state-of-the-art open-weight long-context LMs show that supplanting just 12% of traditional context-extension corpora with OctoLong data yields substantial gains in long-range retrieval, long-term state tracking, repository-level code understanding, and downstream agentic tasks, while also enhancing API usage in short-context coding scenarios.",
      "published": "2026-08-05T17:58:15Z",
      "abstract_url": "http://arxiv.org/abs/2608.05141v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05141v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "OPD-V: Visual On-Policy Self-Distillation with Modality Balance",
      "authors": [
        "Aniri",
        "Jinhe Bi",
        "Peng Liao",
        "Zengjie Jin",
        "Volker Tresp",
        "Fei Shen",
        "Yunpu Ma",
        "Tat-Seng Chua"
      ],
      "abstract": "On-Policy Self-Distillation (OPSD) has become a standard post-training approach for improving visual reasoning in multimodal large language models (MLLMs). Existing methods draw privileged information from diverse input sources to guide self-distillation. Yet these designs overlook Modality Imbalance, a challenge inherent to MLLM reasoning. When textual information dominates generation, the model cannot fully integrate its multimodal input. Consequently, carefully designed privileged information remains underused, limiting the effectiveness of OPSD. To examine this limitation, we construct a Positive Teacher with the Zoom-In Image and a Negative Teacher with the Mask Image, which exhibit different degrees of Modality Imbalance. Changes in their reasoning correctness and token logits reveal that Modality Balance can itself serve as privileged information. Motivated by this finding, we introduce OPD-V, a visual OPSD paradigm that instantiates such information through the Positive Teacher and Negative Teacher. Positive Modality-Balance Logits Margins define a Modality-Balance Trust Region that selects the on-policy tokens used for self-distillation. Experiments across 6 benchmarks, 4 MLLM backbones, and 5 post-training methods show that OPD-V consistently improves reasoning performance while reducing training cost.",
      "published": "2026-08-05T17:53:06Z",
      "abstract_url": "http://arxiv.org/abs/2608.05131v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05131v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "SSTQ:Privacy-Preserving Vector Quantization via Subsampled Stochastic TurboQuant",
      "authors": [
        "Adel Javanmard",
        "David P. Woodruff",
        "Vahab Mirrokni"
      ],
      "abstract": "Achieving local differential privacy in distributed optimization while maintaining low communication cost remains challenging. Existing vector quantization methods, such as vqSGD, use high-dimensional geometric constructions but incur unfavorable dimension-dependent variance. In this work, we propose Subsampled Stochastic TurboQuant (SSTQ), a framework that combines overcomplete equal-norm tight frames, coordinate subsampling, and privacy-aware one-dimensional quantization. SSTQ includes two variants: a Flat Randomized Response version and a Metric-Aware Laplace version, the latter being better suited to higher codebook bit-width regimes. We show that SSTQ achieves optimal mean squared error scaling while using only $\\lceil \\log_2 N \\rceil + b$ bits per client, where $N = Θ(d)$ is the frame size. We also derive a surrogate privacy-aware codebook objective that reduces the codebook-dependent MSE scaling from $O(4^b)$ to $O(2^b)$. Finally, we empirically evaluate SSTQ against established baselines on federated learning tasks using CIFAR-10 and Fashion-MNIST, demonstrating favorable utility and communication efficiency.",
      "published": "2026-08-05T17:51:25Z",
      "abstract_url": "http://arxiv.org/abs/2608.05127v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05127v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Chained Recursive Language Models for Multi-Iteration Reasoning",
      "authors": [
        "Purbesh Mitra",
        "Sennur Ulukus"
      ],
      "abstract": "Long context reasoning in large language models (LLMs) is usually constrained by the fact that a single inference trajectory has to simultaneously explore the context, store intermediate state, verify evidence, and produce the final answer. This becomes particularly difficult in tasks that require extraction, counting, ordering, or multi-hop reasoning, where an early mistake can propagate until the final response. In this work, we propose Chained Recursive Language Models (Chained RLM), an inference-time architecture, in which the same underlying model is called repeatedly as a sequence of fresh reasoning roots. Each root receives the original problem and context, but does not inherit the full conversational history. Instead, it receives a compact plain-text summary, a plain-text blackboard, and some durable task-specific artifacts written by predecessor roots. The motivation is to manage the context by chopping into partial tasks rather than one large inference response; in each staged computation, intermediate artifacts can be inspected, corrected, and extended by a later fresh inference by the same model. We describe the system model, handoff mechanism, artifact workspace, and evaluation protocol for this system. We study when fresh-context artifact continuation gives a measurable gain in accuracy over direct LLM answering even with recursive tool-calling.",
      "published": "2026-08-05T17:50:08Z",
      "abstract_url": "http://arxiv.org/abs/2608.05124v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05124v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IT",
        "cs.LG",
        "eess.SP"
      ]
    },
    {
      "title": "Robust and Efficient Motion Reasoning for Privacy-Aware Classroom Incident Recognition",
      "authors": [
        "Paritosh Parmar",
        "Landy Lan",
        "Hong Yang",
        "Chen Yi",
        "Chiat Pin Tay"
      ],
      "abstract": "Can computer vision help make classrooms safer? In this pilot study, we investigate privacy-aware and computationally efficient classroom incident recognition from CCTV-style observations. This setting remains underexplored, with limited benchmarks and few methods designed for the privacy, efficiency, and generalization demands of real-world deployment. We introduce a novel hybrid benchmark combining generative CCTV-style videos with real-world classroom pose data, and propose a lightweight, but robust motion-reasoning framework motivated by the observation that many incidents differ more in motion direction, speed, acceleration, and intensity than in pose alone. To that end, our method first constructs hierarchical kinematic representations of human actions. Our method then distills hierarchical, multi-order kinematic reasoning from a large teacher into a much smaller single-order student, enabling efficient per-person inference while preserving expressive motion understanding. Experiments show that our model outperforms substantially larger baselines at less than one-tenth of their computational cost, while also demonstrating stronger out-of-domain motion reasoning and zero-shot synthetic-to-real generalization. We will publicly release the benchmark, codebase, and supporting tools to facilitate further research in privacy-aware classroom safety.",
      "published": "2026-08-05T17:46:28Z",
      "abstract_url": "http://arxiv.org/abs/2608.05115v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05115v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.ET",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "Representational separation between unitary and channel quantum generative models via shared classical randomness at shallow depth",
      "authors": [
        "Arunava Majumder",
        "Marius Krumm",
        "Hendrik Poulsen Nautrup",
        "Hans J. Briegel"
      ],
      "abstract": "Near-term quantum hardware limits circuit depth and often imposes geometrically local connectivity for quantum generative models, restricting the output distributions accessible to shallow unitary Born models. Introducing stochasticity into a unitary quantum Born model can improve the empirical generative performance of the resulting channel model and, for a restricted small-scale architecture, has been proven to represent a strictly larger family of distributions than its unitary counterpart. However, whether such randomness provides a provable separation at fixed shallow depth for arbitrarily large systems has remained open. Here, we show that shared classical randomness, a comparatively weak resource from entanglement theory, is sufficient to establish such a strict scalable representational separation over the corresponding shallow unitary Born model. More specifically, we augment bounded-connectivity shallow unitary circuits, followed by computational-basis measurements, with spatially separated local Pauli operations, whose joint application is controlled by a single classically sampled random bit. The resulting shallow-depth channel model generates long-range correlations in the classical output distribution that no purely unitary shallow-depth model with bounded connectivity can reproduce. For one-dimensional nearest-neighbour architectures, reproducing such distributions with a purely unitary model can require depth $Ω(N)$ in the worst case. We further show that measurement-based quantum computation (MBQC) provides a natural implementation of the required shared classical randomness through suitable adaptation of the random measurement outcomes. Numerical experiments on MBQC-based generative models support the analytical results.",
      "published": "2026-08-05T17:44:36Z",
      "abstract_url": "http://arxiv.org/abs/2608.05110v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05110v1",
      "categories": [
        "quant-ph",
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "Capability-Gated Planning: Cost-to-Goal Discovery and the Limits of Myopic Experiment Selection",
      "authors": [
        "Ahmed Hassoon",
        "Mark Dredze"
      ],
      "abstract": "Systems that automate scientific discovery must repeatedly decide which experiment to run, which hypothesis to test, which tool to build, and when to stop. Many systems make these decisions by maximizing a myopic score such as expected information gain per unit cost or a learned plausibility score. We identify a structural limitation of this approach. Some actions are constructive: they acquire an epistemic capability (an instrument, assay, pipeline, simulator, or abstraction) whose value lies not in the information returned immediately but in the future actions it makes available. When the least-cost route to a confident answer requires a chain of such constructions, a planner that scores actions only by information obtainable within a bounded horizon cannot value the first construction: it yields no information within the horizon and is dominated by any measurement with positive information, however small. We formulate goal-directed discovery as a stochastic shortest-path problem in belief space in which constructive experiments change the downstream action graph, and prove that for every lookahead depth d there is an instance on which every myopic information-maximizing planner has an unbounded approximation ratio, and a related instance on which it never reaches the goal. The mechanism is a capability-indistinguishability lemma: within the horizon, acquiring a capability can be observationally indistinguishable from paying for a null action. This establishes capability gating as a reachability axis of difficulty distinct from curvature (submodularity) and information order (adaptivity gaps). We introduce CG-Plan, an incremental replanner with a capability-aware cost-to-go heuristic h = h_cap + h_exp. In a controlled testbed, the performance gap appears only under gating, persists for every fixed horizon, and arises when near-miss hypotheses come from a data-consistent proposer.",
      "published": "2026-08-05T17:25:17Z",
      "abstract_url": "http://arxiv.org/abs/2608.05085v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05085v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MultiPathFormer: Towards a Foundation Model for Multipath Wireless Propagation",
      "authors": [
        "Blessed Guda",
        "Kayley Sze",
        "Carlee Joe-Wong"
      ],
      "abstract": "Recent advances in machine learning have enabled training of wireless foundation models, which aim to support tasks such as channel estimation, beam prediction, and localization based on wireless signals. Existing wireless foundation models typically pretrain on channel tensors using masked reconstruction over subcarriers, antennas, or time but ignore the physical characteristics of wireless propagation. In this work, we propose to instead use multipath propagation as the fundamental pretraining object. We present MultiPathFormer, an autoregressive foundation model that represents each transmitter-receiver link as an ordered sequence of continuous-valued path tokens and pretrains with next-path prediction. We introduce an Environmental RAG (retrieval-augmented generation) mechanism and a first-path codebook on top of the transformer backbone, leveraging environment knowledge to improve path statistics estimation like delay and power by up to 59%. MultiPathFormer pretrained on 27 environments transfers to unseen users and, after scenario-specific fine-tuning, outperforms training the corresponding models from scratch in new environments. Across downstream tasks, it outperforms SOTA channel-based foundation models, achieving 5.57 m mean localization error, 0.914 top-3 beam accuracy, 0.994 line-of-sight classification accuracy, and 0.561 channel estimation NMSE. These results show that path-level pretraining can learn reusable representations of wireless propagation.",
      "published": "2026-08-05T17:20:16Z",
      "abstract_url": "http://arxiv.org/abs/2608.05076v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05076v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SP"
      ]
    },
    {
      "title": "Provable Limits and Certified Deferral for Verbalized Uncertainty in Small Language Models",
      "authors": [
        "Jianru Shen"
      ],
      "abstract": "Small open-weight language models increasingly run in private, offline, and cost-sensitive settings, where the key deployment question is not only what a model answers but when it should defer to a human. We study whether verbalized confidence can support risk-controlled deferral, evaluating eleven instruction-tuned models from three families, 0.5B to 14B parameters, on ARC-Challenge and TruthfulQA with 25,168 local predictions. Three theoretical results delimit what calibration can provide: strictly monotone calibration preserves the risk-coverage frontier and error-detection AUROC; temperature scaling cannot calibrate models whose confidence stays above one half while accuracy falls below it; and a Clopper-Pearson procedure converts a 200-question calibration set into a finite-sample risk certificate under an i.i.d. deployment assumption. Empirically, eight of 22 model-task pairs hit the temperature-scaling infeasibility floor within one percentage point of the predicted bound. Platt scaling reduces ECE to as low as 0.02, yet certified autonomy at a 20% risk budget is granted to only three model-task pairs and to none at 10%. We also identify and repair an answer-ordering artifact in the multiple-choice form of TruthfulQA. Calibration gives confidence semantics; certified deferral determines when small models are safe to use.",
      "published": "2026-08-05T17:11:04Z",
      "abstract_url": "http://arxiv.org/abs/2608.05064v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05064v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Hardware Design and Security in the Era of Chiplets and LLMs",
      "authors": [
        "Johann Knechtel",
        "Ozgur Sinanoglu",
        "Paul V. Gratz",
        "Ramesh Karri"
      ],
      "abstract": "The semiconductor industry is undergoing a dual revolution: the shift toward heterogeneous 2.5D chiplet systems and the integration of Large Language Models (LLMs) into Electronic Design Automation (EDA) flows. While these paradigms offer unprecedented benefits in yield, modularity, design productivity, etc., they radically expand the hardware attack surface. This paper provides a unified analysis of these frontiers, ranging from attacks on chiplet systems (including hardware stacks for LLM acceleration) across architectural, logical, and physical levels, to various exploits against LLM-driven EDA pipelines. To secure chiplet systems, we review a powerful defense approach that leverages 2.5D split manufacturing and active interposers for physically isolated Root of Trust (RoT) architectures. To secure LLM-driven EDA pipelines, we first identify native threats and then review state-of-the-art defense techniques. Finally, we discuss how LLM systems can advance hardware security efforts for modern systems, including chiplets.",
      "published": "2026-08-05T17:09:13Z",
      "abstract_url": "http://arxiv.org/abs/2608.05063v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05063v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.AR"
      ]
    },
    {
      "title": "MarsCast: Transfer Learning of AI Weather Foundation Models to Planetary Atmospheres",
      "authors": [
        "M. L. Carroll",
        "J. Li",
        "S. D. Guzewich",
        "G. Villanueva",
        "J. A. Caraballo-Vega",
        "M. J. Frost"
      ],
      "abstract": "We investigate the transferability of Earth weather foundation models to planetary atmospheres by adapting the GraphCast graph neural weather forecasting model to Mars. While GraphCast achieves state-of-the-art performance for terrestrial forecasting, its applicability to non-Earth environments remains unexplored. Using the Mars Climate Database (MCD), which provides global atmospheric fields across vertical altitude levels (similar to Earth pressure levels), we evaluate zero-shot and fine-tuned GraphCast predictions of Martian temperature and wind fields. Zero-shot forecasts produce a surprisingly accurate depiction of current conditions but fail to reproduce diurnal variability and rapidly decay toward climatological mean states. To address this limitation, we fine-tune GraphCast using MCD variables and top-of-atmosphere solar radiation forcing while holding humidity constant. Fine-tuning enables rapid learning of Martian thermal variability. Within as few as 10 training epochs, the model begins to capture the diurnal cycle and forecasts up to 10 days reproduce seasonal and vertical temperature structure. Prediction quality improves with training sample size and exhibits sensitivity to seasonal initialization. These results demonstrate that Earth-trained AI weather models can be adapted to simulate Martian atmospheric dynamics, providing a pathway toward rapid planetary weather prediction to support mission operations, dust storm risk mitigation, and future human exploration.",
      "published": "2026-08-05T17:03:13Z",
      "abstract_url": "http://arxiv.org/abs/2608.05054v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05054v1",
      "categories": [
        "astro-ph.EP",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "The Effect of Perceived Race and Gender on Police Language Use: Experimental Evidence from VR Simulations",
      "authors": [
        "Sandra C. Sandoval",
        "Navita Goyal",
        "Rashawn Ray",
        "Long Doan",
        "Rachel Rudinger",
        "Hal Daumé"
      ],
      "abstract": "Against the backdrop of violence in police interactions with the U.S. public, we explore how deferentially police officers speak to virtual characters depicted as Black adult males in vir- tual reality (VR) simulations. We evaluate the effect of seeing and communicating with these characters through a causal in- ference lens, where the assignment of the Black man character to a police officer and simulation is the treatment variable. Our (marginal) average treatment effect AT E measures the social impact of the character on the deference of officer statements with each turn of the conversation. Soberingly, we find that most officers speak less deferentially to Black man characters, except for White, biracial, and multiracial female officers, es- pecially in settings where the VR character was known to be a suspect. Across a full conversation of a typical VR scene, these marginal AT Es can result in notable changes in def- erence of tone (two to several points difference on a scale of 0-10), above and beyond that due to the initial effect of per- ceiving a Black male character. Even more disconcerting is that this can contribute to conversation breakdowns that po- tentially result in violence or danger to both the public and the police. We also explored the capabilities of large language models (LLMs) for ATE estimation. From our methods com- parison analysis, including model validation against synthetic data, we provide unique scientific insights on LLM-assisted methodologies for ATE estimation. As such, for ATE esti- mation with multilevel data with text, we recommend mixed effects models with the inverse propensity treatment weighted (iptw) approach, which utilized an LLM for text feature cre- ation. While we also tested LLMs for finetuning prediction models ultimately for ATE estimation, we conclude they are an area for further development and refinement.",
      "published": "2026-08-05T16:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2608.05050v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05050v1",
      "categories": [
        "cs.CY",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Gradient Immunity: Null-Space Resistance to Malicious Fine-Tuning",
      "authors": [
        "Yuxuan Huang",
        "Xingyu Zeng",
        "Tianhang Zheng",
        "Chaochao Lu"
      ],
      "abstract": "Released aligned large language models remain vulnerable to malicious downstream finetuning. Existing defenses are largely designed for the fine-tuning-as-a-service (FTaaS) paradigm or rely on downstream users to follow additional safety procedures, and therefore do not directly address the setting we study: a provider controlled partially protected open-weight (PPOW) release setting in which most weights remain trainable while a small safety-critical component is preserved at release. We propose a Unidirectional Safety Gate (USG), instantiated as a Null Space Cubic Layer together with an Inverse Adapter inserted after the final Transformer layer. During downstream fine-tuning, the cubic layer suppresses or blocks gradients from harmful samples whose hidden states fall in a calibrated protected region, while the Inverse Adapter restores the base model's forward behavior. In practice, we calibrate a threshold using defender-held harmful data, allowing protection to generalize to nearby in-distribution harmful samples. Across six evaluated model-dataset settings, USG keeps post-finetuning attack success rate close to the pre-release level under a fixed release threshold, while maintaining high safe-pass rates on easier settings and exhibiting a clearer safety-utility trade-off on unsafe samples from BeaverTails. These results suggest that release-time representation-space blocking can raise the cost of malicious downstream adaptation without requiring downstream cooperation. The code is available at https://github.com/OpenCausaLab/Gradient-Immunity.",
      "published": "2026-08-05T16:55:59Z",
      "abstract_url": "http://arxiv.org/abs/2608.05045v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05045v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "From Score Matrices to Football-Aware Match-State Simulation: An Auditable LLM Harness for Exact-Score Reranking",
      "authors": [
        "Shaopeng Liang"
      ],
      "abstract": "Football score forecasting combines a strong statistical core with a difficult contextual edge. Dynamic Poisson-family models estimate team strength, expected goals, and coherent score probabilities, but do not directly understand roles, tactical matchups, motivation, or how a first goal changes behaviour. Large language models (LLMs) can reason about such concepts, yet are not calibrated probability engines. We combine both components through an auditable information harness. This paper documents four iterations: V1, a dynamic score-driven Dixon-Coles baseline; V2, which maps LLM contextual ratings back into expected-goal parameters; V3, which replaces scalar correction with goal-by-goal simulations over a frozen score-candidate set; and V4, which adds shared first-breakthrough and post-goal cascade judgments, time-aware stopping, and deterministic tail candidates. The harness defines input semantics, supplies pre-match evidence, and constrains the LLM to an inspectable reasoning route. On a chronological replay of the first 150 matches of the 2025-26 English Premier League, V1 achieved 10.0% Top-1 and 26.7% Top-3 exact-score accuracy. V3 reached 12.0% and 30.0%, while V4 reached 14.7% and 30.7%. V4 increased candidate coverage from 77.3% to 84.7%, although no added tail candidate became a Top-3 exact hit. V1's native 1X2 distribution achieved 53.3% argmax accuracy, 0.9878 log loss, 0.5870 Brier score, and 0.2095 ranked probability score. These results are exploratory: the development slice is not an untouched benchmark, and temporal input isolation cannot exclude outcome memory in a closed LLM. The contribution is an auditable hybrid architecture, a clear design evolution, and negative findings showing where football-aware simulation does and does not improve score selection.",
      "published": "2026-08-05T16:34:53Z",
      "abstract_url": "http://arxiv.org/abs/2608.05030v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05030v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Short-term load forecasting under EU-AI Act Requirements in Safety-Critical Environments: Results from a 41-day live challenge on the aggregated German transmission-grid load",
      "authors": [
        "Thomas Bartz-Beielstein"
      ],
      "abstract": "Short-term load forecasting (STLF) play a vital role in the electric power industry. It serves infrastructure that European and German law designate as critical. Determinism, reproducibility, and auditability are engineering requirements rather than optional extras. STLF is no longer purely an accuracy problem. It is also a software-engineering and compliance problem. This paper describes results from a 41-day live challenge that evaluated a complete STLF pipeline for the aggregated German transmission-grid load. The pipeline is based on the open-source Python library spotforecast2-safe, which implements the EU-AI Act Requirements in Safety-Critical Environments by design. The pipeline predicts the 24 hourly load values of a target day from European Network of Transmission System Operators for Electricity (ENTSO-E) data. It includes anomaly detection and gap-aware data preparation, calendar and weather covariates, a recursive multi-step forecasting algorithm, and hyperparameter tuning. Forecast accuracy is measured against the official ENTSO-E day-ahead forecast. The EU-AI act compliant spotforecast2-safe pipeline beats the ENTSO-E baseline. In-context models show competitive performance. Transparent, low-cost, and auditable local models (referred to as macl2l in this paper) are competitive with more than 100-million-parameter large, energy-intensive pre-trained foundation models such as chronos-2. The challenge infrastructure, the complete submission history of all teams, and the frozen final leaderboard are publicly available.",
      "published": "2026-08-05T16:23:31Z",
      "abstract_url": "http://arxiv.org/abs/2608.05018v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05018v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Revealed Rationality: Label-Free Evaluation and Regularization from Representation Theorems",
      "authors": [
        "Isaiah Andrews"
      ],
      "abstract": "Representation theorems in decision theory establish that behavior satisfies certain axioms if and only if it can be rationalized by a well-defined objective. I argue that this ``if and only if'' structure provides a potentially useful foundation for label-free evaluation and regularization of LLMs and other AI systems. Axiom compliance can be checked from the model's own responses to synthetic choice problems, with no external labels or human feedback, and the penalties are readily computable. Because the axioms are necessary and sufficient, the resulting checks exhaust the implications of the relevant rationality standard for the elicited data: a model that passes cannot be rejected on rationality grounds by any further test of the same data. I discuss three instantiations: probabilistic coherence via a theorem of de Finetti, preference rationality via Afriat's theorem, and subjective expected utility via a theorem of Echenique and Saito (2015), each yielding a continuous penalty that is zero whenever behavior can be rationalized. Since coherence does not restrict which objective rationalizes behavior, these penalties complement rather than replace other evaluation and training signals.",
      "published": "2026-08-05T16:21:18Z",
      "abstract_url": "http://arxiv.org/abs/2608.05015v1",
      "pdf_url": "https://arxiv.org/pdf/2608.05015v1",
      "categories": [
        "econ.TH",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ORACLE: A Multi-Objective Reinforcement Learning-Based Analog Circuit Design Optimizer with Large Language Models-Guided Exploration",
      "authors": [
        "Osei Brempong",
        "Mohammed Ayman Habib",
        "Vivan Poddar",
        "Morteza Fayazi"
      ],
      "abstract": "Analog circuit design automation using reinforcement learning (RL) has emerged as a promising approach for reducing manual effort. However, many existing RL-based methods focus on single-objective optimization. Even methods designed for multi-objective (MO) problems often reduce multiple design specifications to a single scalar reward. This simplification limits the ability to capture the true Pareto trade-off among competing objectives and often leads to suboptimal designs. Moreover, requiring the model to be retrained from scratch whenever the desired MO specifications change remains a key limitation. To address these challenges, we present ORACLE, an open-source RL-based framework for MO analog circuit design optimization that replaces scalar reward optimization with vector-valued learning and preference-aware conditioning. ORACLE represents a true MO analog circuit design optimizer that uses a preference vector to specify the relative weights of multiple objectives, enabling a single trained model to generate designs across diverse trade-off settings without retraining. We further propose two preference-guidance strategies, namely normalized-weight guidance and cosine-aligned guidance, to improve convergence. In addition, we incorporate a large language model (LLM)-guided action selection mechanism to filter actions that are likely to lead to suboptimal designs or increased runtime. Our results show that, on multiple circuit topologies with 2,000 test cases, ORACLE reduces runtime by 20.4x - 104.4x compared to state-of-the-art approaches. It also meets 99.9% of the 2,000 target specifications, and achieves 5.1x - 318.6x better figure of merit in the resulting output specs.",
      "published": "2026-08-05T16:09:12Z",
      "abstract_url": "http://arxiv.org/abs/2608.04999v1",
      "pdf_url": "https://arxiv.org/pdf/2608.04999v1",
      "categories": [
        "eess.SY",
        "cs.AI"
      ]
    },
    {
      "title": "Protoreasoning in Tiny Transformers",
      "authors": [
        "Eduardo Valle",
        "Fergal Reid"
      ],
      "abstract": "We show that tiny transformers can profitably employ a simple form of Chain of Thought, which we call protoreasoning, allowing us to study step-by-step reasoning on ~1M-parameter models and opening up opportunities for much more detailed experimentation and analysis than is feasible for larger models. Current Large Language Models exhibit impressive step-by-step reasoning, but we have yet to understand its generality, i.e., when and how LLMs learn genuinely general algorithms rather than \"bags of heuristics.\" Such questions are hard to settle on compute-intensive frontier models trained on opaque data. To work at model scales far below the threshold for natural-language competence, we define reasoning-friendly tasks on Dyck languages (sentences of correctly nested brackets). We find that protoreasoning traces substantially close the out-of-distribution generalization gap, and ablations confirm that the trace's content, not merely its extra tokens, drives the gain.",
      "published": "2026-08-05T15:51:36Z",
      "abstract_url": "http://arxiv.org/abs/2608.04980v1",
      "pdf_url": "https://arxiv.org/pdf/2608.04980v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models",
      "authors": [
        "Bohai Gu",
        "Yueyang Yuan",
        "Taiyi Wu",
        "Dazhao Du",
        "Jian Liu",
        "Xiaoyi Pang",
        "Jie Zhang",
        "Xiaocheng Lu",
        "Haobin Zhong",
        "Xiaotong Zhao",
        "Alan Zhao",
        "Song Guo"
      ],
      "abstract": "Interactive video world models are essential for long-horizon planning and exploration, yet they suffer from compounding errors. Post-training methods such as reinforcement learning (RL) can improve these models, but they hit a verification bottleneck: for arbitrary action sequences, no ground-truth future state exists to measure long-term drift. Our key insight is that reversible action cycles make this verification possible: a sequence composed with its inverse must analytically return to the initial state, yielding annotation-free supervision on long-horizon correctness. Building on this, we introduce WorldCycle, a self-verifiable RL framework that constructs closed action cycles and their repeated executions from ordinary action sequences, and optimizes two complementary rewards: a spatial closure reward enforcing symmetry between mirrored forward and reverse segments, and a temporal consistency reward aligning states across repeated cycle executions. These rewards force the model to learn actions as consistent state operators rather than memorized temporal patterns, and extend naturally to out-of-distribution composite action cycles that the base model handles poorly. We further release CycleBench, a diagnostic benchmark for state-returning ability under complex action structures. WorldCycle reduces state returning drift by up to 44% and lifts composite-action accuracy nearly 4x over the base model, providing a vital foundation for physically grounded world models.",
      "published": "2026-08-05T15:34:47Z",
      "abstract_url": "http://arxiv.org/abs/2608.04964v1",
      "pdf_url": "https://arxiv.org/pdf/2608.04964v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "CheMLFlow: An Open-Source Platform for Cheminformatics and Materials Informatics Applications",
      "authors": [
        "Brendan Smith",
        "Susana Lopez-Moreno",
        "Eric Dolores-Cuenca",
        "Sangil Kim",
        "Jose L. Mendoza-Cortes",
        "Nijamudheen Abdulrahiman"
      ],
      "abstract": "CheMLFlow is an open-source platform for building and executing end-to-end, high-throughput, and agentic workflows for scientific and technological applications. CheMLFlow targets a common bottleneck in scientific machine learning development, where researchers often need to assemble data acquisition, curation, representation, model training, validation, screening, interpretation, and reporting into a reproducible pipeline, even when their primary research contribution concerns only one stage. CheMLFlow provides modular workflow components, ready-to-run reference pipelines, standardized artifacts, and evaluation outputs that reduce orchestration overhead and support benchmarking across methods and datasets. The platform is designed to be extensible, reproducible, and automation friendly, with pluggable representations and models, deterministic splits, explicit run artifacts, batch execution, and report generation. As scientific software increasingly moves toward agent assisted experimentation, CheMLFlow's configuration driven workflows and structured outputs also provide a practical interface for coding agents to help users construct experiments, inspect results, and summarize findings under human supervision. This article describes the system architecture, core workflows, and benchmarks that reach literature performance for quantum mechanical, physicochemical and bioactivity property prediction, and use cases involving time series datasets demonstrating applications beyond molecular chemistry datasets.",
      "published": "2026-08-05T15:08:29Z",
      "abstract_url": "http://arxiv.org/abs/2608.04942v1",
      "pdf_url": "https://arxiv.org/pdf/2608.04942v1",
      "categories": [
        "cs.LG",
        "cond-mat.mtrl-sci",
        "cond-mat.other",
        "cs.AI",
        "physics.chem-ph"
      ]
    }
  ]
};
