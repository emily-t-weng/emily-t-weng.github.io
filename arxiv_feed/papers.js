const PAPERS_DATA = {
  "last_updated": "2026-08-31 05:09:53 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning",
      "authors": [
        "Nan Wang",
        "Mohit Yadav",
        "Jonathan Wulff",
        "Aidan Rosenbaum",
        "Kezhou Chen",
        "Yuvan Sharma",
        "Xu Dong",
        "Yiwei Tao"
      ],
      "abstract": "Tendon-driven hands are anthropomorphic, and moving the actuators off the joints is what makes a hand of this capability affordable to build. Two effects produce that saving. Routing force through a cable removes the requirement that a motor fit inside the joint it drives, so smaller and cheaper motors suffice, and one motor can drive several joints through a single cable, so fewer motors are needed. They are also harder to learn on than a direct-drive hand. The underactuated transmission that produces the saving is itself difficult to represent in a simulator, and the joints one cable drives are not independently commandable. We present Aero Hand Open, a tendon-driven anthropomorphic hand that is released simulation-ready. Three things ship with it. A simulation model reproduces the cable transmission itself. An identified actuation map connects that model to the motor commands in both directions, including the three-way coupling of the thumb. A reinforcement learning package trains policies for the hand. Together they let a policy be trained entirely in simulation and run on the hand with no fine-tuning and no state estimation. We release the mechanical design, the simulation model, the identified mapping, the training environment and the deployment stack.",
      "published": "2026-08-28T17:53:48Z",
      "abstract_url": "http://arxiv.org/abs/2608.28578v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28578v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning a Size-Weight Frontier for Synthetic-Augmented Inference",
      "authors": [
        "Chengpiao Huang",
        "Kaizheng Wang"
      ],
      "abstract": "Synthetic data can improve statistical inference when real data are scarce, but naively treating synthetic samples as real data can introduce bias and lead to unreliable inference. We develop a general framework for synthetic-augmented inference across a population of related tasks. It characterizes synthetic augmentation by the number of synthetic observations and their weight. Central to our framework is a size-weight frontier that specifies, for each weight, the largest synthetic sample size for which all smaller sizes attain the target task-marginal coverage. We estimate this frontier from historical tasks, and establish a finite-sample coverage guarantee simultaneously for all size-weight configurations on or below the estimated frontier. In experiments using large language model responses to augment opinion survey data, our procedure achieves target coverage and substantially narrows confidence intervals.",
      "published": "2026-08-28T17:52:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.28576v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28576v1",
      "categories": [
        "stat.ME",
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "Blog: Survey of Optimizers",
      "authors": [
        "Ruoran Xu"
      ],
      "abstract": "Neural-network optimization in 2025-2026 is no longer well described as a succession of new Adam variants. The design space has expanded from coordinates to matrices and layers, from fixed training horizons to policies over time, and from mathematical update rules to state representations that must survive sharding and low-precision computation. This survey organizes recent optimizers and training optimization methods along four largely independent axes: temporal estimation, update geometry, horizon management, and representation and systems. It connects the spectral normalization of Muon, the historical matrix statistics of Shampoo and SOAP, adaptive and hybrid matrix methods, memory-efficient optimizers, schedule-free training, small-batch corrections, and quantized optimizer states. The central empirical conclusion is deliberately non-triumphal: matrix-aware methods represent a genuine advance, but there is no context-independent replacement for AdamW. Rankings change with model scale, data-to-parameter ratio, batch size, schedule, parameter partition, tuning budget, and whether the target metric is tokens, FLOPs, wall-clock time, or memory. The practical consequence is a compositional view of optimizer design and a stricter protocol for evaluating optimizer claims.",
      "published": "2026-08-28T17:35:11Z",
      "abstract_url": "http://arxiv.org/abs/2608.28557v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28557v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "An Enclosed Mode Is a Gauge Choice: Topology Relative to Reach in Certified Code World Models",
      "authors": [
        "Javier Aguilar Martín"
      ],
      "abstract": "A code world model accepted by a sampling gate can be exactly right on everything the gate can see and arbitrarily wrong beyond it. We characterize what a certified model can know, and what its errors can cost, when the omission is an annular freeze mode enclosing an unreachable interior. The gate quotient makes the question precise: acceptance-with-certainty determines the model exactly on the reachable query set; beyond reach is gauge. On a minimal ring instrument we prove the extreme case (a wrong-topology filled-disc artifact unfalsifiable by any sampling gate and bitwise harmless at play) and measure, with LLM synthesis across three model families, how one knob (a channel of width gamma) walks the same artifact through three regimes: unfalsifiable-and-harmless, falsifiable-and-costly, and instantly falsified. Three principles organize the empirics. First, danger is topology relative to reach: a channel the planner can use collapses the blind model's exploitation (play cost 1.09 to ~0 over a knee at gamma ~ 0.1), while a hidden channel with the same first Betti number keeps it at full strength (1.12). Second, repair is parameter-bound and sensor-bound: no family recovers the region from outside evidence; from inside, models pose the right topology but cannot pin its parameters, and the posed topology tracks the guiding persistent-homology summary's wrong beta_1 (a sensor with a measured geometric resolution limit), not the truth. Third, mitigation must match the error's dimension and direction: point fences fail against the one-dimensional boundary, a dimension-matched persisted fence collapses exploitation to a two-lesson transient (0.999 to 0.058), and the dual freedom certificate collapses the invented-mode failure symmetrically (1.769 to 0.029). In n dimensions the shell makes misidentification near-certain while the danger stays fully exploitable: the two axes are independent.",
      "published": "2026-08-28T17:14:58Z",
      "abstract_url": "http://arxiv.org/abs/2608.28541v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28541v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LLM-Based Agents for Software and Systems Security: Approaches, Applications, and Assessment",
      "authors": [
        "Jingjing Nie",
        "Jiawei Guo",
        "Krishna Meda",
        "Haipeng Cai"
      ],
      "abstract": "Software and systems security workflows are typically procedural: analysts inspect heterogeneous artifacts, form hypotheses, invoke tools, interpret outputs, and revise plans. Large language model (LLM)-based agents, which can plan, use tools, retain state, and revise actions across multi-step workflows, are being rapidly adopted to automate this work. Given the consequences of delegating security decisions to autonomous systems, understanding how such agents are built, used, and assessed is crucial. Yet to this date, there remains a lack of systematic understanding of what has been done and how far we are in this field: the term \"agent\" is applied inconsistently, applications differ sharply in risk, and assessment protocols are often incomparable. To gain a comprehensive and coherent view of this area hence inform relevant future research, this paper provides a systematic literature review of the (1) technical approaches, including agent architecture, perception, memory, reasoning and planning, action space, orchestration, and self-improvement, (2) applications, with respect to the security tasks served, and (3) assessment, including the datasets, outcome and trajectory metrics, safety measures, and baselines considered, over the peer-reviewed literature spanning the emergence of this area (2023--2026). Our synthesis reveals a field that has built agents able to act but not yet agents whose authority is bounded or whose behavior is auditable. In addition to knowledge systematization, we also extend our insights into the limitations of and challenges faced by current approach, application, and assessment designs, which shed light on potentially promising future research directions.",
      "published": "2026-08-28T16:19:01Z",
      "abstract_url": "http://arxiv.org/abs/2608.28490v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28490v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "How Proper Scoring Rules Shape LLM Forecasting",
      "authors": [
        "Benjamin Turtel",
        "Paul Wilczewski",
        "Kris Skotheim",
        "Ville A. Satopää",
        "Philip E. Tetlock"
      ],
      "abstract": "This paper evaluates how reward function choice shapes the performance and behavior of LLM forecasters. We compare five proper scoring rules as training objectives for binary forecasts of resolved real-world events. Although the rules share the same theoretical incentive for truthful probability reporting, the resulting models differ in calibration, probability use, and estimated profiles of bias, information, and noise, with smaller differences in aggregate accuracy and discrimination. The Brier-trained model has the lowest observed Brier score and highest AUC-ROC, while the log-trained model has the highest observed log score and lowest calibration error. Models with similar aggregate performance also reach that performance through different combinations of bias, information, and noise. Proper scoring rules therefore need not behave interchangeably as training objectives. Reward choice may shape not only how well an LLM forecasts, but how its forecasting errors are structured. Each condition uses a single seed, so some differences may reflect training stochasticity.",
      "published": "2026-08-28T16:08:51Z",
      "abstract_url": "http://arxiv.org/abs/2608.28482v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28482v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry",
      "authors": [
        "Samuel Xiao",
        "Judy Song",
        "Rory Hu",
        "Ziliang Zong"
      ],
      "abstract": "Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and mathematical reasoning. However, their ability to translate informal mathematical problems into formal representations remains underexplored. This limitation is particularly important for neuro-symbolic geometry systems such as AlphaGeometry, whose theorem-proving engine requires inputs in a specialized domain-specific language (DSL). Although AlphaGeometry achieves near-IMO gold-medalist performance, manually converting natural-language problems into its formal syntax remains a significant usability bottleneck. To address this challenge, we introduce the Natural Language to AlphaGeometry Benchmark (NL2AGBench), which evaluates LLMs in translating English geometry problems into AlphaGeometry-compatible formal representations. NL2AGBench uses execution-based verification within AlphaGeometry to assess translation quality rather than relying solely on textual similarity. We evaluate ten state-of-the-art open- and closed-source LLMs across multiple parameter scales and analyze executable translation accuracy, syntactic correctness, and error characteristics. Our experiments reveal a substantial performance gap between closed- and open-source models: leading closed-source models achieve executable translation rates above 80%, while even the largest open-source models struggle to consistently preserve geometric constraints and produce valid formalizations. We introduce an error taxonomy distinguishing syntax and logic errors and investigate mitigation strategies, including few-shot prompting, fine-tuning, and human-guided hinting, which yield measurable improvements across multiple model families.",
      "published": "2026-08-28T16:07:16Z",
      "abstract_url": "http://arxiv.org/abs/2608.28481v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28481v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "ARC-CT: Anatomy-Routed Contrastive Vision-Language Learning for 3D Chest CT",
      "authors": [
        "Huseyin Umut Isik",
        "Mehmet Alp Ozaydin",
        "Sila Kurugol",
        "Şeyda Ertekin"
      ],
      "abstract": "Contrastive vision-language learning uses paired chest CT volumes and radiology reports to learn abnormality classifiers without manually annotated labels. However, two characteristics of chest CT challenge conventional global contrastive learning. First, many critical abnormalities are small or anatomically localized, and pooling an en- tire volume into a single embedding may dilute their visual evidence. Second, the standard contrastive objective treats every other scan in a batch as a negative. Because many chest CTs share abnormalities, this objective incorrectly pushes co-positive pairs apart. We propose Anatomy-Routed Contrastive Learning for 3D Chest CT (ARC-CT), a region-aware framework that addresses these limitations using only la- bels extracted from reports by an LLM, with no manual annotations or bounding boxes. ARC-CT combines three components: (1) an Anato- myQFormer localizing evidence via queries constrained by automatically generated organ masks; (2) a label-Jaccard soft InfoNCE objective in- tegrating the standard one-hot target with the label-set overlap of each pair, which reduces false-negative penalties between studies that share clinical findings; and (3) an organ-level alignment loss connecting mask- pooled visual features to organ-specific report text extracted offline with a large language model. ARC-CT achieves a 0.86 mask-free macro AUC across 18 abnormalities using a compact 3D ResNet-18 backbone. Over- all, ARC-CT outperforms both comparable efficient baselines and sev- eral larger transformer models. Our code and weights are available at https://github.com/arc-ct/arc-ct.",
      "published": "2026-08-28T15:45:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.28455v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28455v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Learning to Use Tools: Reinforcement Learning for Tool-Integrated Mathematical Reasoning",
      "authors": [
        "Minghui Xu",
        "Zi Wang"
      ],
      "abstract": "Current large language models (LLMs) increasingly benefit from external tool integration, especially for tasks requiring reliable computation and verification. Motivated by this, we study calculator tool calling for improving mathematical reasoning on the Countdown task. We first analyze reasoning failures and find that calculation errors account for a substantial portion of incorrect responses. We then construct supervised fine-tuning datasets to teach the model useful tool-use patterns and how to interpret returned outputs. Building on this tool-formatted policy, we apply several on-policy reinforcement learning methods, including RLOO, RLOO++, GRPO, and DAPO, using automatically verifiable final-answer rewards. To enable a more reliable evaluation, we construct a fresh 1,024-problem held-out Countdown benchmark with no exact overlap with the training data. Our results show that calculator tool integration consistently improves both SFT and RL baselines, yielding roughly 10 percentage-point gains across pass@k. Among the RL methods, Tool-DAPO achieves the strongest performance, improving pass@1 from 35.8% for Tool-SFT to 66.0%. Further analysis shows that RL encourages more effective tool use even when only final-answer rewards are provided. These findings suggest that tool integration reduces arithmetic and verification errors, while RL increases the probability of correct reasoning traces.",
      "published": "2026-08-28T15:35:03Z",
      "abstract_url": "http://arxiv.org/abs/2608.28447v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28447v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "LongPIBench: A Long-Context Benchmark for Prompt Injection",
      "authors": [
        "Yupei Liu",
        "Yuqi Jia",
        "Neil Zhenqiang Gong",
        "Jinyuan Jia"
      ],
      "abstract": "Prompt injection attacks pose a serious security risk to large language models in real-world applications. However, existing prompt injection benchmarks primarily focus on short-context inputs, leaving the attacks and defenses in long-context settings largely unexplored. This gap leads to a substantial overestimation of the effectiveness of current defenses. In this paper, we bridge the gap by introducing LongPIBench, a long-context benchmark for prompt injection covering 4 realistic application scenarios: paper peer review, resume screening, code review, and email summary. For each scenario, we construct a synthetic dataset and a real-world dataset, with context lengths ranging from thousands to tens of thousands of tokens. The evaluation results on LongPIBench reveal significant vulnerabilities of prompt injection defenses under long-context settings: even simple heuristic prompt injection attacks achieve high success rates and frequently bypass state-of-the-art defenses. We hope LongPIBench can serve as a practical benchmark for systematically evaluating prompt injection defenses in realistic long-context scenarios.",
      "published": "2026-08-28T15:00:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.28411v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28411v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "VERA-8B: Evidence-Grounded Audit Risk Reasoning from SEC Filings",
      "authors": [
        "Menghan Liu",
        "Elynn Chen"
      ],
      "abstract": "Across audit applications, judgments must be supported by reasonable evidence. However, standard financial language models prioritize fluency over evidence. They are built for general financial reasoning and may produce plausible but ambiguous answers, creating a grounding gap that makes them unsuitable for audit work. We address this gap with VERA-8B, a new end-to-end audit reasoning system that identifies audit risks before enforcement actions occur. Constructing such a model raises several challenges, as no prior machine learning work targets pre-enforcement audit prediction. To our knowledge, we are the first to unify SFT and GRPO for evidence-grounded audit reasoning under one evidence standard, achieving performance that surpasses all evaluated baselines. Because auditing cannot tolerate unsupported claims, we introduce abstention and uncertainty qualification to defer uncertain or evidence-incomplete cases. Finally, we design an AuditBridge to ground model reasoning for practical audit work. It transforms raw filings into verified records and then into reviewer-ready reports, bridging finance and computation with broad generality. Together, these components produce auditable, review-ready outputs suitable for practical audit work.",
      "published": "2026-08-28T14:55:49Z",
      "abstract_url": "http://arxiv.org/abs/2608.28402v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28402v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "RetailAgent: Structured Adverse Timing in Self-Conditioned Multimodal LLM Trading Agents",
      "authors": [
        "Yupeng Zhang",
        "Liuyuan Jiang",
        "Hongyi Huang",
        "Bingheng Li",
        "Lisha Chen"
      ],
      "abstract": "In financial markets, a sequential policy that reacts systematically to price movements may become predictable to other market participants. This paper studies whether large language model (LLM) agents exhibit such directional structure through RetailAgent, an experimental framework in which an LLM observes anonymized intraday equity price histories and permitted state, then repeatedly chooses long (hold the stock) or flat (stay out) before the subsequent interval return is revealed. We compare returns during long and flat intervals along the same stock's intraday path after removing the overall fraction of long decisions. This exposure-matched measure reveals persistent negative timing across modality, horizon, state, and model family. Shuffling saved action sequences substantially attenuates the effect, showing that alignment between actions and subsequent returns drives the negative score. Feeding self-authored memories into decisions further increases policy persistence, while timing becomes more negative among stock-days on which the agent uses both actions. These results reveal stable, recoverable directional structure in sequential LLM financial decisions and a behavioral signal for studying how another participant could respond to a predictable policy.",
      "published": "2026-08-28T14:53:08Z",
      "abstract_url": "http://arxiv.org/abs/2608.28399v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28399v1",
      "categories": [
        "cs.AI",
        "q-fin.TR"
      ]
    },
    {
      "title": "Timing-Aware Repurchase Prediction for Web-Scale E-Commerce: Survival Models for Multi-Surface Grocery Recommendation",
      "authors": [
        "Akshay Kekuda",
        "Shreeranjani Srirangamsridharan",
        "Ishan Bhatt",
        "Yanan Cao",
        "Sinduja Subramaniam",
        "Evren Korpeoglu",
        "Kaushiki Nag",
        "Kannan Achan"
      ],
      "abstract": "Repurchase recommenders in e-commerce are commonly framed as a binary question asking \"will this customer buy this item within W days\", a formulation that requires a separately trained model for every horizon of interest. We replace this stack with survival models that predict time-to-repurchase directly, and evaluate them on millions of customers from a major grocery e-commerce platform across more than thirty ablation configurations. Our study makes three contributions. First, an empirical hazard analysis reveals a slightly decreasing marginal hazard (k ~ 0.9), differing from the common intuition that grocery items become more likely to be repurchased the longer since the last purchase (increasing hazard, k > 1). Log-Normal achieves the best marginal fit (R^2 = 0.998) and the best ranking, despite Weibull providing the best conditional residual fit, revealing an apparent discrepancy we analyze in detail. Second, a single Accelerated Failure Time (AFT) model replaces three per-horizon binary classifiers, matching or exceeding each at its own horizon while using roughly 3x fewer total trees. Feature importance reshuffles under the survival objective: channel-cadence and recency signals rise while aggregate frequency counts fall. Third, a 4-parameter parametric calibration maps raw survival CDFs to per-horizon probabilities with zero cross-horizon monotonicity violations. Calibration quality varies by an order of magnitude across the AFT family: Exponential AFT (Weibull k=1) achieves expected calibration error (ECE) ~1e-4, roughly 10x lower than Log-Normal, while ranking metrics agree within 0.3% relative. We adopt Exponential AFT for probability-consuming surfaces and Log-Normal for pure ranking, exposing a principled calibration-ranking trade-off within a single AFT family.",
      "published": "2026-08-28T14:49:03Z",
      "abstract_url": "http://arxiv.org/abs/2608.28393v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28393v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When Linguistic and Internal Confidence Diverge in Large Language Models",
      "authors": [
        "Hefan Zhang",
        "Bingquan Zhang",
        "Ming Cheng",
        "Saeed Hassanpour",
        "Weicheng Ma",
        "Soroush Vosoughi"
      ],
      "abstract": "Users often ask large language models (LLMs) to report how confident they are, but it is unclear whether such linguistic confidence tracks the model's internal confidence. We study this question across 8 classification tasks, 2 generation tasks and 30 models from three families. For classification, we compare linguistic confidence with logits-based confidence along three axes: association, magnitude agreement and calibration. For generation, we test whether linguistic confidence tracks semantic-entropy-based uncertainty. The axes frequently diverge. Instance-level association is weak on average, although it improves on easier items and for stronger base models. Instruction-tuned models often report higher confidence and sometimes show higher association, but they also have larger confidence gaps and worse calibration. Prompt design mostly changes the distribution of reported confidence. Attitude cues inflate confidence without improving alignment, while score exemplars can preserve rank-order signal when they avoid collapsed confidence values. Regression analyses show that distributional properties of confidence scores explain much of the observed alignment pattern, with model metadata playing a smaller role after controls. These results support a lossy-channel view of linguistic confidence. A more dispersed verbal confidence distribution can carry useful rank information, but it does not make the scores calibrated. Linguistic confidence should therefore be evaluated with multi-axis diagnostics before being used in downstream reliability pipelines.",
      "published": "2026-08-28T14:37:31Z",
      "abstract_url": "http://arxiv.org/abs/2608.28382v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28382v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "GRACE:Gradient-guided Coreset Selection for LLM Unlearning",
      "authors": [
        "Praveen Bushipaka",
        "Andrea D'Angelo",
        "Lucia Passaro",
        "Tommaso Cucinotta"
      ],
      "abstract": "Machine Unlearning methods for Large Language Models typically assume pre-specified forget and retain sets. In realistic settings, however, requests may provide only a few examples of undesired behavior, requiring forget and retain sets to be inferred from heterogeneous corpora. We study this data-selection problem and propose GRACE , a gradient-guided coreset selection method that constructs both forget and retain sets for LLM unlearning. GRACE first computes a forget direction from seed examples that elicit the undesired behavior, then selects a compact forget coreset whose gradients approximate this direction using non-negative orthogonal matching pursuit. To preserve model utility, it selects retain examples after projecting out the forget direction and applying clustered orthogonal matching pursuit in the remaining gradient space. Across two target domains, two model families, and four unlearning algorithms, GRACE improves model utility while maintaining comparable forget quality, with particularly consistent gains over prior gradient-based selection methods.",
      "published": "2026-08-28T14:12:49Z",
      "abstract_url": "http://arxiv.org/abs/2608.28361v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28361v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Propagating construction-time knowledge quality into medical question answering: A framework grounded in clinical guidelines",
      "authors": [
        "Jie Hu",
        "Junjie Wang",
        "Shan Lu",
        "Yifang Hu",
        "Gong Cheng",
        "Yun Liu"
      ],
      "abstract": "Large language models have facilitated knowledge graph (KG) construction from clinical guidelines, but extracted triples vary in structural validity and evidential support. Meanwhile, graph-augmented question answering (QA) systems typically optimize query relevance during retrieval, with limited reuse of quality information produced during KG construction. This creates a disconnect between construction-time quality control and inference-time evidence use. We investigate whether construction-time triple quality can serve as a persistent signal for downstream evidence selection and presentation. We propose a quality-aware framework that models structural conformance (SchemaConf) and evidential support (EvidScore) as complementary dimensions and fuses them into a per-triple quality signal, Q(t). Rather than using quality solely for filtering, the framework retains Q(t) and derived quality tiers as graph attributes and propagates them into quality-weighted subgraph retrieval and tier-conditioned evidence prompting, while preserving passage-level provenance. Experiments on Chinese diabetes clinical guidelines show that the utility of the quality signal is distribution dependent. Under cross-version and cross-model shift, the fused Q(t) provides stronger triple-quality discrimination than either component alone (AUC 0.748 vs. 0.703 for EvidScore and 0.645 for SchemaConf). In guideline-grounded QA, propagating construction-time quality reduces required-knowledge omission from 16.3% to 5.3% and conflicting outputs from 16.3% to 2.7%, with an evidence-grounded precision of 81.6% and near-zero invalid citations. Blinded clinician ratings favor the full framework over no retrieval (4.68 vs. 4.21 on a five-point scale) and approach the oracle condition (4.80), while cross-generator experiments show consistent trends.",
      "published": "2026-08-28T14:11:20Z",
      "abstract_url": "http://arxiv.org/abs/2608.28360v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28360v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla",
      "authors": [
        "Rowzatul Zannat",
        "Abdullah Al Shafi",
        "K. M. Azharul Hasan",
        "Atia Shahnaz Ipa"
      ],
      "abstract": "Medical question answering (QA) systems have become crucial tools for providing reliable health information. But they remain very unexplored for low-resource languages like Bangla due to limited datasets and systems tailored to these languages. To address this, we introduce BanglaMed-QA, a robust QA system specifically designed for the Bangla medical domain. The process begins with building a structured medical knowledge base that includes 4,493 QA pairs in 9 categories under 506 diseases. To improve semantic comprehension, domain-specific root word dictionaries and synonym sets are proposed, in addition to part-of-speech tagging for anaphora resolution. We adopt supervised machine learning models in which SVM is found to be the best model to categorize questions. Multiple similarity metrics, including cosine, Jaccard, BM25, and Levenshtein, are applied with soft and hard voting methods for query matching. The performance of the QA system has been evaluated in two aspects, with a 95% F1 score in an automated evaluation and an average human satisfaction rating of 0.9 out of 1.0. This validates the real-world application of BanglaMed-QA in closing the healthcare information gap for Bangla speakers.",
      "published": "2026-08-28T13:38:35Z",
      "abstract_url": "http://arxiv.org/abs/2608.28329v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28329v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers",
      "authors": [
        "Abrar Alotaibi",
        "Muhammad Shahid Jabbar",
        "Sadam Al-Azani",
        "Moataz Ahmed"
      ],
      "abstract": "Practitioners defend large language models (LLMs) by stacking defenses, assuming the layers compound. A stack is an ensemble, and ensembles compound only under a condition the LLM security literature recommends but never measures: the members must fail on different inputs. Two instruments make that measurable. The Adversary Access-Tier Model (AATM) grades an adversary by the access it holds, from system-only (A0) to influence over training data (A4). A cost model sorts defenses into five classes of inference-time overhead; because two classes require training weights or reading activations, they tier the defender as AATM tiers the adversary. From these we derive how a stack behaves, and the quantities a defender cares about diverge: coverage saturates within a tier, cost rises by class, false refusals accumulate as a union, and residual attack success falls multiplicatively only under independence. We measure that independence. Running one adaptive adversary against a seven-layer stack, failure correlation is positive in all fifteen measurable pairs ($φ$ from $0.30$ to $0.75$), and the joint residual exceeds the multiplicative prediction by up to $0.172$. Stratifying on behavior difficulty dissolves most of the association, so the dependence is predominantly common-cause, but it survives permutation inference, majority-vote grader labels, and externally calibrated thresholds. The same stack refuses four in five benign prompts while remaining statistically indistinguishable from its strongest single layer. The dependence is architectural rather than sampling-based: members correlate through the model they all wrap, so no wider member pool weakens it. Diversity therefore selects stack members but does not predict what an assembled stack delivers, which has to be measured end to end.",
      "published": "2026-08-28T13:36:06Z",
      "abstract_url": "http://arxiv.org/abs/2608.28327v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28327v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "MAIL: Memory-driven, Adaptive, Incremental, and Literature-grounded Framework for Hypothesis Generation in Chemistry",
      "authors": [
        "Mahdi Babaei",
        "Xueshen Li",
        "Yutao Kuang",
        "Jolene P. Reid",
        "Yu Gan"
      ],
      "abstract": "The ever-expanding volume of the chemical literature offers unprecedented opportunities to generate novel and impactful hypotheses. However, the bottleneck lies in efficiently navigating this vast knowledge base to formulate high-quality, experimentally meaningful insights. While Large Language Models (LLMs) show promise for this task, existing methods often rely on static inspiration corpora, predefined heuristics, or laborious human-in-the-loop pipelines and decision-support frameworks that limit scalability and novelty. In this work, we propose an automated approach, a Memory-augmented, Adaptive, Incremental, and Literature-grounded (MAIL) framework for hypothesis generation in chemistry. Our MAIL method formulates hypothesis generation as a temporally grounded, memory-driven reasoning process, where hypotheses emerge from an evolving conceptual path that continuously accumulates and reinterprets prior knowledge. We evaluated the MAIL framework on a public TOMATO-Chem dataset and a newly curated and disseminated high-novelty nature/science challenge (HN-NS) dataset. Across both datasets, MAIL generates structurally coherent and mechanistically plausible hypotheses, achieves the highest MIOS and MPOS by more effectively recovering the central ideas and methodological elements of the historical target hypotheses, and obtains the highest overall expert-evaluation scores for scientific quality. These results demonstrate the potential of LLMs to autonomously explore chemical domains and generate hypotheses that are both innovative and chemically plausible.",
      "published": "2026-08-28T13:25:27Z",
      "abstract_url": "http://arxiv.org/abs/2608.28315v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28315v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Deriving Scaling Laws for OpenEuroLLM Models: Learning Rate, Batch Size and Loss",
      "authors": [
        "Niccolò Ajroldi",
        "Diana Alexandra Onutu",
        "Haider Al-Tahan",
        "Jörg Franke",
        "Sampo Pyysalo",
        "Jenia Jitsev",
        "Aaron Klein"
      ],
      "abstract": "We study the scaling behavior of learning rate and batch size in pretraining dense large language models on English-prevalent corpora. Beyond scaling \\textit{jointly optimal} learning rates and batch sizes, we investigate their \\textit{marginal} evolution with model capacity and data scale and develop a model that captures these relationships. As we employ a Warmup-Stable-Decay learning rate schedule, we further investigate the gains from learning rate annealing over a broad range of hyperparameters settings, models and data budgets, and whether the optimal learning rate and batch size \\textit{transfer} between the stable and decay phases. Finally, we characterize the dependence of loss on model capacity and dataset size, evaluating recently proposed scaling forms that explicitly model their interaction. We find these approaches particularly effective at capturing both undertraining and overtraining regimes across our experiments. This study establishes a first baseline and scaling procedure for the development of future OpenEuroLLM models. We open-source the complete collection of pretraining runs used in this study.",
      "published": "2026-08-28T13:16:56Z",
      "abstract_url": "http://arxiv.org/abs/2608.28308v1",
      "pdf_url": "https://arxiv.org/pdf/2608.28308v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
