const PAPERS_DATA = {
  "last_updated": "2026-07-15 03:15:01 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Do AI Agents Know When a Task Is Simple? Toward Complexity-Aware Reasoning and Execution",
      "authors": [
        "Junjie Yin",
        "Xinyu Feng"
      ],
      "abstract": "Large language model (LLM) agents increasingly automate multi-step engineering and informatics workflows, yet they rarely ask how much effort a task actually requires. They often follow a maximum-context-first strategy--re-reading files and dependencies they have already seen--turning a one-line edit into a small code-base audit. We argue the missing capability is task-aware execution-scope estimation: judging a task's difficulty, the information it truly needs, and the shortest reliable path before committing budget. We formalize minimum-sufficient execution and the Agent Cognitive Redundancy Ratio (ACRR), and propose E3 (Estimate, Execute, Expand): the agent estimates an initial operating point, executes a minimum viable path, and expands scope only when verification fails. On MSE-Bench--a deterministic benchmark of 121 edits in a capability-controlled simulator--E3 matches the strongest baseline's 100% success while cutting cost by 85%, tokens by 91%, and inspected files by 92%, and further beats a strong adaptive retrieval baseline by 16%; the gains survive held-out instruction wording and essentially every cost weighting. A companion real-model harness (LLM-Case) corroborates the effect on a live gpt-4o agent editing a real open-source library, with every candidate patch graded by actually running the project's real pytest suite against a measured oracle: the over-reading is milder but real, and E3 is the leanest and fastest policy at comparable task success--its one shortfall a provider rate-limit, not a wrong edit. We frame this as a controlled probe of execution redundancy, not a measurement of any deployed agent, and position task-aware execution as a step toward engineering-grounded AI (EGAI)--agents whose effort is anchored in the engineering reality of the task. We release the framework and benchmark.",
      "published": "2026-07-14T17:59:31Z",
      "abstract_url": "http://arxiv.org/abs/2607.13034v1",
      "pdf_url": "https://arxiv.org/pdf/2607.13034v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.SE",
        "eess.SY"
      ]
    },
    {
      "title": "TerraZero: Procedural Driving Simulation for Zero-Demonstration Self-Play at Scale",
      "authors": [
        "Zhouchonghao Wu",
        "Akshay Rangesh",
        "Weixin Li",
        "Wei-Jer Chang",
        "Zachary Lee",
        "Tim Wang",
        "Wei Zhan"
      ],
      "abstract": "Training robust autonomous driving agents requires a simulator that is fast enough for reinforcement learning at scale, realistic enough to ground behavior in real-world map structure, and diverse enough to cover the safety-critical long tail that logged data rarely contains. We present TerraZero, a procedural driving simulator and self-play training stack. A configurable C engine runs simulation on the CPU and policy inference on the GPU over a zero-copy path, sustaining 1.3M agent-steps per second on a single server-grade GPU, far faster than existing object-level simulators, while keeping fidelity lighter single-agent systems omit: heterogeneous agents, multiple dynamics models, and full traffic-rule enforcement. TerraZero treats logged data only as a source of real-world map geometry, populating each map with randomized rule-based road users and signal controllers and randomizing agent dynamics, rewards, and sizes per episode, so a map yields an unbounded set of scenarios. Every reported policy trains from scratch by reinforcement learning alone on a compute-efficient self-play recipe across GPUs, with zero human demonstrations and no fallback planner at inference. Policies generalize zero-shot across cities and datasets, including emergent left-hand-traffic driving without explicit supervision. As an ego policy, TerraZero is the first fully learned policy to top the InterPlan long-tail benchmark, ahead of larger learned planners; on routine-driving val14 it ranks among the best approaches and is the safest, posting the best collision and time-to-collision scores. On Waymo Open Sim Agents realism the same recipe outperforms other demonstration-free methods and is competitive with the strongest reference-anchored self-play method. One stack serves both roles: driving policies across dynamics for cars and trucks, and sim agents that jointly control vehicles, pedestrians, and cyclists.",
      "published": "2026-07-14T17:59:02Z",
      "abstract_url": "http://arxiv.org/abs/2607.13028v1",
      "pdf_url": "https://arxiv.org/pdf/2607.13028v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "PalmClaw: A Native On-Device Agent Framework for Mobile Phones",
      "authors": [
        "Hongru Cai",
        "Yongqi Li",
        "Ran Wei",
        "Wenjie Li"
      ],
      "abstract": "Large Language Model (LLM) agents have moved beyond generating responses to executing multi-step tasks by calling tools, observing the results, and iteratively deciding the next action. Most agent systems run on desktops or servers, which support tool use and task automation. Mobile devices are also important agent environments because they are widely accessible and contain users' data, sensors, and daily-use applications. Existing mobile agents mainly operate smartphones through graphical user interface (GUI) actions such as tapping, swiping, and typing, which often form long, interface-dependent sequences, cannot directly access device capabilities, and make execution boundaries difficult to define. We present \\textbf{PalmClaw}, an open-source agent framework that runs natively on mobile phones and manages the sessions, memory, skills, tools, and agent loop directly on the device. PalmClaw exposes device capabilities as device tools with explicit arguments, structured results, and clearly defined execution boundaries. This design enables agents to use mobile capabilities directly while keeping each action explicit and controlled. Experiments show an 11.5\\% relative improvement in task success and a 94.9\\% reduction in completion time over the strongest baseline, with lower setup burden and traces illustrating how execution boundaries are applied. Code is available at https://github.com/ModalityDance/PalmClaw.",
      "published": "2026-07-14T17:58:57Z",
      "abstract_url": "http://arxiv.org/abs/2607.13027v1",
      "pdf_url": "https://arxiv.org/pdf/2607.13027v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FormalAnalyticGeo: A Neural-Symbolic Based Framework for Multimodal Analytic Geometry Problem Generation",
      "authors": [
        "Ruoran Xu",
        "Wending Gao",
        "Qiufeng Wang"
      ],
      "abstract": "Math reasoning has achieved significant progress with the rapid advancement of Multimodal Large Language Models (MLLMs), however analytic geometry remains largely underexplored, primarily due to the scarcity of annotated samples. Existing diagram generation approaches struggle with analytic geometry: template methods cannot handle constraint-driven layouts, and generative models lack the geometric precision to render annotated conic curves correctly. We present FormalAnalyticGeo, a scalable framework for fully automatic generation of multimodal analytic geometry problems. Leveraging the rigor of formal languages, we design the framework around CDL (Condition Description Language), a formal intermediate representation that bridges free-form problem text with precise diagram rendering via a Signed Distance Field (SDF) engine. The framework employs four specialized LLM components in sequence: a Generator that produces diverse analytic geometry problems, a Formalizer that converts each problem into CDL for SDF-based rendering, a Measurer that extracts ground-truth answers through vision-based measurement on the rendered diagrams, and a Quality Verifier that checks outputs at three stages. Structured feedback from the Quality Verifier drives automatic retry, forming a closed loop that eliminates any need for human annotation. Applying FormalAnalyticGeo at scale yields AnalyticGeo7K, a dataset of over 7K verified multimodal problems, each with aligned text, diagram, formal annotation, and ground truth.Experiments show that the generated problems achieve a median ground-truth relative error of 0.70\\%, with 82.3\\% of answers falling within 5\\% of the exact symbolic solution. Our framework and dataset will be publicly released.",
      "published": "2026-07-14T17:24:57Z",
      "abstract_url": "http://arxiv.org/abs/2607.12982v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12982v1",
      "categories": [
        "cs.AI",
        "cs.MA",
        "cs.SC"
      ]
    },
    {
      "title": "Form, Not Content? A Preregistered, Placebo-Controlled Evaluation of Learned Error-Conditioned Self-Repair Through Prompts and Weights in Frozen Small Code Models",
      "authors": [
        "Mehmet Iscan"
      ],
      "abstract": "Frozen small code LLMs are deployed locally, yet the information guiding a retry after a failed attempt is still measured without placebo controls in the self-repair literature. We treat a failed program as a conjecture and an execution counterexample as an oracle-relative refutation, and introduce PoPE (Popperian Placebo-controlled Evaluation): a methodology for measuring whether evidence that falsifies LLM-generated code can be used operationally by that same model. In PoPE, error content is paired with channel-specific placebos that keep the predeclared scaffold while ablating task-relevant content or deranging the task-error assignment. Frozen small code models (0.5-1.5B) are evaluated under preregistered rules through a prompt channel and a weight channel (small-data adapter training), with four generations per arm-unit pair. In the prompt channel, public-tier screening unlocked 12 units under the content-ablated form placebo versus 10 under the live error-pattern arm on a 40-unit resistant band; the result was recorded as mechanism-null. In the weight channel, an 8-8 tie was observed between the error-content adapter and the intervention-free baseline (p=1.0), while the SHA-deranged placebo adapter stayed ahead with 10 unlocks; content-attributable superiority was not confirmed. These results do not constitute evidence of equivalence or non-inferiority. Equivalence was not tested separately. Findings are restricted to the public-tier screening endpoint; hidden-tier confirmation was deferred by design. We read this not as compiled criticism disappearing as information, but as the loss of its external role in testing a new conjecture: when a representation learned from the oracle is written back into the generation state, testing is replaced by conditioning. No working JEPA-RL controller is claimed. PoPE is presented as a placebo-controlled, retestable measurement standard.",
      "published": "2026-07-14T16:59:42Z",
      "abstract_url": "http://arxiv.org/abs/2607.12962v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12962v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Reproducible Reservoir Computing with Thermally Driven Superparamagnets: Controlling Temperature Sensitivity",
      "authors": [
        "Zhengfei Chen",
        "Alex Welbourne",
        "Matthew O. A. Ellis",
        "Dan A. Allwood",
        "Eleni Vasilaki",
        "Thomas J. Hayward"
      ],
      "abstract": "Unconventional computing systems must demonstrate robust performance under real-world environmental conditions to enable practical deployments. We have recently proposed superparamagnetic nanodot ensembles driven by strain-induced magnetoelectric coupling as exciting candidates for use as ultra-low energy consumption reservoir computing substrates. However, because their dynamics are governed by thermal activation effects, these systems are intrinsically sensitive to ambient temperature fluctuations, leading to degraded task performance when operated outside the temperature range used during training. In this paper we simulate how temperature variations affect the magnetization dynamics of such superparamagnetic ensembles, and quantify how this affects task performance. We then show how heterogeneous nanodot patterns that incorporate different sizes of nanodots with different characteristic timescales for thermal activation mitigate this problem. Benchmark results on the NARMA-10 task show that introducing optimized heterogeneity stabilizes performance of the reservoirs across a wide range of ambient temperatures (5-35°C), with little loss of ultimate performance. We also characterize the trade-off between performance and temperature stability and show that it can be tuned via reservoir hyperparameters. Our study demonstrates a key step in making these novel devices suitable for real-world deployment.",
      "published": "2026-07-14T14:57:12Z",
      "abstract_url": "http://arxiv.org/abs/2607.12840v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12840v1",
      "categories": [
        "cs.ET",
        "cond-mat.mes-hall",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Accelerating Masked Diffusion Large Language Models: A Survey of Efficient Inference Techniques",
      "authors": [
        "Daehoon Gwak",
        "Minhyung Lee",
        "Junwoo Park",
        "Jaegul Choo"
      ],
      "abstract": "Diffusion large language models (dLLMs) offer a theoretical advantage in parallel generation over standard autoregressive models. However, parallel generation alone does not guarantee practical speedups. Realizing this efficiency requires specialized inference mechanisms, such as diffusion-aware caching and reuse. Consequently, as inference efficiency becomes a prerequisite for practical deployment, recent research has actively explored acceleration techniques across algorithms, architectures, and systems. However, rigorous comparisons remain difficult, as end-to-end latency stems from intricate trade-offs between algorithmic, architectural, and system-level factors that are often conflated in existing benchmarks. In this survey, we introduce a unified latency decomposition framework for dLLMs to disentangle these factors and analyze their impact on inference speed in real deployments. Guided by this framework, we categorize acceleration techniques along three axes covering algorithmic innovations, architectural and system optimizations, and inference-time scaling. Finally, we provide guidelines for reproducible benchmarking and highlight open challenges for realizing the full potential of parallel generation.",
      "published": "2026-07-14T14:48:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.12829v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12829v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Solution of the Hempel's statistical ambiguity problem and Causal AI",
      "authors": [
        "Evgenii Vityaev"
      ],
      "abstract": "This paper addresses Carl Hempel's longstanding problem of statistical ambiguity in inductive-statistical inference, in which contradictory predictions are derived from statistical laws. To avoid such predictions, Carl Hempel proposed the Requirement of Maximal Specificity (RMS) for the statistical laws used in the inference. An analysis of the RMS refinements made by Wesley Salmon, Alberto Coffa, and James Fetzer led to the following definition of maximally specific statistical laws: \"the lawlike premises of an adequate explanation must specify all and only those properties whose presence or absence made a difference to the occurrence of its explanandum-phenomenon.\" However, there was no proof of a solution to the statistical ambiguity problem based on this definition. We use Nancy Cartwright's definition of causes that raise probabilities across background contexts, and then introduce the concept of Causal Rules. Then we define a special semantic probabilistic inference procedure that incrementally refines these causal rules by incorporating all statistically relevant information. This procedure yields Maximally Specific Causal Relationships (MSCRs), for which we prove (Theorem 1) that predictions derived from them are consistent. This resolves the statistical ambiguity problem. The semantic probabilistic inference procedure provides a probabilistic causal learning system, which may be used in such new areas as Causal AI and Causal Machine Learning. They fundamentally explore causal inference as a tool for understanding cause-and-effect relationships within complex systems. Properties similar to RMS remain under discussion. Several notions related to RMS are considered: invariant feature learning, invariant causal prediction, and spurious association.",
      "published": "2026-07-14T14:44:48Z",
      "abstract_url": "http://arxiv.org/abs/2607.12826v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12826v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters?",
      "authors": [
        "Kaiwen Zheng",
        "Junchen Fu",
        "Wenhao Deng",
        "Hu Han",
        "Joemon M. Jose",
        "Xuri Ge"
      ],
      "abstract": "Recent advances in multimodal large language models (MLLMs) have significantly improved the performance of multimodal emotion recognition (MER) and enabled interpretable description generation by jointly modeling video, audio, and language, etc. However, these performance improvements are often accompanied by an increase in model parameter size (e.g, at least 7B), which simultaneously incurs high computational costs and reduces inference efficiency, thereby hindering real-time deployment on resource-constrained platforms such as robots and mobile devices. This raises a fundamental question: do we really need the multimodal MER model larger than 1B parameters for high-quality MER? In this paper, we challenge the assumption that larger models are inherently necessary and proposes a lightweight MER framework (called Light-MER), which achieves better and faster multimodal sentiment understanding and recognition through knowledge distillation. It can transfer knowledge from a strong, large-scale teacher model to a lightweight sub-billion-parameter student model, aiming to preserve rich multimodal emotion reasoning and recognition while substantially improving deployment efficiency. Specifically, we introduce two new optimization strategies to enhance knowledge transfer: (1) a new optimal transport loss that combines Sliced Wasserstein Distance with hidden-state alignment, and (2) a new multi-reward optimization strategy based on GRPO that balances MER performance and efficiency, aimed at further enhancing the learning capabilities of student models. Extensive experiments on nine benchmark datasets demonstrate that Light-MER achieves state-of-the-art performance while significantly improving inference efficiency. This highlights the strong potential of small multimodal emotion language models for future research. Code is available at https://github.com/GAIR-Lab/Light-MER.",
      "published": "2026-07-14T14:01:29Z",
      "abstract_url": "http://arxiv.org/abs/2607.12787v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12787v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.MM"
      ]
    },
    {
      "title": "Constraint-Aware Aggregation for Federated Reinforcement Learning in Microgrid Energy Coordination",
      "authors": [
        "Usman Haider",
        "Karl Mason"
      ],
      "abstract": "Federated Reinforcement Learning (FedRL) enables coordination of distributed energy resources without sharing raw local data, but standard aggregation methods such as FedAvg do not account for system-level constraints, often leading to unsafe global behavior. In this work, we study constraint-aware aggregation for federated reinforcement learning in distributed energy coordination. We propose aggregation rules that incorporate both local performance and estimated constraint violation into the server-side update. Among these, a simple penalty-based rule, $w_i \\propto R_i - αV_i$, consistently provides the most reliable trade-off between reward and safety, without requiring dual optimization or modifications to local training. \\textcolor{black}{We evaluate our approach on DairyGridEnv, a benchmark modeling multiple farms coordinating battery storage under stochastic demand and a shared grid capacity constraint, and further assess robustness using real load-driven demand profiles from Finland and the German FIELD dataset. Across multiple seeds, penalty-based aggregation substantially reduces violations while improving reward relative to FedAvg in both synthetic and real load-driven settings.} A combined reward-violation scheme exposes a tunable trade-off via $λ$, but is less stable. These results demonstrate that lightweight aggregation strategies can substantially improve empirical safety in federated reinforcement learning while preserving standard communication protocols.",
      "published": "2026-07-14T13:33:25Z",
      "abstract_url": "http://arxiv.org/abs/2607.12763v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12763v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LLMs Can See the Smoke but not the Fire: Evaluating Abductive Reasoning with Elenchos",
      "authors": [
        "Julius Steiglechner",
        "Lucas Mahler",
        "Gabriele Lohmann"
      ],
      "abstract": "Large language models (LLMs) excel at pattern recognition and text generation, but their capacity for abductive inference - inferring latent hypotheses that explain observed behavior - remains poorly understood. Here, we introduce Elenchos (named after the Socratic method of cross-examination), a generative evaluation framework that measures abductive reasoning as a structural inverse problem. Given a reference formal system, such as the lambda-calculus, and a potentially mutated counterpart, agents must determine whether a mutation has occurred and infer the rule modifications responsible for the resulting behavioral differences. Evaluating frontier and mid-tier LLMs reveals a consistent detection-attribution dissociation: models often recognize that a system has been altered but struggle to identify the latent mutations causing the observed discrepancies. Performance degrades substantially under interacting mutations, where models frequently recover only a subset of the underlying mutations. Preliminary evidence also suggests diminishing returns from increased inference-time reasoning, with only modest improvements under larger reasoning budgets, though this finding requires further validation.",
      "published": "2026-07-14T13:03:39Z",
      "abstract_url": "http://arxiv.org/abs/2607.12733v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12733v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning-based Probabilistic Load Forecasting with Post-hoc and In-model Uncertainty",
      "authors": [
        "Sarah Al-Shareeda",
        "Gulcihan Ozdemir",
        "Heung Seok Jeon"
      ],
      "abstract": "Smart-building load forecasters are often trained offline on dense, multivariate, high-frequency data, but deployment may provide only hourly, feature-limited inputs. Missing features must then be reconstructed, and their errors can propagate through the model. If this input uncertainty is not reflected, prediction intervals may become miscalibrated, affecting demand-response scheduling. Our work examines where uncertainty should be placed once inference inputs are reconstructed. We develop a unified one-day-ahead probabilistic forecasting framework that aligns temporal resolution, reconstructs the unavailable inputs, and derives causal features, and we compare a modular post-hoc residual-quantile scheme with an integrated in-model quantile-learning scheme. The comparison uses three mid-scale Deep Learning (DL) backbones: recurrent, hybrid recurrent, and attention-based Temporal Fusion Transformer (TFT) models, under identical inputs, forecasting horizon, preprocessing rules, and training budgets. Results show that uncertainty placement is backbone-dependent. Integrated quantile learning is most reliable with the TFT, yielding 2.2-3.6% MAPE and 28-83W RMSE on the labeled test window, while producing intervals about 5x narrower than the modular intervals at the closest-to-nominal coverage level. Diebold-Mariano tests support the TFT ranking and the mixed behavior of the recurrent backbones. A reconstruction-sensitivity test shows that reconstructed inputs increase the Quantile Score (QS) by 106% while interval width remains nearly unchanged, indicating that the model does not automatically absorb reconstruction-induced uncertainty. Robustness checks against non-DL baselines and seasonal hold-out weeks support this ranking. Our results expose the limits of post-hoc residual quantiles when inference depends on reconstructed inputs.",
      "published": "2026-07-14T13:02:18Z",
      "abstract_url": "http://arxiv.org/abs/2607.12730v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12730v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Bulkhead: Automated Semantic Detection and Remediation of Container Escape Vulnerabilities",
      "authors": [
        "Qiyuan Fan",
        "Zhi Li",
        "Junjie Li",
        "XiaoFeng Wang",
        "Bin Yuan",
        "Deqing Zou"
      ],
      "abstract": "Filesystem isolation in container ecosystems is often weakened by cross-boundary path misresolution, causing path traversal (PaTra) vulnerabilities. These vulnerabilities stem from insecure host-container interactions and have become increasingly pervasive as cloud systems mount shared resources, such as GPUs and agent workspaces, into containers to support AI workloads. Existing defenses remain inadequate. Kernel-level protections are intrusive, can destabilize system calls, and have therefore not been accepted into the Linux mainline. Detection methods rely on static rule matching or manual code auditing. Static rules can flag path-related functions but fail to capture the semantics needed to determine whether a host-container interaction exists, causing many false positives. Manual review requires domain expertise, making it costly, inefficient, and difficult to scale. To address this threat, we present Bulkhead, an automated framework that integrates large language models (LLMs) with formal methods for semantic vulnerability discovery and remediation. Bulkhead uses a multi-agent system to identify and repair PaTra vulnerabilities through multi-dimensional knowledge patterns generalized from known cases. It first applies high-risk functional patterns to locate entry points for cross-boundary interactions in containerized code, then uses call-chain patterns to recover the corresponding execution paths at suitable depth. The Detection pipeline analyzes these call chains against the application scenarios and threat model, identifying vulnerabilities such as missing security checks and TOCTOU flaws in cross-boundary interactions, and generating proof-of-concept (PoC) exploits for validation. These PoCs then guide patch generation. To ensure remediation correctness, the Patch pipeline performs assertion-driven verification using predefined model-checking templates.",
      "published": "2026-07-14T12:52:56Z",
      "abstract_url": "http://arxiv.org/abs/2607.12723v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12723v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts",
      "authors": [
        "Jincheng Xie",
        "Runheng Liu",
        "Heyan Huang",
        "Yawen Ling",
        "Hanbin Dai",
        "Yu Zheng",
        "Wen Hu"
      ],
      "abstract": "Sparse Mixture-of-Experts (MoE) models have become an important approach for scaling Large Language Models (LLMs), but their inference efficiency depends strongly on expert activation patterns. Speculative decoding (SD) accelerates autoregressive generation by verifying multiple draft tokens in parallel, yet existing draft selection strategies primarily optimize acceptance likelihood. In large-scale MoE models, however, selecting draft tokens also determines the union of experts activated during verification. We observe that confidence-driven SD can introduce \\textit{expert scattering}: high-probability draft tokens may route to disjoint experts, increasing expert-weight memory traffic and reducing the speedup from speculation. Motivated by this observation, we revisit draft-tree selection under the non-uniform memory-cost structure of MoE inference. We propose \\textsc{EcoSpec}, a cost-aware speculative decoding framework that incorporates predicted marginal expert activation cost into draft selection. With a lightweight expert predictor and a dynamic expert buffer, \\textsc{EcoSpec} favors draft paths that preserve high acceptance likelihood while reusing experts already covered by the current verification set, without modifying the target-model verification rule. We evaluate \\textsc{EcoSpec} on three large-scale MoE models, including DeepSeek-V3.1 (671B), Qwen3-235B-A22B, and GPT-OSS-120B, across reasoning, coding, question-answering, and dialogue benchmarks. \\textsc{EcoSpec} consistently reduces active expert footprints and improves end-to-end decoding speed, achieving up to $1.62\\times$ speedup. These results show that accounting for expert activation cost is important for efficient speculative decoding in large-scale MoE models.",
      "published": "2026-07-14T12:22:55Z",
      "abstract_url": "http://arxiv.org/abs/2607.12696v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12696v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.DC"
      ]
    },
    {
      "title": "Evidence-Grounded Verified Agentic Reasoning: A Path Toward Eliminating LLM Hallucination in Empirical Inference via Tool-Attested Kernel Proofs",
      "authors": [
        "Junyu Ren"
      ],
      "abstract": "Tool access alone does not make LLM empirical reasoning governable: accepted outputs need not descend from attested evidence, and accepted deductions need not hold up under formal scrutiny. We present EG-VAR (Evidence-Grounded Verified Agentic Reasoning), a Lean 4-based tool-calling architecture in which the Lean kernel is the sole minter of Verified claims via tool-attestation axioms and declared source lifts. Every verified output structurally descends from an attested tool call (Thm. 3.1) and a kernel-checked chain of valid inference (Thm. 3.2); residual outputs are honest Abstain with a replayable audit trail. On a subcollection of TableBench numerical reasoning (n=120), EG-VAR attains 120/120 versus a 95% same-tool baseline; on counterfactual stress tests (5 domains x 2 models), EG-VAR stays 100% source-faithful while same-tool drops to 80-90% (no-tool 50-80%). With the LLM as deployment-time formalizer, residual semantic-formalization error is 3.3% on Sonnet and 1.7% on Opus. We position EG-VAR as a technical-governance interface for high-stakes empirical claims: a formal sidecar makes the target proposition, source scope, evidence boundary, proof obligation, and abstention condition auditable, eliminating unsupported Verified outputs today while turning formalization errors, lift and source-authority disputes, ambiguities, and abstentions into explicit audit targets. Over time, typed sidecars in datasets, APIs, public records, and AI-generated documents can amortize this formalization burden into reusable infrastructure.",
      "published": "2026-07-14T11:33:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.12650v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12650v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CY",
        "cs.SE"
      ]
    },
    {
      "title": "Atomic Units of X: The Compression Layer of Intelligence",
      "authors": [
        "Sachin Dev Duggal",
        "Pradyumna Swarnalatha Ramanna",
        "Alexandros Vassiliades"
      ],
      "abstract": "This paper proposes a theoretical framework for understanding intelligence as a process of atomic compression and compositional reuse. We argue that cognitive, biological, computational, and organizational systems achieve scalable intelligence by decomposing complex phenomena into reusable atomic units that can be recombined into higher-order structures. Drawing on evidence from cognitive science, information theory, evolutionary biology, software engineering, medicine, legal reasoning, education, music, and artificial intelligence, the paper develops the concept of atomic units as fundamental compression layers that support efficiency, transfer, interpretability, and evolvability. The central contribution is the Compression Calculus, a formal framework for comparing surface-level representations with atomic representations and for describing how compression gains compound across abstraction layers. We introduce the Compounding Cascade thesis, according to which each additional layer of abstraction multiplicatively increases representational efficiency rather than merely adding incremental savings. The paper further argues that contemporary AI systems often operate at suboptimal levels of representation, relying on token-level processing or document-level retrieval rather than stable, concept-level atomic structures. In this view, large language models are best understood not as complete knowledge architectures, but as dynamic fusion engines capable of navigating, sequencing, and recombining atomic units. The framework provides a foundation for designing self-evolving knowledge systems that can discover, refine, and compose new primitives over time. By reframing intelligence as compression through compositional abstraction, the paper offers a unifying perspective on expertise, knowledge representation, explainable AI, and the future architecture of adaptive intelligent systems.",
      "published": "2026-07-14T11:11:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.12634v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12634v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making?",
      "authors": [
        "Minh Khoi Ho",
        "Zihao Zhu",
        "Runchuan Zhu",
        "Levina Li",
        "Zhiwen Fan",
        "Zhangyang Wang",
        "Junyuan Hong"
      ],
      "abstract": "As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users' emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents.",
      "published": "2026-07-14T11:09:51Z",
      "abstract_url": "http://arxiv.org/abs/2607.12631v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12631v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Multi-Perspective Agentic Program Repair via Code Property Graphs and Temporal Execution Graphs",
      "authors": [
        "Zhili Huang",
        "Ling Xu",
        "Hongyu Zhang"
      ],
      "abstract": "Large language models (LLMs) have improved automated program repair (APR), but two limitations remain. First, raw execution traces are often too large and repetitive to serve as effective model context. Second, repeated patch sampling may produce different implementations without yielding distinct root-cause hypotheses or repair strategies. We present CT-Repair, an agentic APR framework representing static and dynamic evidence as queryable Code Property Graph (CPG) and Temporal Execution Graph (TEG). CT-Repair applies a three-stage filtering pipeline to construct compact TEGs. Three finite-state-machine-guided agents analyze each bug from static, dynamic, and hybrid perspectives and independently produce evidence-grounded repair strategies. A strategy-guided generation procedure instantiates these strategies as candidate patches and uses validation feedback to refine the most promising strategy. We evaluate CT-Repair on 854 Java bugs from Defects4J v3.0. In the mixed-model configuration, CT-Repair correctly repairs 489 bugs. Under a controlled GPT-5.4-mini configuration, it repairs 388 bugs, 19 and 30 more than ReinFix and RepairAgent, respectively. The union of the three evidence perspectives repairs 99 more bugs than the strongest individual perspective. The filtering pipeline also compacts runtime evidence, with execution filtering narrowing the candidate method scope by 94.85% on average and behavior filtering further reducing retained runtime records by 55.97%. These results show that structured runtime evidence and multi-perspective reasoning can improve repair effectiveness without relying solely on a larger patch-generation budget.",
      "published": "2026-07-14T10:33:29Z",
      "abstract_url": "http://arxiv.org/abs/2607.12605v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12605v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Deep Learning-based Surrogate Modelling of the LOD Method for Multiscale Problems",
      "authors": [
        "Marc Haltmayer",
        "Jaemin Seo",
        "Yuseung Lee",
        "Sungyeop Lee",
        "Jaehoon Jeong",
        "Jae Yong Lee"
      ],
      "abstract": "Multiscale problems are notoriously difficult to tackle using traditional numerical methods, as accurately resolving fine-scale features often requires prohibitively fine discretizations. This challenge is particularly pronounced in applications such as materials science, fluid dynamics, climate systems, chemical processes, and complex networks. Recent neural operator models provide a promising data-driven alternative, but frequently struggle to achieve sufficient accuracy in the presence of strongly heterogeneous or oscillatory coefficients. In this work, we focus on the solution of elliptic PDEs with rough and high-contrast inputs. The Localized Orthogonal Decomposition (LOD) method is a well-established numerical approach for such problems, but it comes, however, at a substantial computational cost. We investigate the performance of popular neural operator architectures on these challenging multiscale problems and identify key limitations in their ability to resolve fine-scale structure. To overcome these challenges, we introduce LOD-MSNO (LOD-Multiscale Neural Operator), a hybrid approach that leverages the LOD method as a strong multiscale prior by building on its representation of the solution as a linear combination of problem-adapted basis functions, while addressing its main computational bottlenecks through data-driven operator learning. We further provide theoretical error estimates for the proposed coefficient-learning framework. Lastly, we demonstrate the potential of our proposed method to outperform current neural operator baselines in terms of accuracy for challenging multiscale inputs, while mainly retaining the computational efficiency of neural operator models.",
      "published": "2026-07-14T09:41:49Z",
      "abstract_url": "http://arxiv.org/abs/2607.12570v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12570v1",
      "categories": [
        "math.NA",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel",
      "authors": [
        "Niccolò Caselli",
        "Salvatore Lo Sardo",
        "Francesco Massafra",
        "Ippokratis Pantelidis",
        "Samuele Punzo",
        "Sathya Kamesh Bhethanabhotla"
      ],
      "abstract": "We investigate whether temporal hierarchy can improve LeWorldModel on long-horizon goal-conditioned control. We introduce Hi-LeWM, an extension that freezes the pretrained low-level LeWM and adds high-level planning over latent subgoals. We evaluate Hi-LeWM on PushT and Cube across increasing goal offsets. Hierarchy does not automatically improve performance: at short horizons, the best configuration uses a one-step high-level horizon, while longer horizons reveal a mismatch between the learned high-level action space and the inference-time search distribution. Experiments with true future latent subgoals show that the frozen low-level controller can execute well-aligned intermediate targets, indicating that high-level subgoal generation is the main bottleneck. Unconstrained search can select latent macro-actions that appear favorable under the learned model but produce poor control targets. Constraining search around macro-actions encoded from training trajectories, with appropriate subgoal execution timing, recovers useful hierarchical regimes, improving over flat LeWM by +11.3 percentage points at medium-range horizons and +14.7 percentage points at the longest PushT horizon. Overall, temporal abstraction can benefit compact frozen LeWM, but only when high-level search remains compatible with the low-level controller",
      "published": "2026-07-14T09:18:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.12547v1",
      "pdf_url": "https://arxiv.org/pdf/2607.12547v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
