const PAPERS_DATA = {
  "last_updated": "2026-11-20 06:52:40 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning",
      "authors": [
        "Yuxin Jiang",
        "Wenqiang Sun",
        "Jiaqi Li",
        "Yuchao Gu",
        "Lvmin Zhang",
        "Ivor Tsang",
        "Mike Zheng Shou"
      ],
      "abstract": "Videos reveal how the world changes, but rarely the actions that cause those changes. Latent action learning recovers such supervision from visual transitions, yet often learns the underlying dynamics largely from scratch. We ask whether a pretrained video generator can instead provide the prior for both inverse and forward dynamics. We introduce O$^{2}$LaF, built on a simple principle: one prior, two dynamics. We initialize both models from the same pretrained video DiT and adapt them to their distinct roles: a feed-forward transition encoder with learnable action queries for inverse dynamics, and a causal, action-conditioned video generator for forward dynamics. By controlling information access and optimizing both dynamics end-to-end through a compact bottleneck, O$^{2}$LaF learns transferable actions from unlabeled videos. Across navigation and robotic manipulation, O$^{2}$LaF sets a new state of the art in few-shot action probing and enables stronger zero-shot action transfer across visual contexts and embodiments. Beyond generative video priors, we further extend our framework to predictive representation priors such as V-JEPA2, achieving competitive performance. Our findings suggest a broader paradigm for latent-action learning: repurposing pretrained video models to discover transferable actions and predict their consequences.",
      "published": "2026-11-19T18:58:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.26617v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26617v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Active Feature Acquisition by Distilling an Oracle Teacher",
      "authors": [
        "Yang Meng",
        "Zixin Ding",
        "Robert L. Grossman",
        "Yuxin Chen"
      ],
      "abstract": "Acquiring every feature of an instance is often costly: a measurement can carry a monetary price, a time delay, or a privacy cost. Active Feature Acquisition (AFA) addresses this by learning a policy that decides, per instance, which features to acquire under a budget. Existing AFA methods train this policy by optimizing a self-constructed utility target rather than distilling an explicit teacher. We construct this teacher explicitly. At training time the label and the unobserved feature values are available, and together they determine the teacher in closed form: an oracle that ranks every candidate by its effect on the downstream predictor's error. OracleImitator distills this ranking into a deployable policy by soft-target imitation. Across multimodal medical diagnosis and the AFABench benchmark, OracleImitator matches or outperforms existing AFA methods across the budget range, recovering 90% of the attainable accuracy gain at under a quarter of the acquisition cost. Because the teacher is an explicit algorithmic object rather than an implicit optimization target, we can also measure directly how much of its ranking is recoverable by a policy restricted to what is visible at deployment.",
      "published": "2026-11-19T18:52:57Z",
      "abstract_url": "http://arxiv.org/abs/2611.26598v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26598v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning",
      "authors": [
        "Terence D. Agbeyegbe",
        "Toneukarin G. Agbeyegbe"
      ],
      "abstract": "Self-consistency certification treats repeated sampling from a language model as sequential evidence accumulation and stops when an anytime-valid e-process crosses a threshold, guaranteeing finite-sample Type I error control by Ville's inequality. Prior constructions assume sampled rollouts are conditionally independent. We show this assumption fails in practice: reasoning streams are positively correlated through shared reasoning prefixes or across-prompt heterogeneity, changing not merely certificate efficiency but the hypothesis certified. Modeling dependence via a per-unit latent state (the random-effects / intraclass-correlation structure) induces exchangeable, equi-correlated increments with correlation $ρ$. Correlation splits the single null into a realized null about the drawn belief and a population null about average model behavior, which coincide only when $ρ=0$. We show: (i) the standard betting e-process remains anytime-valid for the realized null, not the population null; (ii) read as a population-null test, a suitable mixture has asymptotic crossing probability at least $π=P(μ(Z)>0)$, potentially exceeding nominal $ε$; (iii) the within-unit mean's effective sample size saturates at $1/ρ$, so extra within-unit rollouts cannot restore population validity; and (iv) a prompt-level e-process restores population validity, with optimal allocation at one rollout per prompt under a pure rollout budget. Simulations and Claude Haiku 4.5 rollouts confirm this regime dependence: on GSM8K ($\\hatρ=0.435$, $\\hatθ=+0.96$) the gap is dormant, whereas on AIME ($\\hatρ=0.480$, $\\hatθ=-0.15$) the single-unit rule crosses on 24% of prompts at nominal $ε=0.05$, while the prompt-level procedure declines (0.00). This correctness result identifies which reliability claim an anytime-valid certificate actually supports under dependence.",
      "published": "2026-11-19T18:47:51Z",
      "abstract_url": "http://arxiv.org/abs/2611.26592v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26592v1",
      "categories": [
        "stat.ME",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Probing the \"Conscious\" Mind: Selective Access and Global Workspace in Vision-Language Models",
      "authors": [
        "Jiahao Huang",
        "Jun Wang"
      ],
      "abstract": "Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated in a shared workspace, and made available to specialised processes. We introduce the Multimodal Concept Lens (MC-Lens), a multimodal Jacobian lens, to study these functions in VLM concept processing. Its vision encoder and language decoder lenses map intermediate states to a shared semantic space defined by the decoder's output vocabulary to compare concept accessibility across modalities. MC-Lens supports internal concept interventions to examine the role of accessed concepts in downstream computation. (i) Accessibility and selective access. Multimodal concept information is readable in intermediate states and selectively accessible as tasks require. We extend \"readout carriers\" from one name token to a token set to better capture and express concept information and support counterfactual experiments. (ii) Concepts in downstream computation. Counterfactual experiments show that accessed concept information participates in downstream computation and reasoning. (iii) Development of cross-modal alignment and selective access. We develop two J-space metrics tracking the development of cross-modal concept alignment more clearly than hidden state cosine in controlled VLM training. The experiment also shows that reliable selective access develops later than concept accessibility. (iv) Shared functions across models. Across VLM families and sizes, we find shared patterns of concept accessibility, query-dependent selectivity and cross-task intervention effects despite differences in internal organisation. These findings offer an experimental basis to study GWT in multimodal artificial systems and how it can guide cross-modal alignment improvements.",
      "published": "2026-11-19T18:46:50Z",
      "abstract_url": "http://arxiv.org/abs/2611.26590v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26590v1",
      "categories": [
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Replication of Agentic Security Events: Audit and Investigation (ROGUE-AI)",
      "authors": [
        "Mengxiao Wang",
        "Mahbub Alam",
        "Kalyan Nakka",
        "Nitesh Saxena"
      ],
      "abstract": "Agents based on large language models (LLMs) can now delete files and send payments on behalf of their users. News media reports routinely describe failures of these agents (e.g., Replit's coding agent deleting a production database), but a report often only says what went wrong and rarely (in)validates the incident or finds its root cause. Incident catalogs collect these reports, and benchmarks record whether agents fail, but neither replicates an incident nor tests the cause its report might identify. In this paper, we present ROGUE-AI (Replication of Agentic Security Events: Audit and Investigation), a multi-agent framework that closes the above gap and rebuilds each reported incident offline together with a matched control that lacks only the blamed cause, and runs agents on both. Our results suggest that ROGUE-AI is able to reproduce the reported harm in 89% of its 108 attempted replications. Agents built on five current GPT, Claude, Gemini and DeepSeek models made the reported mistake in 18% to 33% of runs, against at most 1% without the blamed cause. Of the 72 incidents with real-world harm, 62 had no underlying attacker, and in these, agents did what their task asked for but missed an ordinary hazard, such as a database shared with the live site. Replications without an attacker caused more harm than replications of prompt injection for all five models. Defenses against prompt injection look for injected instructions, but these replications contain none. Further, a warning after each tool output that names the hazard and its consequence lowered harm for all five models, whereas a note of the same length that does not mention the hazard left most of the harm in place. Built from real incidents, the ROGUE-AI replications form a reusable audit framework that vendors can rerun before each new release.",
      "published": "2026-11-19T18:23:35Z",
      "abstract_url": "http://arxiv.org/abs/2611.26553v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26553v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models",
      "authors": [
        "Md. Abdullah Al Zami",
        "Suraiya Shanaz",
        "Mahdi Hasan",
        "Shoaib Ahmed Dipu",
        "Farig Yousuf Sadeque",
        "Md. Sabbir Ahmed"
      ],
      "abstract": "Linear probes are the cheapest LLM lie detectors but often fail to transfer across kinds of lie. Label-free directions from honest and deceptive role-play are untested at 70B scale and on lies about knowledge, where a lie and an honest denial can be identical. We widen a published persona bank with roles that deny knowledge they have, plus honest counterparts. At each model's middle layer the resulting persona direction ranks lies across Liars' Bench better than single-dataset probes on three of its four models; it fails on Qwen, even when Qwen stays in role far more often. On Llama-3.3-70B-Instruct it beats directions from a random persona bank, but its gain over the published direction is within the range of random widenings. On lies about knowledge, persona directions mainly track a reply's wording: where lies and honest denials share their words, our three main contrasts stay inside the random-direction band on Llama and Qwen and reach only its edge on Gemma, though on Mistral one clears it. A knowledge direction contrasts questions the model can answer with ones nobody can, each followed by the same scripted denial. It separates lies from identical honest denials on two models and narrowly on a third, but not on the fourth; its other results partly depend on the benchmark's wording. On the Gender-Secret fine-tune it clears the band by a few hundredths on Llama and fails on Mistral. The sum of the two directions ranks lies better than a supervised probe at the same depth on all four models (on one, both near chance) but is level with the persona direction on Llama's average and behind that probe on held-out lies. More varied role-play questions do not help the persona direction on Liars' Bench.",
      "published": "2026-11-19T18:20:01Z",
      "abstract_url": "http://arxiv.org/abs/2611.26545v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26545v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models",
      "authors": [
        "Chang Yang",
        "Yao Xuan",
        "Wendan Yan",
        "Nhat Le",
        "Jiang Liu",
        "Longhao Jin",
        "Zhen Peggy Yao",
        "Alireza Vahdatpour",
        "Djordje Gligorijevic"
      ],
      "abstract": "Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system improves normalized entropy by 0.31% on average within each model's budget, and the trimming step cuts pipeline cost by 2.3% with no regression beyond measurement noise. The optimized models were launched, and in online A/B tests they improved a primary online metric by 0.20%.",
      "published": "2026-11-19T18:12:04Z",
      "abstract_url": "http://arxiv.org/abs/2611.26531v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26531v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes",
      "authors": [
        "Francesco Caporali",
        "Jason M. Altschuler",
        "Boris Hanin"
      ],
      "abstract": "We present a new adaptive stepsize rule that biases gradient descent toward flatter minima in both theory and practice, and establish the first uniform-in-time guarantees for both sharpness and loss reduction for a broad class of locally quadratic objectives. Our critically damped Polyak (CDP) stepsizes multiply the classical Polyak rule by $4$ and a mild damping factor. The factor $4$ is critical in that it produces equioscillations around the manifold of minimizers $\\mathcal{M}$ to leading order. Damping provably drives convergence to $\\mathcal{M}$. CDP thus operates at the edge of stability by adapting its stepsizes to local curvature. Once the loss is small, its projected iterates follow gradient descent on $\\log{λ_1(\\nabla^2 \\mathcal{L})}$ along $\\mathcal{M}$ to leading order, paralleling the flatness bias of normalized gradient descent (Arora et al., 2022). Unlike normalized gradient descent, CDP step lengths shrink with the distance to $\\mathcal{M}$, enabling convergence to an interpolating solution that locally minimizes sharpness at finite learning rate. We establish our theory for a broad class of non-negative losses with locally quadratic Morse-Bott structure near $\\mathcal{M}$, capturing regular interpolation regimes in smooth overparameterized neural networks. Empirically, CDP stepsizes produce a flattening effect similar to that of Sharpness-Aware Minimization (SAM, Foret et al., 2021), reducing sharpness even in settings beyond our theory, including nonlinear teacher-student networks and fine-tuning of pre-trained LLMs.",
      "published": "2026-11-19T17:35:37Z",
      "abstract_url": "http://arxiv.org/abs/2611.26475v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26475v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.OC"
      ]
    },
    {
      "title": "Reasoning-Trace Analysis Using Large Language Models for Interpretable Q-Matrix Construction",
      "authors": [
        "Jihong Zhang",
        "Zijing Jin",
        "Mingfeng Xue"
      ],
      "abstract": "Large language models (LLMs) can generate Q-matrices for diagnostic classification models that approximate expert specifications, but evaluations have focused on the final matrices, leaving the basis of individual item-attribute assignments unexamined. This opacity limits the responsible use of AI in measurement. We propose Reasoning-Trace Analysis (RTA), a framework in which an LLM generates attribute-labeled reasoning steps through chain-of-thought (CoT) prompting and these steps are analyzed to reconstruct, compare, and inspect each assignment. RTA is intended to supplement expert review and empirical validation. In a simulation study with perturbed fraction subtraction items, four LLMs recovered the required attributes and withheld attributes the items did not require. In an empirical study with 11 fraction subtraction items, responses from 536 examinees, four LLMs, up to three temperatures, and 100 replications per condition, CoT-derived Q-matrices overlapped substantially with the reference Q-matrix (mean overlap .809) but less than directly generated Q-matrices (.877), and they varied more across replications. Inspection of the traces linked disagreements to item-type coding, labeling rules, and how operations were expressed as steps. Q-matrices with similar overlap differed in fit to examinee responses. RTA provides inspectable evidence that matrix-level agreement and fit do not.",
      "published": "2026-11-19T17:03:16Z",
      "abstract_url": "http://arxiv.org/abs/2611.26422v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26422v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling",
      "authors": [
        "Yuchen Zhou",
        "Lili Zheng",
        "Yuxin Chen",
        "Gen Li"
      ],
      "abstract": "Flow matching is a versatile framework for generative modeling that transports a source distribution to a target data distribution. A key advantage is its flexibility in the choice of source distribution, such as heavy-tailed and multimodal sources. Yet finite-step sampling guarantees for non-Gaussian sources remain limited. We establish a nonasymptotic convergence theory for Euler discretization of flow matching with linear interpolation. Suppose the source distribution on $\\mathbb{R}^d$ has an everywhere positive, continuously differentiable density, finite total variance $\\mathsf{TotVar}_{\\mathsf{base}}$, and an $L$-Lipschitz score. Using the population velocity field and a suitable time grid, this sampler achieves total variation error at most $\\varepsilon$ in $\\widetilde{O}(Ld\\mathsf{TotVar}_{\\mathsf{base}}/\\varepsilon)$ iterations, provided the target distribution satisfies a mild second-moment condition. These assumptions cover, among many examples, source distributions such as multivariate Student's $t$ distributions with more than two degrees of freedom and finite Gaussian mixtures with shared covariance, while accommodating target distributions with neither a smooth density nor bounded support.",
      "published": "2026-11-19T16:54:03Z",
      "abstract_url": "http://arxiv.org/abs/2611.26407v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26407v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.IT",
        "cs.LG"
      ]
    },
    {
      "title": "Training LLMs to Follow Reasoning Instructions with Minimal Edits",
      "authors": [
        "Alexandra Yakovleva",
        "Luat Gia Khoi Nguyen",
        "Juho Kannala",
        "Harri Valpola",
        "Alexander Ilin"
      ],
      "abstract": "Large language models increasingly rely on extended reasoning traces, yet the training signals used to shape this behavior remain coarse. Supervised fine-tuning on full trajectories forces the model to imitate an entire demonstration, overconstraining its output distribution, while trajectory-level reinforcement learning provides only weak credit assignment, rewarding or penalizing whole completions without identifying where they succeed or fail. We propose edit-based training, a general framework for learning from a model's own trajectories via localized textual corrections. Given a sampled completion, an editor identifies the first point of failure under a specified criterion and proposes a minimal replacement span. The model is then supervised only on these edited spans. By design, this concentrates supervision where the model's reasoning first breaks down and aims to preserve its existing reasoning patterns elsewhere. The method applies both offline and online, with the latter continually adapting supervision to the model's evolving failure modes. We instantiate this approach on reasoning instruction following with Qwen3-32B on the ReasonIF benchmark. Compared to full-trajectory imitation and trajectory-level reinforcement learning baselines, edit-based training substantially improves compliance with the reasoning instruction while largely preserving the model's skills on untargeted tasks. These results suggest that localized correction is an effective complement to these approaches for shaping how LLMs reason.",
      "published": "2026-11-19T16:18:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.26342v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26342v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ClosureBench: A Benchmark for Reasoning under Closure Assumptions",
      "authors": [
        "Francesco Belardinelli",
        "Flavio Corradini",
        "Matteo Leonesi"
      ],
      "abstract": "AI agents built on large language models (LLMs) often answer questions from incomplete records. Missing information might be interpreted as unknown or false, depending on the completeness of the source. Current evaluations fail to distinguish these two aspects, as they are usually interwoven with other reasoning demands. We present ClosureBench, a benchmark that tests whether LLMs follow the stated rules for dealing with missing information. Every base scenario is tested under OWA, CWA, and LCWA, and only the treatment of missing information changes between variants. ClosureBench provides a verified base benchmark as well as agent extensions for action choice, source scope, and updates. A model is considered correct on a base scenario only if it provides correct answers in all of its closure variants. Across 960 instances, overall accuracy can mask substantial closure mistakes. Models achieve overall accuracy from 71.18% to 99.86% across replicated runs, but their core switch accuracy, measured on the scenarios whose gold label actually changes with the contract, varies between 18.23% and 99.31%. Large discrepancies emerge under LCWA. For example, some agents choose unknown despite the prompt indicating that missing information should be treated as false. Our findings demonstrate that evaluations of AI agents should not only measure whether agents can reason over existing facts but also whether they use the appropriate assumption when information is absent.",
      "published": "2026-11-19T16:15:04Z",
      "abstract_url": "http://arxiv.org/abs/2611.26335v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26335v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Control-Theoretic Approximation to Predictive Coding Dynamics",
      "authors": [
        "Ryan Fayyazi",
        "Kyle Daruwalla",
        "Mitra Javadzadeh"
      ],
      "abstract": "Backpropagation remains the dominant method for training deep neural networks, but its reliance on non-local computations has motivated the search for biologically plausible alternatives. Two prominent frameworks, predictive coding (PC) and deep feedback control (DFC), offer distinct approaches to local learning, yet their relationship remains unclear. Here, we show that DFC arises as a local approximation to predictive coding dynamics near fixed points, with exact equivalence in shallow linear networks and systematic deviations arising with depth, nonlinearity, and distance from equilibrium. We characterize how this approximation, which we call Approx PC, deviates from PC in its activity dynamics, and show that these deviations can be mitigated by relinearizing the feedback controller and approaching the output target through a sequence of intermediate nudges. In networks trained on MNIST and FashionMNIST, Approx PC reaches task performance comparable to PC trained with the same nudges, and weights learned with it transfer directly to PC inference. Our results unify energy-based and control-theoretic perspectives on learning and open the door to applying tools from control theory to the training of predictive coding networks.",
      "published": "2026-11-19T16:04:50Z",
      "abstract_url": "http://arxiv.org/abs/2611.26319v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26319v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.NC"
      ]
    },
    {
      "title": "Client Exits - A Way to Increase LLM Transparency on the Client Side",
      "authors": [
        "Marc Jansen"
      ],
      "abstract": "Textual recommendations generated by large language models (LLMs) can be difficult to inspect. This paper introduces \\emph{Client Exits}, a lightweight mechanism for enriching LLM responses while separating semantic intent from client-specific presentation. The model inserts predefined, machine-interpretable markers into its output, and the receiving client decides whether and how to realize the corresponding enrichments. This allows supplementary information to be presented as graphics, text, speech, or other modalities without prescribing a concrete interface. We implement a prototype combining a finetuned Qwen model with a modified LibreChat client. Using the Question--Option--Criteria (QOC) approach including three XML-based client exits. We evaluate two QOC-oriented model variants trained on essentially the same examples, with client-exit annotations added for one variant. Across 100 test questions, the client-exit model includes all three markers in every response. An LLM judge, informed about the mechanism without being instructed to favor it, prefers this model's answers in 66\\% of the comparisons. These results demonstrate successful marker generation on the evaluated test set and support the feasibility of client-controlled enrichment.",
      "published": "2026-11-19T15:58:09Z",
      "abstract_url": "http://arxiv.org/abs/2611.26309v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26309v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "What If Chain-of-Thought Is Not Enough?",
      "authors": [
        "Marc Jansen"
      ],
      "abstract": "Reasoning capabilities in large language models have evolved from prompting-based methods to internalized model behaviors. In this paper, we investigate whether the internal thinking model of an LLM can be deliberately changed and whether such a change is beneficial. Instead of relying on Chain-of-Thought, we train models to follow Question--Option--Criteria (QOC), a structured decision process that decomposes a task into explicit questions, competing options, and evaluation criteria. We present a three-generation development pipeline that ranges from a basic QOC decision core to richer answer synthesis and statistically enriched QOC reasoning. We then evaluate the resulting models with respect to both internal consistency and output quality. Across the reported experiments, QOC-based fine-tuning improves answer quality relative to non-fine-tuned base models and yields particularly strong gains once models are large enough to execute the underlying QOC computations reliably. The results suggest that the internal thinking model of an LLM is trainable and that structured alternatives to Chain-of-Thought can provide a promising path toward more reliable, interpretable, and task-adapted decision support.",
      "published": "2026-11-19T15:43:51Z",
      "abstract_url": "http://arxiv.org/abs/2611.26282v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26282v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "The Stability Gap in Continual Reinforcement Learning",
      "authors": [
        "Sebastian Pusch",
        "Matthia Sabatelli",
        "Gido M. van de Ven"
      ],
      "abstract": "Continual reinforcement learning (CRL) aims to enable agents to adapt to changing tasks while preserving previously acquired capabilities. Evaluation often relies on final aggregate metrics that can hide temporary performance degradation during adaptation. This work introduces a simple CRL benchmark that varies observation representations while preserving environment dynamics and rewards, supporting proof-of-concept demonstrations, inexpensive experimentation, and easily reproducible sanity checks. Using this benchmark, we demonstrate the stability gap in CRL: a substantial drop in prior-task performance after the introduction of new tasks, followed by partial or complete recovery. This pattern occurs despite the agent having complete access to past tasks.",
      "published": "2026-11-19T15:26:41Z",
      "abstract_url": "http://arxiv.org/abs/2611.26252v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26252v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs",
      "authors": [
        "Yang Wan",
        "Linchao Zhu"
      ],
      "abstract": "AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the environment and executes the experiments that the tested agent designs, reading the raw materials of the laboratory and calling calculators that are built from its records. Compared with a language model that sees only the task description, the TwinLab agent almost eliminates readings that deviate from the real records and inconsistent answers to a repeated request, and closed-book probes control for answers recalled from memory. We find that these tasks remain hard for frontier models: the strongest, Claude Fable 5.2, averages a score of only 45.0, and we conduct an empirical study of where it fails.",
      "published": "2026-11-19T15:25:34Z",
      "abstract_url": "http://arxiv.org/abs/2611.26248v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26248v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction",
      "authors": [
        "Rafid Enayet",
        "Imtiaz Ahmed",
        "Md Habibor Rahman"
      ],
      "abstract": "Developing generalizable, data-efficient machine learning (ML) models for advanced manufacturing remains challenging because experimental datasets are limited and process-property relationships vary across materials and operating conditions. In laser powder bed fusion (LPBF), melt pool width, depth, and melting mode (conduction, transition, or keyhole) govern process stability and defect formation. Conventional ML models typically use dimensional process parameters, which represent laser-material interactions only implicitly and may limit cross-material transfer. This study evaluates whether two physics-informed dimensionless features, normalized enthalpy and Peclet number, combining laser processing conditions with material thermophysical properties, improve melt pool prediction over raw process parameters. We systematically benchmark seven models: four classical ML methods (Gaussian process, support vector machine, random forest, and XGBoost) and three pretrained tabular foundation models (TabPFN, TabDPT, and TabICLv2). Experiments span 316L stainless steel, IN625, IN718, and combined-material datasets, covering regression of melt pool width and depth and classification of melting modes. Dimensionless features yield comparable or superior predictive performance in most settings, with the largest gains in cross-material depth prediction (R^2 = 0.96). Tabular foundation models generally achieve stronger classification performance than classical baselines, particularly for multi-material datasets. These findings indicate that combining physically meaningful dimensionless representations with pretrained tabular models can improve data-efficient melt pool modeling and support more transferable process-window identification in LPBF.",
      "published": "2026-11-19T15:09:41Z",
      "abstract_url": "http://arxiv.org/abs/2611.26235v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26235v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Sightline: Calibrated Data-Flow Summaries for LLM-Integrated Applications",
      "authors": [
        "Zihao Xu",
        "Yuekang Li",
        "Ruitao Feng",
        "Haifeng Shen"
      ],
      "abstract": "Programs increasingly call large language models (LLMs), filling prompt placeholders with program values and consuming the output. Such a call crosses the boundary between natural language and programming language (NL/PL). To follow data across it, an analysis needs a data-flow summary of which placeholders the output uses, but no code describes what the model does. Analyses of these LLM-integrated applications therefore taint every output with every input, creating spurious flows, or ask a second model to label each call, which is costly. We prove, and mechanize in Lean 4, that a placeholder flows to the output of a transformer, to first order, only through the attention that reads it (no reading, no flow), and that a faint reading or sensitivity cannot certify that a placeholder does not flow. Unlike reading, the log-probability gradients and Fisher signals of existing methods vanish on confident outputs, and single-span removal fails on duplicated content. Our tool, Sightline, reads a recorded call in one forward pass, learns from labels which reads are uses, and calibrates the pruning to miss at most a chosen fraction $α$ of true flows, on average over call sites, when deployment is exchangeable with calibration. On 2,103 unseen repository families and four models, Sightline is the most accurate summary against the seven closest methods and a hidden-state probe. Calibrated at $α$ = 1%, it misses about 1% of true flows and prunes 43.9-65.6% of spurious flows, 1.6 to 2.6 times as many as the best baseline, and still 43.8-60.1% when calibrated on another collection. Through an open surrogate, it reads models treated as closed about as accurately as they read themselves, without their labels. In the information-flow control of an LLM agent, it removes about half of the labels from inputs that do not flow.",
      "published": "2026-11-19T15:00:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.26221v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26221v1",
      "categories": [
        "cs.PL",
        "cs.AI"
      ]
    },
    {
      "title": "Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy",
      "authors": [
        "Ilia Zavidnyi",
        "Alexander Kovrigin",
        "Arina Puchkova",
        "Kirill Gelvan",
        "Sergey Titov"
      ],
      "abstract": "Reasoning models often become more accurate when they can generate longer answers at test time. This behavior is known as test-time scaling. However, RL with verifiable rewards (RLVR) usually trains them with a fixed rollout cap. We find that this cap changes model behavior, not only training cost. After training, models use few additional tokens beyond the training budget. They outperform the base models near that budget, but their accuracy stops improving and falls below the base model at larger budgets. The rollout cap thus becomes a learned constraint, that limits model's test-time scaling. We call this effect budget internalization. We observe it across Qwen3.5-4B, Gemma4-E2B, and Nemotron-3-Nano-4B and two reasoning training tasks: mathematical reasoning and diverse, procedurally generated problems. This effect also transfers from math training to science and coding tasks. Training-objective level ablations on Qwen3.5-4B show that most changes recover little test-time scaling. Stronger KL regularization and lower training-data truncation preserve more scaling, but stronger KL reduces in-budget accuracy. Forcing an answer on truncated training rollouts largely removes budget internalization, but also removes most of the in-budget gain. SFT on artificially truncated traces reproduces budget internalization, whereas SFT on naturally short traces does not, which shows that the effect is not specific to RL. This result links budget internalization to length-censored post-training rather than to an RL-specific objective term. Thus, a rollout cap does not only control training cost. It can trade low-budget gains for lower performance when additional inference compute is available.",
      "published": "2026-11-19T14:20:41Z",
      "abstract_url": "http://arxiv.org/abs/2611.26172v1",
      "pdf_url": "https://arxiv.org/pdf/2611.26172v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
