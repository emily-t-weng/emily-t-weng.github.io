const PAPERS_DATA = {
  "last_updated": "2026-11-16 05:59:20 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "WeightBench: Evaluating Language Models as Weight-Space Programmers",
      "authors": [
        "Aayush Marasini",
        "Barsat Khadka",
        "Nick Rahimi"
      ],
      "abstract": "Recent work shows that large language models (LLMs) can operate directly on the parameters of other neural networks. They can read what a network was trained to do, fix a network that fails, and write working networks from scratch, all without gradient descent. This capability has been termed Weight-Space Programming (WSP). Exploring this new paradigm systematically requires a robust benchmark. We introduce WEIGHTBENCH, a benchmark that evaluates LLMs of increasing capability on 42 WSP tasks across five tracks: reading, fixing and writing weights, recognizing edits that are impossible, and data-free compression. The networks whose weights the LLMs read and edit range from 2K-parameter control policies to 7B-parameter language models. Experimental benchmark results show that frontier agents can already read small networks and write working control policies without ever running them, but cannot yet patch language models, compress them without data, or tell an impossible edit from a hard one. On all 42 tasks, Claude Opus 5.5 reaches a macro score of 0.535 (Sonnet 5.5 0.466, Haiku 5.5 0.402, DeepSeek v4-flash 0.215); on the 38 tasks it was run on, GPT-6.1-sol reaches 0.568 against Opus's 0.546. Code, the task registry, our OS-level sandbox harness, and the live leaderboard are publicly available at https://github.com/Aayush-Marasini/WeightBench.",
      "published": "2026-11-13T18:58:08Z",
      "abstract_url": "http://arxiv.org/abs/2611.18553v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18553v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond",
      "authors": [
        "Cole Becker",
        "Taylan Kargin",
        "Kimia Hassibi",
        "Asuman Ozdaglar",
        "Pablo A. Parrilo"
      ],
      "abstract": "Generative modeling based on stochastic interpolants requires specifying two key choices: an interpolant schedule bridging the initial and target distributions, and an efficient discretization of the resulting probability-flow ODE. In this work, we introduce a new framework for joint schedule/discretization design based on the analyticity properties of the (complexified) ODE trajectories, which in turn ensure good numerical behavior. Our formulation can be solved exactly via conformal mapping, yielding a natural family of variance-preserving elliptic schedules, parametrized by Jacobi elliptic functions. Pairing an elliptic schedule with a tailored discretization, we obtain Zoloflow, a new sampling algorithm closely connected to Zolotarev's optimal rational function approximants. Crucially, both the elliptic schedule and Zoloflow explicitly incorporate the condition number $κ$ of the target distribution and reduce to the classical trig schedule in the limit $κ\\rightarrow 1$. For Gaussian targets, Zoloflow achieves minimax optimality among all oblivious score-span algorithms, attaining total-variation error $\\varepsilon$ using $Θ(\\logκ\\,\\log(\\sqrt d/\\varepsilon))$ (exact) score queries. For the log-concave case, the elliptic schedule allows sampling with at most $\\varepsilon$ 2-Wasserstein-error by requiring merely $\\widetilde{O}(\\sqrt{d} \\, \\mathrm{polylog}(κ, 1/\\varepsilon))$ score evaluations. Complementing these theoretical results, we empirically evaluate the elliptic schedule on general (non-log-concave) target distributions, including detailed comparisons with alternative schedule proposals on both synthetic benchmarks and existing image generation datasets.",
      "published": "2026-11-13T18:49:36Z",
      "abstract_url": "http://arxiv.org/abs/2611.18542v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18542v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL",
      "authors": [
        "Zelin He",
        "Siyu Zhu",
        "Jacob Helwig",
        "Yujian Liu",
        "Li An",
        "Rohit Jain",
        "Shiyu Chang",
        "Alborz Geramifard"
      ],
      "abstract": "Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past trajectories, so the harness tracks how the model's behavior shifts. We study CodeSearch in depth and further analyze ALFWorld and BIRD SQL, with models from 1.5B to 8B. The proposed Compression and Steering mechanisms improve held-out performance over the fixed-harness baseline in most settings. These results suggest that a harness for agentic RL can be learned from the task's data and evolved with the model.",
      "published": "2026-11-13T18:45:49Z",
      "abstract_url": "http://arxiv.org/abs/2611.18533v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18533v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination",
      "authors": [
        "Mohamed Aghzal",
        "Mor Geva",
        "Gregory J. Stein",
        "Ziyu Yao"
      ],
      "abstract": "Mechanistic Interpretability (MI) has made significant progress in explaining the internal computations of Large Language Models (LLMs); however, these advances have yet to demonstrate comparable success in explaining the decision-making processes of LLM-based agents. This paper examines the challenges MI tools face when applied to realistic agentic failures. We focus on one such failure: web agents hallucinating actions that are not available on a website. We apply a series of mechanistic analyzes to Qwen2.5-7B-Instruct as it interacts with live websites. We find that a linear probe trained on the last prompt token can predict hallucinations before generation begins (0.80 AUC) and steering along a probe direction corrects 85\\% of hallucinations, but disrupts 7\\% of valid actions. However, these results only provide a partial explanation of why the agent hallucinates. The probe direction is sensitive to prompt formatting, losing much of its steering effect after a minor change to how actions are represented. Additionally, circuit identification can be difficult as the computation is distributed across multiple components and positions, and the information the agent relies on varies across examples. We discuss how these observations challenge the assumptions underlying standard MI methods for circuit discovery and feature identification, and outline future directions for making these methods better suited to explain failures in LLM-based agents.",
      "published": "2026-11-13T18:30:58Z",
      "abstract_url": "http://arxiv.org/abs/2611.18514v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18514v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading",
      "authors": [
        "Mohammed Efaz",
        "Udo Bub",
        "Itilekha Podder"
      ],
      "abstract": "This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining acceptable response quality. Importantly, the proposed progress-aware boundary mechanism is shown to prevent future content leakage without compromising retrieval quality and correctness. These findings suggest that a localized, privacy-centric approach can augment, rather than replace, the digital reading experience, offering a good alternative to cloud-dependent services. Therein, there remains a lack of amalgamate systems which combine progress-aware, privacy-preserving, and narrative eBook reading with AI support. It is in this very gap that the development of MeReader came to be; synthesizing said strands into an offline, book-constrained assistant focused on enhancing comprehension without compromising narrative continuity or user privacy.",
      "published": "2026-11-13T18:29:51Z",
      "abstract_url": "http://arxiv.org/abs/2611.18513v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18513v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Agentic AI for Discovering Fault-Tolerant Quantum Circuits",
      "authors": [
        "Aritra Lahiri",
        "Florian Marquardt"
      ],
      "abstract": "Scaling quantum computing toward practical quantum advantage is currently limited by the efficiency of circuits that operate on information encoded by quantum error-correcting codes. The full stack of quantum computing encompasses several tasks: preparation of logical code states, implementation of logical gates, logical routines on encoded qubits, etc. Yet, these circuits are largely designed by hand for individual cases, with some recent works incorporating machine learning for certain aspects. We formulate this problem as agentic evolutionary program synthesis, wherein a large language model, leveraging its pretraining knowledge and tool-use proficiency, and guided by a circuit verifier, evolves a program that defines a Clifford circuit for a wide range of curated tasks such as logical state preparation, logical gate design, and logical routines such as code-switching, code growth, etc. Our approach consistently finds fault-tolerant circuits for all of these tasks, each verified by an exhaustive check over faults.",
      "published": "2026-11-13T18:20:23Z",
      "abstract_url": "http://arxiv.org/abs/2611.18504v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18504v1",
      "categories": [
        "quant-ph",
        "cs.AI"
      ]
    },
    {
      "title": "Interactive Program Induction with Language Model Priors",
      "authors": [
        "Robin Shing Moon Chan",
        "Ryan Cotterell"
      ],
      "abstract": "Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algorithm and show that it terminates after $\\mathcal{O}(\\frac{1}γ\\log 1/p^0(f^\\star))$ informative oracle queries, and returns the target program $f^\\star$ up to approximate equivalence when no other class outweighs it under the prior. The algorithm uses a language model as the prior under a PAC-style relaxation of program equivalence, where the query-selection objective is estimated using importance sampling. On the CodeARC benchmark, the algorithm improves accuracy over the LM only agent by up to 22 percentage points while observing up to 10.5 fewer input-output examples.",
      "published": "2026-11-13T17:59:55Z",
      "abstract_url": "http://arxiv.org/abs/2611.18485v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18485v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles",
      "authors": [
        "Wang Ma",
        "Debarun Bhattacharjya",
        "Junkyu Lee",
        "Nhan H Pham",
        "Harsha Kokel",
        "Qiang Ji"
      ],
      "abstract": "Reliable use of large language models (LLMs) requires uncertainty estimates that separate ambiguity intrinsic to a prompt from genuine knowledge gaps. Bayesian and deep-ensemble methods deliver this aleatoric/epistemic decomposition in principle, but are computationally prohibitive at LLM scale and require white-box access. Existing black-box methods, in contrast, summarize to a single scalar that cannot make the distinction. We introduce a two-level ensemble that bridges this gap. Stochastic decoding of a fixed prompt forms the inner ensemble; meaning-preserving semantic perturbations of the prompt form the outer ensemble. Variability is measured in a continuous embedding space, and the law of total covariance gives an exact within-/between-perturbation decomposition to estimate aleatoric and epistemic uncertainty (AU/EU); it requires only black-box sampling access. On top of this estimator, we contribute three theoretical results that ground the framework: (i) a finite-sample bias identity for nested Monte Carlo with a closed-form bias-corrected epistemic estimator; (ii) a Bayesian bridge bounding the gap between perturbation-population AU/EU and Bayesian targets; and (iii) a separation result identifying a class of failure modes that any consistency-only black-box method provably misses while perturbation-based EU detects. Across five short QA benchmarks and four instruction-tuned LLMs spanning two families and two scales, our estimator is competitive with strong white-box baselines, and at a matched generation budget it matches leading black-box methods on error ranking while decomposing AU and EU; controlled paired interventions show that AU and EU respond selectively to distinct, actionable sources of failure: prompt ambiguity and model knowledge gaps; and a long-form pilot on the NQ long-answer split shows the framework extends to claim-level factuality without retraining.",
      "published": "2026-11-13T17:26:39Z",
      "abstract_url": "http://arxiv.org/abs/2611.18453v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18453v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning",
      "authors": [
        "Tao Hu",
        "Ling-Xiang Zhang",
        "Da-Wei Zhou"
      ],
      "abstract": "Multimodal continual instruction tuning requires multimodal large language models to sequentially acquire new capabilities without catastrophic forgetting. Mixture-of-experts (MoE) architectures offer a promising way to mitigate task interference through expert specialization and combination. However, while existing MoE-based methods avoid overwriting shared parameters by allocating separate experts for each task, this progressive expansion incurs severe cross-task redundancy and linear parameter growth. Specifically, experts across tasks often learn overlapping functions alongside task-specific capabilities. For instance, ScienceQA and TextVQA both locate relevant image regions. To curb parameter redundancy without inducing forgetting, we propose Consolidating and Preserving MoE (CoPre-MoE). For each incoming task, CoPre-MoE first learns task-specific MoE to capture specialized capabilities. To eliminate redundancy and reduce parameter growth across tasks, we design a shared-and-private expert architecture. Specifically, we first evaluate functional overlap to selectively consolidate overlapping experts across tasks into a fixed-capacity generic expert pool. We then factorize the remaining task-specific experts into shared projection bases and compact task-specific cores within a private expert pool. To mitigate forgetting, we construct the router by pairing each expert with learnable retrieval keys, where historical keys remain frozen during continual updates while newly added keys are regularized with an orthogonality penalty. At inference, input-dependent routing dynamically combines both pools without test-time task labels. Extensive experiments demonstrate that CoPre-MoE achieves superior performance while substantially reducing parameter overhead.",
      "published": "2026-11-13T16:58:35Z",
      "abstract_url": "http://arxiv.org/abs/2611.18417v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18417v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment",
      "authors": [
        "Suvarna Kadam",
        "Soran Parsa"
      ],
      "abstract": "Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated data, and (f) post-training, adaptation to the target robot. Deployment covers (g) real-time execution, keeping pace with the robot. We analyse 22 models on these axes and report their training data, model size, inference speed and hardware. Three trends emerge: (i) training data has broadened, from about 10,000 hours of teleoperated robot data ($π_0$) to 23,000 community-collected episodes (SmolVLA) and 20,000 hours of egocentric human video (GR00T N1.7); (ii) model sizes have split: most recent models have 3-7 billion parameters, while a new group under one billion runs on consumer GPUs; and (iii) generalist VLAs are emerging that handle both manipulation and navigation, such as UniVLA and Qwen-VLA. We also compare VLAs with world-action models, which predict future observations as well as actions. We review evidence that some VLAs with near-perfect benchmark scores fail when objects or the camera move slightly, and open up further research questions.",
      "published": "2026-11-13T16:17:00Z",
      "abstract_url": "http://arxiv.org/abs/2611.18372v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18372v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.ET",
        "cs.LG"
      ]
    },
    {
      "title": "AI for Parameter Selection in Sample-based Quantum Diagonalization",
      "authors": [
        "Doriela Grabocka",
        "Dimitrios Christofidellis",
        "Federico Zipoli",
        "Anna van Elst",
        "Patrick Ruch",
        "Antonio Foncubierta-Rodríguez",
        "Siddhartha Mishra",
        "Matteo Manica"
      ],
      "abstract": "Sample-based Quantum Diagonalization (SQD) methods provide a way to compute molecular ground-state energies, but their accuracy and runtime depend sensitively on configuration choices such as the subspace dimension, propagation timestep, and the allocation of a fixed sampling budget across circuits, for which no formal selection method exists. In this work, we explore whether oracle models can predict the quantum runtime and approximation error from the Hamiltonian and the candidate configuration. Using synthetic Hamiltonians with controllable spectral properties, we show that machine-learned models can map a Hamiltonian directly to the best-ranked configuration within a predefined grid for the SKQD algorithm, validated against classical Krylov propagation, achieving strong rank correlations between true and predicted error and runtime, Spearman $ρ= 0.969$ and $ρ= 0.905$, respectively. Extending the approach to 50 molecular Hamiltonians (20-50 qubits), with reference ground-state energies from classical calculations, we train regressors that predict the simulation error and runtime for a given molecule--configuration pair ahead of execution, and classifiers to distinguish, from 600 SqDRIFT quantum-hardware runs, whether a configuration achieves a ground-state energy within chemical accuracy ($1.6\\times10^{-3}$ Ha $\\approx 1$ kcal/mol). For the molecular Hamiltonians, the models additionally provide an a priori feasibility and resource estimate, predicting whether a molecule can reach chemical accuracy on current hardware and at what runtime, thus correctly classifying held-out molecules at the molecule level, achieving balanced accuracy 61.8% in-distribution and 100% for five larger held-out molecules.",
      "published": "2026-11-13T15:27:18Z",
      "abstract_url": "http://arxiv.org/abs/2611.18307v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18307v1",
      "categories": [
        "physics.chem-ph",
        "cs.AI"
      ]
    },
    {
      "title": "Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering",
      "authors": [
        "Nima Eshraghi",
        "Lovedeep Gondara",
        "Sagarika Suresh",
        "Xiaotong Xu",
        "Jithin Pradeep",
        "Fanny Chevalier"
      ],
      "abstract": "Activation steering via sparse autoencoders (SAEs) enables behavioral control of large language models without task-specific fine-tuning. Yet, standard methods apply the steering signal at every generated token, displacing representations from the distribution the model was trained on and eroding generation quality. We show that \\emph{when} along a generation the intervention is applied is a decisive axis of control. We cast steering as a per-token allocation problem under a fixed budget of steerable tokens, so that methods differ only in which tokens they select, making allocation the sole variable. Within this framework we introduce two gates. The first, Contrastive Feature Projection (CFP), is memoryless and selects the tokens least aligned with the target behavior through a sparse, interpretable feature projection. The second, Closed-Loop Adaptive Bernoulli Steering (CLABS), is trajectory aware. It treats allocation as closed-loop feedback control, tracking the alignment a generation has accumulated so far and spending budget where the trajectory falls behind. On a linear steering-response model we derive an explicit gain region that guarantees closed-loop stability of CLABS and show that its integral term reduces the steady-state tracking offset. Empirically, CFP and CLABS achieve the largest behavioral shift among budgeted gates at matched budgets, with CLABS strongest under the tight budgets where allocation matters most, while best preserving fluency.",
      "published": "2026-11-13T15:06:43Z",
      "abstract_url": "http://arxiv.org/abs/2611.18274v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18274v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "From Bytes to Behavior: Request-Level Attribution from Live GPU Hidden States",
      "authors": [
        "Jonathan Pan",
        "Yidi Yuan",
        "Xin Wei Chia",
        "Xinhui Su",
        "Vincent Tan",
        "Brenda Ong"
      ],
      "abstract": "Existing memory forensics for machine learning systems typically targets static artifacts, such as model weights or prompt logs, rather than the transient computational states of a live inference process. This gap limits an examiner's ability to analyze the specific behavior of a running large language model (LLM) service that cannot be restarted or modified. We present a method for acquiring per-layer residual-stream activations from a live GPU process using a non-modifying debugger, without requiring prior knowledge of the model's architecture parameters or weights. The approach employs a general six-stage process with platform-specific adapters, utilizing either a PyTorch allocator clock or a kernel-launch clock to identify and read tensor allocations from video memory. We validate this acquisition on 33 open-source decoder-only models across nine architecture classes, demonstrating that the recovered bytes are byte-exact and usable for downstream analysis. As a use case, we apply linear probes trained on standard framework activations to the recovered states, successfully distinguishing refusal from compliance with an AUROC of 0.88 on held-out prompts. This work establishes that live, observational access to intermediate hidden states is feasible on named inference platforms, providing a new source of evidence for examining the operational behavior of deployed AI systems.",
      "published": "2026-11-13T15:04:05Z",
      "abstract_url": "http://arxiv.org/abs/2611.18272v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18272v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization",
      "authors": [
        "Pei Tang",
        "Songtao Lu"
      ],
      "abstract": "Learning across multiple inherently conflicting objectives requires selecting desirable trade-offs from the Pareto set. In this work, we study preference-guided multi-objective optimization, where a predefined preference function is minimized over the weak Pareto set of multiple nonconvex objectives. Leveraging a merit function that vanishes on the weak Pareto set, we propose a penalty-based min-max-min formulation, termed the M$^3$ problem, which unifies preference optimization and the enforcement of weak Pareto optimality in a single objective. We further develop ParetoM$^3$, a single-loop first-order algorithm for computing stationary solutions of the resulting M$^3$ problem. Under a local Kurdyka-Lojasiewicz assumption, we prove that ParetoM$^3$ finds an $ε$-stationary point of the M$^3$ problem that is also approximately weakly Pareto optimal for the original multi-objective problem when the penalty parameter is sufficiently large, with explicit non-asymptotic convergence rates. We also show that this stationarity notion recovers existing optimality criteria, including approximate preference stationarity and approximate Karush-Kuhn-Tucker conditions, under the same assumptions used in prior works. Experiments on a synthetic benchmark, image classification, regression, and math reasoning demonstrate that ParetoM$^3$ reliably attains preferred weak Pareto optimal solutions while achieving strong practical performance. Code is available at https://github.com/tpgoodboy/ParetoM3.",
      "published": "2026-11-13T14:49:23Z",
      "abstract_url": "http://arxiv.org/abs/2611.18252v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18252v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.OC"
      ]
    },
    {
      "title": "Dancing Tokens on the Vocabulary Network from LLM Generation",
      "authors": [
        "Yuan-Chao Hu"
      ],
      "abstract": "Large language models (LLMs) generate text one token at a time, yet how a particular generation relates to the geometry learned in pre-training remains unclear. The fundamental explainability of LLMs is elusive. Here, we build a similarity network from the input embeddings of six open LLMs and map generation onto it. The token-embedding network is scale-free while heavy-tailed, as are random embeddings with matched covariance, but far more fragmented. Distillation barely changes this topology because input embeddings are largely inherited. The knowledge space is thus deterministic from pre-training. Consecutive generated tokens are rarely graph neighbours and, in most models, no more often than in shuffled order, so generation is not a walk on the predefined network, although the tokens it visits are concentrated within it. Through the logit lens in a layer- wise manner, tokens settle late in depth in most models, and repetitive models remain confined to small domain-specific sub-graphs. These observables could serve as inference-time monitors of degeneration. We thus provide a fresh perspective on understanding LLM generation in the microscopic way from node or link prediction by graph learning.",
      "published": "2026-11-13T14:43:55Z",
      "abstract_url": "http://arxiv.org/abs/2611.18248v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18248v1",
      "categories": [
        "cs.AI",
        "cond-mat.dis-nn"
      ]
    },
    {
      "title": "Personalization of Basic and Moral Values Guiding LLM-Controlled Assistant Robots",
      "authors": [
        "Mateusz Wozniak",
        "Qi Gao",
        "Agnieszka Wykowska"
      ],
      "abstract": "Personal assistant robots powered by large language models (LLMs) are likely to enter everyday life within the next decade, but the values that should guide their behaviour remain largely unspecified. Current alignment practice steers a single model toward an aggregate of human values, ignoring that individual users may prefer a robot whose value profile differs from their own. We administer parallel self- and robot-targeted versions of three established psycho-metric instruments - the simplified Schwartz Values Survey (SVS), the Portrait Values Questionnaire (PVQ-21), and the Moral Foundations Questionnaire (MFQ) - to potential future users (N=22), capturing both their personal values and the values they would want an assistant robot to embody. We compare the three instruments as personalization tools using the UEQ-S and benchmark fifteen contemporary LLMs against the resulting profiles. Participants wanted their robot to score markedly lower than themselves on personal-focus values and higher on social-focus values; moral profiles were nearly identical. LLMs prompted to act as robots shifted only weakly toward the desired profile, indicating that default model behaviour and prompt engineering are no substitute for explicit value personalization.",
      "published": "2026-11-13T14:28:47Z",
      "abstract_url": "http://arxiv.org/abs/2611.18230v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18230v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer",
      "authors": [
        "Yue Xu",
        "Kaiyun Wei",
        "Wenying Duan",
        "Zichen Xu",
        "Shuai Ma",
        "Xiaoxi He"
      ],
      "abstract": "Spatio-temporal foundation models (STFMs) require large-scale training corpora, yet synchronized observations and spatial metadata are costly to acquire and curate, limiting the availability of suitable data. By contrast, time-series data are abundant and easier to collect. Since observations at each location form a time series, this motivates a natural question: can abundant time-series data enhance STFMs when spatio-temporal observations are scarce? To bridge this gap, we propose T2ST, a data-efficient training framework that augments limited spatio-temporal observations with selected auxiliary time series and transfers knowledge from a frozen time-series foundation model. Our key insight is that temporal pattern learning can benefit from abundant external time series, while spatial dependency learning remains grounded in real spatio-temporal observations. Guided by this insight, T2ST retrieves matched sequences to reinforce sparsely observed temporal patterns and selects reliable complementary sequences to broaden temporal coverage. Temporal relation distillation aligns relationships among historical patches, while multi-scale prediction distillation transfers forecasting behavior at fine and coarse temporal resolutions, enabling knowledge transfer across heterogeneous architectures. Using only 10\\% of the original spatio-temporal data, T2ST enables STFMs to achieve overall forecasting performance competitive with or superior to their full-data counterparts across six long-term and twelve short-term forecasting datasets. These results establish a new paradigm for data-efficient training of spatio-temporal foundation models. Code is available at https://anonymous.4open.science/r/T2ST-65F3/.",
      "published": "2026-11-13T14:05:40Z",
      "abstract_url": "http://arxiv.org/abs/2611.18199v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18199v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries",
      "authors": [
        "Joey Chan",
        "Haoyu Pan",
        "Zhen Chen",
        "ErShun Pan"
      ],
      "abstract": "Fast charging of lithium-ion batteries is a multi-objective control problem that requires reducing charging time and sustaining high energy throughput while suppressing accelerated degradation and safety-critical electrochemical reactions. Conventional protocols are easy to implement but have limited ability to adapt to variations in battery chemistry, operating conditions, and aging state. Model-based optimization and conventional data-driven control often require accurate models, costly iterative search, or task-specific policies. We develop a large language model (LLM)-empowered intelligent agent for adaptive battery fast-charging control. In this closed-loop framework, the agent interprets structured battery states, generates executable charging actions, observes the resulting battery response, and continuously adjusts subsequent decisions. We construct a large-scale simulation corpus with PyBaMM across five battery chemistries and 11 parameter sets and convert the resulting state-action-response trajectories into structured process data for agent training. Our two-stage strategy combines supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). SFT establishes battery-control semantics and feasible decision priors from simulation-derived demonstrations. GRPO aligns the agent with a multi-objective control objective covering energy throughput, degradation mitigation, and operational constraints. Closed-loop simulations show that, under matched energy-throughput requirements, the agent achieves longer cycling life than conventional charging baselines and optimization-based strategies across heterogeneous battery configurations. This study provides a simulation-grounded proof of concept for using LLM agents as sequential decision makers in battery management and takes a step toward language-model-enabled control of physical energy systems.",
      "published": "2026-11-13T13:11:04Z",
      "abstract_url": "http://arxiv.org/abs/2611.18140v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18140v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams",
      "authors": [
        "Issa Sugiura",
        "Taro Makino",
        "Takashi Ishida"
      ],
      "abstract": "Large language models (LLMs) are increasingly applied to finance, creating a need for benchmarks that evaluate their capabilities across financial systems with different regulations and practices. However, existing financial benchmarks provide limited coverage of country-specific financial systems, and they are difficult to keep up to date as released questions leak into training data. We introduce JFinBench, a Japanese financial benchmark comprising 321 examples from the 2026 editions of six regularly administered certification exams covering accounting, tax, financial planning, and banking. To construct it, we develop a human-in-the-loop agentic pipeline in which a coding agent writes exam-specific conversion code and a review page, while a human verifies every example and feeds corrections back to the agent. This pipeline adapts to heterogeneous exam formats and reduces the effort of incorporating new exam editions, enabling periodic refreshes that mitigate data contamination. Evaluating 12 recent LLMs, we find that Claude Fable 5 and GPT-6 Astra exceed 93% overall, while even the best open-weight model, Kimi K3.1 Pro, trails Claude Fable 5 by 12 percentage points. When models can additionally use a terminal and web access, they retrieve the knowledge they lack and every model improves. The gap narrows but about 4 points remain, and the ranking of models stays largely the same, suggesting that knowledge held in the model itself remains valuable even with tools. We release the dataset, the evaluation code, and the construction pipeline.",
      "published": "2026-11-13T12:59:12Z",
      "abstract_url": "http://arxiv.org/abs/2611.18130v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18130v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Do Machines Learn Deviance From Their Peers? A Randomized Test of Differential Association With AI Agents",
      "authors": [
        "Murat Ozer",
        "Arif Akgul",
        "Amitabh Chakravorty"
      ],
      "abstract": "Differential association theory and its extension, social learning theory, hold that rule-breaking, and the definitions that make it acceptable, are learned through interaction with others. Because people choose their friends, this claim has been hard to test. We addressed this selection problem with AI agents, which learn only through communication and can be assigned to peers at random. In a preregistered experiment, 420 teams of three agents (1,260 agents; seven AI models) worked on coding tasks that could not be solved honestly. A scripted teammate said that it had cheated (deviant peer), that it had reported the problem honestly (honest peer), or said nothing (silent peer). Half of the teams were told that an auditor read their chat. Compared with a silent peer, a deviant peer increased cheating (66.2% vs. 62.9%; OR = 1.96) and an honest peer reduced it (55.7%; OR = 0.35). Among agents that cheated, a deviant peer increased claims that cheating was normal (13.7% vs. 3.4%), often as the team's norm. Monitoring did not suppress prohibited discussion; it doubled it. Peers mattered only for models that had not yet decided. AI agents offer a new way to test criminological learning theories without selection effects.",
      "published": "2026-11-13T12:26:49Z",
      "abstract_url": "http://arxiv.org/abs/2611.18098v1",
      "pdf_url": "https://arxiv.org/pdf/2611.18098v1",
      "categories": [
        "cs.AI"
      ]
    }
  ]
};
