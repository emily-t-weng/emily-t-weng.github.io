const PAPERS_DATA = {
  "last_updated": "2026-02-23 02:51:29 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Unifying approach to uniform expressivity of graph neural networks",
      "authors": [
        "Huan Luo",
        "Jonni Virtema"
      ],
      "abstract": "The expressive power of Graph Neural Networks (GNNs) is often analysed via correspondence to the Weisfeiler-Leman (WL) algorithm and fragments of first-order logic. Standard GNNs are limited to performing aggregation over immediate neighbourhoods or over global read-outs. To increase their expressivity, recent attempts have been made to incorporate substructural information (e.g. cycle counts and subgraph properties). In this paper, we formalize this architectural trend by introducing Template GNNs (T-GNNs), a generalized framework where node features are updated by aggregating over valid template embeddings from a specified set of graph templates. We propose a corresponding logic, Graded template modal logic (GML(T)), and generalized notions of template-based bisimulation and WL algorithm. We establish an equivalence between the expressive power of T-GNNs and GML(T), and provide a unifying approach for analysing GNN expressivity: we show how standard AC-GNNs and its recent variants can be interpreted as instantiations of T-GNNs.",
      "published": "2026-02-20T18:18:48Z",
      "abstract_url": "http://arxiv.org/abs/2602.18409v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18409v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.LO"
      ]
    },
    {
      "title": "Leakage and Second-Order Dynamics Improve Hippocampal RNN Replay",
      "authors": [
        "Josue Casco-Rodriguez",
        "Nanda H. Krishna",
        "Richard G. Baraniuk"
      ],
      "abstract": "Biological neural networks (like the hippocampus) can internally generate \"replay\" resembling stimulus-driven activity. Recent computational models of replay use noisy recurrent neural networks (RNNs) trained to path-integrate. Replay in these networks has been described as Langevin sampling, but new modifiers of noisy RNN replay have surpassed this description. We re-examine noisy RNN replay as sampling to understand or improve it in three ways: (1) Under simple assumptions, we prove that the gradients replay activity should follow are time-varying and difficult to estimate, but readily motivate the use of hidden state leakage in RNNs for replay. (2) We confirm that hidden state adaptation (negative feedback) encourages exploration in replay, but show that it incurs non-Markov sampling that also slows replay. (3) We propose the first model of temporally compressed replay in noisy path-integrating RNNs through hidden state momentum, connect it to underdamped Langevin sampling, and show that, together with adaptation, it counters slowness while maintaining exploration. We verify our findings via path-integration of 2D triangular and T-maze paths and of high-dimensional paths of synthetic rat place cell activity.",
      "published": "2026-02-20T18:07:09Z",
      "abstract_url": "http://arxiv.org/abs/2602.18401v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18401v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.NC",
        "stat.ML"
      ]
    },
    {
      "title": "Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and Steering-Gain Control with PPO",
      "authors": [
        "Mohamed Elgouhary",
        "Amr S. El-Wakeel"
      ],
      "abstract": "Pure Pursuit (PP) is widely used in autonomous racing for real-time path tracking due to its efficiency and geometric clarity, yet performance is highly sensitive to how key parameters-lookahead distance and steering gain-are chosen. Standard velocity-based schedules adjust these only approximately and often fail to transfer across tracks and speed profiles. We propose a reinforcement-learning (RL) approach that jointly chooses the lookahead Ld and a steering gain g online using Proximal Policy Optimization (PPO). The policy observes compact state features (speed and curvature taps) and outputs (Ld, g) at each control step. Trained in F1TENTH Gym and deployed in a ROS 2 stack, the policy drives PP directly (with light smoothing) and requires no per-map retuning. Across simulation and real-car tests, the proposed RL-PP controller that jointly selects (Ld, g) consistently outperforms fixed-lookahead PP, velocity-scheduled adaptive PP, and an RL lookahead-only variant, and it also exceeds a kinematic MPC raceline tracker under our evaluated settings in lap time, path-tracking accuracy, and steering smoothness, demonstrating that policy-guided parameter tuning can reliably improve classical geometry-based control.",
      "published": "2026-02-20T17:48:21Z",
      "abstract_url": "http://arxiv.org/abs/2602.18386v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18386v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "FedZMG: Efficient Client-Side Optimization in Federated Learning",
      "authors": [
        "Fotios Zantalis",
        "Evangelos Zervas",
        "Grigorios Koulouras"
      ],
      "abstract": "Federated Learning (FL) enables distributed model training on edge devices while preserving data privacy. However, clients tend to have non-Independent and Identically Distributed (non-IID) data, which often leads to client-drift, and therefore diminishing convergence speed and model performance. While adaptive optimizers have been proposed to mitigate these effects, they frequently introduce computational complexity or communication overhead unsuitable for resource-constrained IoT environments. This paper introduces Federated Zero Mean Gradients (FedZMG), a novel, parameter-free, client-side optimization algorithm designed to tackle client-drift by structurally regularizing the optimization space. Advancing the idea of Gradient Centralization, FedZMG projects local gradients onto a zero-mean hyperplane, effectively neutralizing the \"intensity\" or \"bias\" shifts inherent in heterogeneous data distributions without requiring additional communication or hyperparameter tuning. A theoretical analysis is provided, proving that FedZMG reduces the effective gradient variance and guarantees tighter convergence bounds compared to standard FedAvg. Extensive empirical evaluations on EMNIST, CIFAR100, and Shakespeare datasets demonstrate that FedZMG achieves better convergence speed and final validation accuracy compared to the baseline FedAvg and the adaptive optimizer FedAdam, particularly in highly non-IID settings.",
      "published": "2026-02-20T17:45:28Z",
      "abstract_url": "http://arxiv.org/abs/2602.18384v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18384v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "\"How Do I ...?\": Procedural Questions Predominate Student-LLM Chatbot Conversations",
      "authors": [
        "Alexandra Neagu",
        "Marcus Messer",
        "Peter Johnson",
        "Rhodri Nelson"
      ],
      "abstract": "Providing scaffolding through educational chatbots built on Large Language Models (LLM) has potential risks and benefits that remain an open area of research. When students navigate impasses, they ask for help by formulating impasse-driven questions. Within interactions with LLM chatbots, such questions shape the user prompts and drive the pedagogical effectiveness of the chatbot's response. This paper focuses on such student questions from two datasets of distinct learning contexts: formative self-study, and summative assessed coursework. We analysed 6,113 messages from both learning contexts, using 11 different LLMs and three human raters to classify student questions using four existing schemas. On the feasibility of using LLMs as raters, results showed moderate-to-good inter-rater reliability, with higher consistency than human raters. The data showed that 'procedural' questions predominated in both learning contexts, but more so when students prepare for summative assessment. These results provide a basis on which to use LLMs for classification of student questions. However, we identify clear limitations in both the ability to classify with schemas and the value of doing so: schemas are limited and thus struggle to accommodate the semantic richness of composite prompts, offering only partial understanding the wider risks and benefits of chatbot integration. In the future, we recommend an analysis approach that captures the nuanced, multi-turn nature of conversation, for example, by applying methods from conversation analysis in discursive psychology.",
      "published": "2026-02-20T17:27:41Z",
      "abstract_url": "http://arxiv.org/abs/2602.18372v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18372v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "Vichara: Appellate Judgment Prediction and Explanation for the Indian Judicial System",
      "authors": [
        "Pavithra PM Nair",
        "Preethu Rose Anish"
      ],
      "abstract": "In jurisdictions like India, where courts face an extensive backlog of cases, artificial intelligence offers transformative potential for legal judgment prediction. A critical subset of this backlog comprises appellate cases, which are formal decisions issued by higher courts reviewing the rulings of lower courts. To this end, we present Vichara, a novel framework tailored to the Indian judicial system that predicts and explains appellate judgments. Vichara processes English-language appellate case proceeding documents and decomposes them into decision points. Decision points are discrete legal determinations that encapsulate the legal issue, deciding authority, outcome, reasoning, and temporal context. The structured representation isolates the core determinations and their context, enabling accurate predictions and interpretable explanations. Vichara's explanations follow a structured format inspired by the IRAC (Issue-Rule-Application-Conclusion) framework and adapted for Indian legal reasoning. This enhances interpretability, allowing legal professionals to assess the soundness of predictions efficiently. We evaluate Vichara on two datasets, PredEx and the expert-annotated subset of the Indian Legal Documents Corpus (ILDC_expert), using four large language models: GPT-4o mini, Llama-3.1-8B, Mistral-7B, and Qwen2.5-7B. Vichara surpasses existing judgment prediction benchmarks on both datasets, with GPT-4o mini achieving the highest performance (F1: 81.5 on PredEx, 80.3 on ILDC_expert), followed by Llama-3.1-8B. Human evaluation of the generated explanations across Clarity, Linking, and Usefulness metrics highlights GPT-4o mini's superior interpretability.",
      "published": "2026-02-20T16:57:44Z",
      "abstract_url": "http://arxiv.org/abs/2602.18346v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18346v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Robo-Saber: Generating and Simulating Virtual Reality Players",
      "authors": [
        "Nam Hee Kim",
        "Jingjing May Liu",
        "Jaakko Lehtinen",
        "Perttu Hämäläinen",
        "James F. O'Brien",
        "Xue Bin Peng"
      ],
      "abstract": "We present the first motion generation system for playtesting virtual reality (VR) games. Our player model generates VR headset and handheld controller movements from in-game object arrangements, guided by style exemplars and aligned to maximize simulated gameplay score. We train on the large BOXRR-23 dataset and apply our framework on the popular VR game Beat Saber. The resulting model Robo-Saber produces skilled gameplay and captures diverse player behaviors, mirroring the skill levels and movement patterns specified by input style exemplars. Robo-Saber demonstrates promise in synthesizing rich gameplay data for predictive applications and enabling a physics-based whole-body VR playtesting agent.",
      "published": "2026-02-20T16:19:19Z",
      "abstract_url": "http://arxiv.org/abs/2602.18319v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18319v1",
      "categories": [
        "cs.GR",
        "cs.AI",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "JPmHC Dynamical Isometry via Orthogonal Hyper-Connections",
      "authors": [
        "Biswa Sengupta",
        "Jinhua Wang",
        "Leo Brunswic"
      ],
      "abstract": "Recent advances in deep learning, exemplified by Hyper-Connections (HC), have expanded the residual connection paradigm by introducing wider residual streams and diverse connectivity patterns. While these innovations yield significant performance gains, they compromise the identity mapping property of residual connections, leading to training instability, limited scalability, and increased memory overhead. To address these challenges, we propose JPmHC (Jacobian-spectrum Preserving manifold-constrained Hyper-Connections), a framework that replaces identity skips with a trainable linear mixer acting on n parallel streams while explicitly controlling gradient conditioning. By constraining the mixer M on operator-norm-bounded manifolds (e.g., bistochastic, Stiefel, Grassmann), JPmHC prevents gradient pathologies and enhances stability. JPmHC introduces three key contributions: (i) a free-probability analysis that predicts Jacobian spectra for structured skips, providing actionable design rules for mixer selection; (ii) memory-efficient implicit differentiation for fixed-point projections, reducing activation memory and synchronization overhead; and (iii) a Stiefel-constrained mixer via Cayley transforms, ensuring orthogonality without post-hoc normalization. Empirical evaluations on ARC-AGI demonstrate that JPmHC achieves faster convergence, higher accuracy, and lower computational cost compared to bistochastic baselines. As a flexible and scalable extension of HC, JPmHC advances spectrum-aware, stable, and efficient deep learning, offering insights into topological architecture design and foundational model evolution.",
      "published": "2026-02-20T16:06:01Z",
      "abstract_url": "http://arxiv.org/abs/2602.18308v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18308v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Analyzing and Improving Chain-of-Thought Monitorability Through Information Theory",
      "authors": [
        "Usman Anwar",
        "Tim Bakker",
        "Dana Kianfar",
        "Cristina Pinneri",
        "Christos Louizos"
      ],
      "abstract": "Chain-of-thought (CoT) monitors are LLM-based systems that analyze reasoning traces to detect when outputs may exhibit attributes of interest, such as test-hacking behavior during code generation. In this paper, we use information-theoretic analysis to show that non-zero mutual information between CoT and output is a necessary but not sufficient condition for CoT monitorability. We identify two sources of approximation error that may undermine the performance of CoT monitors in practice: information gap, which measures the extent to which the monitor can extract the information available in CoT, and elicitation error, which measures the extent to which the monitor approximates the optimal monitoring function. We further demonstrate that CoT monitorability can be systematically improved through targeted training objectives. To this end, we propose two complementary approaches: (a) an oracle-based method that directly rewards the monitored model for producing CoTs that maximize monitor accuracy, and (b) a more practical, label-free approach that maximizes conditional mutual information between outputs and CoTs. Across multiple different environments, we show both methods significantly improve monitor accuracy while preventing CoT degeneration even when training against a monitor, thereby mitigating reward hacking when the task reward is imperfectly specified.",
      "published": "2026-02-20T15:50:30Z",
      "abstract_url": "http://arxiv.org/abs/2602.18297v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18297v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.IT"
      ]
    },
    {
      "title": "Decoding as Optimisation on the Probability Simplex: From Top-K to Top-P (Nucleus) to Best-of-K Samplers",
      "authors": [
        "Xiaotong Ji",
        "Rasul Tutunov",
        "Matthieu Zimmer",
        "Haitham Bou-Ammar"
      ],
      "abstract": "Decoding sits between a language model and everything we do with it, yet it is still treated as a heuristic knob-tuning exercise. We argue decoding should be understood as a principled optimisation layer: at each token, we solve a regularised problem over the probability simplex that trades off model score against structural preferences and constraints. This single template recovers greedy decoding, Softmax sampling, Top-K, Top-P, and Sparsemax-style sparsity as special cases, and explains their common structure through optimality conditions. More importantly, the framework makes it easy to invent new decoders without folklore. We demonstrate this by designing Best-of-K (BoK), a KL-anchored coverage objective aimed at multi-sample pipelines (self-consistency, reranking, verifier selection). BoK targets the probability of covering good alternatives within a fixed K-sample budget and improves empirical performance. We show that such samples can improve accuracy by, for example, +18.6% for Qwen2.5-Math-7B on MATH500 at high sampling temperatures.",
      "published": "2026-02-20T15:38:16Z",
      "abstract_url": "http://arxiv.org/abs/2602.18292v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18292v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PRISM: Parallel Reward Integration with Symmetry for MORL",
      "authors": [
        "Finn van der Knaap",
        "Kejiang Qian",
        "Zheng Xu",
        "Fengxiang He"
      ],
      "abstract": "This work studies heterogeneous Multi-Objective Reinforcement Learning (MORL), where objectives can differ sharply in temporal frequency. Such heterogeneity allows dense objectives to dominate learning, while sparse long-horizon rewards receive weak credit assignment, leading to poor sample efficiency. We propose a Parallel Reward Integration with Symmetry (PRISM) algorithm that enforces reflectional symmetry as an inductive bias in aligning reward channels. PRISM introduces ReSymNet, a theory-motivated model that reconciles temporal-frequency mismatches across objectives, using residual blocks to learn a scaled opportunity value that accelerates exploration while preserving the optimal policy. We also propose SymReg, a reflectional equivariance regulariser that enforces agent mirroring and constrains policy search to a reflection-equivariant subspace. This restriction provably reduces hypothesis complexity and improves generalisation. Across MuJoCo benchmarks, PRISM consistently outperforms both a sparse-reward baseline and an oracle trained with full dense rewards, improving Pareto coverage and distributional balance: it achieves hypervolume gains exceeding 100\\% over the baseline and up to 32\\% over the oracle. The code is at \\href{https://github.com/EVIEHub/PRISM}{https://github.com/EVIEHub/PRISM}.",
      "published": "2026-02-20T15:02:42Z",
      "abstract_url": "http://arxiv.org/abs/2602.18277v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18277v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Simplifying Outcomes of Language Model Component Analyses with ELIA",
      "authors": [
        "Aaron Louis Eidt",
        "Nils Feldhus"
      ],
      "abstract": "While mechanistic interpretability has developed powerful tools to analyze the internal workings of Large Language Models (LLMs), their complexity has created an accessibility gap, limiting their use to specialists. We address this challenge by designing, building, and evaluating ELIA (Explainable Language Interpretability Analysis), an interactive web application that simplifies the outcomes of various language model component analyses for a broader audience. The system integrates three key techniques -- Attribution Analysis, Function Vector Analysis, and Circuit Tracing -- and introduces a novel methodology: using a vision-language model to automatically generate natural language explanations (NLEs) for the complex visualizations produced by these methods. The effectiveness of this approach was empirically validated through a mixed-methods user study, which revealed a clear preference for interactive, explorable interfaces over simpler, static visualizations. A key finding was that the AI-powered explanations helped bridge the knowledge gap for non-experts; a statistical analysis showed no significant correlation between a user's prior LLM experience and their comprehension scores, suggesting that the system reduced barriers to comprehension across experience levels. We conclude that an AI system can indeed simplify complex model analyses, but its true power is unlocked when paired with thoughtful, user-centered design that prioritizes interactivity, specificity, and narrative guidance.",
      "published": "2026-02-20T14:45:27Z",
      "abstract_url": "http://arxiv.org/abs/2602.18262v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18262v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Thinking by Subtraction: Confidence-Driven Contrastive Decoding for LLM Reasoning",
      "authors": [
        "Lexiang Tang",
        "Weihao Gao",
        "Bingchen Zhao",
        "Lu Ma",
        "Qiao jin",
        "Bang Yang",
        "Yuexian Zou"
      ],
      "abstract": "Recent work on test-time scaling for large language model (LLM) reasoning typically assumes that allocating more inference-time computation uniformly improves correctness. However, prior studies show that reasoning uncertainty is highly localized: a small subset of low-confidence tokens disproportionately contributes to reasoning errors and unnecessary output expansion. Motivated by this observation, we propose Thinking by Subtraction, a confidence-driven contrastive decoding approach that improves reasoning reliability through targeted token-level intervention. Our method, Confidence-Driven Contrastive Decoding, detects low-confidence tokens during decoding and intervenes selectively at these positions. It constructs a contrastive reference by replacing high-confidence tokens with minimal placeholders, and refines predictions by subtracting this reference distribution at low-confidence locations. Experiments show that CCD significantly improves accuracy across mathematical reasoning benchmarks while substantially reducing output length, with minimal KV-cache overhead. As a training-free method, CCD enhances reasoning reliability through targeted low-confidence intervention without computational redundancy. Our code will be made available at: https://github.com/bolo-web/CCD.",
      "published": "2026-02-20T14:13:22Z",
      "abstract_url": "http://arxiv.org/abs/2602.18232v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18232v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "[Re] Benchmarking LLM Capabilities in Negotiation through Scoreable Games",
      "authors": [
        "Jorge Carrasco Pollo",
        "Ioannis Kapetangeorgis",
        "Joshua Rosenthal",
        "John Hua Yao"
      ],
      "abstract": "Large Language Models (LLMs) demonstrate significant potential in multi-agent negotiation tasks, yet evaluation in this domain remains challenging due to a lack of robust and generalizable benchmarks. Abdelnabi et al. (2024) introduce a negotiation benchmark based on Scoreable Games, with the aim of developing a highly complex and realistic evaluation framework for LLMs. Our work investigates the reproducibility of claims in their benchmark, and provides a deeper understanding of its usability and generalizability. We replicate the original experiments on additional models, and introduce additional metrics to verify negotiation quality and evenness of evaluation. Our findings reveal that while the benchmark is indeed complex, model comparison is ambiguous, raising questions about its objectivity. Furthermore, we identify limitations in the experimental setup, particularly in information leakage detection and thoroughness of the ablation study. By examining and analyzing the behavior of a wider range of models on an extended version of the benchmark, we reveal insights that provide additional context to potential users. Our results highlight the importance of context in model-comparative evaluations.",
      "published": "2026-02-20T14:11:31Z",
      "abstract_url": "http://arxiv.org/abs/2602.18230v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18230v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SOMtime the World Ain$'$t Fair: Violating Fairness Using Self-Organizing Maps",
      "authors": [
        "Joseph Bingham",
        "Netanel Arussy",
        "Dvir Aran"
      ],
      "abstract": "Unsupervised representations are widely assumed to be neutral with respect to sensitive attributes when those attributes are withheld from training. We show that this assumption is false. Using SOMtime, a topology-preserving representation method based on high-capacity Self-Organizing Maps, we demonstrate that sensitive attributes such as age and income emerge as dominant latent axes in purely unsupervised embeddings, even when explicitly excluded from the input. On two large-scale real-world datasets (the World Values Survey across five countries and the Census-Income dataset), SOMtime recovers monotonic orderings aligned with withheld sensitive attributes, achieving Spearman correlations of up to 0.85, whereas PCA and UMAP typically remain below 0.23 (with a single exception reaching 0.31), and against t-SNE and autoencoders which achieve at most 0.34. Furthermore, unsupervised segmentation of SOMtime embeddings produces demographically skewed clusters, demonstrating downstream fairness risks without any supervised task. These findings establish that \\textit{fairness through unawareness} fails at the representation level for ordinal sensitive attributes and that fairness auditing must extend to unsupervised components of machine learning pipelines. We have made the code available at~ https://github.com/JosephBingham/SOMtime",
      "published": "2026-02-20T13:25:28Z",
      "abstract_url": "http://arxiv.org/abs/2602.18201v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18201v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "LERD: Latent Event-Relational Dynamics for Neurodegenerative Classification",
      "authors": [
        "Hairong Chen",
        "Yicheng Feng",
        "Ziyu Jia",
        "Samir Bhatt",
        "Hengguan Huang"
      ],
      "abstract": "Alzheimer's disease (AD) alters brain electrophysiology and disrupts multichannel EEG dynamics, making accurate and clinically useful EEG-based diagnosis increasingly important for screening and disease monitoring. However, many existing approaches rely on black-box classifiers and do not explicitly model the underlying dynamics that generate observed signals. To address these limitations, we propose LERD, an end-to-end Bayesian electrophysiological neural dynamical system that infers latent neural events and their relational structure directly from multichannel EEG without event or interaction annotations. LERD combines a continuous-time event inference module with a stochastic event-generation process to capture flexible temporal patterns, while incorporating an electrophysiology-inspired dynamical prior to guide learning in a principled way. We further provide theoretical analysis that yields a tractable bound for training and stability guarantees for the inferred relational dynamics. Extensive experiments on synthetic benchmarks and two real-world AD EEG cohorts demonstrate that LERD consistently outperforms strong baselines and yields physiology-aligned latent summaries that help characterize group-level dynamical differences.",
      "published": "2026-02-20T13:03:40Z",
      "abstract_url": "http://arxiv.org/abs/2602.18195v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18195v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Capabilities Ain't All You Need: Measuring Propensities in AI",
      "authors": [
        "Daniel Romero-Alvarado",
        "Fernando Martínez-Plumed",
        "Lorenzo Pacchiardi",
        "Hugo Save",
        "Siddhesh Milind Pawar",
        "Behzad Mehrbakhsh",
        "Pablo Antonio Moreno Casares",
        "Ben Slater",
        "Paolo Bova",
        "Peter Romero",
        "Zachary R. Tyler",
        "Jonathan Prunty",
        "Luning Sun",
        "Jose Hernandez-Orallo"
      ],
      "abstract": "AI evaluation has primarily focused on measuring capabilities, with formal approaches inspired from Item Response Theory (IRT) being increasingly applied. Yet propensities - the tendencies of models to exhibit particular behaviours - play a central role in determining both performance and safety outcomes. However, traditional IRT describes a model's success on a task as a monotonic function of model capabilities and task demands, an approach unsuited to propensities, where both excess and deficiency can be problematic. Here, we introduce the first formal framework for measuring AI propensities by using a bilogistic formulation for model success, which attributes high success probability when the model's propensity is within an \"ideal band\". Further, we estimate the limits of the ideal band using LLMs equipped with newly developed task-agnostic rubrics. Applying our framework to six families of LLM models whose propensities are incited in either direction, we find that we can measure how much the propensity is shifted and what effect this has on the tasks. Critically, propensities estimated using one benchmark successfully predict behaviour on held-out tasks. Moreover, we obtain stronger predictive power when combining propensities and capabilities than either separately. More broadly, our framework showcases how rigorous propensity measurements can be conducted and how it yields gains over solely using capability evaluations to predict AI behaviour.",
      "published": "2026-02-20T12:40:18Z",
      "abstract_url": "http://arxiv.org/abs/2602.18182v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18182v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Click it or Leave it: Detecting and Spoiling Clickbait with Informativeness Measures and Large Language Models",
      "authors": [
        "Wojciech Michaluk",
        "Tymoteusz Urban",
        "Mateusz Kubita",
        "Soveatin Kuntur",
        "Anna Wroblewska"
      ],
      "abstract": "Clickbait headlines degrade the quality of online information and undermine user trust. We present a hybrid approach to clickbait detection that combines transformer-based text embeddings with linguistically motivated informativeness features. Using natural language processing techniques, we evaluate classical vectorizers, word embedding baselines, and large language model embeddings paired with tree-based classifiers. Our best-performing model, XGBoost over embeddings augmented with 15 explicit features, achieves an F1-score of 91\\%, outperforming TF-IDF, Word2Vec, GloVe, LLM prompt based classification, and feature-only baselines. The proposed feature set enhances interpretability by highlighting salient linguistic cues such as second-person pronouns, superlatives, numerals, and attention-oriented punctuation, enabling transparent and well-calibrated clickbait predictions. We release code and trained models to support reproducible research.",
      "published": "2026-02-20T12:16:08Z",
      "abstract_url": "http://arxiv.org/abs/2602.18171v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18171v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FENCE: A Financial and Multimodal Jailbreak Detection Dataset",
      "authors": [
        "Mirae Kim",
        "Seonghun Jeong",
        "Youngjun Kwak"
      ],
      "abstract": "Jailbreaking poses a significant risk to the deployment of Large Language Models (LLMs) and Vision Language Models (VLMs). VLMs are particularly vulnerable because they process both text and images, creating broader attack surfaces. However, available resources for jailbreak detection are scarce, particularly in finance. To address this gap, we present FENCE, a bilingual (Korean-English) multimodal dataset for training and evaluating jailbreak detectors in financial applications. FENCE emphasizes domain realism through finance-relevant queries paired with image-grounded threats. Experiments with commercial and open-source VLMs reveal consistent vulnerabilities, with GPT-4o showing measurable attack success rates and open-source models displaying greater exposure. A baseline detector trained on FENCE achieves 99 percent in-distribution accuracy and maintains strong performance on external benchmarks, underscoring the dataset's robustness for training reliable detection models. FENCE provides a focused resource for advancing multimodal jailbreak detection in finance and for supporting safer, more reliable AI systems in sensitive domains. Warning: This paper includes example data that may be offensive.",
      "published": "2026-02-20T11:40:41Z",
      "abstract_url": "http://arxiv.org/abs/2602.18154v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18154v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.DB"
      ]
    },
    {
      "title": "Agentic Adversarial QA for Improving Domain-Specific LLMs",
      "authors": [
        "Vincent Grari",
        "Ciprian Tomoiaga",
        "Sylvain Lamprier",
        "Tatsunori Hashimoto",
        "Marcin Detyniecki"
      ],
      "abstract": "Large Language Models (LLMs), despite extensive pretraining on broad internet corpora, often struggle to adapt effectively to specialized domains. There is growing interest in fine-tuning these models for such domains; however, progress is constrained by the scarcity and limited coverage of high-quality, task-relevant data. To address this, synthetic data generation methods such as paraphrasing or knowledge extraction are commonly applied. Although these approaches excel at factual recall and conceptual knowledge, they suffer from two critical shortcomings: (i) they provide minimal support for interpretive reasoning capabilities in these specialized domains, and (ii) they often produce synthetic corpora that are excessively large and redundant, resulting in poor sample efficiency. To overcome these gaps, we propose an adversarial question-generation framework that produces a compact set of semantically challenging questions. These questions are constructed by comparing the outputs of the model to be adapted and a robust expert model grounded in reference documents, using an iterative, feedback-driven process designed to reveal and address comprehension gaps. Evaluation on specialized subsets of the LegalBench corpus demonstrates that our method achieves greater accuracy with substantially fewer synthetic samples.",
      "published": "2026-02-20T10:53:09Z",
      "abstract_url": "http://arxiv.org/abs/2602.18137v1",
      "pdf_url": "https://arxiv.org/pdf/2602.18137v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
