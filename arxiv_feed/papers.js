const PAPERS_DATA = {
  "last_updated": "2026-02-18 02:50:49 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching",
      "authors": [
        "Zhen Wu",
        "Xiaoyu Huang",
        "Lujie Yang",
        "Yuanhang Zhang",
        "Koushil Sreenath",
        "Xi Chen",
        "Pieter Abbeel",
        "Rocky Duan",
        "Angjoo Kanazawa",
        "Carmelo Sferrazza",
        "Guanya Shi",
        "C. Karen Liu"
      ],
      "abstract": "While recent advances in humanoid locomotion have achieved stable walking on varied terrains, capturing the agility and adaptivity of highly dynamic human motions remains an open challenge. In particular, agile parkour in complex environments demands not only low-level robustness, but also human-like motion expressiveness, long-horizon skill composition, and perception-driven decision-making. In this paper, we present Perceptive Humanoid Parkour (PHP), a modular framework that enables humanoid robots to autonomously perform long-horizon, vision-based parkour across challenging obstacle courses. Our approach first leverages motion matching, formulated as nearest-neighbor search in a feature space, to compose retargeted atomic human skills into long-horizon kinematic trajectories. This framework enables the flexible composition and smooth transition of complex skill chains while preserving the elegance and fluidity of dynamic human motions. Next, we train motion-tracking reinforcement learning (RL) expert policies for these composed motions, and distill them into a single depth-based, multi-skill student policy, using a combination of DAgger and RL. Crucially, the combination of perception and skill composition enables autonomous, context-aware decision-making: using only onboard depth sensing and a discrete 2D velocity command, the robot selects and executes whether to step over, climb onto, vault or roll off obstacles of varying geometries and heights. We validate our framework with extensive real-world experiments on a Unitree G1 humanoid robot, demonstrating highly dynamic parkour skills such as climbing tall obstacles up to 1.25m (96% robot height), as well as long-horizon multi-obstacle traversal with closed-loop adaptation to real-time obstacle perturbations.",
      "published": "2026-02-17T18:59:11Z",
      "abstract_url": "http://arxiv.org/abs/2602.15827v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15827v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing",
      "authors": [
        "Zarif Ikram",
        "Arad Firouzkouhi",
        "Stephen Tu",
        "Mahdi Soltanolkotabi",
        "Paria Rashidinejad"
      ],
      "abstract": "A central challenge in large language model (LLM) editing is capability preservation: methods that successfully change targeted behavior can quietly game the editing proxy and corrupt general capabilities, producing degenerate behaviors reminiscent of proxy/reward hacking. We present CrispEdit, a scalable and principled second-order editing algorithm that treats capability preservation as an explicit constraint, unifying and generalizing several existing editing approaches. CrispEdit formulates editing as constrained optimization and enforces the constraint by projecting edit updates onto the low-curvature subspace of the capability-loss landscape. At the crux of CrispEdit is expressing capability constraint via Bregman divergence, whose quadratic form yields the Gauss-Newton Hessian exactly and even when the base model is not trained to convergence. We make this second-order procedure efficient at the LLM scale using Kronecker-factored approximate curvature (K-FAC) and a novel matrix-free projector that exploits Kronecker structure to avoid constructing massive projection matrices. Across standard model-editing benchmarks, CrispEdit achieves high edit success while keeping capability degradation below 1% on average across datasets, significantly improving over prior editors.",
      "published": "2026-02-17T18:58:04Z",
      "abstract_url": "http://arxiv.org/abs/2602.15823v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15823v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Decision Quality Evaluation Framework at Pinterest",
      "authors": [
        "Yuqi Tian",
        "Robert Paine",
        "Attila Dobi",
        "Kevin O'Sullivan",
        "Aravindh Manickavasagam",
        "Faisal Farooq"
      ],
      "abstract": "Online platforms require robust systems to enforce content safety policies at scale. A critical component of these systems is the ability to evaluate the quality of moderation decisions made by both human agents and Large Language Models (LLMs). However, this evaluation is challenging due to the inherent trade-offs between cost, scale, and trustworthiness, along with the complexity of evolving policies. To address this, we present a comprehensive Decision Quality Evaluation Framework developed and deployed at Pinterest. The framework is centered on a high-trust Golden Set (GDS) curated by subject matter experts (SMEs), which serves as a ground truth benchmark. We introduce an automated intelligent sampling pipeline that uses propensity scores to efficiently expand dataset coverage. We demonstrate the framework's practical application in several key areas: benchmarking the cost-performance trade-offs of various LLM agents, establishing a rigorous methodology for data-driven prompt optimization, managing complex policy evolution, and ensuring the integrity of policy content prevalence metrics via continuous validation. The framework enables a shift from subjective assessments to a data-driven and quantitative practice for managing content safety systems.",
      "published": "2026-02-17T18:45:55Z",
      "abstract_url": "http://arxiv.org/abs/2602.15809v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15809v1",
      "categories": [
        "stat.AP",
        "cs.AI"
      ]
    },
    {
      "title": "The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety",
      "authors": [
        "Max Springer",
        "Chung Peng Lee",
        "Blossom Metevier",
        "Jane Castleman",
        "Bohdan Turbal",
        "Hayoung Jung",
        "Zeyu Shen",
        "Aleksandra Korolova"
      ],
      "abstract": "Fine-tuning aligned language models on benign tasks unpredictably degrades safety guardrails, even when training data contains no harmful content and developers have no adversarial intent. We show that the prevailing explanation, that fine-tuning updates should be orthogonal to safety-critical directions in high-dimensional parameter space, offers false reassurance: we show this orthogonality is structurally unstable and collapses under the dynamics of gradient descent. We then resolve this through a novel geometric analysis, proving that alignment concentrates in low-dimensional subspaces with sharp curvature, creating a brittle structure that first-order methods cannot detect or defend. While initial fine-tuning updates may indeed avoid these subspaces, the curvature of the fine-tuning loss generates second-order acceleration that systematically steers trajectories into alignment-sensitive regions. We formalize this mechanism through the Alignment Instability Condition, three geometric properties that, when jointly satisfied, lead to safety degradation. Our main result establishes a quartic scaling law: alignment loss grows with the fourth power of training time, governed by the sharpness of alignment geometry and the strength of curvature coupling between the fine-tuning task and safety-critical parameters. These results expose a structural blind spot in the current safety paradigm. The dominant approaches to safe fine-tuning address only the initial snapshot of a fundamentally dynamic problem. Alignment fragility is not a bug to be patched; it is an intrinsic geometric property of gradient descent on curved manifolds. Our results motivate the development of curvature-aware methods, and we hope will further enable a shift in alignment safety analysis from reactive red-teaming to predictive diagnostics for open-weight model deployment.",
      "published": "2026-02-17T18:39:15Z",
      "abstract_url": "http://arxiv.org/abs/2602.15799v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15799v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Enhancing Building Semantics Preservation in AI Model Training with Large Language Model Encodings",
      "authors": [
        "Suhyung Jang",
        "Ghang Lee",
        "Jaekun Lee",
        "Hyunjun Lee"
      ],
      "abstract": "Accurate representation of building semantics, encompassing both generic object types and specific subtypes, is essential for effective AI model training in the architecture, engineering, construction, and operation (AECO) industry. Conventional encoding methods (e.g., one-hot) often fail to convey the nuanced relationships among closely related subtypes, limiting AI's semantic comprehension. To address this limitation, this study proposes a novel training approach that employs large language model (LLM) embeddings (e.g., OpenAI GPT and Meta LLaMA) as encodings to preserve finer distinctions in building semantics. We evaluated the proposed method by training GraphSAGE models to classify 42 building object subtypes across five high-rise residential building information models (BIMs). Various embedding dimensions were tested, including original high-dimensional LLM embeddings (1,536, 3,072, or 4,096) and 1,024-dimensional compacted embeddings generated via the Matryoshka representation model. Experimental results demonstrated that LLM encodings outperformed the conventional one-hot baseline, with the llama-3 (compacted) embedding achieving a weighted average F1-score of 0.8766, compared to 0.8475 for one-hot encoding. The results underscore the promise of leveraging LLM-based encodings to enhance AI's ability to interpret complex, domain-specific building semantics. As the capabilities of LLMs and dimensionality reduction techniques continue to evolve, this approach holds considerable potential for broad application in semantic elaboration tasks throughout the AECO industry.",
      "published": "2026-02-17T18:26:36Z",
      "abstract_url": "http://arxiv.org/abs/2602.15791v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15791v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "This human study did not involve human subjects: Validating LLM simulations as behavioral evidence",
      "authors": [
        "Jessica Hullman",
        "David Broska",
        "Huaman Sun",
        "Aaron Shaw"
      ],
      "abstract": "A growing literature uses large language models (LLMs) as synthetic participants to generate cost-effective and nearly instantaneous responses in social science experiments. However, there is limited guidance on when such simulations support valid inference about human behavior. We contrast two strategies for obtaining valid estimates of causal effects and clarify the assumptions under which each is suitable for exploratory versus confirmatory research. Heuristic approaches seek to establish that simulated and observed human behavior are interchangeable through prompt engineering, model fine-tuning, and other repair strategies designed to reduce LLM-induced inaccuracies. While useful for many exploratory tasks, heuristic approaches lack the formal statistical guarantees typically required for confirmatory research. In contrast, statistical calibration combines auxiliary human data with statistical adjustments to account for discrepancies between observed and simulated responses. Under explicit assumptions, statistical calibration preserves validity and provides more precise estimates of causal effects at lower cost than experiments that rely solely on human participants. Yet the potential of both approaches depends on how well LLMs approximate the relevant populations. We consider what opportunities are overlooked when researchers focus myopically on substituting LLMs for human participants in a study.",
      "published": "2026-02-17T18:18:38Z",
      "abstract_url": "http://arxiv.org/abs/2602.15785v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15785v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ChartEditBench: Evaluating Grounded Multi-Turn Chart Editing in Multimodal Language Models",
      "authors": [
        "Manav Nitin Kapadnis",
        "Lawanya Baghel",
        "Atharva Naik",
        "Carolyn Rosé"
      ],
      "abstract": "While Multimodal Large Language Models (MLLMs) perform strongly on single-turn chart generation, their ability to support real-world exploratory data analysis remains underexplored. In practice, users iteratively refine visualizations through multi-turn interactions that require maintaining common ground, tracking prior edits, and adapting to evolving preferences. We introduce ChartEditBench, a benchmark for incremental, visually grounded chart editing via code, comprising 5,000 difficulty-controlled modification chains and a rigorously human-verified subset. Unlike prior one-shot benchmarks, ChartEditBench evaluates sustained, context-aware editing. We further propose a robust evaluation framework that mitigates limitations of LLM-as-a-Judge metrics by integrating execution-based fidelity checks, pixel-level visual similarity, and logical code verification. Experiments with state-of-the-art MLLMs reveal substantial degradation in multi-turn settings due to error accumulation and breakdowns in shared context, with strong performance on stylistic edits but frequent execution failures on data-centric transformations. ChartEditBench, establishes a challenging testbed for grounded, intent-aware multimodal programming.",
      "published": "2026-02-17T17:45:34Z",
      "abstract_url": "http://arxiv.org/abs/2602.15758v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15758v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "UrbanVerse: Learning Urban Region Representation Across Cities and Tasks",
      "authors": [
        "Fengze Sun",
        "Egemen Tanin",
        "Shanika Karunasekera",
        "Zuqing Li",
        "Flora D. Salim",
        "Jianzhong Qi"
      ],
      "abstract": "Recent advances in urban region representation learning have enabled a wide range of applications in urban analytics, yet existing methods remain limited in their capabilities to generalize across cities and analytic tasks. We aim to generalize urban representation learning beyond city- and task-specific settings, towards a foundation-style model for urban analytics. To this end, we propose UrbanVerse, a model for cross-city urban representation learning and cross-task urban analytics. For cross-city generalization, UrbanVerse focuses on features local to the target regions and structural features of the nearby regions rather than the entire city. We model regions as nodes on a graph, which enables a random walk-based procedure to form \"sequences of regions\" that reflect both local and neighborhood structural features for urban region representation learning. For cross-task generalization, we propose a cross-task learning module named HCondDiffCT. This module integrates region-conditioned prior knowledge and task-conditioned semantics into the diffusion process to jointly model multiple downstream urban prediction tasks. HCondDiffCT is generic. It can also be integrated with existing urban representation learning models to enhance their downstream task effectiveness. Experiments on real-world datasets show that UrbanVerse consistently outperforms state-of-the-art methods across six tasks under cross-city settings, achieving up to 35.89% improvements in prediction accuracy.",
      "published": "2026-02-17T17:28:48Z",
      "abstract_url": "http://arxiv.org/abs/2602.15750v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15750v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MRC-GAT: A Meta-Relational Copula-Based Graph Attention Network for Interpretable Multimodal Alzheimer's Disease Diagnosis",
      "authors": [
        "Fatemeh Khalvandi",
        "Saadat Izadi",
        "Abdolah Chalechale"
      ],
      "abstract": "Alzheimer's disease (AD) is a progressive neurodegenerative condition necessitating early and precise diagnosis to provide prompt clinical management. Given the paramount importance of early diagnosis, recent studies have increasingly focused on computer-aided diagnostic models to enhance precision and reliability. However, most graph-based approaches still rely on fixed structural designs, which restrict their flexibility and limit generalization across heterogeneous patient data. To overcome these limitations, the Meta-Relational Copula-Based Graph Attention Network (MRC-GAT) is proposed as an efficient multimodal model for AD classification tasks. The proposed architecture, copula-based similarity alignment, relational attention, and node fusion are integrated as the core components of episodic meta-learning, such that the multimodal features, including risk factors (RF), Cognitive test scores, and MRI attributes, are first aligned via a copula-based transformation in a common statistical space and then combined by a multi-relational attention mechanism. According to evaluations performed on the TADPOLE and NACC datasets, the MRC-GAT model achieved accuracies of 96.87% and 92.31%, respectively, demonstrating state-of-the-art performance compared to existing diagnostic models. Finally, the proposed model confirms the robustness and applicability of the proposed method by providing interpretability at various stages of disease diagnosis.",
      "published": "2026-02-17T17:15:32Z",
      "abstract_url": "http://arxiv.org/abs/2602.15740v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15740v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.QM"
      ]
    },
    {
      "title": "Spanning the Visual Analogy Space with a Weight Basis of LoRAs",
      "authors": [
        "Hila Manor",
        "Rinon Gal",
        "Haggai Maron",
        "Tomer Michaeli",
        "Gal Chechik"
      ],
      "abstract": "Visual analogy learning enables image manipulation through demonstration rather than textual description, allowing users to specify complex transformations difficult to articulate in words. Given a triplet $\\{\\mathbf{a}$, $\\mathbf{a}'$, $\\mathbf{b}\\}$, the goal is to generate $\\mathbf{b}'$ such that $\\mathbf{a} : \\mathbf{a}' :: \\mathbf{b} : \\mathbf{b}'$. Recent methods adapt text-to-image models to this task using a single Low-Rank Adaptation (LoRA) module, but they face a fundamental limitation: attempting to capture the diverse space of visual transformations within a fixed adaptation module constrains generalization capabilities. Inspired by recent work showing that LoRAs in constrained domains span meaningful, interpolatable semantic spaces, we propose LoRWeB, a novel approach that specializes the model for each analogy task at inference time through dynamic composition of learned transformation primitives, informally, choosing a point in a \"space of LoRAs\". We introduce two key components: (1) a learnable basis of LoRA modules, to span the space of different visual transformations, and (2) a lightweight encoder that dynamically selects and weighs these basis LoRAs based on the input analogy pair. Comprehensive evaluations demonstrate our approach achieves state-of-the-art performance and significantly improves generalization to unseen visual transformations. Our findings suggest that LoRA basis decompositions are a promising direction for flexible visual manipulation. Code and data are in https://research.nvidia.com/labs/par/lorweb",
      "published": "2026-02-17T17:02:38Z",
      "abstract_url": "http://arxiv.org/abs/2602.15727v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15727v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "eess.IV"
      ]
    },
    {
      "title": "Recursive Concept Evolution for Compositional Reasoning in Large Language Models",
      "authors": [
        "Sarim Chaudhry"
      ],
      "abstract": "Large language models achieve strong performance on many complex reasoning tasks, yet their accuracy degrades sharply on benchmarks that require compositional reasoning, including ARC-AGI-2, GPQA, MATH, BBH, and HLE. Existing methods improve reasoning by expanding token-level search through chain-of-thought prompting, self-consistency, or reinforcement learning, but they leave the model's latent representation space fixed. When the required abstraction is not already encoded in this space, performance collapses. We propose Recursive Concept Evolution (RCE), a framework that enables pretrained language models to modify their internal representation geometry during inference. RCE introduces dynamically generated low-rank concept subspaces that are spawned when representational inadequacy is detected, selected through a minimum description length criterion, merged when synergistic, and consolidated via constrained optimization to preserve stability. This process allows the model to construct new abstractions rather than recombining existing ones. We integrate RCE with Mistral-7B and evaluate it across compositional reasoning benchmarks. RCE yields 12-18 point gains on ARC-AGI-2, 8-14 point improvements on GPQA and BBH, and consistent reductions in depth-induced error on MATH and HLE.",
      "published": "2026-02-17T17:01:42Z",
      "abstract_url": "http://arxiv.org/abs/2602.15725v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15725v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation",
      "authors": [
        "Shutian Gu",
        "Chengkai Huang",
        "Ruoyu Wang",
        "Lina Yao"
      ],
      "abstract": "Vision-and-Language Navigation (VLN) requires an agent to follow natural-language instructions and navigate through previously unseen environments. Recent approaches increasingly employ large language models (LLMs) as high-level navigators due to their flexibility and reasoning capability. However, prompt-based LLM navigation often suffers from inefficient decision-making, as the model must repeatedly interpret instructions from scratch and reason over noisy and verbose navigable candidates at each step. In this paper, we propose a retrieval-augmented framework to improve the efficiency and stability of LLM-based VLN without modifying or fine-tuning the underlying language model. Our approach introduces retrieval at two complementary levels. At the episode level, an instruction-level embedding retriever selects semantically similar successful navigation trajectories as in-context exemplars, providing task-specific priors for instruction grounding. At the step level, an imitation-learned candidate retriever prunes irrelevant navigable directions before LLM inference, reducing action ambiguity and prompt complexity. Both retrieval modules are lightweight, modular, and trained independently of the LLM. We evaluate our method on the Room-to-Room (R2R) benchmark. Experimental results demonstrate consistent improvements in Success Rate, Oracle Success Rate, and SPL on both seen and unseen environments. Ablation studies further show that instruction-level exemplar retrieval and candidate pruning contribute complementary benefits to global guidance and step-wise decision efficiency. These results indicate that retrieval-augmented decision support is an effective and scalable strategy for enhancing LLM-based vision-and-language navigation.",
      "published": "2026-02-17T17:00:11Z",
      "abstract_url": "http://arxiv.org/abs/2602.15724v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15724v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Random Wavelet Features for Graph Kernel Machines",
      "authors": [
        "Valentin de Bassompierre",
        "Jean-Charles Delvenne",
        "Laurent Jacques"
      ],
      "abstract": "Node embeddings map graph vertices into low-dimensional Euclidean spaces while preserving structural information. They are central to tasks such as node classification, link prediction, and signal reconstruction. A key goal is to design node embeddings whose dot products capture meaningful notions of node similarity induced by the graph. Graph kernels offer a principled way to define such similarities, but their direct computation is often prohibitive for large networks. Inspired by random feature methods for kernel approximation in Euclidean spaces, we introduce randomized spectral node embeddings whose dot products estimate a low-rank approximation of any specific graph kernel. We provide theoretical and empirical results showing that our embeddings achieve more accurate kernel approximations than existing methods, particularly for spectrally localized kernels. These results demonstrate the effectiveness of randomized spectral constructions for scalable and principled graph representation learning.",
      "published": "2026-02-17T16:45:15Z",
      "abstract_url": "http://arxiv.org/abs/2602.15711v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15711v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SP"
      ]
    },
    {
      "title": "A Content-Based Framework for Cybersecurity Refusal Decisions in Large Language Models",
      "authors": [
        "Meirav Segal",
        "Noa Linder",
        "Omer Antverg",
        "Gil Gekker",
        "Tomer Fichman",
        "Omri Bodenheimer",
        "Edan Maor",
        "Omer Nevo"
      ],
      "abstract": "Large language models and LLM-based agents are increasingly used for cybersecurity tasks that are inherently dual-use. Existing approaches to refusal, spanning academic policy frameworks and commercially deployed systems, often rely on broad topic-based bans or offensive-focused taxonomies. As a result, they can yield inconsistent decisions, over-restrict legitimate defenders, and behave brittlely under obfuscation or request segmentation. We argue that effective refusal requires explicitly modeling the trade-off between offensive risk and defensive benefit, rather than relying solely on intent or offensive classification. In this paper, we introduce a content-based framework for designing and auditing cyber refusal policies that makes offense-defense tradeoffs explicit. The framework characterizes requests along five dimensions: Offensive Action Contribution, Offensive Risk, Technical Complexity, Defensive Benefit, and Expected Frequency for Legitimate Users, grounded in the technical substance of the request rather than stated intent. We demonstrate that this content-grounded approach resolves inconsistencies in current frontier model behavior and allows organizations to construct tunable, risk-aware refusal policies.",
      "published": "2026-02-17T16:12:21Z",
      "abstract_url": "http://arxiv.org/abs/2602.15689v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15689v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Estimating Human Muscular Fatigue in Dynamic Collaborative Robotic Tasks with Learning-Based Models",
      "authors": [
        "Feras Kiki",
        "Pouya P. Niaz",
        "Alireza Madani",
        "Cagatay Basdogan"
      ],
      "abstract": "Assessing human muscle fatigue is critical for optimizing performance and safety in physical human-robot interaction(pHRI). This work presents a data-driven framework to estimate fatigue in dynamic, cyclic pHRI using arm-mounted surface electromyography(sEMG). Subject-specific machine-learning regression models(Random Forest, XGBoost, and Linear Regression predict the fraction of cycles to fatigue(FCF) from three frequency-domain and one time-domain EMG features, and are benchmarked against a convolutional neural network(CNN) that ingests spectrograms of filtered EMG. Framing fatigue estimation as regression (rather than classification) captures continuous progression toward fatigue, supporting earlier detection, timely intervention, and adaptive robot control. In experiments with ten participants, a collaborative robot under admittance control guided repetitive lateral (left-right) end-effector motions until muscular fatigue. Average FCF RMSE across participants was 20.8+/-4.3% for the CNN, 23.3+/-3.8% for Random Forest, 24.8+/-4.5% for XGBoost, and 26.9+/-6.1% for Linear Regression. To probe cross-task generalization, one participant additionally performed unseen vertical (up-down) and circular repetitions; models trained only on lateral data were tested directly and largely retained accuracy, indicating robustness to changes in movement direction, arm kinematics, and muscle recruitment, while Linear Regression deteriorated. Overall, the study shows that both feature-based ML and spectrogram-based DL can estimate remaining work capacity during repetitive pHRI, with the CNN delivering the lowest error and the tree-based models close behind. The reported transfer to new motion patterns suggests potential for practical fatigue monitoring without retraining for every task, improving operator protection and enabling fatigue-aware shared autonomy, for safer fatigue-adaptive pHRI control.",
      "published": "2026-02-17T16:08:11Z",
      "abstract_url": "http://arxiv.org/abs/2602.15684v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15684v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.HC",
        "eess.SP",
        "eess.SY"
      ]
    },
    {
      "title": "Revisiting Northrop Frye's Four Myths Theory with Large Language Models",
      "authors": [
        "Edirlei Soares de Lima",
        "Marco A. Casanova",
        "Antonio L. Furtado"
      ],
      "abstract": "Northrop Frye's theory of four fundamental narrative genres (comedy, romance, tragedy, satire) has profoundly influenced literary criticism, yet computational approaches to his framework have focused primarily on narrative patterns rather than character functions. In this paper, we present a new character function framework that complements pattern-based analysis by examining how archetypal roles manifest differently across Frye's genres. Drawing on Jungian archetype theory, we derive four universal character functions (protagonist, mentor, antagonist, companion) by mapping them to Jung's psychic structure components. These functions are then specialized into sixteen genre-specific roles based on prototypical works. To validate this framework, we conducted a multi-model study using six state-of-the-art Large Language Models (LLMs) to evaluate character-role correspondences across 40 narrative works. The validation employed both positive samples (160 valid correspondences) and negative samples (30 invalid correspondences) to evaluate whether models both recognize valid correspondences and reject invalid ones. LLMs achieved substantial performance (mean balanced accuracy of 82.5%) with strong inter-model agreement (Fleiss' $κ$ = 0.600), demonstrating that the proposed correspondences capture systematic structural patterns. Performance varied by genre (ranging from 72.7% to 89.9%) and role (52.5% to 99.2%), with qualitative analysis revealing that variations reflect genuine narrative properties, including functional distribution in romance and deliberate archetypal subversion in satire. This character-based approach demonstrates the potential of LLM-supported methods for computational narratology and provides a foundation for future development of narrative generation methods and interactive storytelling applications.",
      "published": "2026-02-17T16:02:52Z",
      "abstract_url": "http://arxiv.org/abs/2602.15678v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15678v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry",
      "authors": [
        "Deniz Kucukahmetler",
        "Maximilian Jean Hemmann",
        "Julian Mosig von Aehrenfeld",
        "Maximilian Amthor",
        "Christian Deubel",
        "Nico Scherf",
        "Diaaeldin Taha"
      ],
      "abstract": "Neural networks can accurately forecast complex dynamical systems, yet how they internally represent underlying latent geometry remains poorly understood. We study neural forecasters through the lens of representational alignment, introducing anchor-based, geometry-agnostic relative embeddings that remove rotational and scaling ambiguities in latent spaces. Applying this framework across seven canonical dynamical systems - ranging from periodic to chaotic - we reveal reproducible family-level structure: multilayer perceptrons align with other MLPs, recurrent networks with RNNs, while transformers and echo-state networks achieve strong forecasts despite weaker alignment. Alignment generally correlates with forecasting accuracy, yet high accuracy can coexist with low alignment. Relative geometry thus provides a simple, reproducible foundation for comparing how model families internalize and represent dynamical structure.",
      "published": "2026-02-17T16:00:08Z",
      "abstract_url": "http://arxiv.org/abs/2602.15676v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15676v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra",
      "authors": [
        "Xiachong Feng",
        "Liang Zhao",
        "Weihong Zhong",
        "Yichong Huang",
        "Yuxuan Gu",
        "Lingpeng Kong",
        "Xiaocheng Feng",
        "Bing Qin"
      ],
      "abstract": "Current methods for personality control in Large Language Models rely on static prompting or expensive fine-tuning, failing to capture the dynamic and compositional nature of human traits. We introduce PERSONA, a training-free framework that achieves fine-tuning level performance through direct manipulation of personality vectors in activation space. Our key insight is that personality traits appear as extractable, approximately orthogonal directions in the model's representation space that support algebraic operations. The framework operates through three stages: Persona-Base extracts orthogonal trait vectors via contrastive activation analysis; Persona-Algebra enables precise control through vector arithmetic (scalar multiplication for intensity, addition for composition, subtraction for suppression); and Persona-Flow achieves context-aware adaptation by dynamically composing these vectors during inference. On PersonalityBench, our approach achieves a mean score of 9.60, nearly matching the supervised fine-tuning upper bound of 9.61 without any gradient updates. On our proposed Persona-Evolve benchmark for dynamic personality adaptation, we achieve up to 91% win rates across diverse model families. These results provide evidence that aspects of LLM personality are mathematically tractable, opening new directions for interpretable and efficient behavioral control.",
      "published": "2026-02-17T15:47:58Z",
      "abstract_url": "http://arxiv.org/abs/2602.15669v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15669v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens",
      "authors": [
        "Shiqi Liu",
        "Zeyu He",
        "Guojian Zhan",
        "Letian Tao",
        "Zhilong Zheng",
        "Jiang Wu",
        "Yinuo Wang",
        "Yang Guan",
        "Kehua Sheng",
        "Bo Zhang",
        "Keqiang Li",
        "Jingliang Duan",
        "Shengbo Eben Li"
      ],
      "abstract": "Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often experience late-stage performance collapse, leading to degraded reasoning quality and unstable training. We derive that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. Building on this result, we prove that training instability is driven by a tiny fraction of tokens, approximately 0.01\\%, which we term \\emph{spurious tokens}. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. Motivated by this observation, we propose Spurious-Token-Aware Policy Optimization (STAPO) for large-scale model refining, which selectively masks such updates and renormalizes the loss over valid tokens. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13\\% over GRPO, 20-Entropy and JustRL.",
      "published": "2026-02-17T14:46:48Z",
      "abstract_url": "http://arxiv.org/abs/2602.15620v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15620v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Intracoronary Optical Coherence Tomography Image Processing and Vessel Classification Using Machine Learning",
      "authors": [
        "Amal Lahchim",
        "Lambros Athanasiou"
      ],
      "abstract": "Intracoronary Optical Coherence Tomography (OCT) enables high-resolution visualization of coronary vessel anatomy but presents challenges due to noise, imaging artifacts, and complex tissue structures. This paper proposes a fully automated pipeline for vessel segmentation and classification in OCT images using machine learning techniques. The proposed method integrates image preprocessing, guidewire artifact removal, polar-to-Cartesian transformation, unsupervised K-means clustering, and local feature extraction. These features are used to train Logistic Regression and Support Vector Machine classifiers for pixel-wise vessel classification. Experimental results demonstrate excellent performance, achieving precision, recall, and F1-score values up to 1.00 and overall classification accuracy of 99.68%. The proposed approach provides accurate vessel boundary detection while maintaining low computational complexity and requiring minimal manual annotation. This method offers a reliable and efficient solution for automated OCT image analysis and has potential applications in clinical decision support and real-time medical image processing.",
      "published": "2026-02-17T13:47:27Z",
      "abstract_url": "http://arxiv.org/abs/2602.15579v1",
      "pdf_url": "https://arxiv.org/pdf/2602.15579v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    }
  ]
};
