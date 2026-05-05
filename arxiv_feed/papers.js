const PAPERS_DATA = {
  "last_updated": "2026-05-05 03:36:09 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "SpecKV: Adaptive Speculative Decoding with Compression-Aware Gamma Selection",
      "authors": [
        "Shikhar Shukla"
      ],
      "abstract": "Speculative decoding accelerates large language model (LLM) inference by using a small draft model to propose candidate tokens that a larger target model verifies. A critical hyperparameter in this process is the speculation length~$γ$, which determines how many tokens the draft model proposes per step. Nearly all existing systems use a fixed~$γ$ (typically~4), yet empirical evidence suggests that the optimal value varies across task types and, crucially, depends on the compression level applied to the target model. In this paper, we present \\textbf{SpecKV}, a lightweight adaptive controller that selects~$γ$ per speculation step using signals extracted from the draft model itself. We profile speculative decoding across 4~task categories, 4~speculation lengths, and 3~compression levels (FP16, INT8, NF4), collecting 5,112 step-level records with per-step acceptance rates, draft entropy, and draft confidence. We demonstrate that the optimal~$γ$ shifts across compression regimes and that draft model confidence and entropy are strong predictors of acceptance rate (correlation~$\\approx 0.56$). SpecKV uses a small MLP trained on these signals to maximize expected tokens per speculation step, achieving a 56.0\\% improvement over the fixed-$γ$=4 baseline with only 0.34\\,ms overhead per decision ($<$0.5\\% of step time). The improvement is statistically significant ($p < 0.001$, paired bootstrap test). We release all profiling data, trained models, and notebooks as open-source artifacts.",
      "published": "2026-05-04T17:55:05Z",
      "abstract_url": "http://arxiv.org/abs/2605.02888v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02888v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.DC",
        "eess.SY"
      ]
    },
    {
      "title": "Enhancing RL Generalizability in Robotics through SHAP Analysis of Algorithms and Hyperparameters",
      "authors": [
        "Lingxiao Kong",
        "Cong Yang",
        "Oya Deniz Beyan",
        "Zeyd Boukhers"
      ],
      "abstract": "Despite significant advances in Reinforcement Learning (RL), model performance remains highly sensitive to algorithm and hyperparameter configurations, while generalization gaps across environments complicate real-world deployment. Although prior work has studied RL generalization, the relative contribution of specific configurations to the generalization gap has not been quantitatively decomposed and systematically leveraged for configuration selection. To address this limitation, we propose an explainable framework that evaluates RL performance across robotic environments using SHapley Additive exPlanations (SHAP) to quantify configuration impacts. We establish a theoretical foundation connecting Shapley values to generalizability, empirically analyze configuration impact patterns, and introduce SHAP-guided configuration selection to enhance generalization. Our results reveal distinct patterns across algorithms and hyperparameters, with consistent configuration impacts across diverse tasks and environments. By applying these insights to configuration selection, we achieve improved RL generalizability and provide actionable guidance for practitioners.",
      "published": "2026-05-04T17:41:04Z",
      "abstract_url": "http://arxiv.org/abs/2605.02867v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02867v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "Standing on the Shoulders of Giants: Stabilized Knowledge Distillation for Cross--Language Code Clone Detection",
      "authors": [
        "Mohamad Khajezade",
        "Fatemeh H. Fard",
        "Mohamed Sami Shehata"
      ],
      "abstract": "Cross-language code clone detection (X-CCD) is challenging because semantically equivalent programs written in different languages often share little surface similarity. Although large language models (LLMs) have shown promise for semantic clone detection, their use as black-box systems raises concerns about cost, reproducibility, privacy, and unreliable output formatting. In particular, compact open-source models often struggle to follow reasoning-oriented prompts and to produce outputs that can be consistently mapped to binary clone labels. To address these limitations, we propose a knowledge distillation framework that transfers reasoning capabilities from DeepSeek-R1 into compact open-source student models for X-CCD. Using cross-language code pairs derived from Project CodeNet, we construct reasoning-oriented synthetic training data and fine-tune Phi3 and Qwen-Coder with LoRA adapters. We further introduce response stabilization methods, including forced conclusion prompting, a binary classification head, and a contrastive classification head, and evaluate model behavior using both predictive metrics and response rate. Experiments on Python--Java, Rust--Java, Rust--Python, and Rust--Ruby show that knowledge distillation consistently improves the reliability of compact models and often improves predictive performance, especially under distribution shift. In addition, classification-head variants substantially reduce inference time compared to generation-based inference. Overall, our results show that reasoning-oriented distillation combined with response stabilization makes compact open-source models more practical and reliable for X-CCD detection.",
      "published": "2026-05-04T17:37:16Z",
      "abstract_url": "http://arxiv.org/abs/2605.02860v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02860v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "A second-order method on the Stiefel manifold via Newton$\\unicode{x2013}$Schulz",
      "authors": [
        "Xinhui Xiong",
        "Bin Gao",
        "P. -A. Absil"
      ],
      "abstract": "Retraction-free approaches offer attractive low-cost alternatives to Riemannian methods on the Stiefel manifold, but they are often first-order, which may limit the efficiency under high-accuracy requirements. To this end, we propose a second-order method landing on the Stiefel manifold without invoking retractions, which is proved to enjoy local quadratic (or superlinear for its inexact variant) convergence. The update consists of the sum of (i) a component tangent to the level set of the constraint-defining function that aims to reduce the objective and (ii) a component normal to the same level set that reduces the infeasibility. Specifically, we construct the normal component via Newton$\\unicode{x2013}$Schulz, a fixed-point iteration for orthogonalization. Moreover, we establish a geometric connection between the Newton$\\unicode{x2013}$Schulz iteration and Stiefel manifolds, in which Newton$\\unicode{x2013}$Schulz moves along the normal space. For the tangent component, we formulate a modified Newton equation that incorporates Newton$\\unicode{x2013}$Schulz. Numerical experiments on the orthogonal Procrustes problem, principal component analysis, and real-data independent component analysis illustrate that the proposed method performs better than the existing methods.",
      "published": "2026-05-04T17:18:11Z",
      "abstract_url": "http://arxiv.org/abs/2605.02838v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02838v1",
      "categories": [
        "math.OC",
        "cs.AI",
        "cs.LG",
        "math.NA"
      ]
    },
    {
      "title": "First-Order Efficiency for Probabilistic Value Estimation via A Statistical Viewpoint",
      "authors": [
        "Ziqi Liu",
        "Kiljae Lee",
        "Yuan Zhang",
        "Weijing Tang"
      ],
      "abstract": "Probabilistic values, including Shapley values and semivalues, provide a model-agnostic framework to attribute the behavior of a black-box model to data points or features, with a wide range of applications including explainable artificial intelligence and data valuation. However, their exact computation requires utility evaluations over exponentially many coalitions, making Monte Carlo approximation essential in modern machine learning applications. Existing estimators are often developed through different identification strategies, including weighted averages, self-normalized weighting, regression adjustment, and weighted least squares. Our key observation is that these seemingly distinct constructions share a common first-order error structure, in which the leading term is an augmented inverse-probability weighted influence term determined by the sampling law and a working surrogate function. This first-order representation yields an explicit expression for the leading mean squared error (MSE), which characterizes how the sampling law and the surrogate jointly determine statistical efficiency. Guided by this criterion, we propose an Efficiency-Aware Surrogate-adjusted Estimator (EASE) that directly chooses the sampling law and surrogate to minimize the first-order MSE. We demonstrate that EASE consistently outperforms state-of-the-art estimators for various probabilistic values.",
      "published": "2026-05-04T17:02:17Z",
      "abstract_url": "http://arxiv.org/abs/2605.02827v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02827v1",
      "categories": [
        "cs.AI",
        "stat.ME",
        "stat.ML"
      ]
    },
    {
      "title": "SCPRM: A Schema-aware Cumulative Process Reward Model for Knowledge Graph Question Answering",
      "authors": [
        "Jiujiu Chen",
        "Yazheng Liu",
        "Sihong Xie",
        "Hui Xiong"
      ],
      "abstract": "Large language models excel at complex reasoning, yet evaluating their intermediate steps remains challenging. Although process reward models provide step-wise supervision, they often suffer from a risk compensation effect, where incorrect steps are offset by later correct ones, assigning high rewards to flawed reasoning paths. This issue is further exacerbated in knowledge graph (KG) reasoning, as there may exist multiple paths between the start and end entities in the KGs, and a risky step can make the reasoning path flawed. Those limitations are problematic in risk-sensitive tasks such as medical and legal KG reasoning. To address the issues, we propose a Schema-aware Cumulative Process Reward Model (SCPRM) that evaluates reasoning paths by conditioning on the reasoning prefix , and incorporating schema distance between current reasoning step and the implicit target parsed from the query, which provides cumulative and future rewards to guide the path explorations. We further integrate SCPRM into Monte Carlo Tree Search (MCTS) as SCPRM-MCTS to conduct multi-hop reasoning on KGs for question answering (QA) tasks. Across medical and legal KGQA and CWQ, SCPRM-MCTS improves the performance of Hits@k by an average of 1.18% over strong baselines, demonstrating more accurate and risk-sensitive reasoning evaluation.",
      "published": "2026-05-04T16:56:01Z",
      "abstract_url": "http://arxiv.org/abs/2605.02819v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02819v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Fine-Grained Graph Generation through Latent Mixture Scheduling",
      "authors": [
        "Nidhi Vakil",
        "Hadi Amiri"
      ],
      "abstract": "Structure aware graph generation aims to generate graphs that satisfy given topological properties. It has applications in domains such as drug discovery, social network modeling, and knowledge graph construction. Unlike existing methods that only provide coarse control over graph properties, we introduce a novel conditional variational autoencoder for fine-grained structural control in graph generation. The approach refines the decoder's latent space by dynamically aligning graph- and property-driven representations to improve both graph fidelity and control satisfaction. Specifically, the approach implements a mixture scheduler that progressively integrates graph and control priors. Experiments on five real-world datasets show the efficacy of the proposed model compared to recent baselines, achieving high generation quality while maintaining high controllability.",
      "published": "2026-05-04T16:23:01Z",
      "abstract_url": "http://arxiv.org/abs/2605.02780v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02780v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A decoupled diffusion planner that adapts to changing cost limits by using cost-conditioned generation for safety and reward gradients for performance",
      "authors": [
        "Rufeng Chen",
        "Zhaofan Zhang",
        "Zhejiang Yang",
        "Hechang Chen",
        "Sihong Xie"
      ],
      "abstract": "Offline safe reinforcement learning often requires policies to adapt at deployment time to safety budgets that vary across episodes or change within a single episode. While diffusion-based planners enable flexible trajectory generation, existing guidance schemes often treat reward improvement and constraint satisfaction as competing gradient objectives, which can lead to unreliable safety compliance under cost limits. We reinterpret adaptive safe trajectory generation as sampling from a constrained trajectory distribution, where the budget restricts the trajectory region, and reward shapes preferences within that region. This perspective motivates Safe Decoupled Guidance Diffusion (SDGD), which conditions classifier-free guidance on the cost limit to bias sampling toward trajectories satisfying the specified limit, while using reward-gradient guidance to refine trajectories for higher return. Because direct reward guidance can increase return while also steering samples toward trajectories with higher cumulative cost, we introduce Feasible Trajectory Relabeling (FTR) to reshape reward targets and discourage such directions. We further provide a first-order sampling-time analysis showing that FTR suppresses reward-induced cost drift under a prefix-restorative alignment condition. Extensive evaluations on the DSRL benchmark show that SDGD achieves the strongest safety compliance among baselines, satisfying the constraint on 94.7% of tasks (36/38), while obtaining the highest reward among safe methods on 21 tasks.",
      "published": "2026-05-04T16:19:42Z",
      "abstract_url": "http://arxiv.org/abs/2605.02777v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02777v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "U-Define: Designing User Workflows for Hard and Soft Constraints in LLM-Based Planning",
      "authors": [
        "Christine P Lee",
        "Xinyu Jessica Wang",
        "Aws Albarghouthi",
        "David Porfirio",
        "Bilge Mutlu"
      ],
      "abstract": "LLMs are increasingly used for end-user task planning, yet their black-box nature limits users' ability to ensure reliability and control. While recent systems incorporate verification techniques, it remains unclear how users can effectively apply such rigid constraints to represent intent or adapt to real-world variability. For example, prior work finds that hard-only constraints are too rigid, and numeric flexibility weights confuse users. We investigate how interaction workflows can better support users in applying constraints to guide LLM-generated plans, examining whether abstracting strictness into high-level types (i.e., hard and soft) paired with distinct verification mechanisms helps users more reliably express and align intent. We present U-Define, a system that lets users define constraints in natural language and categorize them as either hard rules that must not be violated or soft preferences that allow flexibility. U-Define verifies these types through complementary methods: formal model checking for hard constraints and LLM-as-judge evaluation for soft ones. Through a technical evaluation and user studies with general and expert participants, we find that user-defined constraint types improve perceived usefulness, performance, and satisfaction while maintaining usability. These findings provide insights for designing flexible yet reliable constraint-based workflows.",
      "published": "2026-05-04T16:05:40Z",
      "abstract_url": "http://arxiv.org/abs/2605.02765v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02765v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "Bolek: A Multimodal Language Model for Molecular Reasoning",
      "authors": [
        "Frederic Grabowski",
        "Jacek Szczerbiński",
        "Maciej Jaśkowski",
        "Kalina Jasińska-Kobus",
        "Paweł Dąbrowski-Tumański",
        "Tomasz Jetka",
        "Bartosz Topolski"
      ],
      "abstract": "Molecular property models increasingly support high-stakes drug-discovery decisions, but their outputs are often difficult to audit: classical predictors return scores without rationale, while language models can produce fluent explanations weakly grounded in the input molecule. We introduce Bolek, a compact multimodal language model that grounds natural-language reasoning in molecular structure by injecting a Morgan fingerprint embedding into an instruction-tuned text decoder. Bolek is fine-tuned on molecular alignment tasks, including molecule description, RDKit descriptor prediction, and substructure detection, and on downstream reasoning over 15 TDC binary classification tasks using synthetic chains-of-thought anchored in concrete molecular features. Across these tasks, Bolek outperforms its Qwen3-4B-Instruct base on all endpoints in yes/no mode and on 13 of 15 in chain-of-thought mode, raising mean ROC/PR AUC from 0.55 to 0.76. It also outperforms TxGemma-9B-Chat on 13 of 15 binary classification tasks despite being less than half its size. Bolek's explanations are more grounded than those of the baseline LLMs: it cites numerical descriptors 10-100x more often per chain-of-thought, and the cited values agree strongly with RDKit for key descriptors such as TPSA, MolLogP, and MolWt (Spearman rho = 0.87-0.91). Generalisation extends beyond the training panel: on 15 unseen TDC classification endpoints, Bolek matches TxGemma on five, and it produces non-trivial rank correlations on three held-out regression endpoints despite never seeing downstream regression during training. These results suggest that targeted modality injection and reasoning supervision tied to verifiable molecular features can yield compact, auditable molecular reasoning models.",
      "published": "2026-05-04T15:46:39Z",
      "abstract_url": "http://arxiv.org/abs/2605.02745v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02745v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.BM"
      ]
    },
    {
      "title": "AI-Generated Smells: An Analysis of Code and Architecture in LLM and Agent-Driven Development",
      "authors": [
        "Yuecai Zhu",
        "Nikolaos Tsantalis",
        "Peter C. Rigby"
      ],
      "abstract": "The promise of Large Language Models in automated software engineering is often measured by functional correctness, overlooking the critical issue of long term maintainability. This paper presents a systematic audit of technical debt in AI-generated software, revealing that AI does not eliminate flaws but rather introduces a distinct machine signature of defects. Our multi-scale analysis, spanning single-file algorithmic tasks and complex, agent generated systems, identifies a fundamental Reasoning-Complexity Trade-off: as models become more capable, they generate increasingly bloated and coupled code. This architectural decay is so pronounced that we establish a Volume-Quality Inverse Law, where code volume is a near perfect predictor of structural degradation. Crucially, we demonstrate that neither functional correctness nor detailed prompting mitigates this decay. These findings challenge the current paradigm of prompt-driven generation, reframing the central problem of AI-based software engineering from one of code generation to one of architectural complexity management. We conclude that future progress depends on equipping agents with explicit architectural foresight to ensure the software they build is not just functional, but also maintainable.",
      "published": "2026-05-04T15:41:13Z",
      "abstract_url": "http://arxiv.org/abs/2605.02741v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02741v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "ProPACT: A Proactive AI-Driven Adaptive Collaborative Tutor for Pair Programming",
      "authors": [
        "Anahita Golrang",
        "Kshitij Sharma",
        "olga viberg"
      ],
      "abstract": "Effective pair programming depends on coordination of attention, cognitive effort, and joint regulation over time, yet most adaptive learning systems remain individual-centric and reactive. This paper introduces ProPACT, a proactive AI-driven adaptive collaborative tutor that treats collaboration itself as the object of instruction. ProPACT constructs a multimodal dyadic learner model based on Joint Visual Attention (JVA), Joint Mental Effort (JME), and individual mental effort, and employs an XGBoost-based forecasting model to predict emerging suboptimal collaboration states up to 30 seconds in advance. These predictions drive a hierarchical adaptive policy that delivers minimally intrusive scaffolds while fading support during productive collaboration. A within-subject study with 26 pair-programming dyads shows that proactive feedback significantly improves debugging success, task efficiency, feedback uptake, and post-intervention gains in JVA and JME, demonstrating the potential of forecast-driven dyadic adaptivity for real-time collaborative learning regulation.",
      "published": "2026-05-04T15:12:49Z",
      "abstract_url": "http://arxiv.org/abs/2605.02703v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02703v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning Equivariant Neural-Augmented Object Dynamics From Few Interactions",
      "authors": [
        "Sergio Orozco",
        "Tushar Kusnur",
        "Brandon May",
        "George Konidaris",
        "Laura Herlant"
      ],
      "abstract": "Learning data-efficient object dynamics models for robotic manipulation remains challenging, especially for deformable objects. A popular approach is to model objects as sets of 3D particles and learn their motion using graph neural networks. In practice, this is not enough to maintain physical feasibility over long horizons and may require large amounts of interaction data to learn. We introduce PIEGraph, a novel approach to combining analytical physics and data-driven models to capture object dynamics for both rigid and deformable bodies using limited real-world interaction data. PIEGraph consists of two components: (1) a \\textbf{P}hysically \\textbf{I}nformed particle-based analytical model (implemented as a spring--mass system) to enforce physically feasible motion, and (2) an \\textbf{E}quivariant \\textbf{Graph} Neural Network with a novel action representation that exploits symmetries in particle interactions to guide the analytical model. We evaluate PIEGraph in simulation and on robot hardware for reorientation and repositioning tasks with ropes, cloth, stuffed animals and rigid objects. We show that our method enables accurate dynamics prediction and reliable downstream robotic manipulation planning, which outperforms state of the art baselines.",
      "published": "2026-05-04T15:11:22Z",
      "abstract_url": "http://arxiv.org/abs/2605.02699v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02699v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Hybrid Inspection and Task-Based Access Control in Zero-Trust Agentic AI",
      "authors": [
        "Majed El Helou",
        "Benjamin Ryder",
        "Chiara Troiani",
        "Jean Diaconu",
        "Hervé Muyal",
        "Marcelo Yannuzzi"
      ],
      "abstract": "Authorizing Large Language Model (LLM)-driven agents to dynamically invoke tools and access protected resources introduces significant security risks, and the risks grow dramatically as agents engage in multi-turn conversations and scale toward distributed collaboration. A compromised or malicious agentic application can tamper with tool calls, falsify results, or request permissions beyond the scope of the subject's intended tasks, which could go unnoticed with current delegated authorization flows given their lack of visibility into the original subject's intent. In light of this, we make the following contributions towards Continuous Agent Semantic Authorization (CASA). First, we propose a hybrid runtime enforcement model that combines deterministic and semantic controls enabled by a zero-trust interception layer. Five deterministic controls enforce structural and data-integrity guarantees over the message flow, while a semantic inspection layer evaluates whether tool call choices align with the intended tasks commissioned to the agent. Second, differently from prior Task-Based Access Control (TBAC) techniques that operate on single-turn interactions, we decompose the semantic layer into two stages: i) a task-extraction step that distills the subject's objectives from multi-turn conversations at the interception layer, and ii) a task-tool semantic matching step at the authorization server that evaluates whether the requested tools are appropriate for the extracted tasks. Third, we extend the ASTRA dataset that we introduced in a prior work, by generating novel conversation-tool datasets with multi-turn interactions containing relevant and irrelevant tool calls for a given task. Lastly, we provide the first experimental results for TBAC under multi-turn conversations.",
      "published": "2026-05-04T15:00:37Z",
      "abstract_url": "http://arxiv.org/abs/2605.02682v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02682v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Trustworthy AI Suffers from Invariance Conflicts and Causality is The Solution",
      "authors": [
        "Ruta Binkyte",
        "Ivaxi Sheth",
        "Zhijing Jin",
        "Mohammad Havaei",
        "Bernhard Schölkopf",
        "Mario Fritz"
      ],
      "abstract": "As artificial intelligence (AI), including machine learning (ML) models and foundation models (FMs), is increasingly deployed in high-stakes domains, ensuring their trustworthiness has become a central challenge. However, the core trustworthy AI objectives, such as fairness, robustness, privacy, and explainability, are hard to achieve simultaneously, especially while preserving utility. This position paper argues that causality is necessary to understand and balance trade-offs in performance and multiple objectives of trustworthy AI. We ground our arguments in re-interpreting trustworthy AI trade-offs as incompatible invariance requirements under different changes to the data-generating process. We then illustrate that causality provides a unifying framework for understanding how trade-offs in trustworthy AI arise, and how they can be softened or resolved through selective invariance. This perspective applies to both classical ML models and large-scale FMs. Our paper discusses how causal assumptions may be applied explicitly or implicitly in modern large-scale systems. Finally, we outline open challenges and opportunities for using causality to build more trustworthy AI.",
      "published": "2026-05-04T14:26:28Z",
      "abstract_url": "http://arxiv.org/abs/2605.02640v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02640v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Dependency Parsing Across the Resource Spectrum: Evaluating Architectures on High and Low-Resource Languages",
      "authors": [
        "Kevin Guan",
        "Happy Buzaaba",
        "Christiane Fellbaum"
      ],
      "abstract": "Transformer-based models achieve state-of-the-art dependency parsing for high-resource languages, yet their advantage over simpler architectures in low-resource settings remains poorly understood. We evaluate four parsers -- the Biaffine LSTM, Stack-Pointer Network, AfroXLMR-large, and RemBERT -- across ten typologically diverse languages, with a focus on low-resource African languages. We find that the Biaffine LSTM consistently outperforms transformer models in low-resource regimes, with transformers recovering their advantage as training data increases. The crossover falls within a resource range typical of treebanks for under-resourced languages. Morphological complexity (measured via MATTR) emerges as a significant secondary predictor of transformers' relative disadvantage after controlling for corpus size. These results indicate that the Biaffine LSTM may be better suited for syntactic tool development in low-resource regimes until sufficient annotated data is available to leverage the representational capacity of pre-trained transformers.",
      "published": "2026-05-04T13:55:32Z",
      "abstract_url": "http://arxiv.org/abs/2605.02608v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02608v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation",
      "authors": [
        "Berk Çiçek",
        "Mert K. Er",
        "Özgür S. Öğüz"
      ],
      "abstract": "While Large Language Models (LLMs) and Vision-Language Models (VLMs) demonstrate remarkable capabilities in high-level reasoning and semantic understanding, applying them directly to contact-rich manipulation remains a challenge due to their lack of explicit physical grounding and inability to perform adaptive control. To bridge this gap, we propose CoRAL (Contact-Rich Adaptive LLM-based control), a modular framework that enables zero-shot planning by decoupling high-level reasoning from low-level control. Unlike black-box policies, CoRAL uses LLMs not as direct controllers, but as cost designers that synthesize context-aware objective functions for a sampling-based motion planner (MPPI). To address the ambiguity of physical parameters in visual data, we introduce a neuro-symbolic adaptation loop: a VLM provides semantic priors for environmental dynamics, such as mass and friction estimates, which are then explicitly refined in real time via online system identification, while the LLM iteratively modulates the cost-function structure to correct strategic errors based on interaction feedback. Furthermore, a retrieval-based memory unit allows the system to reuse successful strategies across recurrent tasks. This hierarchical architecture ensures real-time control stability by decoupling high-level semantic reasoning from reactive execution, effectively bridging the gap between slow LLM inference and dynamic contact requirements. We validate CoRAL on both simulation and real-world hardware across challenging and novel tasks, such as flipping objects against walls by leveraging extrinsic contacts. Experiments demonstrate that CoRAL outperforms state-of-the-art VLA and foundation-model-based planner baselines by boosting success rates over 50% on average in unseen contact-rich scenarios, effectively handling sim-to-real gaps through its adaptive physical understanding.",
      "published": "2026-05-04T13:49:19Z",
      "abstract_url": "http://arxiv.org/abs/2605.02600v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02600v1",
      "categories": [
        "cs.RO",
        "cs.AI"
      ]
    },
    {
      "title": "Foundation-Model-Based Agents in Industrial Automation: Purposes, Capabilities, and Open Challenges",
      "authors": [
        "Vincent Henkel",
        "Felix Gehlhoff",
        "David Kube",
        "Asaad Almutareb",
        "Luis Cruz",
        "Bernd Hellingrath",
        "Philip Koch",
        "Christoph Legat",
        "Florian Mohr",
        "Michael Oberle",
        "Felix Ocker",
        "Thorsten Schoeler",
        "Mario Thron",
        "Nico Andre Töpfer",
        "Lucas Vogt",
        "Yuchen Xia"
      ],
      "abstract": "Foundation models, particularly large language models, are increasingly integrated into agent architectures for industrial tasks such as decision support, process monitoring, and engineering automation. Yet evidence on their purposes, capabilities, and limitations remains fragmented across domains. This work examines how mature foundation-model-based agent systems are in industrial contexts, how their functional profile differs from conventional agent systems, and which limitations persist. A systematic literature survey following the PRISMA 2020 guideline is presented, screening 2,341 publications and synthesising a corpus of 88 publications through a structured coding scheme. The results show that reported systems are predominantly at prototype and early validation stages (75.0% at TRL 4-6), with deployment-oriented evidence remaining rare (9.1%). Operational goals are most frequently positioned in user assistance, monitoring, and process optimisation, while conventional production-control purposes such as planning and scheduling are less prominent. Compared with an established baseline for industrial agent systems, the capability profile reveals substantial gains in human interaction (+37%) and dealing with uncertainty (+35%), but a pronounced deficit in negotiation (-39%). The most widely reported limitations concern lack of generalization, hallucination and output instability, data scarcity, and inference latency. A working definition of foundation-model-based industrial agents is also proposed, bridging conventional agent theory, automation-engineering standards, and the foundation-model paradigm.",
      "published": "2026-05-04T13:44:22Z",
      "abstract_url": "http://arxiv.org/abs/2605.02592v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02592v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Beyond State Machines: Executing Network Procedures with Agentic Tool-Calling Sequences",
      "authors": [
        "Purna Sai Garigipati",
        "Onur Ayan",
        "Kishor Chandra Joshi",
        "Xueli An"
      ],
      "abstract": "Agentic AI will be an essential enabling technology for designing future mobile communication systems, which could provide flexible and customized services, automate complex network operations, and drive autonomous decision-making across the network. This work studies how Large Language Model (LLM)-based network AI agents can be utilized to execute network procedures expressed as sequences of tool invocations. We investigate four approaches, which differ in how the agent obtains the procedure and in how execution is distributed between the agent and the underlying tools. We evaluated the latency and execution correctness across these approaches using a User Equipment (UE) IP allocation procedure as a case study. Furthermore, we conduct a stress test to examine how many sequential procedural steps an LLM agent can reliably execute before failure. Our results show that approaches relying on iterative agent-side reasoning incur higher latency and are more prone to execution errors, while approaches where the procedure is encapsulated within a single tool, which internally orchestrates the required steps by invoking other tools, reduce latency by limiting repeated reasoning. The stress-test results further show that the model with advanced tool-calling capability maintains reliable execution over longer procedures than the other evaluated models; however, all models exhibit reliability degradation as procedure length increases, revealing clear execution limits in multi-step tool-based workflows. To systematically analyze failures in procedure execution, we introduce a procedure-specific error taxonomy that categorizes deviations in multi-step procedural execution.",
      "published": "2026-05-04T13:34:20Z",
      "abstract_url": "http://arxiv.org/abs/2605.02584v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02584v1",
      "categories": [
        "cs.NI",
        "cs.AI"
      ]
    },
    {
      "title": "On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length",
      "authors": [
        "Sunghwan Kim",
        "Junhee Cho",
        "Beong-woo Kwak",
        "Taeyoon Kwon",
        "Liang Wang",
        "Nan Yang",
        "Xingxing Zhang",
        "Furu Wei",
        "Jinyoung Yeo"
      ],
      "abstract": "Large language models (LLMs) have shown promise as interactive agents that solve tasks through extended sequences of environment interactions. While prior work has primarily focused on system-level optimizations or algorithmic improvements, the role of task horizon length in shaping training dynamics remains poorly understood. In this work, we present a systematic empirical study that examines horizon length through controlled task constructions. Specifically, we construct controlled tasks in which agents face identical decision rules and reasoning structures, but differ only in the length of action sequences required for successful completion. Our results reveal that increasing horizon length alone constitutes a training bottleneck, inducing severe training instability driven by exploration difficulties and credit assignment challenges. We demonstrate that horizon reduction is a key principle to address this limitation, stabilizing training and achieving better performance in long-horizon tasks. Moreover, we find that horizon reduction is related to stronger generalization across horizon lengths: models trained under reduced horizons generalize more effectively to longer-horizon variants at inference time, a phenomenon we refer to as horizon generalization.",
      "published": "2026-05-04T13:25:05Z",
      "abstract_url": "http://arxiv.org/abs/2605.02572v1",
      "pdf_url": "https://arxiv.org/pdf/2605.02572v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
