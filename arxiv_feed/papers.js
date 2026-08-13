const PAPERS_DATA = {
  "last_updated": "2026-08-13 02:31:36 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses",
      "authors": [
        "Cheng Qian",
        "Wenting Zhao",
        "Liangwei Yang",
        "Heng Wang",
        "Jielin Qiu",
        "Heng Ji",
        "Silvio Savarese",
        "Huan Wang",
        "Shelby Heinecke"
      ],
      "abstract": "Recent work on distillation transfers the capabilities of large models to smaller ones often by updating the latter's parameters, through teacher forcing, on-policy distillation, and related training-time methods. In this paper, we ask whether such transfer can instead occur at test time. We study strong-to-weak scaffolding: whether a stronger builder model can construct inference-time harnesses that help a weaker target model solve tasks more reliably without any parameter updates. Using four representative Theory-of-Mind benchmarks, each builder model uses 5% of the data as a validation set to iteratively refine its harness over multiple rounds, after which the finalized harness is evaluated on the full test set. Empirically, this form of test-time capability transfer is highly effective, nearly doubling average target-model performance from 0.49 to 0.91. Our analysis shows that the gains come primarily from offloading unstable model reasoning into deterministic code, benchmark-specific routing, and strict answer-format enforcement, rather than from encouraging the target model to reason more extensively or sample more broadly. We further find that builder-model reasoning effort improves harness quality monotonically, platform effects are modest relative to the builder model's own capability, and weaker target models receive the largest gains. These results suggest that inference-time harness design is an important complement to conventional training-time distillation, enabling strong models to transfer cognitive structure to weaker models without retraining.",
      "published": "2026-08-12T17:53:18Z",
      "abstract_url": "http://arxiv.org/abs/2608.12307v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12307v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Redistribution-based Cost Inference Improves Sparse Safe Offline RL",
      "authors": [
        "Ebenezer Gelo",
        "Geraud Nangue Tasse",
        "Steven James",
        "Benjamin Rosman"
      ],
      "abstract": "Safe offline RL typically assumes access to dense per-step cost annotations, but in practice supervisors provide only trajectory-level stop-feedback: a binary signal at the first unsafe transition, with no per-step attribution. We frame this as a temporal credit assignment problem and propose the Redistribution-based Cost Inference (RCI) framework, which converts sparse stop-feedback into dense per-step costs via return decomposition, then trains a constrained offline policy on the augmented dataset. We show that return-equivalent redistribution preserves the feasible policy set and the optimal Lagrangian in a CMDP, establishing that the transformation is lossless in theory while yielding better-conditioned cost critic learning in practice. Experiments on highway driving and robotic manipulation demonstrate substantially lower violation rates than sparse and classifier-based baselines, with robustness to heterogeneous dataset compositions and label noise.",
      "published": "2026-08-12T17:53:15Z",
      "abstract_url": "http://arxiv.org/abs/2608.12306v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12306v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Constructing Dynamic Master Logic Models as Knowledge Graphs for Complex System Diagnostics Using Retrieval-Augmented Large Language Models",
      "authors": [
        "Saman Marandi",
        "Yu-Shu Hu",
        "Mohammad Modarres"
      ],
      "abstract": "Dynamic Master Logic (DML) provides a hierarchical framework for representing system behavior by linking functional objectives to underlying structural elements. However, DML construction typically relies on expert interpretation of technical documentation, limiting scalability for complex systems. This study presents a framework for automated construction of DML models from system descriptions and their representation as Knowledge Graphs (KG-DML), using Retrieval-Augmented Generation and Large Language Models as enabling tools. Building on prior work with small-scale systems, the framework extends automated KG-DML construction and evaluation to substantially larger and more complex systems. Model construction proceeds across the DML hierarchy using targeted retrieval while preserving functional dependencies and explicit logical relationships. The resulting KG-DML supports diagnostic reasoning, safety assessment, upward failure propagation, and downward dependency tracing. A multi-level validation methodology evaluates layer-specific precision and recall, logical gate consistency, and overall structural integrity. Application to the Low-Pressure Coolant Injection system of a decommissioned Boiling Water Reactor demonstrates consistent reconstruction across repeated runs. The results show that automated KG-DML construction can transform technical documentation into executable functional models for diagnostic and reliability analysis.",
      "published": "2026-08-12T17:50:39Z",
      "abstract_url": "http://arxiv.org/abs/2608.12304v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12304v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Trial-and-Error: Agentic Optimization for Image-to-Video Adherence",
      "authors": [
        "Aman Tyagi",
        "Hemanth Boinpally",
        "Jonathan Chen",
        "Douglas Gebert",
        "Steven Hickson"
      ],
      "abstract": "Modern black-box Image-to-Video (I2V) models offer powerful capabilities in automated content creation, yet their lack of fine-grained control and reliability presents significant challenges in professional workflows. Their inherent stochasticity causes minor variations in textual prompts or hyperparameters to yield drastically different outputs often necessitating inefficient, brute-force trial-and-error processes. To address these limitations, we introduce the ``Agentic Self-Improvement\" framework, which reframes video synthesis into a closed-loop, goal-directed optimization. Our framework systematically navigates the generation parameter space using a novel two-stage approach. In the first stage, an iterative prompt optimization loop uses a multimodal Large Language Model (mLLM) to refine the input prompt. This refinement implements two automated evaluations: Davidsonian Scene Graph (DSG) queries ensure semantic adherence, and Common Mistake Questions (CMQ) for artifact detection. At the second stage, we use Bayesian optimization to efficiently co-optimize stochastic seeds and CFG scales. This search is guided by a suite of quality metrics, including the novel Video-Text Adherence (VTA) score derived from the DSG and CMQ evaluations. Our framework significantly outperforms unguided search methods: in human preference studies, videos generated via our agentic approach were strongly preferred over baseline outputs, achieving win rates up to 69\\%. This work provides a practical and extensible methodology for enhancing the predictability and control of state-of-the-art video generation models, moving the field beyond speculative curiosities toward reliable, production-ready tools.",
      "published": "2026-08-12T17:35:16Z",
      "abstract_url": "http://arxiv.org/abs/2608.12290v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12290v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.MM"
      ]
    },
    {
      "title": "Diagram-MMU: A Multi-Modal Benchmark for Scientific Diagrams",
      "authors": [
        "Weihao Bo",
        "Shan Zhang",
        "Yanpeng Sun",
        "Jie Liu",
        "Yongke Yao",
        "Jinhao Du",
        "Wei He",
        "Kai Zou",
        "Zechao Li",
        "Jingdong Wang"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) have been growing the capability for scientific writing and collaboration. For example, OpenAI Prism is a free workspace for scientific writing and collaboration. One important feature in Prism is turning scientific diagrams directly into LaTeX TikZ code. In this paper, we build a benchmark, Diagram-MMU, a multi-modal benchmark designed to assess MLLMs' ability for scientific diagram parsing and understanding. Diagram-MMU features 3.7k curated diagrams and 18.3k human-validated questions across six domains. It evaluates MLLMs on three tasks common in vibe writing workspaces: diagram-to-code parsing, diagram-to-code editing, and diagram question answering, alongside agentic settings per task. The evaluation of 12 MLLMs reveals that diagram-to-code tasks are more challenging than diagram question answering: models can reason well over diagrams but struggle to parse and edit them, underscoring the need for methods to enhance MLLMs' capability in diagram-to-code generation. Under agentic settings, most models improve parsing and editing performance but degrade on question answering, while Claude-4.6 Opus consistently improves across all three tasks. Project Page: https://vi-ocean.github.io/projects/diagram-mmu.",
      "published": "2026-08-12T17:04:13Z",
      "abstract_url": "http://arxiv.org/abs/2608.12262v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12262v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "One Frozen Simulator Is Not Enough: Simulator Collapse in Multi-Agent RL",
      "authors": [
        "Simon Yu",
        "Nicholas Tomlin",
        "Marwa Abdulhai",
        "Ximing Lu",
        "Derek Chong",
        "Abe Hou",
        "Dilara Soylu",
        "Sergey Levine",
        "Christopher D. Manning",
        "Weiyan Shi"
      ],
      "abstract": "Multi-agent reinforcement learning for human-AI interaction typically relies on a single large language model to simulate user behavior. We show that this approach systematically fails to generalize, and trace the failure to simulator collapse: because the simulator LLM is mode-collapsed, an LLM policy trained against it overfits to narrow strategies that exploit the simulator's dominant mode, and such a policy transfers poorly to unseen simulators and real users. We formalize this collapse theoretically and propose two complementary solutions, one at inference time and one at training time. The inference-time solution, Verbalized Sampling, broadens the simulator's behavior by sampling from a verbalized response distribution, reducing mode collapse. The training-time solution, Co-Training, jointly optimizes the policy against a population of trainable simulators, preventing it from overfitting to any single simulator's mode. We validate both solutions on three multi-turn benchmarks: Persuasion for Good, $τ^2$-bench, and CooperBench. Verbalized Sampling improves held-out success by up to 9% over single-simulator RL, and Co-Training pushes gains further to 14%; the human study shows similar gain on real users. Both solutions preserve the policy diversity that collapses under single-simulator RL. To support further work in this direction, we release SCOPE, an open-source framework for Population Co-Training multi-agent RL. More broadly, our results suggest that the diversity of the training environment, not only the policy, is critical to the generalization of multi-turn RL to real-world deployment.",
      "published": "2026-08-12T16:55:50Z",
      "abstract_url": "http://arxiv.org/abs/2608.12253v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12253v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Information Abundance Paradox: Long-Context Training Undermines Parametric Knowledge",
      "authors": [
        "Arda Uzunoglu",
        "Benjamin van Durme",
        "Daniel Khashabi"
      ],
      "abstract": "Large language models are increasingly trained and deployed with long contexts that span documents, code repositories, and interaction histories. This scaling reflects the implicit assumption that training on longer contexts will only help the model by exposing it to richer evidence. We challenge this view by studying how the context window shapes a model's mode of learning, shifting it between parametric internalization and contextualization. We propose the Information Abundance Paradox, which hypothesizes that abundant relevant information in the training context can reduce the incentive to encode that information parametrically, thereby increasing reliance on context. In pretraining with long documents, increasing the context window improves language modeling, natural language understanding, and closed-book MCQA only up to an intermediate optimum, after which performance consistently declines. In supervised fine-tuning, more task-relevant train-time context improves performance with supporting context, but reduces robustness when context is absent or misleading at test time. Our analysis suggests that this behavior arises when longer context provides a lower complexity solution. Mechanistically, training with informative context shifts gradient pressure from feed-forward networks, often linked to parametric knowledge, toward attention modules, and causal interventions show that this shift increases reliance on context during inference. Overall, these findings support the Information Abundance Paradox and suggest that scaling toward near-infinite context is not simply a matter of supplying more data, even when high-quality long-context data is abundant.",
      "published": "2026-08-12T16:13:05Z",
      "abstract_url": "http://arxiv.org/abs/2608.12218v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12218v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Learning-Based Behavior Planning for Automated Driving: Real-World Integration and Deployment",
      "authors": [
        "Jean-Pierre Busch",
        "Guido Linden",
        "Jan Bergmann",
        "Lutz Eckstein"
      ],
      "abstract": "Recent research in machine and deep learning has shown the potential of learningbased motion planning approaches to improve the driving behavior of automated vehicles, especially in complex environments. However, their complex nature and lack of transparency can hinder explainability and trustworthiness and complicate safety assurance. Motivated by these challenges, we propose a hybrid planning architecture that combines the advantages of machine learning with the verifiability and the determinism of classical approaches. Specifically, we developed a deep neural network to interpret complex traffic scenes and propose driving behavior, while an optimization-based supervision layer validates this proposal and enforces explicit drivability and safety constraints. We evaluate the learned planner's driving behavior in open-loop studies on real-world urban data, discuss system integration aspects for stable closed-loop operation, and report results from real-world deployment on our research vehicle karl..",
      "published": "2026-08-12T15:52:18Z",
      "abstract_url": "http://arxiv.org/abs/2608.12198v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12198v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "NetlistBench: Evaluating LLM Reliability in SPICE Netlist Recognition and Manipulation",
      "authors": [
        "Jiarui Ma",
        "Jianghan Wang",
        "Yuheng Ma",
        "Ziyi Zhuang",
        "Xiaoguang Liu"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly used in circuit design workflows, yet their reliability on simulator-facing SPICE netlist recognition and manipulation remains poorly understood and is rarely separated from high-level design reasoning. Although netlists are textual, they encode structured circuit objects through topology and parameters. We present \\textbf{NetlistBench}, a structure-verified benchmark for SPICE netlist recognition and manipulation. NetlistBench contains 2,342 cases across 24 task families, covering parameter and connectivity recognition and edits, hierarchical operations, equivalence judgment, and long-horizon compound editing. Model outputs are evaluated by a deterministic structure-aware oracle. Across six non-thinking LLMs, performance varies substantially with operation-level structural complexity. Simple local edits reach $96\\%$--$100\\%$ accuracy, while device addition drops to $41\\%$--$83\\%$ and equivalence judgment to $49\\%$--$90\\%$. Enabling reasoning substantially improves weaker models but does not eliminate structure-preservation failures, with performance still degrading sharply as the edit horizon increases. NetlistBench identifies netlist reliability as a distinct bottleneck for trustworthy LLM-based circuit design automation.",
      "published": "2026-08-12T15:51:52Z",
      "abstract_url": "http://arxiv.org/abs/2608.12197v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12197v1",
      "categories": [
        "eess.SY",
        "cs.AI"
      ]
    },
    {
      "title": "HYDRA: Hyperbolic Dynamic Representation Architecture for Kolmogorov-Arnold Networks",
      "authors": [
        "Zhao Su",
        "Yuxin Xia",
        "Haoran Li",
        "Jun Shen",
        "Qi Zhu",
        "Qingguo Zhou",
        "Binbin Yong"
      ],
      "abstract": "Kolmogorov-Arnold Networks (KANs) enhance nonlinear function approximation by replacing scalar weights with learnable univariate functions. However, assigning an independent function to every connection results in substantial parameter redundancy, limiting their scalability and efficiency. To reduce this redundancy, we introduce \\textbf{HY}perbolic \\textbf{D}ynamic \\textbf{R}epresentation \\textbf{A}rchitecture (HYDRA), a parameter-efficient hyperbolic extension of KAN that combines spline-based functional learning with representations in the Poincaré ball. HYDRA maps vector-valued inputs into a bounded hyperbolic latent space, performs KAN-style updates in tangent space, and employs a low-rank prototype block to share functional transformations across hidden dimensions. The resulting hyperbolic representations provide a structured radial coordinate for interpretation, while radius control improves training stability by preventing boundary saturation. Extensive experiments across eight benchmark datasets demonstrate that HYDRA consistently achieves competitive or superior predictive performance while improving parameter efficiency and representation interpretability.",
      "published": "2026-08-12T15:48:36Z",
      "abstract_url": "http://arxiv.org/abs/2608.12194v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12194v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "How to Spend Your Oracle Budget: Practical Guidance for Protein Structure Prediction Models",
      "authors": [
        "Aleksandra Kalisz",
        "Jack Simons",
        "Krisztina Sinkovics",
        "Noam Ghenassia",
        "Shikha Surana",
        "Henry Moss",
        "Paul Duckworth"
      ],
      "abstract": "Foundation models for protein structure prediction remain unreliable on certain targets. External oracles can flag and correct these failures, but biological oracles are expensive, making oracle budget a critical constraint. Existing guidance methods, such as FK-steering, DPO, and Best K-of-N sampling, differ in how they spend this budget, yet no systematic comparison exists to guide method selection. To bridge this gap, we benchmark these methods alongside the recently proposed Optimisation Over Outputs (O3), which applies off-the-shelf optimisers within a generative model's latent subspace. We extend the usage of O3 to protein structure prediction models. Overall, our work provides the first practical reference for oracle budget-aware guidance. Our evaluation on two protein targets, calmodulin (1CLL) and E. coli aspartate transcarbamoylase (9EEH), reveals that no single method consistently dominates across all budgets and oracles. Specifically, O3 proves most effective at low oracle budgets, while FK-steering and DPO demonstrate improved performance as the budget increases. We distil these findings into actionable recommendations for practitioners operating under real-world oracle-budget constraints.",
      "published": "2026-08-12T15:46:57Z",
      "abstract_url": "http://arxiv.org/abs/2608.12192v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12192v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Machine Learning-Based Cyber Defense for Cloud Infrastructure: An Adaptive Deep Q-Network Architecture for Intelligent Intrusion Detection and Automated Threat Mitigation",
      "authors": [
        "Md Yassir Mottalib",
        "Md Yousuf",
        "Eklachur Rahman Bhuiyan",
        "S M Ahsan Habib",
        "Sonjoy Kumar Dey",
        "Md. Salahuddin Gazi",
        "Molay Kumar Roy",
        "Asaduzzaman Anik"
      ],
      "abstract": "With the increasing complexity of cyber assaults in cloud environments, adaptable security solutions are needed that can support real-time detection and autonomous response. In this paper, we propose a reinforcement learning-based dynamic cyber defense framework. We deploy a Deep Q-Network (DQN) to train effective defensive strategies to counteract the evolving cyberattacks. We leverage the CICIDS2017 dataset for model creation and the UNSW-NB15 dataset for external validation, involving preprocessing of data, feature engineering, and adaptive policy learning. We compare the proposed DQN with decision tree, support vector machine, random forest, XGBoost, and multilayer perceptron models. The proposed DQN achieves an accuracy of 99.72%, a precision of 99.68%, a recall of 99.65%, an F1-score of 99.66%, and an ROC-AUC of 0.999, while the false positive rate is 0.31%, the false negative rate is 0.35%, and the detection latency is 15 ms. The framework achieved 99.54% attack mitigation rate, demonstrating strong adaptive and real-time defensive capabilities. These results demonstrate the potential of reinforcement learning as a powerful and scalable approach for autonomous cybersecurity in modern cloud environments.",
      "published": "2026-08-12T15:46:17Z",
      "abstract_url": "http://arxiv.org/abs/2608.12190v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12190v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Who Thinks Best Depends on How Long You Let Them: Budget-Dependent Rankings in LLM Evaluation",
      "authors": [
        "Rodrigo Guedes de Souza",
        "Alison R. Panisson"
      ],
      "abstract": "Standard evaluation of large language models assumes stable model rankings across inference conditions. We challenge this assumption by varying the token generation budget, i.e., the maximum tokens a model may produce, across seven levels (64--4,096), evaluating four models on three reasoning benchmarks (56,476 inferences). We report four findings: (i) 3--19% of items exhibit non-monotone behavior (accuracy decreasing with more budget), even after controlling for truncation, and this phenomenon is model-specific (cross-model overlap: 6--14%). (ii) Model rankings reverse across budgets on all benchmarks ($p {<} 0.01$, McNemar). (iii) Oracle analysis reveals model complementarity up to $+27.8$pp, most pronounced at constrained budgets. (iv) A budget-aware router captures 14.1% of the oracle gap cross-domain; budget features help within-domain ($+1.6$ to $+5.7$pp) but are domain-specific and hurt transfer ($-1.2$pp). These results argue for budget-conditioned evaluation protocols.",
      "published": "2026-08-12T15:11:35Z",
      "abstract_url": "http://arxiv.org/abs/2608.12150v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12150v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "A corpus-specific clinical RAG system matches or outperforms newer frontier LLMs on HealthBench",
      "authors": [
        "Praveen Reddy",
        "Charuta Mandke",
        "Suvrankar Datta",
        "Sarah Khan",
        "Siddharth Reddy Anthireddy",
        "Shitij Arora",
        "Vishal Singh"
      ],
      "abstract": "General-purpose large language models (LLMs) have recently been reported to match or exceed specialized clinical AI tools on medical benchmarks, but such comparisons draw on a narrow set of systems and on benchmarks developed largely in high-income settings. We evaluate VITA, a retrieval-augmented generation (RAG) system purpose-built for contextual knowledge retrieval in India and other low- and middle-income (LMIC) settings. VITA retrieves from a curated corpus of disease-specific guidelines, India-specific antimicrobial resistance data, national formulary constraints, and resource-limited care protocols; its architecture and corpus are proprietary, but the benchmark, the physician-written rubrics, and our full response and scoring outputs are public for independent verification. On 4,023 English-language HealthBench questions (80.5% of the benchmark), scored with a GPT-4.1 judge, VITA ranked first with 51.9% of possible rubric points, ahead of GPT-5.4 (46.1%), o4-mini (44.3%), Gemini 3.1 Pro (42.6%), and Claude Sonnet 4.6 (37.3%), and scored highest on 45.4% of questions. To test robustness to newer models and judge lineage, a 500-question subset was re-run against current-generation models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Pro, Grok 4.3) and graded by a neutral open-weight judge (DeepSeek-V4-Pro) sharing no lineage with any system tested. Here the gap narrowed to parity: VITA and GPT-5.5 were statistically indistinguishable on mean per-question score, while VITA led on points-weighted score and won the most questions. VITA's advantages in accuracy and completeness persisted under the neutral judge; its communication scores were lower. These results indicate that a purpose-built clinical RAG system remains competitive with frontier LLMs on an open benchmark, consistent with corpus specificity as a design variable that improves grounding at some cost to communication polish.",
      "published": "2026-08-12T14:55:46Z",
      "abstract_url": "http://arxiv.org/abs/2608.12138v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12138v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.HC",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "Adversarial Resilience of Poisson-Process Submodular Maximization over Matroids: From Robust Offline Optimization to Full-Bandit Learning",
      "authors": [
        "Vaneet Aggarwal"
      ],
      "abstract": "We study nonnegative submodular maximization subject to a general matroid when the offline algorithm is given an arbitrary controlled value oracle. Our main result is an adversarial resilience theorem for the Spiteful Greedy Swap Poisson Process (SGS-Poisson): without modifying its Poisson intensity, single-element exchange rule, or spiteful drop step, the algorithm retains limiting approximation factors $1/e$ for non-monotone objectives and $1-1/e$ for monotone objectives. More precisely, under every controlled oracle $\\widehat f$ satisfying $|\\widehat f(S)-f(S)|\\le ξ$ for every set $S$, our implementation returns a feasible set with expected value at least $(1/e-\\varepsilon)\\OPT-O(kξ)$ and $(1-1/e-\\varepsilon)\\OPT-O(kξ)$, respectively, using $\\widetilde O(nk^2\\varepsilon^{-2})$ oracle calls. As a consequence, the offline-to-online reduction yields full-bandit CMAB algorithms for general matroid-constrained submodular rewards with exact limiting approximation-regret factors $1/e$ and $1-1/e$ and $\\widetilde O(n^{1/5}k^{4/5}T^{4/5})$ regret.",
      "published": "2026-08-12T14:54:15Z",
      "abstract_url": "http://arxiv.org/abs/2608.12134v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12134v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CC",
        "math.OC"
      ]
    },
    {
      "title": "Confidence Calibration of Deep Learning Systems",
      "authors": [
        "Coby Penso"
      ],
      "abstract": "In high-stakes applications, reliable confidence estimates are as important as the predictions themselves. Confidence calibration ensures that predicted probabilities reflect the likelihood of correctness, making it essential for safe deployment of deep learning models. However, existing methods typically assume access to clean validation data, which is often unrealistic due to label noise and domain shifts. This thesis develops methods for improving calibration under these conditions. First, we address calibration under label noise. Standard methods can produce misleading confidence estimates when labels are unreliable. We propose a framework that uses an estimated noise model to reconstruct noise-free confidence estimates by modeling the relationship between noisy and clean label distributions. We extend this approach to Conformal Prediction (CP), which provides set-valued predictions with guaranteed coverage. Our noise-aware CP method estimates clean conformity scores despite label noise, enabling reliable uncertainty quantification. Next, we study calibration in unsupervised domain adaptation, where a model trained on a labeled source domain is adapted to an unlabeled target domain. Since labeled target data are unavailable, we estimate target-domain accuracy from source performance and domain discrepancies, enabling calibration without target labels. We also consider privacy-preserving settings in which user labels and model outputs must remain protected. We propose a locally differentially private conformal prediction framework that provides valid uncertainty quantification while maintaining privacy guarantees and balancing privacy, computational feasibility, and prediction reliability. Our results bridge calibration theory and practical deployment in safety-critical applications, contributing to reliable, privacy-preserving, and noise-resilient neural network predictions.",
      "published": "2026-08-12T14:23:14Z",
      "abstract_url": "http://arxiv.org/abs/2608.12100v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12100v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Faithful, Sufficient and Understandable: Rethinking Graph Counterfactual Explanations via Discrete Diffusion Inversion",
      "authors": [
        "David Bechtoldt",
        "Sidney Bender"
      ],
      "abstract": "Graph Neural Networks (GNNs) achieve strong predictive performance on graph-structured data across domains such as chemistry, biology, and network analysis, yet they provide no intrinsic explanation of their predictions. This limits their adoption in high-stakes and safety-critical settings. Counterfactual explanations address this by revealing the minimal structural modifications that would change a model's prediction. On graphs, however, such a modification is hard to produce. The search space is discrete and combinatorial, and a valid answer must respect categorical node and edge types together with domain rules such as chemical valency in the case of molecular graphs. Existing explainers give up one of two things. Either edits are not held on the data manifold, or the search does not span the full edit space. We propose Graph Diffusion Counterfactual Explanation via Inversion (GDCE-I), which gives up neither. A discrete denoising diffusion model with a novel discrete inversion scheme enables distribution-aware edits leveraging the whole domain edit space. We further address the incomplete and inconsistent evaluation of graph counterfactuals by deriving a framework of explanation desiderata and applying it to every method under one shared protocol. Across four benchmarks, GDCE-I outperforms related work by a large margin on the defined framework. For the molecular domain, we further qualitatively show that GDCE-I attains interpretable in-distribution solutions.",
      "published": "2026-08-12T14:04:49Z",
      "abstract_url": "http://arxiv.org/abs/2608.12083v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12083v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Better Slots, Better Worlds: Representation Quality & Robustness in Object-Centric World Models",
      "authors": [
        "Shukrullo Nazirjonov",
        "Sai Prasanna",
        "Anna Manasyan",
        "Georg Martius"
      ],
      "abstract": "Learning world models from offline trajectories enables agents to accomplish different tasks through planning. Object-centric (OC) representations, which decompose a scene into a set of slots that bind to its objects, have been proposed as an inductive bias for world models that are more sample-efficient and generalize better. Yet prior object-centric world models (OCWMs) take the slot encoder as given and evaluate only in-distribution, leaving open whether the object-centric bias actually delivers for planning and what within the OCWM drives it. We conduct a controlled study of OCWMs for visual model-predictive control along two axes: object-centric representation quality and generalization under distribution shift relative to scene-centric models. We find that (i) planning success correlates positively with unsupervised slot-quality metrics (FG-ARI, mBO), though the gains saturate at high slot quality; (ii) with well-bound slots, the auxiliary proprioception inputs and masking inductive bias that prior methods relied on become unnecessary; and (iii) under unseen distribution shifts, the OCWM with well-bound slots is more robust overall than the end-to-end trained scene-centric LeWM, while DINO-WM, built on similar frozen pretrained features, remains competitive -- pointing to pretrained features as a key contributor to robustness.",
      "published": "2026-08-12T14:02:36Z",
      "abstract_url": "http://arxiv.org/abs/2608.12078v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12078v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Mechanist: AI as a Scientific Instrument for Discovering the Mechanisms of Intelligence",
      "authors": [
        "Mengru Wang",
        "Junfeng Fang",
        "Shuofei Qiao",
        "Zhenqian Xu",
        "Haoming Xu",
        "Haoxiong Wang",
        "Shumin Deng",
        "Linyi Yang",
        "Zhixiang Cui",
        "Xin Xu",
        "Yunzhi Yao",
        "Buqiang Xu",
        "Fei Shen",
        "Haozhe Luo",
        "Yunxiang Wei",
        "Ningyu Zhang",
        "Julian McAuley",
        "Tat Seng Chua",
        "Huajun Chen"
      ],
      "abstract": "AI models have achieved remarkable success across diverse domains, yet the mechanisms underlying their capabilities and the risks they may pose remain poorly understood. As AI development becomes faster and increasingly automated, mechanistic exploration remains largely manual, widening the gap between what models can do and our ability to understand and control them. To bridge this gap, we introduce Mechanist, an agentic system that uses AI as a scientific instrument for the autonomous discovery of mechanisms underlying AI intelligence. To support autonomous mechanistic discovery, we construct an interpretability-focused knowledge graph of approximately 13,000 papers and integrate it with a multidisciplinary database of 43 million papers spanning 26 fields. We further curate a library of 32 foundational methods for mechanism analysis, causal intervention, and validation. Compared with Claude Code and existing AI-scientist systems, Mechanist generates more valuable mechanism hypotheses and executes experiments more reliably. Mechanist also demonstrates a progression from discovering model behaviors to explaining and controlling AI models. Specifically, Mechanist first uncovers a counterintuitive safety risk in scientific laboratories, showing that unsafe traits can transfer across modalities through apparently safe training data. Mechanist then develops a mechanism theory of belief, revealing how models represent world knowledge, form beliefs, infer the beliefs of others, and how these mechanisms emerge during pretraining. Finally, Mechanist translates these mechanistic insights into practical interventions that improve model performance across diverse scenarios and steer scientific foundation models toward generating DNA sequences with specified properties.",
      "published": "2026-08-12T13:19:42Z",
      "abstract_url": "http://arxiv.org/abs/2608.12036v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12036v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.HC",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Uncertainty-Aware Probabilistic Constrained Clustering from Entangled Pairwise Supervision",
      "authors": [
        "Shaojie Zhang",
        "Ke Chen"
      ],
      "abstract": "Pairwise constrained clustering typically relies on hard must-link/cannot-link labels, whereas realistic pairwise supervision may be real-valued and entangle intrinsic ambiguity, expert judgment, and stochastic corruption. Existing deep constrained clustering (DCC) methods mainly target hard, expert-agnostic constraints, treating soft labels mostly numerically rather than semantically. We formalize this setting as uncertainty-aware probabilistic constrained clustering (UPCC), defining a canonical aleatoric target through a heterogeneous observation process and analyzing its conditional identifiability. We introduce ProbPair, an angular pairwise objective for probabilistic relations, and build ECI-PP, an estimator--corrector--integrator framework that refines imperfect supervision via belief estimation, correction, and reliability-aware integration. Across challenging probabilistic supervision settings, experiments on diverse benchmarks show that ECI-PP outperforms state-of-the-art DCC methods and remains robust with a shared default configuration.",
      "published": "2026-08-12T13:07:19Z",
      "abstract_url": "http://arxiv.org/abs/2608.12027v1",
      "pdf_url": "https://arxiv.org/pdf/2608.12027v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
