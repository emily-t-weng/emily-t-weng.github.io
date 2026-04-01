const PAPERS_DATA = {
  "last_updated": "2026-04-01 03:30:59 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Aligned, Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought?",
      "authors": [
        "Max Kaufmann",
        "David Lindner",
        "Roland S. Zimmermann",
        "and Rohin Shah"
      ],
      "abstract": "Chain-of-Thought (CoT) monitoring, in which automated systems monitor the CoT of an LLM, is a promising approach for effectively overseeing AI systems. However, the extent to which a model's CoT helps us oversee the model - the monitorability of the CoT - can be affected by training, for instance by the model learning to hide important features of its reasoning. We propose and empirically validate a conceptual framework for predicting when and why this occurs. We model LLM post-training as an RL environment where the reward decomposes into two terms: one term depending on final outputs and another term depending on the CoT. Our framework allows us to classify these two terms as \"aligned\", \"orthogonal\", or \"in-conflict\" before training. We predict that training with in-conflict terms will reduce monitorability, orthogonal terms will not affect it, and aligned terms will improve it. To validate our framework, we use it to classify a set of RL environments, train LLMs within those environments, and evaluate how training affects CoT monitorability. We find that (1) training with \"in-conflict\" reward terms reduces CoT monitorability and (2) optimizing in-conflict reward terms is difficult.",
      "published": "2026-03-31T17:38:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.30036v1",
      "pdf_url": "https://arxiv.org/pdf/2603.30036v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Tucker Attention: A generalization of approximate attention mechanisms",
      "authors": [
        "Timon Klein",
        "Jonas Kusch",
        "Sebastian Sager",
        "Stefan Schnake",
        "Steffen Schotthöfer"
      ],
      "abstract": "The pursuit of reducing the memory footprint of the self-attention mechanism in multi-headed self attention (MHA) spawned a rich portfolio of methods, e.g., group-query attention (GQA) and multi-head latent attention (MLA). The methods leverage specialized low-rank factorizations across embedding dimensions or attention heads. From the point of view of classical low-rank approximation, these methods are unconventional and raise questions of which objects they really approximate and how to interpret the low-rank behavior of the resulting representations. To answer these questions, this work proposes a generalized view on the weight objects in the self-attention layer and a factorization strategy, which allows us to construct a parameter efficient scheme, called Tucker Attention. Tucker Attention requires an order of magnitude fewer parameters for comparable validation metrics, compared to GQA and MLA, as evaluated in LLM and ViT test cases. Additionally, Tucker Attention~encompasses GQA, MLA, MHA as special cases and is fully compatible with flash-attention and rotary position embeddings (RoPE). This generalization strategy yields insights of the actual ranks achieved by MHA, GQA, and MLA, and further enables simplifications for MLA.",
      "published": "2026-03-31T17:30:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.30033v1",
      "pdf_url": "https://arxiv.org/pdf/2603.30033v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "The Triadic Cognitive Architecture: Bounding Autonomous Action via Spatio-Temporal and Epistemic Friction",
      "authors": [
        "Davide Di Gioia"
      ],
      "abstract": "Current autonomous AI agents, driven primarily by Large Language Models (LLMs), operate in a state of cognitive weightlessness: they process information without an intrinsic sense of network topology, temporal pacing, or epistemic limits. Consequently, heuristic agentic loops (e.g., ReAct) can exhibit failure modes in interactive environments, including excessive tool use under congestion, prolonged deliberation under time decay, and brittle behavior under ambiguous evidence. In this paper, we propose the Triadic Cognitive Architecture (TCA), a unified mathematical framework that grounds machine reasoning in continuous-time physics. By synthesizing nonlinear filtering theory, Riemannian routing geometry, and optimal control, we formally define the concept of Cognitive Friction. We map the agent's deliberation process to a coupled stochastic control problem where information acquisition is path-dependent and physically constrained. Rather than relying on arbitrary heuristic stop-tokens, the TCA uses an HJB-motivated stopping boundary and instantiates a rollout-based approximation of belief-dependent value-of-information with a net-utility halting condition. Through empirical validation in a simulated Emergency Medical Diagnostic Grid (EMDG), we demonstrate that while greedy baselines over-deliberate under latency and congestion costs, the triadic policy reduces time-to-action while improving patient viability without degrading diagnostic accuracy in this environment.",
      "published": "2026-03-31T17:30:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.30031v1",
      "pdf_url": "https://arxiv.org/pdf/2603.30031v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models",
      "authors": [
        "Md Saad",
        "Sajjad Hussain",
        "Mohd Suhaib"
      ],
      "abstract": "This paper introduces a new hybrid framework that combines Reinforcement Learning (RL) and Large Language Models (LLMs) to improve robotic manipulation tasks. By utilizing RL for accurate low-level control and LLMs for high level task planning and understanding of natural language, the proposed framework effectively connects low-level execution with high-level reasoning in robotic systems. This integration allows robots to understand and carry out complex, human-like instructions while adapting to changing environments in real time. The framework is tested in a PyBullet-based simulation environment using the Franka Emika Panda robotic arm, with various manipulation scenarios as benchmarks. The results show a 33.5% decrease in task completion time and enhancements of 18.1% and 36.4% in accuracy and adaptability, respectively, when compared to systems that use only RL. These results underscore the potential of LLM-enhanced robotic systems for practical applications, making them more efficient, adaptable, and capable of interacting with humans. Future research will aim to explore sim-to-real transfer, scalability, and multi-robot systems to further broaden the framework's applicability.",
      "published": "2026-03-31T17:19:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.30022v1",
      "pdf_url": "https://arxiv.org/pdf/2603.30022v1",
      "categories": [
        "cs.RO",
        "cs.AI"
      ]
    },
    {
      "title": "Architecting Secure AI Agents: Perspectives on System-Level Defenses Against Indirect Prompt Injection Attacks",
      "authors": [
        "Chong Xiang",
        "Drew Zagieboylo",
        "Shaona Ghosh",
        "Sanjay Kariyappa",
        "Kai Greshake",
        "Hanshen Xiao",
        "Chaowei Xiao",
        "G. Edward Suh"
      ],
      "abstract": "AI agents, predominantly powered by large language models (LLMs), are vulnerable to indirect prompt injection, in which malicious instructions embedded in untrusted data can trigger dangerous agent actions. This position paper discusses our vision for system-level defenses against indirect prompt injection attacks. We articulate three positions: (1) dynamic replanning and security policy updates are often necessary for dynamic tasks and realistic environments; (2) certain context-dependent security decisions would still require LLMs (or other learned models), but should only be made within system designs that strictly constrain what the model can observe and decide; (3) in inherently ambiguous cases, personalization and human interaction should be treated as core design considerations. In addition to our main positions, we discuss limitations of existing benchmarks that can create a false sense of utility and security. We also highlight the value of system-level defenses, which serve as the skeleton of agentic systems by structuring and controlling agent behaviors, integrating rule-based and model-based security checks, and enabling more targeted research on model robustness and human interaction.",
      "published": "2026-03-31T17:15:46Z",
      "abstract_url": "http://arxiv.org/abs/2603.30016v1",
      "pdf_url": "https://arxiv.org/pdf/2603.30016v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Phyelds: A Pythonic Framework for Aggregate Computing",
      "authors": [
        "Gianluca Aguzzi",
        "Davide Domini",
        "Nicolas Farabegoli",
        "Mirko Viroli"
      ],
      "abstract": "Aggregate programming is a field-based coordination paradigm with over a decade of exploration and successful applications across domains including sensor networks, robotics, and IoT, with implementations in various programming languages, such as Protelis, ScaFi (Scala), and FCPP (C++). A recent research direction integrates machine learning with aggregate computing, aiming to support large-scale distributed learning and provide new abstractions for implementing learning algorithms. However, existing implementations do not target data science practitioners, who predominantly work in Python--the de facto language for data science and machine learning, with a rich and mature ecosystem. Python also offers advantages for other use cases, such as education and robotics (e.g., via ROS). To address this gap, we present Phyelds, a Python library for aggregate programming. Phyelds offers a fully featured yet lightweight implementation of the field calculus model of computation, featuring a Pythonic API and an architecture designed for seamless integration with Python's machine learning ecosystem. We describe the design and implementation of Phyelds and illustrate its versatility across domains, from well-known aggregate computing patterns to federated learning coordination and integration with a widely used multi-agent reinforcement learning simulator.",
      "published": "2026-03-31T16:57:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.29999v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29999v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.PL"
      ]
    },
    {
      "title": "Quantifying Cross-Modal Interactions in Multimodal Glioma Survival Prediction via InterSHAP: Evidence for Additive Signal Integration",
      "authors": [
        "Iain Swift",
        "JingHua Ye",
        "Ruairi O'Reilly"
      ],
      "abstract": "Multimodal deep learning for cancer prognosis is commonly assumed to benefit from synergistic cross-modal interactions, yet this assumption has not been directly tested in survival prediction settings. This work adapts InterSHAP, a Shapley interaction index-based metric, from classification to Cox proportional hazards models and applies it to quantify cross-modal interactions in glioma survival prediction. Using TCGA-GBM and TCGA-LGG data (n=575), we evaluate four fusion architectures combining whole-slide image (WSI) and RNA-seq features. Our central finding is an inverse relationship between predictive performance and measured interaction: architectures achieving superior discrimination (C-index 0.64$\\to$0.82) exhibit equivalent or lower cross-modal interaction (4.8\\%$\\to$3.0\\%). Variance decomposition reveals stable additive contributions across all architectures (WSI${\\approx}$40\\%, RNA${\\approx}$55\\%, Interaction${\\approx}$4\\%), indicating that performance gains arise from complementary signal aggregation rather than learned synergy. These findings provide a practical model auditing tool for comparing fusion strategies, reframe the role of architectural complexity in multimodal fusion, and have implications for privacy-preserving federated deployment.",
      "published": "2026-03-31T16:39:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.29977v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29977v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.QM"
      ]
    },
    {
      "title": "Bethe Ansatz with a Large Language Model",
      "authors": [
        "Balázs Pozsgay",
        "István Vona"
      ],
      "abstract": "We explore the capability of a Large Language Model (LLM) to perform specific computations in mathematical physics: the task is to compute the coordinate Bethe Ansatz solution of selected integrable spin chain models. We select three integrable Hamiltonians for which the solutions were unpublished; two of the Hamiltonians are actually new. We observed that the LLM semi-autonomously solved the task in all cases, with a few mistakes along the way. These were corrected after the human researchers spotted them. The results of the LLM were checked against exact diagonalization (performed by separate programs), and the derivations were also checked by the authors. The Bethe Ansatz solutions are interesting in themselves. Our second model manifestly breaks left-right invariance, but it is PT-symmetric, therefore its solution could be interesting for applications in Generalized Hydrodynamics. And our third model is solved by a special form of the nested Bethe Ansatz, where the model is interacting, but the nesting level has a free fermionic structure lacking $U(1)$-invariance. This structure appears to be unique and it was found by the LLM. We used ChatGPT 5.2 Pro and 5.4 Pro by OpenAI.",
      "published": "2026-03-31T16:06:19Z",
      "abstract_url": "http://arxiv.org/abs/2603.29932v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29932v1",
      "categories": [
        "cond-mat.stat-mech",
        "cs.AI",
        "hep-th"
      ]
    },
    {
      "title": "End-to-End Image Compression with Segmentation Guided Dual Coding for Wind Turbines",
      "authors": [
        "Raül Pérez-Gonzalo",
        "Andreas Espersen",
        "Søren Forchhammer",
        "Antonio Agudo"
      ],
      "abstract": "Transferring large volumes of high-resolution images during wind turbine inspections introduces a bottleneck in assessing and detecting severe defects. Efficient coding must preserve high fidelity in blade regions while aggressively compressing the background. In this work, we propose an end-to-end deep learning framework that jointly performs segmentation and dual-mode (lossy and lossless) compression. The segmentation module accurately identifies the blade region, after which our region-of-interest (ROI) compressor encodes it at superior quality compared to the rest of the image. Unlike conventional ROI schemes that merely allocate more bits to salient areas, our framework integrates: (i) a robust segmentation network (BU-Netv2+P) with a CRF-regularized loss for precise blade localization, (ii) a hyperprior-based autoencoder optimized for lossy compression, and (iii) an extended bits-back coder with hierarchical models for fully lossless blade reconstruction. Furthermore, our ROI framework removes the sequential dependency in bits-back coding by reusing background-coded bits, enabling parallelized and efficient dual-mode compression. To the best of our knowledge, this is the first fully integrated learning-based ROI codec combining segmentation, lossy, and lossless compression, ensuring that subsequent defect detection is not compromised. Experiments on a large-scale wind turbine dataset demonstrate superior compression performance and efficiency, offering a practical solution for automated inspections.",
      "published": "2026-03-31T16:01:23Z",
      "abstract_url": "http://arxiv.org/abs/2603.29927v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29927v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence",
      "authors": [
        "Georgii Mikriukov",
        "Grégoire Montavon",
        "Marina M. -C. Höhne"
      ],
      "abstract": "Post-hoc explanation methods are widely used to interpret black-box predictions, but their generation is often computationally expensive and their reliability is not guaranteed. We propose epistemic uncertainty as a low-cost proxy for explanation reliability: high epistemic uncertainty identifies regions where the decision boundary is poorly defined and where explanations become unstable and unfaithful. This insight enables two complementary use cases: `improving worst-case explanations' (routing samples to cheap or expensive XAI methods based on expected explanation reliability), and `recalling high-quality explanations' (deferring explanation generation for uncertain samples under constrained budget). Across four tabular datasets, five diverse architectures, and four XAI methods, we observe a strong negative correlation between epistemic uncertainty and explanation stability. Further analysis shows that epistemic uncertainty distinguishes not only stable from unstable explanations, but also faithful from unfaithful ones. Experiments on image classification confirm that our findings generalize beyond tabular data.",
      "published": "2026-03-31T15:56:10Z",
      "abstract_url": "http://arxiv.org/abs/2603.29915v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29915v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SISA: A Scale-In Systolic Array for GEMM Acceleration",
      "authors": [
        "Luigi Altamura",
        "Alessio Cicero",
        "Mateo Vázquez Maceiras",
        "Mohammad Ali Maleki",
        "Pedro Trancoso"
      ],
      "abstract": "The currently dominant AI/ML workloads, such as Large Language Models (LLMs), rely on the efficient execution of General Matrix-Matrix Multiplication (GEMM) operations. Thus, most systems are equipped with dedicated matrix hardware accelerators based on square Systolic Arrays (SAs) of Processing Elements (PEs). While this organization was effective for traditional Deep Neural Networks (DNNs), LLMs introduce input-dependent and highly skewed matrices, leading to underutilized SA resources. To address this challenge, we propose SISA (Scale-In Systolic Array), a novel SA architecture that partitions the traditional square array into horizontal rectangular slabs. With minimal overhead, SISA exposes parallelism through independently scheduled slabs for efficient execution of small or skewed matrix shapes, while retaining full-array operation for large GEMMs. SISA achieves up to 8.52x speedup and 93% energy-delay-product (EDP) reduction for representative LLMs compared to a state-of-the-art monolithic SA with the same number of PEs.",
      "published": "2026-03-31T15:55:31Z",
      "abstract_url": "http://arxiv.org/abs/2603.29913v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29913v1",
      "categories": [
        "cs.AR",
        "cs.AI"
      ]
    },
    {
      "title": "C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving",
      "authors": [
        "Zhihong Cui",
        "Haoran Tang",
        "Tianyi Li",
        "Yushuai Li",
        "Peiyuan Guan",
        "Amir Taherkordi",
        "Tor Skeie"
      ],
      "abstract": "Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived commonsense with a trust mechanism to guide trajectory planning. C-TRAIL operates through a closed-loop Recall, Plan, and Update cycle: the Recall module queries an LLM for semantic relations and quantifies their reliability via a dual-trust mechanism; the Plan module injects trust-weighted commonsense into Monte Carlo Tree Search (MCTS) through a Dirichlet trust policy; and the Update module adaptively refines trust scores and policy parameters from environmental feedback. Experiments on four simulated scenarios in Highway-env and two real-world levelXData datasets (highD, rounD) show that C-TRAIL consistently outperforms state-of-the-art baselines, reducing ADE by 40.2%, FDE by 51.7%, and improving SR by 16.9 percentage points on average. The source code is available at https://github.com/ZhihongCui/CTRAIL.",
      "published": "2026-03-31T15:53:29Z",
      "abstract_url": "http://arxiv.org/abs/2603.29908v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29908v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ATP-Bench: Towards Agentic Tool Planning for MLLM Interleaved Generation",
      "authors": [
        "Yinuo Liu",
        "Zi Qian",
        "Heng Zhou",
        "Jiahao Zhang",
        "Yajie Zhang",
        "Zhihang Li",
        "Mengyu Zhou",
        "Erchao Zhao",
        "Xiaoxi Jiang",
        "Guanjun Jiang"
      ],
      "abstract": "Interleaved text-and-image generation represents a significant frontier for Multimodal Large Language Models (MLLMs), offering a more intuitive way to convey complex information. Current paradigms rely on either image generation or retrieval augmentation, yet they typically treat the two as mutually exclusive paths, failing to unify factuality with creativity. We argue that the next milestone in this field is Agentic Tool Planning, where the model serves as a central controller that autonomously determines when, where, and which tools to invoke to produce interleaved responses for visual-critical queries. To systematically evaluate this paradigm, we introduce ATP-Bench, a novel benchmark comprising 7,702 QA pairs (including 1,592 VQA pairs) across eight categories and 25 visual-critical intents, featuring human-verified queries and ground truths. Furthermore, to evaluate agentic planning independent of end-to-end execution and changing tool backends, we propose a Multi-Agent MLLM-as-a-Judge (MAM) system. MAM evaluates tool-call precision, identifies missed opportunities for tool use, and assesses overall response quality without requiring ground-truth references. Our extensive experiments on 10 state-of-the-art MLLMs reveal that models struggle with coherent interleaved planning and exhibit significant variations in tool-use behavior, highlighting substantial room for improvement and providing actionable guidance for advancing interleaved generation. Dataset and code are available at https://github.com/Qwen-Applications/ATP-Bench.",
      "published": "2026-03-31T15:47:59Z",
      "abstract_url": "http://arxiv.org/abs/2603.29902v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29902v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ShapE-GRPO: Shapley-Enhanced Reward Allocation for Multi-Candidate LLM Training",
      "authors": [
        "Rui Ai",
        "Yu Pan",
        "David Simchi-Levi",
        "Chonghuan Wang"
      ],
      "abstract": "In user-agent interaction scenarios such as recommendation, brainstorming, and code suggestion, Large Language Models (LLMs) often generate sets of candidate recommendations where the objective is to maximize the collective utility of the entire set rather than individual candidates independently. However, existing reinforcement learning post-training paradigms, such as Group Relative Policy Optimization (GRPO), typically assign the same set-level scalar reward to every candidate in the set. This leads to noisy training signals where poor candidates free-ride on the high reward produced by a single strong peer, resulting in suboptimal exploration. To address this, we propose Shapley-Enhanced GRPO (ShapE-GRPO). By leveraging the permutation-invariant nature of set-level utility, we derive a Shapley-enhanced formulation from cooperative game theory to decompose set-level rewards into granular, candidate-specific signals. We show that our formulation preserves the fundamental axioms of the Shapley value while remaining computationally efficient with polynomial-time complexity. Empirically, ShapE-GRPO consistently outperforms standard GRPO across diverse datasets with accelerated convergence during training.",
      "published": "2026-03-31T15:24:15Z",
      "abstract_url": "http://arxiv.org/abs/2603.29871v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29871v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA",
      "authors": [
        "Yi Chen",
        "Yuying Ge",
        "Hui Zhou",
        "Mingyu Ding",
        "Yixiao Ge",
        "Xihui Liu"
      ],
      "abstract": "The development of Vision-Language-Action (VLA) models has been significantly accelerated by pre-trained Vision-Language Models (VLMs). However, most existing end-to-end VLAs treat the VLM primarily as a multimodal encoder, directly mapping vision-language features to low-level actions. This paradigm underutilizes the VLM's potential in high-level decision making and introduces training instability, frequently degrading its rich semantic representations. To address these limitations, we introduce DIAL, a framework bridging high-level decision making and low-level motor execution through a differentiable latent intent bottleneck. Specifically, a VLM-based System-2 performs latent world modeling by synthesizing latent visual foresight within the VLM's native feature space; this foresight explicitly encodes intent and serves as the structural bottleneck. A lightweight System-1 policy then decodes this predicted intent together with the current observation into precise robot actions via latent inverse dynamics. To ensure optimization stability, we employ a two-stage training paradigm: a decoupled warmup phase where System-2 learns to predict latent futures while System-1 learns motor control under ground-truth future guidance within a unified feature space, followed by seamless end-to-end joint optimization. This enables action-aware gradients to refine the VLM backbone in a controlled manner, preserving pre-trained knowledge. Extensive experiments on the RoboCasa GR1 Tabletop benchmark show that DIAL establishes a new state-of-the-art, achieving superior performance with 10x fewer demonstrations than prior methods. Furthermore, by leveraging heterogeneous human demonstrations, DIAL learns physically grounded manipulation priors and exhibits robust zero-shot generalization to unseen objects and novel configurations during real-world deployment on a humanoid robot.",
      "published": "2026-03-31T15:02:27Z",
      "abstract_url": "http://arxiv.org/abs/2603.29844v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29844v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "From Density Matrices to Phase Transitions in Deep Learning: Spectral Early Warnings and Interpretability",
      "authors": [
        "Max Hennick",
        "Guillaume Corlouer"
      ],
      "abstract": "A key problem in the modern study of AI is predicting and understanding emergent capabilities in models during training. Inspired by methods for studying reactions in quantum chemistry, we present the ``2-datapoint reduced density matrix\". We show that this object provides a computationally efficient, unified observable of phase transitions during training. By tracking the eigenvalue statistics of the 2RDM over a sliding window, we derive two complementary signals: the spectral heat capacity, which we prove provides early warning of second-order phase transitions via critical slowing down, and the participation ratio, which reveals the dimensionality of the underlying reorganization. Remarkably, the top eigenvectors of the 2RDM are directly interpretable making it straightforward to study the nature of the transitions. We validate across four settings distinct settings: deep linear networks, induction head formation, grokking, and emergent misalignment. We then discuss directions for future work using the 2RDM.",
      "published": "2026-03-31T14:34:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.29805v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29805v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Reasoning-Driven Synthetic Data Generation and Evaluation",
      "authors": [
        "Tim R. Davidson",
        "Benoit Seguin",
        "Enrico Bacis",
        "Cesar Ilharco",
        "Hamza Harkous"
      ],
      "abstract": "Although many AI applications of interest require specialized multi-modal models, relevant data to train such models is inherently scarce or inaccessible. Filling these gaps with human annotators is prohibitively expensive, error-prone, and time-consuming, leading model builders to increasingly consider synthetic data as a scalable alternative. However, existing synthetic data generation methods often rely on manual prompts, evolutionary algorithms, or extensive seed data from the target distribution - limiting their scalability, explainability, and control. In this paper, we introduce Simula: a novel reasoning-driven framework for data generation and evaluation. It employs a seedless, agentic approach to generate synthetic datasets at scale, allowing users to define desired dataset characteristics through an explainable and controllable process that enables fine-grained resource allocation. We show the efficacy of our approach on a variety of datasets, rigorously testing both intrinsic and downstream properties. Our work (1) offers guidelines for synthetic data mechanism design, (2) provides insights into generating and evaluating synthetic data at scale, and (3) unlocks new opportunities for developing and deploying AI in domains where data scarcity or privacy concerns are paramount.",
      "published": "2026-03-31T14:26:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.29791v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29791v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "CausalPulse: An Industrial-Grade Neurosymbolic Multi-Agent Copilot for Causal Diagnostics in Smart Manufacturing",
      "authors": [
        "Chathurangi Shyalika",
        "Utkarshani Jaimini",
        "Cory Henson",
        "Amit Sheth"
      ],
      "abstract": "Modern manufacturing environments demand real-time, trustworthy, and interpretable root-cause insights to sustain productivity and quality. Traditional analytics pipelines often treat anomaly detection, causal inference, and root-cause analysis as isolated stages, limiting scalability and explainability. In this work, we present CausalPulse, an industry-grade multi-agent copilot that automates causal diagnostics in smart manufacturing. It unifies anomaly detection, causal discovery, and reasoning through a neurosymbolic architecture built on standardized agentic protocols. CausalPulse is being deployed in a Robert Bosch manufacturing plant, integrating seamlessly with existing monitoring workflows and supporting real-time operation at production scale. Evaluations on both public (Future Factories) and proprietary (Planar Sensor Element) datasets show high reliability, achieving overall success rates of 98.0% and 98.73%. Per-criterion success rates reached 98.75% for planning and tool use, 97.3% for self-reflection, and 99.2% for collaboration. Runtime experiments report end-to-end latency of 50-60s per diagnostic workflow with near-linear scalability (R^2=0.97), confirming real-time readiness. Comparison with existing industrial copilots highlights distinct advantages in modularity, extensibility, and deployment maturity. These results demonstrate how CausalPulse's modular, human-in-the-loop design enables reliable, interpretable, and production-ready automation for next-generation manufacturing.",
      "published": "2026-03-31T13:54:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.29755v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29755v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Spontaneous Functional Differentiation in Large Language Models: A Brain-Like Intelligence Economy",
      "authors": [
        "Junjie Zhang",
        "Zhen Shen",
        "Gang Xiong",
        "Xisong Dong"
      ],
      "abstract": "The evolution of intelligence in artificial systems provides a unique opportunity to identify universal computational principles. Here we show that large language models spontaneously develop synergistic cores where information integration exceeds individual parts remarkably similar to the human brain. Using Integrated Information Decomposition across multiple architectures we find that middle layers exhibit synergistic processing while early and late layers rely on redundancy. This organization is dynamic and emerges as a physical phase transition as task difficulty increases. Crucially ablating synergistic components causes catastrophic performance loss confirming their role as the physical entity of abstract reasoning and bridging artificial and biological intelligence.",
      "published": "2026-03-31T13:36:08Z",
      "abstract_url": "http://arxiv.org/abs/2603.29735v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29735v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding",
      "authors": [
        "Joakim Edin",
        "Andreas Motzfeldt",
        "Simon Flachs",
        "Lars Maaløe"
      ],
      "abstract": "Medical coding translates free-text clinical documentation into standardized codes drawn from classification systems that contain tens of thousands of entries and are updated annually. It is central to billing, clinical research, and quality reporting, yet remains largely manual, slow, and error-prone. Existing automated approaches learn to predict a fixed set of codes from labeled data, thereby preventing adaptation to new codes or different coding systems without retraining on different data. They also provide no explanation for their predictions, limiting trust in safety-critical settings. We introduce Symphony for Medical Coding, a system that approaches the task the way expert human coders do: by reasoning over the clinical narrative with direct access to the coding guidelines. This design allows Symphony to operate across any coding system and to provide span-level evidence linking each predicted code to the text that supports it. We evaluate on two public benchmarks and three real-world datasets spanning inpatient, outpatient, emergency, and subspecialty settings across the United States and the United Kingdom. Symphony achieves state-of-the-art results across all settings, establishing itself as a flexible, deployment-ready foundation for automated clinical coding.",
      "published": "2026-03-31T13:11:07Z",
      "abstract_url": "http://arxiv.org/abs/2603.29709v1",
      "pdf_url": "https://arxiv.org/pdf/2603.29709v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
