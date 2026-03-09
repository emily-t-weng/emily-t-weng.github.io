const PAPERS_DATA = {
  "last_updated": "2026-03-09 02:49:31 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "BEVLM: Distilling Semantic Knowledge from LLMs into Bird's-Eye View Representations",
      "authors": [
        "Thomas Monninger",
        "Shaoyuan Xie",
        "Qi Alfred Chen",
        "Sihao Ding"
      ],
      "abstract": "The integration of Large Language Models (LLMs) into autonomous driving has attracted growing interest for their strong reasoning and semantic understanding abilities, which are essential for handling complex decision-making and long-tail scenarios. However, existing methods typically feed LLMs with tokens from multi-view and multi-frame images independently, leading to redundant computation and limited spatial consistency. This separation in visual processing hinders accurate 3D spatial reasoning and fails to maintain geometric coherence across views. On the other hand, Bird's-Eye View (BEV) representations learned from geometrically annotated tasks (e.g., object detection) provide spatial structure but lack the semantic richness of foundation vision encoders. To bridge this gap, we propose BEVLM, a framework that connects a spatially consistent and semantically distilled BEV representation with LLMs. Through extensive experiments, we show that BEVLM enables LLMs to reason more effectively in cross-view driving scenes, improving accuracy by 46%, by leveraging BEV features as unified inputs. Furthermore, by distilling semantic knowledge from LLMs into BEV representations, BEVLM significantly improves closed-loop end-to-end driving performance by 29% in safety-critical scenarios.",
      "published": "2026-03-06T18:59:55Z",
      "abstract_url": "http://arxiv.org/abs/2603.06576v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06576v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Boosting deep Reinforcement Learning using pretraining with Logical Options",
      "authors": [
        "Zihan Ye",
        "Phil Chau",
        "Raban Emunds",
        "Jannis Blüml",
        "Cedric Derstroff",
        "Quentin Delfosse",
        "Oleg Arenz",
        "Kristian Kersting"
      ],
      "abstract": "Deep reinforcement learning agents are often misaligned, as they over-exploit early reward signals. Recently, several symbolic approaches have addressed these challenges by encoding sparse objectives along with aligned plans. However, purely symbolic architectures are complex to scale and difficult to apply to continuous settings. Hence, we propose a hybrid approach, inspired by humans' ability to acquire new skills. We use a two-stage framework that injects symbolic structure into neural-based reinforcement learning agents without sacrificing the expressivity of deep policies. Our method, called Hybrid Hierarchical RL (H^2RL), introduces a logical option-based pretraining strategy to steer the learning policy away from short-term reward loops and toward goal-directed behavior while allowing the final policy to be refined via standard environment interaction. Empirically, we show that this approach consistently improves long-horizon decision-making and yields agents that outperform strong neural, symbolic, and neuro-symbolic baselines.",
      "published": "2026-03-06T18:55:15Z",
      "abstract_url": "http://arxiv.org/abs/2603.06565v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06565v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Artificial Intelligence for Detecting Fetal Orofacial Clefts and Advancing Medical Education",
      "authors": [
        "Yuanji Zhang",
        "Yuhao Huang",
        "Haoran Dou",
        "Xiliang Zhu",
        "Chen Ling",
        "Zhong Yang",
        "Lianying Liang",
        "Jiuping Li",
        "Siying Liang",
        "Rui Li",
        "Yan Cao",
        "Yuhan Zhang",
        "Jiewei Lai",
        "Yongsong Zhou",
        "Hongyu Zheng",
        "Xinru Gao",
        "Cheng Yu",
        "Liling Shi",
        "Mengqin Yuan",
        "Honglong Li",
        "Xiaoqiong Huang",
        "Chaoyu Chen",
        "Jialin Zhang",
        "Wenxiong Pan",
        "Alejandro F. Frangi",
        "Guangzhi He",
        "Xin Yang",
        "Yi Xiong",
        "Linliang Yin",
        "Xuedong Deng",
        "Dong Ni"
      ],
      "abstract": "Orofacial clefts are among the most common congenital craniofacial abnormalities, yet accurate prenatal detection remains challenging due to the scarcity of experienced specialists and the relative rarity of the condition. Early and reliable diagnosis is essential to enable timely clinical intervention and reduce associated morbidity. Here we show that an artificial intelligence system, trained on over 45,139 ultrasound images from 9,215 fetuses across 22 hospitals, can diagnose fetal orofacial clefts with sensitivity and specificity exceeding 93% and 95% respectively, matching the performance of senior radiologists and substantially outperforming junior radiologists. When used as a medical copilot, the system raises junior radiologists' sensitivity by more than 6%. Beyond direct diagnostic assistance, the system also accelerates the development of clinical expertise. A pilot study involving 24 radiologists and trainees demonstrated that the model can improve the expertise development for rare conditions. This dual-purpose approach offers a scalable solution for improving both diagnostic accuracy and specialist training in settings where experienced radiologists are scarce.",
      "published": "2026-03-06T18:06:20Z",
      "abstract_url": "http://arxiv.org/abs/2603.06522v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06522v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics",
      "authors": [
        "Kartik Sharma",
        "Rakshit S. Trivedi"
      ],
      "abstract": "Activation steering methods enable inference-time control of large language model (LLM) behavior without retraining, but current approaches face a fundamental trade-off: sample-efficient methods suboptimally capture steering signals from labeled examples, while methods that better extract these signals require hundreds to thousands of examples. We introduce COLD-Steer, a training-free framework that steers LLM activations by approximating the representational changes that would result from gradient descent on in-context examples. Our key insight is that the effect of fine-tuning on a small set of examples can be efficiently approximated at inference time without actual parameter updates. We formalize this through two complementary approaches: (i) a unit kernel approximation method that updates the activations directly using gradients with respect to them, normalized across examples, and (ii) a finite-difference approximation requiring only two forward passes regardless of example count. Experiments across a variety of steering tasks and benchmarks demonstrate that COLD-Steer achieves upto 95% steering effectiveness while using 50 times fewer samples compared to the best baseline. COLD-Steer facilitates accommodating diverse perspectives without extensive demonstration data, which we validate through our experiments on pluralistic alignment tasks. Our framework opens new possibilities for adaptive, context-aware model control that can flexibly address varying loss-driven human preferences through principled approximation of learning dynamics rather than specialized training procedures.",
      "published": "2026-03-06T17:27:27Z",
      "abstract_url": "http://arxiv.org/abs/2603.06495v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06495v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "NOBLE: Accelerating Transformers with Nonlinear Low-Rank Branches",
      "authors": [
        "Ethan Smith"
      ],
      "abstract": "We introduce NOBLE (Nonlinear lOw-rank Branch for Linear Enhancement), an architectural augmentation that adds nonlinear low-rank branches to transformer linear layers. Unlike LoRA and other parameter-efficient fine-tuning (PEFT) methods, NOBLE is designed for pretraining from scratch. The branch is a permanent part of the architecture as opposed to an adapter for finetuning on top of frozen weights. The branch computes σ(xWdown)Wup where σ is a learnable nonlinearity. We evaluate several activation functions and find that CosNet, a two-layer cosine nonlinearity with learnable frequency and phase with a linear projection in between them in the bottleneck space, performs best. NOBLE achieves substantial improvements with minimal overhead: up to 1.47x step speedup to reach baseline eval loss (up to 32% fewer training steps), with as low as 4% additional parameters and 7% step time overhead, resulting in up to 1.22x net wallclock speedup. Experiments on LLMs (250M and 1.5B parameters), BERT, VQGAN, and ViT consistently show improved training efficiency. We identify one caveat: Mixup/CutMix augmentation interferes with NOBLE's benefits in Imagenet classification along with other stochastic augmentations, but when disabled, ViT also improves. This discrepancy is possibly explained by regularization techniques that encourage smoother fits to the target function while NOBLE may specialize more in sharper aspects of the target function.",
      "published": "2026-03-06T17:22:04Z",
      "abstract_url": "http://arxiv.org/abs/2603.06492v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06492v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.NE"
      ]
    },
    {
      "title": "PONTE: Personalized Orchestration for Natural Language Trustworthy Explanations",
      "authors": [
        "Vittoria Vineis",
        "Matteo Silvestri",
        "Lorenzo Antonelli",
        "Filippo Betello",
        "Gabriele Tolomei"
      ],
      "abstract": "Explainable Artificial Intelligence (XAI) seeks to enhance the transparency and accountability of machine learning systems, yet most methods follow a one-size-fits-all paradigm that neglects user differences in expertise, goals, and cognitive needs. Although Large Language Models can translate technical explanations into natural language, they introduce challenges related to faithfulness and hallucinations. To address these challenges, we present PONTE (Personalized Orchestration for Natural language Trustworthy Explanations), a human-in-the-loop framework for adaptive and reliable XAI narratives. PONTE models personalization as a closed-loop validation and adaptation process rather than prompt engineering. It combines: (i) a low-dimensional preference model capturing stylistic requirements; (ii) a preference-conditioned generator grounded in structured XAI artifacts; and (iii) verification modules enforcing numerical faithfulness, informational completeness, and stylistic alignment, optionally supported by retrieval-grounded argumentation. User feedback iteratively updates the preference state, enabling quick personalization. Automatic and human evaluations across healthcare and finance domains show that the verification-refinement loop substantially improves completeness and stylistic alignment over validation-free generation. Human studies further confirm strong agreement between intended preference vectors and perceived style, robustness to generation stochasticity, and consistently positive quality assessments.",
      "published": "2026-03-06T17:12:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.06485v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06485v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Abductive Reasoning with Syllogistic Forms in Large Language Models",
      "authors": [
        "Hirohiko Abe",
        "Risako Ando",
        "Takanobu Morishita Kentaro Ozeki",
        "Koji Mineshima",
        "Mitsuhiro Okada"
      ],
      "abstract": "Research in AI using Large-Language Models (LLMs) is rapidly evolving, and the comparison of their performance with human reasoning has become a key concern. Prior studies have indicated that LLMs and humans share similar biases, such as dismissing logically valid inferences that contradict common beliefs. However, criticizing LLMs for these biases might be unfair, considering our reasoning not only involves formal deduction but also abduction, which draws tentative conclusions from limited information. Abduction can be regarded as the inverse form of syllogism in its basic structure, that is, a process of drawing a minor premise from a major premise and conclusion. This paper explores the accuracy of LLMs in abductive reasoning by converting a syllogistic dataset into one suitable for abduction. It aims to investigate whether the state-of-the-art LLMs exhibit biases in abduction and to identify potential areas for improvement, emphasizing the importance of contextualized reasoning beyond formal deduction. This investigation is vital for advancing the understanding and application of LLMs in complex reasoning tasks, offering insights into bridging the gap between machine and human cognition.",
      "published": "2026-03-06T16:06:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.06428v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06428v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "CLoPA: Continual Low Parameter Adaptation of Interactive Segmentation for Medical Image Annotation",
      "authors": [
        "Parhom Esmaeili",
        "Chayanin Tangwiriyasakul",
        "Eli Gibson",
        "Sebastien Ourselin",
        "M. Jorge Cardoso"
      ],
      "abstract": "Interactive segmentation enables clinicians to guide annotation, but existing zero-shot models like nnInteractive fail to consistently reach expert-level performance across diverse medical imaging tasks. Because annotation campaigns produce a growing stream of task-specific labelled data, online adaptation of the segmentation model is a natural complement to zero-shot inference. We propose CLoPA, a continual adaptation strategy that tunes a small fraction of nnInteractive's parameters on the annotation cache, triggered by lightweight episode scheduling. CLoPA requires no new parameters or changes to the inference pipeline, and operates entirely within the existing annotation workflow. Across eight Medical Segmentation Decathlon tasks spanning diverse anatomical targets and imaging characteristics, CLoPA rapidly elevates performance to expert-level, even for tasks where nnInteractive previously failed, with the majority of gains realised after a single training episode. We show that the benefits of tuning different parameter groups depends on task characteristics and data regimes. Also, that for targets with complex geometries (e.g., hepatic vessels), instance normalisation and low-level feature tuning saturates, suggesting a need for deeper feature-representation alignment in the most challenging scenarios.",
      "published": "2026-03-06T16:03:20Z",
      "abstract_url": "http://arxiv.org/abs/2603.06426v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06426v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Reference Architecture of Reinforcement Learning Frameworks",
      "authors": [
        "Xiaoran Liu",
        "Istvan David"
      ],
      "abstract": "The surge in reinforcement learning (RL) applications gave rise to diverse supporting technology, such as RL frameworks. However, the architectural patterns of these frameworks are inconsistent across implementations and there exists no reference architecture (RA) to form a common basis of comparison, evaluation, and integration. To address this gap, we propose an RA of RL frameworks. Through a grounded theory approach, we analyze 18 state-of-the-practice RL frameworks and, by that, we identify recurring architectural components and their relationships, and codify them in an RA. To demonstrate our RA, we reconstruct characteristic RL patterns. Finally, we identify architectural trends, e.g., commonly used components, and outline paths to improving RL frameworks.",
      "published": "2026-03-06T15:51:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.06413v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06413v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Talk Freely, Execute Strictly: Schema-Gated Agentic AI for Flexible and Reproducible Scientific Workflows",
      "authors": [
        "Joel Strickland",
        "Arjun Vijeta",
        "Chris Moores",
        "Oliwia Bodek",
        "Bogdan Nenchev",
        "Thomas Whitehead",
        "Charles Phillips",
        "Karl Tassenberg",
        "Gareth Conduit",
        "Ben Pellegrini"
      ],
      "abstract": "Large language models (LLMs) can now translate a researcher's plain-language goal into executable computation, yet scientific workflows demand determinism, provenance, and governance that are difficult to guarantee when an LLM decides what runs. Semi-structured interviews with 18 experts across 10 industrial R&D stakeholders surface 2 competing requirements--deterministic, constrained execution and conversational flexibility without workflow rigidity--together with boundary properties (human-in-the-loop control and transparency) that any resolution must satisfy. We propose schema-gated orchestration as the resolving principle: the schema becomes a mandatory execution boundary at the composed-workflow level, so that nothing runs unless the complete action--including cross-step dependencies--validates against a machine-checkable specification. We operationalize the 2 requirements as execution determinism (ED) and conversational flexibility (CF), and use these axes to review 20 systems spanning 5 architectural groups along a validation-scope spectrum. Scores are assigned via a multi-model protocol--15 independent sessions across 3 LLM families--yielding substantial-to-near-perfect inter-model agreement (Krippendorff a=0.80 for ED and a=0.98 for CF), demonstrating that multi-model LLM scoring can serve as a reusable alternative to human expert panels for architectural assessment. The resulting landscape reveals an empirical Pareto front--no reviewed system achieves both high flexibility and high determinism--but a convergence zone emerges between the generative and workflow-centric extremes. We argue that a schema-gated architecture, separating conversational from execution authority, is positioned to decouple this trade-off, and distill 3 operational principles--clarification-before-execution, constrained plan-act orchestration, and tool-to-workflow-level gating--to guide adoption.",
      "published": "2026-03-06T15:40:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.06394v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06394v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Kinetic-based regularization: Learning spatial derivatives and PDE applications",
      "authors": [
        "Abhisek Ganguly",
        "Santosh Ansumali",
        "Sauro Succi"
      ],
      "abstract": "Accurate estimation of spatial derivatives from discrete and noisy data is central to scientific machine learning and numerical solutions of PDEs. We extend kinetic-based regularization (KBR), a localized multidimensional kernel regression method with a single trainable parameter, to learn spatial derivatives with provable second-order accuracy in 1D. Two derivative-learning schemes are proposed: an explicit scheme based on the closed-form prediction expressions, and an implicit scheme that solves a perturbed linear system at the points of interest. The fully localized formulation enables efficient, noise-adaptive derivative estimation without requiring global system solving or heuristic smoothing. Both approaches exhibit quadratic convergence, matching second-order finite difference for clean data, along with a possible high-dimensional formulation. Preliminary results show that coupling KBR with conservative solvers enables stable shock capture in 1D hyperbolic PDEs, acting as a step towards solving PDEs on irregular point clouds in higher dimensions while preserving conservation laws.",
      "published": "2026-03-06T15:32:27Z",
      "abstract_url": "http://arxiv.org/abs/2603.06380v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06380v1",
      "categories": [
        "math.NA",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ESAA-Security: An Event-Sourced, Verifiable Architecture for Agent-Assisted Security Audits of AI-Generated Code",
      "authors": [
        "Elzo Brito dos Santos Filho"
      ],
      "abstract": "AI-assisted software generation has increased development speed, but it has also amplified a persistent engineering problem: systems that are functionally correct may still be structurally insecure. In practice, prompt-based security review with large language models often suffers from uneven coverage, weak reproducibility, unsupported findings, and the absence of an immutable audit trail. The ESAA architecture addresses a related governance problem in agentic software engineering by separating heuristic agent cognition from deterministic state mutation through append-only events, constrained outputs, and replay-based verification. This paper presents ESAA-Security, a domain-specific specialization of ESAA for agent-assisted security auditing of software repositories, with particular emphasis on AI-generated or AI-modified code. ESAA-Security structures auditing as a governed execution pipeline with four phases reconnaissance, domain audit execution, risk classification, and final reporting and operationalizes the workflow into 26 tasks, 16 security domains, and 95 executable checks. The framework produces structured check results, vulnerability inventories, severity classifications, risk matrices, remediation guidance, executive summaries, and a final markdown/JSON audit report. The central idea is that security review should not be modeled as a free-form conversation with an LLM, but as an evidence-oriented audit process governed by contracts and events. In ESAA-Security, agents emit structured intentions under constrained protocols; the orchestrator validates them, persists accepted outputs to an append-only log, reprojects derived views, and verifies consistency through replay and hashing. The result is a traceable, reproducible, and risk-oriented audit architecture whose final report is auditable by construction.",
      "published": "2026-03-06T15:15:26Z",
      "abstract_url": "http://arxiv.org/abs/2603.06365v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06365v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "CLAIRE: Compressed Latent Autoencoder for Industrial Representation and Evaluation -- A Deep Learning Framework for Smart Manufacturing",
      "authors": [
        "Mohammadhossein Ghahramani",
        "Mengchu Zhou"
      ],
      "abstract": "Accurate fault detection in high-dimensional industrial environments remains a major challenge due to the inherent complexity, noise, and redundancy in sensor data. This paper introduces CLAIRE, i.e., a hybrid end-to-end learning framework that integrates unsupervised deep representation learning with supervised classification for intelligent quality control in smart manufacturing systems. It employs an optimized deep autoencoder to transform raw input into a compact latent space, effectively capturing the intrinsic data structure while suppressing irrelevant or noisy features. The learned representations are then fed into a downstream classifier to perform binary fault prediction. Experimental results on a high-dimensional dataset demonstrate that CLAIRE significantly outperforms conventional classifiers trained directly on raw features. Moreover, the framework incorporates a post hoc phase, using a game-theory-based interpretability technique, to analyze the latent space and identify the most informative input features contributing to fault predictions. The proposed framework highlights the potential of integrating explainable AI with feature-aware regularization for robust fault detection. The modular and interpretable nature of the proposed framework makes it highly adaptable, offering promising applications in other domains characterized by complex, high-dimensional data, such as healthcare, finance, and environmental monitoring.",
      "published": "2026-03-06T15:11:58Z",
      "abstract_url": "http://arxiv.org/abs/2603.06361v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06361v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Dynamic Chunking Diffusion Transformer",
      "authors": [
        "Akash Haridas",
        "Utkarsh Saxena",
        "Parsa Ashrafi Fashi",
        "Mehdi Rezagholizadeh",
        "Vikram Appia",
        "Emad Barsoum"
      ],
      "abstract": "Diffusion Transformers process images as fixed-length sequences of tokens produced by a static $\\textit{patchify}$ operation. While effective, this design spends uniform compute on low- and high-information regions alike, ignoring that images contain regions of varying detail and that the denoising process progresses from coarse structure at early timesteps to fine detail at late timesteps. We introduce the Dynamic Chunking Diffusion Transformer (DC-DiT), which augments the DiT backbone with a learned encoder-router-decoder scaffold that adaptively compresses the 2D input into a shorter token sequence in a data-dependent manner using a chunking mechanism learned end-to-end with diffusion training. The mechanism learns to compress uniform background regions into fewer tokens and detail-rich regions into more tokens, with meaningful visual segmentations emerging without explicit supervision. Furthermore, it also learns to adapt its compression across diffusion timesteps, using fewer tokens at noisy stages and more tokens as fine details emerge. On class-conditional ImageNet $256{\\times}256$, DC-DiT consistently improves FID and Inception Score over both parameter-matched and FLOP-matched DiT baselines across $4{\\times}$ and $16{\\times}$ compression, showing this is a promising technique with potential further applications to pixel-space, video and 3D generation. Beyond accuracy, DC-DiT is practical: it can be upcycled from pretrained DiT checkpoints with minimal post-training compute (up to $8{\\times}$ fewer training steps) and composes with other dynamic computation methods to further reduce generation FLOPs.",
      "published": "2026-03-06T14:59:11Z",
      "abstract_url": "http://arxiv.org/abs/2603.06351v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06351v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "MoEless: Efficient MoE LLM Serving via Serverless Computing",
      "authors": [
        "Hanfei Yu",
        "Bei Ouyang",
        "Shwai He",
        "Ang Li",
        "Hao Wang"
      ],
      "abstract": "Large Language Models (LLMs) have become a cornerstone of AI, driving progress across diverse domains such as content creation, search and recommendation systems, and AI-assisted workflows. To alleviate extreme training costs and advancing model scales, Mixture-of-Experts (MoE) has become a popular backbone for modern LLMs, which are commonly served in distributed deployment using expert parallelism (EP). However, MoE's sparse activation mechanism leads to severe expert load imbalance, where a few experts become overloaded while others remain idle, resulting in expert stragglers that inflate inference latency and serving cost. Existing expert load balancing solutions assume static resource configurations on serverful infrastructures, limiting expert scalability and elasticity, and resulting in either costly real-time expert swapping or degraded generation quality. We present MoEless, the first serverless MoE serving framework that mitigates expert load imbalance and accelerates inference via serverless experts. MoEless employs lightweight, layer-aware predictors to accurately estimate incoming expert load distributions and proactively identify stragglers. We design optimized expert scaling and placement strategies to maximize function locality, improve GPU utilization, and balance loads across experts and GPUs. MoEless is prototyped on top of Megatron-LM and deployed on an eight-GPU testbed. Experiments with open-source MoE models and real-world workloads show that MoEless reduces inference latency by 43% and inference cost by 84% compared to state-of-the-art solutions.",
      "published": "2026-03-06T14:58:16Z",
      "abstract_url": "http://arxiv.org/abs/2603.06350v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06350v1",
      "categories": [
        "cs.DC",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "AI End-to-End Radiation Treatment Planning Under One Second",
      "authors": [
        "Simon Arberet",
        "Riqiang Gao",
        "Martin Kraus",
        "Florin C. Ghesu",
        "Wilko Verbakel",
        "Mamadou Diallo",
        "Anthony Magliari",
        "Venkatesan Karuppusamy",
        "Sushil Beriwal",
        "REQUITE Consortium",
        "Ali Kamen",
        "Dorin Comaniciu"
      ],
      "abstract": "Artificial intelligence-based radiation therapy (RT) planning has the potential to reduce planning time and inter-planner variability, improving efficiency and consistency in clinical workflows. Most existing automated approaches rely on multiple dose evaluations and corrections, resulting in plan generation times of several minutes. We introduce AIRT (Artificial Intelligence-based Radiotherapy), an end-to-end deep-learning framework that directly infers deliverable treatment plans from CT images and structure contours. AIRT generates single-arc VMAT prostate plans, from imaging and anatomical inputs to leaf sequencing, in under one second on a single Nvidia A100 GPU. The framework includes a differentiable dose feedback, an adversarial fluence map shaping, and a plan generation augmentation to improve plan quality and robustness. The model was trained on more than 10,000 intact prostate cases. Non-inferiority to RapidPlan Eclipse was demonstrated across target coverage and OAR sparing metrics. Target homogeneity (HI = 0.10 $\\pm$ 0.01) and OAR sparing were similar to reference plans when evaluated using AcurosXB. These results represent a significant step toward ultra-fast standardized RT planning and a streamlined clinical workflow.",
      "published": "2026-03-06T14:45:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.06338v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06338v1",
      "categories": [
        "eess.IV",
        "cs.AI",
        "cs.LG",
        "eess.SY",
        "physics.med-ph"
      ]
    },
    {
      "title": "SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement",
      "authors": [
        "Subramanyam Sahoo",
        "Aman Chadha",
        "Vinija Jain",
        "Divya Chaudhary"
      ],
      "abstract": "Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Goal Drift Index (GDI), a learned multi-signal detector combining semantic, lexical, structural, and distributional measures; (ii) constraint preservation checks that enforce safety-critical invariants such as syntactic correctness and non-hallucination; and (iii) regression-risk quantification to flag improvement cycles that undo prior gains. Across 189 tasks in code generation, mathematical reasoning, and truthfulness, SAHOO produces substantial quality gains, including 18.3 percent improvement in code tasks and 16.8 percent in reasoning, while preserving constraints in two domains and maintaining low violations in truthfulness. Thresholds are calibrated on a small validation set of 18 tasks across three cycles. We further map the capability-alignment frontier, showing efficient early improvement cycles but rising alignment costs later and exposing domain-specific tensions such as fluency versus factuality. SAHOO therefore makes alignment preservation during recursive self-improvement measurable, deployable, and systematically validated at scale.",
      "published": "2026-03-06T14:44:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.06333v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06333v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Structured Exploration vs. Generative Flexibility: A Field Study Comparing Bandit and LLM Architectures for Personalised Health Behaviour Interventions",
      "authors": [
        "Dominik P. Hofer",
        "Haochen Song",
        "Rania Islambouli",
        "Laura Hawkins",
        "Ananya Bhattacharjee",
        "Meredith Franklin",
        "Joseph Jay Williams",
        "Jan D. Smeddinck"
      ],
      "abstract": "Behaviour Change Techniques (BCTs) are central to digital health interventions, yet selecting and delivering effective techniques remains challenging. Contextual bandits enable statistically grounded optimisation of BCT selection, while Large Language Models (LLMs) offer flexible, context-sensitive message generation. We conducted a 4-week study on physical activity motivation (N=54; 9 post-study interviews) that compared five daily messaging approaches: random templates, contextual bandit with templates, LLM generation, hybrid bandit+LLM, and LLM with interaction history. LLM-based approaches were rated substantially more helpful than templates, but no significant differences emerged among LLM conditions. Unexpectedly, bandit optimisation for BCTs selection yielded no additional perceived helpfulness compared with LLM-only approaches. Unconstrained LLMs focused heavily on a single BCT, whereas bandit systems enforced systematic exploration-exploitation across techniques. Quantitative and qualitative findings suggest contextual acknowledgement of user input drove perceived helpfulness. We contribute design suggestions for reflective AI health behaviour change systems that address a trade-off between structured exploration and generative autonomy.",
      "published": "2026-03-06T14:42:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.06330v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06330v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "From Entropy to Calibrated Uncertainty: Training Language Models to Reason About Uncertainty",
      "authors": [
        "Azza Jenane",
        "Nassim Walha",
        "Lukas Kuhn",
        "Florian Buettner"
      ],
      "abstract": "Large Language Models (LLMs) that can express interpretable and calibrated uncertainty are crucial in high-stakes domains. While methods to compute uncertainty post-hoc exist, they are often sampling-based and therefore computationally expensive or lack calibration. We propose a three-stage pipeline to post-train LLMs to efficiently infer calibrated uncertainty estimates for their responses. First, we compute fine-grained entropy-based uncertainty scores on the training data, capturing the distributional variability of model outputs in embedding space. Second, these scores are calibrated via Platt scaling, producing reliable and human-interpretable uncertainty signals. Finally, the target LLM is post-trained via reinforcement learning to align its policy with these calibrated signals through a verifiable reward function. Unlike post-hoc uncertainty estimation methods, our approach provides interpretable and computationally efficient uncertainty estimates at test time. Experiments show that models trained with our pipeline achieve better calibration than baselines and generalize to unseen tasks without further processing, suggesting that they learn a robust uncertainty reasoning behavior.",
      "published": "2026-03-06T14:21:42Z",
      "abstract_url": "http://arxiv.org/abs/2603.06317v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06317v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning Where the Physics Is: Probabilistic Adaptive Sampling for Stiff PDEs",
      "authors": [
        "Akshay Govind Srinivasan",
        "Balaji Srinivasan"
      ],
      "abstract": "Modeling stiff partial differential equations (PDEs) with sharp gradients remains a significant challenge for scientific machine learning. While Physics-Informed Neural Networks (PINNs) struggle with spectral bias and slow training times, Physics-Informed Extreme Learning Machines (PIELMs) offer a rapid, closed-form linear solution but are fundamentally limited by physics-agnostic, random initialization. We introduce the Gaussian Mixture Model Adaptive PIELM (GMM-PIELM), a probabilistic framework that learns a probability density function representing the ``location of physics'' for adaptively sampling kernels of PIELMs. By employing a weighted Expectation-Maximization (EM) algorithm, GMM-PIELM autonomously concentrates radial basis function centers in regions of high numerical error, such as shock fronts and boundary layers. This approach dynamically improves the conditioning of the hidden layer without the expensive gradient-based optimization(of PINNs) or Bayesian search. We evaluate our methodology on 1D singularly perturbed convection-diffusion equations with diffusion coefficients $ν=10^{-4}$. Our method achieves $L_2$ errors up to $7$ orders of magnitude lower than baseline RBF-PIELMs, successfully resolving exponentially thin boundary layers while retaining the orders-of-magnitude speed advantage of the ELM architecture.",
      "published": "2026-03-06T13:46:38Z",
      "abstract_url": "http://arxiv.org/abs/2603.06287v1",
      "pdf_url": "https://arxiv.org/pdf/2603.06287v1",
      "categories": [
        "cs.CE",
        "cs.AI",
        "cs.LG",
        "math.AP"
      ]
    }
  ]
};
