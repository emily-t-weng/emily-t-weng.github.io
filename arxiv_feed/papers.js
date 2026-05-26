const PAPERS_DATA = {
  "last_updated": "2026-05-26 04:15:09 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "From Model Scaling to System Scaling: Scaling the Harness in Agentic AI",
      "authors": [
        "Shangding Gu"
      ],
      "abstract": "This paper studies the next major bottleneck in agentic AI as system scaling, not only model scaling: the design of auditable, persistent, modular, and verifiable architectures around foundation models. We refer to this shift as scaling the harness: treating the structured execution layer around a foundation model as a first-class object of design, evaluation, and optimization. Although recent large language models enable agents to use tools, retrieve information, maintain memory, and execute long-horizon workflows, evaluation remains largely model-centric, often reducing agents to final-task success while treating memory, retrieval, tool use, orchestration, verification, and governance as secondary implementation details. This framing is increasingly inadequate because agent performance emerges from the interaction among the foundation model, memory substrate, context constructor, skill-routing layer, orchestration loop, and verification-and-governance layer. Together, these components form the agent harness, which translates model capability into long-horizon agent behavior. We study scaling the harness through three core bottlenecks: context governance, trustworthy memory, and dynamic skill routing, together with the orchestration and governance mechanisms that coordinate and constrain them. We further outline a research agenda for harness-level benchmarks that go beyond one-shot task success to measure trajectory quality, memory hygiene, context efficiency, communication fidelity, verification cost, and safe evolution over time. To make the discussion concrete, we develop CheetahClaws: https://github.com/SafeRL-Lab/cheetahclaws, a Python-native reference harness, and compare it with Claude Code and OpenClaw. Our main claim is that future progress in agentic AI will depend as much on system design as on stronger foundation models.",
      "published": "2026-05-25T17:59:36Z",
      "abstract_url": "http://arxiv.org/abs/2605.26112v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26112v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Squeezing Capacity from Multimodal Large Language Models for Subject-driven Generation",
      "authors": [
        "Shuhong Zheng",
        "Aashish Kumar Misraa",
        "Yu-Teng Li",
        "Yu-Jhe Li",
        "Igor Gilitschenski"
      ],
      "abstract": "Subject-driven image generation aims to synthesize new images that preserve the identity of the given subject while following textual instructions. Existing approaches often encode text and reference images separately. This limits cross-modal reasoning abilities and causes copy-paste artifacts. Recent frameworks that connect multimodal models and diffusion models improve instruction following, but largely overlook identity preservation. To address these limitations, we condition diffusion models on Multimodal Large Language Models (MLLMs) that jointly encode text and reference images, and augment it with VAE-based identity conditioning. A novel Dual Layer Aggregation (DLA) module is designed to aggregate multi-level MLLM features for optimal conditioning, and a multi-stage denoising strategy is applied to progressively balance the semantic information from MLLM and fine-detail identity from VAE during inference. Extensive experiments demonstrate that our approach harmonizes multimodal understanding with identity preservation, mitigates copy-paste issues, and achieves superior performance regarding human preference on subject-driven image generation. Our project website is available at https://zsh2000.github.io/squeeze-mllm-subject-gen/.",
      "published": "2026-05-25T17:59:35Z",
      "abstract_url": "http://arxiv.org/abs/2605.26111v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26111v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "cs.MM"
      ]
    },
    {
      "title": "Beyond Summaries: Structure-Aware Labeling of Code Changes with Large Language Models",
      "authors": [
        "Bar Weiss",
        "Antonio Abu-Nassar",
        "Adi Sosnovich",
        "Karen Yorav"
      ],
      "abstract": "Code review is a critical practice in software engineering, yet the growing scale and frequency of code patches in modern projects, together with the widespread adoption of AI code assistants, make manual review increasingly challenging. Identifying the types of changes within a patch, such as renames, moves, or logic modifications, can substantially improve review efficiency by enabling prioritization, filtering, and automation. However, existing LLM-based approaches to code review have largely focused on summarization and comment generation, leaving structured code reviews underexplored. In this paper, we present a systematic study of using large language models (LLMs) for taxonomy-based labeling of code changes in a code patch. We introduce a two-stage pipeline that assigns labels to diff hunks and then refines them to capture structural relationships and semantic attributes, such as rename propagation and type changes. Our approach employs few-shot prompting to produce language-agnostic and customizable labels, without the engineering overhead of traditional static-analysis pipelines. We evaluate four LLMs across multiple context configurations on a manually curated benchmark of natural and synthetic patches. Our best configuration achieves up to $84\\%$ recall and $81\\%$ precision, with high accuracy in extracting relational and attribute metadata. These results suggest that LLM-based labeling can effectively complement static analysis by enabling flexible, multilingual, and automation-friendly code review workflows.",
      "published": "2026-05-25T17:56:46Z",
      "abstract_url": "http://arxiv.org/abs/2605.26100v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26100v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Language Models Need Sleep",
      "authors": [
        "Sangyun Lee",
        "Sean McLeish",
        "Tom Goldstein",
        "Giulia Fanti"
      ],
      "abstract": "Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning.",
      "published": "2026-05-25T17:55:39Z",
      "abstract_url": "http://arxiv.org/abs/2605.26099v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26099v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "OrpQuant: Geometric Orthogonal Residual Projection for Multiplier-Free Power-of-Two Transformer Quantization",
      "authors": [
        "Maoyang Xiang",
        "Bo Wang",
        "Tao Luo"
      ],
      "abstract": "The deployment of Large Language Models (LLMs) and Vision Transformers (ViTs) on edge devices is significantly constrained by memory limitations and the critical timing bottlenecks introduced by dense Multiply-Accumulate (MAC) arrays. In the ultra-low bit regime, logarithmic Power-of-Two (PoT) quantization provides a hardware-efficient alternative by replacing MAC operations with bit-shifts. However, the non-uniform exponential lattice is inherently limited by a \\textbf{Low Angular Resolution Regime}, a structural flaw that becomes particularly pronounced at sub-4-bit thresholds, leading to a notable degradation of high-dimensional feature manifolds. To address this geometric limitation, we propose Orthogonal Residual Projection (ORP), an algorithm-hardware co-design framework. By formulating quantization as a dual-basis geometric projection, ORP adaptively synthesizes a higher-resolution residual lattice using strictly shift-and-add operations. Furthermore, ORP's analytical solver offers a practical alternative to computationally intensive gradient-based optimization, reducing the full-model calibration time for LLaMA-2-7B to approximately \\textbf{15 minutes}. Extensive evaluations demonstrate ORP's applicability across modalities and its hardware efficiency. Under the 3-bit (W3/A16) constraint, ORP achieves a perplexity of 6.10 on LLaMA-2-7B, comparing favorably to conventional MAC-intensive baselines like AWQ without relying on asymmetric scaling, while maintaining competitive accuracy in 4-bit scenarios. At the silicon level, standard-cell RTL synthesis at a 28nm node indicates that ORP effectively mitigates the timing bottlenecks associated with dense multiplier trees.",
      "published": "2026-05-25T17:52:46Z",
      "abstract_url": "http://arxiv.org/abs/2605.26092v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26092v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Claw-Anything: Benchmarking Always-On Personal Assistants with Broader Access to User's Digital World",
      "authors": [
        "Yusong Lin",
        "Xinyuan Liang",
        "Haiyang Wang",
        "Qipeng Gu",
        "Siqi Cheng",
        "Jiangui Chen",
        "Shuzhe Wu",
        "Feiyang Pan",
        "Lue Fan",
        "Sanyuan Zhao",
        "Dandan Tu"
      ],
      "abstract": "Large language model agents are increasingly envisioned as always-on personal assistants with access to anything relevant in the user's digital world. Yet current systems operate over only narrow slices of that world, limiting context-sensitive reasoning and effective assistance. Existing benchmarks similarly provide only partial user state and therefore fail to capture performance in such a broad, always-on setting. To address this gap, we introduce Claw-Anything, a benchmark that expands agent context along three dimensions: long-horizon activity histories, interdependent backend services, and integrated GUI and CLI interaction across multiple devices. To instantiate this setting, we simulate months of user activity through multi-round event injection, producing complex world states and realistic noise, including irrelevant events and conflicting signals. Agents must reason over rich contextual environments while remaining robust to such noise. This expanded scope also enables the evaluation of proactive assistance, requiring agents to anticipate user needs and deliver timely recommendations. Experiments show that GPT-5.5 achieves only 34.5% pass@1, substantially below prior benchmarks, underscoring a gap between current agent capabilities and the demands of always-on personal assistance. Alongside the benchmark, we release an automated data-generation pipeline that yields 2,000 training environments and improves the base model by 23.7%, demonstrating its utility of scalable data infrastructure.",
      "published": "2026-05-25T17:50:04Z",
      "abstract_url": "http://arxiv.org/abs/2605.26086v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26086v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking Weak Supervision in Anomaly Detection: A Comprehensive Benchmark",
      "authors": [
        "Xu Yao",
        "Siyuan Zhou",
        "Wu Zhenbo",
        "Chaochuan Hou",
        "Shuang Liang",
        "Shiping wang",
        "Hailiang Huang",
        "Songqiao Han",
        "Minqi Jiang"
      ],
      "abstract": "Weakly supervised anomaly detection (WSAD) has developed in three primary directions: incomplete, inexact, and inaccurate supervision. However, these directions remain isolated, lacking a unified framework to assess whether they address unique challenges or share fundamental mechanics. This paper introduces WSADBench, the first benchmark that unifies evaluation across distinct weakly supervised scenarios, benchmarking diverse approaches from specialized WSAD methods to advanced tabular foundation models. WSADBench establishes standardized protocols to evaluate 36 algorithms across 4 modalities by systematically varying label quantity, granularity, and quality, revealing the performance boundaries of various methods. Based on over 700K experiments, WSADBench reveals four critical insights: (i) Strong intrinsic correlations exist between these weak supervision scenarios, challenging the isolation of current research directions. (ii) Specialized WSAD algorithms excel only in extreme label-scarcity regimes but are quickly dominated by tabular foundation models and general classification methods as supervision increases or in OOD scenarios. (iii) Unlabeled data shows inconsistent utility across settings, with marginal gains compared to label refinement. (iv) Models exhibit asymmetric sensitivity to different types of label noise. We release WSADBench as an open-source benchmark with code and datasets to facilitate future WSAD research: https://github.com/SUFE-AILAB/WSADBench.",
      "published": "2026-05-25T17:32:58Z",
      "abstract_url": "http://arxiv.org/abs/2605.26068v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26068v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding",
      "authors": [
        "Rustem Takhanov",
        "Zhenisbek Assylbekov"
      ],
      "abstract": "Conditionally positive definite (CPD) kernels are defined with respect to a function class $\\mathcal{F}$. It is well known that such a kernel $K$ is associated with its native space (defined analogously to an RKHS), which in turn gives rise to a learning method -- called conditional kernel ridge regression (conditional KRR) due to its analogy with KRR -- where the estimated regression function is penalized by the square of its native space norm. This method is of interest because it can be viewed as classical linear regression, with features specified by $\\mathcal{F}$, followed by the application of standard KRR to the residual (unexplained) component of the target variable. Methods of this type have recently attracted increasing attention. We study the statistical properties of this method by reducing its behavior to that of KRR with another fixed kernel, called the residual kernel. Our main theoretical result shows that such a reduction is indeed possible, at the cost of an additional term in the expected test risk, bounded by $\\mathcal{O}(1/\\sqrt{N})$, where $N$ is the sample size and the hidden constant depends on the class $\\mathcal{F}$ and the input distribution. This reduction enables us to analyze conditional KRR in the case where $K$ is positive definite and $\\mathcal{F}$ is given by the first $k$ principal eigenfunctions in the Mercer decomposition of $K$. We also consider the setting where $\\mathcal{F}$ consists of $k$ random features from a random feature representation of $K$. It turns out that these two settings are closely related. Both our theoretical analysis and experiments confirm that conditional KRR outperforms standard KRR in these cases whenever the $\\mathcal{F}$-component of the regression function is more pronounced than the residual part.",
      "published": "2026-05-25T17:31:54Z",
      "abstract_url": "http://arxiv.org/abs/2605.26067v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26067v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Neuronal Stochastic Attention Circuit (NSAC) for Probabilistic Representation Learning",
      "authors": [
        "Waleed Razzaq",
        "Yun-Bo Zhao"
      ],
      "abstract": "Reliable quantification of uncertainty estimates in continuous-time (CT) representation learning remains nascent, particularly within CT attention architectures. We introduce the Neuronal Stochastic Attention Circuit (NSAC), a novel biologically-inspired CT attention architecture that reformulates attention logit computation as the solution of an Ornstein-Uhlenbeck stochastic differential equation modulated by input-dependent, nonlinear interlinked gates derived from repurposed C.elegans Neuronal Circuit Policies (NCPs) wiring mechanism. It induces Gaussian distribution over logits that propagates principled stochasticity through logistic-normal distribution over attention weights to yield probabilistic output. A two-term objective function combining Gaussian negative log-likelihood with an epistemic-separation regularizer enforces higher predictive variance and enables joint quantification of aleatoric and epistemic uncertainty. Empirically, we implement NSAC in a diverse set of learning tasks including: (i) irregular CT function approximation; (ii) multivariate regression; (iii) long-range forecasting; (iv) Industry 4.0; and (v) the lane-keeping of autonomous vehicles. We observe that the NSAC remains competitive against several baselines in terms of accuracy and produces reasonably well-calibrated uncertainty estimates while being interpretable at the neuronal cell level.",
      "published": "2026-05-25T17:19:14Z",
      "abstract_url": "http://arxiv.org/abs/2605.26061v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26061v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges",
      "authors": [
        "Parth Darshan",
        "Abhishek Divekar"
      ],
      "abstract": "Customizing an LLM judge to a specific task or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) doesn't apply to the multi-objective textual gradient setting. We test five decomposition modes of textual gradient optimizers by varying how much cross-task information the loss, gradient and optimizer LLMs share. In 6 of 10 configurations, we observe that optimization never improves over the initial prompt. Gradient specificity drops by 59% (from 9.0 to 3.7) when the gradient LLM processes multiple criteria jointly. Separately, we observe that naively combining per-task instructions into a single prompt degrades Spearman's rho by -5.3%. These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge customization using textual feedback.",
      "published": "2026-05-25T17:08:55Z",
      "abstract_url": "http://arxiv.org/abs/2605.26046v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26046v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG",
        "cs.MA",
        "cs.SE"
      ]
    },
    {
      "title": "L2IR: Revealing Latent Intent in Graph Fraud Detection",
      "authors": [
        "Jinsheng Guo",
        "Zhenhao Weng",
        "Yibo Liu",
        "Yan Qiao",
        "Meng Li"
      ],
      "abstract": "Graph fraud detection has long depended on Graph Neural Networks (GNNs) to propagate and aggregate information across relational data. A critical obstacle in practice, however, is that fraudsters frequently disguise themselves by forging numerous connections with benign users, causing fraud signals to be progressively diluted during neighborhood aggregation and undermining detection reliability. While recent efforts have used Large Language Models (LLMs) to provide rich semantic cues for fraud detection, the underlying intent behind suspicious connections remains insufficiently explored. Compounding this issue, the scarcity of annotated fraud samples makes it difficult to train detectors that remain robust under heavy camouflage. To address these gaps, we propose L2IR, an LLM-driven Latent Intent Revealing framework for graph fraud detection. By uncovering latent intent from both user behaviors and suspicious connections, L2IR extracts intent-aware representations from raw behavioral traces and reasons about the true purpose behind individual connections, effectively distinguishing supportive links from misleading ones. It further incorporates adaptive self-training to enhance robustness under limited supervision. Evaluations on two real-world datasets characterized by pervasive camouflage demonstrate that L2IR surpasses strong baselines and can function as a plug-in enhancement for a range of GNN-based detectors, improving AUPRC by up to 8.27%.",
      "published": "2026-05-25T17:06:13Z",
      "abstract_url": "http://arxiv.org/abs/2605.26040v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26040v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CITYREP: A Unified Benchmark for Urban Representations Across Cities, Tasks, and Modalities",
      "authors": [
        "Junyuan Liu",
        "Xinglei Wang",
        "Zichao Zeng",
        "Jiazhuang Feng",
        "Quan Qin",
        "Ilya Ilyankou",
        "Guangsheng Dong",
        "Tao Cheng"
      ],
      "abstract": "Urban representation learning encodes complex urban environments into general-purpose embeddings for diverse downstream tasks and emerging urban foundation models. However, current evaluations are limited, typically focusing on one or two cities and tasks and relying on random splits that introduce spatial leakage, leading to inflated performance and weak support for cross-location generalization and fair comparison. To address this, we propose CityRep, a unified benchmark that evaluates urban representations across data modalities, cities, and tasks using spatially structured splits. CityRep consists of three key components: (1) a spatial unit-agnostic evaluation framework that supports heterogeneous urban representations through a standardized alignment module; (2) a unified evaluation protocol using block-based spatial splits to mitigate spatial leakage and enable rigorous model comparison; and (3) an extensible multi-city, multi-task benchmark suite spanning 8 cities and 8 tasks across regression, classification, and distribution prediction. We evaluate 11 representative urban representation models. Results show that performance is highly sensitive to the split protocol, with random splits inflating scores and altering model rankings. We also observe substantial variability across cities and tasks, underscoring the need for generalization-aware evaluation. CityRep is released as a reproducible benchmark with datasets, evaluation pipelines, and diagnostic tools to facilitate fair comparison and support future research in urban representation learning towards urban foundation models.",
      "published": "2026-05-25T17:03:46Z",
      "abstract_url": "http://arxiv.org/abs/2605.26036v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26036v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution",
      "authors": [
        "Zixin Jessie Chen",
        "Zhuo Chen",
        "Archer Wang",
        "Jeff Gore",
        "William T. Freeman",
        "Congyue Deng",
        "Marin Soljačić"
      ],
      "abstract": "Creating images from noise is image generation; reconstructing fine details from coarse inputs is super-resolution. Despite their practical differences, both can be understood as reversing information loss across scales. We introduce $\\textbf{SKILD}$, a $\\textbf{S}$cale-invariant $\\textbf{K}$-Space $\\textbf{I}$mage $\\textbf{L}$earning $\\textbf{D}$iffusion model that unifies generation and continuous super-resolution within a single unconditional framework. Both natural images and critical physical systems exhibit scale invariance, and we leverage it to design a forward process that attenuates image content from fine to coarse scales while injecting spectrum-matched Gaussian noise, making scale an explicit coordinate of the diffusion dynamics. The same trained reverse process performs generation and continuous super-resolution by varying only the starting timestep: $\\textit{no task-specific architecture, no conditioning branch, no classifier-free guidance, no retraining per scale factor}$. Empirically, SKILD reaches FID $2.65$ and Inception Score $9.63$ on unconditional CIFAR-10, performs $2\\times$--$8\\times$ super-resolution on ImageNet from a single unconditional checkpoint while outperforming conditional models across perceptual metrics, and reconstructs critical Ising models whose connected four-point correlations closely track the ground truth.",
      "published": "2026-05-25T17:01:21Z",
      "abstract_url": "http://arxiv.org/abs/2605.26032v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26032v1",
      "categories": [
        "cs.CV",
        "cond-mat.stat-mech",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring",
      "authors": [
        "Adina Scheinfeld",
        "Haotan Zhang",
        "Shang Mu",
        "Rudolf L. M. van Herten",
        "Lucas Stoffl",
        "Ali Erturk",
        "Zhuhao Wu",
        "Johannes C. Paetzold"
      ],
      "abstract": "Light sheet fluorescence microscopy (LSM) enables high-resolution, three-dimensional (3D) imaging of biological specimens, providing rich volumetric data for studying cellular organization, pathology, and vascular networks. However, the size, dimensionality, and annotation burden of LSM data make supervised deep learning approaches costly and difficult to scale. Additionally, despite the abundance of unannotated LSM volumes, foundation models for this modality remain underexplored due to computational challenges and the complexity of volumetric representation learning. In this work, we introduce a 3D foundation model for LSM data, pretrained on a large curated collection of 3D images spanning multiple organisms, stains, and imaging protocols. We learn transferable volumetric representations by jointly optimizing for masked reconstruction and image-text alignment. The pretrained backbone drastically reduces the annotation burden, enabling efficient, few-shot adaptation for varied downstream tasks. We evaluate this approach on downstream segmentation, classification, and deblurring. Our results demonstrate consistent improvements over baselines, (1) when measured using standard evaluation metrics and (2) when rigorously assessed by domain experts. This highlights the potential of foundation model pretraining to reduce annotation requirements while improving performance across diverse LSM analysis tasks. Pretrained model weights and code for pretraining and finetuning are publicly available: https://github.com/AdinaScheinfeld/lsm_fm_public_repo.git.",
      "published": "2026-05-25T16:50:58Z",
      "abstract_url": "http://arxiv.org/abs/2605.26026v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26026v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Retrieval-Augmented Detection of Potentially Abusive Clauses in Chilean Terms of Service",
      "authors": [
        "Christoffer Loeffler",
        "Tomás Rey Pizarro",
        "Daniel Ignacio Miranda Vásquez",
        "Andrea Martínez Freile"
      ],
      "abstract": "Online Terms of Service often function as contracts of adhesion, creating asymmetries that may expose consumers to potentially abusive clauses. In Chile, assessing such clauses is legally challenging because some provisions clearly violate mandatory consumer law, whereas others depend on broader standards such as good faith and contractual imbalance. We present a retrieval-augmented generation framework for the automated detection and classification of potentially abusive clauses in Chilean Terms of Service. Designed for local execution, it combines efficient clause detection, hybrid dense--sparse retrieval, reranking, and prompt augmentation to support medium-sized open-weight language models. We also introduce the Chilean Abusive Terms of Service Extended corpus, comprising 100 contracts and 10,029 annotated clauses in 24 legally grounded categories spanning illegal, dark, and gray clauses. Experiments comparing commercial and open-weight language models, fine-tuned encoders, and traditional baselines show that retrieval-augmented prompting substantially improves performance and enables local models to approach larger cloud-based systems at lower computational and token cost. The study also contributes a refined legal annotation scheme and a practical design for AI-assisted consumer contract review.",
      "published": "2026-05-25T16:38:10Z",
      "abstract_url": "http://arxiv.org/abs/2605.26019v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26019v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "AdvantageFlow: Advantage-Weighted Least Squares for RL in Flow Models",
      "authors": [
        "Branislav Kveton",
        "Anup Rao",
        "Subhojyoti Mukherjee",
        "Krishna Kumar Singh",
        "Viet Dac Lai"
      ],
      "abstract": "We introduce AdvantageFlow, a forward-process reinforcement learning algorithm for rectified flow models. Unlike Flow-GRPO, which optimizes the reverse process, we optimize an advantage-weighted forward-process prediction loss. This optimization problem is unstable when advantages are negative and the loss becomes non-convex. We stabilize it by rollout policy regularization, which reduces variance and arises from fitting a local reward-improving target distribution. We evaluate AdvantageFlow on image generation tasks with Stable Diffusion 3.5 Medium. It outperforms both Flow-GRPO and a state-of-the-art forward-process RL baseline based on negative-aware fine-tuning.",
      "published": "2026-05-25T16:32:14Z",
      "abstract_url": "http://arxiv.org/abs/2605.26013v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26013v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Learning in Low-Dimensional Subspaces: Orthogonal Bottlenecks for Reinforcement Learning",
      "authors": [
        "Aleksandar Todorov",
        "Matthia Sabatelli"
      ],
      "abstract": "Deep reinforcement learning (RL) agents commonly rely on high-dimensional neural representations, despite growing evidence that task-relevant value and policy structure may be intrinsically low-dimensional. In this work, we present a simple yet effective representation-level prior that inserts a fixed orthonormal projection to constrain encoder features to a low-dimensional subspace, requiring no auxiliary objectives, pretraining, or changes to the underlying RL algorithm. Under a linear realizability assumption, we prove that when the bottleneck dimension exceeds the intrinsic rank of the optimal value function in feature space, the bottleneck preserves expressivity and leaves the induced gradient dynamics unchanged up to an equivalent low-dimensional parameterization. Empirically, we find that across both single and multi-task benchmarks, baseline performance is either matched or improved once the bottleneck dimension exceeds a small task-dependent threshold; in many cases, value representations can be compressed to extremely low dimensions without loss, and the minimal sufficient dimension depends far more on environment complexity than encoder width. In addition, we analyze representation geometry and find that orthogonal bottlenecks stabilize feature norms and are associated with higher effective rank. Together, these results support a representation-space interpretation of the manifold hypothesis in reinforcement learning and position orthogonal bottlenecks as a lightweight, architecture-agnostic mechanism for shaping RL representations.",
      "published": "2026-05-25T16:31:33Z",
      "abstract_url": "http://arxiv.org/abs/2605.26012v1",
      "pdf_url": "https://arxiv.org/pdf/2605.26012v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SafeCtrl-RL: Inference-Time Adaptive Behaviour Control for LLM Dialogue via RL-Driven Prompt Optimisation",
      "authors": [
        "Michael Orme",
        "Yanchao Yu",
        "Zhiyuan Tan"
      ],
      "abstract": "Ensuring safe and contextually appropriate behaviour in Large Language Models (LLMs) remains a critical challenge for real-world deployment. We present \\textbf{SafeCtrl-RL}, an inference-time behavioural control framework that enables adaptive safety regulation without model retraining or parameter modification. The method formulates dialogue generation as a sequential decision process, where a reinforcement learning agent dynamically selects prompt adjustment strategies based on contextual feedback. This allows unsafe behaviours to be suppressed through iterative refinement, which we conceptualise as inference-time behavioural unlearning. Evaluated across multiple LLMs and unsafe dialogue scenarios, SafeCtrl-RL consistently improves safety and response quality, outperforms existing prompt-based optimisation methods, and achieves favourable performance--efficiency trade-offs. **Warning: This paper may contain examples of harmful language, and reader discretion is recommended.",
      "published": "2026-05-25T16:03:38Z",
      "abstract_url": "http://arxiv.org/abs/2605.25984v1",
      "pdf_url": "https://arxiv.org/pdf/2605.25984v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Creative Quality Alignment: Expert Tacit Knowledge Transfer via Chain-of-Thought Fine-Tuning",
      "authors": [
        "Bo Zou",
        "Chao Xu"
      ],
      "abstract": "This paper provides an empirical implementation of the creative quality metric proposed in Calibrated Surprise (Zou & Xu, 2026a). The question this paper addresses is: does this mathematical claim hold at the engineering level? To make the answer as general as possible, we deliberately choose the strictest engineering conditions: low data cost and a small base model. Training data comes from approximately 100 expert chain-of-thought (CoT) annotations produced by the BC Protocol (Zou & Xu, 2026b). We also identify a data bias: most publicly available alignment datasets are skewed toward craft-related knowledge, while audience modeling and reality-logic coverage are systematically weak. We use the term Creative Quality Alignment (CQA) to describe this class of engineering methods. We also offer a supporting theoretical observation: in an LLM with a single conditional distribution architecture, calibrating the appreciation side automatically transfers to the generation side via architectural duality. This is the structural reason why ~100 CoT examples are sufficient -- not a purely empirical observation like LIMA (Zhou et al., 2023).",
      "published": "2026-05-25T15:52:10Z",
      "abstract_url": "http://arxiv.org/abs/2605.25977v1",
      "pdf_url": "https://arxiv.org/pdf/2605.25977v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability",
      "authors": [
        "Bo Zou",
        "Chao Xu"
      ],
      "abstract": "Large language models (LLMs) face a dual challenge in creative capability evaluation: existing benchmarks (e.g., Story Cloze Test, HellaSwag) measure models' discriminative ability over narrative continuation using multiple-choice recognition paradigms, rather than directly measuring creative generation capability; rubric-based scoring and LLM-as-Judge methods rely on subjective dimension assessment or natural language model outputs, and cannot provide objective, automated scoring mechanisms. This paper proposes QUIET (Quality Understanding via Interlocked Evaluation Testing), a diagnostic benchmark for LLM creative capability based on multi-blank cascaded story cloze. QUIET sets N blanks (10-20) in a story with complete structure, with each blank accompanied by an explicit content constraint, and cascade dependency relationships between blanks -- the content filled into earlier blanks constrains the feasible solution space for later blanks. The evaluated model (or human participants) fills all blanks in open-ended generation mode; the results are scored by an information-theoretic automated scoring protocol without human grading. The scoring protocol directly operationalizes the \"calibrated surprise\" theoretical framework (Zou & Xu, 2026a). For each blank k, a composite score is computed: score = satisfy * (1 + lambda * surprise), where lambda = 1.0. Here, \"satisfy\" measures how well the blank filling satisfies the content constraint (objective logical reasoning judgment, not subjective aesthetic scoring), and \"surprise\" measures the degree of surprise given that the constraint is satisfied. Creative answers that do not satisfy the constraint score zero; answers that satisfy the constraint but are mediocre score low; answers that satisfy the constraint and are surprising score high.",
      "published": "2026-05-25T15:29:58Z",
      "abstract_url": "http://arxiv.org/abs/2605.25955v1",
      "pdf_url": "https://arxiv.org/pdf/2605.25955v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
