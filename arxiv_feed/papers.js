const PAPERS_DATA = {
  "last_updated": "2026-03-24 02:46:38 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention",
      "authors": [
        "Xinyan Wang",
        "Xiaogeng Liu",
        "Chaowei Xiao"
      ],
      "abstract": "Large Reasoning Models (LRMs) achieve strong accuracy on challenging tasks by generating long Chain-of-Thought traces, but suffer from overthinking. Even after reaching the correct answer, they continue generating redundant reasoning steps. This behavior increases latency and compute cost and can also lead to answer drift. Existing mitigation methods either require training-heavy backbone modification or rely on hand-crafted heuristics that do not truly capture overthinking patterns. We propose ROM, the first method that formulates overthinking mitigation as a streaming prediction-and-control problem. ROM attaches a lightweight detection head to the late-layer hidden states of a frozen large language model backbone. It monitors tokens in real time and triggers an early transition to the final answer once overthinking is detected. We also introduce token-level supervision based on solution correctness boundaries and a data augmentation strategy that reduces distilled-data bias. Across seven benchmarks, ROM achieves the highest accuracy (93.51%), the shortest responses (1,159 tokens), and the best response efficiency. Compared with the vanilla baseline, it reduces response length by 47.2% and improves efficiency by 121%. These results show that streaming detection is a promising approach to real-time overthinking mitigation.",
      "published": "2026-03-23T14:26:57Z",
      "abstract_url": "http://arxiv.org/abs/2603.22016v1",
      "pdf_url": "https://arxiv.org/pdf/2603.22016v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "λ-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks",
      "authors": [
        "Cristian Pérez-Corral",
        "Alberto Fernández-Hernández",
        "Jose I. Mestre",
        "Manuel F. Dolz",
        "Enrique S. Quintana-Ortí"
      ],
      "abstract": "Gaussian Error Linear Unit (GELU) is a widely used smooth alternative to Rectifier Linear Unit (ReLU), yet many deployment, compression, and analysis toolchains are most naturally expressed for piecewise-linear (ReLU-type) networks. We study a hardness-parameterized formulation of GELU, f(x;λ)=xΦ(λ x), where Φ is the Gaussian CDF and λ \\in [1, infty) controls gate sharpness, with the goal of turning smooth gated training into a controlled path toward ReLU-compatible models. Learning λ is non-trivial: naive updates yield unstable dynamics and effective gradient attenuation, so we introduce a constrained reparameterization and an optimizer-aware update scheme. Empirically, across a diverse set of model--dataset pairs spanning MLPs, CNNs, and Transformers, we observe structured layerwise hardness profiles and assess their robustness under different initializations. We further study a deterministic ReLU-ization strategy in which the learned gates are progressively hardened toward a principled target, enabling a post-training substitution of λ-GELU by ReLU with reduced disruption. Overall, λ-GELU provides a minimal and interpretable knob to profile and control gating hardness, bridging smooth training with ReLU-centric downstream pipelines.",
      "published": "2026-03-23T13:58:19Z",
      "abstract_url": "http://arxiv.org/abs/2603.21991v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21991v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TREX: Trajectory Explanations for Multi-Objective Reinforcement Learning",
      "authors": [
        "Dilina Rajapakse",
        "Juan C. Rosero",
        "Ivana Dusparic"
      ],
      "abstract": "Reinforcement Learning (RL) has demonstrated its ability to solve complex decision-making problems in a variety of domains, by optimizing reward signals obtained through interaction with an environment. However, many real-world scenarios involve multiple, potentially conflicting objectives that cannot be easily represented by a single scalar reward. Multi-Objective Reinforcement Learning (MORL) addresses this limitation by enabling agents to optimize several objectives simultaneously, explicitly reasoning about trade-offs between them. However, the ``black box\" nature of the RL models makes the decision process behind chosen objective trade-offs unclear. Current Explainable Reinforcement Learning (XRL) methods are typically designed for single scalar rewards and do not account for explanations with respect to distinct objectives or user preferences. To address this gap, in this paper we propose TREX, a Trajectory based Explainability framework to explain Multi-objective Reinforcement Learning policies, based on trajectory attribution. TREX generates trajectories directly from the learned expert policy, across different user preferences and clusters them into semantically meaningful temporal segments. We quantify the influence of these behavioural segments on the Pareto trade-off by training complementary policies that exclude specific clusters, measuring the resulting relative deviation on the observed rewards and actions compared to the original expert policy. Experiments on multi-objective MuJoCo environments - HalfCheetah, Ant and Swimmer, demonstrate the framework's ability to isolate and quantify the specific behavioural patterns.",
      "published": "2026-03-23T13:55:14Z",
      "abstract_url": "http://arxiv.org/abs/2603.21988v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21988v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SecureBreak -- A dataset towards safe and secure models",
      "authors": [
        "Marco Arazzi",
        "Vignesh Kumar Kembu",
        "Antonino Nocera"
      ],
      "abstract": "Large language models are becoming pervasive core components in many real-world applications. As a consequence, security alignment represents a critical requirement for their safe deployment. Although previous related works focused primarily on model architectures and alignment methodologies, these approaches alone cannot ensure the complete elimination of harmful generations. This concern is reinforced by the growing body of scientific literature showing that attacks, such as jailbreaking and prompt injection, can bypass existing security alignment mechanisms. As a consequence, additional security strategies are needed both to provide qualitative feedback on the robustness of the obtained security alignment at the training stage, and to create an ``ultimate'' defense layer to block unsafe outputs possibly produced by deployed models. To provide a contribution in this scenario, this paper introduces SecureBreak, a safety-oriented dataset designed to support the development of AI-driven solutions for detecting harmful LLM outputs caused by residual weaknesses in security alignment. The dataset is highly reliable due to careful manual annotation, where labels are assigned conservatively to ensure safety. It performs well in detecting unsafe content across multiple risk categories. Tests with pre-trained LLMs show improved results after fine-tuning on SecureBreak. Overall, the dataset is useful both for post-generation safety filtering and for guiding further model alignment and security improvements.",
      "published": "2026-03-23T13:41:05Z",
      "abstract_url": "http://arxiv.org/abs/2603.21975v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21975v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Parameter-Efficient Fine-Tuning for Medical Text Summarization: A Comparative Study of Lora, Prompt Tuning, and Full Fine-Tuning",
      "authors": [
        "Ulugbek Shernazarov",
        "Rostislav Svitsov",
        "Bin Shi"
      ],
      "abstract": "Fine-tuning large language models for domain-specific tasks such as medical text summarization demands substantial computational resources. Parameter-efficient fine-tuning (PEFT) methods offer promising alternatives by updating only a small fraction of parameters. This paper compares three adaptation approaches-Low-Rank Adaptation (LoRA), Prompt Tuning, and Full Fine-Tuning-across the Flan-T5 model family on the PubMed medical summarization dataset. Through experiments with multiple random seeds, we demonstrate that LoRA consistently outperforms full fine-tuning, achieving 43.52 +/- 0.18 ROUGE-1 on Flan-T5-Large with only 0.6% trainable parameters compared to 40.67 +/- 0.21 for full fine-tuning. Sensitivity analyses examine the impact of LoRA rank and prompt token count. Our findings suggest the low-rank constraint provides beneficial regularization, challenging assumptions about the necessity of full parameter updates. Code is available at https://github.com/eracoding/llm-medical-summarization",
      "published": "2026-03-23T13:35:11Z",
      "abstract_url": "http://arxiv.org/abs/2603.21970v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21970v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence",
      "authors": [
        "Peter Fasogbon",
        "Ugurcan Budak",
        "Patrice Rondao Alface",
        "Hamed Rezazadegan Tavakoli"
      ],
      "abstract": "The pruning of 3D Gaussian splats is essential for reducing their complexity to enable efficient storage, transmission, and downstream processing. However, most of the existing pruning strategies depend on camera parameters, rendered images, or view-dependent measures. This dependency becomes a hindrance in emerging camera-agnostic exchange settings, where splats are shared directly as point-based representations (e.g., .ply). In this paper, we propose a camera-agnostic, one-shot, post-training pruning method for 3D Gaussian splats that relies solely on attribute-derived neighbourhood descriptors. As our primary contribution, we introduce a hybrid descriptor framework that captures structural and appearance consistency directly from the splat representation. Building on these descriptors, we formulate pruning as a statistical evidence estimation problem and introduce a Beta evidence model that quantifies per-splat reliability through a probabilistic confidence score. Experiments conducted on standardized test sequences defined by the ISO/IEC MPEG Common Test Conditions (CTC) demonstrate that our approach achieves substantial pruning while preserving reconstruction quality, establishing a practical and generalizable alternative to existing camera-dependent pruning strategies.",
      "published": "2026-03-23T12:52:08Z",
      "abstract_url": "http://arxiv.org/abs/2603.21933v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21933v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Deep Reinforcement Learning and The Tale of Two Temporal Difference Errors",
      "authors": [
        "Juan Sebastian Rojas",
        "Chi-Guhn Lee"
      ],
      "abstract": "The temporal difference (TD) error was first formalized in Sutton (1988), where it was first characterized as the difference between temporally successive predictions, and later, in that same work, formulated as the difference between a bootstrapped target and a prediction. Since then, these two interpretations of the TD error have been used interchangeably in the literature, with the latter eventually being adopted as the standard critic loss in deep reinforcement learning (RL) architectures. In this work, we show that these two interpretations of the TD error are not always equivalent. In particular, we show that increasingly-nonlinear deep RL architectures can cause these interpretations of the TD error to yield increasingly different numerical values. Then, building on this insight, we show how choosing one interpretation of the TD error over the other can affect the performance of deep RL algorithms that utilize the TD error to compute other quantities, such as with deep differential (i.e., average-reward) RL methods. All in all, our results show that the default interpretation of the TD error as the difference between a bootstrapped target and a prediction does not always hold in deep RL settings.",
      "published": "2026-03-23T12:43:36Z",
      "abstract_url": "http://arxiv.org/abs/2603.21921v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21921v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Not All Layers Are Created Equal: Adaptive LoRA Ranks for Personalized Image Generation",
      "authors": [
        "Donald Shenaj",
        "Federico Errica",
        "Antonio Carta"
      ],
      "abstract": "Low Rank Adaptation (LoRA) is the de facto fine-tuning strategy to generate personalized images from pre-trained diffusion models. Choosing a good rank is extremely critical, since it trades off performance and memory consumption, but today the decision is often left to the community's consensus, regardless of the personalized subject's complexity. The reason is evident: the cost of selecting a good rank for each LoRA component is combinatorial, so we opt for practical shortcuts such as fixing the same rank for all components. In this paper, we take a first step to overcome this challenge. Inspired by variational methods that learn an adaptive width of neural networks, we let the ranks of each layer freely adapt during fine-tuning on a subject. We achieve it by imposing an ordering of importance on the rank's positions, effectively encouraging the creation of higher ranks when strictly needed. Qualitatively and quantitatively, our approach, LoRA$^2$, achieves a competitive trade-off between DINO, CLIP-I, and CLIP-T across 29 subjects while requiring much less memory and lower rank than high rank LoRA versions. Code: https://github.com/donaldssh/NotAllLayersAreCreatedEqual.",
      "published": "2026-03-23T12:13:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.21884v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21884v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SmaAT-QMix-UNet: A Parameter-Efficient Vector-Quantized UNet for Precipitation Nowcasting",
      "authors": [
        "Nikolas Stavrou",
        "Siamak Mehrkanoon"
      ],
      "abstract": "Weather forecasting supports critical socioeconomic activities and complements environmental protection, yet operational Numerical Weather Prediction (NWP) systems remain computationally intensive, thus being inefficient for certain applications. Meanwhile, recent advances in deep data-driven models have demonstrated promising results in nowcasting tasks. This paper presents SmaAT-QMix-UNet, an enhanced variant of SmaAT-UNet that introduces two key innovations: a vector quantization (VQ) bottleneck at the encoder-decoder bridge, and mixed kernel depth-wise convolutions (MixConv) replacing selected encoder and decoder blocks. These enhancements both reduce the model's size and improve its nowcasting performance. We train and evaluate SmaAT-QMix-UNet on a Dutch radar precipitation dataset (2016-2019), predicting precipitation 30 minutes ahead. Three configurations are benchmarked: using only VQ, only MixConv, and the full SmaAT-QMix-UNet. Grad-CAM saliency maps highlight the regions influencing each nowcast, while a UMAP embedding of the codewords illustrates how the VQ layer clusters encoder outputs. The source code for SmaAT-QMix-UNet is publicly available on GitHub \\footnote{\\href{https://github.com/nstavr04/MasterThesisSnellius}{https://github.com/nstavr04/MasterThesisSnellius}}.",
      "published": "2026-03-23T12:09:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.21879v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21879v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "P^2O: Joint Policy and Prompt Optimization",
      "authors": [
        "Xinyu Lu",
        "Kaiqi Zhang",
        "Jinglin Yang",
        "Boxi Cao",
        "Yaojie Lu",
        "Hongyu Lin",
        "Min He",
        "Xianpei Han",
        "Le Sun"
      ],
      "abstract": "Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, vanilla RLVR suffers from inefficient exploration, particularly when confronting \"hard samples\" that yield nearzero success rates. In such scenarios, the reliance on sparse outcome rewards typically results in zero-advantage estimates, effectively starving the model of supervision signals despite the high informational value of these instances. To address this, we propose P^2O, a novel framework that synergizes Prompt Optimization with Policy Optimization. P^2O identifies hard samples during training iterations and leverages the GeneticPareto (GEPA) prompt optimization algorithm to evolve prompt templates that guide the model toward discovering successful trajectories. Crucially, unlike traditional prompt engineering methods that rely on input augmentation, P^2O distills the reasoning gains induced by these optimized prompts directly into the model parameters. This mechanism provides denser positive supervision signals for hard samples and accelerates convergence. Extensive experiments demonstrate that P^2O not only achieves superior performance on in-distribution datasets but also exhibits strong generalization, yielding substantial improvements on out-of-distribution benchmarks (+4.7% avg.).",
      "published": "2026-03-23T12:08:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.21877v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21877v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Reasoning or Rhetoric? An Empirical Analysis of Moral Reasoning Explanations in Large Language Models",
      "authors": [
        "Aryan Kasat",
        "Smriti Singh",
        "Aman Chadha",
        "Vinija Jain"
      ],
      "abstract": "Do large language models reason morally, or do they merely sound like they do? We investigate whether LLM responses to moral dilemmas exhibit genuine developmental progression through Kohlberg's stages of moral development, or whether alignment training instead produces reasoning-like outputs that superficially resemble mature moral judgment without the underlying developmental trajectory. Using an LLM-as-judge scoring pipeline validated across three judge models, we classify more than 600 responses from 13 LLMs spanning a range of architectures, parameter scales, and training regimes across six classical moral dilemmas, and conduct ten complementary analyses to characterize the nature and internal coherence of the resulting patterns. Our results reveal a striking inversion: responses overwhelmingly correspond to post-conventional reasoning (Stages 5-6) regardless of model size, architecture, or prompting strategy, the effective inverse of human developmental norms, where Stage 4 dominates. Most strikingly, a subset of models exhibit moral decoupling: systematic inconsistency between stated moral justification and action choice, a form of logical incoherence that persists across scale and prompting strategy and represents a direct reasoning consistency failure independent of rhetorical sophistication. Model scale carries a statistically significant but practically small effect; training type has no significant independent main effect; and models exhibit near-robotic cross-dilemma consistency producing logically indistinguishable responses across semantically distinct moral problems. We posit that these patterns constitute evidence for moral ventriloquism: the acquisition, through alignment training, of the rhetorical conventions of mature moral reasoning without the underlying developmental trajectory those conventions are meant to represent.",
      "published": "2026-03-23T11:43:49Z",
      "abstract_url": "http://arxiv.org/abs/2603.21854v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21854v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "On the Number of Conditional Independence Tests in Constraint-based Causal Discovery",
      "authors": [
        "Marc Franquesa Monés",
        "Jiaqi Zhang",
        "Caroline Uhler"
      ],
      "abstract": "Learning causal relations from observational data is a fundamental problem with wide-ranging applications across many fields. Constraint-based methods infer the underlying causal structure by performing conditional independence tests. However, existing algorithms such as the prominent PC algorithm need to perform a large number of independence tests, which in the worst case is exponential in the maximum degree of the causal graph. Despite extensive research, it remains unclear if there exist algorithms with better complexity without additional assumptions. Here, we establish an algorithm that achieves a better complexity of $p^{\\mathcal{O}(s)}$ tests, where $p$ is the number of nodes in the graph and $s$ denotes the maximum undirected clique size of the underlying essential graph. Complementing this result, we prove that any constraint-based algorithm must perform at least $2^{Ω(s)}$ conditional independence tests, establishing that our proposed algorithm achieves exponent-optimality up to a logarithmic factor in terms of the number of conditional independence tests needed. Finally, we validate our theoretical findings through simulations, on semi-synthetic gene-expression data, and real-world data, demonstrating the efficiency of our algorithm compared to existing methods in terms of number of conditional independence tests needed.",
      "published": "2026-03-23T11:33:43Z",
      "abstract_url": "http://arxiv.org/abs/2603.21844v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21844v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ME",
        "stat.ML"
      ]
    },
    {
      "title": "CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter",
      "authors": [
        "Hanyin Cheng",
        "Xingjian Wu",
        "Yang Shu",
        "Zhongwen Rao",
        "Lujia Pan",
        "Bin Yang",
        "Chenjuan Guo"
      ],
      "abstract": "Most existing Time Series Foundation Models (TSFMs) use channel independent modeling and focus on capturing and generalizing temporal dependencies, while neglecting the correlations among channels or overlooking the different aspects of correlations. However, these correlations play a vital role in Multivariate time series forecasting. To address this, we propose a CoRrelation-aware Adapter (CoRA), a lightweight plug-and-play method that requires only fine-tuning with TSFMs and is able to capture different types of correlations, so as to improve forecast performance. Specifically, to reduce complexity, we innovatively decompose the correlation matrix into low-rank Time-Varying and Time-Invariant components. For the Time-Varying component, we further design learnable polynomials to learn dynamic correlations by capturing trends or periodic patterns. To learn positive and negative correlations that appear only among some channels, we introduce a novel dual contrastive learning method that identifies correlations through projection layers, regulated by a Heterogeneous-Partial contrastive loss during training, without introducing additional complexity in the inference stage. Extensive experiments on 10 real-world datasets demonstrate that CoRA can improve TSFMs in multivariate forecasting performance.",
      "published": "2026-03-23T11:13:02Z",
      "abstract_url": "http://arxiv.org/abs/2603.21828v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21828v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Ctrl-A: Control-Driven Online Data Augmentation",
      "authors": [
        "Jesper B. Christensen",
        "Ciaran Bench",
        "Spencer A. Thomas",
        "Hüsnü Aslan",
        "David Balslev-Harder",
        "Nadia A. S. Smith",
        "Alessandra Manzin"
      ],
      "abstract": "We introduce ControlAugment (Ctrl-A), an automated data augmentation algorithm for image-vision tasks, which incorporates principles from control theory for online adjustment of augmentation strength distributions during model training. Ctrl-A eliminates the need for initialization of individual augmentation strengths. Instead, augmentation strength distributions are dynamically, and individually, adapted during training based on a control-loop architecture and what we define as relative operation response curves. Using an operation-dependent update procedure provides Ctrl-A with the potential to suppress augmentation styles that negatively impact model performance, alleviating the need for manually engineering augmentation policies for new image-vision tasks. Experiments on the CIFAR-10, CIFAR-100, and SVHN-core benchmark datasets using the common WideResNet-28-10 architecture demonstrate that Ctrl-A is highly competitive with existing state-of-the-art data augmentation strategies.",
      "published": "2026-03-23T11:03:02Z",
      "abstract_url": "http://arxiv.org/abs/2603.21819v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21819v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors",
      "authors": [
        "Yuze Qin",
        "Qingyong Li",
        "Zhiqing Guo",
        "Wen Wang",
        "Yan Liu",
        "Yangli-ao Geng"
      ],
      "abstract": "Precipitation nowcasting is critical for disaster mitigation and aviation safety. However, radar-only models frequently suffer from a lack of large-scale atmospheric context, leading to performance degradation at longer lead times. While integrating meteorological variables predicted by weather foundation models offers a potential remedy, existing architectures fail to reconcile the profound representational heterogeneities between radar imagery and meteorological data. To bridge this gap, we propose PW-FouCast, a novel frequency-domain fusion framework that leverages Pangu-Weather forecasts as spectral priors within a Fourier-based backbone. Our architecture introduces three key innovations: (i) Pangu-Weather-guided Frequency Modulation to align spectral magnitudes and phases with meteorological priors; (ii) Frequency Memory to correct phase discrepancies and preserve temporal evolution; and (iii) Inverted Frequency Attention to reconstruct high-frequency details typically lost in spectral filtering. Extensive experiments on the SEVIR and MeteoNet benchmarks demonstrate that PW-FouCast achieves state-of-the-art performance, effectively extending the reliable forecast horizon while maintaining structural fidelity. Our code is available at https://github.com/Onemissed/PW-FouCast.",
      "published": "2026-03-23T10:05:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.21768v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21768v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "The Presupposition Problem in Representation Genesis",
      "authors": [
        "Yiling Wu"
      ],
      "abstract": "Large language models are the first systems to achieve high cognitive performance without clearly undergoing representation genesis: the transition from a non-representing physical system to one whose states guide behavior in a content-sensitive way. Prior cognitive systems had already made this transition before we could examine it, and philosophy of mind treated genesis as a background condition rather than an explanatory target. LLMs provide a case that does not clearly involve this transition, making the genesis question newly urgent: if genesis did not occur, which cognitive capacities are affected, and why? We currently lack the conceptual resources to answer this. The reason, this paper argues, is structural. Major frameworks in philosophy of mind, including the Language of Thought hypothesis, teleosemantics, predictive processing, enactivism, and genetic phenomenology, share a common feature when applied to the genesis question: at some explanatory step, each deploys concepts whose explanatory purchase depends on the system already being organized as a representer. This pattern, which we call the Representation Presupposition structure, generates systematic explanatory deferral. Attempts to explain the first acquisition of content-manipulable representation within the existing categorical vocabulary import resources from the representational side of the transition itself. We call this the Representation Regress. The paper offers a conceptual diagnosis rather than a new theory, establishing the structure of the problem and deriving two minimum adequacy conditions for any account that avoids this pattern. LLMs make the absence of such a theory consequential rather than merely theoretical.",
      "published": "2026-03-23T09:37:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.21745v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21745v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning",
      "authors": [
        "Andreas Sauter",
        "Yuyue Zhao",
        "Jacopo Urbani",
        "Wenxiang Hu",
        "Zaiqiao Meng",
        "Lun Zhou",
        "Xiaohui Yan",
        "Yougang Lyu"
      ],
      "abstract": "Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often rely on rubric-based scalar rewards that provide global quality scores but lack actionable granularity. Conversely, language-based refinement methods are typically confined to inference-time prompting, targeting models that are not explicitly optimized to internalize such critiques. To bridge this gap, we propose \\textbf{EvoIdeator}, a framework that facilitates the evolution of scientific ideas by aligning the RL training objective with \\textbf{checklist-grounded feedback}. EvoIdeator leverages a structured judge model to generate two synergistic signals: (1) \\emph{lexicographic rewards} for multi-dimensional optimization, and (2) \\emph{fine-grained language feedback} that offers span-level critiques regarding grounding, feasibility, and methodological rigor. By integrating these signals into the RL loop, we condition the policy to systematically utilize precise feedback during both optimization and inference. Extensive experiments demonstrate that EvoIdeator, built on Qwen3-4B, significantly outperforms much larger frontier models across key scientific metrics. Crucially, the learned policy exhibits strong generalization to diverse external feedback sources without further fine-tuning, offering a scalable and rigorous path toward self-refining autonomous ideation.",
      "published": "2026-03-23T09:15:26Z",
      "abstract_url": "http://arxiv.org/abs/2603.21728v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21728v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning",
      "authors": [
        "Shuo Wang",
        "Ziyu Chen",
        "Ming Tang"
      ],
      "abstract": "Fine-tuning large language models (LLMs) with backpropagation achieves high performance but incurs substantial memory overhead, limiting scalability on resource-constrained hardware. Zeroth-order (ZO) optimization provides a memory-efficient alternative by relying solely on forward passes, yet it typically suffers from slow or unstable convergence due to high-variance gradient estimates. Sparse ZO updates partially address this issue by perturbing only a subset of parameters, but their effectiveness hinges on selecting informative parameters, which is challenging in ZO optimization because each query yields only scalar feedback. We propose \\textbf{Adaptive Curvature-Guided Sparse Zeroth-Order Optimization (CurvZO)}, which tracks curvature signals online from scalar ZO feedback and leverages these signals to construct a parameter-wise sampling distribution for selecting coordinates at each update, reducing the variance of the sparse ZO gradient estimator. Moreover, CurvZO dynamically adapts the perturbation budget to the evolving curvature signal distribution, yielding sparse ZO updates that remain both focused and sufficiently exploratory. Extensive experiments on OPT and Llama across diverse NLP tasks show that CurvZO consistently improves fine-tuning performance and reduces training time over ZO baselines. It improves accuracy by up to 4.4 points and achieves up to a $2\\times$ speedup, while preserving memory efficiency.",
      "published": "2026-03-23T09:13:45Z",
      "abstract_url": "http://arxiv.org/abs/2603.21725v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21725v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "FISformer: Replacing Self-Attention with a Fuzzy Inference System in Transformer Models for Time Series Forecasting",
      "authors": [
        "Bulent Haznedar",
        "Levent Karacan"
      ],
      "abstract": "Transformers have achieved remarkable progress in time series forecasting, yet their reliance on deterministic dot-product attention limits their capacity to model uncertainty and nonlinear dependencies across multivariate temporal dimensions. To address this limitation, we propose FISFormer, a Fuzzy Inference System-driven Transformer that replaces conventional attention with a FIS Interaction mechanism. In this framework, each query-key pair undergoes a fuzzy inference process for every feature dimension, where learnable membership functions and rule-based reasoning estimate token-wise relational strengths. These FIS-derived interaction weights capture uncertainty and provide interpretable, continuous mappings between tokens. A softmax operation is applied along the token axis to normalize these weights, which are then combined with the corresponding value features through element-wise multiplication to yield the final context-enhanced token representations. This design fuses the interpretability and uncertainty modeling of fuzzy logic with the representational power of Transformers. Extensive experiments on multiple benchmark datasets demonstrate that FISFormer achieves superior forecasting accuracy, noise robustness, and interpretability compared to state-of-the-art Transformer variants, establishing fuzzy inference as an effective alternative to conventional attention mechanisms.",
      "published": "2026-03-23T09:12:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.21724v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21724v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SemEval-2026 Task 12: Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models",
      "authors": [
        "Pengfei Cao",
        "Mingxuan Yang",
        "Yubo Chen",
        "Chenlong Zhang",
        "Mingxuan Liu",
        "Kang Liu",
        "Jun Zhao"
      ],
      "abstract": "Understanding why real-world events occur is important for both natural language processing and practical decision-making, yet direct-cause inference remains underexplored in evidence-rich settings. To address this gap, we organized SemEval-2026 Task 12: Abductive Event Reasoning (AER).\\footnote{The task data is available at https://github.com/sooo66/semeval2026-task12-dataset.git} The task asks systems to identify the most plausible direct cause of a target event from supporting evidence. We formulate AER as an evidence-grounded multiple-choice benchmark that captures key challenges of real-world causal reasoning, including distributed evidence, indirect background factors, and semantically related but non-causal distractors. The shared task attracted 122 participants and received 518 submissions. This paper presents the task formulation, dataset construction pipeline, evaluation setup, and system results. AER provides a focused benchmark for abductive reasoning over real-world events and highlights challenges for future work on causal reasoning and multi-document understanding.",
      "published": "2026-03-23T09:06:24Z",
      "abstract_url": "http://arxiv.org/abs/2603.21720v1",
      "pdf_url": "https://arxiv.org/pdf/2603.21720v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    }
  ]
};
