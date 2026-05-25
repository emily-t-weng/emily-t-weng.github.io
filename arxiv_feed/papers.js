const PAPERS_DATA = {
  "last_updated": "2026-05-25 04:40:09 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws",
      "authors": [
        "Xu Ouyang",
        "Deyi Liu",
        "Yuhang Cai",
        "Jing Liu",
        "Yuan Yang",
        "Chen Zheng",
        "Thomas Hartvigsen",
        "Yiyuan Ma"
      ],
      "abstract": "Existing scaling laws for Large Language Models (LLMs), predominantly monotonic power laws, fail to explain emerging non-monotonic phenomena such as catastrophic overtraining and quantization-induced degradation, where performance deteriorates despite increased compute. We propose the Shannon Scaling Law, a unified theoretical framework that models LLM training as information transmission over a noisy channel, grounded in the Shannon-Hartley theorem. By mapping model parameters to channel bandwidth and training tokens to signal power, our formulation explicitly captures the interaction between learning signal and intrinsic noise. This perspective reveals a fundamental Shannon capacity for LLMs: scaling model size or data without preserving a sufficient signal-to-noise ratio (SNR) inevitably amplifies noise, inducing a transition from monotonic improvement to U-shaped performance degradation. We validate our theory through experiments on Pythia and OLMo2 under perturbations, including Gaussian noise, quantization and supervised fine-tuning on math, QA and code tasks. The Shannon Scaling Law consistently outperforms classical scaling laws and recent perturbation-aware laws, achieving strong $R^2$ scores and accurately capturing loss basins missed by prior approaches. It also extrapolates: fitted on $\\leq$6.9B Pythia models with $\\leq$180B tokens, it predicts the unseen 12B model up to 307B tokens at pooled $R^2{=}0.847$, while monotonic baselines collapse.",
      "published": "2026-05-22T17:59:38Z",
      "abstract_url": "http://arxiv.org/abs/2605.23901v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23901v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.IT"
      ]
    },
    {
      "title": "ETCHR: Editing To Clarify and Harness Reasoning",
      "authors": [
        "Beichen Zhang",
        "Yuhong Liu",
        "Jinsong Li",
        "Yuhang Zang",
        "Jiaqi Wang",
        "Dahua Lin"
      ],
      "abstract": "Multimodal Large Language Models have advanced visual reasoning, yet a purely textual chain of thought remains a bottleneck for questions that require fine-grained focus or view transformations. The ''think with images'' paradigm narrows this gap, but existing approaches are either constrained by fixed predefined toolkits or produce noisy intermediate images from unified multimodal methods. We pursue a third option: using a dedicated image editing model and decouple it with an understanding model. However, off-the-shelf image editors fail as reasoning assistants with two complementary gaps: a language-side gap, where editors trained as passive instruction-followers cannot map an abstract question to an appropriate visual transformation, and a generation-side gap, where edit correctness degrades as reasoning depth grows. Guided by this analysis, we introduce ETCHR (Editing To Clarify and Harness Reasoning), a question-conditioned, reasoning-aware image editor decoupled from the downstream understanding model and trained with a two-stage recipe targeted at the two gaps: Reasoning Imitation via supervised fine-tuning on edit trajectories, followed by Reasoning Enhancement with VLM-derived rewards for edit correctness and downstream reasoning accuracy. Since the editor is decoupled, ETCHR plugs into different open- and closed-source MLLMs in a training-free manner. Across five task families (fine-grained perception, chart understanding, logic reasoning, jigsaw restoration, and 3D understanding), ETCHR raises average Pass@1 from 55.95 to 60.77 (+4.82) with Qwen3-VL-8B, from 65.08 to 70.55 (+5.47) with Gemini-3.1-Flash-Lite, and from 76.55 to 81.16 (+4.61) with the 1T-parameter MoE model Kimi K2.5.",
      "published": "2026-05-22T17:58:28Z",
      "abstract_url": "http://arxiv.org/abs/2605.23897v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23897v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Good Token Hunting: A Hitchhiker's Guide to Token Selection for Visual Geometry Transformers",
      "authors": [
        "Shuhong Zheng",
        "Michael Oechsle",
        "Erik Sandström",
        "Marie-Julie Rakotosaona",
        "Federico Tombari",
        "Igor Gilitschenski"
      ],
      "abstract": "Visual geometry transformers have become powerful architectures for multi-view 3D reconstruction, enabling joint prediction of multiple 3D attributes in a feed-forward manner. However, their computational cost grows quadratically with the input sequence length due to the global attention layers inside these models. This limits both their scalability and efficiency. In this work, we address this challenge with a simple yet general strategy: restricting the number of key/value tokens that each query interacts with during global attention. To achieve effective token selection, we introduce a two-stage framework. First, an inter-frame selection step operates at the frame level to identify frames that should be preserved. Second, an intra-frame selection step further discards more redundant tokens within the selected frames. Our analysis highlights the advantage of a diversity-based strategy for inter-frame selection, which ensures broad coverage of the scene. For intra-frame selection, we show that layer-aware sparsification is necessary, with the selection process guided by the entropy of the global attention pattern. Our approach offers a superior speed-accuracy trade-off compared to existing solutions. Extensive experiments show that it accelerates visual geometry transformers by over 85% for scenes with 500 images while maintaining, or even improving, baseline performance, which hints that how our token selection strategy can play a crucial role in future applications of visual geometry transformers. Our project website is available at https://zsh2000.github.io/good-token-hunting.github.io.",
      "published": "2026-05-22T17:55:13Z",
      "abstract_url": "http://arxiv.org/abs/2605.23892v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23892v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "CHRONOS: Temporally-Aware Multi-Agent Coordination for Evolving Data Marketplaces",
      "authors": [
        "Joydeep Chandra"
      ],
      "abstract": "Temporal knowledge-graph data marketplaces face three coupled failures in static designs: stale hybrid index shortcuts reduce recall as edges evolve, stationary Shapley pricing misattributes value after distribution shifts, and uncoordinated agents over-consume a shared differential-privacy budget. We present CHRONOS, a three-layer architecture providing a unified treatment of these challenges with explicit public and private separation. Layer one applies neural-ODE temporal decay to shortcut edges, providing a per-query expected recall-loss bound of Big-O of Pq lambda delta t, with a monotone-envelope guarantee reducing bound looseness to 1.8 to 3.2 times observed loss. Layer two conditions Shapley valuation on detected changepoints and provides finite-sample error guarantees under noise. Layer three uses EXP3-IX to achieve Big-O of the square root of T log T regret while enforcing epsilon and delta differential privacy via moments accounting. CHRONOS releases a privatized affinity matrix per epoch using the Gaussian mechanism; all retrieval and ranking are post-processing, incurring no extra privacy cost. We provide multi-epoch settlement, scalability analysis for 500 sellers, and comparisons against accelerated baselines. Across four benchmarks, CHRONOS shows 0.937 recall at ten, 2.74 queries per second, 161 ms latency, and total epsilon of 4.25 at delta of 10 to the power of negative 6 under zCDP composition. These results indicate a competitive operating point. A limitation is that at this privacy level, released valuations remain noise-dominated; utility derives primarily from public index routing and adaptive scheduling driven by low-sensitivity statistics.",
      "published": "2026-05-22T17:47:45Z",
      "abstract_url": "http://arxiv.org/abs/2605.23887v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23887v1",
      "categories": [
        "cs.DB",
        "cs.AI",
        "cs.CR",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs",
      "authors": [
        "Rim Assouel",
        "Amir Bar",
        "Michal Drozdzal",
        "Adriana Romero-Soriano"
      ],
      "abstract": "Despite remarkable progress in Multimodal Large Language Models (MLLMs), these models still struggle with fine-grained understanding tasks. In this work, we propose Procedurally Generated Tasks (PGT), a simple data-driven framework that serves a dual purpose: inducing fine-grained visual understanding and acting as a low-cost diagnostic tool to identify the source of perception failures. By overlaying unambiguous geometric primitives on images, PGT generate additional dense supervision that disentangles visual grounding capability from semantic priors. Extensive experiments on relational, quantitative, and 3D/depth understanding benchmarks show that PGT yields remarkable gains across diverse architectures. Instruction tuning MLLMs on LLaVA-v1.5-Instruct augmented with PGT data results in improvements of up to +20% on the What'sUp benchmark and +13.3% on CV-Bench-2D, while maintaining general perception capabilities. Moreover, finetuning state-of-the-art MLLMs on PGT data leads to boosts of up to +5.5% on What'sUp and +8.3% on CV-Bench-2D. These findings demonstrate that PGT effectively address the bottleneck of fine-grained perception, revealing that many spatial reasoning deficits stem from inadequate supervision signals rather than inherent architectural or resolution limitations.",
      "published": "2026-05-22T17:45:01Z",
      "abstract_url": "http://arxiv.org/abs/2605.23883v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23883v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Human Decision-Making with Persuasive and Narrative LLM Explanations",
      "authors": [
        "Laura R. Marusich",
        "Mary Grace Kozuch Dhooghe",
        "Jonathan Z. Bakdash",
        "Murat Kantarcioglu"
      ],
      "abstract": "Large language models (LLMs) have the potential to aid and improve human decision-making in classification tasks, not only by providing fairly accurate predictions, but also in their ability to generate cogent narrative explanations of those predictions. Prior work has demonstrated that people generally find AI narrative explanations to be understandable, trustworthy, and convincing for changing beliefs and opinions; however, less is known about the impact of narrative explanations on objective human decision-making performance. Here we conduct a large-scale human behavioral experiment to evaluate decision-making performance with LLM-generated narrative explanations of varying persuasiveness. We found the degree of persuasiveness, or lack thereof, for LLM-based explanations did not meaningfully impact decision accuracy over a simple AI prediction alone, in agreement with typical results with explainable AI based on feature importance. We found evidence that narratives increased reliance on AI, but both when the AI prediction was correct and incorrect. Exploratory analyses also indicated that the more persuasive narratives may have had a detrimental effect on decision response times and the ability to discriminate between a correct and incorrect AI prediction. Overall, this work indicates that including narrative explanations with AI predictions may involve tradeoffs for decision-making performance, and more work is needed to determine how and when narrative explanations impact human decision-making.",
      "published": "2026-05-22T17:25:02Z",
      "abstract_url": "http://arxiv.org/abs/2605.23867v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23867v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "Leveraging Foundation Models for Causal Generative Modeling",
      "authors": [
        "Aneesh Komanduri",
        "Xintao Wu"
      ],
      "abstract": "Causal generative modeling is essential for developing reliable and transparent AI systems capable of counterfactual reasoning. While existing approaches focus on integrating causal constraints during the training of generative models, they often lack a unified framework to leverage the zero-shot reasoning capabilities of pretrained foundation models. We introduce FM-CGM, a modular framework for end-to-end visual causal reasoning using pretrained foundation models. FM-CGM formalizes the causal pipeline through three core components: a concept extractor, a concept manipulator, and a counterfactual generator. By leveraging a large reasoning model for causal inference and a text-to-image diffusion model for generation, our approach enables zero-shot causal discovery, intervention, and counterfactual generation. We then develop Causal Semantic Guidance (CSG), a cross-attention-based mechanism that ensures semantic interventions propagate to descendant concepts while preserving invariant regions. We empirically show that our approach can identify plausible causal structures and is suitable for faithful counterfactual image generation.",
      "published": "2026-05-22T17:20:17Z",
      "abstract_url": "http://arxiv.org/abs/2605.23861v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23861v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "It's the humans, not the data: Geopolitical bias in LLMs originates in post-training, amplified by the language of the prompt",
      "authors": [
        "Stuart Bladon",
        "Brinnae Bent"
      ],
      "abstract": "It has generally been assumed that geopolitical bias in language models originates from the training data used during the pre-training phase. We tested seven open-weight LLM pairs consisting of the base model (pre-training only) and the chat model (pre-training and post-training) from seven labs on a paired-scenario forced-choice probe over 28 country pairs in English, French, and Chinese, and found that geopolitical bias originates in post-training rather than in pre-training. Across seven AI labs, six showed shifts in the direction associated with the country or region of the model developer after post-training. This shift is strongest in Alibaba's Qwen 2.5: while the base is neutral on China-favourability (-0.15 log-odds, p=0.15), the post-trained chat variant is at +2.91 (p<10^-4), an 18x shift in odds. We also observe shifts in biases toward other countries across all models. Additionally, the magnitude of this shift depends on the language used to prompt the model: the French-made Mistral becomes pro-France only under French prompting (FR-EN shift +1.91, p<10^-4). These findings suggest that geopolitical preferences in language models are not simply inherited from large-scale internet data but are actively shaped during post-training, highlighting the need for greater transparency, auditing, and oversight of alignment processes that influence how models represent nations, cultures, and political perspectives.",
      "published": "2026-05-22T16:29:02Z",
      "abstract_url": "http://arxiv.org/abs/2605.23825v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23825v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Binary Edits Robust Multimodal Knowledge Editing with Adversarial Subspace Alignment",
      "authors": [
        "Haoyuan Wang",
        "Xiaohao Liu",
        "Jiajie Su",
        "Jianmao Xiao",
        "Chaochao Chen"
      ],
      "abstract": "Multimodal large language models (MLLMs) need efficient mechanisms to update knowledge without degrading existing capabilities. While intrinsic multimodal knowledge editing achieves strong reliability and locality, it often exhibits limited generality, failing to propagate edits across semantically equivalent visual and linguistic variations. This issue arises from the lack of explicit semantic supervision, rigid editing scopes, and biased anchoring to individual samples in high-dimensional multimodal spaces. We address robust intrinsic multimodal knowledge editing by explicitly targeting generalization. We formalize robustness through knowledge units that group semantically equivalent multimodal inputs and define generality as consistent predictions within each unit. To expose fragile semantic regions, we introduce Latent Adversarial Robustification (LAR), which generates adversarial yet semantically coherent variants in the joint latent space. We further propose Rank-Constrained Subspace Learning (RCSL), enforcing low-rank alignment of adversarial representations at the edit layer via a singular value-based objective. Extensive analysis demonstrates the effectiveness of ASAM empirically.",
      "published": "2026-05-22T15:46:10Z",
      "abstract_url": "http://arxiv.org/abs/2605.23780v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23780v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "MemAudit: Post-hoc Auditing of Poisoned Agent Memory via Causal Attribution and Structural Anomaly Detection",
      "authors": [
        "Zhewen Tan",
        "Yilun Yao",
        "Huiyan Jin",
        "Wenhan Yu",
        "Guoan Wang",
        "Mengyuan Fan",
        "liang lu",
        "Feng Liu",
        "Xiangzheng Zhang",
        "Duohe Ma",
        "Tong Yang",
        "Lin Sun"
      ],
      "abstract": "Large language model agents increasingly rely on persistent memory to store past interactions, retrieve relevant demonstrations, and improve long-horizon task execution. However, this memory mechanism also creates a practical security vulnerability: an adversarial user may inject malicious records into the agent's memory through ordinary interaction, and these records can later be retrieved to steer the agent's reasoning and actions. Existing defenses primarily focus on online intervention, such as prompt filtering or output blocking, but they do not address the post-hoc question of which stored memories are responsible after harmful behavior has already been observed. We propose \\textbf{MemAudit}, a post-hoc causal memory auditing framework for memory-augmented LLM agents. The framework combines two complementary signals: (1) a counterfactual memory influence score that measures each memory's causal contribution to harmful outputs, and (2) a memory consistency graph that identifies structurally anomalous memories within the broader memory store. We evaluate MemAudit against MINJA, a query-only memory injection attack in which malicious records are generated and stored through normal agent interactions rather than direct memory-bank modification. Across both QA and reasoning-agent settings, MemAudit substantially reduces attack success rates under realistic post-hoc auditing scenarios. The results show that QA attack success is reduced from $70\\%$ to $0\\%$, while RAP attack success drops from $83.3\\%$ to $0\\%$.",
      "published": "2026-05-22T15:03:13Z",
      "abstract_url": "http://arxiv.org/abs/2605.23723v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23723v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "OnePred: Next-Query Prediction via Recursive Intent Memory in Multi-Turn Conversations",
      "authors": [
        "Jiangwang Chen",
        "Bowen Zhang",
        "Zixin Song",
        "Jiazheng Kang",
        "Xiao Yang",
        "Da Zhu",
        "Guanjun Jiang"
      ],
      "abstract": "Although large language model (LLM) conversational systems process millions of multi-turn dialogues daily, they remain fundamentally reactive: they respond only after the user types a query. A key step toward proactive interaction is next-query prediction, which anticipates the user's subsequent query based solely on the preceding dialogue. Progress on this task is hindered by the lack of dedicated benchmarks and a fundamental efficiency--quality trade-off: naively concatenating full dialogue history incurs linearly growing token consumption, while truncating to the latest turn discards crucial cross-turn context. Our key insight is that accurate prediction does not require re-reading raw history; it suffices to track the user's evolving intent trajectory across topics, unresolved needs, and interest shifts. We propose OnePred, which maintains a recursively updated memory as its sole cross-turn context, bounding the per-turn cost independently of conversation length. We train the model via a two-stage reinforcement learning pipeline that first teaches what to predict, then what to compress, shaping the memory into a prediction-oriented intent chain. To establish a rigorous testbed, we introduce NQP-Bench, spanning three diverse subsets. Experiments demonstrate that OnePred reduces per-turn token consumption by up to 22$\\times$ compared to full-history inputs while consistently exceeding all baselines in prediction quality, with larger gains on longer conversations. Our code is publicly available at https://github.com/ZBWpro/OnePred.",
      "published": "2026-05-22T14:16:21Z",
      "abstract_url": "http://arxiv.org/abs/2605.23668v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23668v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception",
      "authors": [
        "Liupeng Li",
        "Haoqian Kang",
        "Zhenyu Lu",
        "Jinpeng Wang",
        "Bin Chen",
        "Ke Chen",
        "Yaowei Wang"
      ],
      "abstract": "High-resolution (HR) image perception presents a key bottleneck for multimodal large language models (MLLMs). While visual search offers a promising solution, existing methods struggle with the trade-off between coverage and efficiency. Visual expert-assisted search is efficient but prone to blind spots when proposals fail, whereas scan-based search guarantees coverage at the cost of computational redundancy and semantic fragmentation. To address this dilemma, we introduce CVSearch, a training-free adaptive framework that dynamically schedules search strategies via an Assess-then-Search workflow. Specifically, CVSearch first invokes expert-assisted search when global information is insufficient, and only triggers a novel semantic-aware scanning mechanism upon failure. Distinct from rigid grid partitioning, this efficient scanning paradigm incorporates Semantic Guided Adaptive Patching to decompose images into semantically consistent regions, effectively mitigating object fragmentation. Furthermore, we devise a Dynamic Bottom-Up Search strategy driven by a Visual Complexity prior to enable efficient and precise iterative exploration of local details. Extensive experiments on HR benchmarks demonstrate that CVSearch achieves state-of-the-art accuracy while substantially improving search efficiency. Code is released at https://github.com/liliupeng28/ICML26-CVSearch.",
      "published": "2026-05-22T14:07:44Z",
      "abstract_url": "http://arxiv.org/abs/2605.23655v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23655v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.MM"
      ]
    },
    {
      "title": "Learning Through Noise: Why Subliminal Learning Works and When It Fails",
      "authors": [
        "Vincent C. Brockers",
        "Roman D. Ventzke",
        "Valentin Neuhaus",
        "Belén Hidalgo-Ogalde",
        "Viola Priesemann"
      ],
      "abstract": "In the context of artificial neural networks, subliminal learning refers to the transfer of task-relevant knowledge or unintended biases from teacher to student models through distillation on task-unrelated input$\\unicode{x2013}$output pairs. Prior explanations tie this effect to shared or closely matched teacher$\\unicode{x2013}$student initialization. We show that a closely matched initialization is not necessary. Instead, subliminal learning is governed by compatible output heads. Using a controlled MNIST setting, we split outputs into an auxiliary head (for auxiliary, task-unrelated noise signals) and a class head (for classification) to demonstrate subliminal learning occurs$\\unicode{x2014}$even when we randomly initialize hidden layers and remove layers, add new layers, or change the architecture (MLP-to-CNN). Compatible auxiliary heads enable transfer of a recoverable teacher signal, bringing the student's representations closer to the teacher's. When the class heads remain compatible as well, students trained only on task-unrelated noise can approach, and in favorable regimes match, teacher-level task performance. Our setting enables us to develop a theory that explains the mechanism of subliminal learning and to derive upper bounds on when subliminal learning fails. Together, our results turn subliminal learning from a surprising transfer effect into a theoretically grounded mechanism with predictable limits.",
      "published": "2026-05-22T13:59:13Z",
      "abstract_url": "http://arxiv.org/abs/2605.23645v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23645v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Adversarial Vulnerability Under Temporal Concept Drift: A Longitudinal Study of Android Malware Detection",
      "authors": [
        "Ahmed Sabbah",
        "Mohammed Kharma",
        "Radi Jarrar",
        "Samer Zein",
        "David Mohaisen"
      ],
      "abstract": "We present a longitudinal, drift-aware evaluation of adversarial robustness across more than a decade of Android applications using static and dynamic feature representations extracted from emulator and real-device executions. The dataset is organized into yearly slices and evaluated under three deployment protocols that emulate realistic learning scenarios: (1) same-year training and testing, (2) cross-year deployment without model updates, and (3) expanding-window retraining with cumulative historical data. Across multiple classifier families, adversarial examples are generated using FGSM and SPSA under feasibility constraints. We measure clean performance, Adversarial Accuracy (AA), Attack Success Rate (ASR), and introduce temporal linkage metrics -- RobustDrop, $Δ$ASR, and Adversarial Amplification Factor (AAF) -- to quantify the relationship between distribution shift and robustness degradation.nResults show that temporal separation is associated with reduced adversarial robustness under the evaluated transfer-based feature-space setting. As the train-test gap increases, clean accuracy and adversarial accuracy decline, while attack success exhibits configuration-dependent increases, particularly under FGSM perturbations and static features. Expanding-window retraining mitigates, but does not eliminate, robustness loss under continued distributional evolution. These findings indicate that temporal drift should be considered when assessing the long-term robustness of intelligent detection systems under evolving data distributions and highlight the need for drift-aware robustness assessment frameworks in long-lived adversarial environments.",
      "published": "2026-05-22T13:29:45Z",
      "abstract_url": "http://arxiv.org/abs/2605.23623v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23623v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DiLaDiff: Distilled Latent-Augmented Diffusion for Language Modeling",
      "authors": [
        "Jean-Marie Lemercier",
        "Tomas Geffner",
        "Karsten Kreis",
        "Morteza Mardani",
        "Arash Vahdat",
        "Ante Jukić"
      ],
      "abstract": "Diffusion language models intrinsically fail to capture correlations between decoded tokens, which leads to a harsh trade-off between sampling quality and throughput. To solve this issue, we propose DiLaDiff, a variant of masked diffusion language models with three components: (1) a continuous latent space with semantic capabilities, learned by an auto-encoder fine-tuned from an existing masked diffusion language model; (2) a latent diffusion model learning the prior over the encoder distribution; (3) a consistency model distilling the learned prior into a few-step latent generative model. We show that, even without distillation, our latent-guided diffusion model outperforms the masked diffusion baseline while significantly accelerating inference. Consistency distillation further lowers the computational overhead of continuous diffusion, such that the latent is generated in negligible time compared to discrete decoding.",
      "published": "2026-05-22T13:15:59Z",
      "abstract_url": "http://arxiv.org/abs/2605.23605v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23605v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Preisach Attention: A Hysteretic Model of Sequential Memory",
      "authors": [
        "Piotr Frydrych"
      ],
      "abstract": "We introduce the Preisach Attention Layer (PAL), a novel sequence modelling architecture grounded in the classical Preisach hysteresis operator from mathematical physics. PAL replaces the softmax attention mechanism with a binary relay operator parameterised by learned activation and deactivation thresholds, maintaining a stack of local extrema as its internal state. A single-layer PAL-Transformer with O(1) depth is Turing-complete under arbitrary precision arithmetic, achievable through simulation of a two-stack pushdown automaton -- in contrast to the O(log n) depth required by standard hard-attention transformers. Second, we prove that the function classes computable by PAL and by the transformer are incomparable: PAL computes historical range statistics in O(1) layers that require O(log n) layers for transformers, while transformers support random-access retrieval that PAL cannot perform without auxiliary state. The separating property is rate-independence -- PAL responds only to the sequence of local extrema, not to absolute token positions or temporal spacing. Third, we show that the extremum stack constitutes a minimal sufficient statistic of the input history for all rate-independent functionals, providing a formal analogue of the wiping property in classical hysteresis theory. PAL is thus an efficient architecture for tasks with long episodic memory and weak positional dependence, with O(n log n) total inference cost versus O(n^2) for standard attention.",
      "published": "2026-05-22T13:12:04Z",
      "abstract_url": "http://arxiv.org/abs/2605.23603v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23603v1",
      "categories": [
        "cs.LG",
        "cond-mat.dis-nn",
        "cs.AI",
        "cs.NE"
      ]
    },
    {
      "title": "Cost-Effective Model Evaluation with Meta-Learning",
      "authors": [
        "Trinh Pham",
        "Viet Huynh",
        "Hongzhi Yin",
        "Quoc Viet Hung Nguyen",
        "Thanh Tam Nguyen"
      ],
      "abstract": "The rapid growth of machine learning has produced an ever-expanding ecosystem of models, making it increasingly challenging to verify the reliability of newly released models on unseen, unlabeled data. Conventional evaluation pipelines depend on expensive annotation, repeated fine-tuning, or narrow assumptions that fail to transfer across model families. We present MetaEvaluator, a cost-effective, model-agnostic framework for rapid, label-free assessment of unseen models spanning diverse architectures and modalities. MetaEvaluator leverages meta-learning over a pool of reference models to obtain a transferable initialization, enabling accurate evaluation of new models while amortizing cost across the pool and removing the need for per-model retraining. To the best of our knowledge, this is the first model-agnostic framework capable of evaluating new models on entirely unlabeled datasets. Extensive experiments show that MetaEvaluator produces stable and accurate performance estimates at substantially reduced cost compared to conventional approaches, making scalable benchmarking of emerging models on unlabeled data practical.",
      "published": "2026-05-22T13:05:34Z",
      "abstract_url": "http://arxiv.org/abs/2605.23595v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23595v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "cs.ET",
        "cs.PF"
      ]
    },
    {
      "title": "HARNESS-LM: A Three-Phase Training Recipe for Harnessing SLMs in Sponsored Search Retrieval",
      "authors": [
        "Vipul Gupta",
        "Shikhar Mohan",
        "Lakshya Kumar",
        "Pranjal Chitale",
        "Nikit Begwani",
        "Amit Singh",
        "Manik Varma"
      ],
      "abstract": "In the competitive landscape of sponsored search, balancing retrieval quality with production latency is a critical challenge. While large retrieval models based on Small Language Models (SLMs) such as Qwen3-Embedding-4B/8B set strong upper bounds on public benchmarks, their deployment in high-throughput, latency-sensitive environments remains impractical. In this paper, we present HARNESS-LM (HLM), a three-phase training framework for transferring the capabilities of large-scale retrievers into compact, cost-efficient models. The approach comprises: (1) training a high-performance reference (\"teacher\") retriever by fine-tuning a billion-parameter-scale SLM; (2) aligning query representations via an L2 objective to distill knowledge into a sub-600M parameter student encoder; and (3) applying a final contrastive refinement stage to optimize the student for retrieval performance. We also present a comprehensive empirical study of key design choices, including alignment objectives, embedding dimensionality, model scale, architecture, and optimization strategies, to identify configurations that are most effective in production settings. On a real-world Bing Ads evaluation benchmark, HLM recovers over 98% of the reference retriever's precision across multiple settings, while delivering up to 27x lower online query-encoder latency and 20x higher throughput on NVIDIA A100 GPUs. Online A/B testing on Bing Ads further shows a +1% Revenue, +0.6% Impression, and +0.4% Click uplift over the current ensemble of retrievers running in production with the deployed 190M parameter model, clearly highlighting the practical efficacy of the HLM recipe in a real-world sponsored search setting.",
      "published": "2026-05-22T12:39:56Z",
      "abstract_url": "http://arxiv.org/abs/2605.23572v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23572v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Understanding Goal Generalisation in Sequential Reinforcement Learning",
      "authors": [
        "Jason Ross Brown",
        "Edward James Young"
      ],
      "abstract": "Reinforcement learning agents often exhibit unintended goal-directed behaviour outside their training distribution, but we currently lack a principled understanding of how such agents will generalise to novel environments based on their training history. We address this gap for agents trained sequentially on one or more tasks. We study over 100 sequential training pipelines, evaluating behaviour across over 250 out-of-distribution environments. We find that salient features drive generalisation, and that goals learnt early in training can persist and influence those acquired later. To explain these phenomena, we introduce latent policy gradients, a method that predicts what out-of-distribution behaviour a training pipeline will likely induce. Our method simulates the evolution of low-dimensional latent variables during training according to what would achieve high reward on the training objective with respect to a simple model of how the latent variables map to behaviour. It achieves strong predictive accuracy, generalises to unseen types of training pipeline, and is interpretable. Our findings demonstrate that while out-of-distribution RL agent behaviour is dependent on the whole training pipeline, this dependence has an underlying structure we can capture, laying groundwork for understanding goal generalisation from a developmental perspective.",
      "published": "2026-05-22T12:31:18Z",
      "abstract_url": "http://arxiv.org/abs/2605.23565v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23565v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for Whole-Slide Image VQA",
      "authors": [
        "Chunze Yang",
        "Qidong Liu",
        "Wenjie Zhao",
        "Yue Tang",
        "Jiusong Ge",
        "Di Zhang",
        "Jiashuai Liu",
        "Lei Wu",
        "Junbo Lu",
        "Ni Zhang",
        "Xian Wu",
        "Zeyu Gao",
        "Chen Li"
      ],
      "abstract": "Whole-slide image visual question answering (WSI-VQA) frames pathology as an extreme-context search problem: to answer a free-form clinical query, a system must first navigate a gigapixel slide under a strict inspection budget to locate sparse, high-resolution evidence. Existing approaches largely fall into two paradigms: i) supervised pathology multimodal large language models (MLLMs) and agents can absorb localization and reasoning into learned modules, but they often couple navigation to task-specific supervision and retraining, limiting their practicality; ii) training-free pathology agents avoid this cost by keeping core models frozen, but often follow a question-first design, constructing the initial candidate set mainly from query-conditioned relevance. This can miss decisive morphology that is not named in the question, and force heavier inference-time scaffolding. To address this challenge, we introduce PathNavigate, a training-free pathology agent built around a scan-search-readout routine. Before question matching, PathNavigate scans the current slide at low magnification with a shared online memory module over frozen pathology features, producing a slide-specific surprise field that marks an abnormal-region pool. It then applies question-conditioned PLIP relevance only within this pool to select high-magnification search targets. Finally, it extracts local high-magnification evidence and answers with a frozen perceptor-adjudicator stack, using the same online memory as slide-level context. Experiments on WSI-VQA and SlideBench-BCNB show that the proposed scan-search-readout design improves answer accuracy and yields more interpretable evidence-selection trajectories with higher efficiency.The code is available online.",
      "published": "2026-05-22T12:25:43Z",
      "abstract_url": "http://arxiv.org/abs/2605.23559v1",
      "pdf_url": "https://arxiv.org/pdf/2605.23559v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    }
  ]
};
