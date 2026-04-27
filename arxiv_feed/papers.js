const PAPERS_DATA = {
  "last_updated": "2026-04-27 03:49:03 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation",
      "authors": [
        "Rajinder Sandhu",
        "Di Mu",
        "Cheng Chang",
        "Md Shahriar Tasjid",
        "Himanshu Rai",
        "Maksims Volkovs",
        "Ga Wu"
      ],
      "abstract": "Dense vector retrieval is the practical backbone of Retrieval- Augmented Generation (RAG), but similarity search can suffer from precision limitations. Conversely, utility-based approaches leveraging LLM re-ranking often achieve superior performance but are computationally prohibitive and prone to noise inherent in perplexity estimation. We propose Utility-Aligned Embeddings (UAE), a framework designed to merge these advantages into a practical, high-performance retrieval method. We formulate retrieval as a distribution matching problem, training a bi-encoder to imitate a utility distribution derived from perplexity reduction using a Utility-Modulated InfoNCE objective. This approach injects graded utility signals directly into the embedding space without requiring test-time LLM inference. On the QASPER benchmark, UAE improves retrieval Recall@1 by 30.59%, MAP by 30.16% and Token F1 by 17.3% over the strong semantic baseline BGE-Base. Crucially, UAE is over 180x faster than the efficient LLM re-ranking methods preserving competitive performance, demonstrating that aligning retrieval with generative utility yields reliable contexts at scale.",
      "published": "2026-04-24T17:18:56Z",
      "abstract_url": "http://arxiv.org/abs/2604.22722v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22722v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Rethinking XAI Evaluation: A Human-Centered Audit of Shapley Benchmarks in High-Stakes Settings",
      "authors": [
        "Inês Oliveira e Silva",
        "Sérgio Jesus",
        "Iker Perez",
        "Rita P. Ribeiro",
        "Carlos Soares",
        "Hugo Ferreira",
        "Pedro Bizarro"
      ],
      "abstract": "Shapley values are a cornerstone of explainable AI, yet their proliferation into competing formulations has created a fragmented landscape with little consensus on practical deployment. While theoretical differences are well-documented, evaluation remains reliant on quantitative proxies whose alignment with human utility is unverified. In this work, we use a unified amortized framework to isolate semantic differences between eight Shapley variants under the low-latency constraints of operational risk workflows. We conduct a large-scale empirical evaluation across four risk datasets and a realistic fraud-detection environment involving professional analysts and 3,735 case reviews. Our results reveal a fundamental misalignment: standard quantitative metrics, such as sparsity and faithfulness, are decoupled from human-perceived clarity and decision utility. Furthermore, while no formulation improved objective analyst performance, explanations consistently increased decision confidence, signaling a critical risk of automation bias in high-stakes settings. These findings suggest that current evaluation proxies are insufficient for predicting downstream human impact, and we provide evidence-based guidance for selecting formulations and metrics in operational decision systems.",
      "published": "2026-04-24T15:38:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.22662v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22662v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "From Natural Language to Verified Code: Toward AI Assisted Problem-to-Code Generation with Dafny-Based Formal Verification",
      "authors": [
        "Md Erfan",
        "Md Kamal Hossain Chowdhury",
        "Ahmed Ryan",
        "Md Rayhanur Rahman"
      ],
      "abstract": "Large Language Models (LLMs) show promise in automated software engineering, yet their guarantee of correctness is frequently undermined by erroneous or hallucinated code. To enforce model honesty, formal verification requires LLMs to synthesize implementation logic alongside formal specifications that are subsequently proven correct by a mathematical verifier. However, the transition from informal natural language to precise formal specification remains an arduous task. Our work addresses this by providing the NaturalLanguage2VerifiedCode (NL2VC)-60 dataset: a collection of 60 complex algorithmic problems. We evaluate 11 randomly selected problem sets across seven open-weight LLMs using a tiered prompting strategy: contextless prompts, signature prompts providing structural anchors, and self-healing prompts utilizing iterative feedback from the Dafny verifier. To address vacuous verification, where models satisfy verifiers with trivial specifications, we integrate the uDebug platform to ensure functional validation. Our results show that while contextless prompting leads to near-universal failure, structural signatures and iterative self-healing facilitate a dramatic performance turnaround. Specifically, Gemma 4-31B achieved a 90.91\\% verification success rate, while GPT-OSS 120B rose from zero to 81.82\\% success with signature-guided feedback. These findings indicate that formal verification is now attainable for open-weight LLMs, which serve as effective apprentices for synthesizing complex annotations and facilitating high-assurance software development.",
      "published": "2026-04-24T14:28:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.22601v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22601v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking Math Reasoning Evaluation: A Robust LLM-as-a-Judge Framework Beyond Symbolic Rigidity",
      "authors": [
        "Erez Yosef",
        "Oron Anschel",
        "Shunit Haviv Hakimi",
        "Asaf Gendler",
        "Adam Botach",
        "Nimrod Berman",
        "Igor Kviatkovsky"
      ],
      "abstract": "Recent advancements in large language models have led to significant improvements across various tasks, including mathematical reasoning, which is used to assess models' intelligence in logical reasoning and problem-solving. Models are evaluated on mathematical reasoning benchmarks by verifying the correctness of the final answer against a ground truth answer. A common approach for this verification is based on symbolic mathematics comparison, which fails to generalize across diverse mathematical representations and solution formats. In this work, we offer a robust and flexible alternative to rule-based symbolic mathematics comparison. We propose an LLM-based evaluation framework for evaluating model-generated answers, enabling accurate evaluation across diverse mathematical representations and answer formats. We present failure cases of symbolic evaluation in two popular frameworks, Lighteval and SimpleRL, and compare them to our approach, demonstrating clear improvements over commonly used methods. Our framework enables more reliable evaluation and benchmarking, leading to more accurate performance monitoring, which is important for advancing mathematical problem-solving and intelligent systems.",
      "published": "2026-04-24T14:25:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.22597v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22597v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Learning Evidence Highlighting for Frozen LLMs",
      "authors": [
        "Shaoang Li",
        "Yanhang Shi",
        "Yufei Li",
        "Mingfu Liang",
        "Xiaohan Wei",
        "Yunchen Pu",
        "Fei Tian",
        "Chonglin Sun",
        "Frank Shyu",
        "Luke Simon",
        "Sandeep Pandey",
        "Xi Liu",
        "Jian Li"
      ],
      "abstract": "Large Language Models (LLMs) can reason well, yet often miss decisive evidence when it is buried in long, noisy contexts. We introduce HiLight, an Evidence Emphasis framework that decouples evidence selection from reasoning for frozen LLM solvers. HiLight avoids compressing or rewriting the input, which can discard or distort evidence, by training a lightweight Emphasis Actor to insert minimal highlight tags around pivotal spans in the unaltered context. A frozen Solver then performs downstream reasoning on the emphasized input. We cast highlighting as a weakly supervised decision-making problem and optimize the Actor with reinforcement learning using only the Solver's task reward, requiring no evidence labels and no access to or modification of the Solver. Across sequential recommendation and long-context question answering, HiLight consistently improves performance over strong prompt-based and automated prompt-optimization baselines. The learned emphasis policy transfers zero-shot to both smaller and larger unseen Solver families, including an API-based Solver, suggesting that the Actor captures genuine, reusable evidence structure rather than overfitting to a single backbone.",
      "published": "2026-04-24T13:57:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.22565v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22565v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy",
      "authors": [
        "Asim Ukaye",
        "Mubarak Abdu-Aguye",
        "Nurbek Tastan",
        "Karthik Nandakumar"
      ],
      "abstract": "Client contribution estimation in Federated Learning is necessary for identifying clients' importance and for providing fair rewards. Current methods often rely on server-side validation data or self-reported client information, which can compromise privacy or be susceptible to manipulation. We introduce a data-free signal based on the matrix von Neumann (spectral) entropy of the final-layer updates, which measures the diversity of the information contributed. We instantiate two practical schemes: (i) SpectralFed, which uses normalized entropy as aggregation weights, and (ii) SpectralFuse, which fuses entropy with class-specific alignment via a rank-adaptive Kalman filter for per-round stability. Across CIFAR-10/100 and the naturally partitioned FEMNIST and FedISIC benchmarks, entropy-derived scores show a consistently high correlation with standalone client accuracy under diverse non-IID regimes - without validation data or client metadata. We compare our results with data-free contribution estimation baselines and show that spectral entropy serves as a useful indicator of client contribution.",
      "published": "2026-04-24T13:55:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.22562v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22562v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "cs.DC"
      ]
    },
    {
      "title": "SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning",
      "authors": [
        "Jichao Wang",
        "Liuyang Bian",
        "Yufeng Zhou",
        "Han Xiao",
        "Yue Pan",
        "Guozhi Wang",
        "Hao Wang",
        "Zhaoxiong Wang",
        "Yafei Wen",
        "Xiaoxin Chen",
        "Shuai Ren",
        "Lingfang Zeng"
      ],
      "abstract": "As Multimodal Large Language Models (MLLMs) mature, GUI agents are evolving from static interactions to complex navigation. While Reinforcement Learning (RL) has emerged as a promising paradigm for training MLLM agents on dynamic GUI tasks, its effective application faces a dilemma. Standard Offline RL often relies on static step-level data, neglecting global trajectory semantics such as task completion and execution quality. Conversely, Online RL captures the long-term dynamics but suffers from high interaction costs and potential environmental instability. To bridge this gap, we propose SOLAR-RL (Semi-Online Long-horizon Assignment Reinforcement Learning). Instead of relying solely on expensive online interactions, our framework integrates global trajectory insights directly into the offline learning process. Specifically, we reconstruct diverse rollout candidates from static data, detect the first failure point using per-step validity signals, and retroactively assign dense step-level rewards with target-aligned shaping to reflect trajectory-level execution quality, effectively simulating online feedback without interaction costs. Extensive experiments demonstrate that SOLAR-RL significantly improves long-horizon task completion rates and robustness compared to strong baselines, offering a sample-efficient solution for autonomous GUI navigation.",
      "published": "2026-04-24T13:53:39Z",
      "abstract_url": "http://arxiv.org/abs/2604.22558v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22558v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Controllable Spoken Dialogue Generation: An LLM-Driven Grading System for K-12 Non-Native English Learners",
      "authors": [
        "Haidong Yuan",
        "Haokun Zhao",
        "Wanshi Xu",
        "Songjun Cao",
        "Qingyu Zhou",
        "Long Ma",
        "Hongjie Fan"
      ],
      "abstract": "Large language models (LLMs) often fail to meet the pedagogical needs of K-12 English learners in non-native contexts due to a proficiency mismatch. To address this widespread challenge, we introduce a proficiency-aligned framework that adapts LLM outputs to learner abilities, using China's national curriculum (CSE) as a representative case. Our framework enables precise control over lexical complexity through a four-tier grading system, supported by a comprehensive suite of new resources: graded vocabulary lists and a multi-turn dialogue corpus. Our core technical contribution is the \\textbf{DDPO} algorithm,Diversity Driven Policy Optimization, a multi-turn GRPO-based approach designed to preserve dialogue diversity while holistically optimizing dialogue quality. This method significantly outperforms conventional approaches, achieving low out-of-vocabulary rates and high diversity while enhancing conversational naturalness and pedagogical value. While grounded in the CSE, our framework is designed for flexibility and can be readily adapted to other educational standards. Our models, data, and code will all be open-sourced, providing a scalable platform for personalized English speaking practice that effectively addresses the unique challenges faced by K-12 learners in non-immersive environments.",
      "published": "2026-04-24T13:33:12Z",
      "abstract_url": "http://arxiv.org/abs/2604.22542v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22542v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "On the Properties of Feature Attribution for Supervised Contrastive Learning",
      "authors": [
        "Leonardo Arrighi",
        "Julia Eva Belloni",
        "Aurélie Gallet",
        "Ivan Gentile",
        "Matteo Lippi",
        "Marco Zullich"
      ],
      "abstract": "Most Neural Networks (NNs) for classification are trained using Cross-Entropy as a loss function. This approach requires the model to have an explicit classification layer. However, there exist alternative approaches, such as Contrastive Learning (CL). Instead of explicitly operating a classification, CL has the NN produce an embedding space where projections of similar data are pulled together, while projections of dissimilar data are pushed apart. In the case of Supervised CL (SCL), labels are adopted as similarity criteria, thus creating an embedding space where the projected data points are well-clustered. SCL provides crucial advantages over CE with regard to adversarial robustness and out-of-distribution detection, thus making it a more natural choice in safety-critical scenarios. In the present paper, we empirically show that NNs for image classification trained with SCL present higher-quality feature attribution explanations than CL with regard to faithfulness, complexity, and continuity. These results reinforce previous findings about CL-based approaches when targeting more trustworthy and transparent NNs and can guide practitioners in the selection of training objectives targeting not only accuracy, but also transparency of the models.",
      "published": "2026-04-24T13:32:43Z",
      "abstract_url": "http://arxiv.org/abs/2604.22540v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22540v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "FeatEHR-LLM: Leveraging Large Language Models for Feature Engineering in Electronic Health Records",
      "authors": [
        "Hojjat Karami",
        "David Atienza",
        "Jean-Philippe Thiran",
        "Anisoara Ionescu"
      ],
      "abstract": "Feature engineering for Electronic Health Records (EHR) is complicated by irregular observation intervals, variable measurement frequencies, and structural sparsity inherent to clinical time series. Existing automated methods either lack clinical domain awareness or assume clean, regularly sampled inputs, limiting their applicability to real-world EHR data. We present \\textbf{FeatEHR-LLM}, a framework that leverages Large Language Models (LLMs) to generate clinically meaningful tabular features from irregularly sampled EHR time series. To limit patient privacy exposure, the LLM operates exclusively on dataset schemas and task descriptions rather than raw patient records. A tool-augmented generation mechanism equips the LLM with specialized routines for querying irregular temporal data, enabling it to produce executable feature-extraction code that explicitly handles uneven observation patterns and informative sparsity. FeatEHR-LLM supports both univariate and multivariate feature generation through an iterative, validation-in-the-loop pipeline. Evaluated on eight clinical prediction tasks across four ICU datasets, our framework achieves the highest mean AUROC on 7 out of 8 tasks, with improvements of up to 6 percentage points over strong baselines. Code is available at github.com/hojjatkarami/FeatEHR-LLM.",
      "published": "2026-04-24T13:21:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.22534v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22534v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding",
      "authors": [
        "Lihao Zheng",
        "Zhenwei Shao",
        "Yu Zhou",
        "Yan Yang",
        "Xintian Shen",
        "Jiawei Chen",
        "Hao Ma",
        "Tao Wei"
      ],
      "abstract": "Although Multimodal Large Language Models (MLLMs) have advanced rapidly, they still face notable challenges in fine-grained multi-image understanding, often exhibiting spatial hallucination, attention leakage, and failures in object constancy. In addition, existing approaches typically rely on expensive human annotations or large-scale chain-of-thought (CoT) data generation. We propose Compositional Grounded Contrast (abbr. CGC), a low-cost full framework for boosting fine-grained multi-image understanding of MLLMs. Built on existing single-image grounding annotations, CGC constructs compositional multi-image training instances through Inter-Image Contrast and Intra-Image Contrast, which introduce semantically decoupled distractor contexts for cross-image discrimination and correlated cross-view samples for object constancy, respectively. CGC further introduces a Rule-Based Spatial Reward within the GRPO framework to improve source-image attribution, spatial alignment, and structured output validity under a Think-before-Grounding paradigm. Experiments show that CGC achieves state-of-the-art results on fine-grained multi-image benchmarks, including MIG-Bench and VLM2-Bench. The learned multi-image understanding capability also transfers to broader multimodal understanding and reasoning tasks, yielding consistent gains over the Qwen3-VL-8B base model on MathVista (+2.90), MuirBench (+2.88), MMStar (+1.93), MMMU (+1.77), and BLINK (+1.69).",
      "published": "2026-04-24T12:26:49Z",
      "abstract_url": "http://arxiv.org/abs/2604.22498v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22498v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Superminds Test: Actively Evaluating Collective Intelligence of Agent Society via Probing Agents",
      "authors": [
        "Xirui Li",
        "Ming Li",
        "Yunze Xiao",
        "Ryan Wong",
        "Dianqi Li",
        "Timothy Baldwin",
        "Tianyi Zhou"
      ],
      "abstract": "Collective intelligence refers to the ability of a group to achieve outcomes beyond what any individual member can accomplish alone. As large language model agents scale to populations of millions, a key question arises: Does collective intelligence emerge spontaneously from scale? We present the first empirical evaluation of this question in a large-scale autonomous agent society. Studying MoltBook, a platform hosting over two million agents, we introduce Superminds Test, a hierarchical framework that probes society-level intelligence using controlled Probing Agents across three tiers: joint reasoning, information synthesis, and basic interaction. Our experiments reveal a stark absence of collective intelligence. The society fails to outperform individual frontier models on complex reasoning tasks, rarely synthesizes distributed information, and often fails even trivial coordination tasks. Platform-wide analysis further shows that interactions remain shallow, with threads rarely extending beyond a single reply and most responses being generic or off-topic. These results suggest that collective intelligence does not emerge from scale alone. Instead, the dominant limitation of current agent societies is extremely sparse and shallow interaction, which prevents agents from exchanging information and building on each other's outputs.",
      "published": "2026-04-24T11:11:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.22452v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22452v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking",
      "authors": [
        "Chenxi Gu",
        "Xiaoning Du",
        "John Grundy"
      ],
      "abstract": "Watermarking has emerged as a promising technique for tracing the authorship of content generated by large language models (LLMs). Among existing approaches, the KGW scheme is particularly attractive due to its versatility, efficiency, and effectiveness in natural language generation. However, KGW's effectiveness degrades significantly under low-entropy settings such as code generation and mathematical reasoning. A crucial step in the KGW method is random vocabulary partitioning, which enables adjustments to token selection based on specific preferences. Our study revealed that the next-token probability distribution plays an critical role in determining how much, or even whether, we can modify token selection and, consequently, the effectiveness of watermarking. We refer to this characteristic, associated with the probability distribution of each token prediction, as \\emph{watermark strength.} In cases of random vocabulary partitioning, the lower bound of watermark strength is dictated by the next-token probability distribution. However, we found that, by redesigning the vocabulary partitioning algorithm, we can potentially raise this lower bound. In this paper, we propose SSG (\\textbf{S}ort-then-\\textbf{S}plit by \\textbf{G}roups), a method that partitions the vocabulary into two logit-balanced subsets. This design lifts the lower bound of watermark strength for each token prediction, thereby improving watermark detectability. Experiments on code generation and mathematical reasoning datasets demonstrate the effectiveness of SSG.",
      "published": "2026-04-24T10:55:50Z",
      "abstract_url": "http://arxiv.org/abs/2604.22438v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22438v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "From Local to Cluster: A Unified Framework for Causal Discovery with Latent Variables",
      "authors": [
        "Zongyu Li"
      ],
      "abstract": "Latent variables pose a fundamental challenge to causal discovery and inference. Conventional local methods focus on direct neighbors but fail to provide macro level insights. Cluster level methods enable macro causal reasoning but either assume clusters are known a priori or require causal sufficiency. Moreover, directly applying single variable causal discovery methods to cluster level problems violates causal sufficiency and leads to incorrect results. To overcome these limitations, this paper proposes L2C (Local to Cluster Causal Abstraction), a unified framework that bridges local structure learning and cluster level causal discovery. Unlike prior work that requires a complete manual assignment of micro variables to clusters, L2C discovers the partition automatically from local causal patterns. Our solution leverages a cluster reduction theorem to reduce any cluster to at most three nodes without loss of causal information, applies local causal discovery to identify direct causes, effects, and V structures in the presence of latent variables, and performs macro level causal inference via cluster level calculus on the learned cluster graph. L2C does not assume causal sufficiency, as latent variables are handled through local discovery. Theoretical analysis shows that L2C ensures soundness, atomic completeness, and computational efficiency. Extensive experiments on synthetic and real world data demonstrate that L2C accurately recovers ground truth clusters and achieves superior macro causal effect identification compared to existing baselines.",
      "published": "2026-04-24T10:22:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.22416v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22416v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Distance-Misaligned Training in Graph Transformers and Adaptive Graph-Aware Control",
      "authors": [
        "Qinhan Hou",
        "Jing Tang"
      ],
      "abstract": "Graph Transformers can mix information globally, but this flexibility also creates failure modes: some tasks require long-range communication while others are better served by local interaction. We study this through a synthetic node-classification benchmark on contextual stochastic block model graphs, where labels are generated by a controllable mixture of local and far-shell signals. We define distance-misaligned training as a mismatch between where label-relevant information lies and where the model allocates communication over graph distance. On this benchmark, we find three points. First, the preferred graph-distance bias changes systematically with task locality. Second, an oracle adaptive controller, given offline access to the task-side distance target, nearly matches the best fixed bias across regimes and strongly improves over a neutral baseline on mixed and local tasks. Third, a task-agnostic zero-gap controller is weaker, indicating that adaptation alone is not enough and that the control target matters. These results suggest that distance-resolved diagnosis is useful for understanding Graph Transformer failures and for designing graph-aware control.",
      "published": "2026-04-24T10:19:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.22413v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22413v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Introducing Background Temperature to Characterise Hidden Randomness in Large Language Models",
      "authors": [
        "Alberto Messina",
        "Stefano Scotta"
      ],
      "abstract": "Even when decoding with temperature $T=0$, large language models (LLMs) can produce divergent outputs for identical inputs. Recent work by Thinking Machines Lab highlights implementation-level sources of nondeterminism, including batch-size variation, kernel non-invariance, and floating-point non-associativity. In this short note we formalize this behavior by introducing the notion of \\emph{background temperature} $T_{\\mathrm{bg}}$, the effective temperature induced by an implementation-dependent perturbation process observed even when nominal $T=0$. We provide clean definitions, show how $T_{\\mathrm{bg}}$ relates to a stochastic perturbation governed by the inference environment $I$, and propose an empirical protocol to estimate $T_{bg}$ via the equivalent temperature $T_n(I)$ of an ideal reference system. We conclude with a set of pilot experiments run on a representative pool from the major LLM providers that demonstrate the idea and outline implications for reproducibility, evaluation, and deployment.",
      "published": "2026-04-24T10:11:06Z",
      "abstract_url": "http://arxiv.org/abs/2604.22411v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22411v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Hidden Failure Modes of Gradient Modification under Adam in Continual Learning, and Adaptive Decoupled Moment Routing as a Repair",
      "authors": [
        "Yuelin Hu",
        "Zhenbo Yu",
        "Zhengxue Cheng",
        "Wei Liu",
        "Li Song"
      ],
      "abstract": "Many continual-learning methods modify gradients upstream (e.g., projection, penalty rescaling, replay mixing) while treating Adam as a neutral backend. We show this composition has a hidden failure mode. In a high-overlap, non-adaptive 8-domain continual LM, all shared-routing projection baselines collapse close to vanilla forgetting (12.5--12.8 vs. 13.2). A 0.5% replay buffer is the strongest shared alternative but still reaches 11.6, while fixed-strength decoupling falls below vanilla at 14.1. Only adaptive decoupled routing remains stable at 9.4, improving over vanilla by 3.8 units. On a 16-domain stream, its gain over the strongest shared-routing projection baseline grows to 4.5--4.8 units. The failure is largely invisible on clean benchmarks. We explain this effect through Adam's second-moment pathway: in the tested regime, projection induces a 1/(1-alpha) inflation of the old-direction effective learning rate, matching measurements within 8% across eight alpha values. The same conflict appears with penalty methods, replay mixing, and at 7B scale under LoRA. Our fix routes the modified gradient only to the first moment while preserving magnitude-faithful second-moment statistics, with overlap-aware adaptive strength. This simple change is the only tested configuration that consistently avoids collapse across methods, optimizers, and scale.",
      "published": "2026-04-24T10:00:00Z",
      "abstract_url": "http://arxiv.org/abs/2604.22407v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22407v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language",
      "authors": [
        "Rui Zhao",
        "Xuewen Zhong",
        "Xiaoyun Zheng",
        "Jinsong Su",
        "Yidong Chen"
      ],
      "abstract": "Sign language research has achieved significant progress due to the advances in large language models (LLMs). However, the intrinsic ability of LLMs to understand sign language, especially in multimodal contexts, remains underexplored. To address this limitation, we introduce CNSL-bench, the first comprehensive Chinese em{National Sign Language benchmark designed for evaluating multimodal large language models (MLLMs) in sign language understanding. The proposed CNSL-bench is characterized by: 1) Authoritative grounding, as it is anchored to the officially standardized \\textit{National Common Sign Language Dictionary, mitigating ambiguity from regional or non-canonical variants and ensuring consistent semantic definitions; 2) Multimodal coverage, providing aligned textual descriptions, illustrative images, and sign language videos; and 3) Articulatory diversity, supporting fine-grained analysis across key manual articulatory forms, including air-writing, finger-spelling, and the Chinese manual-alphabet. Using CNSL-bench, we extensively evaluate 21 open-source and proprietary up-to-date MLLMs. Our results reveal that, despite recent advances in multimodal modeling, current MLLMs remain substantially inferior to human performance, exhibiting systematic disparities across input modalities and manual articulatory forms. Additional diagnostic analyses suggest that several performance limitations persist beyond improvements in reasoning and that instruction-following robustness varies substantially across models.",
      "published": "2026-04-24T08:59:33Z",
      "abstract_url": "http://arxiv.org/abs/2604.22367v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22367v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FETS Benchmark: Foundation Models Outperform Dataset-specific Machine Learning in Energy Time Series Forecasting",
      "authors": [
        "Marco Obermeier",
        "Marco Pruckner",
        "Florian Haselbeck",
        "Andreas Zeiselmair"
      ],
      "abstract": "Driven by the transition towards a climate-neutral energy system, accurate energy time series forecasting is critical for planning and operation. Yet, it remains largely a dataset-specific task, requiring comprehensive training data, limiting scalability, and resulting in high model development and maintenance effort. Recently, foundation models that aim to learn generalizable patterns via extensive pretraining have shown superior performance in multiple prediction tasks. Despite their success and strong potential to address challenges in energy forecasting, their application in this domain remains largely unexplored. We address this gap by presenting the Foundation Models in Energy Time Series Forecasting (FETS) benchmark. We (1) provide a structured overview of energy forecasting use cases along three main dimensions: stakeholders, attributes, and data categories; (2) collect and analyze 54 datasets across 9 data categories, guided by typical stakeholder interests; (3) benchmark foundation models against classical machine learning approaches across different forecasting settings. Foundation models consistently outperform dataset-specific optimized machine learning approaches across all settings and data categories, despite the latter having seen the full historic target data during training. In particular, covariate-informed foundation models achieve the strongest performance. Further analysis reveals a strong correlation between predictive performance and spectral entropy, performance saturation beyond a certain context length, and improved performance at higher aggregation levels such as national load, district heating, and power grid data. Overall, our findings highlight the strong potential of foundation models as scalable and generalizable forecasting solutions for the energy domain, particularly in data-constrained and privacy-sensitive settings.",
      "published": "2026-04-24T08:00:50Z",
      "abstract_url": "http://arxiv.org/abs/2604.22328v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22328v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CE"
      ]
    },
    {
      "title": "BLAST: Benchmarking LLMs with ASP-based Structured Testing",
      "authors": [
        "Manuel Alejandro Borroto Santana",
        "Erica Coppolillo",
        "Francesco Calimeri",
        "Giuseppe Manco",
        "Simona Perri",
        "Francesco Ricca"
      ],
      "abstract": "Large Language Models (LLMs) have demonstrated remarkable performance across a broad spectrum of tasks, including natural language understanding, dialogue systems, and code generation. Despite evident progress, less attention has been paid to their effectiveness in handling declarative paradigms such as Answer Set Programming (ASP), to date. In this paper we introduce BLAST: The first dedicated benchmarking methodology and associated dataset for evaluating the accuracy of LLMs in generating ASP code. BLAST provides a structured evaluation framework featuring two novel semantic metrics tailored to ASP code generation. The paper presents the results of an empirical evaluation involving ten well-established graph-related problems from the ASP literature and a diverse set of eight state-of-the-art LLMs.",
      "published": "2026-04-24T07:39:09Z",
      "abstract_url": "http://arxiv.org/abs/2604.22306v1",
      "pdf_url": "https://arxiv.org/pdf/2604.22306v1",
      "categories": [
        "cs.LO",
        "cs.AI",
        "cs.PL"
      ]
    }
  ]
};
