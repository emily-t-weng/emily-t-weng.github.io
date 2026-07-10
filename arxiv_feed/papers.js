const PAPERS_DATA = {
  "last_updated": "2026-07-10 03:59:51 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "SLORR: Simple and Efficient In-Training Low-Rank Regularization",
      "authors": [
        "David González-Martínez",
        "Shiwei Liu"
      ],
      "abstract": "Low-rank factorization is widely used to compress neural networks, but modern models are often not naturally amenable to aggressive factorization without significant accuracy loss. Existing training-time low-rank regularizers can improve compressibility, but they often require SVDs of large weight matrices, modify the model architecture (introducing additional trainable parameters), or rely on stateful cached quantities. To address these limitations, we introduce SLORR, a simple, stateless, and architecture-preserving framework for in-training low-rank regularization, instantiated with two main variants based on the Hoyer sparsity metric and the nuclear norm. SLORR directly regularizes the original weight matrices using GPU-friendly approximations for the forward and backward passes of the regularizers, for which we provide approximation guarantees. We first evaluate SLORR on ImageNet-1K across short-horizon continued training of ResNet-50, ViT-B/16, and ViT-L/16, and pretraining of ResNet-18, where SLORR induces compressibility while introducing less than 8% training overhead. We further evaluate SLORR-Hoyer in LLM pretraining at 135M and 560M scales: SLORR-trained compressed models preserve performance substantially better than unregularized models while adding less than 1% average training overhead.",
      "published": "2026-07-09T17:51:50Z",
      "abstract_url": "http://arxiv.org/abs/2607.08754v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08754v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Dimensionality Reduction Meets Network Science: Sensemaking on UMAP's kNN Graph",
      "authors": [
        "Duen Horng Chau",
        "Donghao Ren",
        "Fred Hohman",
        "Dominik Moritz"
      ],
      "abstract": "While UMAP is widely used for exploring high-dimensional data, typical workflows focus on its lower-dimensional embedding, largely overlooking the rich k-nearest-neighbor (kNN) graph that UMAP constructs internally. This graph encodes the data manifold in its original high-dimensional space, before the distortion that UMAP's 2D projection introduces. We demonstrate the untapped potential of this internal representation, showing how standard graph algorithms applied to this graph enhance data sensemaking: (1) PageRank identifies representative data points, (2) k-core decomposition reveals dense core regions versus sparse periphery, and (3) clustering coefficient detects tight-knit neighborhoods with highly-similar data points. Through quantitative and qualitative evaluation on MNIST and Fashion MNIST, we show that these graph-based analyses are not only practical but also competitive with or complementary to purpose-built methods (e.g., k-medoids for exemplar selection, HDBSCAN for density-based clustering).",
      "published": "2026-07-09T17:47:08Z",
      "abstract_url": "http://arxiv.org/abs/2607.08746v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08746v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DS",
        "cs.HC"
      ]
    },
    {
      "title": "AUTOPILOT VQA: Benchmarking Vision-Language Models for Incident-Centric Dashcam Understanding",
      "authors": [
        "Siddharth Damodharan",
        "Radhika Gupta",
        "Ali Alshami",
        "Ryan Rabinowitz",
        "Jugal Kalita"
      ],
      "abstract": "Recent advances in Vision-Language Models, Large Language Models, and Multimodal Large Language Models have improved autonomous driving tasks such as scene understanding, decision making, trajectory prediction, and visual question answering. However, evaluating whether these models can reliably reason about safety-critical incidents remains challenging. To address this gap, we present AUTOPILOT-VQA, an incident-centric visual question answering benchmark for dashcam video understanding. The dataset evaluates different systems through structured questions designed around real-world driving incidents and near-incidents. The benchmark covers diverse safety-relevant categories, including weather and lighting conditions, traffic environment, road layout, road surface state, signage, involved entities, accident occurrence, impact location, and avoidability-related reasoning. By requiring models to answer grounded questions about both contextual scene properties and event-level incident details, AUTOPILOT-VQA moves beyond object recognition toward temporally grounded, safety-aware reasoning. The dataset is released as part of the AUTOPILOT CVPR 2026 competition and provides a standardized benchmark for assessing the reliability of autonomous driving systems in different scenarios. Our benchmark support developments for more interpretable, robust, and safety-conscious vision-language systems for real-world autonomous driving.",
      "published": "2026-07-09T17:46:24Z",
      "abstract_url": "http://arxiv.org/abs/2607.08745v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08745v1",
      "categories": [
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows",
      "authors": [
        "Emanuele Quinto",
        "Carlo Andrea Rozzi",
        "Francesco Zanitti"
      ],
      "abstract": "Large language model (LLM) applications increasingly use explicit workflows for tool use, retrieval, branching, checkpointing, and human approval. Existing workflow systems already address many execution concerns. This paper proposes a Lisp-inspired but language-independent conceptual model: symbolic forms, object identity, and live-image thinking are used as explanatory lenses, not implementation commitments. In this model, workflow definitions, workflow instances, inference records, context snapshots, and dependency relations are represented as persistent knowledge objects in a shared knowledge substrate. Its central semantic distinction is between derive and infer: derive is deterministic computation over available state; infer is mediated LLM judgment under declared context and executor-controlled capability policy. The result is a preliminary conceptual account of semantic persistence: workflows do not merely produce knowledge and leave traces, but can themselves be represented as inspectable, resumable, and reviewable knowledge objects, while formal transition semantics remain future work.",
      "published": "2026-07-09T17:40:46Z",
      "abstract_url": "http://arxiv.org/abs/2607.08740v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08740v1",
      "categories": [
        "cs.AI",
        "cs.PL",
        "cs.SE"
      ]
    },
    {
      "title": "The Illusion of Equivalency: Statistical Characterization of Quantization Effects in LLMs",
      "authors": [
        "Baha Rababah",
        "Cuneyt Gurcan Akcora",
        "Carson K. Leung"
      ],
      "abstract": "Post-training quantization is widely used to deploy large language models in resource-constrained settings, yet its evaluation relies almost exclusively on accuracy and perplexity. We show that these metrics fail to capture behavioral changes induced by quantization. We introduce correctness agreement, a decision-level metric that measures overlap in correct predictions between a base model and its quantized variants, independent of absolute accuracy. Across multiple models and quantization schemes from 8-bit to 2-bit, we find that behavioral divergence emerges under moderate quantization even when task performance appears preserved. To explain this effect, we analyze quantization as a structural operator on attention weights and quantify layer-wise distortions using statistical and distributional measures. Our results reveal non-linear breakpoints at low bit-widths and show that query and key projections are consistently more sensitive than value and output projections. These findings expose an illusion of equivalence between base and quantized models and motivate behavioral evaluation beyond conventional performance metrics.",
      "published": "2026-07-09T17:35:02Z",
      "abstract_url": "http://arxiv.org/abs/2607.08734v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08734v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Pose-to-Biomechanics: Bridging 3D Human Pose Estimation and Biomechanical Attribute Prediction",
      "authors": [
        "Ayda Eghbalian",
        "Kevin Desai"
      ],
      "abstract": "Recent progress in 3D human pose estimation has made markerless recovery of skeletal motion increasingly accurate and scalable. However, most pose estimators remain optimized for geometric keypoint accuracy, while many real-world applications in rehabilitation, sports science, ergonomics, and clinical movement analysis require biomechanical quantities that describe how the body moves, loads, and activates. In this work, we propose BioModule, a lightweight plug-in temporal transformer that attaches downstream of any 3D pose estimator and predicts biomechanical attributes from standard 17-joint 3D skeletons. BioModule is estimator-agnostic and requires no modification of the upstream pose model, enabling existing pose estimators to be extended toward physically interpretable motion analysis. To train and evaluate BioModule, we construct a large-scale aligned dataset pairing Human3.6M video and 3D keypoints with the biomechanical label space of Human3.6Mplus. We establish and verify anatomical correspondence between coordinate systems of the two datasets, enabling frame-accurate cross-modal supervision. Using this aligned supervision, BioModule predicts biomechanical quantities. We further benchmark BioModule across seven state-of-the-art 3D pose estimators, providing the first systematic analysis of how upstream pose estimation quality propagates to downstream biomechanical prediction fidelity. The results position BioModule as a compact, modular bridge between vision-based pose estimation and biomechanically meaningful human motion analysis.",
      "published": "2026-07-09T17:31:16Z",
      "abstract_url": "http://arxiv.org/abs/2607.08725v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08725v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Practical Investigation of Training-free Relaxed Speculative Decoding",
      "authors": [
        "Guoxuan Xia",
        "Luka Ribar",
        "Paul Balanca"
      ],
      "abstract": "Speculative decoding accelerates sampling from an autoregressive LLM by using a faster auxiliary model to draft tokens which are then verified in parallel by the LLM. Standard speculative decoding is lossless: its rejection and resampling steps exactly preserve the LLM's sampling distribution. Recent work argues that relaxing this strict guarantee can yield further speed-ups, controlled capability-speed trade-offs, or even capability gains. We practically investigate training-free relaxed speculative decoding techniques, unify existing approaches within a shared framework, benchmark them on contemporary settings, and distil takeaways and empirical findings for practitioners. Important takeaways include: relaxation can require considerable capability evaluation unlike lossless speculative decoding, and many relaxed approaches rely on a drafter that is a good language model, making them unsuited for lightweight dedicated multi-token-prediction drafters.",
      "published": "2026-07-09T16:50:41Z",
      "abstract_url": "http://arxiv.org/abs/2607.08690v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08690v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search",
      "authors": [
        "Xiaoshuai Song",
        "Liancheng Zhang",
        "Kangzhi Zhao",
        "Yutao Zhu",
        "Zhongyuan Wang",
        "Guanting Dong",
        "Jinghan Yang",
        "Han Li",
        "Kun Gai",
        "Ji-Rong Wen",
        "Zhicheng Dou"
      ],
      "abstract": "Large language model (LLM)-based web search agents are transforming information seeking from simple factoid question answering into complex, deep-and-wide search and research-oriented tasks. A single ReAct-style agent is constrained by one long trajectory and limited context, making it difficult to handle depth and coverage simultaneously. Existing multi-agent systems improve search coverage through parallel execution and aggregation, but still exhibit clear limitations in recursive depth, collaboration adaptability, and evidence-grounded expansion. We propose WebSwarm, a progressive recursive delegation framework that jointly constructs task decomposition, recursive expansion, and agent collaboration during inference. WebSwarm dynamically instantiates agentic search nodes, each coupling a local objective with a search mode that specifies how the node should organize search and collaboration. Each node can either solve its objective itself or further delegate child nodes; after solving, it returns evidence and results upward, enabling parent nodes to further expand, revise, or aggregate the search process. To guide this process, WebSwarm first probes how task-relevant information is organized on the web to ground subsequent node expansion, and reuses process-level experience across homogeneous sibling nodes. Experiments on BrowseComp-Plus, WideSearch, DeepWideSearch, and GISA show that WebSwarm consistently outperforms single-agent and multi-agent baselines on deep, wide, and interleaved deep-and-wide tasks. Further analyses of ablation, task difficulty, web tool efficiency, and model generalization explain WebSwarm's effectiveness and provide insights for multi-agent search systems.",
      "published": "2026-07-09T16:28:49Z",
      "abstract_url": "http://arxiv.org/abs/2607.08662v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08662v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "Multi-Modal, Multi-Environment Machine Teaching for Robust Reward Learning",
      "authors": [
        "Ali Larian",
        "Qian Lin",
        "Chang Zong Wu",
        "Daniel S. Brown"
      ],
      "abstract": "As autonomous agents are increasingly deployed across diverse operational contexts, aligning their behavior with human intent demands reward functions that remain robust to such changes rather than overfitting to any single environment. Inverse reinforcement learning (IRL) provides a principled way to infer such objectives from human feedback. However, existing analyses of optimal teaching approaches for IRL focus on single-environment, demonstration-only settings, leaving underexplored how heterogeneous feedback modalities and environment dynamics jointly constrain reward functions that generalize across multiple environments. Because demonstrations in one MDP entangle reward information with that environments specific structure, the resulting rewards frequently fail to generalize when the agent is deployed in a new setting. We first analyze how different feedback modalities constrain rewards, showing that, in the unlimited-data regime, comparisons impose strictly stronger global constraints than other modalities. Beyond this theoretical analysis, we introduce a hierarchical machine teaching algorithm for reward learning that operates across multiple MDPs. The algorithm first greedily selects informative environments that expose complementary reward constraints, then strategically queries low-cost feedback within those environments. Empirically, our method achieves substantially lower regret and stronger generalization to held-out environments than uniform teaching baselines under identical feedback budgets, demonstrating the importance of multi-environment, multi-modal teaching for learning dynamics-robust reward functions.",
      "published": "2026-07-09T16:18:16Z",
      "abstract_url": "http://arxiv.org/abs/2607.08647v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08647v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "UltraX: Refining Pre-Training Data at Scale with Adaptive Programmatic Editing",
      "authors": [
        "Xinlong Zhao",
        "Dongsheng Liu",
        "Hengyu Zhao",
        "Zixuan Fu",
        "Zheng Wang",
        "Jie Cai",
        "Jie Zhou",
        "Qiang Ma",
        "Xuanhe Zhou",
        "Xu Han",
        "Yudong Wang",
        "Zhiyuan Liu"
      ],
      "abstract": "As available training data approaches its physical limit, gains from Scaling Laws have begun to diminish. Consequently, improving Large Language Models (LLMs) now depends less on data expansion and more on higher-quality data utilization. However, in the context of large-scale corpora, existing refinement methodologies face significant limitations in quality, efficiency, and reliability: Rule-based approaches are constrained by fixed heuristics and struggle with instance-level variations; LLM-based approaches improve quality but fail to meet the efficiency and reliability requirements of large-scale data processing. To address these challenges, we propose UltraX, a function-calling refinement framework for large-scale pre-training data that completes the editing function space by introducing insertion in addition to deletion and modification, enabling fine-grained instance-level editing. Specifically, UltraX builds a reliable program-supervision generation pipeline. In this pipeline, dataset-adaptive prompt optimization first guides an expert LLM to produce high-quality end-to-end refined texts, and Line Alignment Mapping and Dynamic Context Replacement then convert original-refined text pairs into structured program supervision. Meanwhile, UltraX improves supervision quality and stabilizes the training distribution with low-confidence example filtering and ratio-controlled sampling by operation combination. During inference and execution, it normalizes and validates model outputs through sliding-window prediction, global operation aggregation, and systematic post-processing, improving the stability and reliability of large-scale execution. Experiments show that UltraX achieves the highest average performance across all corpora and also matches or surpasses baselines with fewer training tokens, demonstrating stronger data efficiency and refinement reliability.",
      "published": "2026-07-09T16:18:07Z",
      "abstract_url": "http://arxiv.org/abs/2607.08646v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08646v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "The complexities of patient-centred conversational artificial intelligence",
      "authors": [
        "João Matos",
        "Olivia Buege",
        "Donny Cheung",
        "Gary S. Collins",
        "Paula Dhiman",
        "Nan Li",
        "Bingyu Mao",
        "Benjamin W. Nelson",
        "Michail Ouroutzoglou",
        "Paul Varghese",
        "Jonathan Amar"
      ],
      "abstract": "Consumer-facing health chatbots powered by large language models (LLMs) are increasingly used for symptom assessment. However, chatbot development and evaluation often rely on cooperative, articulate, simulated patients. We analysed 2,053 real patient-chatbot conversations and found that communication patterns and expression of emotions vary widely across users. We developed a patient simulator that separately models clinical content, emotional state, conversational strategy, and communication style. In a Turing-inspired evaluation of realism with 15 human graders, simulated conversations were nearly indistinguishable from real ones, with human graders achieving an accuracy of 55%. We used five distinct patient personae, across 1,164 clinician-graded cases, to evaluate the performance of four LLMs in urgency assessment. We found that communication style can significantly alter triage outcomes. Patient-centred conversational artificial intelligence must accommodate communication diversity: systems designed for idealised, rather than realistic, interactions risk underperforming and amplifying health disparities when deployed in the real world.",
      "published": "2026-07-09T15:56:55Z",
      "abstract_url": "http://arxiv.org/abs/2607.08625v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08625v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "When Structured Sparse Autoencoders Learn Consistent Concepts Across Modalities",
      "authors": [
        "Weiduo Liao",
        "Yunqiao Yang",
        "Ying Wei"
      ],
      "abstract": "Sparse autoencoders (SAEs) have emerged as a promising technique for mechanistic interpretability by learning a set of sparse latent features in large models, each of which encodes a distinct concept. However, in vision-language models (VLMs), vanilla SAEs struggle to learn modality-consistent concepts, with concepts often exhibiting fragmented coverage (i.e., disjoint regions) in the visual modality. To address this challenge, we propose a Structured Sparse AutoEncoder ($S^2AE$) that enforces concept consistency from both semantic and spatial perspectives in the visual modality. Specifically, we group image patches based on Transformer attention similarity and spatial proximity, and introduce a structured sparsity regularization when training the vanilla SAE. The regularization consists of exclusive sparsity for inter-group concept disentanglement and group sparsity for intra-group concept consistency, which drives the latent neurons by SAEs to specialize in distinct, semantically grounded concepts. Evaluated on the \\texttt{Qwen2.5-VL-7B-Instruct} model, the method achieves 6.06% average improvement in semantic alignment (mIoU) and 60.81 in representational efficiency (lower l0 norm) while maintaining near-perfect reconstruction fidelity with an Explained Variance above 99%. Cross-modal analysis further demonstrates that $S^2AE$ enhances neuronal monosemanticity by this visual structural prior, achieving a 3.08% average gain in semantic consistency and a 2.37% average gain in monosemanticity scores for both modalities of multimodal features, thereby fostering more coherent and disentangled representations.",
      "published": "2026-07-09T15:35:08Z",
      "abstract_url": "http://arxiv.org/abs/2607.08605v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08605v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Towards Precision Therapy in Hepatocellular Carcinoma: A Clinical-Reasoning LLM for Risk Stratification and Treatment Guidance",
      "authors": [
        "Peng Cui",
        "Jitao Wang",
        "Siyan Xue",
        "Yao Huang",
        "Haoming Xia",
        "Dong Li",
        "Dengxiang Liu",
        "Weilin Wang",
        "Liping Liu",
        "Leida Zhang",
        "Yunfu Cui",
        "Tao Peng",
        "Daolin Ji",
        "Haitao Zhao",
        "Wei Zhang",
        "Xiaojuan Wang",
        "Weijie Ma",
        "Zongren Ding",
        "Jinlong Li",
        "Yuan Ding",
        "Jiajing Zhao",
        "Zhiyu Chen",
        "Chengkun Yang",
        "Ziyue Huang",
        "Jiaqi Liu",
        "Fusheng Liu",
        "Yang Zhou",
        "Xiaojuan Wang",
        "Zhongquan Sun",
        "Shiyun Bao",
        "Xiaojun Wang",
        "Ming Yang",
        "Guangxin Li",
        "Bin Shu",
        "Yong Liao",
        "Hongxuan Li",
        "Yao Tang",
        "Shizhong Yang",
        "Yongyi Zeng",
        "Yufeng Yuan",
        "Yinpeng Dong",
        "Jihui Hao",
        "Jun Zhu",
        "Jiahong Dong"
      ],
      "abstract": "Hepatocellular carcinoma (HCC) is a common malignancy and a leading cause of cancer-related mortality. Current guidelines and staging systems provide coarse categories, but often miss within-stage heterogeneity and the clinical context in electronic medical records (EMRs). We present HCC-STAR (Hepatocellular Carcinoma Staging, Treatment And pRognosis), a clinically aligned large language model that reads routine EMR narratives and jointly outputs risk score-based staging, ranked guideline-consistent treatments with evidence-based rationales, and individualized survival estimates. We curated about 30,000 HCC cases from SEER and expanded them into EMR-style narrative training data using a clinician-validated, prompt-based augmentation workflow. On this corpus, we developed a knowledge-aligned reasoning framework optimized with a step-verifiable composite reward, moving beyond text-level memorization of clinical guidelines. In a multi-center cohort of 6,668 patients from 12 hospitals in China, HCC-STAR achieved state-of-the-art performance in treatment recommendation and risk stratification compared with clinical guidelines and competitive models, including GPT-5 and Gemini-2.5 Pro. Hypothetical overall-survival analysis showed a median survival of 51 months under adherence to HCC-STAR recommendations, compared with 29 and 32 months under BCLC and CNLC. In clinician-centric evaluations, blinded hepatobiliary specialists rated HCC-STAR's reasoning and evidence-based justifications as trustworthy. The model surpassed resident and attending physicians in treatment accuracy and helped physicians make more accurate decisions faster when used as an assistant. These findings support HCC-STAR as a reliable and verifiable decision-support system for risk stratification and precision therapy in HCC.",
      "published": "2026-07-09T15:33:08Z",
      "abstract_url": "http://arxiv.org/abs/2607.08602v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08602v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CommuniWave:A Machine Learning Model for Quantifying the Degree of Temporary Informal Behavior in Urban Communities",
      "authors": [
        "Hongye Yang",
        "Shien Liu",
        "Zhihao Xie"
      ],
      "abstract": "For urban managers and designers, improving the functional attributes of urban communities to enhance territorial resilience in the face of complexity and uncertainty is crucial. Currently, community planning often follows a top-down approach and lacks effective metrics to quantify informal behaviors of residents, leading to frequent conflicts with original plans. This study introduces CommuniWave, a machine learning model designed to efficiently detect and quantify the Degree of Informal Behavior (DIB) in urban communities. The model integrates a Behavior Capture Net (BCN) based on mmaction2, a self-developed YOLOv10 model (YLX), and a Behavior Eval Model (BEM) using random forest. Ultimately, by generating DIB fluctuation charts from street videos, the model facilitates dynamic monitoring, supporting urban managers in making refined decisions to enhance the overall resilience of communities.",
      "published": "2026-07-09T14:45:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.08554v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08554v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "DocMaster: A Hierarchical Structure-Aware System for Document Analysis",
      "authors": [
        "Ziqi Chen",
        "Yingli Zhou",
        "Fangyuan Zhang",
        "Quanqing Xu",
        "Chuanhui Yang",
        "Yixiang Fang"
      ],
      "abstract": "Leveraging large language models (LLMs) to analyze complex documents -- such as academic papers, technical manuals, and financial reports -- has emerged as a mainstream and critical task in both research and industry. In practice, users must first filter relevant documents from large collections and then conduct in-depth analysis (e.g. question answering) over the selected subset, yet existing systems flatten documents into plain-text chunks, discarding the rich hierarchical structures (sections, tables, figures, equations) and degrading downstream performance. We present DocMaster, a hierarchical structure-aware document analysis system. DocMaster parses documents into hierarchical document trees preserving original layouts and constructs a structure-aware semantic index that enables accurate document filtering and in-depth analysis. We demonstrate DocMaster through an interactive web interface that enables users to upload document collections, construct tree-based and multi-view semantic indices, filter relevant documents via natural-language conditions, and perform follow-up question answering over the filtered results. The source code, data, and demo are available at https://doc-master.github.io/.",
      "published": "2026-07-09T14:33:47Z",
      "abstract_url": "http://arxiv.org/abs/2607.08539v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08539v1",
      "categories": [
        "cs.DB",
        "cs.AI"
      ]
    },
    {
      "title": "AI-guided stimuli discovery and generation to optimize facial emotion perception studies in autism",
      "authors": [
        "Kushin Mukherjee",
        "Na Yeon Kim",
        "Maren Wehrheim",
        "Ralph Adolphs",
        "Kohitij Kar"
      ],
      "abstract": "Understanding perceptual differences between autistic and neurotypical adults requires behavioral assays that are sensitive, reliable, and mechanistically informative. Facial emotion perception is a useful test case because group differences have been reported, but findings vary across studies. Here we show that this variability may reflect image-level sparsity: autistic-neurotypical differences in emotion judgments were concentrated in a small subset of diagnostic facial expressions rather than spread uniformly across stimuli. We trained population-specific artificial neural network models to predict image-level judgments for autistic and neurotypical participants, then used these models to select novel faces predicted to maximize group separation. In an independent cohort, model-selected images produced larger behavioral differences than matched random images. We then used the same models with a generative adversarial network to transform diagnostic images toward greater predicted group agreement. In phenotype-matched validation, synthesized images reduced behavioral separation relative to their matched originals. These results establish a model-guided framework for discovering and transforming stimuli that reveal population-specific perceptual differences. More broadly, they show how behavioral phenotyping can move beyond averaging across fixed stimulus sets toward optimized assays that identify the conditions under which neurodivergent perception diverges or converges.",
      "published": "2026-07-09T14:29:31Z",
      "abstract_url": "http://arxiv.org/abs/2607.08533v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08533v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Cognitive-structured Multimodal Agent for Multimodal Understanding, Generation, and Editing",
      "authors": [
        "Feng Wang",
        "Canmiao Fu",
        "Zhipeng Huang",
        "Chen Li",
        "Jing Lyu",
        "Ge Li"
      ],
      "abstract": "Recent unified multimodal models show a single architecture can jointly perform vision/language understanding and image generation/editing. However, they repeatedly feed all historical visual and textual inputs into a shared context window, limiting long-horizon multimodal dialogue due to visual token explosion and unreliable cross-turn referencing. We propose a Cognitive-structured Multimodal Agent that externalizes visual information into an Episodic Visual Memory and selectively reactivates relevant episodes during reasoning. The agent consists of a Perceptual Abstraction Engine for structured visual abstraction, a Cognitive Retrieval Engine for cross-turn memory retrieval, and a Multimodal Executive Controller for autonomous task inference and action planning. To address the lack of turn-level retrieval supervision in existing datasets, we develop a Unified Scenario Engine that programmatically generates structured multi-turn conversations with fine-grained retrieval annotations, enabling reinforcement learning to optimize abstraction and retrieval policies. We also construct a long-horizon visual-dialogue benchmark stratified by difficulty to evaluate episodic visual recall. Our 8B agent achieves 91.4% retrieval accuracy over 20-turn sessions, surpassing 32B baselines by +8.2% while nearly halving per-turn inference time (23.1s -> 12.7s). We further present the Cognitive-structured Multimodal Agent Harness (CMA-Harness), a tool-augmented deployment of the same cognitive structure integrating persistent multimodal memory, web access, image generation/editing/composition tools, and OpenAI-compatible serving. Structured memory and modular decision-making offer a more scalable, efficient paradigm for long-horizon multimodal agents than monolithic parameter scaling. Code: https://github.com/caseclose/cma-harness ; Project page: https://caseclose.github.io/cma-harness/",
      "published": "2026-07-09T13:55:55Z",
      "abstract_url": "http://arxiv.org/abs/2607.08497v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08497v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "ADORN: Adaptive Drift handling for Open RAN using Reinforcement Learning",
      "authors": [
        "Ashit Kumar Subudhi",
        "Bhargav Chirumamilla",
        "Shubham Vaishnav",
        "Mduduzi C. Hlophe",
        "Praveen Kumar Donta",
        "Andrea Fumagalli",
        "Venkateswarlu Gudepu",
        "Koteswararao Kondepu"
      ],
      "abstract": "Dynamic traffic variations in Open Radio Access Networks (O-RAN) lead to drift, which degrades the performance of Artificial Intelligence/Machine Learning (AI/ML) models. Traditional retraining approaches maintain forecasting accuracy but incur high computational cost and may lead to violations of Service Level Agreements (SLAs). This work proposes a Q-learning-based adaptive retraining approach that formulates the retraining decision as a Markov Decision Process (MDP), where a Reinforcement Learning (RL) agent learns a policy that balances forecasting accuracy and retraining cost. The proposed approach incorporates a multi-expert Long Short-Term Memory (LSTM) ensemble to mitigate catastrophic forgetting and improve robustness across diverse traffic conditions. Experimental results show that the proposed approach effectively reduces retraining overhead compared to greedy and random baselines, while maintaining system performance within predefined limits.",
      "published": "2026-07-09T13:05:42Z",
      "abstract_url": "http://arxiv.org/abs/2607.08443v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08443v1",
      "categories": [
        "cs.NI",
        "cs.AI"
      ]
    },
    {
      "title": "Predicting Male Fertility Using Machine Learning: A Semen Parameters Based Analysis with the VISEM Dataset",
      "authors": [
        "Shahnawaz Qureshi",
        "Raja Khurram Shahzad",
        "Muhammad Fozan",
        "Emal Kawal",
        "Syed Aziz Shah",
        "Sattam Al-Anazi",
        "Syed MuhammadZeeshan Iqbal"
      ],
      "abstract": "Male infertility is a significant yet often underdiagnosed aspect of reproductive health, with semen analysis serving as the cornerstone of clinical evaluation. To address this problem, this study investigates the use of machine learning algorithms to classify male fertility status based on key semen parameters, i.e., sperm concentration, motility, and morphology, using the VISEM dataset. This dataset includes semen samples from 85 participants, classified into three categories, i.e., Fertile, Sub-Fertile, and Infertile, according to the World Health Organization's criteria. After pre-processing and feature engineering, the dataset was used to train and assess multiple classification models using the LazyPredict framework. Among the more than 40 algorithms tested, the Nearest Centroid classifier achieved an accuracy of 94.2%, outperforming other models such as Support Vector Machines and Quadratic Discriminant Analysis. The model's robustness was validated using 5-fold cross-validation and multiclass ROC-AUC analysis. This study illustrates that machine learning models can provide fast, accurate, and objective assessments of semen quality, potentially supporting clinical decision-making in andrology and assisted reproductive technologies. These findings emphasize the growing potential of machine learning to enhance fertility diagnostics and inform patient-specific treatment strategies.",
      "published": "2026-07-09T12:51:46Z",
      "abstract_url": "http://arxiv.org/abs/2607.08429v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08429v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "DrugGen 2: A disease-aware language model for enhancing drug discovery",
      "authors": [
        "Ali Motahharynia",
        "Mohammadreza Ghaffarzadeh-Esfahani",
        "Mahsa Sheikholeslami",
        "Navid Mazrouei",
        "Matin Irajpour",
        "Yousof Gheisari",
        "Hajar Sirous"
      ],
      "abstract": "Current computational approaches for drug design typically focus on generating molecules conditioned on specific targets or general molecular properties, often neglecting the influence of disease context on target behavior and therapeutic outcomes. To address this gap, we introduce DrugGen-2, a novel generative model that designs small molecules conditioned on both disease ontology and target protein sequences. DrugGen-2 was developed by fine-tuning a pre-trained GPT-2 model on a curated dataset of approved drugs linked to their diseases and targets, using a two-step strategy of supervised fine-tuning followed by reinforcement learning via group relative policy optimization (GRPO). This process was guided by reward functions optimizing for chemical validity, novelty, diversity, and high predicted binding affinity. When evaluated on five protein targets relevant to diabetic nephropathy, DrugGen-2 significantly outperformed baseline models (DrugGPT and DrugGen). It demonstrated a superior capacity to generate unique molecules, exhibited greater structural similarity to approved drugs, and achieved improved predicted binding affinities across all targets. Molecular docking analyses further supported these findings, identifying candidate ligands with strong binding potential, including compounds with predicted affinities (-9.917, -9.485, and -9.367) exceeding those of reference drugs such as enalapril for angiotensin-converting enzyme (-8.283). By integrating disease-specific context into molecular generation, DrugGen-2 advances AI-assisted drug discovery, offering a powerful tool for de novo design and drug repurposing that accounts for the complex interplay between diseases and molecular targets.",
      "published": "2026-07-09T12:29:33Z",
      "abstract_url": "http://arxiv.org/abs/2607.08404v1",
      "pdf_url": "https://arxiv.org/pdf/2607.08404v1",
      "categories": [
        "q-bio.QM",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
