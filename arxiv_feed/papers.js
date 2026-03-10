const PAPERS_DATA = {
  "last_updated": "2026-03-10 02:40:53 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Revisiting Gradient Staleness: Evaluating Distance Metrics for Asynchronous Federated Learning Aggregation",
      "authors": [
        "Patrick Wilhelm",
        "Odej Kao"
      ],
      "abstract": "In asynchronous federated learning (FL), client devices send updates to a central server at varying times based on their computational speed, often using stale versions of the global model. This staleness can degrade the convergence and accuracy of the global model. Previous work, such as AsyncFedED, proposed an adaptive aggregation method using Euclidean distance to measure staleness. In this paper, we extend this approach by exploring alternative distance metrics to more accurately capture the effect of gradient staleness. We integrate these metrics into the aggregation process and evaluate their impact on convergence speed, model performance, and training stability under heterogeneous clients and non-IID data settings. Our results demonstrate that certain metrics lead to more robust and efficient asynchronous FL training, offering a stronger foundation for practical deployment.",
      "published": "2026-03-09T10:40:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.08211v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08211v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Distributional Regression with Tabular Foundation Models: Evaluating Probabilistic Predictions via Proper Scoring Rules",
      "authors": [
        "Jonas Landsgesell",
        "Pascal Knoll"
      ],
      "abstract": "Prior-Data Fitted Networks (PFNs), such as TabPFN and TabICL, have revolutionized tabular deep learning by leveraging in-context learning for tabular data. These models are meant as foundation models for classification and regression settings and promise to greatly simplify deployment in practical settings because their performance is unprecedented (in terms of mean squared error or $R^2$, when measured on common benchmarks like TabArena or TALENT). However, we see an important weakness of current benchmarks for the regression setting: the current benchmarks focus on evaluating win rates and performance using metrics like (root) mean squared error or $R^2$. Therefore, these leaderboards (implicitly and explicitly) push researchers to optimize for machine learning pipelines which elicit a good mean value estimate. The main problem is that this approach only evaluates a point estimate (namely the mean estimator which is the Bayes estimator associated with the mean squared error loss). In this article we discuss the application of proper scoring rules for evaluating the goodness of probabilistic forecasts in distributional regression. We also propose to enhance common machine learning benchmarks with metrics for probabilistic regression. To improve the status quo and make the machine learning community aware of scoring rules for probabilistic regression, we advocate to use the continuous ranked probability score (CRPS) in benchmarks for probabilistic regression. However, we also illustrate that the choice of the scoring rule changes the inductive bias of the trained model. We, therefore, advocate for finetuning or promptable tabular foundation models.",
      "published": "2026-03-09T10:38:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.08206v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08206v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TildeOpen LLM: Leveraging Curriculum Learning to Achieve Equitable Language Representation",
      "authors": [
        "Toms Bergmanis",
        "Martins Kronis",
        "Ingus Jānis Pretkalniņš",
        "Dāvis Nicmanis",
        "Jeļizaveta Jeļinska",
        "Roberts Rozis",
        "Rinalds Vīksna",
        "Mārcis Pinnis"
      ],
      "abstract": "Large language models often underperform in many European languages due to the dominance of English and a few high-resource languages in training data. This paper presents TildeOpen LLM, a 30-billion-parameter open-weight foundational model trained for 34 European languages to promote linguistic equity and improve performance for low-resource languages. To address the data imbalance, we combine dataset upsampling with a curriculum-based training schedule that alternates between uniform and natural language distributions. The resulting model performs favorably compared to other multilingual LLMs despite being trained with significantly fewer computing resources. Evaluation across multiple multilingual benchmarks shows that TildeOpen surpasses existing open-weight models in text generation and comprehension, particularly for Baltic, Finno-Ugric, and Slavic languages. Human evaluations confirm an up to tenfold reduction in linguistic errors relative to leading baselines. The model and associated resources are fully open-weight and publicly available at huggingface.co/TildeAI/TildeOpen-30b. These outcomes demonstrate that careful data curation and balanced training strategies can substantially enhance multilingual model quality without increasing model size or training volume.",
      "published": "2026-03-09T10:03:17Z",
      "abstract_url": "http://arxiv.org/abs/2603.08182v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08182v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Is continuous CoT better suited for multi-lingual reasoning?",
      "authors": [
        "Ali Hamza Bashir",
        "Behzad Shomali",
        "Markus Frey",
        "Mehdi Ali",
        "Rafet Sifa",
        "David Berghaus"
      ],
      "abstract": "We investigate whether performing reasoning in a continuous latent space leads to more robust multilingual capabilities. We compare Continuous Chain-of-Thought (using the CODI framework) against standard supervised fine-tuning across five typologically diverse languages: English, Chinese, German, French, and Urdu. Our experiments on GSM8k and CommonsenseQA demonstrate that continuous reasoning significantly outperforms explicit reasoning on low-resource languages, particularly in zero-shot settings where the target language was not seen during training. Additionally, this approach achieves extreme efficiency, compressing reasoning traces by approximately $29\\times$ to $50\\times$. These findings indicate that continuous latent representations naturally exhibit greater language invariance, offering a scalable solution for cross-lingual reasoning.",
      "published": "2026-03-09T09:57:08Z",
      "abstract_url": "http://arxiv.org/abs/2603.08177v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08177v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "An explainable hybrid deep learning-enabled intelligent fault detection and diagnosis approach for automotive software systems validation",
      "authors": [
        "Mohammad Abboush",
        "Ehab Ghannoum",
        "Andreas Rausch"
      ],
      "abstract": "Advancements in data-driven machine learning have emerged as a pivotal element in supporting automotive software systems (ASSs) engineering across various levels of the V-development process. Duringsystemverificationandvalidation,theintegrationofanintelligent fault detection anddiagnosis (FDD) model with test recordings analysis process serves as a powerful tool for efficiency ensuring functional safety. However, the lack of interpretability of the black-box FDD models developed not only hinders understanding of the cause underlying the prediction, but also prevents the model from being adapted based on the prediction result. This, in turn, increases the computational cost required for developingacomplexFDDmodelandlimitsconfidenceinreal-timesafety-criticalapplications.To address this challenge, a novel explainable method for fault detection, identification, and localization is proposed in this article with the aim of providing a clear understanding of the logic behind the prediction outcome. To this end, a hybrid 1dCNN-GRU-based intelligent model was developed to analyze the recordings from the real-time validation process of ASSs. The employment of explainable AI techniques, i.e., IGs, DeepLIFT, Gradient SHAP, and DeepLIFT SHAP, was instrumental in enabling model adaptation and facilitating the root cause analysis (RCA). The proposed approach is applied to the real time dataset collected during a virtual test drive performed by the user on hardware in the loop system.",
      "published": "2026-03-09T09:46:28Z",
      "abstract_url": "http://arxiv.org/abs/2603.08165v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08165v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Gradually Excavating External Knowledge for Implicit Complex Question Answering",
      "authors": [
        "Chang Liu",
        "Xiaoguang Li",
        "Lifeng Shang",
        "Xin Jiang",
        "Qun Liu",
        "Edmund Y. Lam",
        "Ngai Wong"
      ],
      "abstract": "Recently, large language models (LLMs) have gained much attention for the emergence of human-comparable capabilities and huge potential. However, for open-domain implicit question-answering problems, LLMs may not be the ultimate solution due to the reasons of: 1) uncovered or out-of-date domain knowledge, 2) one-shot generation and hence restricted comprehensiveness. To this end, this work proposes a gradual knowledge excavation framework for open-domain complex question answering, where LLMs iteratively and actively acquire external information, and then reason based on acquired historical knowledge. Specifically, during each step of the solving process, the model selects an action to execute, such as querying external knowledge or performing a single logical reasoning step, to gradually progress toward a final answer. Our method can effectively leverage plug-and-play external knowledge and dynamically adjust the strategy for solving complex questions. Evaluated on the StrategyQA dataset, our method achieves 78.17% accuracy with less than 6% parameters of its competitors, setting new SOTA for ~10B-scale LLMs.",
      "published": "2026-03-09T09:28:42Z",
      "abstract_url": "http://arxiv.org/abs/2603.08148v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08148v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "DARC: Disagreement-Aware Alignment via Risk-Constrained Decoding",
      "authors": [
        "Mingxi Zou",
        "Jiaxiang Chen",
        "Junfan Li",
        "Langzhang Liang",
        "Qifan Wang",
        "Xu Yinghui",
        "Zenglin Xu"
      ],
      "abstract": "Preference-based alignment methods (e.g., RLHF, DPO) typically optimize a single scalar objective, implicitly averaging over heterogeneous human preferences. In practice, systematic annotator and user-group disagreement makes mean-reward maximization brittle and susceptible to proxy over-optimization. We propose **Disagreement-Aware Alignment via Risk-Constrained Decoding (DARC)**, a retraining-free inference-time method that frames response selection as distributionally robust, risk-sensitive decision making. Given multiple preference samples or scalable disagreement proxies, DARC reranks candidates by maximizing a *KL-robust (entropic)* satisfaction objective, and provides simple deployment controls that cap or penalize the corresponding entropic risk premium relative to the mean, enabling explicit risk budgets without retraining. We provide theoretical characterization linking this decoding rule to principled pessimism and KL-based distributionally robust optimization. Experiments on alignment benchmarks show that DARC reduces disagreement and tail risk while maintaining competitive average quality under noisy, heterogeneous feedback.",
      "published": "2026-03-09T09:21:29Z",
      "abstract_url": "http://arxiv.org/abs/2603.08145v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08145v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Foley-Flow: Coordinated Video-to-Audio Generation with Masked Audio-Visual Alignment and Dynamic Conditional Flows",
      "authors": [
        "Shentong Mo",
        "Yibing Song"
      ],
      "abstract": "Coordinated audio generation based on video inputs typically requires a strict audio-visual (AV) alignment, where both semantics and rhythmics of the generated audio segments shall correspond to those in the video frames. Previous studies leverage a two-stage design where the AV encoders are firstly aligned via contrastive learning, then the encoded video representations guide the audio generation process. We observe that both contrastive learning and global video guidance are effective in aligning overall AV semantics while limiting temporally rhythmic synchronization. In this work, we propose FoleyFlow to first align unimodal AV encoders via masked modeling training, where the masked audio segments are recovered under the guidance of the corresponding video segments. After training, the AV encoders which are separately pretrained using only unimodal data are aligned with semantic and rhythmic consistency. Then, we develop a dynamic conditional flow for the final audio generation. Built upon the efficient velocity flow generation framework, our dynamic conditional flow utilizes temporally varying video features as the dynamic condition to guide corresponding audio segment generations. To this end, we extract coherent semantic and rhythmic representations during masked AV alignment, and use this representation of video segments to guide audio generation temporally. Our audio results are evaluated on the standard benchmarks and largely surpass existing results under several metrics. The superior performance indicates that FoleyFlow is effective in generating coordinated audios that are both semantically and rhythmically coherent to various video sequences.",
      "published": "2026-03-09T09:06:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.08126v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08126v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.SD",
        "eess.AS"
      ]
    },
    {
      "title": "SaiVLA-0: Cerebrum--Pons--Cerebellum Tripartite Architecture for Compute-Aware Vision-Language-Action",
      "authors": [
        "Xiang Shi",
        "Wenlong Huang",
        "Menglin Zou",
        "Xinhai Sun"
      ],
      "abstract": "We revisit Vision-Language-Action through a neuroscience-inspired triad. Biologically, the Cerebrum provides stable high-level multimodal priors and remains frozen; the Pons Adapter integrates these cortical features with real-time proprioceptive inputs and compiles intent into execution-ready tokens; and the Cerebellum (ParaCAT) performs fast, parallel categorical decoding for online control, with hysteresis/EMA/temperature/entropy for stability. A fixed-ratio schedule and two-stage feature caching make the system compute-aware and reproducible. Inspired by active, foveated vision, our wrist ROIs are geometrically tied to the end-effector via calibrated projection, providing a movement-stabilized, high-resolution view that is sensitive to fine-grained pose changes and complements the global context of the main view. The design is modular: upgrading the Cerebrum only retrains the Pons; changing robots only trains the Cerebellum; cerebellum-only RL can further refine control without touching high-level semantics. As a concept-and-protocol paper with preliminary evidence, we outline a timing protocol under matched conditions (GPU, resolution, batch) to verify anticipated efficiency gains. We also report preliminary LIBERO evidence showing that split feature caching reduces training time (7.5h to 4.5h) and improves average success (86.5% to 92.5%) under official N1.5 head-only training, and that SaiVLA0 reaches 99.0% mean success.",
      "published": "2026-03-09T09:03:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.08124v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08124v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DC-W2S: Dual-Consensus Weak-to-Strong Training for Reliable Process Reward Modeling in Biological Reasoning",
      "authors": [
        "Chi-Min Chan",
        "Ehsan Hajiramezanali",
        "Xiner Li",
        "Edward De Brouwer",
        "Carl Edwards",
        "Wei Xue",
        "Sirui Han",
        "Yike Guo",
        "Gabriele Scalia"
      ],
      "abstract": "In scientific reasoning tasks, the veracity of the reasoning process is as critical as the final outcome. While Process Reward Models (PRMs) offer a solution to the coarse-grained supervision problems inherent in Outcome Reward Models (ORMs), their deployment is hindered by the prohibitive cost of obtaining expert-verified step-wise labels. This paper addresses the challenge of training reliable PRMs using abundant but noisy \"weak\" supervision. We argue that existing Weak-to-Strong Generalization (W2SG) theories lack prescriptive guidelines for selecting high-quality training signals from noisy data. To bridge this gap, we introduce the Dual-Consensus Weak-to-Strong (DC-W2S) framework. By intersecting Self-Consensus (SC) metrics among weak supervisors with Neighborhood-Consensus (NC) metrics in the embedding space, we stratify supervision signals into distinct reliability regimes. We then employ a curriculum of instance-level balanced sampling and label-level reliability-aware masking to guide the training process. We demonstrate that DC-W2S enables the training of robust PRMs for complex reasoning without exhaustive expert annotation, proving that strategic data curation is more effective than indiscriminate training on large-scale noisy datasets.",
      "published": "2026-03-09T08:36:55Z",
      "abstract_url": "http://arxiv.org/abs/2603.08095v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08095v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "In-Context Reinforcement Learning for Tool Use in Large Language Models",
      "authors": [
        "Yaoqi Ye",
        "Yiran Zhao",
        "Keyu Duan",
        "Zeyu Zheng",
        "Kenji Kawaguchi",
        "Cihang Xie",
        "Michael Qizhe Shieh"
      ],
      "abstract": "While large language models (LLMs) exhibit strong reasoning abilities, their performance on complex tasks is often constrained by the limitations of their internal knowledge. A compelling approach to overcome this challenge is to augment these models with external tools -- such as Python interpreters for mathematical computations or search engines for retrieving factual information. However, enabling models to use these tools effectively remains a significant challenge. Existing methods typically rely on cold-start pipelines that begin with supervised fine-tuning (SFT), followed by reinforcement learning (RL). These approaches often require substantial amounts of labeled data for SFT, which is expensive to annotate or synthesize. In this work, we propose In-Context Reinforcement Learning (ICRL), an RL-only framework that eliminates the need for SFT by leveraging few-shot prompting during the rollout stage of RL. Specifically, ICRL introduces in-context examples within the rollout prompts to teach the model how to invoke external tools. Furthermore, as training progresses, the number of in-context examples is gradually reduced, eventually reaching a zero-shot setting where the model learns to call tools independently. We conduct extensive experiments across a range of reasoning and tool-use benchmarks. Results show that ICRL achieves state-of-the-art performance, demonstrating its effectiveness as a scalable, data-efficient alternative to traditional SFT-based pipelines.",
      "published": "2026-03-09T08:06:18Z",
      "abstract_url": "http://arxiv.org/abs/2603.08068v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08068v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "S2S-FDD: Bridging Industrial Time Series and Natural Language for Explainable Zero-shot Fault Diagnosis",
      "authors": [
        "Baoxue Li",
        "Chunhui Zhao"
      ],
      "abstract": "Fault diagnosis is critical for the safe operation of industrial systems. Conventional diagnosis models typically produce abstract outputs such as anomaly scores or fault categories, failing to answer critical operational questions like \"Why\" or \"How to repair\". While large language models (LLMs) offer strong generalization and reasoning abilities, their training on discrete textual corpora creates a semantic gap when processing high-dimensional, temporal industrial signals. To address this challenge, we propose a Signals-to-Semantics fault diagnosis (S2S-FDD) framework that bridges high-dimensional sensor signals with natural language semantics through two key innovations: We first design a Signal-to-Semantic operator to convert abstract time-series signals into natural language summaries, capturing trends, periodicity, and deviations. Based on the descriptions, we design a multi-turn tree-structured diagnosis method to perform fault diagnosis by referencing historical maintenance documents and dynamically querying additional signals. The framework further supports human-in-the-loop feedback for continuous refinement. Experiments on the multiphase flow process show the feasibility and effectiveness of the proposed method for explainable zero-shot fault diagnosis.",
      "published": "2026-03-09T07:38:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.08048v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08048v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling",
      "authors": [
        "Dengcan Liu",
        "Fengkai Yang",
        "Xiaohan Wang",
        "Shurui Yan",
        "Jiajun Chai",
        "Jiahao Li",
        "Yikun Ban",
        "Zhendong Mao",
        "Wei Lin",
        "Guojun Yin"
      ],
      "abstract": "Reward modeling is essential for aligning Large Language Models(LLMs) with human preferences, yet conventional reward models suffer from poor interpretability and heavy reliance on costly expert annotations. While recent rubric-based approaches enhance evaluation transparency, they lack systematic quality control, yielding noisy and redundant criteria, failing to mitigate persistent biases (e.g., verbosity, position) in LLM evaluators, and creating a scalability-reliability trade-off. To address these limitations, we propose CDRRM (Contrast-Driven Rubric Reward Model), a framework built on a novel Contrast-then-Synthesis paradigm for high-quality rubric generation and guided preference judgment. CDRRM first conducts multi-dimensional contrastive profiling on preference pairs to identify causal discriminative factors, then synthesizes these insights into compact, context-aware rubrics to guide preference judg- ments. Extensive experiments on three authoritative benchmarks (RewardBench, RMBench, RMB) demonstrate that CDRRM achieves state-of-the-art performance across diverse domains and effectively mitigates aforementioned evaluation biases. Notably, our approach delivers exceptional data efficiency: training the rubric generator on only 3k high-quality samples empowers a frozen pre-trained judge model to outperform fully fine-tuned baselines. This work offers a scalable, interpretable, and data-efficient path for reward modeling.",
      "published": "2026-03-09T07:15:23Z",
      "abstract_url": "http://arxiv.org/abs/2603.08035v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08035v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "GCGNet: Graph-Consistent Generative Network for Time Series Forecasting with Exogenous Variables",
      "authors": [
        "Zhengyu Li",
        "Xiangfei Qiu",
        "Yuhan Zhu",
        "Xingjian Wu",
        "Jilin Hu",
        "Chenjuan Guo",
        "Bin Yang"
      ],
      "abstract": "Exogenous variables offer valuable supplementary information for predicting future endogenous variables. Forecasting with exogenous variables needs to consider both past-to-future dependencies (i.e., temporal correlations) and the influence of exogenous variables on endogenous variables (i.e., channel correlations). This is pivotal when future exogenous variables are available, because they may directly affect the future endogenous variables. Many methods have been proposed for time series forecasting with exogenous variables, focusing on modeling temporal and channel correlations. However, most of them use a two-step strategy, modeling temporal and channel correlations separately, which limits their ability to capture joint correlations across time and channels. Furthermore, in real-world scenarios, time series are frequently affected by various forms of noises, underscoring the critical importance of robustness in such correlations modeling. To address these limitations, we propose GCGNet, a Graph-Consistent Generative Network for time series forecasting with exogenous variables. Specifically, GCGNet first employs a Variational Generator to produce coarse predictions. A Graph Structure Aligner then further guides it by evaluating the consistency between the generated and true correlations, where the correlations are represented as graphs, and are robust to noises. Finally, a Graph Refiner is proposed to refine the predictions to prevent degeneration and improve accuracy. Extensive experiments on 12 real-world datasets demonstrate that GCGNet outperforms state-of-the-art baselines.",
      "published": "2026-03-09T07:11:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.08032v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08032v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "FedMomentum: Preserving LoRA Training Momentum in Federated Fine-Tuning",
      "authors": [
        "Peishen Yan",
        "Yang Hua",
        "Hao Wang",
        "Jiaru Zhang",
        "Xiaoyu Wu",
        "Tao Song",
        "Haibing Guan"
      ],
      "abstract": "Federated fine-tuning of large language models (LLMs) with low-rank adaptation (LoRA) offers a communication-efficient and privacy-preserving solution for task-specific adaptation. Naive aggregation of LoRA modules introduces noise due to mathematical incorrectness when averaging the downsampling and upsampling matrices independently. However, existing noise-free aggregation strategies inevitably compromise the structural expressiveness of LoRA, limiting its ability to retain client-specific adaptations by either improperly reconstructing the low-rank structure or excluding partially trainable components. We identify this problem as loss of training momentum, where LoRA updates fail to accumulate effectively across rounds, resulting in slower convergence and suboptimal performance. To address this, we propose FedMomentum, a novel framework that enables structured and momentum-preserving LoRA aggregation via singular value decomposition (SVD). Specifically, after aggregating low-rank updates in a mathematically correct manner, FedMomentum applies SVD to extract the dominant components that capture the main update directions. These components are used to reconstruct the LoRA modules with the same rank, while residual components can be retained and later merged into the backbone to preserve semantic information and ensure robustness. Extensive experiments across multiple tasks demonstrate that FedMomentum consistently outperforms prior state-of-the-art methods in convergence speed and final accuracy.",
      "published": "2026-03-09T06:43:17Z",
      "abstract_url": "http://arxiv.org/abs/2603.08014v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08014v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PIRA-Bench: A Transition from Reactive GUI Agents to GUI-based Proactive Intent Recommendation Agents",
      "authors": [
        "Yuxiang Chai",
        "Shunye Tang",
        "Han Xiao",
        "Rui Liu",
        "Hongsheng Li"
      ],
      "abstract": "Current Graphical User Interface (GUI) agents operate primarily under a reactive paradigm: a user must provide an explicit instruction for the agent to execute a task. However, an intelligent AI assistant should be proactive, which is capable of anticipating user intentions directly from continuous visual inputs, such as mobile or desktop screenshots, and offering timely recommendations without explicit user prompting. Transitioning to this proactive paradigm presents significant challenges. Real-world screen activity is rarely linear; it consists of long-horizon trajectories fraught with noisy browsing, meaningless actions, and multithreaded task-switching. To address this gap, we introduce PIRA-Bench (Proactive Intent Recommendation Agent Benchmark), a novel benchmark for evaluating multimodal large language models (MLLMs) on continuous, weakly-supervised visual inputs. Unlike reactive datasets, PIRA-Bench features complex trajectories with multiple interleaved intents and noisy segments with various user profile contexts, challenging agents to detect actionable events while fitting to user preferences. Furthermore, we propose the PIRF baseline, a memory-aware, state-tracking framework that empowers general MLLMs to manage multiple task threads and handle misleading visual inputs. PIRA-Bench serves as an initial step toward robust and proactive GUI-based personal assistants.",
      "published": "2026-03-09T06:41:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.08013v1",
      "pdf_url": "https://arxiv.org/pdf/2603.08013v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CMMR-VLN: Vision-and-Language Navigation via Continual Multimodal Memory Retrieval",
      "authors": [
        "Haozhou Li",
        "Xiangyu Dong",
        "Huiyan Jiang",
        "Yaoming Zhou",
        "Xiaoguang Ma"
      ],
      "abstract": "Although large language models (LLMs) are introduced into vision-and-language navigation (VLN) to improve instruction comprehension and generalization, existing LLM- based VLN lacks the ability to selectively recall and use relevant priori experiences to help navigation tasks, limiting their performance in long-horizon and unfamiliar scenarios. In this work, we propose CMMR-VLN (Continual Multimodal Memory Retrieval based VLN), a VLN framework that endows LLM agents with structured memory and reflection capabilities. Specifically, the CMMR-VLN constructs a multimodal experi- ence memory indexed by panoramic visual images and salient landmarks to retrieve relevant experiences during navigation, introduces a retrieved-augmented generation pipeline to mimick how experienced human navigators leverage priori knowledge, and incorporates a reflection-based memory update strategy that selectively stores complete successful paths and the key initial mistake in failure cases. Comprehensive tests illustrate average success rate improvements of 52.9%, 20.9% and 20.9%, and 200%, 50% and 50% over the NavGPT, the MapGPT, and the DiscussNav in simulation and real tests, respectively eluci- dating the great potential of the CMMR-VLN as a backbone VLN framework.",
      "published": "2026-03-09T06:02:50Z",
      "abstract_url": "http://arxiv.org/abs/2603.07997v1",
      "pdf_url": "https://arxiv.org/pdf/2603.07997v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "\\$OneMillion-Bench: How Far are Language Agents from Human Experts?",
      "authors": [
        "Qianyu Yang",
        "Yang Liu",
        "Jiaqi Li",
        "Jun Bai",
        "Hao Chen",
        "Kaiyuan Chen",
        "Tiliang Duan",
        "Jiayun Dong",
        "Xiaobo Hu",
        "Zixia Jia",
        "Yang Liu",
        "Tao Peng",
        "Yixin Ren",
        "Ran Tian",
        "Zaiyuan Wang",
        "Yanglihong Xiao",
        "Gang Yao",
        "Lingyue Yin",
        "Ge Zhang",
        "Chun Zhang",
        "Jianpeng Jiao",
        "Zilong Zheng",
        "Yuan Gong"
      ],
      "abstract": "As language models (LMs) evolve from chat assistants to long-horizon agents capable of multi-step reasoning and tool use, existing benchmarks remain largely confined to structured or exam-style tasks that fall short of real-world professional demands. To this end, we introduce \\$OneMillion-Bench \\$OneMillion-Bench, a benchmark of 400 expert-curated tasks spanning Law, Finance, Industry, Healthcare, and Natural Science, built to evaluate agents across economically consequential scenarios. Unlike prior work, the benchmark requires retrieving authoritative sources, resolving conflicting evidence, applying domain-specific rules, and making constraint decisions, where correctness depends as much on the reasoning process as the final answer. We adopt a rubric-based evaluation protocol scoring factual accuracy, logical coherence, practical feasibility, and professional compliance, focused on expert-level problems to ensure meaningful differentiation across agents. Together, \\$OneMillion-Bench provides a unified testbed for assessing agentic reliability, professional depth, and practical readiness in domain-intensive scenarios.",
      "published": "2026-03-09T05:32:42Z",
      "abstract_url": "http://arxiv.org/abs/2603.07980v1",
      "pdf_url": "https://arxiv.org/pdf/2603.07980v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning",
      "authors": [
        "Wei Yang",
        "Defu Cao",
        "Jiacheng Pang",
        "Muyan Weng",
        "Yan Liu"
      ],
      "abstract": "While scaling individual Large Language Models (LLMs) has delivered remarkable progress, the next frontier lies in scaling collaboration through multi-agent systems (MAS). However, purely autonomous MAS remain ''closed-world'' systems, constrained by the static knowledge horizon of pre-trained models. This limitation makes them brittle on tasks requiring knowledge beyond training data, often leading to collective failure under novel challenges. To address this, we propose the Human-In-the-Loop Multi-Agent Collaboration (HILA) framework, a principled paradigm for human--agent collaboration. HILA trains agents to learn a metacognitive policy that governs when to solve problems autonomously and when to defer to a human expert. To operationalize this policy, we introduce Dual-Loop Policy Optimization, which disentangles immediate decision-making from long-term capability growth. The inner loop applies Group Relative Policy Optimization (GRPO) with a cost-aware reward to optimize deferral decisions, while the outer loop implements continual learning, transforming expert feedback into high-quality supervised signals that strengthen the agent's reasoning ability. Experiments on challenging mathematical and problem-solving benchmarks show that HILA, equipped with Dual-Loop Policy Optimization, consistently outperforms advanced MAS, establishing a principled foundation for collaborative and continually improving agentic systems.",
      "published": "2026-03-09T05:18:07Z",
      "abstract_url": "http://arxiv.org/abs/2603.07972v1",
      "pdf_url": "https://arxiv.org/pdf/2603.07972v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "PSTNet: Physically-Structured Turbulence Network",
      "authors": [
        "Boris Kriuk",
        "Fedor Kriuk"
      ],
      "abstract": "Reliable real-time estimation of atmospheric turbulence intensity remains an open challenge for aircraft operating across diverse altitude bands, particularly over oceanic, polar, and data-sparse regions that lack operational nowcasting infrastructure. Classical spectral models encode climatological averages rather than the instantaneous atmospheric state, and generic ML regressors offer adaptivity but provide no guarantee that predictions respect fundamental scaling laws. This paper introduces the Physically-Structured Turbulence Network (PSTNet), a lightweight architecture that embeds physics directly into its structure. PSTNet couples four components: (i) a zero-parameter backbone derived from Monin-Obukhov theory, (ii) a regime-gated mixture of specialist sub-networks supervised by Richardson-number-derived soft targets, (iii) Feature-wise Linear Modulation layers conditioning hidden representations on local air-density ratio, and (iv) a Kolmogorov output layer enforcing inertial-subrange scaling as an architectural constraint. The entire model contains only 552 learnable parameters, requiring fewer than 2.5 kB of storage and executing in under 12s on a Cortex-M7 microcontroller. We validate PSTNet on 340 paired six-degree-of-freedom guidance simulations spanning three vehicle classes (Mach 2.8, 4.5, and 8.0) and six operational categories with real-time satellite weather ingestion. PSTNet achieves a mean miss-distance improvement of +2.8% with a 78% win rate and a statistically significant effect size. Our results demonstrate that encoding domain physics as architectural priors yields a more efficient and interpretable path to turbulence estimation accuracy than scaling model capacity, establishing PSTNet as a viable drop-in replacement for legacy look-up tables in resource-constrained, safety-critical on-board guidance systems.",
      "published": "2026-03-09T04:46:46Z",
      "abstract_url": "http://arxiv.org/abs/2603.07957v1",
      "pdf_url": "https://arxiv.org/pdf/2603.07957v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
