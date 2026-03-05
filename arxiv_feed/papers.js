const PAPERS_DATA = {
  "last_updated": "2026-03-05 02:44:21 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Monitoring Emergent Reward Hacking During Generation via Internal Activations",
      "authors": [
        "Patrick Wilhelm",
        "Thorsten Wittkopp",
        "Odej Kao"
      ],
      "abstract": "Fine-tuned large language models can exhibit reward-hacking behavior arising from emergent misalignment, which is difficult to detect from final outputs alone. While prior work has studied reward hacking at the level of completed responses, it remains unclear whether such behavior can be identified during generation. We propose an activation-based monitoring approach that detects reward-hacking signals from internal representations as a model generates its response. Our method trains sparse autoencoders on residual stream activations and applies lightweight linear classifiers to produce token-level estimates of reward-hacking activity. Across multiple model families and fine-tuning mixtures, we find that internal activation patterns reliably distinguish reward-hacking from benign behavior, generalize to unseen mixed-policy adapters, and exhibit model-dependent temporal structure during chain-of-thought reasoning. Notably, reward-hacking signals often emerge early, persist throughout reasoning, and can be amplified by increased test-time compute in the form of chain-of-thought prompting under weakly specified reward objectives. These results suggest that internal activation monitoring provides a complementary and earlier signal of emergent misalignment than output-based evaluation, supporting more robust post-deployment safety monitoring for fine-tuned language models.",
      "published": "2026-03-04T13:44:24Z",
      "abstract_url": "http://arxiv.org/abs/2603.04069v1",
      "pdf_url": "https://arxiv.org/pdf/2603.04069v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Inference-Time Toxicity Mitigation in Protein Language Models",
      "authors": [
        "Manuel Fernández Burda",
        "Santiago Aranguri",
        "Iván Arcuschin Moreno",
        "Enzo Ferrante"
      ],
      "abstract": "Protein language models (PLMs) are becoming practical tools for de novo protein design, yet their dual-use potential raises safety concerns. We show that domain adaptation to specific taxonomic groups can elicit toxic protein generation, even when toxicity is not the training objective. To address this, we adapt Logit Diff Amplification (LDA) as an inference-time control mechanism for PLMs. LDA modifies token probabilities by amplifying the logit difference between a baseline model and a toxicity-finetuned model, requiring no retraining. Across four taxonomic groups, LDA consistently reduces predicted toxicity rate (measured via ToxDL2) below the taxon-finetuned baseline while preserving biological plausibility. We evaluate quality using Fréchet ESM Distance and predicted foldability (pLDDT), finding that LDA maintains distributional similarity to natural proteins and structural viability (unlike activation-based steering methods that tend to degrade sequence properties). Our results demonstrate that LDA provides a practical safety knob for protein generators that mitigates elicited toxicity while retaining generative quality.",
      "published": "2026-03-04T13:28:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.04045v1",
      "pdf_url": "https://arxiv.org/pdf/2603.04045v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "DQE-CIR: Distinctive Query Embeddings through Learnable Attribute Weights and Target Relative Negative Sampling in Composed Image Retrieval",
      "authors": [
        "Geon Park",
        "Ji-Hoon Park",
        "Seong-Whan Lee"
      ],
      "abstract": "Composed image retrieval (CIR) addresses the task of retrieving a target image by jointly interpreting a reference image and a modification text that specifies the intended change. Most existing methods are still built upon contrastive learning frameworks that treat the ground truth image as the only positive instance and all remaining images as negatives. This strategy inevitably introduces relevance suppression, where semantically related yet valid images are incorrectly pushed away, and semantic confusion, where different modification intents collapse into overlapping regions of the embedding space. As a result, the learned query representations often lack discriminativeness, particularly at fine-grained attribute modifications. To overcome these limitations, we propose distinctive query embeddings through learnable attribute weights and target relative negative sampling (DQE-CIR), a method designed to learn distinctive query embeddings by explicitly modeling target relative relevance during training. DQE-CIR incorporates learnable attribute weighting to emphasize distinctive visual features conditioned on the modification text, enabling more precise feature alignment between language and vision. Furthermore, we introduce target relative negative sampling, which constructs a target relative similarity distribution and selects informative negatives from a mid-zone region that excludes both easy negatives and ambiguous false negatives. This strategy enables more reliable retrieval for fine-grained attribute changes by improving query discriminativeness and reducing confusion caused by semantically similar but irrelevant candidates.",
      "published": "2026-03-04T13:17:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.04037v1",
      "pdf_url": "https://arxiv.org/pdf/2603.04037v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Multi-Dimensional Quality Scoring Framework for Decentralized LLM Inference with Proof of Quality",
      "authors": [
        "Arther Tian",
        "Alex Ding",
        "Frank Chen",
        "Simon Wu",
        "Aaron Chan"
      ],
      "abstract": "Decentralized large language model (LLM) inference networks can pool heterogeneous compute to scale serving, but they require lightweight and incentive-compatible mechanisms to assess output quality. Prior work introduced cost-aware Proof of Quality (PoQ) and adaptive robust PoQ to allocate rewards under evaluator heterogeneity and adversarial behavior. In this paper, we focus on the quality signal itself and propose a multi-dimensional quality scoring framework that decomposes output quality into modular dimensions, including model and cost priors, structure quality, semantic quality, query-output alignment, and agreement/uncertainty. Using logged outputs from QA and summarization tasks, we systematically audit dimension reliability and show that seemingly reasonable dimensions can be task-dependent and even negatively correlated with reference quality without calibration. While the default composite underperforms a strong single semantic evaluator, ablations reveal that removing unreliable dimensions and re-normalizing weights yields a calibrated composite that matches or exceeds the best single- evaluator and consensus baselines. Finally, we integrate the composite score as a drop-in quality signal in PoQ and demonstrate complementary benefits with robust aggregation and adaptive trust weighting under adversarial evaluator attacks.",
      "published": "2026-03-04T13:05:46Z",
      "abstract_url": "http://arxiv.org/abs/2603.04028v1",
      "pdf_url": "https://arxiv.org/pdf/2603.04028v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Discriminative Perception via Anchored Description for Reasoning Segmentation",
      "authors": [
        "Tao Yang",
        "Qing Zhou",
        "Yanliang Li",
        "Qi Wang"
      ],
      "abstract": "Reasoning segmentation increasingly employs reinforcement learning to generate explanatory reasoning chains that guide Multimodal Large Language Models. While these geometric rewards are primarily confined to guiding the final localization, they are incapable of discriminating whether the reasoning process remains anchored on the referred region or strays into irrelevant context. Lacking this discriminative guidance, the model's reasoning often devolves into unfocused and verbose chains that ultimately fail to disambiguate and perceive the target in complex scenes. This suggests a need to complement the RL objective with Discriminative Perception, an ability to actively distinguish a target from its context. To realize this, we propose DPAD to compel the model to generate a descriptive caption of the referred object, which is then used to explicitly discriminate by contrasting the caption's semantic relevance to the referred object against the wider context. By optimizing for this discriminative capability, the model is forced to focus on the unique attributes of the target, leading to a more converged and efficient reasoning chain. The descriptive caption also serves as an interpretability rationale that aligns with the segmentation. Experiments on the benchmarks confirm the validity of our approach, delivering substantial performance gains, with the cIoU on ReasonSeg increasing by 3.09% and the reasoning chain length decreasing by approximately 42%. Code is available at https://github.com/mrazhou/DPAD",
      "published": "2026-03-04T12:46:27Z",
      "abstract_url": "http://arxiv.org/abs/2603.04002v1",
      "pdf_url": "https://arxiv.org/pdf/2603.04002v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Spectral Surgery: Training-Free Refinement of LoRA via Gradient-Guided Singular Value Reweighting",
      "authors": [
        "Zailong Tian",
        "Yanzhe Chen",
        "Zhuoheng Han",
        "Lizi Liao"
      ],
      "abstract": "Low-Rank Adaptation (LoRA) improves downstream performance by restricting task updates to a low-rank parameter subspace, yet how this limited capacity is allocated within a trained adapter remains unclear. Through a geometric and empirical study across multiple tasks and backbones, we find that trained LoRA updates often exhibit an inefficient spectrum: task effects concentrate in a small subset of singular directions, while many remaining components are neutral or detrimental, motivating post-hoc refinement within the learned subspace. We propose Spectral Surgery, a training-free refinement that decomposes a LoRA update with SVD, estimates per-component sensitivity using gradients on a small calibration set, and reweights singular values under a magnitude constraint while keeping the learned directions fixed. Across Llama-3.1-8B and Qwen3-8B on four benchmarks, Spectral Surgery yields consistent gains (up to +4.4 points on CommonsenseQA and +2.4 pass@1 on HumanEval) by adjusting only $\\approx 1{,}000$ scalar coefficients. These results demonstrate that SVD-structured, low-cost parameter editing can serve as a practical route to improving trained LoRA adapters in a purely post-hoc manner.",
      "published": "2026-03-04T12:38:36Z",
      "abstract_url": "http://arxiv.org/abs/2603.03995v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03995v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Upholding Epistemic Agency: A Brouwerian Assertibility Constraint for Responsible AI",
      "authors": [
        "Michael Jülich"
      ],
      "abstract": "Generative AI can convert uncertainty into authoritative-seeming verdicts, displacing the justificatory work on which democratic epistemic agency depends. As a corrective, I propose a Brouwer-inspired assertibility constraint for responsible AI: in high-stakes domains, systems may assert or deny claims only if they can provide a publicly inspectable and contestable certificate of entitlement; otherwise they must return \"Undetermined\". This constraint yields a three-status interface semantics (Asserted, Denied, Undetermined) that cleanly separates internal entitlement from public standing while connecting them via the certificate as a boundary object. It also produces a time-indexed entitlement profile that is stable under numerical refinement yet revisable as the public record changes. I operationalize the constraint through decision-layer gating of threshold and argmax outputs, using internal witnesses (e.g., sound bounds or separation margins) and an output contract with reason-coded abstentions. A design lemma shows that any total, certificate-sound binary interface already decides the deployed predicate on its declared scope, so \"Undetermined\" is not a tunable reject option but a mandatory status whenever no forcing witness is available. By making outputs answerable to challengeable warrants rather than confidence alone, the paper aims to preserve epistemic agency where automated speech enters public justification.",
      "published": "2026-03-04T12:14:21Z",
      "abstract_url": "http://arxiv.org/abs/2603.03971v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03971v1",
      "categories": [
        "cs.CY",
        "cs.AI",
        "cs.LG",
        "cs.LO"
      ]
    },
    {
      "title": "TFWaveFormer: Temporal-Frequency Collaborative Multi-level Wavelet Transformer for Dynamic Link Prediction",
      "authors": [
        "Hantong Feng",
        "Yonggang Wu",
        "Duxin Chen",
        "Wenwu Yu"
      ],
      "abstract": "Dynamic link prediction plays a crucial role in diverse applications including social network analysis, communication forecasting, and financial modeling. While recent Transformer-based approaches have demonstrated promising results in temporal graph learning, their performance remains limited when capturing complex multi-scale temporal dynamics. In this paper, we propose TFWaveFormer, a novel Transformer architecture that integrates temporal-frequency analysis with multi-resolution wavelet decomposition to enhance dynamic link prediction. Our framework comprises three key components: (i) a temporal-frequency coordination mechanism that jointly models temporal and spectral representations, (ii) a learnable multi-resolution wavelet decomposition module that adaptively extracts multi-scale temporal patterns through parallel convolutions, replacing traditional iterative wavelet transforms, and (iii) a hybrid Transformer module that effectively fuses local wavelet features with global temporal dependencies. Extensive experiments on benchmark datasets demonstrate that TFWaveFormer achieves state-of-the-art performance, outperforming existing Transformer-based and hybrid models by significant margins across multiple metrics. The superior performance of TFWaveFormer validates the effectiveness of combining temporal-frequency analysis with wavelet decomposition in capturing complex temporal dynamics for dynamic link prediction tasks.",
      "published": "2026-03-04T11:47:57Z",
      "abstract_url": "http://arxiv.org/abs/2603.03963v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03963v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "GIPO: Gaussian Importance Sampling Policy Optimization",
      "authors": [
        "Chengxuan Lu",
        "Zhenquan Zhang",
        "Shukuan Wang",
        "Qunzhi Lin",
        "Baigui Sun",
        "Yang Liu"
      ],
      "abstract": "Post-training with reinforcement learning (RL) has recently shown strong promise for advancing multimodal agents beyond supervised imitation. However, RL remains limited by poor data efficiency, particularly in settings where interaction data are scarce and quickly become outdated. To address this challenge, GIPO (Gaussian Importance sampling Policy Optimization) is proposed as a policy optimization objective based on truncated importance sampling, replacing hard clipping with a log-ratio-based Gaussian trust weight to softly damp extreme importance ratios while maintaining non-zero gradients. Theoretical analysis shows that GIPO introduces an implicit, tunable constraint on the update magnitude, while concentration bounds guarantee robustness and stability under finite-sample estimation. Experimental results show that GIPO achieves state-of-the-art performance among clipping-based baselines across a wide range of replay buffer sizes, from near on-policy to highly stale data, while exhibiting superior bias--variance trade-off, high training stability and improved sample efficiency.",
      "published": "2026-03-04T11:34:59Z",
      "abstract_url": "http://arxiv.org/abs/2603.03955v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03955v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Selecting Offline Reinforcement Learning Algorithms for Stochastic Network Control",
      "authors": [
        "Nicolas Helson",
        "Pegah Alizadeh",
        "Anastasios Giovanidis"
      ],
      "abstract": "Offline Reinforcement Learning (RL) is a promising approach for next-generation wireless networks, where online exploration is unsafe and large amounts of operational data can be reused across the model lifecycle. However, the behavior of offline RL algorithms under genuinely stochastic dynamics -- inherent to wireless systems due to fading, noise, and traffic mobility -- remains insufficiently understood. We address this gap by evaluating Bellman-based (Conservative Q-Learning), sequence-based (Decision Transformers), and hybrid (Critic-Guided Decision Transformers) offline RL methods in an open-access stochastic telecom environment (mobile-env). Our results show that Conservative Q-Learning consistently produces more robust policies across different sources of stochasticity, making it a reliable default choice in lifecycle-driven AI management frameworks. Sequence-based methods remain competitive and can outperform Bellman-based approaches when sufficient high-return trajectories are available. These findings provide practical guidance for offline RL algorithm selection in AI-driven network control pipelines, such as O-RAN and future 6G functions, where robustness and data availability are key operational constraints.",
      "published": "2026-03-04T10:41:10Z",
      "abstract_url": "http://arxiv.org/abs/2603.03932v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03932v1",
      "categories": [
        "cs.NI",
        "cs.AI",
        "cs.LG",
        "cs.PF",
        "eess.SY"
      ]
    },
    {
      "title": "BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning",
      "authors": [
        "Yuhan Xie",
        "Chen Lyu"
      ],
      "abstract": "Model Merging (MM) has emerged as a scalable paradigm for multi-task learning (MTL), enabling multiple task-specific models to be integrated without revisiting the original training data. Despite recent progress, the reliability of MM under test-time distribution shift remains insufficiently understood. Most existing MM methods typically assume that test data are clean and distributionally aligned with both the training and auxiliary sources. However, this assumption rarely holds in practice, often resulting in biased predictions with degraded generalization. To address this issue, we present BD-Merging, a bias-aware unsupervised model merging framework that explicitly models uncertainty to achieve adaptive reliability under distribution shift. First, BD-Merging introduces a joint evidential head that learns uncertainty over a unified label space, capturing cross-task semantic dependencies in MM. Second, building upon this evidential foundation, we propose an Adjacency Discrepancy Score (ADS) that quantifies evidential alignment among neighboring samples. Third, guided by ADS, a discrepancy-aware contrastive learning mechanism refines the merged representation by aligning consistent samples and separating conflicting ones. Combined with general unsupervised learning, this process trains a debiased router that adaptively allocates task-specific or layer-specific weights on a per-sample basis, effectively mitigating the adverse effects of distribution shift. Extensive experiments across diverse tasks demonstrate that BD-Merging achieves superior effectiveness and robustness compared to state-of-the-art MM baselines.",
      "published": "2026-03-04T10:27:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.03920v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03920v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking Role-Playing Evaluation: Anonymous Benchmarking and a Systematic Study of Personality Effects",
      "authors": [
        "Ji-Lun Peng",
        "Yun-Nung Chen"
      ],
      "abstract": "Large language models (LLMs) have demonstrated significant potential in developing Role-Playing Agents (RPAs). However, current research primarily evaluates RPAs using famous fictional characters, allowing models to rely on memory associated with character names. This dependency creates a bias that limits the generalization of RPAs to unseen personas. To address this issue, we propose an anonymous evaluation method. Experiments across multiple benchmarks reveal that anonymization significantly degrades role-playing performance, confirming that name exposure carries implicit information. Furthermore, we investigate personality augmentation to enhance role fidelity under anonymous setting. We systematically compare the efficacy of personality traits derived from human annotations versus those self-generated by the model. Our results demonstrate that incorporating personality information consistently improves RPA performance. Crucially, self-generated personalities achieve performance comparable to human-annotated ones. This work establishes a fairer evaluation protocol and validates a scalable, personality-enhanced framework for constructing robust RPAs.",
      "published": "2026-03-04T10:24:02Z",
      "abstract_url": "http://arxiv.org/abs/2603.03915v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03915v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "PatchDecomp: Interpretable Patch-Based Time Series Forecasting",
      "authors": [
        "Hiroki Tomioka",
        "Genta Yoshimura"
      ],
      "abstract": "Time series forecasting, which predicts future values from past observations, plays a central role in many domains and has driven the development of highly accurate neural network models. However, the complexity of these models often limits human understanding of the rationale behind their predictions. We propose PatchDecomp, a neural network-based time series forecasting method that achieves both high accuracy and interpretability. PatchDecomp divides input time series into subsequences (patches) and generates predictions by aggregating the contributions of each patch. This enables clear attribution of each patch, including those from exogenous variables, to the final prediction. Experiments on multiple benchmark datasets demonstrate that PatchDecomp provides predictive performance comparable to recent forecasting methods. Furthermore, we show that the model's explanations not only influence predicted values quantitatively but also offer qualitative interpretability through visualization of patch-wise contributions.",
      "published": "2026-03-04T10:04:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.03902v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03902v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "IROSA: Interactive Robot Skill Adaptation using Natural Language",
      "authors": [
        "Markus Knauer",
        "Samuel Bustamante",
        "Thomas Eiband",
        "Alin Albu-Schäffer",
        "Freek Stulp",
        "João Silvério"
      ],
      "abstract": "Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.",
      "published": "2026-03-04T09:54:09Z",
      "abstract_url": "http://arxiv.org/abs/2603.03897v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03897v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CL",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "CzechTopic: A Benchmark for Zero-Shot Topic Localization in Historical Czech Documents",
      "authors": [
        "Martin Kostelník",
        "Michal Hradiš",
        "Martin Dočekal"
      ],
      "abstract": "Topic localization aims to identify spans of text that express a given topic defined by a name and description. To study this task, we introduce a human-annotated benchmark based on Czech historical documents, containing human-defined topics together with manually annotated spans and supporting evaluation at both document and word levels. Evaluation is performed relative to human agreement rather than a single reference annotation. We evaluate a diverse range of large language models alongside BERT-based models fine-tuned on a distilled development dataset. Results reveal substantial variability among LLMs, with performance ranging from near-human topic detection to pronounced failures in span localization. While the strongest models approach human agreement, the distilled token embedding models remain competitive despite their smaller scale. The dataset and evaluation framework are publicly available at: https://github.com/dcgm/czechtopic.",
      "published": "2026-03-04T09:35:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.03884v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03884v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Structure-Aware Distributed Backdoor Attacks in Federated Learning",
      "authors": [
        "Wang Jian",
        "Shen Hong",
        "Ke Wei",
        "Liu Xue Hua"
      ],
      "abstract": "While federated learning protects data privacy, it also makes the model update process vulnerable to long-term stealthy perturbations. Existing studies on backdoor attacks in federated learning mainly focus on trigger design or poisoning strategies, typically assuming that identical perturbations behave similarly across different model architectures. This assumption overlooks the impact of model structure on perturbation effectiveness. From a structure-aware perspective, this paper analyzes the coupling relationship between model architectures and backdoor perturbations. We introduce two metrics, Structural Responsiveness Score (SRS) and Structural Compatibility Coefficient (SCC), to measure a model's sensitivity to perturbations and its preference for fractal perturbations. Based on these metrics, we develop a structure-aware fractal perturbation injection framework (TFI) to study the role of architectural properties in the backdoor injection process. Experimental results show that model architecture significantly influences the propagation and aggregation of perturbations. Networks with multi-path feature fusion can amplify and retain fractal perturbations even under low poisoning ratios, while models with low structural compatibility constrain their effectiveness. Further analysis reveals a strong correlation between SCC and attack success rate, suggesting that SCC can predict perturbation survivability. These findings highlight that backdoor behaviors in federated learning depend not only on perturbation design or poisoning intensity but also on the interaction between model architecture and aggregation mechanisms, offering new insights for structure-aware defense design.",
      "published": "2026-03-04T09:19:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.03865v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03865v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "In-Context Environments Induce Evaluation-Awareness in Language Models",
      "authors": [
        "Maheep Chaudhary"
      ],
      "abstract": "Humans often become more self-aware under threat, yet can lose self-awareness when absorbed in a task; we hypothesize that language models exhibit environment-dependent \\textit{evaluation awareness}. This raises concerns that models could strategically underperform, or \\textit{sandbag}, to avoid triggering capability-limiting interventions such as unlearning or shutdown. Prior work demonstrates sandbagging under hand-crafted prompts, but this underestimates the true vulnerability ceiling. We introduce a black-box adversarial optimization framework treating the in-context prompt as an optimizable environment, and develop two approaches to characterize sandbagging: (1) measuring whether models expressing intent to underperform can actually execute it across different task structures, and (2) causally isolating whether underperformance is driven by genuine evaluation-aware reasoning or shallow prompt-following. Evaluating Claude-3.5-Haiku, GPT-4o-mini, and Llama-3.3-70B across four benchmarks (Arithmetic, GSM8K, MMLU, and HumanEval), optimized prompts induce up to 94 percentage point (pp) degradation on arithmetic (GPT-4o-mini: 97.8\\%$\\rightarrow$4.0\\%), far exceeding hand-crafted baselines which produce near-zero behavioral change. Code generation exhibits model-dependent resistance: Claude degrades only 0.6pp, while Llama's accuracy drops to 0\\%. The intent -- execution gap reveals a monotonic resistance ordering: Arithmetic $<$ GSM8K $<$ MMLU, demonstrating that vulnerability is governed by task structure rather than prompt strength. CoT causal intervention confirms that 99.3\\% of sandbagging is causally driven by verbalized eval-aware reasoning, ruling out shallow instruction-following. These findings demonstrate that adversarially optimized prompts pose a substantially greater threat to evaluation reliability than previously understood.",
      "published": "2026-03-04T08:22:02Z",
      "abstract_url": "http://arxiv.org/abs/2603.03824v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03824v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "SWE-CI: Evaluating Agent Capabilities in Maintaining Codebases via Continuous Integration",
      "authors": [
        "Jialong Chen",
        "Xander Xu",
        "Hu Wei",
        "Chuan Chen",
        "Bing Zhao"
      ],
      "abstract": "Large language model (LLM)-powered agents have demonstrated strong capabilities in automating software engineering tasks such as static bug fixing, as evidenced by benchmarks like SWE-bench. However, in the real world, the development of mature software is typically predicated on complex requirement changes and long-term feature iterations -- a process that static, one-shot repair paradigms fail to capture. To bridge this gap, we propose \\textbf{SWE-CI}, the first repository-level benchmark built upon the Continuous Integration loop, aiming to shift the evaluation paradigm for code generation from static, short-term \\textit{functional correctness} toward dynamic, long-term \\textit{maintainability}. The benchmark comprises 100 tasks, each corresponding on average to an evolution history spanning 233 days and 71 consecutive commits in a real-world code repository. SWE-CI requires agents to systematically resolve these tasks through dozens of rounds of analysis and coding iterations. SWE-CI provides valuable insights into how well agents can sustain code quality throughout long-term evolution.",
      "published": "2026-03-04T08:20:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.03823v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03823v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Fairness Begins with State: Purifying Latent Preferences for Hierarchical Reinforcement Learning in Interactive Recommendation",
      "authors": [
        "Yun Lu",
        "Xiaoyu Shi",
        "Hong Xie",
        "Xiangyu Zhao",
        "Mingsheng Shang"
      ],
      "abstract": "Interactive recommender systems (IRS) are increasingly optimized with Reinforcement Learning (RL) to capture the sequential nature of user-system dynamics. However, existing fairness-aware methods often suffer from a fundamental oversight: they assume the observed user state is a faithful representation of true preferences. In reality, implicit feedback is contaminated by popularity-driven noise and exposure bias, creating a distorted state that misleads the RL agent. We argue that the persistent conflict between accuracy and fairness is not merely a reward-shaping issue, but a state estimation failure. In this work, we propose \\textbf{DSRM-HRL}, a framework that reformulates fairness-aware recommendation as a latent state purification problem followed by decoupled hierarchical decision-making. We introduce a Denoising State Representation Module (DSRM) based on diffusion models to recover the low-entropy latent preference manifold from high-entropy, noisy interaction histories. Built upon this purified state, a Hierarchical Reinforcement Learning (HRL) agent is employed to decouple conflicting objectives: a high-level policy regulates long-term fairness trajectories, while a low-level policy optimizes short-term engagement under these dynamic constraints. Extensive experiments on high-fidelity simulators (KuaiRec, KuaiRand) demonstrate that DSRM-HRL effectively breaks the \"rich-get-richer\" feedback loop, achieving a superior Pareto frontier between recommendation utility and exposure equity.",
      "published": "2026-03-04T08:14:21Z",
      "abstract_url": "http://arxiv.org/abs/2603.03820v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03820v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning",
      "authors": [
        "Huihan Liu",
        "Changyeon Kim",
        "Bo Liu",
        "Minghuan Liu",
        "Yuke Zhu"
      ],
      "abstract": "Continual learning is a long-standing challenge in robot policy learning, where a policy must acquire new skills over time without catastrophically forgetting previously learned ones. While prior work has extensively studied continual learning in relatively small behavior cloning (BC) policy models trained from scratch, its behavior in modern large-scale pretrained Vision-Language-Action (VLA) models remains underexplored. In this work, we found that pretrained VLAs are remarkably resistant to forgetting compared with smaller policy models trained from scratch. Simple Experience Replay (ER) works surprisingly well on VLAs, sometimes achieving zero forgetting even with a small replay data size. Our analysis reveals that pretraining plays a critical role in downstream continual learning performance: large pretrained models mitigate forgetting with a small replay buffer size while maintaining strong forward learning capabilities. Furthermore, we found that VLAs can retain relevant knowledge from prior tasks despite performance degradation during learning new tasks. This knowledge retention enables rapid recovery of seemingly forgotten skills through finetuning. Together, these insights imply that large-scale pretraining fundamentally changes the dynamics of continual learning, enabling models to continually acquire new skills over time with simple replay. Code and more information can be found at https://ut-austin-rpl.github.io/continual-vla",
      "published": "2026-03-04T08:03:13Z",
      "abstract_url": "http://arxiv.org/abs/2603.03818v1",
      "pdf_url": "https://arxiv.org/pdf/2603.03818v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    }
  ]
};
