const PAPERS_DATA = {
  "last_updated": "2026-03-18 02:52:26 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "ManiTwin: Scaling Data-Generation-Ready Digital Object Dataset to 100K",
      "authors": [
        "Kaixuan Wang",
        "Tianxing Chen",
        "Jiawei Liu",
        "Honghao Su",
        "Shaolong Zhu",
        "Minxuan Wang",
        "Zixuan Li",
        "Yue Chen",
        "Huan-ang Gao",
        "Yusen Qin",
        "Jiawei Wang",
        "Qixuan Zhang",
        "Lan Xu",
        "Jingyi Yu",
        "Yao Mu",
        "Ping Luo"
      ],
      "abstract": "Learning in simulation provides a useful foundation for scaling robotic manipulation capabilities. However, this paradigm often suffers from a lack of data-generation-ready digital assets, in both scale and diversity. In this work, we present ManiTwin, an automated and efficient pipeline for generating data-generation-ready digital object twins. Our pipeline transforms a single image into simulation-ready and semantically annotated 3D asset, enabling large-scale robotic manipulation data generation. Using this pipeline, we construct ManiTwin-100K, a dataset containing 100K high-quality annotated 3D assets. Each asset is equipped with physical properties, language descriptions, functional annotations, and verified manipulation proposals. Experiments demonstrate that ManiTwin provides an efficient asset synthesis and annotation workflow, and that ManiTwin-100K offers high-quality and diverse assets for manipulation data generation, random scene synthesis, and VQA data generation, establishing a strong foundation for scalable simulation data synthesis and policy learning. Our webpage is available at https://manitwin.github.io/.",
      "published": "2026-03-17T17:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2603.16866v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16866v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models",
      "authors": [
        "Tianyu Xie",
        "Jinfa Huang",
        "Yuexiao Ma",
        "Rongfang Luo",
        "Yan Yang",
        "Wang Chen",
        "Yuhui Zeng",
        "Ruize Fang",
        "Yixuan Zou",
        "Xiawu Zheng",
        "Jiebo Luo",
        "Rongrong Ji"
      ],
      "abstract": "Omni-modal large language models (OLMs) redefine human-machine interaction by natively integrating audio, vision, and text. However, existing OLM benchmarks remain anchored to static, accuracy-centric tasks, leaving a critical gap in assessing social interactivity, the fundamental capacity to navigate dynamic cues in natural dialogues. To this end, we propose SocialOmni, a comprehensive benchmark that operationalizes the evaluation of this conversational interactivity across three core dimensions: (i) speaker separation and identification (who is speaking), (ii) interruption timing control (when to interject), and (iii) natural interruption generation (how to phrase the interruption). SocialOmni features 2,000 perception samples and a quality-controlled diagnostic set of 209 interaction-generation instances with strict temporal and contextual constraints, complemented by controlled audio-visual inconsistency scenarios to test model robustness. We benchmarked 12 leading OLMs, which uncovers significant variance in their social-interaction capabilities across models. Furthermore, our analysis reveals a pronounced decoupling between a model's perceptual accuracy and its ability to generate contextually appropriate interruptions, indicating that understanding-centric metrics alone are insufficient to characterize conversational social competence. More encouragingly, these diagnostics from SocialOmni yield actionable signals for bridging the perception-interaction divide in future OLMs.",
      "published": "2026-03-17T17:58:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.16859v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16859v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Unifying Optimization and Dynamics to Parallelize Sequential Computation: A Guide to Parallel Newton Methods for Breaking Sequential Bottlenecks",
      "authors": [
        "Xavier Gonzalez"
      ],
      "abstract": "Massively parallel hardware (GPUs) and long sequence data have made parallel algorithms essential for machine learning at scale. Yet dynamical systems, like recurrent neural networks and Markov chain Monte Carlo, were thought to suffer from sequential bottlenecks. Recent work showed that dynamical systems can in fact be parallelized across the sequence length by reframing their evaluation as a system of nonlinear equations, which can be solved with Newton's method using a parallel associative scan. However, these parallel Newton methods struggled with limitations, primarily inefficiency, instability, and lack of convergence guarantees. This thesis addresses these limitations with methodological and theoretical contributions, drawing particularly from optimization. Methodologically, we develop scalable and stable parallel Newton methods, based on quasi-Newton and trust-region approaches. The quasi-Newton methods are faster and more memory efficient, while the trust-region approaches are significantly more stable. Theoretically, we unify many fixed-point methods into our parallel Newton framework, including Picard and Jacobi iterations. We establish a linear convergence rate for these techniques that depends on the method's approximation accuracy and stability. Moreover, we give a precise condition, rooted in dynamical stability, that characterizes when parallelization provably accelerates a dynamical system and when it cannot. Specifically, the sign of the Largest Lyapunov Exponent of a dynamical system determines whether or not parallel Newton methods converge quickly. In sum, this thesis unlocks scalable and stable methods for parallelizing sequential computation, and provides a firm theoretical basis for when such techniques will and will not work. This thesis also serves as a guide to parallel Newton methods for researchers who want to write the next chapter in this ongoing story.",
      "published": "2026-03-17T17:55:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.16850v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16850v1",
      "categories": [
        "math.NA",
        "cs.AI",
        "cs.DC",
        "math.DS",
        "math.OC"
      ]
    },
    {
      "title": "Internalizing Agency from Reflective Experience",
      "authors": [
        "Rui Ge",
        "Yichao Fu",
        "Yuyang Qian",
        "Junda Su",
        "Yiming Zhao",
        "Peng Zhao",
        "Hao Zhang"
      ],
      "abstract": "Large language models are increasingly deployed as autonomous agents that must plan, act, and recover from mistakes through long-horizon interaction with environments that provide rich feedback. However, prevailing outcome-driven post-training methods (e.g., RL with verifiable rewards) primarily optimize final success signals, leaving rich environment feedback underutilized. Consequently, they often lead to distribution sharpening: the policy becomes better at reproducing a narrow set of already-successful behaviors, while failing to improve the feedback-grounded agency needed to expand problem-solving capacity (e.g., Pass@k) in long-horizon settings. To address this, we propose LEAFE (Learning Feedback-Grounded Agency from Reflective Experience), a framework that internalizes recovery agency from reflective experience. Specifically, during exploration, the agent summarizes environment feedback into actionable experience, backtracks to earlier decision points, and explores alternative branches with revised actions. We then distill these experience-guided corrections into the model through supervised fine-tuning, enabling the policy to recover more effectively in future interactions. Across a diverse set of interactive coding and agentic tasks under fixed interaction budgets, LEAFE consistently improves Pass@1 over the base model and achieves higher Pass@k than outcome-driven baselines (GRPO) and experience-based methods such as Early Experience, with gains of up to 14% on Pass@128.",
      "published": "2026-03-17T17:50:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.16843v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16843v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Prompt Programming for Cultural Bias and Alignment of Large Language Models",
      "authors": [
        "Maksim Eren",
        "Eric Michalak",
        "Brian Cook",
        "Johnny Seales"
      ],
      "abstract": "Culture shapes reasoning, values, prioritization, and strategic decision-making, yet large language models (LLMs) often exhibit cultural biases that misalign with target populations. As LLMs are increasingly used for strategic decision-making, policy support, and document engineering tasks such as summarization, categorization, and compliance-oriented auditing, improving cultural alignment is important for ensuring that downstream analyses and recommendations reflect target-population value profiles rather than default model priors. Previous work introduced a survey-grounded cultural alignment framework and showed that culture-specific prompting can reduce misalignment, but it primarily evaluated proprietary models and relied on manual prompt engineering. In this paper, we validate and extend that framework by reproducing its social sciences survey based projection and distance metrics on open-weight LLMs, testing whether the same cultural skew and benefits of culture conditioning persist outside closed LLM systems. Building on this foundation, we introduce use of prompt programming with DSPy for this problem-treating prompts as modular, optimizable programs-to systematically tune cultural conditioning by optimizing against cultural-distance objectives. In our experiments, we show that prompt optimization often improves upon cultural prompt engineering, suggesting prompt compilation with DSPy can provide a more stable and transferable route to culturally aligned LLM responses.",
      "published": "2026-03-17T17:34:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.16827v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16827v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Surg$Σ$: A Spectrum of Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence",
      "authors": [
        "Zhitao Zeng",
        "Mengya Xu",
        "Jian Jiang",
        "Pengfei Guo",
        "Yunqiu Xu",
        "Zhu Zhuo",
        "Chang Han Low",
        "Yufan He",
        "Dong Yang",
        "Chenxi Lin",
        "Yiming Gu",
        "Jiaxin Guo",
        "Yutong Ban",
        "Daguang Xu",
        "Qi Dou",
        "Yueming Jin"
      ],
      "abstract": "Surgical intelligence has the potential to improve the safety and consistency of surgical care, yet most existing surgical AI frameworks remain task-specific and struggle to generalize across procedures and institutions. Although multimodal foundation models, particularly multimodal large language models, have demonstrated strong cross-task capabilities across various medical domains, their advancement in surgery remains constrained by the lack of large-scale, systematically curated multimodal data. To address this challenge, we introduce Surg$Σ$, a spectrum of large-scale multimodal data and foundation models for surgical intelligence. At the core of this framework lies Surg$Σ$-DB, a large-scale multimodal data foundation designed to support diverse surgical tasks. Surg$Σ$-DB consolidates heterogeneous surgical data sources (including open-source datasets, curated in-house clinical collections and web-source data) into a unified schema, aiming to improve label consistency and data standardization across heterogeneous datasets. Surg$Σ$-DB spans 6 clinical specialties and diverse surgical types, providing rich image- and video-level annotations across 18 practical surgical tasks covering understanding, reasoning, planning, and generation, at an unprecedented scale (over 5.98M conversations). Beyond conventional multimodal conversations, Surg$Σ$-DB incorporates hierarchical reasoning annotations, providing richer semantic cues to support deeper contextual understanding in complex surgical scenarios. We further provide empirical evidence through recently developed surgical foundation models built upon Surg$Σ$-DB, illustrating the practical benefits of large-scale multimodal annotations, unified semantic design, and structured reasoning annotations for improving cross-task generalization and interpretability.",
      "published": "2026-03-17T17:27:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.16822v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16822v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights",
      "authors": [
        "Yi Chen",
        "Daiwei Chen",
        "Sukrut Madhav Chikodikar",
        "Caitlyn Heqi Yin",
        "Ramya Korlakai Vinayak"
      ],
      "abstract": "Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient.",
      "published": "2026-03-17T17:20:08Z",
      "abstract_url": "http://arxiv.org/abs/2603.16817v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16817v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Beyond Accuracy: Evaluating Forecasting Models by Multi-Echelon Inventory Cost",
      "authors": [
        "Swata Marik",
        "Swayamjit Saha",
        "Garga Chatterjee"
      ],
      "abstract": "This study develops a digitalized forecasting-inventory optimization pipeline integrating traditional forecasting models, machine learning regressors, and deep sequence models within a unified inventory simulation framework. Using the M5 Walmart dataset, we evaluate seven forecasting approaches and assess their operational impact under single- and two-echelon newsvendor systems. Results indicate that Temporal CNN and LSTM models significantly reduce inventory costs and improve fill rates compared to statistical baselines. Sensitivity and multi-echelon analyses demonstrate robustness and scalability, offering a data-driven decision-support tool for modern supply chains.",
      "published": "2026-03-17T17:19:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.16815v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16815v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "InCoder-32B: Code Foundation Model for Industrial Scenarios",
      "authors": [
        "Jian Yang",
        "Wei Zhang",
        "Jiajun Wu",
        "Junhang Cheng",
        "Shawn Guo",
        "Haowen Wang",
        "Weicheng Gu",
        "Yaxin Du",
        "Joseph Li",
        "Fanglin Xu",
        "Yizhi Li",
        "Lin Jing",
        "Yuanbo Wang",
        "Yuhan Gao",
        "Ruihao Gong",
        "Chuan Hao",
        "Ran Tao",
        "Aishan Liu",
        "Tuney Zheng",
        "Ganqu Cui",
        "Zhoujun Li",
        "Mingjie Tang",
        "Chenghua Lin",
        "Wayne Xin Zhao",
        "Xianglong Liu",
        "Ming Zhou",
        "Bryan Dai",
        "Weifeng Lv"
      ],
      "abstract": "Recent code large language models have achieved remarkable progress on general programming tasks. Nevertheless, their performance degrades significantly in industrial scenarios that require reasoning about hardware semantics, specialized language constructs, and strict resource constraints. To address these challenges, we introduce InCoder-32B (Industrial-Coder-32B), the first 32B-parameter code foundation model unifying code intelligence across chip design, GPU kernel optimization, embedded systems, compiler optimization, and 3D modeling. By adopting an efficient architecture, we train InCoder-32B from scratch with general code pre-training, curated industrial code annealing, mid-training that progressively extends context from 8K to 128K tokens with synthetic industrial reasoning data, and post-training with execution-grounded verification. We conduct extensive evaluation on 14 mainstream general code benchmarks and 9 industrial benchmarks spanning 4 specialized domains. Results show InCoder-32B achieves highly competitive performance on general tasks while establishing strong open-source baselines across industrial domains.",
      "published": "2026-03-17T17:01:35Z",
      "abstract_url": "http://arxiv.org/abs/2603.16790v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16790v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Finding Common Ground in a Sea of Alternatives",
      "authors": [
        "Jay Chooi",
        "Paul Gölz",
        "Ariel D. Procaccia",
        "Benjamin Schiffer",
        "Shirley Zhang"
      ],
      "abstract": "We study the problem of selecting a statement that finds common ground across diverse population preferences. Generative AI is uniquely suited for this task because it can access a practically infinite set of statements, but AI systems like the Habermas machine leave the choice of generated statement to a voting rule. What it means for this rule to find common ground, however, is not well-defined. In this work, we propose a formal model for finding common ground in the infinite alternative setting based on the proportional veto core from social choice. To provide guarantees relative to these infinitely many alternatives and a large population, we wish to satisfy a notion of proportional veto core using only query access to the unknown distribution of alternatives and voters. We design an efficient sampling-based algorithm that returns an alternative in the (approximate) proportional veto core with high probability and prove matching lower bounds, which show that no algorithm can do the same using fewer queries. On a synthetic dataset of preferences over text, we confirm the effectiveness of our sampling-based algorithm and compare other social choice methods as well as LLM-based methods in terms of how reliably they produce statements in the proportional veto core.",
      "published": "2026-03-17T16:28:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.16751v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16751v1",
      "categories": [
        "cs.GT",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SpecMoE: Spectral Mixture-of-Experts Foundation Model for Cross-Species EEG Decoding",
      "authors": [
        "D. Darankoum",
        "C. Habermacher",
        "J. Volle",
        "S. Grudinin"
      ],
      "abstract": "Decoding the orchestration of neural activity in electroencephalography (EEG) signals is a central challenge in bridging neuroscience with artificial intelligence. Foundation models have made strides in generalized EEG decoding, yet many existing frameworks primarily relying on separate temporal and spectral masking of raw signals during self-supervised pretraining. Such strategies often tend to bias learning toward high-frequency oscillations, as low-frequency rhythmic patterns can be easily inferred from the unmasked signal. We introduce a foundation model that utilizes a novel Gaussian-smoothed masking scheme applied to short-time Fourier transform (STFT) maps. By jointly applying time, frequency, and time-frequency Gaussian masks, we make the reconstruction task much more challenging, forcing the model to learn intricate neural patterns across both high- and low-frequency domains. To effectively recover signals under this aggressive masking strategy, we design SpecHi-Net, a U-shaped hierarchical architecture with multiple encoding and decoding stages. To accelerate large-scale pretraining, we partition the data into three subsets, each used to train an independent expert model. We then combine these models through SpecMoE, a mixture of experts framework guided by a learned spectral gating mechanism. SpecMoE achieves state-of-the-art performance across a diverse set of EEG decoding tasks, including sleep staging, emotion recognition, motor imagery classification, abnormal signal detection, and drug effect prediction. Importantly, the model demonstrates strong cross-species and cross-subject generalization, maintaining high accuracy on both human and murine EEG datasets.",
      "published": "2026-03-17T16:20:14Z",
      "abstract_url": "http://arxiv.org/abs/2603.16739v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16739v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure",
      "authors": [
        "Caglar Yildirim"
      ],
      "abstract": "Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions.",
      "published": "2026-03-17T16:16:35Z",
      "abstract_url": "http://arxiv.org/abs/2603.16734v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16734v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "IQuest-Coder-V1 Technical Report",
      "authors": [
        "Jian Yang",
        "Wei Zhang",
        "Shawn Guo",
        "Zhengmao Ye",
        "Lin Jing",
        "Shark Liu",
        "Yizhi Li",
        "Jiajun Wu",
        "Cening Liu",
        "X. Ma",
        "Yuyang Song",
        "Siwei Wu",
        "Yuwen Li",
        "L. Liao",
        "T. Zheng",
        "Ziling Huang",
        "Zelong Huang",
        "Che Liu",
        "Yan Xing",
        "Renyuan Li",
        "Qingsong Cai",
        "Hanxu Yan",
        "Siyue Wang",
        "Shikai Li",
        "Jason Klein Liu",
        "An Huang",
        "Yongsheng Kang",
        "Jinxing Zhang",
        "Chuan Hao",
        "Haowen Wang",
        "Weicheng Gu",
        "Ran Tao",
        "Mingjie Tang",
        "Peihao Wu",
        "Jianzhou Wang",
        "Xianglong Liu",
        "Weifeng Lv",
        "Bryan Dai"
      ],
      "abstract": "In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through different phases of the pipeline. Our models are developed through the evolutionary pipeline, starting with the initial pre-training consisting of code facts, repository, and completion data. Following that, we implement a specialized mid-training stage that integrates reasoning and agentic trajectories in 32k-context and repository-scale in 128k-context to forge deep logical foundations. The models are then finalized with post-training of specialized coding capabilities, which is bifurcated into two specialized paths: the thinking path (utilizing reasoning-driven RL) and the instruct path (optimized for general assistance). IQuest-Coder-V1 achieves state-of-the-art performance among competitive models across critical dimensions of code intelligence: agentic software engineering, competitive programming, and complex tool use. To address deployment constraints, the IQuest-Coder-V1-Loop variant introduces a recurrent mechanism designed to optimize the trade-off between model capacity and deployment footprint, offering an architecturally enhanced path for efficacy-efficiency trade-off. We believe the release of the IQuest-Coder-V1 series, including the complete white-box chain of checkpoints from pre-training bases to the final thinking and instruction models, will advance research in autonomous code intelligence and real-world agentic systems.",
      "published": "2026-03-17T16:15:31Z",
      "abstract_url": "http://arxiv.org/abs/2603.16733v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16733v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.SE"
      ]
    },
    {
      "title": "Federated Learning with Multi-Partner OneFlorida+ Consortium Data for Predicting Major Postoperative Complications",
      "authors": [
        "Yuanfang Ren",
        "Varun Sai Vemuri",
        "Zhenhong Hu",
        "Benjamin Shickel",
        "Ziyuan Guan",
        "Tyler J. Loftus",
        "Parisa Rashidi",
        "Tezcan Ozrazgat-Baslanti",
        "Azra Bihorac"
      ],
      "abstract": "Background: This study aims to develop and validate federated learning models for predicting major postoperative complications and mortality using a large multicenter dataset from the OneFlorida Data Trust. We hypothesize that federated learning models will offer robust generalizability while preserving data privacy and security. Methods: This retrospective, longitudinal, multicenter cohort study included 358,644 adult patients admitted to five healthcare institutions, who underwent 494,163 inpatient major surgical procedures from 2012-2023. We developed and internally and externally validated federated learning models to predict the postoperative risk of intensive care unit (ICU) admission, mechanical ventilation (MV) therapy, acute kidney injury (AKI), and in-hospital mortality. These models were compared with local models trained on data from a single center and central models trained on a pooled dataset from all centers. Performance was primarily evaluated using area under the receiver operating characteristics curve (AUROC) and the area under the precision-recall curve (AUPRC) values. Results: Our federated learning models demonstrated strong predictive performance, with AUROC scores consistently comparable or superior performance in terms of AUROC and AUPRC across all outcomes and sites. Our federated learning models also demonstrated strong generalizability, with comparable or superior performance in terms of both AUROC and AUPRC compared to the best local learning model at each site. Conclusions: By leveraging multicenter data, we developed robust, generalizable, and privacy-preserving predictive models for major postoperative complications and mortality. These findings support the feasibility of federated learning in clinical decision support systems.",
      "published": "2026-03-17T16:09:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.16723v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16723v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Cost Trade-offs in Matrix Inversion Updates for Streaming Outlier Detection",
      "authors": [
        "Florian Grivet",
        "Louise Travé-Massuyès"
      ],
      "abstract": "Outlier detection identifies data points that deviate significantly from expected patterns, revealing anomalies that may require special attention. Incorporating online learning further improves accuracy by continuously updating the model to reflect the most recent data. When employing the Christoffel function as an outlier score, online learning requires updating the inverse of a matrix following a rank-k update, given the initial inverse. Surprisingly, there is no consensus on the optimal method for this task. This technical note aims to compare three different updating methods: Direct Inversion (DI), Iterative Sherman-Morrison (ISM), and Woodbury Matrix Identity (WMI), to identify the most suitable approach for different scenarios. We first derive the theoretical computational costs of each method and then validate these findings through comprehensive Python simulations run on a CPU. These results allow us to propose a simple, quantitative, and easy-to-remember rule that can be stated qualitatively as follows: ISM is optimal for rank-1 updates, WMI excels for small updates relative to matrix size, and DI is preferable otherwise. This technical note produces a general result for any problem involving a matrix inversion update. In particular, it contributes to the ongoing development of efficient online outlier detection techniques.",
      "published": "2026-03-17T15:51:18Z",
      "abstract_url": "http://arxiv.org/abs/2603.16697v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16697v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making",
      "authors": [
        "Jun Liu",
        "Pu Zhao",
        "Zhenglun Kong",
        "Xuan Shen",
        "Peiyan Dong",
        "Fan Yang",
        "Lin Cui",
        "Hao Tang",
        "Geng Yuan",
        "Wei Niu",
        "Wenbin Zhang",
        "Xue Lin",
        "Gaowen Liu",
        "Yanzhi Wang",
        "Dong Huang"
      ],
      "abstract": "Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.",
      "published": "2026-03-17T15:38:50Z",
      "abstract_url": "http://arxiv.org/abs/2603.16673v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16673v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings?",
      "authors": [
        "Aishwarya Ramasethu",
        "Niyathi Allu",
        "Rohin Garg",
        "Harshwardhan Fartale",
        "Dun Li Chan"
      ],
      "abstract": "Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings.",
      "published": "2026-03-17T15:28:46Z",
      "abstract_url": "http://arxiv.org/abs/2603.16660v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16660v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models",
      "authors": [
        "Xiaojie Gu",
        "Sherry T. Tong",
        "Aosong Feng",
        "Sophia Simeng Han",
        "Jinghui Lu",
        "Yingjian Chen",
        "Yusuke Iwasawa",
        "Yutaka Matsuo",
        "Chanjun Park",
        "Rex Ying",
        "Irene Li"
      ],
      "abstract": "Reasoning-focused large language models (LLMs) have advanced in many NLP tasks, yet their evaluation remains challenging: final answers alone do not expose the intermediate reasoning steps, making it difficult to determine whether a model truly reasons correctly and where failures occur, while existing multi-hop QA benchmarks lack step-level annotations for diagnosing reasoning failures. To address this gap, we propose Omanic, an open-domain multi-hop QA resource that provides decomposed sub-questions and intermediate answers as structural annotations for analyzing reasoning processes. It contains 10,296 machine-generated training examples (OmanicSynth) and 967 expert-reviewed human-annotated evaluation examples (OmanicBench). Systematic evaluations show that state-of-the-art LLMs achieve only 73.11% multiple-choice accuracy on OmanicBench, confirming its high difficulty. Stepwise analysis reveals that CoT's performance hinges on factual completeness, with its gains diminishing under knowledge gaps and errors amplifying in later hops. Additionally, supervised fine-tuning on OmanicSynth brings substantial transfer gains (7.41 average points) across six reasoning and math benchmarks, validating the dataset's quality and further supporting the effectiveness of OmanicSynth as supervision for reasoning-capability transfer. We release the data at https://huggingface.co/datasets/li-lab/Omanic and the code at https://github.com/XiaojieGu/Omanic.",
      "published": "2026-03-17T15:23:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.16654v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16654v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When AI Navigates the Fog of War",
      "authors": [
        "Ming Li",
        "Xirui Li",
        "Tianyi Zhou"
      ],
      "abstract": "Can AI reason about a war before its trajectory becomes historically obvious? Analyzing this capability is difficult because retrospective geopolitical prediction is heavily confounded by training-data leakage. We address this challenge through a temporally grounded case study of the early stages of the 2026 Middle East conflict, which unfolded after the training cutoff of current frontier models. We construct 11 critical temporal nodes, 42 node-specific verifiable questions, and 5 general exploratory questions, requiring models to reason only from information that would have been publicly available at each moment. This design substantially mitigates training-data leakage concerns, creating a setting well-suited for studying how models analyze an unfolding crisis under the fog of war, and provides, to our knowledge, the first temporally grounded analysis of LLM reasoning in an ongoing geopolitical conflict. Our analysis reveals three main findings. First, current state-of-the-art large language models often display a striking degree of strategic realism, reasoning beyond surface rhetoric toward deeper structural incentives. Second, this capability is uneven across domains: models are more reliable in economically and logistically structured settings than in politically ambiguous multi-actor environments. Finally, model narratives evolve over time, shifting from early expectations of rapid containment toward more systemic accounts of regional entrenchment and attritional de-escalation. Since the conflict remains ongoing at the time of writing, this work can serve as an archival snapshot of model reasoning during an unfolding geopolitical crisis, enabling future studies without the hindsight bias of retrospective analysis.",
      "published": "2026-03-17T15:13:10Z",
      "abstract_url": "http://arxiv.org/abs/2603.16642v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16642v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CY"
      ]
    },
    {
      "title": "MLLM-based Textual Explanations for Face Comparison",
      "authors": [
        "Redwan Sony",
        "Anil K Jain",
        "Ross Arun"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) have recently been proposed as a means to generate natural-language explanations for face recognition decisions. While such explanations facilitate human interpretability, their reliability on unconstrained face images remains underexplored. In this work, we systematically analyze MLLM-generated explanations for the unconstrained face verification task on the challenging IJB-S dataset, with a particular focus on extreme pose variation and surveillance imagery. Our results show that even when MLLMs produce correct verification decisions, the accompanying explanations frequently rely on non-verifiable or hallucinated facial attributes that are not supported by visual evidence. We further study the effect of incorporating information from traditional face recognition systems, viz., scores and decisions, alongside the input images. Although such information improves categorical verification performance, it does not consistently lead to faithful explanations. To evaluate the explanations beyond decision accuracy, we introduce a likelihood-ratio-based framework that measures the evidential strength of textual explanations. Our findings highlight fundamental limitations of current MLLMs for explainable face recognition and underscore the need for a principled evaluation of reliable and trustworthy explanations in biometric applications. Code is available at https://github.com/redwankarimsony/LR-MLLMFR-Explainability.",
      "published": "2026-03-17T15:01:00Z",
      "abstract_url": "http://arxiv.org/abs/2603.16629v1",
      "pdf_url": "https://arxiv.org/pdf/2603.16629v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    }
  ]
};
