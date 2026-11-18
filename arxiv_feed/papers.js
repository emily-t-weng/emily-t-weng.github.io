const PAPERS_DATA = {
  "last_updated": "2026-11-18 06:59:17 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching",
      "authors": [
        "Giannis Daras",
        "Marios Papachristou",
        "Adam Klivans",
        "Antonio Torralba",
        "Constantinos Daskalakis"
      ],
      "abstract": "Collecting large-scale, diverse, and high-quality post-training datasets for autoregressive language models is challenging. Consequently, Supervised Fine-Tuning (SFT) is typically applied to datasets significantly smaller than those used for pre-training, which leads to overfitting and the catastrophic forgetting of the tails of the data distribution. To prevent this collapse, we introduce Ambient SFT, a principled framework for leveraging imperfect predictions to augment the fine-tuning set. In our method, a \"frozen\" version of the pre-trained model serves as a noisy teacher that the fine-tuned model must match at a coarse level. In particular, we require that the high-temperature versions of the conditional distributions produced by the post-training model should match the high-temperature versions of the pre-training ones. The temperature hyperparameter controls the level of trust in the pre-trained model, with the limit $T\\to \\infty$ reverting back to regular SFT and $T=1$ to knowledge distillation. We demonstrate that our approach outperforms standard SFT across domain adaptation, instruction-tuning, and reasoning, leading for up to $21\\%$ average relative improvement to standard SFT.",
      "published": "2026-11-17T18:59:37Z",
      "abstract_url": "http://arxiv.org/abs/2611.23229v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23229v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Do LLM Agents Still Need to Be Told How to Self-Improve?",
      "authors": [
        "Shobhita Sundaram",
        "Cansu Sancaktar",
        "Phillip Isola"
      ],
      "abstract": "LLM agents increasingly drive their own learning. Self-improvement methods delegate various components of model optimization to the models themselves. As models become more capable, this motivates a basic question: do these increasingly capable models still need us to specify how they should self-improve? In this paper we study an autonomous paradigm in which, given basic primitives, resources, and a general goal for self-improvement, agents construct their own self-learning algorithm for context-based optimization. Using the setting of gameplay as a testbed, a frozen LLM agent is given a compute budget and tools for collecting experience and updating text documents, and allowed to freely experiment without an externally imposed loop or access to real-world data. We find that sufficiently capable models do self-improve under these conditions, demonstrating strategies that align with classic cognitive frameworks for self-regulated learning and constructs from machine learning literature on self-generated curricula. Furthermore, we study the extent to which the learned text artifacts transfer to external benchmarks, observing improvement across different game horizons, mechanics, and 3D navigation.",
      "published": "2026-11-17T18:57:10Z",
      "abstract_url": "http://arxiv.org/abs/2611.23219v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23219v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Higher-Scoring Coding Models Did Not Get More Work Done",
      "authors": [
        "Yining Hua"
      ],
      "abstract": "New coding models advertise themselves on score improvements of coding benchmarks. Yet no study has tested if a model that scores higher on these benchmarks gets more real work done. On June 12, 2026, a U.S. export control directive banned access to Claude Fable 5 for 19 days, which forced users to switch to other models. During this time, most former Fable 5 users switched to Opus 4.8 (11 points lower on SWE-Bench Pro compared to Fable 5). We treat this suspension as a natural experiment, where the people and projects stayed the same and only the model changed, so we could isolate the effect of Fable 5 on user productivity. We tracked GitHub pull requests of 297 Fable 5 users before, during, and after the suspension. We used 8,908 developers who stayed on Opus 4.8 all the time as the control group. We also repeated the comparison across five model upgrades before and after the suspension in 2026 (1,208 developers, 15,442 pull requests). Fitting a difference-in-differences model with developer and date fixed effects, we found that having access to Fable 5 increased the 7-day merge rate of pull requests by only 1.4 percentage points [-1.4; +4.3], which was not statistically significant. Similarly, no significant increases of merge rates were found with the upgrades (weighted mean: -0.9 percentage points [-1.9; +0.2]); all five estimates were negative (95% intervals included 0, did not differ from each other). We also did not observe a significant increase in the number of pull requests merged per week (+2.4% [-4.9%; +10.2%]).",
      "published": "2026-11-17T18:56:42Z",
      "abstract_url": "http://arxiv.org/abs/2611.23216v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23216v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Fine-Tuning Gemma 4 Edge Models for MATLAB Signal-Processing Code with SFT and QLoRA on a Laptop iGPU",
      "authors": [
        "Mehmet Altan Toksöz"
      ],
      "abstract": "Small language models that run on a laptop are attractive for writing signal-processing code offline, but they make systematic errors on digital signal processing (DSP) conventions. We apply supervised fine-tuning (SFT) to the Gemma 4 edge models E2B and E4B with 66 checked examples covering 11 signal-processing MATLAB subtasks, using low-rank adaptation (LoRA) and quantized LoRA (QLoRA) on the integrated GPU (iGPU) of a laptop, and deploy the 8-bit integer (int8) models with the adapters through OpenVINO. On held-out variants scored by deterministic checkers, the pass rate rises from 18/66 to 37/66 (LoRA) and 37/66 (QLoRA) for E2B and from 32/66 to 47/66 for E4B (QLoRA). Reading all 159 failing programs shows that the adapters remove most errors of convention, such as the sign of a lag, units and array shapes, but few errors in building signals and noise. Applied at run time, the adapter costs 17-19% of the generation speed.",
      "published": "2026-11-17T18:47:37Z",
      "abstract_url": "http://arxiv.org/abs/2611.23205v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23205v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.SE",
        "eess.SP"
      ]
    },
    {
      "title": "WoRD: Diffusion-Based Posterior Inference for Online Goal Recognition",
      "authors": [
        "Almog Anschel",
        "Sarah Keren"
      ],
      "abstract": "Goal recognition is the task of inferring an agent's intended goal from its observed behavior. Most existing work focuses on high-level symbolic recognition, abstracting away the continuous motion through which agents reach their goal. Methods that do reason over continuous trajectories rely on an explicit dynamics model that is hard to acquire and rarely transfers across settings. Neither approach explicitly learns how agents move toward each candidate goal, limiting accuracy on ambiguous trajectory prefixes where such knowledge is most needed. We present WoRD (Waypoint Recognition with Diffusion), which uses a learned trajectory prior to infer the agent's goal from partial observations. At each step, WoRD runs one warm-started denoising chain per candidate goal, steers each chain toward the observed trajectory through a closed-form log-likelihood gradient, and reweights the resulting samples by Bayes' rule into a calibrated posterior over goals. We evaluate WoRD on simulated robot settings, pedestrian trajectories, and vehicle trajectories using four metrics targeting distinct failure modes. Across all three settings, the same inference procedure preserves calibrated multimodal beliefs through ambiguous prefixes where filter- and classifier-based observers collapse to a single mode or grow overconfident.",
      "published": "2026-11-17T18:35:53Z",
      "abstract_url": "http://arxiv.org/abs/2611.23192v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23192v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Retrieval-Augmented Correction for Generalist Medical Image Segmentation",
      "authors": [
        "Vincenzo Marcianò",
        "Xiaoming Zhang",
        "Michela Antonelli",
        "Sebastien Ourselin",
        "Maria A Zuluaga"
      ],
      "abstract": "Automated segmentation is routinely used as the first step in quantitative image-based clinical workflows, where predicted masks are subsequently reviewed and corrected by human operators. Automated correction of the residual structural errors is currently addressed one anatomy at a time, so the number of trained correctors scales with the number of segmentation targets. We present GRACE, a retrieval-augmented refiner that edits an input mask under context drawn from a shared knowledge base of verified prior cases. The refiner is trained once on a multi-organ corpus spanning compact organs, cavities, branching vessels, and tumour lesions. In the held-out dataset-transfer evaluation, three held-out datasets query a fixed knowledge base built from training data only. A separate leave-one-anatomy-out experiment tests parameter-free index extension: verified cases of an unseen solid organ enter the knowledge base, and the refiner stays frozen. A light shape-aware projection head on a frozen multimodal encoder maps embedding similarity to the shape agreement of the masks, so the knowledge base returns cases that the refiner can use as a shape prior. We evaluate GRACE on 18 public datasets spanning CT, MRI and ultrasound, using nine refinement configurations and seven upstream segmentors that span weak, mid, and strong operating regimes. We report each change with the DSC of its input mask. On the weak tier, the input DSC is 0.386 and GRACE adds up to +0.35 DSC. On a strong segmentor, the input DSC is 0.932 and the mean change is small. Transfer to an imaging modality that is absent from training and from the index is partial and depends on the anatomy, and remains an open research direction. Our results suggest retrieval-conditioned refinement is a practical alternative to anatomy-specific correction models. Code: https://github.com/robustml-eurecom/grace",
      "published": "2026-11-17T18:33:42Z",
      "abstract_url": "http://arxiv.org/abs/2611.23189v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23189v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Learning Stratified Policies for Inference-Time Objectives",
      "authors": [
        "Mridul Mahajan",
        "Xuezhou Zhang",
        "Aldo Pacchiano"
      ],
      "abstract": "Inference-time scaling improves the performance of large language models by sampling multiple responses and evaluating their collective utility. However, many existing approaches obtain these samples by repeatedly sampling from a single policy. In this work, we study if the sample budget at inference time can instead be allocated across a population of policies. We introduce stratified policies, where each of $k$ learned policies contributes one response to a $k$-sample inference-time objective. We realize this policy population using separate LoRA adapters on a shared language model backbone and optimize it for pass@$k$. We theoretically motivate the approach by analyzing the linear setting; a population of LoRAs can achieve zero best-of-$k$ approximation error in settings where every single LoRA has strictly positive prediction error. Empirically, we demonstrate that stratified policies consistently improve pass@$k$ over a single-policy baseline across multiple reasoning tasks. Overall, our results highlight how distributing samples across complementary policies can be a more effective use of sample budget than repeated i.i.d. sampling from a single policy at inference time.",
      "published": "2026-11-17T18:06:12Z",
      "abstract_url": "http://arxiv.org/abs/2611.23154v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23154v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "RSI-Prime: Training Models to Train Models",
      "authors": [
        "Dewu Zheng",
        "Yang Ye",
        "Hongyu Zhang",
        "Jun Yu",
        "Yanlin Wang"
      ],
      "abstract": "Enabling large language models (LLMs) to autonomously conduct training and improve target models is an important foundation for recursive self-improvement (RSI). As LLMs expand their role from being trained to conducting training, their model-training capability warrants dedicated study and systematic development. To this end, we construct RSI-Prime, a dataset for improving model-training capability, comprising 350 complete experimental trajectories across seven task categories. To help LLMs learn to make high-quality experimental decisions from training practice, we perform fine-grained decision identification and quality annotation, then apply decision-level loss masking to exclude supervision on low-quality behavior. Experiments with GLM-5.3 on PostTrainBench, a benchmark for autonomous post-training, show that training on RSI-Prime yields a 31.95% relative improvement in overall performance over the original GLM-5.3, highlighting a path toward future RSI systems in which models autonomously sustain their own advancement.",
      "published": "2026-11-17T17:55:30Z",
      "abstract_url": "http://arxiv.org/abs/2611.23138v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23138v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Natural Language Programming: Rethinking Language, Memory, and Cognition in the Era of Generative AI",
      "authors": [
        "Giovanni Adorni",
        "Daniele Grosso"
      ],
      "abstract": "Generative Artificial Intelligence is transforming the role of natural language in computational systems. In classical Natural Language Processing (NLP), language was treated as an object of computation: something to analyse, classify, translate, or generate. Large Language Models (LLMs) challenge this perspective, enabling natural language to function increasingly as an executable cognitive medium through which humans structure reasoning, orchestrate problem solving, and regulate hybrid human/AI workflows. This paper introduces Natural Language Programming (NLPg) as a theoretical and epistemological extension of traditional NLP. Rather than focusing on language understanding alone, NLPg conceptualises prompts as dynamic cognitive artefacts that externalise goals, constraints, assumptions, validation criteria, and intermediate reasoning states. Prompting is thus not merely a technical interaction technique, but a form of dialogic cognitive programming operating within distributed human/AI memory architectures. Building on theories of distributed cognition, extended mind, metacognition, and dialogic learning, the paper proposes the notion of Dialogic Working Memory, describing how prompts and conversational states function as temporary external memory structures supporting reasoning continuity and reflective regulation. We further map existing memory architectures (BTMA, SKGE, RMR) onto this framework, illustrating how episodic, semantic, and metacognitive layers support long-horizon conversational coherence. Finally, implications for computational linguistics and AI/mediated knowledge systems are discussed, suggesting a post symbolic and post interface paradigm in which natural language operates simultaneously as communication medium, executable specification, cognitive scaffold, and distributed memory architecture.",
      "published": "2026-11-17T17:27:01Z",
      "abstract_url": "http://arxiv.org/abs/2611.23106v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23106v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "What's in a Tree ? Hierarchical $n$-gram models as controlled approximators of language",
      "authors": [
        "Aritra Das",
        "Francesco Cagnetta",
        "Darshil Doshi",
        "Victor V. Albert",
        "Maissam Barkeshli",
        "Andrey Gromov"
      ],
      "abstract": "Hierarchical models have been proposed as synthetic settings for studying language. However, their relevance to realistic language corpora remains underexplored. In this work, we introduce tree-grams, autoregressive generative $n$-gram models that build the conditional probability tensor $p(x_n \\mid x_{1:n-1})$ through a hierarchical composition of order-2 and order-3 tensors. $n$ controls the maximal context window, whereas a rank hyperparameter $d$ controls model capacity, yielding an interpretable and systematically scalable family of approximators of language. We show how tree-grams outperform existing $n$-gram baselines, while remaining competitive with deep transformer-based language models. We demonstrate the utility of such controlled approximations by using tree-grams as generative models. In particular, we show that hyperparameter transfer under $μ$P can be made more robust when narrower models are tuned on simpler data, generated by tree-grams with moderate $n$ and $d$. In addition, we show that entropy and effective range of correlations control performance gap after fine-tuning. This is because tree-grams allow us to explicitly control horizon and capacity. When allowed to route across candidate topologies, we observe emergent sparsity in tree-grams. Such concentrated routing permits pruning upto 95\\% of topologies without significant change in performance.",
      "published": "2026-11-17T17:14:01Z",
      "abstract_url": "http://arxiv.org/abs/2611.23093v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23093v1",
      "categories": [
        "cs.CL",
        "cond-mat.dis-nn",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "BugGen 2.0: Rethinking the Roles of LLMs for Scalable RTL Bug Synthesis",
      "authors": [
        "Minh Luu",
        "Michael Quinn",
        "Aakash Tyagi",
        "David Houngninou",
        "Jiang Hu"
      ],
      "abstract": "AI-assisted RTL localization and debugging need ground-truth bug datasets with known locations and observable failures, while testbench grading uses injected bugs to measure detection capability. Creating such datasets at scale remains a major EDA challenge. BugGen 1.0 shows that Large Language Models (LLMs) can generate context-aware RTL mutations. Still, it relies on the model for both choosing where to mutate and generating the edit, which increases inference costs and can lead to invalid choices. BugGen 2.0 changes this division of labor. Deterministic algorithms handle structural analysis, target selection, and validation, while the LLM focuses on localized semantic rewriting. Across four OpenTitan IP blocks and FabScalar with three LLMs, BugGen 2.0 reaches every requested quota of compilable bug scenarios that cause at least two test failures, with 14-16 of 16 configured mutation types represented. In completed same-model comparisons, it reduces token use by 6.8x-15.4x relative to BugGen 1.0 and 1.5x-7.9x relative to Firefly. BugGen 2.0 also expands the dataset without simulating every new bug. After finding accepted scenarios with observable failures, it generates alternative mutations at the same locations, simulates only a subset, and estimates acceptance for the rest. This skips 63.6% of AES and 33.9% of FabScalar variant simulations. Full simulation finds that 77.7% of AES variants and 95.2% of FabScalar variants are accepted, with a 1.09% estimation error on the directly comparable FabScalar set. These results show that focusing LLMs on semantic rewriting lowers inference and simulation cost while improving consistency across models and designs.",
      "published": "2026-11-17T17:11:54Z",
      "abstract_url": "http://arxiv.org/abs/2611.23091v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23091v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Learning to Undo: Transfer Reinforcement Learning under State Space Transformations",
      "authors": [
        "Mridul Mahajan",
        "Aldo Pacchiano",
        "Xuezhou Zhang"
      ],
      "abstract": "Transfer learning in reinforcement learning (RL) has shown strong empirical success. In this work, we take a more principled perspective by studying when and how transferring knowledge between MDPs (a source and a target) can be provably beneficial. Specifically, we consider the case where there exists an undo map such that applying this map to the target's state space recovers the source exactly. We propose an algorithm that learns this map by matching state feature statistics gathered from both MDPs, and then uses it to transfer the source policy to the target. We theoretically justify the algorithm by analyzing the setting when the undo map is linear and the source is linearly-$Q^\\star$ realizable, where our approach has strictly better sample complexity than tabula rasa RL in the target MDP. Empirically, we demonstrate that these benefits extend beyond this regime: on challenging continuous control tasks and Atari games, our method achieves significantly better sample efficiency. Overall, our results highlight how shared structure between tasks can be leveraged for efficient transfer of policies across environments.",
      "published": "2026-11-17T17:06:51Z",
      "abstract_url": "http://arxiv.org/abs/2611.23081v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23081v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Runtime Verification of Multiple Natural Language Criteria for Agent Governance",
      "authors": [
        "Silviu Pitis",
        "Parand A. Alamdari",
        "Jessica Tang",
        "Toryn Q. Klassen",
        "Sheila A. McIlraith"
      ],
      "abstract": "AI agents are governed by hundreds of natural language criteria, yet existing verifiers cannot efficiently evaluate an agent's output against a large set of such criteria and report which are satisfied, violated, or inapplicable. We introduce the VFM, a tree-attention verifier that encodes the shared context and target text, and scores N natural language criteria in parallel, returning independent per-criterion ternary verdicts. The VFM is pretrained on CriteriaBank, a corpus of 394K (context, target, criterion, verdict) tuples spanning over 321K distinct criterion strings. A finetuned VFM-4B outperforms GPT-5.4 on a synthetic governance benchmark. On the DynaBench multi-criterion benchmark, it outperforms DynaGuard-4B on trace-level failure detection, and its per-criterion verdicts localize the violated rule and support focused escalation to a larger chain-of-thought judge. We show that CriteriaBank pretraining improves cross-domain transfer, particularly in the low-data regime, while matching a strong direct-finetuning control at full data. On a single H100, VFM-4B evaluates 1,000 criteria against a 4K-token context in under 2s at <11 GB VRAM, making it suitable for runtime verification.",
      "published": "2026-11-17T17:01:00Z",
      "abstract_url": "http://arxiv.org/abs/2611.23074v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23074v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "FrameRouter: Frame Budget Routing for Long-Video Understanding on Video-MME-v2 and Beyond",
      "authors": [
        "Bingjun Luo",
        "Jialin Guo",
        "Siqi Li"
      ],
      "abstract": "Long-video question answering benchmarks like Video-MME-v2 typically evaluate systems under a uniform frame budget for every question, despite stark variation in the evidence each query actually demands. We analyze this mismatch on Video-MME-v2 and find that different question subsets prefer different frozen frame-budget policies, while uniformly increasing the frame count does not reliably improve all metrics. These findings motivate fixed-mean frame-budget routing, a controlled inference setting in which each question may receive a different frame budget while the average budget matches a uniform reference. We introduce FrameRouter, a training-free test-time framework that combines an evidence-demand router, a budget-constrained allocation rule, and a frozen bank of frame-sampling policies. By controlling how much visual evidence each question receives while leaving the underlying samplers unchanged, FrameRouter is orthogonal to query-aware frame selection. Across Video-MME-v2 and several long-video understanding benchmarks, FrameRouter improves long-video question answering by using the same average visual budget more selectively across questions. Code is available at https://github.com/bingjunluo/FrameRouter.",
      "published": "2026-11-17T16:49:21Z",
      "abstract_url": "http://arxiv.org/abs/2611.23055v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23055v1",
      "categories": [
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "ForecastGym: A Live, Controlled, and Replayable Benchmark for Forecasting Agents",
      "authors": [
        "Yicheng Tao",
        "Yuante Li",
        "Zipei Fan"
      ],
      "abstract": "Agents built on large language models (LLMs) increasingly forecast real-world events by searching the Web and reasoning over what they find. How well such an agent forecasts depends on both the retrieved evidence and the model, making control over that evidence essential for evaluation. Live evaluations can mix retrieval with reasoning when each system searches on its own, while retrospective search can leak later information through date filters or today's ranking of historical pages. Comparing models, search sources, and memory strategies requires both fixed evidence and a record of how it evolved before resolution. We introduce ForecastGym, which records what four commercial web search APIs return for open prediction-market questions every two days and replays this frozen evidence offline once outcomes are known, comparing models, memory, and captured search sources under a fixed evidence-acquisition policy. Its first release covers 150 questions, 614 evidence states, one per question and collection time, and 131k search results. Rerunning the same queries after settlement returns results that all but state the outcome for up to 92% of the questions, despite the tested date filters. On the frozen evidence, all tested LLMs trail the market in forecasting accuracy. Our diagnostic analysis suggests that missed public information contributes more to the gap in high-gap states than misreading does. Switching the search API changes a model's Brier score by up to 0.043, more than the spread across models, while persistent notes score similarly to rereading all evidence at half the input tokens. The evidence a forecaster sees can matter as much as the model that reads it.",
      "published": "2026-11-17T16:44:38Z",
      "abstract_url": "http://arxiv.org/abs/2611.23045v1",
      "pdf_url": "https://arxiv.org/pdf/2611.23045v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Residual-Guided Lightweight Reinforcement Learning for Risk-Aware Trajectory Selection with a Frozen World Model",
      "authors": [
        "Jiaqi Wu",
        "Yongqi Dong",
        "Yijie Zhong",
        "Maple Shao",
        "Yide Tao",
        "Linsen Cai",
        "Weize Cai",
        "Hai L. Vu",
        "Alvaro Garcia Hernandez"
      ],
      "abstract": "In automated driving, world models provide future scene predictions for evaluating candidate trajectories, but fixed selection rules may not capture scene-dependent trade-offs. We propose a residual-guided reinforcement learning (RL) framework for lightweight trajectory selection with a frozen world model and a fixed candidate bank. The framework uses proximal policy optimization (PPO) to fine-tune a joint policy over candidate selection and bounded trajectory residuals. A context-conditioned ranking network learns candidate preferences from feedback obtained by evaluating local perturbations of the selected trajectories. At inference, the selector combines learned preference and predicted risk ranks using fixed weights and returns the selected candidate trajectory without modification. Fine-tuning updates 16,106 parameters across the actor and critic, while the deployed selector retains only the 11,725-parameter ranking network, with the residual branch and critic used exclusively during training. In open-loop evaluation on the nuScenes validation set, the method reduces average Euclidean trajectory error (L2) by 3.99% relative to the RiskWorld model baseline without RL-based selection. Under matched budgets for feedback accesses and actor updates, residual exploration reduces L2 by 0.83% relative to the categorical PPO variant trained without residuals. The method also maintains lower L2 than RiskWorld when risk features are removed or shuffled.",
      "published": "2026-11-17T16:12:55Z",
      "abstract_url": "http://arxiv.org/abs/2611.22998v1",
      "pdf_url": "https://arxiv.org/pdf/2611.22998v1",
      "categories": [
        "cs.AI",
        "cs.ET",
        "cs.LG",
        "cs.RO",
        "eess.SY"
      ]
    },
    {
      "title": "Structured AI ideation yields physical strategies for epithelial barrier repair",
      "authors": [
        "Geonho Nah",
        "Johyun Park",
        "Kyung-Ah Cho",
        "Suyoung Hwang",
        "Yoon Hee Lim",
        "Eun-kyung Min",
        "Seunghyun Moon",
        "Donghyun Koh",
        "Seungwon Kim",
        "Jiho Park",
        "Hojin Yoo",
        "Myungsun Nam",
        "Haneul Choi",
        "Kyungsu Kim",
        "In-Sun Hong",
        "YunJae Jung",
        "Minhyeong Lee"
      ],
      "abstract": "AI systems increasingly support literature analysis, candidate evaluation and hypothesis refinement, but enabling researchers to explore research directions beyond established disciplinary approaches remains challenging. We developed Abelio, an AI system designed to generate novel research directions by expanding investigators' avenues of inquiry. Abelio reformulates research objectives, uses these alternative formulations to identify relevant elements outside conventional disciplinary boundaries, and refines the resulting hypotheses. Each stage integrates divergent exploration with convergent selection through a dedicated subsystem. In quantitative evaluations, proposals generated by the Ideation Core (Abelio's hypothesis-generation core) outperformed those of four frontier large language models in scientific soundness and hypothesis diversity, and three of four in distance from existing literature. Subsystem ablation showed that each subsystem contributes to the property it targets, with the contribution of Element Selection measurable before Refinement. To investigate intestinal epithelial barrier restoration, Abelio proposed physical interventions involving membrane-tethered poly(ethylene glycol) (PEG) and cholesterol-anchored complementary DNA. Both proposals were archived before experimentation and subsequently tested independently in damaged Caco-2 monolayers. PEG preserved barrier function in a membrane-surface-dependent manner, whereas complementary DNA enabled sequence-specific restoration that was reversible on demand. These findings demonstrate how structured research ideation can identify experimentally testable approaches beyond established disciplinary conventions.",
      "published": "2026-11-17T16:05:22Z",
      "abstract_url": "http://arxiv.org/abs/2611.22988v1",
      "pdf_url": "https://arxiv.org/pdf/2611.22988v1",
      "categories": [
        "cs.AI",
        "q-bio.QM"
      ]
    },
    {
      "title": "Large Language Models Have Not Yet Transformed Science",
      "authors": [
        "Ron Bekkerman",
        "Nichole Li",
        "Bahareh Kargar",
        "Mark Somers"
      ],
      "abstract": "The emergence of large language models (LLMs) in late 2022 has raised questions about the future of science, but assessing their impact requires understanding how scientific practices were already evolving. We analyze 2.5 million full-text papers across five disciplines from 2015 to 2025, tracing trends over a broad spectrum of lexical, structural, epistemological, and social dimensions. Aside from frequency shifts in some words, most pre-2023 trends continued with limited detectable breaks. Among the notable exceptions, growth in computer science coauthor teams accelerated significantly in Asia, while apparent declines in biomedical topic diversity largely reflected standardization in article formats. Our study establishes an empirical foundation for monitoring the health of global science as LLM adoption deepens.",
      "published": "2026-11-17T15:57:22Z",
      "abstract_url": "http://arxiv.org/abs/2611.22975v1",
      "pdf_url": "https://arxiv.org/pdf/2611.22975v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY",
        "cs.DL"
      ]
    },
    {
      "title": "High-Dimensional Black-Box Optimization via Guidance Sampling of Diffusion Models",
      "authors": [
        "Xiaoyan Hu",
        "Rohit Sonker",
        "Jalal Butt",
        "Jiayu Chen",
        "Egemen Kolemen",
        "Jeff Schneider"
      ],
      "abstract": "Many real-world optimization problems involve optimizing black-box objectives for which only a limited number of evaluations are available. Bayesian optimization (BO) is a powerful framework for such problems, but its effectiveness often deteriorates in high-dimensional spaces. Recent work leverages diffusion models for black-box optimization, often through pre-training a reward-conditioned score network to steer generation towards high-reward samples. Alternatively, gradient-based guidance methods assume access to a differentiable reward function from which guidance gradients are queried. However, these approaches either require large amounts of labeled data for offline training or access to exact reward gradients, limiting their applicability in black-box settings. We propose BReaD (Black-box Reward-guided Sampling of Diffusion Model), a guidance sampling framework for optimizing black-box rewards without assuming access to a labeled dataset or reward gradients. The proposed method uses a Gaussian process (GP) as a differentiable surrogate for the unknown objective and actively queries labels for the generated samples. We provide a theoretical analysis of the total-variation distance between the generated distribution and the target distribution. Using Upper Confidence Bound (UCB) as guidance signals, our method balances exploration and exploitation during generation and enables the discovery of high-reward regions in the data space. Numerical experiments on scientific discovery and image generation tasks demonstrate that our framework can effectively optimize black-box rewards in low-data scenarios.",
      "published": "2026-11-17T15:51:29Z",
      "abstract_url": "http://arxiv.org/abs/2611.22965v1",
      "pdf_url": "https://arxiv.org/pdf/2611.22965v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Hierarchical Diffusion Transformer for End-to-End Image Generation",
      "authors": [
        "Xiaoyu Wu",
        "Yifei Wang",
        "Weihang Guo",
        "Chen Wei",
        "Lydia E Kavraki",
        "Niloofar Mireshghallah",
        "Steven Wu"
      ],
      "abstract": "Pixel-space diffusion learns visual representations and image generation jointly in a single end-to-end training stage, without a separately pretrained visual encoder. However, plain pixel-space Transformers still trail leading two-stage latent diffusion models in generation quality and training efficiency. We hypothesize that part of this gap arises from retaining one token per patch throughout the backbone, which does not explicitly encourage a visual hierarchy that summarizes related content at progressively coarser levels. To address this, we introduce Hierarchical Diffusion Transformers (HiT), which progressively group related tokens and represent each group with a single token. The main Transformer bottleneck processes these compact group tokens, and ungrouping restores features to their original image positions. A compression budget regulates the average compression ratio across the batch, allowing group memberships and counts to vary across images and diffusion timesteps during both training and generation. The model exhibits emergent semantic structure, learning to group coherent objects and object parts without segmentation labels. In text-to-image generation, HiT further learns to group caption words with image patches depicting the same concepts, without region annotations. HiT achieves 1.48 FID on ImageNet $256\\times256$ and reaches comparable generation quality to JiT using approximately 40\\% of its cumulative training FLOPs. For text-to-image generation, HiT exceeds UniLiP's average score across five benchmarks with roughly $6\\times$ fewer denoiser parameters and $3\\times$ fewer forward FLOPs. These results suggest that learning a visual hierarchy within the generator breaks down the barrier to stronger image generation without external visual pretraining.",
      "published": "2026-11-17T15:47:56Z",
      "abstract_url": "http://arxiv.org/abs/2611.22962v1",
      "pdf_url": "https://arxiv.org/pdf/2611.22962v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
