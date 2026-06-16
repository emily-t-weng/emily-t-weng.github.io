const PAPERS_DATA = {
  "last_updated": "2026-06-16 05:18:26 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "The Importance of Phase in Neural Representations: An Internal Oppenheim-Lim Test of Image Classifiers",
      "authors": [
        "Alper Yıldırım"
      ],
      "abstract": "Oppenheim and Lim (1981) showed that natural images stay recognizable when reconstructed from their Fourier phase alone, while the magnitude carries little of their identity. We ask whether trained image classifiers reproduce this asymmetry inside their hidden layers, and we test it causally: given two images, we transplant the phase of one onto the magnitude of the other at a chosen layer and record which image the prediction follows. In PRISM2D, GFNet, and ViT-B/16 the prediction follows the phase or sign donor, and deleting all image-specific magnitude barely moves accuracy, so identity rides on phase while image-specific magnitude is largely dispensable to the readout. ResNet-50 at first seems to break the pattern, because transplanting sign after its ReLUs does nothing; a fair intervention before the ReLU reveals a strong latent sign code in the late blocks, and a DC-only control shows the readout consumes a channel-wise spatial average. Controls rule out the trivial case in which magnitude simply stops depending on the image. The architectures therefore share a phase/sign identity code but expose it in different bases, set by rectification and readout geometry, which gives a mechanistic account of the texture--shape gap between CNNs and attention models.",
      "published": "2026-06-15T17:54:52Z",
      "abstract_url": "http://arxiv.org/abs/2606.17037v1",
      "pdf_url": "https://arxiv.org/pdf/2606.17037v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "HAMON: Passive Optical Sequence Mixing for Long-Horizon Forecasting",
      "authors": [
        "Alper Yıldırım"
      ],
      "abstract": "Simple linear and frequency-domain models remain surprisingly competitive in long-horizon time-series forecasting, and recent mechanistic evidence suggests that standard forecasting benchmarks may not require the dense superposed representations that make transformers powerful in other domains. This raises a substrate-level question: if the core forecasting operator is often low-complexity and approximately linear, does it need to be implemented as learned digital temporal mixing? We introduce HAMON, a passive diffractive optical forecasting core in which historical values are encoded onto an optical aperture, future positions are left dark, and cascaded trainable phase masks with free-space diffraction shape the forecast directly in the output field. At inference, prediction is performed by a single passive optical propagation pass with no trainable digital sequence-mixing layer. Across standard benchmarks, HAMON outperforms the strongest digital baselines considered on ETTm2 at all horizons and on ETTh2 at all but the longest horizon, improving MSE by up to 14\\% and doing so consistently across horizons rather than at isolated points. It is competitive on Weather and trails the strongest baselines on the remaining ETT settings and on the high-channel-count Traffic and Electricity datasets. Phase encoding, intensity-compatible readout, and phase-scrambling ablations, together with a TorchOptics cross-simulator check, indicate that the forecasts arise from the data-bearing optical field rather than from a digital forecasting head. Because the passive core uses standard Fourier optics, HAMON defines a concrete target for optical hardware and for passive physical sequence mixing.",
      "published": "2026-06-15T17:51:50Z",
      "abstract_url": "http://arxiv.org/abs/2606.17028v1",
      "pdf_url": "https://arxiv.org/pdf/2606.17028v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.AR"
      ]
    },
    {
      "title": "TokenPilot: Cache-Efficient Context Management for LLM Agents",
      "authors": [
        "Buqiang Xu",
        "Zirui Xue",
        "Dianmou Chen",
        "Chenyang Fu",
        "Chiyu Wu",
        "Caiying Huang",
        "Chen Jiang",
        "Jizhan Fang",
        "Xinle Deng",
        "Yijun Chen",
        "Yunzhi Yao",
        "Xuehai Wang",
        "Jin Shang",
        "Gong Yu",
        "Ningyu Zhang"
      ],
      "abstract": "As LLM agents are deployed in long-horizon sessions, context accumulation drives up inference costs. Existing approaches utilize text pruning or dynamic memory eviction to minimize token footprints; however, their unconstrained sequence mutations alter layouts, introducing prefix mismatches and cache invalidation. This reveals a critical trade-off between text sparsity and prompt cache continuity. To address this, we present TokenPilot, a dual-granularity context management framework. Globally, Ingestion-Aware Compaction acts as a framework harness to stabilize prompt prefixes and eliminate open-world environmental noise at the ingestion gate. Locally, Lifecycle-Aware Eviction monitors the ongoing residual utility of context segments, enforcing a conservative batch-turn schedule to offload content segments only when task relevance expires. Experiments on PinchBench and Claw-Eval under both isolated and continuous modes demonstrate that TokenPilot reduces costs by 61% and 56% in isolated mode, and 61% and 87% in continuous mode, while maintaining competitive performance compared to prior systems. TokenPilot has been integrated into LightMem2 at https://github.com/zjunlp/LightMem2.",
      "published": "2026-06-15T17:46:50Z",
      "abstract_url": "http://arxiv.org/abs/2606.17016v1",
      "pdf_url": "https://arxiv.org/pdf/2606.17016v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "TuneJury: An Open Metric for Improving Music Generation Preference Alignment",
      "authors": [
        "Yonghyun Kim",
        "Junwon Lee",
        "Haiwen Xia",
        "Yinghao Ma",
        "Junghyun Koo",
        "Koichi Saito",
        "Yuki Mitsufuji",
        "Chris Donahue"
      ],
      "abstract": "We introduce TuneJury, an open, instance-level pairwise reward model for text-to-music that predicts a music preference score from a text prompt and an audio clip. The released checkpoint is trained on publicly available human-preference labels covering arena-style (A vs. B) votes, metric-alignment preference pairs, crowdsourced pairwise comparisons, and expert aesthetic ratings. The predicted score margin between two clips is well calibrated on our held-out test split, supporting data filtering via a simple score threshold. TuneJury generalizes to both held-out test pairs and out-of-distribution benchmarks, remaining competitive with prior baselines on the latter. For generators released after training, we introduce anchor calibration, a post-hoc, per-system Bradley-Terry calibration that recovers agreement at substantially better data efficiency than from-scratch retraining. The same frozen reward drives consistent reward-axis gains across three downstream applications: inference-time best-of-N selection, DITTO-style latent optimization, and expert-iteration post-training. TuneJury is available at https://github.com/yonghyunk1m/TuneJury.",
      "published": "2026-06-15T17:39:30Z",
      "abstract_url": "http://arxiv.org/abs/2606.17006v1",
      "pdf_url": "https://arxiv.org/pdf/2606.17006v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.LG",
        "cs.MM",
        "eess.AS"
      ]
    },
    {
      "title": "ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation",
      "authors": [
        "Tran Dinh Tien",
        "Zhiqiang Shen"
      ],
      "abstract": "Segment Anything Model 3 (SAM 3) provides a strong frozen backbone for concept-prompted segmentation, but applying it directly to open-vocabulary semantic segmentation (OVSS) is inefficient: full-resolution decoding is typically run over the entire dataset vocabulary, whereas each image contains only a small active subset of classes. We introduce ActiveSAM, a training-free, zero-shot inference framework that turns SAM 3 into an active-vocabulary segmenter. ActiveSAM first canonicalizes and expands class prompts, then estimates an image-conditioned active set from a low-resolution presence preview. Only the retained classes are decoded at full resolution, using bucketed prompt multiplexing with the frozen SAM 3 decoder. The preview stage uses only class-presence evidence and skips unnecessary segmentation-head computation, while the final stage applies margin-aware background calibration to suppress low-confidence pixels. ActiveSAM requires no target-dataset training, no weight updates, and no oracle class-presence labels. Across eight OVSS benchmarks, ActiveSAM improves the speed-accuracy tradeoff of training-free open-vocabulary semantic segmentation, outperforming the current state-of-the-art SegEarth-OV3 by approximately +1.4 mIoU on average while running up to 5.5x faster on large-vocabulary datasets. ActiveSAM also demonstrates the strongest robustness under image corruption that simulates real-world distribution shift, making it well-suited for deployment in noisy-input domains such as autonomous driving and embodied AI. Code is available at https://github.com/VILA-Lab/ActiveSAM.",
      "published": "2026-06-15T17:31:30Z",
      "abstract_url": "http://arxiv.org/abs/2606.16996v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16996v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When in Doubt, Plan It Out: Committed Small Language Model Deliberation for Reactive Reinforcement Learning",
      "authors": [
        "Nathan Gavenski",
        "Juarez Monteiro",
        "Francisco Galuppo",
        "Adriano Veloso",
        "Odinaldo Rodrigues"
      ],
      "abstract": "Reinforcement Learning (RL) policies often degrade in unfamiliar environments because they lack explicit deliberation. We propose Plan, Align, Commit, Think (PACT), a hybrid architecture that combines a fast, reactive RL policy with a slow, deliberative Small Language Model (SLM) planner. PACT invokes the SLM asynchronously to generate and validate candidate action plans. Once a plan is verified through simulation as safe, feasible, and complete, it is executed directly, bypassing the RL policy without retraining or modifying it. Evaluated on three FrozenLake configurations of increasing difficulty, PACT outperforms all baselines while relying on a 2B-parameter SLM backbone, suggesting that deliberative planning and reactive execution are more powerful in concert than either is alone in these settings.",
      "published": "2026-06-15T17:31:22Z",
      "abstract_url": "http://arxiv.org/abs/2606.16995v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16995v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Consensus-based Agentic Large Language Model Framework for Harmonized Tariff Schedule Code Classification",
      "authors": [
        "Truong Thanh Hung Nguyen",
        "Khanh Van Quynh Nguyen",
        "Hoang-Loc Cao",
        "Tri Duong",
        "Phuc Ho",
        "Van Pham",
        "Loc Nguyen",
        "Hung Cao"
      ],
      "abstract": "Accurate Harmonized Tariff Schedule (HTS) code classification is essential for customs clearance, duty assessment, trade statistics, and regulatory compliance in maritime logistics. However, exact HTS classification remains challenging because product descriptions are often short, incomplete, or ambiguous, while correct classification depends on hierarchical tariff structures, legal notes, and jurisdiction-specific rules. This paper proposes an agentic large language model (LLM) framework for Canadian 10-digit HTS code classification in smart-port and maritime logistics environments. The framework integrates multi-agent information retrieval, semantic retrieval over official tariff documents, evidence-grounded reasoning, consensus-based validation, element-wise voting across hierarchical code components, confidence estimation, and human-in-the-loop escalation. We evaluate the framework on a private dataset of 3,300 domain-expert-labeled product records collected from logistics and delivery contexts. Experimental results show that exact 10-digit classification remains difficult even for advanced LLMs, with performance decreasing from coarse chapter-level prediction to fine-grained tariff and statistical suffix assignment. These findings demonstrate the need for evidence-grounded, uncertainty-aware, and human-centered classification workflows rather than fully autonomous single-step prediction. The proposed framework supports more interpretable, accountable, and compliance-oriented HTS classification for maritime logistics and smart-port operations. Our code is available at https://github.com/Analytics-Everywhere-Lab/hts.",
      "published": "2026-06-15T17:24:07Z",
      "abstract_url": "http://arxiv.org/abs/2606.16987v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16987v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Phantoms and Disclosures: a Causal Framework for Auditing Synthetic Data",
      "authors": [
        "Kareem Amin",
        "Rudrajit Das",
        "Alessandro Epasto",
        "Adel Javanmard",
        "Dennis Kraft",
        "Mónica Ribero",
        "Sergei Vassilvitskii"
      ],
      "abstract": "The rapid adoption of generative AI and Large Language Models (LLMs) has spurred interest in synthetic data as a privacy-preserving alternative to sensitive real-world datasets. However, generating high-utility synthetic data often carries the risk of memorizing and regurgitating private information from the training corpus. In this work, we present a customizable empirical auditing framework designed to detect and explain such data disclosures. Our framework introduces a mechanism to distinguish between \"true disclosures\"-where the system directly reproduces a user's information-and \"phantom disclosures''-where the system incidentally generates a user's data. By partitioning input data into training and holdout sets and applying rigorous statistical hypothesis testing, we determine if observed disclosures are consistent with strict privacy baselines, such as zero-learning or specific Differential Privacy (DP) bounds. Crucially, this approach requires no model access, no canary insertion, and no reference model training -only the synthetic output and a held-out control set. We demonstrate that this framework effectively functions as a membership inference attack, providing empirical lower bounds on privacy leakage that are tighter than prior data-based auditing methods. Our approach is model-agnostic, applies to any synthetic data generation mechanism, and requires orders of magnitude fewer computational resources than shadow-model or canary-based alternatives.",
      "published": "2026-06-15T16:54:02Z",
      "abstract_url": "http://arxiv.org/abs/2606.16952v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16952v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.AP",
        "stat.ME",
        "stat.ML"
      ]
    },
    {
      "title": "Scalable Circuit Learning for Interpreting Large Language Models",
      "authors": [
        "Naiyu Yin",
        "Dennis Wei",
        "Tian Gao",
        "Amit Dhurandhar",
        "Karthikeyan Natesan Ramamurthy",
        "Yue Yu"
      ],
      "abstract": "A prominent research direction in mechanistic interpretability is learning sparse circuits over LLM components to reveal how they jointly produce model behavior. However, raw neurons are polysemantic, making learned circuits hard to interpret. Sparse autoencoder (SAE) features alleviate this, but their high dimensionality makes existing intervention-based circuit learning methods computationally prohibitive. We propose CircuitLasso, a scalable circuit-learning approach based on sparse linear regression. CircuitLasso recovers circuits whose structural accuracy matches that of state-of-the-art intervention-based methods on the benchmark data, at a fraction of the computational cost. For interpretability, CircuitLasso efficiently uncovers relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions. Finally, we validate the utility of our learned circuits by leveraging their insights to achieve comparable performance at substantially lower cost on a domain-generalization task.",
      "published": "2026-06-15T16:40:43Z",
      "abstract_url": "http://arxiv.org/abs/2606.16939v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16939v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CrossMaps: Confidence-Aware Open-Vocabulary Semantic Mapping for Rover Navigation",
      "authors": [
        "Jan-Niklas Klein",
        "Sona Ghahremani",
        "Christian Medeiros Adriano",
        "Holger Giese"
      ],
      "abstract": "Rovers rely on perception to maintain spatial maps that encode both objects and sensor quality (e.g., range reliability, lighting artifacts, data density), guiding data fusion, embedding updates, and navigation under partial observability. To study these coupled perception-navigation processes, we present CrossMaps, a real-time confidence-aware open-vocabulary semantic mapping pipeline that constructs language-queryable maps from RGB-D data. Building on VLMaps-style approaches, CrossMaps integrates multi-scale CLIP embeddings with confidence-aware fusion and a dual-memory architecture consisting of Short-Term Memory (STM) and Long-Term Memory (LTM). The STM aggregates noisy visual observations using geometric, semantic, and temporal confidence cues, while confident and coherent cells are promoted to the LTM as persistent semantic landmarks. Designed for deployment with a Jetson Orin-powered UGV alongside SLAM, CrossMaps runs in real time and produces semantic heatmaps that can be queried with natural language to guide rover navigation.",
      "published": "2026-06-15T16:35:01Z",
      "abstract_url": "http://arxiv.org/abs/2606.16935v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16935v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Unified Causal-Origin Taxonomy of Distributional Shifts in Reinforcement Learning",
      "authors": [
        "Ardianto Wibowo",
        "Paulo E Santos",
        "Amer Baghdadi",
        "Matthew Stephenson",
        "Karl Sammut",
        "Jean-Philippe Diguet"
      ],
      "abstract": "Reinforcement learning (RL) systems often degrade when operating conditions differ from those previously encountered, reflecting distributional shifts in the underlying data-generating process. Such shifts may occur between training and evaluation, as in In-Distribution (ID) and Out-of-Distribution (OOD) generalization, or within non-stationary settings where environment dynamics evolve over time. However, the formal relationship between these views remains unclear, and existing work mainly focuses on mitigation rather than the causal origin of shift within the agent-environment interaction. This work develops a unified causal-origin taxonomy that characterizes sources of distributional shift in RL and relates ID/OOD generalization to non-stationary settings. We transfer the classical dataset-shift principle from supervised learning to RL by reformulating distributional shift in terms of the generative interaction process. Using a Partially Observable Markov Decision Process (POMDP), we decompose the interaction into structural components, including the state distribution, observation process, policy, reward, and transition dynamics, together with the shifted-time boundary. The proposed taxonomy distinguishes internal, agent-driven, and external, environment-driven, distributional shifts. The shifted-time boundary perspective further characterizes explicit, implicit, and hybrid shifts. This formulation unifies ID/OOD generalization and non-stationarity as structured changes in the underlying process. We also introduce an evaluation framework for measuring shift impact and adaptation through performance degradation and recovery metrics. By grounding distributional shift in the causal-origin structure of RL, this work supports systematic analysis of robustness under distributional shift.",
      "published": "2026-06-15T16:32:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.16933v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16933v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MA-SBI: Misspecification-Aware Simulation-Based Inference via Side-Channel Guidance",
      "authors": [
        "Arunkumar V",
        "Manoranjan Gandhudi",
        "Gangadharan G. R.",
        "Arun Prakash",
        "S. Senthilkumar"
      ],
      "abstract": "Simulation-based inference (SBI) of latent parameters is often hindered by simulator misspecification, the mismatch between simulated and real-world observations caused by inherent modeling simplifications. RoPE, the recent state-of-the-art for robust SBI, addresses this through optimal transport between learned representations of real and simulated observations, but requires ground-truth parameter calibration pairs that are typically unavailable in the very settings where SBI is needed. What practitioners do have is unstructured side-information such as regime labels, instruction text, and policy bulletins. We propose Misspecification-Aware Simulation-Based Inference (MA-SBI), a calibration-free framework that turns this side-channel into a posterior correction. A learned corrector maps side-channel text to an observation-space shift applied before any pre-trained amortized posterior, requiring no retraining and no parameter ground-truth. Our main theorem bounds achievable bias reduction by the mutual information between misspecification and side-channel, with a non-vacuous constant that extends to all sub-Gaussian noise via Donsker-Varadhan. On hide-the-calibration benchmarks, MA-SBI with text alone matches the oracle posterior across 10 seeds and two backbones (TOST equivalence), while RoPE given more data does not. The two approaches are complementary: where misspecification is structural and recoverable from parameter pairs, RoPE dominates, as the theory predicts. A stochastic variant improves posterior-predictive log-likelihood on real COVID and OxCGRT epidemiological data, and correctly leaves the posterior unchanged on a well-specified cognitive-science corpus.",
      "published": "2026-06-15T16:26:42Z",
      "abstract_url": "http://arxiv.org/abs/2606.16923v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16923v1",
      "categories": [
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Demystifying Variance in Circuit Discovery of LLMs",
      "authors": [
        "Frank Zhengqing Wu",
        "Francesco Tonin",
        "Volkan Cevher"
      ],
      "abstract": "Circuit discovery is a key technique in mechanistic interpretability to pinpoint the model components that are crucial for performing a given task. Although the current state-of-the-art method (EAP-IG) performs well on the metric of (un)faithfulness, it suffers from substantial variability. This includes resampling variance, where the circuit changes when we probe with a new batch of data from the same distribution; rephrasing variance, where the discovered circuit shifts when the prompts are rephrased; and sample-wise variance, where a circuit with low population unfaithfulness exhibits large fluctuations in unfaithfulness across individual samples. This paper studies the roots of these variances. We demonstrate that CEAP, our new circuit discovery method that improves upon EAP-IG with a theoretical guarantee, can substantially lessen resampling variance. We further show that rephrasing variance arises because prompts with different templates tend to activate different circuits in the model. This leads us to argue that it may be challenging to find a comprehensive circuit that explains and controls the model's behavior on a task, which can be expressed in countless templates, suggesting that LLMs may be inherently hard to steer. We show that sparsity, which has been claimed to form more compact and interpretable task circuits, fails to solve this problem. Regarding sample-wise variance, we argue that it is largely benign: extremely poor unfaithfulness scores often stem from how unfaithfulness is defined, rather than from defects in the measured circuits. We show that the magnitude of unfaithfulness is affected by selective contribution scaling, a neural mechanism that accounts for the extremely poor scores sometimes observed.",
      "published": "2026-06-15T16:25:07Z",
      "abstract_url": "http://arxiv.org/abs/2606.16920v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16920v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Weights and Gradients: A Taxonomy of Federated Learning Messages",
      "authors": [
        "Alvaro Javier Vargas Guerrero",
        "Xinguang Wang",
        "Quang Manh Doan",
        "Guy Nagels"
      ],
      "abstract": "Federated Learning is rapidly evolving beyond the exchange of traditional model weights and gradients, yet existing definitions fail to capture the full scope of modern payloads like synthetic data and federated analytics. This paper addresses the gap by proposing a formal mathematical definition of a federated message that accounts for both utility and privacy. We introduce a taxonomy that organizes these exchanges into three categories: model structures, statistical summaries, and data-conditioned representations. By evaluating these groups based on computational demands, communication costs, and privacy risks, we provide a clearer understanding of the trade-offs involved in decentralized training. Our review of 202 recent publications highlights a significant shift since 2021 toward diverse messaging paradigms, signaling a move away from standard deep learning updates toward more specialized information sharing. This framework provides a structured path for future research to optimize federated systems for varying hardware and security requirements.",
      "published": "2026-06-15T16:02:03Z",
      "abstract_url": "http://arxiv.org/abs/2606.16891v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16891v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Compositional Reasoning Depth Predicts Clinical AI Failure: Empirical Evidence Consistent with Transformer Compositionality Limits in Electronic Health Record Question Answering",
      "authors": [
        "Sanjay Basu"
      ],
      "abstract": "Aggregate accuracy benchmarks conceal a systematic structure in how large language models fail at electronic health record (EHR) question answering: questions requiring more inferential steps produce disproportionately more errors. Motivated by theoretical results on transformer compositionality limits, we introduce a pre-specified hop-count taxonomy -- the number of distinct reasoning steps required to answer a clinical question from an EHR -- as a principled predictor of model failure. We annotate 313 clinician-generated MedAlign EHR question-answer pairs across four hop levels and evaluate 301 questions in a within-model ablation (claude-sonnet-4-6, zero-shot vs. extended thinking) and cross-architecture replications (gpt-4o and gpt-5.4-2026-03-05, zero-shot). All three models, spanning two providers and two OpenAI generations (GPT-4 and GPT-5), show monotone accuracy decline with hop count: Claude Sonnet zero-shot falls from 30.6% (hop=1) to 17.6% (hop=4) (Cochran-Armitage z=-2.30, p=0.011; OR per hop 0.72, 95% CI [0.56,0.92], p=0.008); GPT-4o replicates this (37.8% to 14.7%; OR 0.58 [0.45,0.75], p<0.001); and gpt-5.4-2026-03-05 confirms it (37.8% to 23.5%; OR 0.80 [0.66,0.98], p=0.027). A pre-specified context-sufficiency audit shows higher-hop questions are not differentially disadvantaged by EHR truncation (answerability 93-95% at hops 2-4 vs. 79% at hop=1), so the decline reflects compositional reasoning difficulty. Extended thinking did not significantly flatten the accuracy-depth curve across three reasoning conditions, and thinking-token usage scaled with hop count (r=0.31, p<0.0001), consistent with the predicted O(k) computational requirement. Hop count is thus a theory-motivated, cross-architecture predictor of large-language-model error on EHR question answering, with direct implications for deployment risk stratification of clinical AI.",
      "published": "2026-06-15T16:00:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.16890v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16890v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Upper Bounds on the Generalization Error of Deep Learning Models via Local Robustness and Stability",
      "authors": [
        "Abdul-Rauf Nuhu",
        "Parham M. Kebria",
        "Vahid Hemmati",
        "Mahmoud N. Mahmoud",
        "Edward Tunstel",
        "Abdollah Homaifar"
      ],
      "abstract": "Generalization is a critical property of data-driven models, particularly deep learning models deployed in safety-critical applications. Robustness-based generalization bounds have gained attention as a principled way to link robustness properties to generalization performance, often in a data-dependent manner. However, most existing bounds suffer from vacuousness in practical settings, yielding loose upper bounds that greatly exceed the actual error rates and limiting their usefulness for real-world evaluation. While this issue is often attributed to the uncertainty term, a substantial part of the problem originates from the robustness term itself, particularly for the 0-1 loss. Existing approaches typically treat the robustness term as a global measure, ignoring its variation across different sub-regions of the input space. In this work, we propose a generalization bound that addresses this limitation by scaling the robustness term according to the number of stable and unstable samples within each sub-region. Our bounds incorporate both data- and model-dependent factors while maintaining practical relevance (yielding tighter upper bounds on true error). Experiments on models trained on the ImageNet dataset show that our bounds remain consistently non-vacuous and achieve the tightest estimates among existing methods, closely aligning with empirical performance across a range of robust deep neural networks.",
      "published": "2026-06-15T15:55:22Z",
      "abstract_url": "http://arxiv.org/abs/2606.16883v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16883v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens",
      "authors": [
        "Yizhen Yao",
        "Qinglin Zhu",
        "Runcong Zhao",
        "Xiangxiang Dai",
        "Yanzheng Xiang",
        "Yulan He",
        "Lin Gui"
      ],
      "abstract": "Diffusion Large Language Models (dLLMs) offer a promising avenue for parallel generation but face a trade-off between decoding speed and quality. While revocable decoding strategies attempt to mitigate errors by verifying and remasking tokens, they typically operate within a mixed-quality context. This leads to two critical failures: \\textit{Error Propagation}, where new tokens absorb toxic information from erroneous context, and \\textit{Local Error Reinforcement}, where errors mutually reinforce each other to evade detection. To alleviate these challenges, we propose ASRD (Anchor Supervised Revocable Decoding), a training-free framework that operates within the embedding space. ASRD explicitly decouples the decoding context into trusted \\textit{Anchor Tokens}, which are identified via temporal consistency, and uncertain candidates. Leveraging a dynamic Anchor Tokens Cache, we introduce two complementary mechanisms: (1) Anchor-Guided Generation, which injects entropy-weighted anchor signals into masked positions to implicitly rectify attention toward the reliable global skeleton; and (2) Anchor-Perturbed Verification, which applies orthogonal perturbations to uncertain candidate tokens, destabilizing and remasking errors driven by fragile local consensus. Extensive experiments on math and coding benchmarks demonstrate that ASRD outperforms recent remasking baselines, achieving accuracy improvements of up to 6.4\\% while accelerating inference throughput by up to 7.2$\\times$.",
      "published": "2026-06-15T15:23:47Z",
      "abstract_url": "http://arxiv.org/abs/2606.16847v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16847v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Deep Q-Learning on Hölder Spaces",
      "authors": [
        "Qian Qi"
      ],
      "abstract": "We study the operator-theoretic core of Q-learning in continuous-time stochastic control with continuous states and actions. In value-based reinforcement learning, each Q-learning or DQN update is built from a Bellman optimality target; our analysis isolates this target in a diffusion setting and studies its regularity and approximation complexity. Under uniform ellipticity and Hölder-regular coefficients, we show that a Bellman update maps bounded inputs into an anisotropic regularity class, smoothing the state variable while leaving only Lipschitz dependence on the action variable. This yields a compact family of Bellman iterates and motivates a tensor-product DeepONet architecture adapted to the mixed regularity of the problem. We then derive explicit approximation and resource bounds, together with a stiffness--complexity trade-off as the time step $δ\\to 0$. The resulting theory makes a direct contribution to Q-learning theory at the level of Bellman target regularity and approximation in continuous stochastic control. At the same time, we do not claim a full convergence theorem for practical sampled Q-learning with exploration, replay, and stochastic gradient updates.",
      "published": "2026-06-15T15:23:22Z",
      "abstract_url": "http://arxiv.org/abs/2606.16846v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16846v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Robust Dual-Signal Fusion: Hybrid Neuro-Symbolic Gating with Compressed Chain-of-Thought Refinement for Irony Detection in Social Media Texts",
      "authors": [
        "Ankit Bhattacharjee",
        "Krityapriya Bhaumik"
      ],
      "abstract": "Large Language Models (LLMs) natively default to literal semantic interpretations, making zero-shot irony detection a persistent challenge. We introduce the Robust Dual-Signal (RDS) Fusion framework, a hybrid neuro-symbolic architecture that compresses Chain-of-Thought (CoT) reasoning trajectories without Supervised Fine-Tuning (SFT). Evaluated on a strictly held-out TweetEval test set (N=734), RDS achieves 78.1% accuracy and a Macro F1 of 0.777, matching the absolute performance ceiling of the fine-tuned BERTweet. On the heavily imbalanced iSarcasm dataset, the frozen CoT pipeline filters 22.5% of out-of-distribution hallucinations, yielding a zero-shot Macro F1 of 0.6726 and Ironic F1 of 0.4821, outperforming multiple heavily supervised SemEval transformer ensembles. A statistical ablation confirms this structural synergy: adding the symbolic prior to the neural baseline yields no significant gain (p = 0.242), and the marginal benefit of adding the CoT pipeline to that prior is heavily compressed (p = 0.149). Only the complete, concurrent fusion of all three signals achieves a statistically validated improvement over the baseline (p = 0.005).",
      "published": "2026-06-15T15:22:37Z",
      "abstract_url": "http://arxiv.org/abs/2606.16845v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16845v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Models: Reflections on Engineering AI-enabled Systems in a Project-Based Course",
      "authors": [
        "Amir Mashmool",
        "Kishan Ravindra Sawant",
        "Mojtaba Shahin",
        "Nico Hochgeschwender",
        "Rainer Koschke"
      ],
      "abstract": "Teaching Software Engineering for AI-enabled systems entails addressing the integration of AI components within full-scale software architectures under realistic constraints. While machine learning courses emphasize model development, students often lack experience in architectural design, deployment, and monitoring of AI-enabled systems. Empirical evaluations of such system-oriented AI courses remain limited. This paper reflects on the design and implementation of a project-based master's-level course titled AI Algorithms: Theory and Engineering, at the University of Bremen, in which students developed a movie recommendation system while making architectural design decisions to address challenges related to scalability, deployment, and evolving requirements. We conducted a mixed-methods study combining analyses of student submissions and questionnaire responses to investigate integration challenges, learning outcomes, and opportunities for improvement. Our results indicate persistent difficulties in early architectural decisions, heterogeneous ML integration, evolving requirements, and data management, largely due to uneven ML and software engineering expertise. From the educator's perspective, the course fostered system-level reasoning and strengthened awareness of data-centric ML practices in AI-enabled systems.",
      "published": "2026-06-15T15:21:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.16842v1",
      "pdf_url": "https://arxiv.org/pdf/2606.16842v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    }
  ]
};
