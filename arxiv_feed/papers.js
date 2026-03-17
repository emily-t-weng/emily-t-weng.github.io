const PAPERS_DATA = {
  "last_updated": "2026-03-17 02:45:57 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "FairMed-XGB: A Bayesian-Optimised Multi-Metric Framework with Explainability for Demographic Equity in Critical Healthcare Data",
      "authors": [
        "Mitul Goswami",
        "Romit Chatterjee",
        "Arif Ahmed Sekh"
      ],
      "abstract": "Machine learning models deployed in critical care settings exhibit demographic biases, particularly gender disparities, that undermine clinical trust and equitable treatment. This paper introduces FairMed-XGB, a novel framework that systematically detects and mitigates gender-based prediction bias while preserving model performance and transparency. The framework integrates a fairness-aware loss function combining Statistical Parity Difference, Theil Index, and Wasserstein Distance, jointly optimised via Bayesian Search into an XGBoost classifier. Post-mitigation evaluation on seven clinically distinct cohorts derived from the MIMIC-IV-ED and eICU databases demonstrates substantial bias reduction: Statistical Parity Difference decreases by 40 to 51 percent on MIMIC-IV-ED and 10 to 19 percent on eICU; Theil Index collapses by four to five orders of magnitude to near-zero values; Wasserstein Distance is reduced by 20 to 72 percent. These gains are achieved with negligible degradation in predictive accuracy (AUC-ROC drop <0.02). SHAP-based explainability reveals that the framework diminishes reliance on gender-proxy features, providing clinicians with actionable insights into how and where bias is corrected. FairMed-XGB offers a robust, interpretable, and ethically aligned solution for equitable clinical decision-making, paving the way for trustworthy deployment of AI in high-stakes healthcare environments.",
      "published": "2026-03-16T07:57:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.14947v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14947v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Directional Routing in Transformers",
      "authors": [
        "Kevin Taylor"
      ],
      "abstract": "We introduce directional routing, a lightweight mechanism that gives each transformer attention head learned suppression directions controlled by a shared router, at 3.9% parameter cost. We train a 433M-parameter model alongside an identical baseline in a single run, then trace the resulting circuits through mechanistic interpretability. Routing becomes the model's dominant computational pathway. Disabling it collapses factual recall to near-zero probability across all 8 test prompts and drops induction accuracy from 93.4% to 0.0%. Knocking out individual attention heads has negligible effect: the primary mover head's removal actually increases target probability, and induction heads retain 98.6% accuracy without their strongest member. The coordination mechanism is irreplaceable; the components it coordinates are not. The model also self-organizes, without explicit pressure, into two regimes: domain-adaptive routing in early layers and fixed syntactic pruning in late layers, where the least-varying layer is the most critical (+42.6 PPL when disabled). Routing reduces perplexity 31-56% relative to the baseline, though downstream multiple-choice benchmarks do not yet reflect these gains.",
      "published": "2026-03-16T07:28:22Z",
      "abstract_url": "http://arxiv.org/abs/2603.14923v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14923v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Spectrogram features for audio and speech analysis",
      "authors": [
        "Ian McLoughlin",
        "Lam Pham",
        "Yan Song",
        "Xiaoxiao Miao",
        "Huy Phan",
        "Pengfei Cai",
        "Qing Gu",
        "Jiang Nan",
        "Haoyu Song",
        "Donny Soh"
      ],
      "abstract": "Spectrogram-based representations have grown to dominate the feature space for deep learning audio analysis systems, and are often adopted for speech analysis also. Initially, the primary motivator for spectrogram-based representations was their ability to present sound as a two dimensional signal in the time-frequency plane, which not only provides an interpretable physical basis for analysing sound, but also unlocks the use of a wide range of machine learning techniques such as convolutional neural networks, that had been developed for image processing. A spectrogram is a matrix characterised by the resolution and span of its two dimensions, as well as by the representation and scaling of each element. Many possibilities for these three characteristics have been explored by researchers across numerous application areas, with different settings showing affinity for various tasks. This paper reviews the use of spectrogram-based representations and surveys the state-of-the-art to question how front-end feature representation choice allies with back-end classifier architecture for different tasks.",
      "published": "2026-03-16T07:24:58Z",
      "abstract_url": "http://arxiv.org/abs/2603.14917v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14917v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.LG",
        "eess.SP"
      ]
    },
    {
      "title": "Informative Perturbation Selection for Uncertainty-Aware Post-hoc Explanations",
      "authors": [
        "Sumedha Chugh",
        "Ranjitha Prasad",
        "Nazreen Shah"
      ],
      "abstract": "Trust and ethical concerns due to the widespread deployment of opaque machine learning (ML) models motivating the need for reliable model explanations. Post-hoc model-agnostic explanation methods addresses this challenge by learning a surrogate model that approximates the behavior of the deployed black-box ML model in the locality of a sample of interest. In post-hoc scenarios, neither the underlying model parameters nor the training are available, and hence, this local neighborhood must be constructed by generating perturbed inputs in the neighborhood of the sample of interest, and its corresponding model predictions. We propose \\emph{Expected Active Gain for Local Explanations} (\\texttt{EAGLE}), a post-hoc model-agnostic explanation framework that formulates perturbation selection as an information-theoretic active learning problem. By adaptively sampling perturbations that maximize the expected information gain, \\texttt{EAGLE} efficiently learns a linear surrogate explainable model while producing feature importance scores along with the uncertainty/confidence estimates. Theoretically, we establish that cumulative information gain scales as $\\mathcal{O}(d \\log t)$, where $d$ is the feature dimension and $t$ represents the number of samples, and that the sample complexity grows linearly with $d$ and logarithmically with the confidence parameter $1/δ$. Empirical results on tabular and image datasets corroborate our theoretical findings and demonstrate that \\texttt{EAGLE} improves explanation reproducibility across runs, achieves higher neighborhood stability, and improves perturbation sample quality as compared to state-of-the-art baselines such as Tilia, US-LIME, GLIME and BayesLIME.",
      "published": "2026-03-16T06:48:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.14894v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14894v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "LLMs as Signal Detectors: Sensitivity, Bias, and the Temperature-Criterion Analogy",
      "authors": [
        "Jon-Paul Cacioli"
      ],
      "abstract": "Large language models (LLMs) are evaluated for calibration using metrics such as Expected Calibration Error that conflate two distinct components: the model's ability to discriminate correct from incorrect answers (sensitivity) and its tendency toward confident or cautious responding (bias). Signal Detection Theory (SDT) decomposes these components. While SDT-derived metrics such as AUROC are increasingly used, the full parametric framework - unequal-variance model fitting, criterion estimation, z-ROC analysis - has not been applied to LLMs as signal detectors. In this pre-registered study, we treat three LLMs as observers performing factual discrimination across 168,000 trials and test whether temperature functions as a criterion shift analogous to payoff manipulations in human psychophysics. Critically, this analogy may break down because temperature changes the generated answer itself, not only the confidence assigned to it. Our results confirm the breakdown with temperature simultaneously increasing sensitivity (AUC) and shifting criterion. All models exhibited unequal-variance evidence distributions (z-ROC slopes 0.52-0.84), with instruct models showing more extreme asymmetry (0.52-0.63) than the base model (0.77-0.87) or human recognition memory (~0.80). The SDT decomposition revealed that models occupying distinct positions in sensitivity-bias space could not be distinguished by calibration metrics alone, demonstrating that the full parametric framework provides diagnostic information unavailable from existing metrics.",
      "published": "2026-03-16T06:45:59Z",
      "abstract_url": "http://arxiv.org/abs/2603.14893v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14893v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Decision-Level Ordinal Modeling for Multimodal Essay Scoring with Large Language Models",
      "authors": [
        "Han Zhang",
        "Jiamin Su",
        "Li liu"
      ],
      "abstract": "Automated essay scoring (AES) predicts multiple rubric-defined trait scores for each essay, where each trait follows an ordered discrete rating scale. Most LLM-based AES methods cast scoring as autoregressive token generation and obtain the final score via decoding and parsing, making the decision implicit. This formulation is particularly sensitive in multimodal AES, where the usefulness of visual inputs varies across essays and traits. To address these limitations, we propose Decision-Level Ordinal Modeling (DLOM), which makes scoring an explicit ordinal decision by reusing the language model head to extract score-wise logits on predefined score tokens, enabling direct optimization and analysis in the score space. For multimodal AES, DLOM-GF introduces a gated fusion module that adaptively combines textual and multimodal score logits. For text-only AES, DLOM-DA adds a distance-aware regularization term to better reflect ordinal distances. Experiments on the multimodal EssayJudge dataset show that DLOM improves over a generation-based SFT baseline across scoring traits, and DLOM-GF yields further gains when modality relevance is heterogeneous. On the text-only ASAP/ASAP++ benchmarks, DLOM remains effective without visual inputs, and DLOM-DA further improves performance and outperforms strong representative baselines.",
      "published": "2026-03-16T06:40:18Z",
      "abstract_url": "http://arxiv.org/abs/2603.14891v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14891v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Seismic full-waveform inversion based on a physics-driven generative adversarial network",
      "authors": [
        "Xinyi Zhang",
        "Caiyun Liu",
        "Jie Xiong",
        "Qingfeng Yu"
      ],
      "abstract": "Objectives: Full-waveform inversion (FWI) is a high-resolution geophysical imaging technique that reconstructs subsurface velocity models by iteratively minimizing the misfit between predicted and observed seismic data. However, under complex geological conditions, conventional FWI suffers from strong dependence on the initial model and tends to produce unstable results when the data are sparse or contaminated by noise. Methods: To address these limitations, this paper proposes a physics-driven generative adversarial network-based full-waveform inversion method. The proposed approach integrates the data-driven capability of deep neural networks with the physical constraints imposed by the seismic wave equation, and employs adversarial training through a discriminator to enhance the stability and robustness of the inversion results. Results: Experimental results on two representative benchmark geological models demonstrate that the proposed method can effectively recover complex velocity structures and achieves superior performance in terms of structural similarity (SSIM) and signal-to-noise ratio (SNR). Conclusions: This method provides a promising solution for alleviating the initial-model dependence in full-waveform inversion and shows strong potential for practical applications.",
      "published": "2026-03-16T06:24:17Z",
      "abstract_url": "http://arxiv.org/abs/2603.14879v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14879v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "IgPose: A Generative Data-Augmented Pipeline for Robust Immunoglobulin-Antigen Binding Prediction",
      "authors": [
        "Tien-Cuong Bui",
        "Injae Chung",
        "Wonjun Lee",
        "Junsu Ko",
        "Juyong Lee"
      ],
      "abstract": "Predicting immunoglobulin-antigen (Ig-Ag) binding remains a significant challenge due to the paucity of experimentally-resolved complexes and the limited accuracy of de novo Ig structure prediction. We introduce IgPose, a generalizable framework for Ig-Ag pose identification and scoring, built on a generative data-augmentation pipeline. To mitigate data scarcity, we constructed the Structural Immunoglobulin Decoy Database (SIDD), a comprehensive repository of high-fidelity synthetic decoys. IgPose integrates equivariant graph neural networks, ESM-2 embeddings, and gated recurrent units to synergistically capture both geometric and evolutionary features. We implemented interface-focused k-hop sampling with biologically guided pooling to enhance generalization across diverse interfaces. The framework comprises two sub-networks--IgPoseClassifier for binding pose discrimination and IgPoseScore for DockQ score estimation--and achieves robust performance on curated internal test sets and the CASP-16 benchmark compared to physics and deep learning baselines. IgPose serves as a versatile computational tool for high-throughput antibody discovery pipelines by providing accurate pose filtering and ranking. IgPose is available on GitHub (https://github.com/arontier/igpose).",
      "published": "2026-03-16T06:16:03Z",
      "abstract_url": "http://arxiv.org/abs/2603.14870v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14870v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Sample-Efficient Hypergradient Estimation for Decentralized Bi-Level Reinforcement Learning",
      "authors": [
        "Mikoto Kudo",
        "Takumi Tanabe",
        "Akifumi Wachi",
        "Youhei Akimoto"
      ],
      "abstract": "Many strategic decision-making problems, such as environment design for warehouse robots, can be naturally formulated as bi-level reinforcement learning (RL), where a leader agent optimizes its objective while a follower solves a Markov decision process (MDP) conditioned on the leader's decisions. In many situations, a fundamental challenge arises when the leader cannot intervene in the follower's optimization process; it can only observe the optimization outcome. We address this decentralized setting by deriving the hypergradient of the leader's objective, i.e., the gradient of the leader's strategy that accounts for changes in the follower's optimal policy. Unlike prior hypergradient-based methods that require extensive data for repeated state visits or rely on gradient estimators whose complexity can increase substantially with the high-dimensional leader's decision space, we leverage the Boltzmann covariance trick to derive an alternative hypergradient formulation. This enables efficient hypergradient estimation solely from interaction samples, even when the leader's decision space is high-dimensional. Additionally, to our knowledge, this is the first method that enables hypergradient-based optimization for 2-player Markov games in decentralized settings. Experiments highlight the impact of hypergradient updates and demonstrate our method's effectiveness in both discrete and continuous state tasks.",
      "published": "2026-03-16T06:11:00Z",
      "abstract_url": "http://arxiv.org/abs/2603.14867v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14867v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.GT",
        "cs.MA"
      ]
    },
    {
      "title": "IntegratingWeather Foundation Model and Satellite to Enable Fine-Grained Solar Irradiance Forecasting",
      "authors": [
        "Ziqing Ma",
        "Kai Ying",
        "Xinyue Gu",
        "Tian Zhou",
        "Tianyu Zhu",
        "Haifan Zhang",
        "Peisong Niu",
        "Wang Zheng",
        "Cong Bai",
        "Liang Sun"
      ],
      "abstract": "Accurate day-ahead solar irradiance forecasting is essential for integrating solar energy into the power grid. However, it remains challenging due to the pronounced diurnal cycle and inherently complex cloud dynamics. Current methods either lack fine-scale resolution (e.g., numerical weather prediction, weather foundation models) or degrade at longer lead times (e.g., satellite extrapolation). We propose Baguan-solar, a two-stage multimodal framework that fuses forecasts from Baguan, a global weather foundation model, with high-resolution geostationary satellite imagery to produce 24- hour irradiance forecasts at kilometer scale. Its decoupled two-stage design first forecasts day-night continuous intermediates (e.g., cloud cover) and then infers irradiance, while its modality fusion jointly preserves fine-scale cloud structures from satellite and large-scale constraints from Baguan forecasts. Evaluated over East Asia using CLDAS as ground truth, Baguan-solar outperforms strong baselines (including ECMWF IFS, vanilla Baguan, and SolarSeer), reducing RMSE by 16.08% and better resolving cloud-induced transients. An operational deployment of Baguan-solar has supported solar power forecasting in an eastern province in China, since July 2025. Our code is accessible at https://github.com/DAMO-DI-ML/Baguansolar. git.",
      "published": "2026-03-16T05:44:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.14845v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14845v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Real-Time Driver Safety Scoring Through Inverse Crash Probability Modeling",
      "authors": [
        "Joyjit Roy",
        "Samaresh Kumar Singh"
      ],
      "abstract": "Road crashes remain a leading cause of preventable fatalities. Existing prediction models predominantly produce binary outcomes, which offer limited actionable insights for real-time driver feedback. These approaches often lack continuous risk quantification, interpretability, and explicit consideration of vulnerable road users (VRUs), such as pedestrians and cyclists. This research introduces SafeDriver-IQ, a framework that transforms binary crash classifiers into continuous 0-100 safety scores by combining national crash statistics with naturalistic driving data from autonomous vehicles. The framework fuses National Highway Traffic Safety Administration (NHTSA) crash records with Waymo Open Motion Dataset scenarios, engineers domain-informed features, and incorporates a calibration layer grounded in transportation safety literature. Evaluation across 15 complementary analyses indicates that the framework reliably differentiates high-risk from low-risk driving conditions with strong discriminative performance. Findings further reveal that 87% of crashes involve multiple co-occurring risk factors, with non-linear compounding effects that increase the risk to 4.5x baseline. SafeDriver-IQ delivers proactive, explainable safety intelligence relevant to advanced driver-assistance systems (ADAS), fleet management, and urban infrastructure planning. This framework shifts the focus from reactive crash counting to real-time risk prevention.",
      "published": "2026-03-16T05:37:12Z",
      "abstract_url": "http://arxiv.org/abs/2603.14841v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14841v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC",
        "cs.ET"
      ]
    },
    {
      "title": "Ablate and Rescue: A Causal Analysis of Residual Stream Hyper-Connections",
      "authors": [
        "William Peng",
        "Josheev Rai",
        "Kevin Tseng",
        "Siwei Wang",
        "Sean Wu"
      ],
      "abstract": "Multi-stream transformer architectures have recently been proposed as a promising direction for managing representation collapse and the vanishing gradient problem for residual connections, yet their internal mechanisms remain unexplored. In particular, the recently introduced Manifold-Constrained Hyper-Connections (mHC) architecture posits multiple residual streams with constrained interaction, but lacks in-depth mechanistic analysis. We present the first open-source mHC language model (https://huggingface.co/wgpeng/mhc-780m) and analyze the multiple-stream architecture with a suite of representation-level metrics and causal interventions to probe how parallel streams encode and utilize information. Specifically, we introduce a systematic stream ablation-and-rescue framework that enables direct causal comparison of residual streams during inference. Through targeted pairwise interventions and controlled recovery experiments, we distinguish functional redundancy from asymmetric utilization and reveal how information is distributed across streams beyond what is observable from representational similarity alone.",
      "published": "2026-03-16T05:24:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.14833v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14833v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Two Birds, One Projection: Harmonizing Safety and Utility in LVLMs via Inference-time Feature Projection",
      "authors": [
        "Yewon Han",
        "Yumin Seol",
        "EunGyung Kong",
        "Minsoo Jo",
        "Taesup Kim"
      ],
      "abstract": "Existing jailbreak defence frameworks for Large Vision-Language Models often suffer from a safety utility tradeoff, where strengthening safety inadvertently degrades performance on general visual-grounded reasoning tasks. In this work, we investigate whether safety and utility are inherently antagonistic objectives. We focus on a modality induced bias direction consistently observed across datasets, which arises from suboptimal coupling between the Large Language Model backbone and visual encoders. We further demonstrate that this direction undermines performance on both tasks. Leveraging this insight, we propose Two Birds, One Projection, an efficient inference time jailbreak defence that projects cross-modal features onto the null space of the identified bias direction to remove the corresponding components. Requiring only a single forward pass, our method effectively breaks the conventional tradeoff, simultaneously improving both safety and utility across diverse benchmarks.",
      "published": "2026-03-16T04:59:24Z",
      "abstract_url": "http://arxiv.org/abs/2603.14825v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14825v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "SimCert: Probabilistic Certification for Behavioral Similarity in Deep Neural Network Compression",
      "authors": [
        "Jingyang Li",
        "Fu Song",
        "Guoqiang Li"
      ],
      "abstract": "Deploying Deep Neural Networks (DNNs) on resource-constrained embedded systems requires aggressive model compression techniques like quantization and pruning. However, ensuring that the compressed model preserves the behavioral fidelity of the original design is a critical challenge in the safety-critical system design flow. Existing verification methods often lack scalability or fail to handle the architectural heterogeneity introduced by pruning. In this work, we propose SimCert, a probabilistic certification framework for verifying the behavioral similarity of compressed neural networks. Unlike worst-case analysis, SimCert provides quantitative safety guarantees with adjustable confidence levels. Our framework features: (1) A dual-network symbolic propagation method supporting both quantization and pruning; (2) A variance-aware bounding technique using Bernstein's inequality to tighten safety certificates; and (3) An automated verification toolchain. Experimental results on ACAS Xu and computer vision benchmarks demonstrate that SimCert outperforms state-of-the-art baselines.",
      "published": "2026-03-16T04:44:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.14818v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14818v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Universe Routing: Why Self-Evolving Agents Need Epistemic Control",
      "authors": [
        "Zhaohui Geoffrey Wang"
      ],
      "abstract": "A critical failure mode of current lifelong agents is not lack of knowledge, but the inability to decide how to reason. When an agent encounters \"Is this coin fair?\" it must recognize whether to invoke frequentist hypothesis testing or Bayesian posterior inference - frameworks that are epistemologically incompatible. Mixing them produces not minor errors, but structural failures that propagate across decision chains. We formalize this as the universe routing problem: classifying questions into mutually exclusive belief spaces before invoking specialized solvers. Our key findings challenge conventional assumptions: (1) hard routing to heterogeneous solvers matches soft MoE accuracy while being 7x faster because epistemically incompatible frameworks cannot be meaningfully averaged; (2) a 465M-parameter router achieves a 2.3x smaller generalization gap than keyword-matching baselines, indicating semantic rather than surface-level reasoning; (3) when expanding to new belief spaces, rehearsal-based continual learning achieves zero forgetting, outperforming EWC by 75 percentage points, suggesting that modular epistemic architectures are fundamentally more amenable to lifelong learning than regularization-based approaches. These results point toward a broader architectural principle: reliable self-evolving agents may require an explicit epistemic control layer that governs reasoning framework selection.",
      "published": "2026-03-16T03:58:14Z",
      "abstract_url": "http://arxiv.org/abs/2603.14799v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14799v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Multi-Task Genetic Algorithm with Multi-Granularity Encoding for Protein-Nucleotide Binding Site Prediction",
      "authors": [
        "Yiming Gao",
        "Liuyi Xu",
        "Pengshan Cui",
        "Yining Qian",
        "An-Yang Lu",
        "Xianpeng Wang"
      ],
      "abstract": "Accurate identification of protein-nucleotide binding sites is fundamental to deciphering molecular mechanisms and accelerating drug discovery. However, current computational methods often struggle with suboptimal performance due to inadequate feature representation and rigid fusion mechanisms, which hinder the effective exploitation of cross-task information synergy. To bridge this gap, we propose MTGA-MGE, a framework that integrates a Multi-Task Genetic Algorithm with Multi-Granularity Encoding to enhance binding site prediction. Specifically, we develop a Multi-Granularity Encoding (MGE) network that synergizes multi-scale convolutions and self-attention mechanisms to distill discriminative signals from high-dimensional, redundant biological data. To overcome the constraints of static fusion, a genetic algorithm is employed to adaptively evolve task-specific fusion strategies, thereby effectively improving model generalization. Furthermore, to catalyze collaborative learning, we introduce an External-Neighborhood Mechanism (ENM) that leverages biological similarities to facilitate targeted information exchange across tasks. Extensive evaluations on fifteen nucleotide datasets demonstrate that MTGA-MGE not only establishes a new state-of-the-art in data-abundant, high-resource scenarios but also maintains a robust competitive edge in rare, low-resource regimes, presenting a highly adaptive scheme for decoding complex protein-ligand interactions in the post-genomic era.",
      "published": "2026-03-16T03:51:54Z",
      "abstract_url": "http://arxiv.org/abs/2603.14797v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14797v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LaPro-DTA: Latent Dual-View Drug Representations and Salient Protein Feature Extraction for Generalizable Drug--Target Affinity Prediction",
      "authors": [
        "Zihan Dun",
        "Liuyi Xu",
        "An-Yang Lu",
        "Shuang Li",
        "Yining Qian"
      ],
      "abstract": "Drug--target affinity prediction is pivotal for accelerating drug discovery, yet existing methods suffer from significant performance degradation in realistic cold-start scenarios (unseen drugs/targets/pairs), primarily driven by overfitting to training instances and information loss from irrelevant target sequences. In this paper, we propose LaPro-DTA, a framework designed to achieve robust and generalizable DTA prediction. To tackle overfitting, we devise a latent dual-view drug representation mechanism. It synergizes an instance-level view to capture fine-grained substructures with stochastic perturbation and a distribution-level view to distill generalized chemical scaffolds via semantic remapping, thereby enforcing the model to learn transferable structural rules rather than memorizing specific samples. To mitigate information loss, we introduce a salient protein feature extraction strategy using pattern-aware top-$k$ pooling, which effectively filters background noise and isolates high-response bioactive regions. Furthermore, a cross-view multi-head attention mechanism fuses these purified features to model comprehensive interactions. Extensive experiments on benchmark datasets demonstrate that LaPro-DTA significantly outperforms state-of-the-art methods, achieving an 8\\% MSE reduction on the Davis dataset in the challenging unseen-drug setting, while offering interpretable insights into binding mechanisms.",
      "published": "2026-03-16T03:46:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.14792v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14792v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "$p^2$RAG: Privacy-Preserving RAG Service Supporting Arbitrary Top-$k$ Retrieval",
      "authors": [
        "Yulong Ming",
        "Mingyue Wang",
        "Jijia Yang",
        "Cong Wang",
        "Xiaohua Jia"
      ],
      "abstract": "Retrieval-Augmented Generation (RAG) enables large language models to use external knowledge, but outsourcing the RAG service raises privacy concerns for both data owners and users. Privacy-preserving RAG systems address these concerns by performing secure top-$k$ retrieval, which typically is secure sorting to identify relevant documents. However, existing systems face challenges supporting arbitrary $k$ due to their inability to change $k$, new security issues, or efficiency degradation with large $k$. This is a significant limitation because modern long-context models generally achieve higher accuracy with larger retrieval sets. We propose $p^2$RAG, a privacy-preserving RAG service that supports arbitrary top-$k$ retrieval. Unlike existing systems, $p^2$RAG avoids sorting candidate documents. Instead, it uses an interactive bisection method to determine the set of top-$k$ documents. For security, $p^2$RAG uses secret sharing on two semi-honest non-colluding servers to protect the data owner's database and the user's prompt. It enforces restrictions and verification to defend against malicious users and tightly bound the information leakage of the database. The experiments show that $p^2$RAG is 3--300$\\times$ faster than the state-of-the-art PRAG for $k = 16$--$1024$.",
      "published": "2026-03-16T03:19:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.14778v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14778v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation",
      "authors": [
        "Qiyuan Chen",
        "Xian Wu",
        "Yi Wang",
        "Xianhao Chen"
      ],
      "abstract": "Fine-tuning large models on edge devices is severely hindered by the memory-intensive backpropagation (BP) in standard frameworks like federated learning and split learning. While substituting BP with zeroth-order optimization can significantly reduce memory footprints, it typically suffers from prohibitively degraded convergence speed. To resolve this dilemma, we propose Hybrid-Order Split Federated Learning (HO-SFL). By reformulating the split learning process within a Lagrangian framework, HO-SFL decouples the optimization landscape: The server performs precise first-order updates (i.e., BP), whereas clients conduct memory-efficient zeroth-order optimization. This hybrid design not only eliminates the need for client-side BP but also enables dimension-free model aggregation, drastically lowering communication costs. Crucially, we provide a theoretical convergence analysis, demonstrating that HO-SFL mitigates the dimension-dependent convergence slowdown of zeroth-order optimization, achieving a convergence rate comparable to first-order methods. Extensive experiments on tasks across vision and language modalities validate that HO-SFL achieves convergence speeds comparable to first-order baselines while significantly reducing communication costs and client memory footprints.",
      "published": "2026-03-16T03:13:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.14773v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14773v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "OpenHospital: A Thing-in-itself Arena for Evolving and Benchmarking LLM-based Collective Intelligence",
      "authors": [
        "Peigen Liu",
        "Rui Ding",
        "Yuren Mao",
        "Ziyan Jiang",
        "Yuxiang Ye",
        "Yunjun Gao",
        "Ying Zhang",
        "Renjie Sun",
        "Longbin Lai",
        "Zhengping Qian"
      ],
      "abstract": "Large Language Model (LLM)-based Collective Intelligence (CI) presents a promising approach to overcoming the data wall and continuously boosting the capabilities of LLM agents. However, there is currently no dedicated arena for evolving and benchmarking LLM-based CI. To address this gap, we introduce OpenHospital, an interactive arena where physician agents can evolve CI through interactions with patient agents. This arena employs a data-in-agent-self paradigm that rapidly enhances agent capabilities and provides robust evaluation metrics for benchmarking both medical proficiency and system efficiency. Experiments demonstrate the effectiveness of OpenHospital in both fostering and quantifying CI.",
      "published": "2026-03-16T03:12:18Z",
      "abstract_url": "http://arxiv.org/abs/2603.14771v1",
      "pdf_url": "https://arxiv.org/pdf/2603.14771v1",
      "categories": [
        "cs.AI"
      ]
    }
  ]
};
