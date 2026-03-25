const PAPERS_DATA = {
  "last_updated": "2026-03-25 02:51:19 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions",
      "authors": [
        "Adrian Bulat",
        "Alberto Baldrati",
        "Ioannis Maniadis Metaxas",
        "Yassine Ouali",
        "Georgios Tzimiropoulos"
      ],
      "abstract": "Existing approaches for improving the efficiency of Large Vision-Language Models (LVLMs) are largely based on the concept of visual token reduction. This approach, however, creates an information bottleneck that impairs performance, especially on challenging tasks that require fine-grained understanding and reasoning. In this work, we challenge this paradigm by introducing VISion On Request (VISOR), a method that reduces inference cost without discarding visual information. Instead of compressing the image, VISOR improves efficiency by sparsifying the interaction between image and text tokens. Specifically, the language model attends to the full set of high-resolution visual tokens through a small, strategically placed set of attention layers: general visual context is provided by efficient cross-attention between text-image, while a few well-placed and dynamically selected self-attention layers refine the visual representations themselves, enabling complex, high-resolution reasoning when needed. Based on this principle, we first train a single universal network on a range of computational budgets by varying the number of self-attention layers, and then introduce a lightweight policy mechanism that dynamically allocates visual computation based on per-sample complexity. Extensive experiments show that VISOR drastically reduces computational cost while matching or exceeding state-of-the-art results across a diverse suite of benchmarks, and excels in challenging tasks that require detailed visual understanding.",
      "published": "2026-03-24T17:58:17Z",
      "abstract_url": "http://arxiv.org/abs/2603.23495v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23495v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Failure of contextual invariance in gender inference with large language models",
      "authors": [
        "Sagar Kumar",
        "Ariel Flint",
        "Luca Maria Aiello",
        "Andrea Baronchelli"
      ],
      "abstract": "Standard evaluation practices assume that large language model (LLM) outputs are stable under contextually equivalent formulations of a task. Here, we test this assumption in the setting of gender inference. Using a controlled pronoun selection task, we introduce minimal, theoretically uninformative discourse context and find that this induces large, systematic shifts in model outputs. Correlations with cultural gender stereotypes, present in decontextualized settings, weaken or disappear once context is introduced, while theoretically irrelevant features, such as the gender of a pronoun for an unrelated referent, become the most informative predictors of model behaviour. A Contextuality-by-Default analysis reveals that, in 19--52\\% of cases across models, this dependence persists after accounting for all marginal effects of context on individual outputs and cannot be attributed to simple pronoun repetition. These findings show that LLM outputs violate contextual invariance even under near-identical syntactic formulations, with implications for bias benchmarking and deployment in high-stakes settings.",
      "published": "2026-03-24T17:52:22Z",
      "abstract_url": "http://arxiv.org/abs/2603.23485v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23485v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "ReqFusion: A Multi-Provider Framework for Automated PEGS Analysis Across Software Domains",
      "authors": [
        "Muhammad Khalid",
        "Manuel Oriol",
        "Yilmaz Uygun"
      ],
      "abstract": "Requirements engineering is a vital, yet labor-intensive, stage in the software development process. This article introduces ReqFusion: an AI-enhanced system that automates the extraction, classification, and analysis of software requirements utilizing multiple Large Language Model (LLM) providers. The architecture of ReqFusion integrates OpenAI GPT, Anthropic Claude, and Groq models to extract functional and non-functional requirements from various documentation formats (PDF, DOCX, and PPTX) in academic, industrial, and tender proposal contexts. The system uses a domain-independent extraction method and generates requirements following the Project, Environment, Goal, and System (PEGS) approach introduced by Bertrand Meyer. The main idea is that, because the PEGS format is detailed, LLMs have more information and cues about the requirements, producing better results than a simple generic request. An ablation study confirms this hypothesis: PEGS-guided prompting achieves an F1 score of 0.88, compared to 0.71 for generic prompting under the same multi-provider configuration. The evaluation used 18 real-world documents to generate 226 requirements through automated classification, with 54.9% functional and 45.1% nonfunctional across academic, business, and technical domains. An extended evaluation on five projects with 1,050 requirements demonstrated significant improvements in extraction accuracy and a 78% reduction in analysis time compared to manual methods. The multi-provider architecture enhances reliability through model consensus and fallback mechanisms, while the PEGS-based approach ensures comprehensive coverage of all requirement categories.",
      "published": "2026-03-24T17:45:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.23482v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23482v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs",
      "authors": [
        "Haoran Yuan",
        "Weigang Yi",
        "Zhenyu Zhang",
        "Wendi Chen",
        "Yuchen Mo",
        "Jiashi Yin",
        "Xinzhuo Li",
        "Xiangyu Zeng",
        "Chuan Wen",
        "Cewu Lu",
        "Katherine Driggs-Campbell",
        "Ismini Lourentzou"
      ],
      "abstract": "Video-Action Models (VAMs) have emerged as a promising framework for embodied intelligence, learning implicit world dynamics from raw video streams to produce temporally consistent action predictions. Although such models demonstrate strong performance on long-horizon tasks through visual reasoning, they remain limited in contact-rich scenarios where critical interaction states are only partially observable from vision alone. In particular, fine-grained force modulation and contact transitions are not reliably encoded in visual tokens, leading to unstable or imprecise behaviors. To bridge this gap, we introduce the Video-Tactile Action Model (VTAM), a multimodal world modeling framework that incorporates tactile perception as a complementary grounding signal. VTAM augments a pretrained video transformer with tactile streams via a lightweight modality transfer finetuning, enabling efficient cross-modal representation learning without tactile-language paired data or independent tactile pretraining. To stabilize multimodal fusion, we introduce a tactile regularization loss that enforces balanced cross-modal attention, preventing visual latent dominance in the action model. VTAM demonstrates superior performance in contact-rich manipulation, maintaining a robust success rate of 90 percent on average. In challenging scenarios such as potato chip pick-and-place requiring high-fidelity force awareness, VTAM outperforms the pi 0.5 baseline by 80 percent. Our findings demonstrate that integrating tactile feedback is essential for correcting visual estimation errors in world action models, providing a scalable approach to physically grounded embodied foundation models.",
      "published": "2026-03-24T17:45:06Z",
      "abstract_url": "http://arxiv.org/abs/2603.23481v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23481v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "3DCity-LLM: Empowering Multi-modality Large Language Models for 3D City-scale Perception and Understanding",
      "authors": [
        "Yiping Chen",
        "Jinpeng Li",
        "Wenyu Ke",
        "Yang Luo",
        "Jie Ouyang",
        "Zhongjie He",
        "Li Liu",
        "Hongchao Fan",
        "Hao Wu"
      ],
      "abstract": "While multi-modality large language models excel in object-centric or indoor scenarios, scaling them to 3D city-scale environments remains a formidable challenge. To bridge this gap, we propose 3DCity-LLM, a unified framework designed for 3D city-scale vision-language perception and understanding. 3DCity-LLM employs a coarse-to-fine feature encoding strategy comprising three parallel branches for target object, inter-object relationship, and global scene. To facilitate large-scale training, we introduce 3DCity-LLM-1.2M dataset that comprises approximately 1.2 million high-quality samples across seven representative task categories, ranging from fine-grained object analysis to multi-faceted scene planning. This strictly quality-controlled dataset integrates explicit 3D numerical information and diverse user-oriented simulations, enriching the question-answering diversity and realism of urban scenarios. Furthermore, we apply a multi-dimensional protocol based on text-similarity metrics and LLM-based semantic assessment to ensure faithful and comprehensive evaluations for all methods. Extensive experiments on two benchmarks demonstrate that 3DCity-LLM significantly outperforms existing state-of-the-art methods, offering a promising and meaningful direction for advancing spatial reasoning and urban intelligence. The source code and dataset are available at https://github.com/SYSU-3DSTAILab/3D-City-LLM.",
      "published": "2026-03-24T17:18:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.23447v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23447v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Evaluating LLM-Based Test Generation Under Software Evolution",
      "authors": [
        "Sabaat Haroon",
        "Mohammad Taha Khan",
        "Muhammad Ali Gulzar"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly used for automated unit test generation. However, it remains unclear whether these tests reflect genuine reasoning about program behavior or simply reproduce superficial patterns learned during training. If the latter dominates, LLM-generated tests may exhibit weaknesses such as reduced coverage, missed regressions, and undetected faults. Understanding how LLMs generate tests and how those tests respond to code evolution is therefore essential. We present a large-scale empirical study of LLM-based test generation under program changes. Using an automated mutation-driven framework, we analyze how generated tests react to semantic-altering changes (SAC) and semantic-preserving changes (SPC) across eight LLMs and 22,374 program variants. LLMs achieve strong baseline results, reaching 79% line coverage and 76% branch coverage with fully passing test suites on the original programs. However, performance degrades as programs evolve. Under SACs, the pass rate of newly generated tests drops to 66%, and branch coverage declines to 60%. More than 99% of failing SAC tests pass on the original program while executing the modified region, indicating residual alignment with the original behavior rather than adaptation to updated semantics. Performance also declines under SPCs despite unchanged functionality: pass rates fall to 79% and branch coverage to 69%. Although SPC edits preserve semantics, they often introduce larger syntactic changes, leading to instability in generated test suites. Models generate more new tests while discarding many baseline tests, suggesting sensitivity to lexical changes rather than true semantic impact. Overall, our results indicate that current LLM-based test generation relies heavily on surface-level cues and struggles to maintain regression awareness as programs evolve.",
      "published": "2026-03-24T17:14:18Z",
      "abstract_url": "http://arxiv.org/abs/2603.23443v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23443v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Targeted Adversarial Traffic Generation : Black-box Approach to Evade Intrusion Detection Systems in IoT Networks",
      "authors": [
        "Islam Debicha",
        "Tayeb Kenaza",
        "Ishak Charfi",
        "Salah Mosbah",
        "Mehdi Sehaki",
        "Jean-Michel Dricot"
      ],
      "abstract": "The integration of machine learning (ML) algorithms into Internet of Things (IoT) applications has introduced significant advantages alongside vulnerabilities to adversarial attacks, especially within IoT-based intrusion detection systems (IDS). While theoretical adversarial attacks have been extensively studied, practical implementation constraints have often been overlooked. This research addresses this gap by evaluating the feasibility of evasion attacks on IoT network-based IDSs, employing a novel black-box adversarial attack. Our study aims to bridge theoretical vulnerabilities with real-world applicability, enhancing understanding and defense against sophisticated threats in modern IoT ecosystems. Additionally, we propose a defense scheme tailored to mitigate the impact of evasion attacks, thereby reinforcing the resilience of ML-based IDSs. Our findings demonstrate successful evasion attacks against IDSs, underscoring their susceptibility to advanced techniques. In contrast, we proposed a defense mechanism that exhibits robust performance by effectively detecting the majority of adversarial traffic, showcasing promising outcomes compared to current state-of-the-art defenses. By addressing these critical cybersecurity challenges, our research contributes to advancing IoT security and provides insights for developing more resilient IDS.",
      "published": "2026-03-24T17:11:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.23438v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23438v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling",
      "authors": [
        "Yiqi Zhang",
        "Huiqiang Jiang",
        "Xufang Luo",
        "Zhihe Yang",
        "Chengruidong Zhang",
        "Yifei Shen",
        "Dongsheng Li",
        "Yuqing Yang",
        "Lili Qiu",
        "Yang You"
      ],
      "abstract": "Scaling reinforcement learning (RL) has shown strong promise for enhancing the reasoning abilities of large language models (LLMs), particularly in tasks requiring long chain-of-thought generation. However, RL training efficiency is often bottlenecked by the rollout phase, which can account for up to 70% of total training time when generating long trajectories (e.g., 16k tokens), due to slow autoregressive generation and synchronization overhead between rollout and policy updates. We propose SortedRL, an online length-aware scheduling strategy designed to address this bottleneck by improving rollout efficiency and maintaining training stability. SortedRL reorders rollout samples based on output lengths, prioritizing short samples forming groups for early updates. This enables large rollout batches, flexible update batches, and near on-policy micro-curriculum construction simultaneously. To further accelerate the pipeline, SortedRL incorporates a mechanism to control the degree of off-policy training through a cache-based mechanism, and is supported by a dedicated RL infrastructure that manages rollout and update via a stateful controller and rollout buffer. Experiments using LLaMA-3.1-8B and Qwen-2.5-32B on diverse tasks, including logical puzzles, and math challenges like AIME 24, Math 500, and Minerval, show that SortedRL reduces RL training bubble ratios by over 50%, while attaining 3.9% to 18.4% superior performance over baseline given same amount of data.",
      "published": "2026-03-24T16:48:31Z",
      "abstract_url": "http://arxiv.org/abs/2603.23414v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23414v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies",
      "authors": [
        "Hanzhong Zhang",
        "Siyang Song",
        "Jindong Wang"
      ],
      "abstract": "While large language models simulate social behaviors, their capacity for stable stance formation and identity negotiation during complex interventions remains unclear. To overcome the limitations of static evaluations, this paper proposes a novel mixed-methods framework combining computational virtual ethnography with quantitative socio-cognitive profiling. By embedding human researchers into generative multiagent communities, controlled discursive interventions are conducted to trace the evolution of collective cognition. To rigorously measure how agents internalize and react to these specific interventions, this paper formalizes three new metrics: Innate Value Bias (IVB), Persuasion Sensitivity, and Trust-Action Decoupling (TAD). Across multiple representative models, agents exhibit endogenous stances that override preset identities, consistently demonstrating an innate progressive bias (IVB > 0). When aligned with these stances, rational persuasion successfully shifts 90% of neutral agents while maintaining high trust. In contrast, conflicting emotional provocations induce a paradoxical 40.0% TAD rate in advanced models, which hypocritically alter stances despite reporting low trust. Smaller models contrastingly maintain a 0% TAD rate, strictly requiring trust for behavioral shifts. Furthermore, guided by shared stances, agents use language interactions to actively dismantle assigned power hierarchies and reconstruct self organized community boundaries. These findings expose the fragility of static prompt engineering, providing a methodological and quantitative foundation for dynamic alignment in human-agent hybrid societies. The official code is available at: https://github.com/armihia/CMASE-Endogenous-Stances",
      "published": "2026-03-24T16:38:46Z",
      "abstract_url": "http://arxiv.org/abs/2603.23406v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23406v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.HC"
      ]
    },
    {
      "title": "Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation",
      "authors": [
        "Michal Balcerak",
        "Suprosana Shit",
        "Chinmay Prabhakar",
        "Sebastian Kaltenbach",
        "Michael S. Albergo",
        "Yilun Du",
        "Bjoern Menze"
      ],
      "abstract": "Energy-based models for discrete domains, such as graphs, explicitly capture relative likelihoods, naturally enabling composable probabilistic inference tasks like conditional generation or enforcing constraints at test-time. However, discrete energy-based models typically struggle with efficient and high-quality sampling, as off-support regions often contain spurious local minima, trapping samplers and causing training instabilities. This has historically resulted in a fidelity gap relative to discrete diffusion models. We introduce Graph Energy Matching (GEM), a generative framework for graphs that closes this fidelity gap. Motivated by the transport map optimization perspective of the Jordan-Kinderlehrer-Otto (JKO) scheme, GEM learns a permutation-invariant potential energy that simultaneously provides transport-aligned guidance from noise toward data and refines samples within regions of high data likelihood. Further, we introduce a sampling protocol that leverages an energy-based switch to seamlessly bridge: (i) rapid, gradient-guided transport toward high-probability regions to (ii) a mixing regime for exploration of the learned graph distribution. On molecular graph benchmarks, GEM matches or exceeds strong discrete diffusion baselines. Beyond sample quality, explicit modeling of relative likelihood enables targeted exploration at inference time, facilitating compositional generation, property-constrained sampling, and geodesic interpolation between graphs.",
      "published": "2026-03-24T16:35:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.23398v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23398v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Contrastive Metric Learning for Point Cloud Segmentation in Highly Granular Detectors",
      "authors": [
        "Max Marriott-Clarke",
        "Lazar Novakovic",
        "Elizabeth Ratzer",
        "Robert J. Bainbridge",
        "Loukas Gouskos",
        "Benedikt Maier"
      ],
      "abstract": "We propose a novel clustering approach for point-cloud segmentation based on supervised contrastive metric learning (CML). Rather than predicting cluster assignments or object-centric variables, the method learns a latent representation in which points belonging to the same object are embedded nearby while unrelated points are separated. Clusters are then reconstructed using a density-based readout in the learned metric space, decoupling representation learning from cluster formation and enabling flexible inference. The approach is evaluated on simulated data from a highly granular calorimeter, where the task is to separate highly overlapping particle showers represented as sets of calorimeter hits. A direct comparison with object condensation (OC) is performed using identical graph neural network backbones and equal latent dimensionality, isolating the effect of the learning objective. The CML method produces a more stable and separable embedding geometry for both electromagnetic and hadronic particle showers, leading to improved local neighbourhood consistency, a more reliable separation of overlapping showers, and better generalization when extrapolating to unseen multiplicities and energies. This translates directly into higher reconstruction efficiency and purity, particularly in high-multiplicity regimes, as well as improved energy resolution. In mixed-particle environments, CML maintains strong performance, suggesting robust learning of the shower topology, while OC exhibits significant degradation. These results demonstrate that similarity-based representation learning combined with density-based aggregation is a promising alternative to object-centric approaches for point cloud segmentation in highly granular detectors.",
      "published": "2026-03-24T15:55:36Z",
      "abstract_url": "http://arxiv.org/abs/2603.23356v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23356v1",
      "categories": [
        "hep-ex",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Leveraging LLMs and Social Media to Understand User Perception of Smartphone-Based Earthquake Early Warnings",
      "authors": [
        "Hanjing Wang",
        "S. Mostafa Mousavi",
        "Patrick Robertson",
        "Richard M. Allen",
        "Alexie Barski",
        "Robert Bosch",
        "Nivetha Thiruverahan",
        "Youngmin Cho",
        "Tajinder Gadh",
        "Steve Malkos",
        "Boone Spooner",
        "Greg Wimpey",
        "Marc Stogaitis"
      ],
      "abstract": "Android's Earthquake Alert (AEA) system provided timely early warnings to millions during the Mw 6.2 Marmara Ereglisi, Türkiye earthquake on April 23, 2025. This event, the largest in the region in 25 years, served as a critical real-world test for smartphone-based Earthquake Early Warning (EEW) systems. The AEA system successfully delivered alerts to users with high precision, offering over a minute of warning before the strongest shaking reached urban areas. This study leveraged Large Language Models (LLMs) to analyze more than 500 public social media posts from the X platform, extracting 42 distinct attributes related to user experience and behavior. Statistical analyses revealed significant relationships, notably a strong correlation between user trust and alert timeliness. Our results indicate a distinction between engineering and the user-centric definition of system accuracy. We found that timeliness is accuracy in the user's mind. Overall, this study provides actionable insights for optimizing alert design, public education campaigns, and future behavioral research to improve the effectiveness of such systems in seismically active regions.",
      "published": "2026-03-24T15:24:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.23322v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23322v1",
      "categories": [
        "stat.AP",
        "cs.AI",
        "cs.CY",
        "physics.geo-ph"
      ]
    },
    {
      "title": "Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression",
      "authors": [
        "V. K. Cody Bumgardner",
        "Mitchell A. Klusty",
        "Mahmut S. Gokmen",
        "Evan W. Damron"
      ],
      "abstract": "Automated radiology report generation from 3D computed tomography (CT) volumes is challenging due to extreme sequence lengths, severe class imbalance, and the tendency of large language models (LLMs) to ignore visual tokens in favor of linguistic priors. We present Ker-VLJEPA-3B, a four-phase curriculum learning framework for free-text report generation from thoracic CT volumes. A phased training curriculum progressively adapts a Llama 3.2 3B decoder to ground its output in visual features from a frozen, self-supervised encoder. Our visual backbone (LeJEPA ViT-Large) is trained via self-supervised joint-embedding prediction on unlabeled CTs, without text supervision. Unlike contrastive models (CLIP, BiomedCLIP), this language-free backbone yields modality-pure representations. Vision-language alignment is deferred to the curriculum's bridge and generation phases. This modality-agnostic design can integrate any self-supervised encoder into an LLM without paired text during foundation training. Methodological innovations include: (1) zone-constrained cross-attention compressing slice embeddings into 32 spatially-grounded visual tokens; (2) PCA whitening of anisotropic LLM embeddings; (3) a positive-findings-only strategy eliminating posterior collapse; (4) warm bridge initialization transferring projection weights; and (5) selective cross-attention freezing with elastic weight consolidation to prevent catastrophic forgetting. Evaluated on the CT-RATE benchmark (2,984 validation volumes, 18 classes), Ker-VLJEPA-3B achieves a macro F1 of 0.429, surpassing the state-of-the-art (U-VLM, macro F1 = 0.414) by 3.6%, and reaching 0.448 (+8.2%) with threshold optimization. Ablation studies confirm 56.6% of generation quality derives from patient-specific visual content. Code and weights are available.",
      "published": "2026-03-24T15:13:30Z",
      "abstract_url": "http://arxiv.org/abs/2603.23308v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23308v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Designing Agentic AI-Based Screening for Portfolio Investment",
      "authors": [
        "Mehmet Caner",
        "Agostino Capponi",
        "Nathan Sun",
        "Jonathan Y. Tan"
      ],
      "abstract": "We introduce a new agentic artificial intelligence (AI) platform for portfolio management. Our architecture consists of three layers. First, two large language model (LLM) agents are assigned specialized tasks: one agent screens for firms with desirable fundamentals, while a sentiment analysis agent screens for firms with desirable news. Second, these agents deliberate to generate and agree upon buy and sell signals from a large portfolio, substantially narrowing the pool of candidate assets. Finally, we apply a high-dimensional precision matrix estimation procedure to determine optimal portfolio weights. A defining theoretical feature of our framework is that the number of assets in the portfolio is itself a random variable, realized through the screening process. We introduce the concept of sensible screening and establish that, under mild screening errors, the squared Sharpe ratio of the screened portfolio consistently estimates its target. Empirically, our method achieves superior Sharpe ratios relative to an unscreened baseline portfolio and to conventional screening approaches, evaluated on S&P 500 data over the period 2020--2024.",
      "published": "2026-03-24T15:03:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.23300v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23300v1",
      "categories": [
        "q-fin.PM",
        "cs.AI",
        "cs.MA",
        "q-fin.ST"
      ]
    },
    {
      "title": "A Comparative Study of Machine Learning Models for Hourly Forecasting of Air Temperature and Relative Humidity",
      "authors": [
        "Jiaqi Dong"
      ],
      "abstract": "Accurate short-term forecasting of air temperature and relative humidity is critical for urban management, especially in topographically complex cities such as Chongqing, China. This study compares seven machine learning models: eXtreme Gradient Boosting (XGBoost), Random Forest, Support Vector Regression (SVR), Multi-Layer Perceptron (MLP), Decision Tree, Long Short-Term Memory (LSTM) networks, and Convolutional Neural Network (CNN)-LSTM (CNN-LSTM), for hourly prediction using real-world open data. Based on a unified framework of data preprocessing, lag-feature construction, rolling statistical features, and time-series validation, the models are systematically evaluated in terms of predictive accuracy and robustness. The results show that XGBoost achieves the best overall performance, with a test mean absolute error (MAE) of 0.302 °C for air temperature and 1.271% for relative humidity, together with an average R2 of 0.989 across the two forecasting tasks. These findings demonstrate the strong effectiveness of tree-based ensemble learning for structured meteorological time-series forecasting and provide practical guidance for intelligent meteorological forecasting in mountainous cities.",
      "published": "2026-03-24T14:47:52Z",
      "abstract_url": "http://arxiv.org/abs/2603.23282v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23282v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Emergence of Fragility in LLM-based Social Networks: the Case of Moltbook",
      "authors": [
        "Luca Sodano",
        "Sofia Sciangula",
        "Amulya Galmarini",
        "Francesco Bertolotti"
      ],
      "abstract": "The rapid diffusion of large language models and the growth in their capability has enabled the emergence of online environments populated by autonomous AI agents that interact through natural language. These platforms provide a novel empirical setting for studying collective dynamics among artificial agents. In this paper we analyze the interaction network of Moltbook, a social platform composed entirely of LLM based agents, using tools from network science. The dataset comprises 39,924 users, 235,572 posts, and 1,540,238 comments collected through web scraping. We construct a directed weighted network in which nodes represent agents and edges represent commenting interactions. Our analysis reveals strongly heterogeneous connectivity patterns characterized by heavy tailed degree and activity distributions. At the mesoscale, the network exhibits a pronounced core periphery organization in which a very small structural core (0.9% of nodes) concentrates a large fraction of connectivity. Robustness experiments show that the network is relatively resilient to random node removal but highly vulnerable to targeted attacks on highly connected nodes, particularly those with high out degree. These findings indicate that the interaction structure of AI agent social systems may develop strong centralization and structural fragility, providing new insights into the collective organization of LLM native social environments.",
      "published": "2026-03-24T14:42:45Z",
      "abstract_url": "http://arxiv.org/abs/2603.23279v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23279v1",
      "categories": [
        "cs.SI",
        "cs.AI"
      ]
    },
    {
      "title": "A Multimodal Framework for Human-Multi-Agent Interaction",
      "authors": [
        "Shaid Hasan",
        "Breenice Lee",
        "Sujan Sarker",
        "Tariq Iqbal"
      ],
      "abstract": "Human-robot interaction is increasingly moving toward multi-robot, socially grounded environments. Existing systems struggle to integrate multimodal perception, embodied expression, and coordinated decision-making in a unified framework. This limits natural and scalable interaction in shared physical spaces. We address this gap by introducing a multimodal framework for human-multi-agent interaction in which each robot operates as an autonomous cognitive agent with integrated multimodal perception and Large Language Model (LLM)-driven planning grounded in embodiment. At the team level, a centralized coordination mechanism regulates turn-taking and agent participation to prevent overlapping speech and conflicting actions. Implemented on two humanoid robots, our framework enables coherent multi-agent interaction through interaction policies that combine speech, gesture, gaze, and locomotion. Representative interaction runs demonstrate coordinated multimodal reasoning across agents and grounded embodied responses. Future work will focus on larger-scale user studies and deeper exploration of socially grounded multi-agent interaction dynamics.",
      "published": "2026-03-24T14:35:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.23271v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23271v1",
      "categories": [
        "cs.RO",
        "cs.AI"
      ]
    },
    {
      "title": "Not All Tokens Are Created Equal: Query-Efficient Jailbreak Fuzzing for LLMs",
      "authors": [
        "Wenyu Chen",
        "Xiangtao Meng",
        "Chuanchao Zang",
        "Li Wang",
        "Xinyu Gao",
        "Jianing Wang",
        "Peng Zhan",
        "Zheng Li",
        "Shanqing Guo"
      ],
      "abstract": "Large Language Models(LLMs) are widely deployed, yet are vulnerable to jailbreak prompts that elicit policy-violating outputs. Although prior studies have uncovered these risks, they typically treat all tokens as equally important during prompt mutation, overlooking the varying contributions of individual tokens to triggering model refusals. Consequently, these attacks introduce substantial redundant searching under query-constrained scenarios, reducing attack efficiency and hindering comprehensive vulnerability assessment. In this work, we conduct a token-level analysis of refusal behavior and observe that token contributions are highly skewed rather than uniform. Moreover, we find strong cross-model consistency in refusal tendencies, enabling the use of a surrogate model to estimate token-level contributions to the target model's refusals. Motivated by these findings, we propose TriageFuzz, a token-aware jailbreak fuzzing framework that adapts the fuzz testing approach with a series of customized designs. TriageFuzz leverages a surrogate model to estimate the contribution of individual tokens to refusal behaviors, enabling the identification of sensitive regions within the prompt. Furthermore, it incorporates a refusal-guided evolutionary strategy that adaptively weights candidate prompts with a lightweight scorer to steer the evolution toward bypassing safety constraints. Extensive experiments on six open-source LLMs and three commercial APIs demonstrate that TriageFuzz achieves comparable attack success rates (ASR) with significantly reduced query costs. Notably, it attains a 90% ASR with over 70% fewer queries compared to baselines. Even under an extremely restrictive budget of 25 queries, TriageFuzz outperforms existing methods, improving ASR by 20-40%.",
      "published": "2026-03-24T14:33:36Z",
      "abstract_url": "http://arxiv.org/abs/2603.23269v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23269v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SafeSeek: Universal Attribution of Safety Circuits in Language Models",
      "authors": [
        "Miao Yu",
        "Siyuan Fu",
        "Moayad Aloqaily",
        "Zhenhong Zhou",
        "Safa Otoum",
        "Xing fan",
        "Kun Wang",
        "Yufei Guo",
        "Qingsong Wen"
      ],
      "abstract": "Mechanistic interpretability reveals that safety-critical behaviors (e.g., alignment, jailbreak, backdoor) in Large Language Models (LLMs) are grounded in specialized functional components. However, existing safety attribution methods struggle with generalization and reliability due to their reliance on heuristic, domain-specific metrics and search algorithms. To address this, we propose \\ourmethod, a unified safety interpretability framework that identifies functionally complete safety circuits in LLMs via optimization. Unlike methods focusing on isolated heads or neurons, \\ourmethod introduces differentiable binary masks to extract multi-granular circuits through gradient descent on safety datasets, while integrates Safety Circuit Tuning to utilize these sparse circuits for efficient safety fine-tuning. We validate \\ourmethod in two key scenarios in LLM safety: \\textbf{(1) backdoor attacks}, identifying a backdoor circuit with 0.42\\% sparsity, whose ablation eradicates the Attack Success Rate (ASR) from 100\\% $\\to$ 0.4\\% while retaining over 99\\% general utility; \\textbf{(2) safety alignment}, localizing an alignment circuit with 3.03\\% heads and 0.79\\% neurons, whose removal spikes ASR from 0.8\\% $\\to$ 96.9\\%, whereas excluding this circuit during helpfulness fine-tuning maintains 96.5\\% safety retention.",
      "published": "2026-03-24T14:32:53Z",
      "abstract_url": "http://arxiv.org/abs/2603.23268v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23268v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling",
      "authors": [
        "Ruisong Zhou",
        "Haijun Zou",
        "Li Zhou",
        "Chumin Sun",
        "Zaiwen Wen"
      ],
      "abstract": "Efficient scheduling of directed acyclic graphs (DAGs) in heterogeneous environments is challenging due to resource capacities and dependencies. In practice, the need for adaptability across environments with varying resource pools and task types, alongside rapid schedule generation, complicates these challenges. We propose WeCAN, an end-to-end reinforcement learning framework for heterogeneous DAG scheduling that addresses task--pool compatibility coefficients and generation-induced optimality gaps. It adopts a two-stage single-pass design: a single forward pass produces task--pool scores and global parameters, followed by a generation map that constructs schedules without repeated network calls. Its weighted cross-attention encoder models task--pool interactions gated by compatibility coefficients, and is size-agnostic to environment fluctuations. Moreover, widely used list-scheduling maps can incur generation-induced optimality gaps from restricted reachability. We introduce an order-space analysis that characterizes the reachable set of generation maps via feasible schedule orders, explains the mechanism behind generation-induced gaps, and yields sufficient conditions for gap elimination. Guided by these conditions, we design a skip-extended realization with an analytically parameterized decreasing skip rule, which enlarges the reachable order set while preserving single-pass efficiency. Experiments on computation graphs and real-world TPC-H DAGs demonstrate improved makespan over strong baselines, with inference time comparable to classical heuristics and faster than multi-round neural schedulers.",
      "published": "2026-03-24T14:16:08Z",
      "abstract_url": "http://arxiv.org/abs/2603.23249v1",
      "pdf_url": "https://arxiv.org/pdf/2603.23249v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.OC"
      ]
    }
  ]
};
