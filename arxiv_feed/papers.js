const PAPERS_DATA = {
  "last_updated": "2026-08-03 03:38:43 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "AgentHPOBench: A Benchmark For Evaluating LLM Agents as Sequential Hyperparameter Optimizers",
      "authors": [
        "Tianyu Huai",
        "Tingshuo Fan",
        "Xinchi Chen",
        "Yining Zheng",
        "Yuxin Wang",
        "Shuang Chen",
        "Jie Zhou",
        "Xuanjing Huang"
      ],
      "abstract": "As LLMs evolve from code completion systems into autonomous scientific agents, evaluating their ability to conduct experiments has become increasingly important. Existing benchmarks typically focus on static code generation, paper replication, or final answer correctness, but do not directly assess whether agents can interpret experimental evidence and use it to guide subsequent hyperparameter decisions. To address this gap, we introduce AgentHPOBench, a sequential benchmark comprising 30 executable machine learning tasks across seven research categories. Each task begins with a validated baseline run, after which an agent performs several sequential interventions. At each step, the agent observes the accumulated configurations, metrics, and logs before proposing the next valid configuration. We evaluate 12 widely used agents and conventional HPO baselines under a unified protocol. The results show that current agents exhibit measurable experimental optimization ability across domains, but still face clear limitations in sustained iterative refinement, complex log diagnosis, and consistent progress toward reported reference performance.",
      "published": "2026-07-31T16:58:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.29626v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29626v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CENDRe: Concept Extraction with Natural Domain Representations",
      "authors": [
        "Antonia Holzapfel",
        "Andres Felipe Posada Moreno",
        "Sebastian Trimpe"
      ],
      "abstract": "Convolutional neural networks (CNNs) are widely used for time-series classification, but their deployment in critical domains requires understanding the temporal and spectral patterns that drive their predictions. Concept extraction (CE) methods identify such patterns by analyzing representations within the models' latent space. However, existing time-series CE methods have three limitations: they operate only in the time domain and overlook frequency features, predefine the number of concepts, and produce localizations misaligned with the regions the model uses. We address these limitations by proposing CENDRe, a concept extraction method for CNNs. It first discovers concepts by clustering per-timestep latent representations in two stages, where silhouette-guided aggregation selects the number of concepts automatically. Then, it localizes each concept through gradients of a presence score that contrasts the latent representations with their prototypes, producing masks that concentrate on the regions driving the concept. These gradients, propagated through a differentiable invertible mapping of the input such as a Fourier transform, yield localizations for the same concepts in the frequency domain. Finally, each concept receives a relevance score that quantifies its contribution to each class. On synthetic benchmarks, CENDRe achieves representation correctness comparable to state-of-the-art CE methods and significantly higher importance correctness. On real bearing-fault data, CENDRe extracts the frequency bands driving the model's predictions, located in regions commonly inspected for fault diagnosis, producing evidence to assess the model that time-domain CE methods cannot.",
      "published": "2026-07-31T16:56:27Z",
      "abstract_url": "http://arxiv.org/abs/2607.29621v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29621v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning",
      "authors": [
        "Luca Viano",
        "Antoine Moulin",
        "Audrey Huang",
        "Volkan Cevher",
        "Philip Amortila",
        "Dylan J. Foster"
      ],
      "abstract": "Imitation learning (IL)---training an agent to replicate expert behavior from demonstrations---underpins applications from robotics to language model training. Standard approaches such as Behavior Cloning (BC) are known to suffer from compounding errors and performance plateaus, particularly when the learner cannot perfectly represent the expert's policy (as is typical, e.g., in distillation). Two interventions are widely understood empirically to improve performance: querying the expert interactively along the learner's own trajectories, and using value function estimation en route to generating a policy rather than directly fitting the expert's full action distribution. We investigate the nature of these improvements and their potentially surprising interplay. Our main finding is that expert interaction relaxes the representational demands on the learner: one only needs a model capable of realizing the expert's value function, bypassing the (often stricter) requirement of realizing the expert's policy itself. Concretely, we introduce OVI, an interactive on-policy IL algorithm that is statistically efficient whenever the learner can represent the expert's value function and computationally efficient given access to a linear maximization oracle. We complement this with a negative result showing that interaction is necessary. Namely, without stronger assumptions beyond expert-value realizability alone, any offline IL algorithm must scale with the complexity of the expert policy class. Our findings bear out empirically. OVI outperforms offline policy-based (BC), interactive policy-based (DAgger), and offline value-based IL methods, with the largest gains when the learner network is substantially less expressive than the expert's.",
      "published": "2026-07-31T16:52:47Z",
      "abstract_url": "http://arxiv.org/abs/2607.29617v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29617v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "A Human-Centered Validation of the Explainability-Performance Coefficient",
      "authors": [
        "Christian Oliva",
        "Luis F. Lago-Fernández"
      ],
      "abstract": "The rapid adoption of deep learning models in high-risk domains has intensified the need for trustworthy Explainable Artificial Intelligence (XAI). However, objectively evaluating explanation fidelity and aligning XAI metrics with human-centered understanding remain critical open challenges. In this work, we propose a model-agnostic metric, the EPC score, which is an extension of the Explainability-Performance Coefficient (EPC), that quantifies explanation quality by explicitly balancing the trade-off between feature selection sparsity and preserved model performance. Through an empirical validation across tabular, text, and image modalities, we show that the EPC score effectively uncovers operational dependencies among network activations, data dimensionality, and explainer performance. Furthermore, we validate the EPC score against independent human-based explanations, proving that higher EPC scores strongly align with human lexical sentiment judgments and spatial visual annotations.",
      "published": "2026-07-31T16:50:51Z",
      "abstract_url": "http://arxiv.org/abs/2607.29614v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29614v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "FriendBench: Benchmarking Dyadic Familiarity Inference in Humans and Multimodal Large Language Models",
      "authors": [
        "Jeffrey M. Girard",
        "Jason Z. Zheng",
        "Jacqueline R. Vertino",
        "Antony D'Avirro",
        "Benjamin Peloquin"
      ],
      "abstract": "Reading a social situation often depends on behavior, not words alone. We introduce FriendBench, a benchmark for inferring whether two people are already familiar or are meeting as strangers, from a 20-second clip of a dyadic ice-breaker conversation. Every pair answers the same type of prompt, so only the manner of interaction can reveal the answer. Across text, audio, and video, we compare 26 models from seven companies against matched human panels over 96 balanced dyads. The best model and the human crowd are statistically indistinguishable on accuracy in every modality, but reach it differently: humans stay balanced across the two answers, while the strongest models lean toward \"stranger\"---a difference in effective prior, not discrimination. Richer channels help both unequally, and only humans gain from visible behavior on top of speech. We release the stimuli, human ratings, and model predictions.",
      "published": "2026-07-31T16:33:39Z",
      "abstract_url": "http://arxiv.org/abs/2607.29602v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29602v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CV",
        "cs.HC"
      ]
    },
    {
      "title": "DungeonBench: A Benchmark for Rules-Rich Tactical Reasoning in Dungeons & Dragons Combat",
      "authors": [
        "Ismayil Ismayilov",
        "Atakan Kara",
        "Kaan Oktay"
      ],
      "abstract": "Games and simulators make valuable benchmarks by turning decisions into measurable outcomes, but many current suites under-test rules-rich tactical reasoning: the ability to choose well when geometry, timing, resources, objectives, and rule interactions all matter at once. We introduce DungeonBench, a benchmark for tactical reasoning in Dungeons & Dragons combat, built to cover the vast majority of combat-relevant 2014 System Reference Document content whose effects can be resolved by the simulator while retaining mechanics that simplified combat simulators often abstract away. At each step, DungeonBench exposes a complete tactical observation, a pending decision, and an indexed list of executable options spanning movement, attacks, spells, reactions, objectives, preparation, and scarce resources. The task is to value legal choices whose consequences depend on action economy, creature traits, battlefield geometry, timing windows, and future encounters. DungeonBench has two tracks: Encounter, which evaluates local tactical play in single fights, and Day, which links encounters through persistent hit points, spell slots, consumables, preparation, and short-rest timing, forcing policies to trade off immediate tactical advantage against future survivability. The same engine-generated decision stream supports heuristic controllers, language-model policies, learned option rankers, and masked-action reinforcement-learning agents. We evaluate frontier language-model policies on this shared decision stream. Results show that full tactical observations do not saturate the benchmark: frontier policies often win direct encounters, but linked encounter days expose failures in resource budgeting, rest timing, and rule-aware tactical discipline.",
      "published": "2026-07-31T16:03:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.29577v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29577v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "MOT-SR: Multi-Objective Tool-Augmented Scientific Equation Discovery with Large Language Models",
      "authors": [
        "Boxiao Wang",
        "Runxiang Wang",
        "Kai Li",
        "Chongming Li",
        "Zhiwei Chen",
        "Yifan Zhang",
        "Jian Cheng"
      ],
      "abstract": "Symbolic Regression (SR) aims to discover analytical equations from observational data and plays a central role in scientific modeling. While recent Large Language Model (LLM) based approaches show promise, they face two limitations. First, they lack data analysis mechanisms for uncovering variable dependencies, which reduces the efficiency of equation discovery. Second, most methods rely on single-objective evaluation focused solely on fitting error. This neglect of structural complexity and generalization often causes models to converge prematurely to local optima, limiting their ability to explore the broader equation space. We propose Multi-Objective Tool-augmented Symbolic Regression (MOT-SR), a unified framework that integrates external analytical tools to extract structural priors and guide equation generation, while jointly optimizing for accuracy, complexity, and generalization via a multi-objective evaluation module that maintains a dynamic Pareto front. MOT-SR employs two collaborative LLM modules: a Meta Strategy Generator, which selects tools and synthesizes structural optimization strategies based on Pareto-optimal equations, and an Equation Generator, which produces new candidate equations accordingly. The system operates in a closed-loop manner, continuously refining both strategies and equation structures. Across 40 standard tasks, MOT-SR outperforms existing SR methods in accuracy, generalization, and efficiency. We further validate MOT-SR on extreme mass-ratio inspiral (EMRI) orbital modeling, an important problem in space-based gravitational-wave astronomy where small local errors can accumulate substantially over long-term evolution. The discovered interpretable correction achieves the lowest trajectory-level integration error on held-out configurations. These results demonstrate the potential of MOT-SR to enable reliable modeling of long-horizon scientific dynamics.",
      "published": "2026-07-31T15:52:07Z",
      "abstract_url": "http://arxiv.org/abs/2607.29561v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29561v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "AMTFV: Agentic Mathematical Tool-Flow Verification for LLM Self-Correction",
      "authors": [
        "Rui Zou",
        "Yutao Zhu",
        "Mengqi Wei",
        "Ji-Rong Wen"
      ],
      "abstract": "Large language models have demonstrated strong mathematical problem-solving capabilities, yet reliably verifying their candidate answers remains challenging. Existing representative methods mainly revise outputs through natural-language reflection or assist verification by directly generating verification programs; the former may not reliably support exact computation, whereas the latter prematurely couples mathematical modeling with low-level implementation. We propose AMTFV (Agentic Mathematical Tool-Flow Verification). By introducing Mathematical Tool Flow (MTF) as an interrupt--execute--resume interface, AMTFV decouples verification modeling from concrete execution and supports exact computation through a mathematical toolbox. Specifically, the verification agent first constructs a verification workflow, encodes the mathematical objects and computational intent requiring reliable execution in an MTF request, and sends it to the mathematical toolbox agent. The latter parses the request, generates executable calls, and dispatches them to the backend for exact computation. Tool outputs then support candidate-answer adjudication, answer revision, and verification-workflow revision. We evaluate AMTFV on five challenging mathematical reasoning datasets with seven model configurations from DeepSeek, GPT, and Gemini. Experimental results show that AMTFV outperforms the representative baselines evaluated in this study overall; under an individual model configuration, it improves average accuracy over the strongest baseline by up to 8.3 percentage points, with larger gains on samples of medium and high verification complexity.",
      "published": "2026-07-31T15:42:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.29549v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29549v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation",
      "authors": [
        "Gaetano Perrone",
        "Simon Pietro Romano"
      ],
      "abstract": "Standard AI-text detection benchmarks compare human-written text against text generated directly by large language models (LLMs). While prior work has shown that rewriting and paraphrasing can degrade detector performance, it remains unclear whether performance measured on this conventional benchmark predicts detector behavior when human-authored content is rewritten by an LLM. To address this gap, we introduce Authorship-Rewriting Benchmark (ARB), built from 1,800 human source texts (600 each from XSum, WritingPrompts, and OpenWebText) and four open-weight generators (Llama-3.2-3B, Qwen2.5-7B, Mistral-7B, Gemma-2-9B). Each source item yields four matched variants: human-written (HUMAN), direct LLM generation (Free-LLM), LLM-rewritten human text (H2L), and same-generator LLM-rewritten LLM text (LLM2L). We evaluated five detectors (FastDetectGPT, Binoculars-falcon-7b, RADAR, BERT-Defense, RoBERTa-Defense) at a strict 1%-false-positive operating point (TPR@1%FPR). FastDetectGPT and Binoculars-falcon-7b detected 91.2% and 93.5\\% of direct LLM text, but only 30.8% and 15.1% of human text an LLM had rewritten, a drop of 60-78 percentage points. The same detectors retained 78.3% and 83.0% recall when LLM text was rewritten by the same model, a much smaller decline of 10-13 points. RADAR followed the same pattern (66.8% to 12.2%), while BERT-Defense and RoBERTa-Defense stayed below 3% recall across all regimes. These results show that detector performance measured on the conventional human-vs-LLM benchmark does not transfer to human-authored text revised by an LLM, even though the same detectors remain largely robust to LLM-only rewriting.",
      "published": "2026-07-31T15:35:57Z",
      "abstract_url": "http://arxiv.org/abs/2607.29539v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29539v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "TerraNova: A Foundation Model for the Anthropocene",
      "authors": [
        "Carlos Rodriguez-Pardo",
        "Massimo Tavoni"
      ],
      "abstract": "A defining problem of the Anthropocene is to model the physical Earth and human societies as one coupled system, yet no learned representation spans their observational breadth. We argue the obstacle is geometric: the physical Earth is measured as continuous fields that ignore political borders, whereas societies are reported for administrative units. Earth-system foundation models serve the first geometry; coupling it to the second has required lossy averaging over borders. We introduce TerraNova, a foundation model trained on 1,024 physical and societal records in their native geometries: 512 gridded Earth-system fields and 512 national indicators. Dedicated encoders represent location, country, time and task, cross-modal transformers fuse them into a shared spatiotemporal state, and a hypernetwork generates a per-query decoder whose evidential head returns a predictive distribution. Two contrastive objectives couple the representation: a population-weighted alignment between each country and coordinates in its territory, and one to pretrained geospatial embeddings carrying image-derived semantics. Read out through that decoder, the representation is competitive with purpose-built geospatial encoders while spanning axes they do not represent (time, oceans and uncertainty) and supporting country-level capabilities. The frozen backbone reconstructs dense fields from sparse observations and adapts to unseen variables in minutes on consumer hardware.",
      "published": "2026-07-31T15:27:26Z",
      "abstract_url": "http://arxiv.org/abs/2607.29527v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29527v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CY",
        "econ.EM",
        "stat.ML"
      ]
    },
    {
      "title": "DreamQAS: Learning a Decision-Useful World Model for VQE-Efficient Quantum Architecture Search",
      "authors": [
        "Jiayang Niu",
        "Yan Wang",
        "Jie Li",
        "Ke Deng",
        "Azadeh Alavi",
        "Muhammad Usman",
        "Yongli Ren"
      ],
      "abstract": "Reinforcement-learning-based quantum architecture search (RL-QAS) repeatedly optimizes a variational quantum eigensolver (VQE) after extending a circuit, although circuit construction and action legality are deterministic and known. We introduce DreamQAS, a model-based RL framework that preserves these exact circuit dynamics and learns only the expensive post-VQE feedback. A recurrent randomized-prior ensemble predicts an oracle-free score relative to an empirical energy frontier and supports multi-step imagined policy learning over explicit legal circuits. Ranking-based activation, uncertainty-aware pessimism and truncation, and selective real-VQE verification form a reliability-controlled learning loop. Under a common 15,000-episode budget and frozen evaluation for the RL methods, DreamQAS has the lowest mean frozen-policy energy error on four of five molecular tasks and the second-lowest on one. At fine-error targets reached by all seeds of both methods, it uses 1.6x to 2.0x fewer real VQE calls on four tasks and 10.6x fewer on BeH2-8q. Counterfactual action-ranking utility increases across all five tasks, with a mean increase of 0.346 and a 95 percent confidence interval of [0.185, 0.507], while direct greedy and beam use of the same model does not recover the gains of imagined policy learning. Ensemble disagreement also improves risk-coverage over random rejection on all three probed tasks. These results establish a world-model design for QAS whose value lies in decision-useful feedback rather than exact energy prediction.",
      "published": "2026-07-31T14:58:23Z",
      "abstract_url": "http://arxiv.org/abs/2607.29491v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29491v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TFGformer: Multivariate Time Series Forecasting via Time-Frequency Graph Learning and Covariate Fusion",
      "authors": [
        "Yu Sun",
        "Yuan Chang",
        "Xiaohou Shi",
        "Yan Sun"
      ],
      "abstract": "Large-scale multivariate time series from heterogeneous IoT sensors demand accurate long-term forecasting for resource scheduling and predictive maintenance. While recent time series foundation models exhibit strong generalization, they rely on static parametric knowledge and lack dynamic access to external historical patterns during inference. Retrieval-Augmented Generation (RAG) offers a potential remedy, yet its application to time series forecasting is challenged by magnitude variations across heterogeneous sources and the mismatch between historical similarity and future consistency. We propose CrossRAG, a retrieval-augmented forecasting framework that integrates Shape-Aware Memory (SAM) with RevIN normalization for magnitude-robust shape-level retrieval, Future-Consistent Contrastive (FCC) learning to distinguish informative references from hard negatives with similar history but divergent futures, and Cross-Attention Temporal Fusion (CATF) to fuse retrieved historical--future reference pairs into the backbone's representations at the representation level. Experiments on seven public benchmarks show that CrossRAG consistently outperforms both parametric-only baselines and existing retrieval-augmented forecasting methods.",
      "published": "2026-07-31T14:24:26Z",
      "abstract_url": "http://arxiv.org/abs/2607.29459v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29459v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ModelEquivBench: Certifying Multi-Relational Evaluation of LLM-Generated Optimization Models",
      "authors": [
        "Penglin Zhu",
        "Jungang Xu"
      ],
      "abstract": "Large language models increasingly generate optimization models from natural language, but existing evaluation often reduces a generated model and its ground truth to a single equivalent/not-equivalent verdict or an execution-success rate--labels that are neither independently checkable nor faithful to the multiple distinct senses in which two formulations can agree. We present ModelEquivBench, a certifying, multi-relational evaluation system that reports a per-pair semantic profile E0--E6: model construction and exact ingestion (E0), verified representation alignment (E1), same-space and projected feasible-set relations (E2, E3), objective-order equivalence (E4), optimal-value equality (E5), and optimizer-set equivalence (E6). Each decided entry carries relation-appropriate, independently re-checkable evidence: replayable traces or explicit maps for E0--E1, exact-rational certificates for positive E2--E6 conclusions, and explicit witnesses for supported negatives. Incomplete mapping search, unsupported structure, and resource limits produce typed UNKNOWN or N/A outcomes rather than guesses, while unmet prerequisites are reported as ABSENT. Using ModelEquivBench to evaluate three model snapshots--GPT-5.4, Claude Sonnet 4.6, and Qwen3.5-397B-A17B--on the same frozen cohort of 173 base problems (346 cells per model) under a no-repair protocol, the resulting profiles expose distinctions that coarse baselines do not represent: 49, 35, and 25 cells contain executable candidates that are nevertheless certified negative on at least one supported relation, and 25, 8, and 18 structural rejections occur on pairs for which E2 certifies mapped feasible-set equality under a verified map. The three model snapshots fail at different stages of the profile and therefore cannot be meaningfully reduced to a single accuracy score.",
      "published": "2026-07-31T13:55:09Z",
      "abstract_url": "http://arxiv.org/abs/2607.29431v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29431v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Explore Beyond the Boundary Using Entropic Information",
      "authors": [
        "Bumgeun Park",
        "Donghwan Lee"
      ],
      "abstract": "In reinforcement learning, exploration with sparse and delayed rewards presents a significant challenge due to the limited feedback available for guiding the learning process. Addressing this issue requires extensive exploration in the state space to discover valuable reward signals. In this paper, we propose Entropic Information for Exploration (ENTINEX), a novel method that enhances exploration by incentivizing agents to explore beyond the boundaries of the state distribution. ENTINEX achieves this by assigning intrinsic rewards to these boundaries, leveraging entropic information to identify them effectively. Through extensive experimentation, we demonstrate that ENTINEX consistently improves exploration performance in environments characterized by sparse and delayed rewards. Our experimental results show that ENTINEX outperforms existing exploration methods, highlighting its effectiveness in both sparse and delayed reward scenarios.",
      "published": "2026-07-31T13:41:51Z",
      "abstract_url": "http://arxiv.org/abs/2607.29419v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29419v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Stable Autoregressive Speech Generation with Low-Frame-Rate High-Dimensional Continuous Tokens",
      "authors": [
        "Yi Luo",
        "Rongzhi Gu",
        "Jixun Yao"
      ],
      "abstract": "Balancing sequence length, representational capacity, and long-horizon stability is a central problem in autoregressive (AR) speech and audio generation. Representations with higher frame rates or greater capacity can preserve more signal detail, but they also make streaming generation more vulnerable to distribution drift and AR error accumulation. Conversely, shorter and more compressed representations simplify AR modeling, but their limited bandwidth may discard important components and constrain the upper bound of reconstruction fidelity and generation quality. We ask whether a low-frame-rate, high-dimensional, high-bandwidth continuous representation can be co-designed with a streaming generation framework to support robust high-fidelity reconstruction, strong single-token predictability, and superior long-horizon stability. We decompose this goal into two coupled problems: what geometric and statistical properties a high-dimensional representation space should have, and how an AR continuous-token generator should be structured to resist error accumulation. Accordingly, we propose Locodec, a locally encoded codec that shapes its representation space to improve the interpolatability of a lower-dimensional core manifold and the identifiability of the native high-dimensional coordinates, thereby improving the predictability of high-dimensional high-bandwidth tokens. We also propose MP-ELD, a single-token AR flow-matching framework that uses multi-path information routing and residual classifier-free guidance to mitigate error accumulation. Experiments with 8-Hz, 768-dimensional tokens show that our design preserves reconstruction quality, improves single-token predictability, achieves competitive WER, and maintains stable long-form synthesis, without using external SSL/ASR models, pretrained text language models, or post-training stages.",
      "published": "2026-07-31T12:51:24Z",
      "abstract_url": "http://arxiv.org/abs/2607.29363v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29363v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.LG",
        "cs.SD"
      ]
    },
    {
      "title": "Versatile On-device Adaptation at the Edge by Unifying Few-shot, Zero-shot, Continual, and In-context Learning",
      "authors": [
        "Douwe den Blanken",
        "Martin Lefebvre",
        "Charlotte Frenkel"
      ],
      "abstract": "With the ever-increasing pervasiveness of smart edge devices, the demand is growing for applications that can be tailored to users (e.g., custom keyword spotting) or patients (e.g., adaptive health monitoring). Yet, most edge devices rely on fixed inference algorithms and thus cannot learn on-device to personalize predictions. When they can, devices typically support only a specific learning scenario, such as few-shot learning (FSL): going beyond this requires resorting either to another specialized device or to cloud-based retraining, which implies significant energy and latency overheads, a lack of real-time capabilities, and privacy concerns. In this work, we introduce embedder-centric learning (ECL), a framework that unifies four different online learning scenarios: FSL for on-the-fly customization, continual learning (CL) for knowledge accumulation, zero-shot learning (ZSL) for leveraging semantic data, and in-context learning (ICL) for adapting beyond classification. We demonstrate in silicon that ECL can be deployed on resource-constrained devices across four real-world use cases representative of the aforementioned learning scenarios. Our approach establishes a new state-of-the-art performance for FSL character recognition (Omniglot: 96.8% for 5-way 1-shot, 83.3% for 32-way 1-shot), and the first hardware baseline for CL in keyword spotting (NeuroBench keyword FSCIL: 71.8% for 200-way 5-shot). Moreover, we present the first hardware demonstrations of ZSL with semantic data (60.6% for 5-way spoken sentence classification) and ICL (46.2% at the 500th token of RegBench) operating at micro-to-milliwatt power budgets. Therefore, by unifying multiple learning scenarios, we pave the way for smart and versatile devices that can adapt right at the edge, without reliance on the cloud.",
      "published": "2026-07-31T12:37:48Z",
      "abstract_url": "http://arxiv.org/abs/2607.29353v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29353v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.AS"
      ]
    },
    {
      "title": "The persuasive power of large language models does not depend on their perceived national origin",
      "authors": [
        "Ningzhi Liu",
        "Yannic Hinrichs",
        "Jonas R. Kunst"
      ],
      "abstract": "Conversational AI developed by geopolitical rivals reaches citizens worldwide, raising concerns that it could sway public opinion or be rejected as foreign propaganda, with consequences for democratic discourse and information sovereignty. Yet, whether an AI's perceived national origin shapes its persuasive power is unknown. In a preregistered randomized experiment, 403 adults from a nationally representative United States sample held a three-round debate with a chatbot introduced as either American (\"DiscoveryAI\") or Chinese (\"ZhengheAI\"), discussing a political or non-political topic. In all conditions, participants actually conversed with the same model (GPT-4o), instructed to argue against their initial position. We combined pre- and post-conversation self-reports of attitudes, trust, and collective narcissism with computational analyses of 1,209 participant turns, including LLM-coded stance and argumentative conduct, stance-sensitive embeddings, and keyword-masked emotion and toxicity classifiers. The conversations produced substantial attitude changes in every condition. Critically, the nationality label affected neither self-reported attitude change nor expressed stance, concessions, counterarguing, or affect, and equivalence tests and Bayes factors largely supported these null effects. The label's only reliable footprint was lower pre-conversation human-like trust in the Chinese model, whereas functionality trust was unaffected. Political topics slowed stance movement toward the AI's position, and collective narcissism predicted less attitude change regardless of origin, acting as a general barrier rather than an out-group filter. Users thus initially withhold social trust from a rival's AI yet still assimilate its arguments; origin labeling and transparency requirements alone may offer weak protection against foreign influence operations conducted through conversational AI.",
      "published": "2026-07-31T12:08:25Z",
      "abstract_url": "http://arxiv.org/abs/2607.29334v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29334v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "MAGA: Multi-Platform Self-Fusion of GUI Agents via Structured Action Distillation",
      "authors": [
        "Hang Yan",
        "Zhangxuan GU",
        "Beitong Zhou",
        "Jiaxuan Chen",
        "Runze Li",
        "Yusong Hu",
        "Shuheng Shen",
        "Changhua Meng"
      ],
      "abstract": "Graphical user interface (GUI) agents based on large language models are increasingly deployed across mobile, web, and desktop environments. However, existing agents are typically domain-specific, limiting the deployment and user experience. This motivates the consolidation of specialized models into a single cross-environment policy. Weight merging directly merges domain-specific experts but can corrupt executable actions under expert disagreement, while on-policy distillation (OPD) avoids conflicting teacher supervision yet still treats all response tokens equally during distillation, ignoring that action tokens are the only interface between the environment and the agent. To address this, We introduce MAGA that re-allocates training signal according to the structured action. Based on the correctness of the generated action, it suppresses unnecessary or invalid distillation signals and focuses learning on erroneous actions. Besides, a training-only hint optimizes the supervision signal provided by domain-specific teachers without changing the student input. Across two model scales, MAGA achieves the highest mean success rate, outperforming the strongest baseline by 2.0% at 8B and achieves almost the same average performance with teachers.",
      "published": "2026-07-31T11:51:04Z",
      "abstract_url": "http://arxiv.org/abs/2607.29320v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29320v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Tool Specifications Matter: Uncovering and Mitigating Safety Risks in AI Agents",
      "authors": [
        "Minghui Pan",
        "Jiayuxuan Yang",
        "Yuanyuan Yuan",
        "Yu Jiang",
        "Zhenpeng Chen"
      ],
      "abstract": "AI agents extend large language models (LLMs) with external tools, enabling them to perform complex tasks and translate model outputs into consequential real-world actions. Yet LLMs often become substantially less safe when deployed as agents, and the source of this degradation remains poorly understood. In this paper, we identify schema-formatted tool specifications as a primary source of agent safety degradation and show, through white-box representation analysis, that they weaken the model's internal refusal signals and contribute to unsafe tool execution. Building on this finding, we propose SafeKeep, an inference-time safeguard that decouples safety judgment from tool execution: it assesses requests using flattened textual tool specifications while retaining the original schema-formatted specifications for execution. Across two representative benchmarks and four LLMs, including both white-box and black-box models, SafeKeep increases the average refusal rate for harmful requests from 23.8% to 70.6% and reduces the average attack success rate under observation-level prompt injection from 25.6% to 2.5%. It also outperforms existing safeguards and preserves task-handling capability. We release the code and data at https://github.com/snowcatsmoking/SafeKeep .",
      "published": "2026-07-31T10:25:04Z",
      "abstract_url": "http://arxiv.org/abs/2607.29254v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29254v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "CalibratedRubric: Task-Adaptive Rubric Banks for Open-Ended LLM Evaluation",
      "authors": [
        "Mengting Chen",
        "Yanshu Sun",
        "Wanting Liang",
        "Beidi Luan",
        "Rui Sun",
        "Dezhi Chen",
        "Jing Li",
        "Zuo Bai"
      ],
      "abstract": "Reliable evaluation of open-ended LLM outputs requires fine-grained rubrics, yet expert curation is costly and difficult to scale. Existing automated pipelines rely on strict judge unanimity and binary variance filters, which cannot distinguish measurable rubrics from informative ones. We introduce CalibratedRubric, a task-adaptive framework that combines type-specific scoring, Bayesian rubric-measurability filtering, and item response theory (IRT)-based bank assembly. CalibratedRubric estimates each rubric's measurability with a Beta--Bernoulli agreement posterior and uses a submodular information-coverage objective to construct compact rubric banks over the observed capability range. Across financial, healthcare, general, and legal benchmarks, measurability filtering improves human-gold agreement on JudgmentBench from $κ=0.604$ to $0.743$. IRT-based greedy selection improves cross-fitted rank fidelity over random selection across all six evaluated response blocks and requires only 49 rather than 131 rubrics to reach the target correlation on FinResearchBench decision-support tasks. Task-label perturbations further reduce system separation, confirming the practical relevance of task-adaptive scoring. These results support CalibratedRubric as an efficient, uncertainty-aware approach to open-ended LLM evaluation, with calibration gains depending on sufficient judge redundancy.",
      "published": "2026-07-31T10:21:56Z",
      "abstract_url": "http://arxiv.org/abs/2607.29252v1",
      "pdf_url": "https://arxiv.org/pdf/2607.29252v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
