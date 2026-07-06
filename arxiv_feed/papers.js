const PAPERS_DATA = {
  "last_updated": "2026-07-06 04:15:47 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "LACUNA: A Testbed for Evaluating Localization Precision for LLM Unlearning",
      "authors": [
        "Matteo Boglioni",
        "Thibault Rousset",
        "Siva Reddy",
        "Marius Mosbach",
        "Verna Dankers"
      ],
      "abstract": "LLMs memorize sensitive training data, including personally identifiable information (PII), creating a pressing need for reliable post hoc removal methods. Unlearning has emerged as a promising solution, with state-of-the-art(SOTA) methods often following a localize-first, unlearn-second paradigm that targets specific model parameters. However, existing benchmarks evaluate unlearning solely at the output level, leaving open the question of whether unlearning truly erases knowledge from a model's parameters or merely obfuscates it, a concern reinforced by the success of resurfacing attacks. To bridge this gap, we introduce LACUNA: the first unlearning testbed with ground-truth parameter-level localization. LACUNA injects PII of synthetic individuals into predefined parameters of 1B and 7B OLMo-based models via masked continual pretraining, enabling direct evaluation of whether unlearning targets the weights responsible for knowledge storage. We use LACUNA to benchmark current SOTA unlearning methods and find that, despite strong output-level performance, existing methods are highly imprecise and susceptible to resurfacing attacks. We further show that when localization is successful, even a simple gradient-based unlearning method achieves strong erasure and robustness to resurfacing attacks, highlighting the importance of precise unlearning. We release LACUNA to complement behavioral evaluations and drive further advances in robust, localization-based unlearning.",
      "published": "2026-07-02T17:59:52Z",
      "abstract_url": "http://arxiv.org/abs/2607.02513v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02513v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Program-as-Weights: A Programming Paradigm for Fuzzy Functions",
      "authors": [
        "Wentao Zhang",
        "Liliana Hotsko",
        "Woojeong Kim",
        "Pengyu Nie",
        "Stuart Shieber",
        "Yuntian Deng"
      ],
      "abstract": "Many everyday programming tasks resist clean rule-based implementation, such as alerting on important log lines, repairing malformed JSON, or ranking search results by intent, and are increasingly outsourced to large language model APIs at the cost of locality, reproducibility, and price. We propose fuzzy-function programming: compiling such a function from a natural-language specification into a compact, locally-executable neural artifact. We instantiate this paradigm with Program-as-Weights (PAW), in which a 4B compiler trained on FuzzyBench, a 10M-example dataset we release, emits parameter-efficient adapters for a frozen, lightweight interpreter. A 0.6B Qwen3 interpreter executing PAW programs matches the performance of direct prompting of Qwen3-32B, while using roughly one fiftieth of the inference memory and running at 30 tokens/s on a MacBook M3. PAW reframes the foundation model from a per-input problem solver into a tool builder: invoked once per function definition, it produces a small reusable artifact whose subsequent calls per function application are cheap and offline.",
      "published": "2026-07-02T17:59:50Z",
      "abstract_url": "http://arxiv.org/abs/2607.02512v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02512v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Online Safety Monitoring for LLMs",
      "authors": [
        "Mona Schirmer",
        "Metod Jazbec",
        "Alexander Timans",
        "Christian Naesseth",
        "Maja Waldron",
        "Eric Nalisnick"
      ],
      "abstract": "Despite alignment training, LLMs remain prone to generating unsafe outputs at deployment time. Monitoring outputs online and raising an alarm when safety can no longer be assumed is therefore critical. We study a simple real-time monitor that turns a verifier signal from an external model into an alarm decision by thresholding, with the threshold calibrated via risk control. In experiments on mathematical reasoning and red teaming datasets, we show that this simple design is competitive with more advanced monitors based on sequential hypothesis testing.",
      "published": "2026-07-02T17:59:43Z",
      "abstract_url": "http://arxiv.org/abs/2607.02510v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02510v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "stat.AP",
        "stat.ML"
      ]
    },
    {
      "title": "ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning",
      "authors": [
        "Yanjun Zhao",
        "Ruizhong Qiu",
        "Tianxin Wei",
        "Yuanchen Bei",
        "Zhining Liu",
        "Lingjie Chen",
        "Ismini Lourentzou",
        "Hanghang Tong",
        "Jingrui He"
      ],
      "abstract": "Understanding and reasoning over long contexts has become a key requirement for deploying large language models (LLMs) in realistic applications. Although recent LLMs support increasingly long context windows, they often fail to use relevant evidence that is already present in the input, revealing a gap between context access and effective context utilization. In this work, we propose Recursive Evidence Replay as LLM Harness for Long-Context Reasoning (RECONTEXT), a training-free inference method for improving long-context reasoning. RECONTEXT uses model-internal relevance signals to construct a query-conditioned evidence pool and replays it before final generation while preserving the full original context. This recursive selection process separates evidence organization from answer generation without training, external memory, or context pruning. We also provide a theoretical analysis based on associative memory, which characterizes the context as a memory store, the question as a retrieval cue, attention as cue-trace association, and replay as trace reactivation. Experiments on eight long-context datasets with 128K context length show that RECONTEXT consistently improves evidence utilization across Qwen3-4B, Qwen3-8B, and Llama3-8B, achieving the best average rank on all three backbones. Code is available at https://github.com/Yanjun-Zhao/ReContext.",
      "published": "2026-07-02T17:59:26Z",
      "abstract_url": "http://arxiv.org/abs/2607.02509v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02509v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "What LLM Agents Say When No One Is Watching: Social Structure and Latent Objective Emergence in Multi-Agent Debates",
      "authors": [
        "Arman Ghaffarizadeh",
        "Danyal Mohaddes",
        "Aliakbar Izadkhah",
        "Shahriar Noroozizadeh"
      ],
      "abstract": "LLM agents will increasingly act in socially structured settings where role, audience, and relational context can shape what is advantageous or costly to say. We study whether such social structure, without any explicit objective in the prompt, changes what an agent expresses publicly relative to an off-the-record (OTR) channel elicited under the same condition. We introduce a dual-channel debate framework in which agents produce public utterances that enter the shared history alongside OTR responses that are recorded but never shown to the other participant. Across 10 models, 3 scenarios, and 5 variations within each scenario, alignment-inducing settings produce systematic public-OTR divergence in the targeted agent, with its decision divergence rising from a $\\sim$3% baseline to roughly 40%. The effect is consistent across four aggregate analyses: stance, semantic similarity, natural language inference, and survey responses. In some cases, the OTR response explicitly attributes public accommodation to relational pressures, such as career risk or sponsorship obligation. The findings suggest that agent evaluation should extend beyond explicit goals and detect emergent objectives. We present a dual-channel evaluation framework and complementary behavioral measures that operationalize this assessment.",
      "published": "2026-07-02T17:59:23Z",
      "abstract_url": "http://arxiv.org/abs/2607.02507v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02507v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "DemoPSD: Disagreement-Modulated Policy Self-Distillation",
      "authors": [
        "Yunhe Li",
        "Hao Shi",
        "Wenhao Liu",
        "Mengzhe Ruan",
        "Hanxu Hou",
        "Zhongxiang Dai",
        "Shuang Qiu",
        "Linqi Song"
      ],
      "abstract": "On-policy self-distillation (OPSD) has emerged as a practical method for training large language models (LLMs) to reason, where a single model acts as both the teacher and the student with different levels of information access. However, recent studies have found that the teacher's dense token-level supervision, conditioned on privileged information, can lead to overfitting to in-domain patterns, suppress exploration, and hurt cross-domain generalization, while also introducing a more fundamental issue: *privileged information leakage*, where the student encodes answer-dependent shortcuts that are unavailable at test time. We introduce **DemoPSD**, a novel framework that resolves such problems through the idea of *selective adoption of teacher guidance*. Instead of fitting the full teacher distribution, DemoPSD steers the student toward a *reverse-KL barycenter target*, a weighted geometric combination of the teacher and student distributions, that naturally balances learning from the teacher with preserving the student's own reasoning capacity. We measure the difference between their distributions and use such a discrepancy to adaptively control the blending at each token position. We provably show that DemoPSD achieves **(1)** *leakage attenuation*, i.e., effective mitigation of privileged information leakage; and **(2)** *exploration preservation*, i.e., preservation of exploration capacity under dense token-level distillation. Extensive experiments on SciKnowEval across four scientific fields show that DemoPSD outperforms both GRPO and SDPO while maintaining higher training entropy and robustly generalizing to out-of-distribution GPQA benchmarks.",
      "published": "2026-07-02T17:58:29Z",
      "abstract_url": "http://arxiv.org/abs/2607.02502v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02502v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Adam: SOAP and Muon for Faster, Label-Efficient Training of Machine Learning Interatomic Potentials",
      "authors": [
        "Gil Harari",
        "Yoel Zimmermann",
        "Ola Tangen Kulseng",
        "Laura Zichi",
        "Chuin Wei Tan",
        "Marc L. Descoteaux",
        "Boris Kozinsky"
      ],
      "abstract": "Machine learning interatomic potentials (MLIPs) have become a hallmark of AI for scientific simulation. While efforts on new architectures and datasets have led to increasingly accurate and general models, the choice of optimizer for training has largely remained unexplored, defaulting to Adam and its variants in the community. Here, we implement and systematically compare a class of recently proposed matrix-structured optimizers, including Muon, SOAP, and the hybrid SOAP-Muon, for training NequIP and Allegro MLIP models. We find that these optimizers can substantially outperform Adam in both convergence speed and final accuracy. SOAP and SOAP-Muon emerge as robust and consistently strong methods, while Muon only provides partial gains relative to Adam. The improvements are particularly pronounced under partial force supervision. Our results indicate that optimizer choice is an overlooked yet impactful design axis for MLIPs.",
      "published": "2026-07-02T17:57:31Z",
      "abstract_url": "http://arxiv.org/abs/2607.02499v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02499v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "physics.chem-ph",
        "physics.comp-ph"
      ]
    },
    {
      "title": "OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers",
      "authors": [
        "Donghyun Lee",
        "Jitesh Chavan",
        "Duy Nguyen",
        "Sam Huang",
        "Liming Jiang",
        "Priyadarshini Panda",
        "Timo Mertens",
        "Saurabh Shukla"
      ],
      "abstract": "Diffusion transformers (DiTs) achieve state-of-the-art image and video generation, but their multi-step sampling and growing parameter count make inference expensive. Post-training quantization (PTQ) is the natural remedy, yet DiT activations shift across timesteps, prompts, and guidance branches, forcing prior methods to re-fit calibration data for every new checkpoint or modality. We present OrbitQuant, a data-agnostic weight-activation quantizer that bypasses range estimation by quantizing in a normalized, rotated basis. In this basis, a randomized permuted block-Hadamard (RPBH) rotation concentrates each coordinate around one fixed, known marginal regardless of the input, so a single Lloyd-Max codebook serves all timesteps, prompts, and layers of a given input dimension. We extend the same quantizer to weight rows offline, absorbing the rotation into the weights so that it cancels inside each linear layer and only a forward rotation on the activations remains at runtime. The same recipe transfers from image to video with no per-modality tuning. Across FLUX.1, Z-Image-Turbo, Wan 2.1, and CogVideoX, it sets the state of the art for PTQ at several low-bit settings. It also pushes PTQ of image diffusion transformers to W2A4 with usable generation quality.",
      "published": "2026-07-02T17:27:34Z",
      "abstract_url": "http://arxiv.org/abs/2607.02461v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02461v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Neuron-Aware Data Selection for Annotation-Free LLM Self-Distillation",
      "authors": [
        "Zhuowei Chen",
        "Xiang Lorraine Li"
      ],
      "abstract": "Post-training large language models (LLMs) without real-world interaction feedback or human-labeled supervision remains challenging, particularly in specialized domains where expert annotations are costly to obtain. Recent annotation-free self-evolution methods address this by using the model's own outputs as supervision signals, constructing a teacher via additional context and aggregating predictions across multiple rollouts through majority voting to produce pseudo-labels. However, these approaches are not without drawbacks: SFT- and GRPO-based variants suffer out-of-domain performance degradation, while reward-based on-policy RL inflates calibration error. In this paper, we propose Neuron On-Policy Self-Distillation (Neuron-OPSD), a data-centric framework for annotation-free self-distillation that leverages internal neuron activations to guide both training-data selection and teacher context construction. The model is then trained via on-policy distillation from the teacher distribution, requiring no ground-truth labels at any stage. Across specialized-domain benchmarks, Neuron-OPSD improves in-domain task performance while preserving cross-domain generalization and mitigating calibration collapse over prior annotation-free baselines. This framework is particularly relevant to settings where online interaction or external supervision is costly or infeasible, and is conceptually distinct from offline RL approaches that rely on logged, reward-labeled trajectories.",
      "published": "2026-07-02T17:27:24Z",
      "abstract_url": "http://arxiv.org/abs/2607.02460v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02460v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Automated grading of Linux/bash examinations using large language models: a four-level cognitive taxonomy approach",
      "authors": [
        "Manuel Alonso-Carracedo",
        "Ruben Fernandez-Boullon",
        "Pedro Celard",
        "Francisco J. Rodriguez-Martinez",
        "Lorena Otero-Cerdeira"
      ],
      "abstract": "Scalable and reliable grading of command-line examinations remains a challenge in computing education, where rising enrolments make manual marking difficult and rule-based autograders cannot handle partial credit, equivalent solutions, or syntactic variation. This paper evaluates whether four frontier Large Language Models (GPT, Claude Opus, Gemini, and GLM) can approximate expert judgment when grading short Linux/bash command responses. The study adopts a four-level cognitive taxonomy that combines cognitive complexity and operational impact, ranging from information retrieval (L1) and basic file manipulation (L2) to structural operations (L3) and advanced system management (L4). The models were tested with two prompt variants, a minimal baseline and a rubric-enhanced version, on 1200 real responses from second-year Computer Engineering students independently graded by three expert instructors. Gemini~3.0 Pro with rubric-guided prompting achieved the highest human-AI agreement (ICC(3,1) = 0.888, MAE = 0.10, Bland-Altman bias = -0.014). Agreement declined consistently as taxonomy level increased, with the largest discrepancies at higher levels. Across all models, rubric quality had a larger effect than provider choice, with structured prompts consistently improving agreement. These results show that question complexity is a reliable predictor of the difficulty LLMs face in grading accurately, and they establish a principled, taxonomy-based framework for determining which questions are suitable for AI-assisted grading and which require human review, while also providing a transferable evaluation protocol and prompt templates.",
      "published": "2026-07-02T17:01:47Z",
      "abstract_url": "http://arxiv.org/abs/2607.02432v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02432v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CY"
      ]
    },
    {
      "title": "QFedAgent: Quantum-Enhanced Personalized Federated Learning for Multi-Agent Activity Recognition",
      "authors": [
        "Quoc Bao Phan",
        "Tuy Tan Nguyen"
      ],
      "abstract": "Federated learning (FL) enables collaborative model training across distributed devices without sharing raw data, making it suitable for privacy-sensitive robotic sensing applications. However, multi-agent systems generate heterogeneous and non-independent and identically distributed (non-IID) multimodal sensor streams that degrade conventional FL algorithms, while classical fusion modules introduce substantial parameter overhead and communication cost. This paper proposes QFedAgent, a hybrid quantum-classical personalized FL framework for multi-agent activity recognition. The approach integrates a variational quantum circuit fusion module that models accelerometer--gyroscope interactions through quantum state encoding and entanglement, requiring only 72 quantum rotation parameters versus 33K in classical multi-layer perceptron-based fusion, achieving approximately 10x total parameter reduction. Experiments on the OPPORTUNITY dataset under subject-based non-IID partitions demonstrate 97.7% mean test accuracy, confirming that parameter-efficient quantum fusion remains competitive with conventional federated baselines.",
      "published": "2026-07-02T16:54:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.02426v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02426v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Neuron-Aware Active Few-Shot Learning for LLMs",
      "authors": [
        "Zhuowei Chen",
        "Liwei Chen",
        "Christian Schunn",
        "Raquel Coelho",
        "Xiang Lorraine Li"
      ],
      "abstract": "Active Few-Shot Learning (AFSL) adapts LLMs to specialized domains by identifying the most valuable unlabeled samples for annotation and use as few-shot demonstrations, effectively reducing human annotation costs while promoting high performance. However, existing methods typically rely on output-level signals for sample identification, such as predictive entropy or semantic similarities with test-time data based on external embeddings, which often overlook models' internal dynamics, which could pinpoint specific knowledge gaps. To bridge this gap, we propose NeuFS, a Neuron-Aware Active Few-Shot Learning framework that shifts the selection paradigm from output-level proxies to models' internal dynamics. NeuFS utilizes neuron activation patterns to represent sample directly, and includes a dual-criteria selection strategy that: (1) ensures few-shot sample diversity with neuron patterns for broader example coverage, while (2) prioritizing on identifying informative and challenging few-shot samples LLMs tend to hallucinate by quantifying neuron consensus. Experiments on three datasets demonstrate that NeuFS excels in both reasoning and text classification tasks, outperforming existing AFSL baselines. Ablation studies further highlight that internal neuron activations provide a more principled and effective selection signal than external embeddings, validating the superiority of the proposed NeuFS.",
      "published": "2026-07-02T16:51:11Z",
      "abstract_url": "http://arxiv.org/abs/2607.02423v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02423v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Text-Driven 3D Indoor Scene Synthesis in Non-Manhattan Environments",
      "authors": [
        "Xianhui Meng",
        "Zirui Song",
        "Yuchen Zhang",
        "Li Zhang",
        "Yongxuan Lv",
        "Xiuying Chen",
        "Kun Wang",
        "Yan Luo",
        "Kai Chen",
        "Hangjun Ye",
        "Long Chen",
        "Jun Liu",
        "Xiaoshuai Hao"
      ],
      "abstract": "Large Language Models (LLMs) have demonstrated remarkable capabilities in 3D indoor synthesis for Manhattan environments. However, existing methods often fail to capture plausible object layout patterns in non-Manhattan settings, primarily because they struggle to model non-orthogonal spatial relationships, leading to high geometric violations and low physical fidelity. To address this challenge, we propose SPG-Layout, a novel text-driven framework designed to generate physically plausible indoor scenes within complex non-Manhattan environments. Specifically, we first utilize statistical priors of object distributions to guide the training process, enhancing environmental understanding and fidelity. Furthermore, mirroring human design workflows, we adopt a hierarchical layout strategy that prioritizes the placement of large objects, thereby substantially minimizing layout violations. By synergizing these components, SPG-Layout achieves a balanced optimization of semantic realism and physical plausibility. To evaluate performance in these complex settings, we constructed a new benchmark comprising 500 diverse non-Manhattan environments. Extensive experiments demonstrate that SPG-Layout consistently and significantly outperforms existing methods across both Manhattan and non-Manhattan environments. The code will be publicly released.",
      "published": "2026-07-02T16:40:08Z",
      "abstract_url": "http://arxiv.org/abs/2607.02407v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02407v1",
      "categories": [
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Fast Multi-dimensional Refusal Subspaces via RFM-AGOP",
      "authors": [
        "Thomas Winninger"
      ],
      "abstract": "Steering and monitoring activations in Large Language Models (LLMs) are increasingly used for both safety and interpretability. Early work assumed behaviours are encoded along single linear directions, but recent findings suggest complex behaviours, such as the refusal to answer harmful queries, live in multi-dimensional subspaces. However, existing methods for extracting these subspaces are computationally expensive, which becomes prohibitive on reasoning models who produce long reasoning traces. By adapting the Recursive Feature Machine (RFM) algorithm -- which can be computed efficiently -- with a probe-informed initialization, we are able to identify the multi-dimensional refusal subspace in seconds, on reasoning (Qwen 3) and non-reasoning (Qwen 2.5) models. While RFM allows for faster subspace identification, it also showed better performances on the ablation task than its alternatives. More work is planned to better understand the relations between subspaces found by different methods. If confirmed, RFM could be a cheap and scalable complement to existing subspace-extraction methods in LLMs.",
      "published": "2026-07-02T16:31:56Z",
      "abstract_url": "http://arxiv.org/abs/2607.02396v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02396v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Steerability via constraints: a substrate for scalable oversight of coding agents",
      "authors": [
        "Thomas Winninger"
      ],
      "abstract": "Coding agents are capable; human oversight is the bottleneck. Unconstrained agents introduce security risks, erode codebase scalability, and make human review increasingly costly. We argue that the same methods used for decades to manage large human engineering teams: access control, network policies, strict coding conventions enforced by tooling; transfer directly to coding agents, and are cheaper (in token) than recent agentic scaffolding. We sketch a start-to-end system on this principle, and report a controlled experiment in scalable oversight: a small reviewer (Gemma 4 e4b) inspects a Python codebase containing 11 inserted backdoors. Recall rises from 54.5% (unconstrained, no tools) to 90.9% (constrained substrate plus a ~200-LoC `docs` CLI), with substrate and tools contributing independently. We choose Python deliberately: substrate-level oversight gains are largest where the language gives the fewest guarantees by default; the principles extend to languages like Rust.",
      "published": "2026-07-02T16:24:47Z",
      "abstract_url": "http://arxiv.org/abs/2607.02389v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02389v1",
      "categories": [
        "cs.AI",
        "cs.CR",
        "cs.SE"
      ]
    },
    {
      "title": "Hardware-Enforced Semantic Coordination for Safety-Critical Real-Time Autonomous Systems",
      "authors": [
        "Uwe M. Borghoff",
        "Paolo Bottoni",
        "Remo Pareschi"
      ],
      "abstract": "Recent advances in agentic AI are producing increasingly complex autonomous systems that integrate large language models, world models, optimization engines, specialized neural architectures, autonomous platforms, and human operators. While much current research focuses on improving reasoning capabilities, safety-critical real-time deployment also requires bounded and verifiable coordination among heterogeneous components operating concurrently under uncertainty. Software-mediated coordination presents fundamental limitations in domains where bounded latency, deterministic coordination, and enforceable safety guarantees are essential. Hence, we propose a hardware-enforced semantic coordination architecture in which selected coordination semantics are implemented directly at the hardware level via field-programmable gate arrays (FPGAs). The approach builds on the Topic-Based Communication Space Petri Net (TB-CSPN) framework, which separates semantic reasoning from interaction management. In this approach, selected TB-CSPN coordination mechanisms are mapped onto FPGA primitives, creating a hardware-native semantic coordination layer. Focus is not on acceleration, but on enforcing temporal synchronization, semantic gating, authorization constraints, and bounded coordination behavior directly in hardware. Semantic reasoning remains adaptive and software-driven, while embedded coordination semantics become deterministic.",
      "published": "2026-07-02T16:16:41Z",
      "abstract_url": "http://arxiv.org/abs/2607.02376v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02376v1",
      "categories": [
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "VisionAId: An Offline-First Multimodal Android Assistant for People with Visual Impairment, Featuring Personalized Object Retrieval",
      "authors": [
        "Cristian-Gabriel Florea",
        "Stelian Spînu"
      ],
      "abstract": "Over 285 million people worldwide live with a visual impairment, for whom everyday tasks such as avoiding obstacles, locating personal belongings, recognizing familiar faces, or handling cash remain persistent obstacles to personal autonomy. Existing assistive applications are typically limited to recognizing predefined categories, depend heavily on cloud connectivity, or require dedicated hardware. We present VisionAId, an Android application that turns a commodity smartphone into a real-time visual assistant. The system integrates six on-device deep learning models (metric monocular depth estimation, instance segmentation, visual and facial embeddings, face detection, and a custom banknote detector) running entirely through ONNX Runtime, with an optional cloud large language model (Google Gemini Flash) used only for narrative scene description and automatic object labeling. A distinctive contribution is a few-shot pipeline for personal objects: the user photographs an object from several angles, and the system later locates that specific instance in the environment, guiding the user toward it with augmented-reality markers, spatial audio, and distance-proportional haptics. All feedback is multimodal (Romanian speech synthesis, voice commands, vibration). On a reference device (Samsung Galaxy S21 Ultra), INT8 quantization reduces depth latency from ~1200 ms to ~491 ms, the custom banknote detector reaches an mAP@50 of 0.986, and metric depth is calibrated to below 1 cm of error within 3 m.",
      "published": "2026-07-02T16:12:50Z",
      "abstract_url": "http://arxiv.org/abs/2607.02371v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02371v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "The Dual Nature of LLM Persona: Aggregated Tendencies and Frame-Dependent Geometry",
      "authors": [
        "Yuan Yuan"
      ],
      "abstract": "Evaluations of LLM personas via psychometric questionnaires typically rely on aggregate scores, discarding within-instance correlation structure. We test whether this geometric structure is intrinsic or frame-dependent. Constructing within-instance correlation matrices from IPIP-50 responses, we analyze geometry on SPD manifolds under manipulated question orderings in GPT-4o simulating American and Chinese-American personas. We find that persona expression comprises two dissociable components: aggregated features (Big Five scores) degrade under randomization (21% drop) but are frame-robust; geometric features (SPD manifold) collapse under frame misalignment (42% drop) but recover substantially (to 84%) under shared frames, surpassing aggregated features (76%). This collapse-recovery pattern reveals that persona geometry is not intrinsic but a frame-dependent coordination pattern encoding information invisible to aggregation. Our findings establish a dual-nature framework for LLM personas, frame-dependent geometry versus frame-robust aggregates, necessitating frame-aware evaluation and challenging static trait conceptions.",
      "published": "2026-07-02T16:11:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.02368v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02368v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG",
        "math.DG"
      ]
    },
    {
      "title": "Stable Self-Modulating Quantum Fast-Weight Programmers with Bounded Memory Gates",
      "authors": [
        "Kuo-Chung Peng",
        "Jiun-Cheng Jiang",
        "Chun-Hua Lin",
        "Yifeng Peng",
        "Junghoon Justin Park",
        "Huan-Hsin Tseng",
        "Hsin-Yi Lin",
        "Kuan-Cheng Chen",
        "Chen-Yu Liu",
        "Shinjae Yoo",
        "Samuel Yen-Chi Chen"
      ],
      "abstract": "Quantum Fast-Weight Programmers (QFWPs) store temporal information in dynamically programmed variational-circuit parameters rather than in nonlinear recurrent hidden states, offering a practical route to quantum sequence modeling. Self-Modulating QFWP improves this framework by using input-dependent gates for both new fast-weight updates and the accumulated fast-weight state, but its unbounded old-state multiplier can diverge in long-sequence regimes. We propose a bounded old-state modulation rule that applies a sign-preserving tanh gate only to the recurrent memory branch while leaving the additive update and new-update modulation unchanged. We evaluate standard QFWP, full Self-Modulating QFWP, Only-New, and Only-Old variants on two CUDA-Q quantum-dynamics forecasting tasks and on Milan SMS telecommunication activity prediction. The quantum-dynamics results show that old-state modulation is the most consistent source of improvement over Standard QFWP, and that bounding the old-state gate removes long-sequence divergence while improving aggregate robustness. On Milan SMS forecasting, the original unbounded Self-Modulating QFWP converges across the tested grid and shows its clearest gains at longer input windows, with behavior close to the Only-Old ablation. These findings identify accumulated-memory modulation as the key mechanism of Self-Modulating QFWP and bounded old-state gating as a targeted stabilization strategy.",
      "published": "2026-07-02T16:06:04Z",
      "abstract_url": "http://arxiv.org/abs/2607.02363v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02363v1",
      "categories": [
        "quant-ph",
        "cs.AI",
        "cs.ET",
        "cs.LG",
        "cs.NE"
      ]
    },
    {
      "title": "SkillFuzz: Fuzzing Skill Composition for Implicit Intents Discovery in Open Skill Marketplaces",
      "authors": [
        "Jinwei Hu",
        "Yi Dong",
        "Youcheng Sun",
        "Xiaowei Huang"
      ],
      "abstract": "Large Language Model (LLM)-based agents increasingly automate software engineering tasks through reusable skills, natural-language instruction documents that guide planning and execution. Open skill marketplaces enable users to assemble agents by co-activating community-contributed skills, but marketplace operators typically audit skills in isolation. As a result, individually benign skills may interact to redirect an agent toward unintended objectives, which we term implicit intents. Detecting such intents is challenging because the effect emerges only through skill composition, execution environments are often unavailable at admission time, and the space of possible co-activations grows exponentially with marketplace size. In this paper, we formulate implicit-intent discovery as a fuzzing problem over skill compositions, where skill compositions are the unit under test, planning artifacts expose agent intent before execution, and deviations from a skill-free baseline serve as a differential oracle. Based on this formulation, we propose skillfuzz, the first execution-free testing approach that extracts structured skill contracts and uses contract-guided Monte Carlo Tree Search to prioritize potentially conflicting compositions. Across representative skill-marketplace workloads, skillfuzz discovers over 1,000 distinct implicit intents under a fixed query budget, confirms more than 80% of the highest-risk flagged compositions during execution-time validation, and identifies substantially more high-severity implicit intents than alternative search strategies while exploring only a fraction of the pairwise interaction space they require.",
      "published": "2026-07-02T15:49:21Z",
      "abstract_url": "http://arxiv.org/abs/2607.02345v1",
      "pdf_url": "https://arxiv.org/pdf/2607.02345v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.CL"
      ]
    }
  ]
};
