const PAPERS_DATA = {
  "last_updated": "2026-04-14 03:29:35 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Physics-Informed State Space Models for Reliable Solar Irradiance Forecasting in Off-Grid Systems",
      "authors": [
        "Mohammed Ezzaldin Babiker Abdullah"
      ],
      "abstract": "The stable operation of autonomous off-grid photovoltaic systems dictates reliance on solar forecasting algorithms that respect atmospheric thermodynamics. Contemporary deep learning models consistently exhibit critical anomalies, primarily severe temporal phase lags during cloud transients and physically impossible nocturnal power generation. To resolve this divergence between data-driven modeling and deterministic celestial mechanics, this research introduces the Thermodynamic Liquid Manifold Network. The proposed methodology projects 15 meteorological and geometric variables into a Koopman-linearized Riemannian manifold to systematically map complex climatic dynamics. The architecture integrates a Spectral Calibration unit and a multiplicative Thermodynamic Alpha-Gate. This system synthesizes real-time atmospheric opacity with theoretical clear-sky boundary models, structurally enforcing strict celestial geometry compliance. This completely neutralizes phantom nocturnal generation while maintaining zero-lag synchronization during rapid weather shifts. Validated against a rigorous five-year testing horizon in a severe semi-arid climate, the framework achieves an RMSE of 18.31 Wh/m2 and a Pearson correlation of 0.988. The model strictly maintains a zero-magnitude nocturnal error across all 1826 testing days and exhibits a sub-30-minute phase response during high-frequency transients. Comprising exactly 63,458 trainable parameters, this ultra-lightweight design establishes a robust, thermodynamically consistent standard for edge-deployable microgrid controllers.",
      "published": "2026-04-13T17:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2604.11807v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11807v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Solving Physics Olympiad via Reinforcement Learning on Physics Simulators",
      "authors": [
        "Mihir Prabhudesai",
        "Aryan Satpathy",
        "Yangmin Li",
        "Zheyang Qin",
        "Nikash Bhardwaj",
        "Amir Zadeh",
        "Chuan Li",
        "Katerina Fragkiadaki",
        "Deepak Pathak"
      ],
      "abstract": "We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question-answer (QA) pairs, a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question-answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: https://sim2reason.github.io/.",
      "published": "2026-04-13T17:59:40Z",
      "abstract_url": "http://arxiv.org/abs/2604.11805v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11805v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "cs.RO"
      ]
    },
    {
      "title": "C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts",
      "authors": [
        "Chenxi Qing",
        "Junxi Wu",
        "Zheng Liu",
        "Yixiang Qiu",
        "Hongyao Yu",
        "Bin Chen",
        "Hao Wu",
        "Shu-Tao Xia"
      ],
      "abstract": "Recently, large language models (LLMs) are capable of generating highly fluent textual content. While they offer significant convenience to humans, they also introduce various risks, like phishing and academic dishonesty. Numerous research efforts have been dedicated to developing algorithms for detecting AI-generated text and constructing relevant datasets. However, in the domain of Chinese corpora, challenges remain, including limited model diversity and data homogeneity. To address these issues, we propose C-ReD: a comprehensive Chinese Real-prompt AI-generated Detection benchmark. Experiments demonstrate that C-ReD not only enables reliable in-domain detection but also supports strong generalization to unseen LLMs and external Chinese datasets-addressing critical gaps in model diversity, domain coverage, and prompt realism that have limited prior Chinese detection benchmarks. We release our resources at https://github.com/HeraldofLight/C-ReD.",
      "published": "2026-04-13T17:56:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.11796v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11796v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Mechanistic Analysis of Looped Reasoning Language Models",
      "authors": [
        "Hugh Blayney",
        "Álvaro Arroyo",
        "Johan Obando-Ceron",
        "Pablo Samuel Castro",
        "Aaron Courville",
        "Michael M. Bronstein",
        "Xiaowen Dong"
      ],
      "abstract": "Reasoning has become a central capability in large language models. Recent research has shown that reasoning performance can be improved by looping an LLM's layers in the latent dimension, resulting in looped reasoning language models. Despite promising results, few works have investigated how their internal dynamics differ from those of standard feedforward models. In this paper, we conduct a mechanistic analysis of the latent states in looped language models, focusing in particular on how the stages of inference observed in feedforward models compare to those observed in looped ones. To this end, we analyze cyclic recurrence and show that for many of the studied models each layer in the cycle converges to a distinct fixed point; consequently, the recurrent block follows a consistent cyclic trajectory in the latent space. We provide evidence that as these fixed points are reached, attention-head behavior stabilizes, leading to constant behavior across recurrences. Empirically, we discover that recurrent blocks learn stages of inference that closely mirror those of feedforward models, repeating these stages in depth with each iteration. We study how recurrent block size, input injection, and normalization influence the emergence and stability of these cyclic fixed points. We believe these findings help translate mechanistic insights into practical guidance for architectural design.",
      "published": "2026-04-13T17:55:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.11791v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11791v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ClawGuard: A Runtime Security Framework for Tool-Augmented LLM Agents Against Indirect Prompt Injection",
      "authors": [
        "Wei Zhao",
        "Zhe Li",
        "Peixin Zhang",
        "Jun Sun"
      ],
      "abstract": "Tool-augmented Large Language Model (LLM) agents have demonstrated impressive capabilities in automating complex, multi-step real-world tasks, yet remain vulnerable to indirect prompt injection. Adversaries exploit this weakness by embedding malicious instructions within tool-returned content, which agents directly incorporate into their conversation history as trusted observations. This vulnerability manifests across three primary attack channels: web and local content injection, MCP server injection, and skill file injection. To address these vulnerabilities, we introduce \\textsc{ClawGuard}, a novel runtime security framework that enforces a user-confirmed rule set at every tool-call boundary, transforming unreliable alignment-dependent defense into a deterministic, auditable mechanism that intercepts adversarial tool calls before any real-world effect is produced. By automatically deriving task-specific access constraints from the user's stated objective prior to any external tool invocation, \\textsc{ClawGuard} blocks all three injection pathways without model modification or infrastructure change. Experiments across five state-of-the-art language models on AgentDojo, SkillInject, and MCPSafeBench demonstrate that \\textsc{ClawGuard} achieves robust protection against indirect prompt injection without compromising agent utility. This work establishes deterministic tool-call boundary enforcement as an effective defense mechanism for secure agentic AI systems, requiring neither safety-specific fine-tuning nor architectural modification. Code is publicly available at https://github.com/Claw-Guard/ClawGuard.",
      "published": "2026-04-13T17:55:11Z",
      "abstract_url": "http://arxiv.org/abs/2604.11790v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11790v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "ClawGUI: A Unified Framework for Training, Evaluating, and Deploying GUI Agents",
      "authors": [
        "Fei Tang",
        "Zhiqiong Lu",
        "Boxuan Zhang",
        "Weiming Lu",
        "Jun Xiao",
        "Yueting Zhuang",
        "Yongliang Shen"
      ],
      "abstract": "GUI agents drive applications through their visual interfaces instead of programmatic APIs, interacting with arbitrary software via taps, swipes, and keystrokes, reaching a long tail of applications that CLI-based agents cannot. Yet progress in this area is bottlenecked less by modeling capacity than by the absence of a coherent full-stack infrastructure: online RL training suffers from environment instability and closed pipelines, evaluation protocols drift silently across works, and trained agents rarely reach real users on real devices. We present \\textbf{ClawGUI}, an open-source framework addressing these three gaps within a single harness. \\textbf{ClawGUI-RL} provides the first open-source GUI agent RL infrastructure with validated support for both parallel virtual environments and real physical devices, integrating GiGPO with a Process Reward Model for dense step-level supervision. \\textbf{ClawGUI-Eval} enforces a fully standardized evaluation pipeline across 6 benchmarks and 11+ models, achieving 95.8\\% reproduction against official baselines. \\textbf{ClawGUI-Agent} brings trained agents to Android, HarmonyOS, and iOS through 12+ chat platforms with hybrid CLI-GUI control and persistent personalized memory. Trained end to end within this pipeline, \\textbf{ClawGUI-2B} achieves 17.1\\% Success Rate on MobileWorld GUI-Only, outperforming the same-scale MAI-UI-2B baseline by 6.0\\%.",
      "published": "2026-04-13T17:52:04Z",
      "abstract_url": "http://arxiv.org/abs/2604.11784v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11784v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.CV"
      ]
    },
    {
      "title": "General365: Benchmarking General Reasoning in Large Language Models Across Diverse and Challenging Tasks",
      "authors": [
        "Junlin Liu",
        "Shengnan An",
        "Shuang Zhou",
        "Dan Ma",
        "Shixiong Luo",
        "Ying Xie",
        "Yuan Zhang",
        "Wenling Yuan",
        "Yifan Zhou",
        "Xiaoyu Li",
        "Ziwen Wang",
        "Xuezhi Cao",
        "Xunliang Cai"
      ],
      "abstract": "Contemporary large language models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in specialized domains like mathematics and physics. However, their ability to generalize these reasoning skills to more general and broader contexts--often termed general reasoning--remains under-explored. Unlike domain-specific reasoning, general reasoning relies less on expert knowledge but still presents formidable reasoning challenges, such as complex constraints, nested logical branches, and semantic interference. To address this gap, we introduce General365, a benchmark specifically designed to assess general reasoning in LLMs. By restricting background knowledge to a K-12 level, General365 explicitly decouples reasoning from specialized expertise. The benchmark comprises 365 seed problems and 1,095 variant problems across eight categories, ensuring both high difficulty and diversity. Evaluations across 26 leading LLMs reveal that even the top-performing model achieves only 62.8% accuracy, in stark contrast to the near-perfect performances of LLMs in math and physics benchmarks. These results suggest that the reasoning abilities of current LLMs are heavily domain-dependent, leaving significant room for improvement in broader applications. We envision General365 as a catalyst for advancing LLM reasoning beyond domain-specific tasks toward robust, general-purpose real-world scenarios. Code, Dataset, and Leaderboard: https://general365.github.io",
      "published": "2026-04-13T17:44:25Z",
      "abstract_url": "http://arxiv.org/abs/2604.11778v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11778v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Discourse Diversity in Multi-Turn Empathic Dialogue",
      "authors": [
        "Hongli Zhan",
        "Emma S. Gueorguieva",
        "Javier Hernandez",
        "Jina Suh",
        "Desmond C. Ong",
        "Junyi Jessy Li"
      ],
      "abstract": "Large language models (LLMs) produce responses rated as highly empathic in single-turn settings (Ayers et al., 2023; Lee et al., 2024), yet they are also known to be formulaic generators that reuse the same lexical patterns, syntactic templates, and discourse structures across tasks (Jiang et al., 2025; Shaib et al., 2024; Namuduri et al., 2025). Less attention has been paid to whether this formulaicity extends to the level of discourse moves, i.e., what a response does for the person it is addressing. This question is especially consequential for empathic dialogue, where effective support demands not just a kind response at one moment but varied strategies as a conversation unfolds (Stiles et al., 1998). Indeed, prior work shows that LLMs reuse the same tactic sequences more than human supporters in single-turn settings (Gueorguieva et al., 2026). We extend this analysis to multi-turn conversations and find that the rigidity compounds: once a tactic appears in a supporter turn, LLMs reuse it in the next at nearly double the rate of humans (0.50-0.56 vs. 0.27). This pattern holds across LLMs serving as supporters in real emotional support conversations, and is invisible to standard similarity metrics. To address this gap, we introduce MINT (Multi-turn Inter-tactic Novelty Training), the first reinforcement learning framework to optimize discourse move diversity across multi-turn empathic dialogue. The best MINT variant combines an empathy quality reward with a cross-turn tactic novelty signal, improving aggregate empathy by 25.3% over vanilla across 1.7B and 4B models while reducing cross-turn discourse move repetition by 26.3% on the 4B model, surpassing all baselines including quality-only and token-level diversity methods on both measures. These results suggest that what current models lack is not empathy itself, but the ability to vary their discourse moves across a conversation.",
      "published": "2026-04-13T17:17:22Z",
      "abstract_url": "http://arxiv.org/abs/2604.11742v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11742v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Evaluating Cooperation in LLM Social Groups through Elected Leadership",
      "authors": [
        "Ryan Faulkner",
        "Anushka Deshpande",
        "David Guzman Piedrahita",
        "Joel Z. Leibo",
        "Zhijing Jin"
      ],
      "abstract": "Governing common-pool resources requires agents to develop enduring strategies through cooperation and self-governance to avoid collective failure. While foundation models have shown potential for cooperation in these settings, existing multi-agent research provides little insight into whether structured leadership and election mechanisms can improve collective decision making. The lack of such a critical organizational feature ubiquitous in human society presents a significant shortcoming of the current methods. In this work we aim to directly address whether leadership and elections can support improved social welfare and cooperation through multi-agent simulation with LLMs. We present our open-source framework that simulates leadership through elected personas and candidate-driven agendas and carry out an empirical study of LLMs under controlled governance conditions. Our experiments demonstrate that having elected leadership improves social welfare scores by 55.4% and survival time by 128.6% across a range of high performing LLMs. Through the construction of an agent social graph we compute centrality metrics to assess the social influence of leader personas and also analyze rhetorical and cooperative tendencies revealed through a sentiment analysis on leader utterances. This work lays the foundation for further study of election mechanisms in multi-agent systems toward navigating complex social dilemmas.",
      "published": "2026-04-13T16:57:11Z",
      "abstract_url": "http://arxiv.org/abs/2604.11721v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11721v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A Mamba-Based Multimodal Network for Multiscale Blast-Induced Rapid Structural Damage Assessment",
      "authors": [
        "Wanli Ma",
        "Sivasakthy Selvakumaran",
        "Dain G. Farrimond",
        "Adam A. Dennis",
        "Samuel E. Rigby"
      ],
      "abstract": "Accurate and rapid structural damage assessment (SDA) is crucial for post-disaster management, helping responders prioritise resources, plan rescues, and support recovery. Traditional field inspections, though precise, are limited by accessibility, safety risks, and time constraints, especially after large explosions. Machine learning with remote sensing has emerged as a scalable solution for rapid SDA, with Mamba-based networks achieving state-of-the-art performance. However, these methods often require extensive training and large datasets, limiting real-world applicability. Moreover, they fail to incorporate key physical characteristics of blast loading for SDA. To overcome these challenges, we propose a Mamba-based multimodal network for rapid SDA that integrates multi-scale blast-loading information with optical remote sensing images. Evaluated on the 2020 Beirut explosion, our method significantly improves performance over state-of-the-art approaches. Code is available at: https://github.com/IMPACTSquad/Blast-Mamba",
      "published": "2026-04-13T16:43:16Z",
      "abstract_url": "http://arxiv.org/abs/2604.11709v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11709v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Agentic Driving Coach: Robustness and Determinism of Agentic AI-Powered Human-in-the-Loop Cyber-Physical Systems",
      "authors": [
        "Deeksha Prahlad",
        "Daniel Fan",
        "Hokeun Kim"
      ],
      "abstract": "Foundation models, including large language models (LLMs), are increasingly used for human-in-the-loop (HITL) cyber-physical systems (CPS) because foundation model-based AI agents can potentially interact with both the physical environments and human users. However, the unpredictable behavior of human users and AI agents, in addition to the dynamically changing physical environments, leads to uncontrollable nondeterminism. To address this urgent challenge of enabling agentic AI-powered HITL CPS, we propose a reactor-model-of-computation (MoC)-based approach, realized by the open-source Lingua Franca (LF) framework. We also carry out a concrete case study using the agentic driving coach as an application of HITL CPS. By evaluating the LF-based agentic HITL CPS, we identify practical challenges in reintroducing determinism into such agentic HITL CPS and present pathways to address them.",
      "published": "2026-04-13T16:42:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.11705v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11705v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.RO",
        "eess.SY"
      ]
    },
    {
      "title": "Fairness is Not Flat: Geometric Phase Transitions Against Shortcut Learning",
      "authors": [
        "Nicolas Rodriguez-Alvarez",
        "Fernando Rodriguez-Merino"
      ],
      "abstract": "Deep Neural Networks are highly susceptible to shortcut learning, frequently memorizing low-dimensional spurious correlations instead of underlying causal mechanisms. This phenomenon not only degrades out-of-distribution robustness but also induces severe demographic biases in sensitive applications. In this paper, we propose a geometric \\textit{a priori} methodology to mitigate shortcut learning. By deploying a zero-hidden-layer ($N=1$) Topological Auditor, we mathematically isolate features that monopolize the gradient without human intervention. We empirically demonstrate a Capacity Phase Transition: once linear shortcuts are pruned, networks are forced to utilize higher geometric capacity ($N \\geq 16$) to curve the decision boundary and learn ethical representations. Our approach outperforms L1 Regularization -- which collapses into demographic bias -- and operates at a fraction of the computational cost of post-hoc methods like Just Train Twice (JTT), successfully reducing counterfactual gender vulnerability from 21.18\\% to 7.66\\%.",
      "published": "2026-04-13T16:40:26Z",
      "abstract_url": "http://arxiv.org/abs/2604.11704v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11704v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "DreamKG: A KG-Augmented Conversational System for People Experiencing Homelessness",
      "authors": [
        "Javad M Alizadeh",
        "Genhui Zheng",
        "Chiu C Tan",
        "Yuzhou Chen",
        "Omar Martinez",
        "Philip McCallion",
        "Ying Ding",
        "Chenguang Yang",
        "AnneMarie Tomosky",
        "Huanmei Wu"
      ],
      "abstract": "People experiencing homelessness (PEH) face substantial barriers to accessing timely, accurate information about community services. DreamKG addresses this through a knowledge graph-augmented conversational system that grounds responses in verified, up-to-date data about Philadelphia organizations, services, locations, and hours. Unlike standard large language models (LLMs) prone to hallucinations, DreamKG combines Neo4j knowledge graphs with structured query understanding to handle location-aware and time-sensitive queries reliably. The system performs spatial reasoning for distance-based recommendations and temporal filtering for operating hours. Preliminary evaluation shows 59% superiority over Google Search AI on relevant queries and 84% rejection of irrelevant queries. This demonstration highlights the potential of hybrid architectures that combines LLM flexibility with knowledge graph reliability to improve service accessibility for vulnerable populations effectively.",
      "published": "2026-04-13T16:38:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.11703v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11703v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Legal2LogicICL: Improving Generalization in Transforming Legal Cases to Logical Formulas via Diverse Few-Shot Learning",
      "authors": [
        "Jieying Xue",
        "Phuong Minh Nguyen",
        "Ha Thanh Nguyen",
        "May Myo Zin",
        "Ken Satoh"
      ],
      "abstract": "This work aims to improve the generalization of logic-based legal reasoning systems by integrating recent advances in NLP with legal-domain adaptive few-shot learning techniques using LLMs. Existing logic-based legal reasoning pipelines typically rely on fine-tuned models to map natural-language legal cases into logical formulas before forwarding them to a symbolic reasoner. However, such approaches are heavily constrained by the scarcity of high-quality annotated training data. To address this limitation, we propose a novel LLM-based legal reasoning framework that enables effective in-context learning through retrieval-augmented generation. Specifically, we introduce Legal2LogicICL, a few-shot retrieval framework that balances diversity and similarity of exemplars at both the latent semantic representation level and the legal text structure level. In addition, our method explicitly accounts for legal structure by mitigating entity-induced retrieval bias in legal texts, where lengthy and highly specific entity mentions often dominate semantic representations and obscure legally meaningful reasoning patterns. Our Legal2LogicICL constructs informative and robust few-shot demonstrations, leading to accurate and stable logical rule generation without requiring additional training. In addition, we construct a new dataset, named Legal2Proleg, which is annotated with alignments between legal cases and PROLEG logical formulas to support the evaluation of legal semantic parsing. Experimental results on both open-source and proprietary LLMs demonstrate that our approach significantly improves accuracy, stability, and generalization in transforming natural-language legal case descriptions into logical representations, highlighting its effectiveness for interpretable and reliable legal reasoning. Our code is available at https://github.com/yingjie7/Legal2LogicICL.",
      "published": "2026-04-13T16:36:48Z",
      "abstract_url": "http://arxiv.org/abs/2604.11699v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11699v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind",
      "authors": [
        "Hanqi Xiao",
        "Vaidehi Patil",
        "Zaid Khan",
        "Hyunji Lee",
        "Elias Stengel-Eskin",
        "Mohit Bansal"
      ],
      "abstract": "As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, ToM for Steering Beliefs (ToM-SB), in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task.",
      "published": "2026-04-13T16:14:41Z",
      "abstract_url": "http://arxiv.org/abs/2604.11666v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11666v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Why Do Large Language Models Generate Harmful Content?",
      "authors": [
        "Rajesh Ganguli",
        "Raha Moraffah"
      ],
      "abstract": "Large Language Models (LLMs) have been shown to generate harmful content. However, the underlying causes of such behavior remain under explored. We propose a causal mediation analysis-based approach to identify the causal factors responsible for harmful generation. Our method performs a multi-granular analysis across model layers, modules (MLP and attention blocks), and individual neurons. Extensive experiments on state-of-the-art LLMs indicate that harmful generation arises in the later layers of the model, results primarily from failures in MLP blocks rather than attention blocks, and is associated with neurons that act as a gating mechanism for harmful generation. The results indicate that the early layers in the model are used for a contextual understanding of harmfulness in a prompt, which is then propagated through the model, to generate harmfulness in the late layers, as well as a signal indicating harmfulness through MLP blocks. This is then further propagated to the last layer of the model, specifically to a sparse set of neurons, which receives the signal and determines the generation of harmful content accordingly.",
      "published": "2026-04-13T16:11:38Z",
      "abstract_url": "http://arxiv.org/abs/2604.11663v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11663v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Towards Autonomous Mechanistic Reasoning in Virtual Cells",
      "authors": [
        "Yunhui Jang",
        "Lu Zhu",
        "Jake Fawkes",
        "Alisandra Kaye Denton",
        "Dominique Beaini",
        "Emmanuel Noutahi"
      ],
      "abstract": "Large language models (LLMs) have recently gained significant attention as a promising approach to accelerate scientific discovery. However, their application in open-ended scientific domains such as biology remains limited, primarily due to the lack of factually grounded and actionable explanations. To address this, we introduce a structured explanation formalism for virtual cells that represents biological reasoning as mechanistic action graphs, enabling systematic verification and falsification. Building upon this, we propose VCR-Agent, a multi-agent framework that integrates biologically grounded knowledge retrieval with a verifier-based filtering approach to generate and validate mechanistic reasoning autonomously. Using this framework, we release VC-TRACES dataset, which consists of verified mechanistic explanations derived from the Tahoe-100M atlas. Empirically, we demonstrate that training with these explanations improves factual precision and provides a more effective supervision signal for downstream gene expression prediction. These results underscore the importance of reliable mechanistic reasoning for virtual cells, achieved through the synergy of multi-agent and rigorous verification.",
      "published": "2026-04-13T16:10:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.11661v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11661v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "RPA-Check: A Multi-Stage Automated Framework for Evaluating Dynamic LLM-based Role-Playing Agents",
      "authors": [
        "Riccardo Rosati",
        "Edoardo Colucci",
        "Massimiliano Bolognini",
        "Adriano Mancini",
        "Paolo Sernani"
      ],
      "abstract": "The rapid adoption of Large Language Models (LLMs) in interactive systems has enabled the creation of dynamic, open-ended Role-Playing Agents (RPAs). However, evaluating these agents remains a significant challenge, as standard NLP metrics fail to capture the nuances of role adherence, logical consistency, and long-term narrative stability. This paper introduces RPA-Check, a multi-stage automated evaluation framework designed to objectively assess the performance of LLM-based RPAs in complex, constraints-heavy environments. Our methodology is based on a four-step pipeline: (1) Dimension Definition, establishing high-level qualitative behavioral criteria; (2) Augmentation, where these requirements are expanded into granular boolean checklist indicators; (3) Semantic Filtering, to ensure indicator objectivity, no redundancy and agent isolation; and (4) LLM-as-a-Judge Evaluation, which employs chain-of-thought verification to score agent fidelity. We validate this framework by applying it to LLM Court, a serious game for forensic training involving several quantized local models. Experimental results across five distinct legal scenarios demonstrate the framework's ability to identify subtle trade-offs between model size, reasoning depth, and operational stability. Notably, the findings reveal an inverse relationship between parametric scale and procedural consistency, showing that smaller, adequately instruction-tuned models (8-9B) can outperform larger architectures prone to user-alignment bias or sycophancy. RPA-Check thus provides a standardized and reproducible metric for future research in generative agent evaluation within specialized domains.",
      "published": "2026-04-13T16:08:03Z",
      "abstract_url": "http://arxiv.org/abs/2604.11655v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11655v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time",
      "authors": [
        "Haozhe Wang",
        "Cong Wei",
        "Weiming Ren",
        "Jiaming Liu",
        "Fangzhen Lin",
        "Wenhu Chen"
      ],
      "abstract": "Most reward models for visual generation reduce rich human judgments to a single unexplained score, discarding the reasoning that underlies preference. We show that teaching reward models to produce explicit, multi-dimensional critiques before scoring transforms them from passive evaluators into active optimization tools, improving generators in two complementary ways: at training time, structured rationales provide interpretable, fine-grained rewards for reinforcement learning; at test time, a Generate-Critique-Refine loop turns critiques into targeted prompt revisions that improve outputs without any parameter updates. To train such a reward model without costly rationale annotations, we introduce Preference-Anchored Rationalization (PARROT), a principled framework that recovers high-quality rationales from readily available preference data through anchored generation, consistency filtering, and distillation. The resulting model, RationalRewards (8B), achieves state-of-the-art preference prediction among open-source reward models, competitive with Gemini-2.5-Pro, while using 10-20x less training data than comparable baselines. As an RL reward, it consistently improves text-to-image and image-editing generators beyond scalar alternatives. Most strikingly, its test-time critique-and-refine loop matches or exceeds RL-based fine-tuning on several benchmarks, suggesting that structured reasoning can unlock latent capabilities in existing generators that suboptimal prompts fail to elicit.",
      "published": "2026-04-13T15:38:09Z",
      "abstract_url": "http://arxiv.org/abs/2604.11626v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11626v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SCNO: Spiking Compositional Neural Operator -- Towards a Neuromorphic Foundation Model for Nuclear PDE Solving",
      "authors": [
        "Samrendra Roy",
        "Souvik Chakraborty",
        "Rizwan-uddin",
        "Syed Bahauddin Alam"
      ],
      "abstract": "Neural operators have emerged as powerful surrogates for partial differential equation (PDE) solvers, yet they are typically trained as monolithic models for individual PDEs, require energy-intensive GPU hardware, and must be retrained from scratch when new physics emerge. We introduce the Spiking Compositional Neural Operator (SCNO), a modular architecture combining spiking and conventional components that addresses all three limitations. SCNO maintains a library of small spiking neural operator blocks, each trained on a single elementary differential operator (convection, diffusion, reaction), and composes them through a lightweight input-conditioned aggregator to solve coupled PDEs not seen during block training. A small correction network learns cross-coupling residuals while keeping all blocks and the aggregator frozen, preserving zero-forgetting modular expansion by construction. We evaluate SCNO on eight PDE families including five coupled systems and a nuclear-relevant 1-group neutron diffusion equation. SCNO with correction achieves the lowest relative $L^2$ error on four of five coupled PDEs, outperforming both a monolithic spiking DeepONet (by up to 62%, mean over 3 seeds) and a standard ANN DeepONet (by up to 65%), while requiring only 95K trainable parameters versus 462K for the monolithic baseline. To our knowledge, this is the first compositional spiking neural operator and the first proof-of-concept for modular neuromorphic PDE solving with built-in forgetting-free expansion.",
      "published": "2026-04-13T15:36:48Z",
      "abstract_url": "http://arxiv.org/abs/2604.11625v1",
      "pdf_url": "https://arxiv.org/pdf/2604.11625v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
