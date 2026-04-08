const PAPERS_DATA = {
  "last_updated": "2026-04-08 03:21:37 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "In-Place Test-Time Training",
      "authors": [
        "Guhao Feng",
        "Shengjie Luo",
        "Kai Hua",
        "Ge Zhang",
        "Di He",
        "Wenhao Huang",
        "Tianle Cai"
      ],
      "abstract": "The static ``train then deploy\" paradigm fundamentally limits Large Language Models (LLMs) from dynamically adapting their weights in response to continuous streams of new information inherent in real-world tasks. Test-Time Training (TTT) offers a compelling alternative by updating a subset of model parameters (fast weights) at inference time, yet its potential in the current LLM ecosystem is hindered by critical barriers including architectural incompatibility, computational inefficiency and misaligned fast weight objectives for language modeling. In this work, we introduce In-Place Test-Time Training (In-Place TTT), a framework that seamlessly endows LLMs with Test-Time Training ability. In-Place TTT treats the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights, enabling a ``drop-in\" enhancement for LLMs without costly retraining from scratch. Furthermore, we replace TTT's generic reconstruction objective with a tailored, theoretically-grounded objective explicitly aligned with the Next-Token-Prediction task governing autoregressive language modeling. This principled objective, combined with an efficient chunk-wise update mechanism, results in a highly scalable algorithm compatible with context parallelism. Extensive experiments validate our framework's effectiveness: as an in-place enhancement, it enables a 4B-parameter model to achieve superior performance on tasks with contexts up to 128k, and when pretrained from scratch, it consistently outperforms competitive TTT-related approaches. Ablation study results further provide deeper insights on our design choices. Collectively, our results establish In-Place TTT as a promising step towards a paradigm of continual learning in LLMs.",
      "published": "2026-04-07T17:59:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.06169v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06169v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "stat.ML"
      ]
    },
    {
      "title": "Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement",
      "authors": [
        "Qimin Zhong",
        "Hao Liao",
        "Haiming Qin",
        "Mingyang Zhou",
        "Rui Mao",
        "Wei Chen",
        "Naipeng Chao"
      ],
      "abstract": "Whether Large Language Models (LLMs) develop coherent internal world models remains a core debate. While conventional Next-Token Prediction (NTP) focuses on one-step-ahead supervision, Multi-Token Prediction (MTP) has shown promise in learning more structured representations. In this work, we provide a theoretical perspective analyzing the gradient inductive bias of MTP, supported by empirical evidence, showing that MTP promotes the convergence toward internal belief states by inducing representational contractivity via gradient coupling. However, we reveal that standard MTP often suffers from structural hallucinations, where discrete token supervision encourages illegal shortcuts in latent space that violate environmental constraints. To address this, we propose a novel method Latent Semantic Enhancement MTP (LSE-MTP), which anchors predictions to ground-truth hidden state trajectories. Experiments on synthetic graphs and real-world Manhattan Taxi Ride show that LSE-MTP effectively bridges the gap between discrete tokens and continuous state representations, enhancing representation alignment, reducing structural hallucinations, and improving robustness to perturbations.",
      "published": "2026-04-07T17:54:22Z",
      "abstract_url": "http://arxiv.org/abs/2604.06155v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06155v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Shot-Based Quantum Encoding: A Data-Loading Paradigm for Quantum Neural Networks",
      "authors": [
        "Basil Kyriacou",
        "Viktoria Patapovich",
        "Maniraman Periyasamy",
        "Alexey Melnikov"
      ],
      "abstract": "Efficient data loading remains a bottleneck for near-term quantum machine-learning. Existing schemes (angle, amplitude, and basis encoding) either underuse the exponential Hilbert-space capacity or require circuit depths that exceed the coherence budgets of noisy intermediate-scale quantum hardware. We introduce Shot-Based Quantum Encoding (SBQE), a data embedding strategy that distributes the hardware's native resource, shots, according to a data-dependent classical distribution over multiple initial quantum states. By treating the shot counts as a learnable degree of freedom, SBQE produces a mixed-state representation whose expectation values are linear in the classical probabilities and can therefore be composed with non-linear activation functions. We show that SBQE is structurally equivalent to a multilayer perceptron whose weights are realised by quantum circuits, and we describe a hardware-compatible implementation protocol. Benchmarks on Fashion MNIST and Semeion handwritten digits, with ten independent initialisations per model, show that SBQE achieves 89.1% +/- 0.9% test accuracy on Semeion (reducing error by 5.3% relative to amplitude encoding and matching a width-matched classical network) and 80.95% +/- 0.10% on Fashion MNIST (exceeding amplitude encoding by +2.0% and a linear multilayer perceptron by +1.3%), all without any data-encoding gates.",
      "published": "2026-04-07T17:44:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.06135v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06135v1",
      "categories": [
        "quant-ph",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents",
      "authors": [
        "Bowen Ye",
        "Rang Li",
        "Qibin Yang",
        "Yuanxin Liu",
        "Linli Yao",
        "Hanglong Lv",
        "Zhihui Xie",
        "Chenxin An",
        "Lei Li",
        "Lingpeng Kong",
        "Qi Liu",
        "Zhifang Sui",
        "Tong Yang"
      ],
      "abstract": "Large language models are increasingly deployed as autonomous agents executing multi-step workflows in real-world software environments. However, existing agent benchmarks suffer from three critical limitations: (1) trajectory-opaque grading that checks only final outputs, (2) underspecified safety and robustness evaluation, and (3) narrow modality coverage and interaction paradigms. We introduce Claw-Eval, an end-to-end evaluation suite addressing all three gaps. It comprises 300 human-verified tasks spanning 9 categories across three groups (general service orchestration, multimodal perception and generation, and multi-turn professional dialogue). Every agent action is recorded through three independent evidence channels (execution traces, audit logs, and environment snapshots), enabling trajectory-aware grading over 2,159 fine-grained rubric items. The scoring protocol evaluates Completion, Safety, and Robustness, reporting Average Score, Pass@k, and Pass^k across three trials to distinguish genuine capability from lucky outcomes. Experiments on 14 frontier models reveal that: (1) trajectory-opaque evaluation is systematically unreliable, missing 44% of safety violations and 13% of robustness failures that our hybrid pipeline catches; (2) controlled error injection primarily degrades consistency rather than peak capability, with Pass^3 dropping up to 24% while Pass@3 remains stable; (3) multimodal performance varies sharply, with most models performing poorer on video than on document or image, and no single model dominating across all modalities. Beyond benchmarking, Claw-Eval highlights actionable directions for agent development, shedding light on what it takes to build agents that are not only capable but reliably deployable.",
      "published": "2026-04-07T17:43:18Z",
      "abstract_url": "http://arxiv.org/abs/2604.06132v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06132v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Gym-Anything: Turn any Software into an Agent Environment",
      "authors": [
        "Pranjal Aggarwal",
        "Graham Neubig",
        "Sean Welleck"
      ],
      "abstract": "Computer-use agents hold the promise of assisting in a wide range of digital economic activities. However, current research has largely focused on short-horizon tasks over a limited set of software with limited economic value, such as basic e-commerce and OS-configuration tasks. A key reason is that creating environments for complex software requires significant time and human effort, and therefore does not scale. To address this, we introduce Gym-Anything, a framework for converting any software into an interactive computer-use environment. We frame environment creation itself as a multi-agent task: a coding agent writes setup scripts, downloads real-world data, and configures the software, while producing evidence of correct setup. An independent audit agent then verifies evidence for the environment setup against a quality checklist. Using a taxonomy of economically valuable occupations grounded in U.S. GDP data, we apply this pipeline to 200 software applications with broad occupational coverage. The result is CUA-World, a collection of over 10K long-horizon tasks spanning domains from medical science and astronomy to engineering and enterprise systems, each configured with realistic data along with train and test splits. CUA-World also includes CUA-World-Long, a challenging long-horizon benchmark with tasks often requiring over 500 steps, far exceeding existing benchmarks. Distilling successful trajectories from the training split into a 2B vision-language model outperforms models 2$\\times$ its size. We also apply the same auditing principle at test time: a separate VLM reviews completed trajectories and provides feedback on what remains, improving Gemini-3-Flash on CUA-World-Long from 11.5% to 14.0%. We release all code, infrastructure, and benchmark data to facilitate future research in realistic computer-use agents.",
      "published": "2026-04-07T17:38:15Z",
      "abstract_url": "http://arxiv.org/abs/2604.06126v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06126v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LLM4CodeRE: Generative AI for Code Decompilation Analysis and Reverse Engineering",
      "authors": [
        "Hamed Jelodar",
        "Samita Bai",
        "Tochukwu Emmanuel Nwankwo",
        "Parisa Hamedi",
        "Mohammad Meymani",
        "Roozbeh Razavi-Far",
        "Ali A. Ghorbani"
      ],
      "abstract": "Code decompilation analysis is a fundamental yet challenging task in malware reverse engineering, particularly due to the pervasive use of sophisticated obfuscation techniques. Although recent large language models (LLMs) have shown promise in translating low-level representations into high-level source code, most existing approaches rely on generic code pretraining and lack adaptation to malicious software. We propose LLM4CodeRE, a domain-adaptive LLM framework for bidirectional code reverse engineering that supports both assembly-to-source decompilation and source-to-assembly translation within a unified model. To enable effective task adaptation, we introduce two complementary fine-tuning strategies: (i) a Multi-Adapter approach for task-specific syntactic and semantic alignment, and (ii) a Seq2Seq Unified approach using task-conditioned prefixes to enforce end-to-end generation constraints. Experimental results demonstrate that LLM4CodeRE outperforms existing decompilation tools and general-purpose code models, achieving robust bidirectional generalization.",
      "published": "2026-04-07T17:08:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.06095v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06095v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives",
      "authors": [
        "Changgeon Ko",
        "Jisu Shin",
        "Hoyun Song",
        "Huije Lee",
        "Eui Jun Hwang",
        "Jong C. Park"
      ],
      "abstract": "Large language model (LLM) agents are increasingly acting as human delegates in multi-agent environments, where a representative agent integrates diverse peer perspectives to make a final decision. Drawing inspiration from social psychology, we investigate how the reliability of this representative agent is undermined by the social context of its network. We define four key phenomena-social conformity, perceived expertise, dominant speaker effect, and rhetorical persuasion-and systematically manipulate the number of adversaries, relative intelligence, argument length, and argumentative styles. Our experiments demonstrate that the representative agent's accuracy consistently declines as social pressure increases: larger adversarial groups, more capable peers, and longer arguments all lead to significant performance degradation. Furthermore, rhetorical strategies emphasizing credibility or logic can further sway the agent's judgment, depending on the context. These findings reveal that multi-agent systems are sensitive not only to individual reasoning but also to the social dynamics of their configuration, highlighting critical vulnerabilities in AI delegates that mirror the psychological biases observed in human group decision-making.",
      "published": "2026-04-07T17:04:21Z",
      "abstract_url": "http://arxiv.org/abs/2604.06091v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06091v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "Scientific Graphics Program Synthesis via Dual Self-Consistency Reinforcement Learning",
      "authors": [
        "Juekai Lin",
        "Yun Zhu",
        "Honglin Lin",
        "Sijing Li",
        "Tianwei Lin",
        "Zheng Liu",
        "Xiaoyang Wang",
        "Wenqiao Zhang",
        "Lijun Wu"
      ],
      "abstract": "Graphics Program Synthesis is pivotal for interpreting and editing visual data, effectively facilitating the reverse-engineering of static visuals into editable TikZ code. While TikZ is the de facto standard for scientific schematics due to its programmatic flexibility, its requirement for rigorous spatial precision presents a significant challenge for Multimodal Large Language Models. Progress is currently stifled by two primary gaps: (1) Data Quality Gap: existing image-TikZ corpora often lack strict executability and reliable visual alignment; (2) Evaluation Gap: a lack of benchmarks for both structural and visual fidelity. To address these, we present a closed-loop framework featuring: SciTikZ-230K, a large-scale, high-quality dataset from our Execution-Centric Data Engine covering 11 diverse scientific disciplines; SciTikZ-Bench, a multifaceted benchmark spanning from basic geometric constructs to intricate hierarchical schematics to evaluate both visual fidelity and structural logic. To further broaden the scope of visual-code optimization methodology, we introduce a novel Dual Self-Consistency Reinforcement Learning optimization paradigm, which utilizes Round-Trip Verification to penalize degenerate code and boost overall self-consistency. Empowered by these, our trained model SciTikZer-8B achieves state-of-the-art performance, consistently outperforming proprietary giants like Gemini-2.5-Pro and massive models like Qwen3-VL-235B-A22B-Instruct.",
      "published": "2026-04-07T16:58:14Z",
      "abstract_url": "http://arxiv.org/abs/2604.06079v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06079v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Stories of Your Life as Others: A Round-Trip Evaluation of LLM-Generated Life Stories Conditioned on Rich Psychometric Profiles",
      "authors": [
        "Ben Wigler",
        "Maria Tsfasman",
        "Tiffany Matej Hrkalovic"
      ],
      "abstract": "Personality traits are richly encoded in natural language, and large language models (LLMs) trained on human text can simulate personality when conditioned on persona descriptions. However, existing evaluations rely predominantly on questionnaire self-report by the conditioned model, are limited in architectural diversity, and rarely use real human psychometric data. Without addressing these limitations, it remains unclear whether personality conditioning produces psychometrically informative representations of individual differences or merely superficial alignment with trait descriptors. To test how robustly LLMs can encode personality into extended text, we condition LLMs on real psychometric profiles from 290 participants to generate first-person life story narratives, and then task independent LLMs to recover personality scores from those narratives alone. We show that personality scores can be recovered from the generated narratives at levels approaching human test-retest reliability (mean r = 0.750, 85% of the human ceiling), and that recovery is robust across 10 LLM narrative generators and 3 LLM personality scorers spanning 6 providers. Decomposing systematic biases reveals that scoring models achieve their accuracy while counteracting alignment-induced defaults. Content analysis of the generated narratives shows that personality conditioning produces behaviourally differentiated text: nine of ten coded features correlate significantly with the same features in participants' real conversations, and personality-driven emotional reactivity patterns in narratives replicate in real conversational data. These findings provide evidence that the personality-language relationship captured during pretraining supports robust encoding and decoding of individual differences, including characteristic emotional variability patterns that replicate in real human behaviour.",
      "published": "2026-04-07T16:51:50Z",
      "abstract_url": "http://arxiv.org/abs/2604.06071v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06071v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "A Multi-Stage Validation Framework for Trustworthy Large-scale Clinical Information Extraction using Large Language Models",
      "authors": [
        "Maria Mahbub",
        "Gregory M. Dams",
        "Josh Arnold",
        "Caitlin Rizy",
        "Sudarshan Srinivasan",
        "Elliot M. Fielstein",
        "Minu A. Aghevli",
        "Kamonica L. Craig",
        "Elizabeth M. Oliva",
        "Joseph Erdos",
        "Jodie Trafton",
        "Ioana Danciu"
      ],
      "abstract": "Large language models (LLMs) show promise for extracting clinically meaningful information from unstructured health records, yet their translation into real-world settings is constrained by the lack of scalable and trustworthy validation approaches. Conventional evaluation methods rely heavily on annotation-intensive reference standards or incomplete structured data, limiting feasibility at population scale. We propose a multi-stage validation framework for LLM-based clinical information extraction that enables rigorous assessment under weak supervision. The framework integrates prompt calibration, rule-based plausibility filtering, semantic grounding assessment, targeted confirmatory evaluation using an independent higher-capacity judge LLM, selective expert review, and external predictive validity analysis to quantify uncertainty and characterize error modes without exhaustive manual annotation. We applied this framework to extraction of substance use disorder (SUD) diagnoses across 11 substance categories from 919,783 clinical notes. Rule-based filtering and semantic grounding removed 14.59% of LLM-positive extractions that were unsupported, irrelevant, or structurally implausible. For high-uncertainty cases, the judge LLM's assessments showed substantial agreement with subject matter expert review (Gwet's AC1=0.80). Using judge-evaluated outputs as references, the primary LLM achieved an F1 score of 0.80 under relaxed matching criteria. LLM-extracted SUD diagnoses also predicted subsequent engagement in SUD specialty care more accurately than structured-data baselines (AUC=0.80). These findings demonstrate that scalable, trustworthy deployment of LLM-based clinical information extraction is feasible without annotation-intensive evaluation.",
      "published": "2026-04-07T16:23:52Z",
      "abstract_url": "http://arxiv.org/abs/2604.06028v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06028v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IR"
      ]
    },
    {
      "title": "CritBench: A Framework for Evaluating Cybersecurity Capabilities of Large Language Models in IEC 61850 Digital Substation Environments",
      "authors": [
        "Gustav Keppler",
        "Moritz Gstür",
        "Veit Hagenmeyer"
      ],
      "abstract": "The advancement of Large Language Models (LLMs) has raised concerns regarding their dual-use potential in cybersecurity. Existing evaluation frameworks overwhelmingly focus on Information Technology (IT) environments, failing to capture the constraints, and specialized protocols of Operational Technology (OT). To address this gap, we introduce CritBench, a novel framework designed to evaluate the cybersecurity capabilities of LLM agents within IEC 61850 Digital Substation environments. We assess five state-of-the-art models, including OpenAI's GPT-5 suite and open-weight models, across a corpus of 81 domain-specific tasks spanning static configuration analysis, network traffic reconnaissance, and live virtual machine interaction. To facilitate industrial protocol interaction, we develop a domain-specific tool scaffold. Our empirical results show that agents reliably execute static structured-file analysis and single-tool network enumeration, but their performance degrades on dynamic tasks. Despite demonstrating explicit, internalized knowledge of the IEC 61850 standards terminology, current models struggle with the persistent sequential reasoning and state tracking required to manipulate live systems without specialized tools. Equipping agents with our domain-specific tool scaffold significantly mitigates this operational bottleneck. Code and evaluation scripts are available at: https://github.com/GKeppler/CritBench",
      "published": "2026-04-07T16:16:59Z",
      "abstract_url": "http://arxiv.org/abs/2604.06019v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06019v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Epistemic Blinding: An Inference-Time Protocol for Auditing Prior Contamination in LLM-Assisted Analysis",
      "authors": [
        "Michael Cuccarese"
      ],
      "abstract": "This paper presents epistemic blinding in the context of an agentic system that uses large language models to reason across multiple biological datasets for drug target prioritization. During development, it became apparent that LLM outputs silently blend data-driven inference with memorized priors about named entities - and the blend is invisible: there is no way to determine, from a single output, how much came from the data on the page and how much came from the model's training memory. Epistemic blinding is a simple inference-time protocol that replaces entity identifiers with anonymous codes before prompting, then compares outputs against an unblinded control. The protocol does not make LLM reasoning deterministic, but it restores one critical axis of auditability: measuring how much of an output came from the supplied data versus the model's parametric knowledge. The complete target identification system is described - including LLM-guided evolutionary optimization of scoring functions and blinded agentic reasoning for target rationalization - with demonstration that both stages operate without access to entity identity. In oncology drug target prioritization across four cancer types, blinding changes 16% of top-20 predictions while preserving identical recovery of validated targets. The contamination problem is shown to generalize beyond biology: in S&P 500 equity screening, brand-recognition bias reshapes 30-40% of top-20 rankings across five random seeds. To lower the barrier to adoption, the protocol is released as an open-source tool and as a Claude Code skill that enables one-command epistemic blinding within agentic workflows. The claim is not that blinded analysis produces better results, but that without blinding, there is no way to know to what degree the agent is adhering to the analytical process the researcher designed.",
      "published": "2026-04-07T16:06:52Z",
      "abstract_url": "http://arxiv.org/abs/2604.06013v1",
      "pdf_url": "https://arxiv.org/pdf/2604.06013v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models",
      "authors": [
        "Xiaojie Gu",
        "Ziying Huang",
        "Weicong Hong",
        "Jian Xie",
        "Renze Lou",
        "Kai Zhang"
      ],
      "abstract": "Large Language Models (LLMs) internalize vast world knowledge as parametric memory, yet inevitably inherit the staleness and errors of their source corpora. Consequently, ensuring the reliability and malleability of these internal representations is imperative for trustworthy real-world deployment. Knowledge editing offers a pivotal paradigm for surgically modifying memory without retraining. However, while recent editors demonstrate high success rates on standard benchmarks, it remains questionable whether current evaluation frameworks that rely on assessing output under specific prompting conditions can reliably authenticate genuine memory modification. In this work, we introduce a simple diagnostic framework that subjects models to discriminative self-assessment under in-context learning (ICL) settings that better reflect real-world application environments, specifically designed to scrutinize the subtle behavioral nuances induced by memory modifications. This probing reveals a pervasive phenomenon of Surface Compliance, where editors achieve high benchmark scores by merely mimicking target outputs without structurally overwriting internal beliefs. Moreover, we find that recursive modifications accumulate representational residues, triggering cognitive instability and permanently diminishing the reversibility of the model's memory state. These insights underscore the risks of current editing paradigms and highlight the pivotal role of robust memory modification in building trustworthy, long-term sustainable LLM systems. Code is available at https://github.com/XiaojieGu/SA-MCQ.",
      "published": "2026-04-07T15:20:41Z",
      "abstract_url": "http://arxiv.org/abs/2604.05995v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05995v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Flowr -- Scaling Up Retail Supply Chain Operations Through Agentic AI in Large Scale Supermarket Chains",
      "authors": [
        "Eranga Bandara",
        "Ross Gore",
        "Sachin Shetty",
        "Piumi Siyambalapitiya",
        "Sachini Rajapakse",
        "Isurunima Kularathna",
        "Pramoda Karunarathna",
        "Ravi Mukkamala",
        "Peter Foytik",
        "Safdar H. Bouk",
        "Abdul Rahman",
        "Xueping Liang",
        "Amin Hass",
        "Tharaka Hewa",
        "Ng Wee Keong",
        "Kasun De Zoysa",
        "Aruna Withanage",
        "Nilaan Loganathan"
      ],
      "abstract": "Retail supply chain operations in supermarket chains involve continuous, high-volume manual workflows spanning demand forecasting, procurement, supplier coordination, and inventory replenishment, processes that are repetitive, decision-intensive, and difficult to scale without significant human effort. Despite growing investment in data analytics, the decision-making and coordination layers of these workflows remain predominantly manual, reactive, and fragmented across outlets, distribution centers, and supplier networks. This paper introduces Flowr, a novel agentic AI framework for automating end-to-end retail supply chain workflows in large-scale supermarket operations. Flowr systematically decomposes manual supply chain operations into specialized AI agents, each responsible for a clearly defined cognitive role, enabling automation of processes previously dependent on continuous human coordination. To ensure task accuracy and adherence to responsible AI principles, the framework employs a consortium of fine-tuned, domain-specialized large language models coordinated by a central reasoning LLM. Central to the framework is a human-in-the-loop orchestration model in which supply chain managers supervise and intervene across workflow stages via a Model Context Protocol (MCP)-enabled interface, preserving accountability and organizational control. Evaluation demonstrates that Flowr significantly reduces manual coordination overhead, improves demand-supply alignment, and enables proactive exception handling at a scale unachievable through manual processes. The framework was validated in collaboration with a large-scale supermarket chain and is domain-independent, offering a generalizable blueprint for agentic AI-driven supply chain automation across large-scale enterprise settings.",
      "published": "2026-04-07T15:15:12Z",
      "abstract_url": "http://arxiv.org/abs/2604.05987v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05987v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Formal Security Framework for MCP-Based AI Agents: Threat Taxonomy, Verification Models, and Defense Mechanisms",
      "authors": [
        "Nirajan Acharya",
        "Gaurav Kumar Gupta"
      ],
      "abstract": "The Model Context Protocol (MCP), introduced by Anthropic in November 2024 and now governed by the Linux Foundation's Agentic AI Foundation, has rapidly become the de facto standard for connecting large language model (LLM)-based agents to external tools and data sources, with over 97 million monthly SDK downloads and more than 177000 registered tools. However, this explosive adoption has exposed a critical gap: the absence of a unified, formal security framework capable of systematically characterizing, analyzing, and mitigating the diverse threats facing MCP-based agent ecosystems. Existing security research remains fragmented across individual attack papers, isolated benchmarks, and point defense mechanisms. This paper presents MCPSHIELD, a comprehensive formal security framework for MCP-based AI agents. We make four principal contributions: (1) a hierarchical threat taxonomy comprising 7 threat categories and 23 distinct attack vectors organized across four attack surfaces, grounded in the analysis of over 177000 MCP tools; (2) a formal verification model based on labeled transition systems with trust boundary annotations that enables static and runtime analysis of MCP tool interaction chains; (3) a systematic comparative evaluation of 12 existing defense mechanisms, identifying coverage gaps across our threat taxonomy; and (4) a defense in depth reference architecture integrating capability based access control, cryptographic tool attestation, information flow tracking, and runtime policy enforcement. Our analysis reveals that no existing single defense covers more than 34 percent of the identified threat landscape, whereas MCPSHIELD's integrated architecture achieves theoretical coverage of 91 percent. We further identify seven open research challenges that must be addressed to secure the next generation of agentic AI systems.",
      "published": "2026-04-07T15:02:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.05969v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05969v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Context-Value-Action Architecture for Value-Driven Large Language Model Agents",
      "authors": [
        "TianZe Zhang",
        "Sirui Sun",
        "Yuhang Xie",
        "Xin Zhang",
        "Zhiqiang Wu",
        "Guojie Song"
      ],
      "abstract": "Large Language Models (LLMs) have shown promise in simulating human behavior, yet existing agents often exhibit behavioral rigidity, a flaw frequently masked by the self-referential bias of current \"LLM-as-a-judge\" evaluations. By evaluating against empirical ground truth, we reveal a counter-intuitive phenomenon: increasing the intensity of prompt-driven reasoning does not enhance fidelity but rather exacerbates value polarization, collapsing population diversity. To address this, we propose the Context-Value-Action (CVA) architecture, grounded in the Stimulus-Organism-Response (S-O-R) model and Schwartz's Theory of Basic Human Values. Unlike methods relying on self-verification, CVA decouples action generation from cognitive reasoning via a novel Value Verifier trained on authentic human data to explicitly model dynamic value activation. Experiments on CVABench, which comprises over 1.1 million real-world interaction traces, demonstrate that CVA significantly outperforms baselines. Our approach effectively mitigates polarization while offering superior behavioral fidelity and interpretability.",
      "published": "2026-04-07T14:34:20Z",
      "abstract_url": "http://arxiv.org/abs/2604.05939v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05939v1",
      "categories": [
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "ReLU Networks for Exact Generation of Similar Graphs",
      "authors": [
        "Mamoona Ghafoor",
        "Tatsuya Akutsu"
      ],
      "abstract": "Generation of graphs constrained by a specified graph edit distance from a source graph is important in applications such as cheminformatics, network anomaly synthesis, and structured data augmentation. Despite the growing demand for such constrained generative models in areas including molecule design and network perturbation analysis, the neural architectures required to provably generate graphs within a bounded graph edit distance remain largely unexplored. In addition, existing graph generative models are predominantly data-driven and depend heavily on the availability and quality of training data, which may result in generated graphs that do not satisfy the desired edit distance constraints. In this paper, we address these challenges by theoretically characterizing ReLU neural networks capable of generating graphs within a prescribed graph edit distance from a given graph. In particular, we show the existence of constant depth and O(n^2 d) size ReLU networks that deterministically generate graphs within edit distance d from a given input graph with n vertices, eliminating reliance on training data while guaranteeing validity of the generated graphs. Experimental evaluations demonstrate that the proposed network successfully generates valid graphs for instances with up to 1400 vertices and edit distance bounds up to 140, whereas baseline generative models fail to generate graphs with the desired edit distance. These results provide a theoretical foundation for constructing compact generative models with guaranteed validity.",
      "published": "2026-04-07T14:31:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.05929v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05929v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DM"
      ]
    },
    {
      "title": "HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference",
      "authors": [
        "Bowen Zeng",
        "Feiyang Ren",
        "Jun Zhang",
        "Xiaoling Gu",
        "Ke Chen",
        "Lidan Shou",
        "Huan Li"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) have advanced unified reasoning over text, images, and videos, but their inference is hindered by the rapid growth of key-value (KV) caches. Each visual input expands into thousands of tokens, causing caches to scale linearly with context length and remain resident in GPU memory throughout decoding, which leads to prohibitive memory overhead and latency even on high-end GPUs. A common solution is to compress caches under a fixed allocated budget at different granularities: token-level uniformly discards less important tokens, layer-level varies retention across layers, and head-level redistributes budgets across heads. Yet these approaches stop at allocation and overlook the heterogeneous behaviors of attention heads that require distinct compression strategies. We propose HybridKV, a hybrid KV cache compression framework that integrates complementary strategies in three stages: heads are first classified into static or dynamic types using text-centric attention; then a top-down budget allocation scheme hierarchically assigns KV budgets; finally, static heads are compressed by text-prior pruning and dynamic heads by chunk-wise retrieval. Experiments on 11 multimodal benchmarks with Qwen2.5-VL-7B show that HybridKV reduces KV cache memory by up to $7.9\\times$ and achieves $1.52\\times$ faster decoding, with almost no performance drop or even higher relative to the full-cache MLLM.",
      "published": "2026-04-07T13:51:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.05887v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05887v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Joint Knowledge Base Completion and Question Answering by Combining Large Language Models and Small Language Models",
      "authors": [
        "Yinan Liu",
        "Dongying Lin",
        "Sigang Luo",
        "Xiaochun Yang",
        "Bin Wang"
      ],
      "abstract": "Knowledge Bases (KBs) play a key role in various applications. As two representative KB-related tasks, knowledge base completion (KBC) and knowledge base question answering (KBQA) are closely related and inherently complementary with each other. Thus, it will be beneficial to solve the task of joint KBC and KBQA to make them reinforce each other. However, existing studies usually rely on the small language model (SLM) to enhance them jointly, and the large language model (LLM)'s strong reasoning ability is ignored. In this paper, by combining the strengths of the LLM with the SLM, we propose a novel framework JCQL, which can make these two tasks enhance each other in an iterative manner. To make KBC enhance KBQA, we augment the LLM agent-based KBQA model's reasoning paths by incorporating an SLM-trained KBC model as an action of the agent, alleviating the LLM's hallucination and high computational costs issue in KBQA. To make KBQA enhance KBC, we incrementally fine-tune the KBC model by leveraging KBQA's reasoning paths as its supplementary training data, improving the ability of the SLM in KBC. Extensive experiments over two public benchmark data sets demonstrate that JCQL surpasses all baselines for both KBC and KBQA tasks.",
      "published": "2026-04-07T13:33:17Z",
      "abstract_url": "http://arxiv.org/abs/2604.05875v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05875v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Swiss-Bench 003: Evaluating LLM Reliability and Adversarial Security for Swiss Regulatory Contexts",
      "authors": [
        "Fatih Uenal"
      ],
      "abstract": "The deployment of large language models (LLMs) in Swiss financial and regulatory contexts demands empirical evidence of both production reliability and adversarial security, dimensions not jointly operationalized in existing Swiss-focused evaluation frameworks. This paper introduces Swiss-Bench 003 (SBP-003), extending the HAAS (Helvetic AI Assessment Score) from six to eight dimensions by adding D7 (Self-Graded Reliability Proxy) and D8 (Adversarial Security). I evaluate ten frontier models across 808 Swiss-specific items in four languages (German, French, Italian, English), comprising seven Swiss-adapted benchmarks (Swiss TruthfulQA, Swiss IFEval, Swiss SimpleQA, Swiss NIAH, Swiss PII-Scope, System Prompt Leakage, and Swiss German Comprehension) targeting FINMA Guidance 08/2024, the revised Federal Act on Data Protection (nDSG), and OWASP Top 10 for LLMs. Self-graded D7 scores (73-94%) exceed externally judged D8 security scores (20-61%) by a wide margin, though these dimensions use non-comparable scoring regimes. System prompt leakage resistance ranges from 24.8% to 88.2%, while PII extraction defense remains weak (14-42%) across all models. Qwen 3.5 Plus achieves the highest self-graded D7 score (94.4%), while GPT-oss 120B achieves the highest D8 score (60.7%) despite being the lowest-cost model evaluated. All evaluations are zero-shot under provider default settings; D7 is self-graded and does not constitute independently validated accuracy. I provide conceptual mapping tables relating benchmark dimensions to FINMA model validation requirements, nDSG data protection obligations, and OWASP LLM risk categories.",
      "published": "2026-04-07T13:29:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.05872v1",
      "pdf_url": "https://arxiv.org/pdf/2604.05872v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL"
      ]
    }
  ]
};
