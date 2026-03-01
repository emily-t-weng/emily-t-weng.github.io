const PAPERS_DATA = {
  "last_updated": "2026-03-01 03:14:57 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Model Agreement via Anchoring",
      "authors": [
        "Eric Eaton",
        "Surbhi Goel",
        "Marcel Hussing",
        "Michael Kearns",
        "Aaron Roth",
        "Sikata Bela Sengupta",
        "Jessica Sorrell"
      ],
      "abstract": "Numerous lines of aim to control $\\textit{model disagreement}$ -- the extent to which two machine learning models disagree in their predictions. We adopt a simple and standard notion of model disagreement in real-valued prediction problems, namely the expected squared difference in predictions between two models trained on independent samples, without any coordination of the training processes. We would like to be able to drive disagreement to zero with some natural parameter(s) of the training procedure using analyses that can be applied to existing training methodologies. We develop a simple general technique for proving bounds on independent model disagreement based on $\\textit{anchoring}$ to the average of two models within the analysis. We then apply this technique to prove disagreement bounds for four commonly used machine learning algorithms: (1) stacked aggregation over an arbitrary model class (where disagreement is driven to 0 with the number of models $k$ being stacked) (2) gradient boosting (where disagreement is driven to 0 with the number of iterations $k$) (3) neural network training with architecture search (where disagreement is driven to 0 with the size $n$ of the architecture being optimized over) and (4) regression tree training over all regression trees of fixed depth (where disagreement is driven to 0 with the depth $d$ of the tree architecture). For clarity, we work out our initial bounds in the setting of one-dimensional regression with squared error loss -- but then show that all of our results generalize to multi-dimensional regression with any strongly convex loss.",
      "published": "2026-02-26T18:59:32Z",
      "abstract_url": "http://arxiv.org/abs/2602.23360v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23360v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport",
      "authors": [
        "Simon Roschmann",
        "Paul Krzakala",
        "Sonia Mazelet",
        "Quentin Bouniot",
        "Zeynep Akata"
      ],
      "abstract": "The Platonic Representation Hypothesis posits that neural networks trained on different modalities converge toward a shared statistical model of the world. Recent work exploits this convergence by aligning frozen pretrained vision and language models with lightweight alignment layers, but typically relies on contrastive losses and millions of paired samples. In this work, we ask whether meaningful alignment can be achieved with substantially less supervision. We introduce a semi-supervised setting in which pretrained unimodal encoders are aligned using a small number of image-text pairs together with large amounts of unpaired data. To address this challenge, we propose SOTAlign, a two-stage framework that first recovers a coarse shared geometry from limited paired data using a linear teacher, then refines the alignment on unpaired samples via an optimal-transport-based divergence that transfers relational structure without overconstraining the target space. Unlike existing semi-supervised methods, SOTAlign effectively leverages unpaired images and text, learning robust joint embeddings across datasets and encoder pairs, and significantly outperforming supervised and semi-supervised baselines.",
      "published": "2026-02-26T18:55:06Z",
      "abstract_url": "http://arxiv.org/abs/2602.23353v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23353v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "FlashOptim: Optimizers for Memory Efficient Training",
      "authors": [
        "Jose Javier Gonzalez Ortiz",
        "Abhay Gupta",
        "Chris Renard",
        "Davis Blalock"
      ],
      "abstract": "Standard mixed-precision training of neural networks requires many bytes of accelerator memory for each model parameter. These bytes reflect not just the parameter itself, but also its gradient and one or more optimizer state variables. With each of these values typically requiring 4 bytes, training even a 7 billion parameter model can be impractical for researchers with less than 100GB of accelerator memory. We introduce FlashOptim, a suite of optimizations that reduces per-parameter memory by over 50% while preserving model quality and API compatibility. Our approach introduces two key techniques. First, we improve master weight splitting by finding and exploiting a tight bound on its quantization error. Second, we design companding functions that greatly reduce the error in 8-bit optimizer state quantization. Together with 16-bit gradients, these techniques reduce AdamW memory from 16 bytes to 7 bytes per parameter, or 5 bytes with gradient release. They also cut model checkpoint sizes by more than half. Experiments with FlashOptim applied to SGD, AdamW, and Lion show no measurable quality degradation on any task from a collection of standard vision and language benchmarks, including Llama-3.1-8B finetuning.",
      "published": "2026-02-26T18:52:22Z",
      "abstract_url": "http://arxiv.org/abs/2602.23349v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23349v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Utilizing LLMs for Industrial Process Automation",
      "authors": [
        "Salim Fares"
      ],
      "abstract": "A growing number of publications address the best practices to use Large Language Models (LLMs) for software engineering in recent years. However, most of this work focuses on widely-used general purpose programming languages like Python due to their widespread usage training data. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, remains underexplored. This research aims to utilize and integrate LLMs in the industrial development process, solving real-life programming tasks (e.g., generating a movement routine for a robotic arm) and accelerating the development cycles of manufacturing systems.",
      "published": "2026-02-26T18:38:00Z",
      "abstract_url": "http://arxiv.org/abs/2602.23331v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23331v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Toward Expert Investment Teams:A Multi-Agent LLM System with Fine-Grained Trading Tasks",
      "authors": [
        "Kunihiro Miyazaki",
        "Takanobu Kawahara",
        "Stephen Roberts",
        "Stefan Zohren"
      ],
      "abstract": "The advancement of large language models (LLMs) has accelerated the development of autonomous financial trading systems. While mainstream approaches deploy multi-agent systems mimicking analyst and manager roles, they often rely on abstract instructions that overlook the intricacies of real-world workflows, which can lead to degraded inference performance and less transparent decision-making. Therefore, we propose a multi-agent LLM trading framework that explicitly decomposes investment analysis into fine-grained tasks, rather than providing coarse-grained instructions. We evaluate the proposed framework using Japanese stock data, including prices, financial statements, news, and macro information, under a leakage-controlled backtesting setting. Experimental results show that fine-grained task decomposition significantly improves risk-adjusted returns compared to conventional coarse-grained designs. Crucially, further analysis of intermediate agent outputs suggests that alignment between analytical outputs and downstream decision preferences is a critical driver of system performance. Moreover, we conduct standard portfolio optimization, exploiting low correlation with the stock index and the variance of each system's output. This approach achieves superior performance. These findings contribute to the design of agent structure and task configuration when applying LLM agents to trading systems in practical settings.",
      "published": "2026-02-26T18:37:36Z",
      "abstract_url": "http://arxiv.org/abs/2602.23330v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23330v1",
      "categories": [
        "cs.AI",
        "q-fin.TR"
      ]
    },
    {
      "title": "LLM Novice Uplift on Dual-Use, In Silico Biology Tasks",
      "authors": [
        "Chen Bo Calvin Zhang",
        "Christina Q. Knight",
        "Nicholas Kruus",
        "Jason Hausenloy",
        "Pedro Medeiros",
        "Nathaniel Li",
        "Aiden Kim",
        "Yury Orlovskiy",
        "Coleman Breen",
        "Bryce Cai",
        "Jasper Götting",
        "Andrew Bo Liu",
        "Samira Nedungadi",
        "Paula Rodriguez",
        "Yannis Yiming He",
        "Mohamed Shaaban",
        "Zifan Wang",
        "Seth Donoughe",
        "Julian Michael"
      ],
      "abstract": "Large language models (LLMs) perform increasingly well on biology benchmarks, but it remains unclear whether they uplift novice users -- i.e., enable humans to perform better than with internet-only resources. This uncertainty is central to understanding both scientific acceleration and dual-use risk. We conducted a multi-model, multi-benchmark human uplift study comparing novices with LLM access versus internet-only access across eight biosecurity-relevant task sets. Participants worked on complex problems with ample time (up to 13 hours for the most involved tasks). We found that LLM access provided substantial uplift: novices with LLMs were 4.16 times more accurate than controls (95% CI [2.63, 6.87]). On four benchmarks with available expert baselines (internet-only), novices with LLMs outperformed experts on three of them. Perhaps surprisingly, standalone LLMs often exceeded LLM-assisted novices, indicating that users were not eliciting the strongest available contributions from the LLMs. Most participants (89.6%) reported little difficulty obtaining dual-use-relevant information despite safeguards. Overall, LLMs substantially uplift novices on biological tasks previously reserved for trained practitioners, underscoring the need for sustained, interactive uplift evaluations alongside traditional benchmarks.",
      "published": "2026-02-26T18:37:23Z",
      "abstract_url": "http://arxiv.org/abs/2602.23329v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23329v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CR",
        "cs.CY",
        "cs.HC"
      ]
    },
    {
      "title": "Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction",
      "authors": [
        "Rafael R. Baptista",
        "André de Lima Salgado",
        "Ricardo V. Godoy",
        "Marcelo Becker",
        "Thiago Boaventura",
        "Gustavo J. G. Lahr"
      ],
      "abstract": "Leader-follower interaction is an important paradigm in human-robot interaction (HRI). Yet, assigning roles in real time remains challenging for resource-constrained mobile and assistive robots. While large language models (LLMs) have shown promise for natural communication, their size and latency limit on-device deployment. Small language models (SLMs) offer a potential alternative, but their effectiveness for role classification in HRI has not been systematically evaluated. In this paper, we present a benchmark of SLMs for leader-follower communication, introducing a novel dataset derived from a published database and augmented with synthetic samples to capture interaction-specific dynamics. We investigate two adaptation strategies: prompt engineering and fine-tuning, studied under zero-shot and one-shot interaction modes, compared with an untrained baseline. Experiments with Qwen2.5-0.5B reveal that zero-shot fine-tuning achieves robust classification performance (86.66% accuracy) while maintaining low latency (22.2 ms per sample), significantly outperforming baseline and prompt-engineered approaches. However, results also indicate a performance degradation in one-shot modes, where increased context length challenges the model's architectural capacity. These findings demonstrate that fine-tuned SLMs provide an effective solution for direct role assignment, while highlighting critical trade-offs between dialogue complexity and classification reliability on the edge.",
      "published": "2026-02-26T18:20:26Z",
      "abstract_url": "http://arxiv.org/abs/2602.23312v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23312v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "cs.LG",
        "cs.RO",
        "eess.SY"
      ]
    },
    {
      "title": "Conformalized Neural Networks for Federated Uncertainty Quantification under Dual Heterogeneity",
      "authors": [
        "Quang-Huy Nguyen",
        "Jiaqi Wang",
        "Wei-Shinn Ku"
      ],
      "abstract": "Federated learning (FL) faces challenges in uncertainty quantification (UQ). Without reliable UQ, FL systems risk deploying overconfident models at under-resourced agents, leading to silent local failures despite seemingly satisfactory global performance. Existing federated UQ approaches often address data heterogeneity or model heterogeneity in isolation, overlooking their joint effect on coverage reliability across agents. Conformal prediction is a widely used distribution-free UQ framework, yet its applications in heterogeneous FL settings remains underexplored. We provide FedWQ-CP, a simple yet effective approach that balances empirical coverage performance with efficiency at both global and agent levels under the dual heterogeneity. FedWQ-CP performs agent-server calibration in a single communication round. On each agent, conformity scores are computed on calibration data and a local quantile threshold is derived. Each agent then transmits only its quantile threshold and calibration sample size to the server. The server simply aggregates these thresholds through a weighted average to produce a global threshold. Experimental results on seven public datasets for both classification and regression demonstrate that FedWQ-CP empirically maintains agent-wise and global coverage while producing the smallest prediction sets or intervals.",
      "published": "2026-02-26T18:07:45Z",
      "abstract_url": "http://arxiv.org/abs/2602.23296v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23296v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays",
      "authors": [
        "Hyungyung Lee",
        "Hangyul Yoon",
        "Edward Choi"
      ],
      "abstract": "Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings.",
      "published": "2026-02-26T17:51:21Z",
      "abstract_url": "http://arxiv.org/abs/2602.23276v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23276v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Mitigating Legibility Tax with Decoupled Prover-Verifier Games",
      "authors": [
        "Yegon Kim",
        "Juho Lee"
      ],
      "abstract": "As large language models become increasingly capable, it is critical that their outputs can be easily checked by less capable systems. Prover-verifier games can be used to improve checkability of model outputs, but display a degradation in accuracy compared to a baseline trained only to maximize correctness -- a phenonemon named legibility tax. We propose a solution by decoupling the correctness from the checkability condition and instead training a \"translator\" model that turns a fixed solver model's solution into a checkable form. This allows us to first train the solver to maximize correctness, and then train the translator to translate the solver into a checkable form while retaining the solver's answer. To accommodate this new objective of translation, we formulate a decoupled prover-verifier game where the equilibria correspond to faithful and checkable translators.",
      "published": "2026-02-26T17:25:22Z",
      "abstract_url": "http://arxiv.org/abs/2602.23248v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23248v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Agency and Architectural Limits: Why Optimization-Based Systems Cannot Be Norm-Responsive",
      "authors": [
        "Radha Sarma"
      ],
      "abstract": "AI systems are increasingly deployed in high-stakes contexts -- medical diagnosis, legal research, financial analysis -- under the assumption they can be governed by norms. This paper demonstrates that assumption is formally invalid for optimization-based systems, specifically Large Language Models trained via Reinforcement Learning from Human Feedback (RLHF). We establish that genuine agency requires two necessary and jointly sufficient architectural conditions: the capacity to maintain certain boundaries as non-negotiable constraints rather than tradeable weights (Incommensurability), and a non-inferential mechanism capable of suspending processing when those boundaries are threatened (Apophatic Responsiveness). These conditions apply across all normative domains. RLHF-based systems are constitutively incompatible with both conditions. The operations that make optimization powerful -- unifying all values on a scalar metric and always selecting the highest-scoring output -- are precisely the operations that preclude normative governance. This incompatibility is not a correctable training bug awaiting a technical fix; it is a formal constraint inherent to what optimization is. Consequently, documented failure modes - sycophancy, hallucination, and unfaithful reasoning - are not accidents but structural manifestations. Misaligned deployment triggers a second-order risk we term the Convergence Crisis: when humans are forced to verify AI outputs under metric pressure, they degrade from genuine agents into criteria-checking optimizers, eliminating the only component in the system capable of normative accountability. Beyond the incompatibility proof, the paper's primary positive contribution is a substrate-neutral architectural specification defining what any system -- biological, artificial, or institutional -- must satisfy to qualify as an agent rather than a sophisticated instrument.",
      "published": "2026-02-26T17:16:17Z",
      "abstract_url": "http://arxiv.org/abs/2602.23239v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23239v1",
      "categories": [
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Scaling Search Relevance: Augmenting App Store Ranking with LLM-Generated Judgments",
      "authors": [
        "Evangelia Christakopoulou",
        "Vivekkumar Patel",
        "Hemanth Velaga",
        "Sandip Gaikwad"
      ],
      "abstract": "Large-scale commercial search systems optimize for relevance to drive successful sessions that help users find what they are looking for. To maximize relevance, we leverage two complementary objectives: behavioral relevance (results users tend to click or download) and textual relevance (a result's semantic fit to the query). A persistent challenge is the scarcity of expert-provided textual relevance labels relative to abundant behavioral relevance labels. We first address this by systematically evaluating LLM configurations, finding that a specialized, fine-tuned model significantly outperforms a much larger pre-trained one in providing highly relevant labels. Using this optimal model as a force multiplier, we generate millions of textual relevance labels to overcome the data scarcity. We show that augmenting our production ranker with these textual relevance labels leads to a significant outward shift of the Pareto frontier: offline NDCG improves for behavioral relevance while simultaneously increasing for textual relevance. These offline gains were validated by a worldwide A/B test on the App Store ranker, which demonstrated a statistically significant +0.24% increase in conversion rate, with the most substantial performance gains occurring in tail queries, where the new textual relevance labels provide a robust signal in the absence of reliable behavioral relevance labels.",
      "published": "2026-02-26T17:11:26Z",
      "abstract_url": "http://arxiv.org/abs/2602.23234v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23234v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SC-Arena: A Natural Language Benchmark for Single-Cell Reasoning with Knowledge-Augmented Evaluation",
      "authors": [
        "Jiahao Zhao",
        "Feng Jiang",
        "Shaowei Qin",
        "Zhonghui Zhang",
        "Junhao Liu",
        "Guibing Guo",
        "Hamid Alinejad-Rokny",
        "Min Yang"
      ],
      "abstract": "Large language models (LLMs) are increasingly applied in scientific research, offering new capabilities for knowledge discovery and reasoning. In single-cell biology, however, evaluation practices for both general and specialized LLMs remain inadequate: existing benchmarks are fragmented across tasks, adopt formats such as multiple-choice classification that diverge from real-world usage, and rely on metrics lacking interpretability and biological grounding. We present SC-ARENA, a natural language evaluation framework tailored to single-cell foundation models. SC-ARENA formalizes a virtual cell abstraction that unifies evaluation targets by representing both intrinsic attributes and gene-level interactions. Within this paradigm, we define five natural language tasks (cell type annotation, captioning, generation, perturbation prediction, and scientific QA) that probe core reasoning capabilities in cellular biology. To overcome the limitations of brittle string-matching metrics, we introduce knowledge-augmented evaluation, which incorporates external ontologies, marker databases, and scientific literature to support biologically faithful and interpretable judgments. Experiments and analysis across both general-purpose and domain-specialized LLMs demonstrate that (i) under the Virtual Cell unified evaluation paradigm, current models achieve uneven performance on biologically complex tasks, particularly those demanding mechanistic or causal understanding; and (ii) our knowledge-augmented evaluation framework ensures biological correctness, provides interpretable, evidence-grounded rationales, and achieves high discriminative capacity, overcoming the brittleness and opacity of conventional metrics. SC-Arena thus provides a unified and interpretable framework for assessing LLMs in single-cell biology, pointing toward the development of biology-aligned, generalizable foundation models.",
      "published": "2026-02-26T16:50:28Z",
      "abstract_url": "http://arxiv.org/abs/2602.23199v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23199v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ESAA: Event Sourcing for Autonomous Agents in LLM-Based Software Engineering",
      "authors": [
        "Elzo Brito dos Santos Filho"
      ],
      "abstract": "Autonomous agents based on Large Language Models (LLMs) have evolved from reactive assistants to systems capable of planning, executing actions via tools, and iterating over environment observations. However, they remain vulnerable to structural limitations: lack of native state, context degradation over long horizons, and the gap between probabilistic generation and deterministic execution requirements. This paper presents the ESAA (Event Sourcing for Autonomous Agents) architecture, which separates the agent's cognitive intention from the project's state mutation, inspired by the Event Sourcing pattern. In ESAA, agents emit only structured intentions in validated JSON (agent.result or issue.report); a deterministic orchestrator validates, persists events in an append-only log (activity.jsonl), applies file-writing effects, and projects a verifiable materialized view (roadmap.json). The proposal incorporates boundary contracts (AGENT_CONTRACT.yaml), metaprompting profiles (PARCER), and replay verification with hashing (esaa verify), ensuring the immutability of completed tasks and forensic traceability. Two case studies validate the architecture: (i) a landing page project (9 tasks, 49 events, single-agent composition) and (ii) a clinical dashboard system (50 tasks, 86 events, 4 concurrent agents across 8 phases), both concluding with run.status=success and verify_status=ok. The multi-agent case study demonstrates real concurrent orchestration with heterogeneous LLMs (Claude Sonnet 4.6, Codex GPT-5, Antigravity/Gemini 3 Pro, and Claude Opus 4.6), providing empirical evidence of the architecture's scalability beyond single-agent scenarios.",
      "published": "2026-02-26T16:45:59Z",
      "abstract_url": "http://arxiv.org/abs/2602.23193v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23193v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring",
      "authors": [
        "Usman Anwar",
        "Julianna Piskorz",
        "David D. Baek",
        "David Africa",
        "Jim Weatherall",
        "Max Tegmark",
        "Christian Schroeder de Witt",
        "Mihaela van der Schaar",
        "David Krueger"
      ],
      "abstract": "Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \\textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \\textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.",
      "published": "2026-02-26T16:27:24Z",
      "abstract_url": "http://arxiv.org/abs/2602.23163v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23163v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CR",
        "cs.IT",
        "cs.MA"
      ]
    },
    {
      "title": "Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs",
      "authors": [
        "Jayadev Billa"
      ],
      "abstract": "Multimodal LLMs can process speech and images, but they cannot hear a speaker's voice or see an object's texture. We show this is not a failure of encoding: speaker identity, emotion, and visual attributes survive through every LLM layer (3--55$\\times$ above chance in linear probes), yet removing 64--71% of modality-specific variance improves decoder loss. The decoder has no learned use for these directions; their presence is noise. We formalize this as a mismatched decoder problem: a decoder trained on text can only extract information along text-aligned directions. Accessible information is bounded by the Generalized Mutual Information (GMI), with degradation scaling with distributional distance and decoder sensitivity. The bound is a property of the decoder's scoring rule, not of any particular architecture; it applies whether non-text inputs arrive through a learned projection, a discrete codebook, or no explicit adapter at all. We validate this across five models spanning speech and vision. A controlled experiment (two Prismatic VLMs differing only in encoder text-alignment) confirms the bottleneck is the decoder's scoring rule, not the encoder or projection. A LoRA intervention demonstrates the fix: training with an emotion objective improves emotion accessibility ($+$7.5%) without affecting other attributes, confirming that the training objective determines what becomes accessible.",
      "published": "2026-02-26T15:52:48Z",
      "abstract_url": "http://arxiv.org/abs/2602.23136v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23136v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DyGnROLE: Modeling Asymmetry in Dynamic Graphs with Node-Role-Oriented Latent Encoding",
      "authors": [
        "Tyler Bonnet",
        "Marek Rei"
      ],
      "abstract": "Real-world dynamic graphs are often directed, with source and destination nodes exhibiting asymmetrical behavioral patterns and temporal dynamics. However, existing dynamic graph architectures largely rely on shared parameters for processing source and destination nodes, with limited or no systematic role-aware modeling. We propose DyGnROLE (Dynamic Graph Node-Role-Oriented Latent Encoding), a transformer-based architecture that explicitly disentangles source and destination representations. By using separate embedding vocabularies and role-semantic positional encodings, the model captures the distinct structural and temporal contexts unique to each role. Critical to the effectiveness of these specialized embeddings in low-label regimes is a self-supervised pretraining objective we introduce: Temporal Contrastive Link Prediction (TCLP). The pretraining uses the full unlabeled interaction history to encode informative structural biases, enabling the model to learn role-specific representations without requiring annotated data. Evaluation on future edge classification demonstrates that DyGnROLE substantially outperforms a diverse set of state-of-the-art baselines, establishing role-aware modeling as an effective strategy for dynamic graph learning.",
      "published": "2026-02-26T15:51:51Z",
      "abstract_url": "http://arxiv.org/abs/2602.23135v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23135v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.SI"
      ]
    },
    {
      "title": "Multi-Agent Large Language Model Based Emotional Detoxification Through Personalized Intensity Control for Consumer Protection",
      "authors": [
        "Keito Inoshita"
      ],
      "abstract": "In the attention economy, sensational content exposes consumers to excessive emotional stimulation, hindering calm decision-making. This study proposes Multi-Agent LLM-based Emotional deToxification (MALLET), a multi-agent information sanitization system consisting of four agents: Emotion Analysis, Emotion Adjustment, Balance Monitoring, and Personal Guide. The Emotion Analysis Agent quantifies stimulus intensity using a 6-emotion BERT classifier, and the Emotion Adjustment Agent rewrites texts into two presentation modes, BALANCED (neutralized text) and COOL (neutralized text + supplementary text), using an LLM. The Balance Monitoring Agent aggregates weekly information consumption patterns and generates personalized advice, while the Personal Guide Agent recommends a presentation mode according to consumer sensitivity. Experiments on 800 AG News articles demonstrated significant stimulus score reduction (up to 19.3%) and improved emotion balance while maintaining semantic preservation. Near-zero correlation between stimulus reduction and semantic preservation confirmed that the two are independently controllable. Category-level analysis revealed substantial reduction (17.8-33.8%) in Sports, Business, and Sci/Tech, whereas the effect was limited in the World category, where facts themselves are inherently high-stimulus. The proposed system provides a framework for supporting calm information reception of consumers without restricting access to the original text.",
      "published": "2026-02-26T15:37:03Z",
      "abstract_url": "http://arxiv.org/abs/2602.23123v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23123v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design",
      "authors": [
        "Zhuoliang Xie",
        "Fei Liu",
        "Zhenkun Wang",
        "Qingfu Zhang"
      ],
      "abstract": "The Capacitated Vehicle Routing Problem (CVRP), a fundamental combinatorial optimization challenge, focuses on optimizing fleet operations under vehicle capacity constraints. While extensively studied in operational research, the NP-hard nature of CVRP continues to pose significant computational challenges, particularly for large-scale instances. This study presents AILS-AHD (Adaptive Iterated Local Search with Automatic Heuristic Design), a novel approach that leverages Large Language Models (LLMs) to revolutionize CVRP solving. Our methodology integrates an evolutionary search framework with LLMs to dynamically generate and optimize ruin heuristics within the AILS method. Additionally, we introduce an LLM-based acceleration mechanism to enhance computational efficiency. Comprehensive experimental evaluations against state-of-the-art solvers, including AILS-II and HGS, demonstrate the superior performance of AILS-AHD across both moderate and large-scale instances. Notably, our approach establishes new best-known solutions for 8 out of 10 instances in the CVRPLib large-scale benchmark, underscoring the potential of LLM-driven heuristic design in advancing the field of vehicle routing optimization.",
      "published": "2026-02-26T15:12:23Z",
      "abstract_url": "http://arxiv.org/abs/2602.23092v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23092v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "MoDora: Tree-Based Semi-Structured Document Analysis System",
      "authors": [
        "Bangrui Xu",
        "Qihang Yao",
        "Zirui Tang",
        "Xuanhe Zhou",
        "Yeye He",
        "Shihan Yu",
        "Qianqian Xu",
        "Bin Wang",
        "Guoliang Li",
        "Conghui He",
        "Fan Wu"
      ],
      "abstract": "Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document. To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at https://github.com/weAIDB/MoDora.",
      "published": "2026-02-26T14:48:49Z",
      "abstract_url": "http://arxiv.org/abs/2602.23061v1",
      "pdf_url": "https://arxiv.org/pdf/2602.23061v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CL",
        "cs.DB",
        "cs.LG"
      ]
    }
  ]
};
