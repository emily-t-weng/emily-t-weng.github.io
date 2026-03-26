const PAPERS_DATA = {
  "last_updated": "2026-03-26 03:16:13 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Retrieval Improvements Do Not Guarantee Better Answers: A Study of RAG for AI Policy QA",
      "authors": [
        "Saahil Mathur",
        "Ryan David Rittner",
        "Vedant Ajit Thakur",
        "Daniel Stuart Schiff",
        "Tunazzina Islam"
      ],
      "abstract": "Retrieval-augmented generation (RAG) systems are increasingly used to analyze complex policy documents, but achieving sufficient reliability for expert usage remains challenging in domains characterized by dense legal language and evolving, overlapping regulatory frameworks. We study the application of RAG to AI governance and policy analysis using the AI Governance and Regulatory Archive (AGORA) corpus, a curated collection of 947 AI policy documents. Our system combines a ColBERT-based retriever fine-tuned with contrastive learning and a generator aligned to human preferences using Direct Preference Optimization (DPO). We construct synthetic queries and collect pairwise preferences to adapt the system to the policy domain. Through experiments evaluating retrieval quality, answer relevance, and faithfulness, we find that domain-specific fine-tuning improves retrieval metrics but does not consistently improve end-to-end question answering performance. In some cases, stronger retrieval counterintuitively leads to more confident hallucinations when relevant documents are absent from the corpus. These results highlight a key concern for those building policy-focused RAG systems: improvements to individual components do not necessarily translate to more reliable answers. Our findings provide practical insights for designing grounded question-answering systems over dynamic regulatory corpora.",
      "published": "2026-03-25T17:54:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.24580v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24580v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "LensWalk: Agentic Video Understanding by Planning How You See in Videos",
      "authors": [
        "Keliang Li",
        "Yansong Li",
        "Hongze Shen",
        "Mengdi Liu",
        "Hong Chang",
        "Shiguang Shan"
      ],
      "abstract": "The dense, temporal nature of video presents a profound challenge for automated analysis. Despite the use of powerful Vision-Language Models, prevailing methods for video understanding are limited by the inherent disconnect between reasoning and perception: they rely on static, pre-processed information and cannot actively seek raw evidence from video as their understanding evolves. To address this, we introduce LensWalk, a flexible agentic framework that empowers a Large Language Model reasoner to control its own visual observation actively. LensWalk establishes a tight reason-plan-observe loop where the agent dynamically specifies, at each step, the temporal scope and sampling density of the video it observes. Using a suite of versatile, Vision-Language Model based tools parameterized by these specifications, the agent can perform broad scans for cues, focus on specific segments for fact extraction, and stitch evidence from multiple moments for holistic verification. This design allows for progressive, on-demand evidence gathering that directly serves the agent's evolving chain of thought. Without requiring any model fine-tuning, LensWalk delivers substantial, plug-and-play performance gains on multiple model recipes, boosting their accuracy by over 5\\% on challenging long-video benchmarks like LVBench and Video-MME. Our analysis reveals that enabling an agent to control how it sees is key to unlocking more accurate, robust, and interpretable video reasoning.",
      "published": "2026-03-25T17:38:54Z",
      "abstract_url": "http://arxiv.org/abs/2603.24558v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24558v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Evaluating Chunking Strategies For Retrieval-Augmented Generation in Oil and Gas Enterprise Documents",
      "authors": [
        "Samuel Taiwo",
        "Mohd Amaluddin Yusoff"
      ],
      "abstract": "Retrieval-Augmented Generation (RAG) has emerged as a framework to address the constraints of Large Language Models (LLMs). Yet, its effectiveness fundamentally hinges on document chunking - an often-overlooked determinant of its quality. This paper presents an empirical study quantifying performance differences across four chunking strategies: fixed-size sliding window, recursive, breakpoint-based semantic, and structure-aware. We evaluated these methods using a proprietary corpus of oil and gas enterprise documents, including text-heavy manuals, table-heavy specifications, and piping and instrumentation diagrams (P and IDs). Our findings show that structure-aware chunking yields higher overall retrieval effectiveness, particularly in top-K metrics, and incurs significantly lower computational costs than semantic or baseline strategies. Crucially, all four methods demonstrated limited effectiveness on P and IDs, underscoring a core limitation of purely text-based RAG within visually and spatially encoded documents. We conclude that while explicit structure preservation is essential for specialised domains, future work must integrate multimodal models to overcome current limitations.",
      "published": "2026-03-25T17:35:24Z",
      "abstract_url": "http://arxiv.org/abs/2603.24556v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24556v1",
      "categories": [
        "cs.IR",
        "cs.AI"
      ]
    },
    {
      "title": "UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience",
      "authors": [
        "Zichuan Lin",
        "Feiyu Liu",
        "Yijun Yang",
        "Jiafei Lyu",
        "Yiming Gao",
        "Yicheng Liu",
        "Zhicong Lu",
        "Yangbin Yu",
        "Mingyu Yang",
        "Junyou Li",
        "Deheng Ye",
        "Jie Jiang"
      ],
      "abstract": "Autonomous mobile GUI agents have attracted increasing attention along with the advancement of Multimodal Large Language Models (MLLMs). However, existing methods still suffer from inefficient learning from failed trajectories and ambiguous credit assignment under sparse rewards for long-horizon GUI tasks. To that end, we propose UI-Voyager, a novel two-stage self-evolving mobile GUI agent. In the first stage, we employ Rejection Fine-Tuning (RFT), which enables the continuous co-evolution of data and models in a fully autonomous loop. The second stage introduces Group Relative Self-Distillation (GRSD), which identifies critical fork points in group rollouts and constructs dense step-level supervision from successful trajectories to correct failed ones. Extensive experiments on AndroidWorld show that our 4B model achieves an 81.0% Pass@1 success rate, outperforming numerous recent baselines and exceeding human-level performance. Ablation and case studies further verify the effectiveness of GRSD. Our method represents a significant leap toward efficient, self-evolving, and high-performance mobile GUI automation without expensive manual data annotation.",
      "published": "2026-03-25T17:10:29Z",
      "abstract_url": "http://arxiv.org/abs/2603.24533v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24533v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "No Single Metric Tells the Whole Story: A Multi-Dimensional Evaluation Framework for Uncertainty Attributions",
      "authors": [
        "Emily Schiller",
        "Teodor Chiaburu",
        "Marco Zullich",
        "Luca Longo"
      ],
      "abstract": "Research on explainable AI (XAI) has frequently focused on explaining model predictions. More recently, methods have been proposed to explain prediction uncertainty by attributing it to input features (uncertainty attributions). However, the evaluation of these methods remains inconsistent as studies rely on heterogeneous proxy tasks and metrics, hindering comparability. We address this by aligning uncertainty attributions with the well-established Co-12 framework for XAI evaluation. We propose concrete implementations for the correctness, consistency, continuity, and compactness properties. Additionally, we introduce conveyance, a property tailored to uncertainty attributions that evaluates whether controlled increases in epistemic uncertainty reliably propagate to feature-level attributions. We demonstrate our evaluation framework with eight metrics across combinations of uncertainty quantification and feature attribution methods on tabular and image data. Our experiments show that gradient-based methods consistently outperform perturbation-based approaches in consistency and conveyance, while Monte-Carlo dropconnect outperforms Monte-Carlo dropout in most metrics. Although most metrics rank the methods consistently across samples, inter-method agreement remains low. This suggests no single metric sufficiently evaluates uncertainty attribution quality. The proposed evaluation framework contributes to the body of knowledge by establishing a foundation for systematic comparison and development of uncertainty attribution methods.",
      "published": "2026-03-25T17:02:13Z",
      "abstract_url": "http://arxiv.org/abs/2603.24524v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24524v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Claudini: Autoresearch Discovers State-of-the-Art Adversarial Attack Algorithms for LLMs",
      "authors": [
        "Alexander Panfilov",
        "Peter Romov",
        "Igor Shilov",
        "Yves-Alexandre de Montjoye",
        "Jonas Geiping",
        "Maksym Andriushchenko"
      ],
      "abstract": "LLM agents like Claude Code can not only write code but also be used for autonomous AI research and engineering \\citep{rank2026posttrainbench, novikov2025alphaevolve}. We show that an \\emph{autoresearch}-style pipeline \\citep{karpathy2026autoresearch} powered by Claude Code discovers novel white-box adversarial attack \\textit{algorithms} that \\textbf{significantly outperform all existing (30+) methods} in jailbreaking and prompt injection evaluations. Starting from existing attack implementations, such as GCG~\\citep{zou2023universal}, the agent iterates to produce new algorithms achieving up to 40\\% attack success rate on CBRN queries against GPT-OSS-Safeguard-20B, compared to $\\leq$10\\% for existing algorithms (\\Cref{fig:teaser}, left). The discovered algorithms generalize: attacks optimized on surrogate models transfer directly to held-out models, achieving \\textbf{100\\% ASR against Meta-SecAlign-70B} \\citep{chen2025secalign} versus 56\\% for the best baseline (\\Cref{fig:teaser}, middle). Extending the findings of~\\cite{carlini2025autoadvexbench}, our results are an early demonstration that incremental safety and security research can be automated using LLM agents. White-box adversarial red-teaming is particularly well-suited for this: existing methods provide strong starting points, and the optimization objective yields dense, quantitative feedback. We release all discovered attacks alongside baseline implementations and evaluation code at https://github.com/romovpa/claudini.",
      "published": "2026-03-25T16:50:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.24511v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24511v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA",
      "authors": [
        "John Ray B. Martinez"
      ],
      "abstract": "Miscalibrated confidence scores are a practical obstacle to deploying AI in clinical settings. A model that is always overconfident offers no useful signal for deferral. We present a multi-agent framework that combines domain-specific specialist agents with Two-Phase Verification and S-Score Weighted Fusion to improve both calibration and discrimination in medical multiple-choice question answering. Four specialist agents (respiratory, cardiology, neurology, gastroenterology) generate independent diagnoses using Qwen2.5-7B-Instruct. Each diagnosis is then subjected to a two-phase self-verification process that measures internal consistency and produces a Specialist Confidence Score (S-score). The S-scores drive a weighted fusion strategy that selects the final answer and calibrates the reported confidence. We evaluate across four experimental settings, covering 100-question and 250-question high-disagreement subsets of both MedQA-USMLE and MedMCQA. Calibration improvement is the central finding, with ECE reduced by 49-74% across all four settings, including the harder MedMCQA benchmark where these gains persist even when absolute accuracy is constrained by knowledge-intensive recall demands. On MedQA-250, the full system achieves ECE = 0.091 (74.4% reduction over the single-specialist baseline) and AUROC = 0.630 (+0.056) at 59.2% accuracy. Ablation analysis identifies Two-Phase Verification as the primary calibration driver and multi-agent reasoning as the primary accuracy driver. These results establish that consistency-based verification produces more reliable uncertainty estimates across diverse medical question types, providing a practical confidence signal for deferral in safety-critical clinical AI applications.",
      "published": "2026-03-25T16:22:53Z",
      "abstract_url": "http://arxiv.org/abs/2603.24481v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24481v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Integrating Causal Machine Learning into Clinical Decision Support Systems: Insights from Literature and Practice",
      "authors": [
        "Domenique Zipperling",
        "Lukas Schmidt",
        "Benedikt Hahn",
        "Niklas Kühl",
        "Steven Kimbrough"
      ],
      "abstract": "Current clinical decision support systems (CDSSs) typically base their predictions on correlation, not causation. In recent years, causal machine learning (ML) has emerged as a promising way to improve decision-making with CDSSs by offering interpretable, treatment-specific reasoning. However, existing research often emphasizes model development rather than designing clinician-facing interfaces. To address this gap, we investigated how CDSSs based on causal ML should be designed to effectively support collaborative clinical decision-making. Using a design science research methodology, we conducted a structured literature review and interviewed experienced physicians. From these, we derived eight empirically grounded design requirements, developed seven design principles, and proposed nine practical design features. Our results establish guidance for designing CDSSs that deliver causal insights, integrate seamlessly into clinical workflows, and support trust, usability, and human-AI collaboration. We also reveal tensions around automation, responsibility, and regulation, highlighting the need for an adaptive certification process for ML-based medical products.",
      "published": "2026-03-25T16:00:06Z",
      "abstract_url": "http://arxiv.org/abs/2603.24448v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24448v1",
      "categories": [
        "cs.HC",
        "cs.AI"
      ]
    },
    {
      "title": "CUA-Suite: Massive Human-annotated Video Demonstrations for Computer-Use Agents",
      "authors": [
        "Xiangru Jian",
        "Shravan Nayak",
        "Kevin Qinghong Lin",
        "Aarash Feizi",
        "Kaixin Li",
        "Patrice Bechard",
        "Spandana Gella",
        "Sai Rajeswar"
      ],
      "abstract": "Computer-use agents (CUAs) hold great promise for automating complex desktop workflows, yet progress toward general-purpose agents is bottlenecked by the scarcity of continuous, high-quality human demonstration videos. Recent work emphasizes that continuous video, not sparse screenshots, is the critical missing ingredient for scaling these agents. However, the largest existing open dataset, ScaleCUA, contains only 2 million screenshots, equating to less than 20 hours of video. To address this bottleneck, we introduce CUA-Suite, a large-scale ecosystem of expert video demonstrations and dense annotations for professional desktop computer-use agents. At its core is VideoCUA, which provides approximately 10,000 human-demonstrated tasks across 87 diverse applications with continuous 30 fps screen recordings, kinematic cursor traces, and multi-layerfed reasoning annotations, totaling approximately 55 hours and 6 million frames of expert video. Unlike sparse datasets that capture only final click coordinates, these continuous video streams preserve the full temporal dynamics of human interaction, forming a superset of information that can be losslessly transformed into the formats required by existing agent frameworks. CUA-Suite further provides two complementary resources: UI-Vision, a rigorous benchmark for evaluating grounding and planning capabilities in CUAs, and GroundCUA, a large-scale grounding dataset with 56K annotated screenshots and over 3.6 million UI element annotations. Preliminary evaluation reveals that current foundation action models struggle substantially with professional desktop applications (~60% task failure rate). Beyond evaluation, CUA-Suite's rich multimodal corpus supports emerging research directions including generalist screen parsing, continuous spatial control, video-based reward modeling, and visual world models. All data and models are publicly released.",
      "published": "2026-03-25T15:52:56Z",
      "abstract_url": "http://arxiv.org/abs/2603.24440v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24440v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Enes Causal Discovery",
      "authors": [
        "Alexis Kafantaris"
      ],
      "abstract": "Enes The proposed architecture is a mixture of experts, which allows for the model entities, such as the causal relationships, to be further parameterized. More specifically, an attempt is made to exploit a neural net as implementing neurons poses a great challenge for this dataset. To explain, a simple and fast Pearson coefficient linear model usually achieves good scores. An aggressive baseline that requires a really good model to overcome that is. Moreover, there are major limitations when it comes to causal discovery of observational data. Unlike the sachs one did not use interventions but only prior knowledge; the most prohibiting limitation is that of the data which is addressed. Thereafter, the method and the model are described and after that the results are presented.",
      "published": "2026-03-25T15:47:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.24436v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24436v1",
      "categories": [
        "cs.NE",
        "cs.AI",
        "cs.LG",
        "cs.SC"
      ]
    },
    {
      "title": "AI-Supervisor: Autonomous AI Research Supervision via a Persistent Research World Model",
      "authors": [
        "Yunbo Long"
      ],
      "abstract": "Existing automated research systems operate as stateless, linear pipelines, generating outputs without maintaining a persistent understanding of the research landscape. They process papers sequentially, propose ideas without structured gap analysis, and lack mechanisms for agents to verify or refine each other's findings. We present AutoProf (Autonomous Professor), a multi-agent orchestration framework where specialized agents provide end-to-end AI research supervision driven by human interests, from literature review through gap discovery, method development, evaluation, and paper writing, via autonomous exploration and self-correcting updates. Unlike sequential pipelines, AutoProf maintains a continuously evolving Research World Model implemented as a Knowledge Graph, capturing methods, benchmarks, limitations, and unexplored gaps as shared memory across agents. The framework introduces three contributions: first, structured gap discovery that decomposes methods into modules, evaluates them across benchmarks, and identifies module-level gaps; second, self-correcting discovery loops that analyze why modules succeed or fail, detect benchmark biases, and assess evaluation adequacy; third, self-improving development loops using cross-domain mechanism search to iteratively address failing components. All agents operate under a consensus mechanism where findings are validated before being committed to the shared model. The framework is model-agnostic, supports mainstream large language models, and scales elastically with token budget from lightweight exploration to full-scale investigation.",
      "published": "2026-03-25T15:16:51Z",
      "abstract_url": "http://arxiv.org/abs/2603.24402v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24402v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Exploring How Fair Model Representations Relate to Fair Recommendations",
      "authors": [
        "Bjørnar Vassøy",
        "Benjamin Kille",
        "Helge Langseth"
      ],
      "abstract": "One of the many fairness definitions pursued in recent recommender system research targets mitigating demographic information encoded in model representations. Models optimized for this definition are typically evaluated on how well demographic attributes can be classified given model representations, with the (implicit) assumption that this measure accurately reflects \\textit{recommendation parity}, i.e., how similar recommendations given to different users are. We challenge this assumption by comparing the amount of demographic information encoded in representations with various measures of how the recommendations differ. We propose two new approaches for measuring how well demographic information can be classified given ranked recommendations. Our results from extensive testing of multiple models on one real and multiple synthetically generated datasets indicate that optimizing for fair representations positively affects recommendation parity, but also that evaluation at the representation level is not a good proxy for measuring this effect when comparing models. We also provide extensive insight into how recommendation-level fairness metrics behave for various models by evaluating their performances on numerous generated datasets with different properties.",
      "published": "2026-03-25T15:12:20Z",
      "abstract_url": "http://arxiv.org/abs/2603.24396v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24396v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When AI Meets Early Childhood Education: Large Language Models as Assessment Teammates in Chinese Preschools",
      "authors": [
        "Xingming Li",
        "Runke Huang",
        "Yanan Bao",
        "Yuye Jin",
        "Yuru Jiao",
        "Qingyong Hu"
      ],
      "abstract": "High-quality teacher-child interaction (TCI) is fundamental to early childhood development, yet traditional expert-based assessment faces a critical scalability challenge. In large systems like China's-serving 36 million children across 250,000+ kindergartens-the cost and time requirements of manual observation make continuous quality monitoring infeasible, relegating assessment to infrequent episodic audits that limit timely intervention and improvement tracking. In this paper, we investigate whether AI can serve as a scalable assessment teammate by extracting structured quality indicators and validating their alignment with human expert judgments. Our contributions include: (1) TEPE-TCI-370h (Tracing Effective Preschool Education), the first large-scale dataset of naturalistic teacher-child interactions in Chinese preschools (370 hours, 105 classrooms) with standardized ECQRS-EC and SSTEW annotations; (2) We develop Interaction2Eval, a specialized LLM-based framework addressing domain-specific challenges-child speech recognition, Mandarin homophone disambiguation, and rubric-based reasoning-achieving up to 88% agreement; (3) Deployment validation across 43 classrooms demonstrating an 18x efficiency gain in the assessment workflow, highlighting its potential for shifting from annual expert audits to monthly AI-assisted monitoring with targeted human oversight. This work not only demonstrates the technical feasibility of scalable, AI-augmented quality assessment but also lays the foundation for a new paradigm in early childhood education-one where continuous, inclusive, AI-assisted evaluation becomes the engine of systemic improvement and equitable growth.",
      "published": "2026-03-25T15:05:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.24389v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24389v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "MolEvolve: LLM-Guided Evolutionary Search for Interpretable Molecular Optimization",
      "authors": [
        "Xiangsen Chen",
        "Ruilong Wu",
        "Yanyan Lan",
        "Ting Ma",
        "Yang Liu"
      ],
      "abstract": "Despite deep learning's success in chemistry, its impact is hindered by a lack of interpretability and an inability to resolve activity cliffs, where minor structural nuances trigger drastic property shifts. Current representation learning, bound by the similarity principle, often fails to capture these structural-activity discontinuities. To address this, we introduce MolEvolve, an evolutionary framework that reformulates molecular discovery as an autonomous, look-ahead planning problem. Unlike traditional methods that depend on human-engineered features or rigid prior knowledge, MolEvolve leverages a Large Language Model (LLM) to actively explore and evolve a library of executable chemical symbolic operations. By utilizing the LLM to cold start and an Monte Carlo Tree Search (MCTS) engine for test-time planning with external tools (e.g. RDKit), the system self-discovers optimal trajectories autonomously. This process evolves transparent reasoning chains that translate complex structural transformations into actionable, human-readable chemical insights. Experimental results demonstrate that MolEvolve's autonomous search not only evolves transparent, human-readable chemical insights, but also outperforms baselines in both property prediction and molecule optimization tasks.",
      "published": "2026-03-25T15:01:03Z",
      "abstract_url": "http://arxiv.org/abs/2603.24382v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24382v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CE"
      ]
    },
    {
      "title": "Evidence of an Emergent \"Self\" in Continual Robot Learning",
      "authors": [
        "Adidev Jhunjhunwala",
        "Judah Goldfeder",
        "Hod Lipson"
      ],
      "abstract": "A key challenge to understanding self-awareness has been a principled way of quantifying whether an intelligent system has a concept of a \"self,\" and if so how to differentiate the \"self\" from other cognitive structures. We propose that the \"self\" can be isolated by seeking the invariant portion of cognitive process that changes relatively little compared to more rapidly acquired cognitive knowledge and skills, because our self is the most persistent aspect of our experiences. We used this principle to analyze the cognitive structure of robots under two conditions: One robot learns a constant task, while a second robot is subjected to continual learning under variable tasks. We find that robots subjected to continual learning develop an invariant subnetwork that is significantly more stable (p < 0.001) compared to the control. We suggest that this principle can offer a window into exploring selfhood in other cognitive AI systems.",
      "published": "2026-03-25T14:27:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.24350v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24350v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Enhancing Efficiency and Performance in Deepfake Audio Detection through Neuron-level dropin & Neuroplasticity Mechanisms",
      "authors": [
        "Yupei Li",
        "Shuaijie Shao",
        "Manuel Milling",
        "Björn Schuller"
      ],
      "abstract": "Current audio deepfake detection has achieved remarkable performance using diverse deep learning architectures such as ResNet, and has seen further improvements with the introduction of large models (LMs) like Wav2Vec. The success of large language models (LLMs) further demonstrates the benefits of scaling model parameters, but also highlights one bottleneck where performance gains are constrained by parameter counts. Simply stacking additional layers, as done in current LLMs, is computationally expensive and requires full retraining. Furthermore, existing low-rank adaptation methods are primarily applied to attention-based architectures, which limits their scope. Inspired by the neuronal plasticity observed in mammalian brains, we propose novel algorithms, dropin and further plasticity, that dynamically adjust the number of neurons in certain layers to flexibly modulate model parameters. We evaluate these algorithms on multiple architectures, including ResNet, Gated Recurrent Neural Networks, and Wav2Vec. Experimental results using the widely recognised ASVSpoof2019 LA, PA, and FakeorReal dataset demonstrate consistent improvements in computational efficiency with the dropin approach and a maximum of around 39% and 66% relative reduction in Equal Error Rate with the dropin and plasticity approach among these dataset, respectively. The code and supplementary material are available at Github link.",
      "published": "2026-03-25T14:22:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.24343v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24343v1",
      "categories": [
        "cs.SD",
        "cs.AI"
      ]
    },
    {
      "title": "Large Language Model Guided Incentive Aware Reward Design for Cooperative Multi-Agent Reinforcement Learning",
      "authors": [
        "Dogan Urgun",
        "Gokhan Gungor"
      ],
      "abstract": "Designing effective auxiliary rewards for cooperative multi-agent systems remains a precarious task; misaligned incentives risk inducing suboptimal coordination, especially where sparse task feedback fails to provide sufficient grounding. This study introduces an automated reward design framework that leverages large language models to synthesize executable reward programs from environment instrumentation. The procedure constrains candidate programs within a formal validity envelope and evaluates their efficacy by training policies from scratch under a fixed computational budget; selection depends exclusively on the sparse task return. The framework is evaluated across four distinct Overcooked-AI layouts characterized by varied corridor congestion, handoff dependencies, and structural asymmetries. Iterative search generations consistently yield superior task returns and delivery counts, with the most pronounced gains occurring in environments dominated by interaction bottlenecks. Diagnostic analysis of the synthesized shaping components indicates increased interdependence in action selection and improved signal alignment in coordination-intensive tasks. These results demonstrate that the search for objectivegrounded reward programs can mitigate the burden of manual engineering while producing shaping signals compatible with cooperative learning under finite budgets.",
      "published": "2026-03-25T14:05:59Z",
      "abstract_url": "http://arxiv.org/abs/2603.24324v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24324v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Cost-Sensitive Neighborhood Aggregation for Heterophilous Graphs: When Does Per-Edge Routing Help?",
      "authors": [
        "Eyal Weiss"
      ],
      "abstract": "Recent work distinguishes two heterophily regimes: adversarial, where cross-class edges dilute class signal and harm classification, and informative, where the heterophilous structure itself carries useful signal. We ask: when does per-edge message routing help, and when is a uniform spectral channel sufficient? To operationalize this question we introduce Cost-Sensitive Neighborhood Aggregation (CSNA), a GNN layer that computes pairwise distance in a learned projection and uses it to soft-route each message through concordant and discordant channels with independent transformations. Under a contextual stochastic block model we show that cost-sensitive weighting preserves class-discriminative signal where mean aggregation provably attenuates it, provided $w_+/w_- > q/p$. On six benchmarks with uniform tuning, CSNA is competitive with state-of-the-art methods on adversarial-heterophily datasets (Texas, Wisconsin, Cornell, Actor) but underperforms on informative-heterophily datasets (Chameleon, Squirrel) -- precisely the regime where per-edge routing has no useful decomposition to exploit. The pattern is itself the finding: the cost function's ability to separate edge types serves as a diagnostic for the heterophily regime, revealing when fine-grained routing adds value over uniform channels and when it does not. Code is available at https://github.com/eyal-weiss/CSNA-public .",
      "published": "2026-03-25T13:28:31Z",
      "abstract_url": "http://arxiv.org/abs/2603.24291v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24291v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Embracing Heteroscedasticity for Probabilistic Time Series Forecasting",
      "authors": [
        "Yijun Wang",
        "Qiyuan Zhuang",
        "Xiu-Shen Wei"
      ],
      "abstract": "Probabilistic time series forecasting (PTSF) aims to model the full predictive distribution of future observations, enabling both accurate forecasting and principled uncertainty quantification. A central requirement of PTSF is to embrace heteroscedasticity, as real-world time series exhibit time-varying conditional variances induced by nonstationary dynamics, regime changes, and evolving external conditions. However, most existing non-autoregressive generative approaches to PTSF, such as TimeVAE and $K^2$VAE, rely on MSE-based training objectives that implicitly impose a homoscedastic assumption, thereby fundamentally limiting their ability to model temporal heteroscedasticity. To address this limitation, we propose the Location-Scale Gaussian VAE (LSG-VAE), a simple but effective framework that explicitly parameterizes both the predictive mean and time-dependent variance through a location-scale likelihood formulation. This design enables LSG-VAE to faithfully capture heteroscedastic aleatoric uncertainty and introduces an adaptive attenuation mechanism that automatically down-weights highly volatile observations during training, leading to improved robustness in trend prediction. Extensive experiments on nine benchmark datasets demonstrate that LSG-VAE consistently outperforms fifteen strong generative baselines while maintaining high computational efficiency suitable for real-time deployment.",
      "published": "2026-03-25T12:48:50Z",
      "abstract_url": "http://arxiv.org/abs/2603.24254v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24254v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "DVM: Real-Time Kernel Generation for Dynamic AI Models",
      "authors": [
        "Jingzhi Fang",
        "Xiong Gao",
        "Renwei Zhang",
        "Zichun Ye",
        "Lei Chen",
        "Jie Zhao",
        "Chengnuo Huang",
        "Hui Xu",
        "Xuefeng Jin"
      ],
      "abstract": "Dynamism is common in AI computation, e.g., the dynamic tensor shapes and the dynamic control flows in models. Due to the long compilation time, existing runtime compilation damages the model efficiency, while the offline compilers either suffer from the long compilation time and device memory footprint to cover all the possible execution instances of a dynamic model, or sacrifice optimization opportunities for usability. In this paper, we rethink the feasibility of runtime compilation for dynamic models and identify that the key for it to work is to speed up the compilation or hide the compilation overhead. To do this, we propose a real-time compiler, DVM. In DVM, we design a runtime operator compiler based on a bytecode virtual machine to perform effective and efficient compilation for each dynamic operator instance given its input. Specifically, instead of compiling programs into machine code, we encode the operator program into bytecode on the CPU and decode the bytecode into virtual instructions for direct execution on the NPU. Based on the runtime operator compiler, we further propose an operator fuser, which performs symbol-deduction-based fusion on static graphs and runtime fusion on dynamic graphs. Both pattern- and stacking-based fusion are supported to increase fusion opportunities. Evaluation on operators, subgraphs, and models shows that, compared with TorchInductor, PyTorch-eager and MindSpore-graph-O0, we are up to 11.77$\\times$ better in terms of the operator/model efficiency and up to 5 orders of magnitude faster in terms of the maximum compilation time.",
      "published": "2026-03-25T12:24:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.24239v1",
      "pdf_url": "https://arxiv.org/pdf/2603.24239v1",
      "categories": [
        "cs.PL",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
