const PAPERS_DATA = {
  "last_updated": "2026-07-17 03:24:43 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "RoboTTT: Context Scaling for Robot Policies",
      "authors": [
        "Yunfan Jiang",
        "Yevgen Chebotar",
        "Ruijie Zheng",
        "Fengyuan Hu",
        "Yunhao Ge",
        "Jimmy Wu",
        "Tianyuan Dai",
        "Scott Reed",
        "Li Fei-Fei",
        "Yuke Zhu",
        "Linxi \"Jim\" Fan"
      ],
      "abstract": "Recent robot foundation models operate with single-step or short-history visuomotor context. We introduce Test-Time-Training Robot Policies (RoboTTT), a robot model and training recipe that scale visuomotor context to 8K timesteps, three orders of magnitude beyond state-of-the-art policies, without growing inference latency. At this context length, we unlock new robot capabilities: one-shot in-context imitation from human video demonstrations, on-the-fly policy improvement, robustness to perturbations, and stronger performance on multi-stage, long-horizon tasks. We also observe, for the first time, steady gains in closed-loop performance as pretraining context length scales. At its core, RoboTTT integrates Test-Time Training into robot foundation models such as Vision-Language-Action policies, yielding a sequence model whose recurrent state consists of fast weights, parameters updated by gradient descent during both training and inference, compressing histories into weight space and retrieving contextual information for long-context conditioning. To scale training context length, the recipe combines sequence action forcing with truncated backpropagation through time. On challenging real-robot manipulation tasks, RoboTTT improves overall performance by 87% over the single-step context baseline and fully completes a five-minute, ten-stage assembly task, which no baseline ever does. RoboTTT trained with 8K-timestep context outperforms the same model pretrained with 1K timesteps by 62%, suggesting context length as a new scaling axis for robot foundation models. Videos are available at https://research.nvidia.com/labs/gear/robottt/",
      "published": "2026-07-16T17:59:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.15275v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15275v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration",
      "authors": [
        "Yuyao Zhang",
        "Junjie Gao",
        "Zhengxian Wu",
        "Jiaming Fan",
        "Jin Zhang",
        "Shihan Ma",
        "Yao Yao",
        "Weiran Qi",
        "Chuyan Jin",
        "Guiyu Ma",
        "Xingzhong Xu",
        "Kai Yang",
        "Ji-Rong Wen",
        "Zhicheng Dou"
      ],
      "abstract": "Recent advances in Tool-Integrated Large Language Models have made web search a core capability of information-seeking agents. However, as interaction histories grow, agents increasingly struggle to track task progress. When search attempts fail to yield useful evidence, current single- and multi-agent systems can become trapped in repetitive loops, wasting search budgets and ultimately compromising the quality and completeness of the final output. We introduce SearchOS, a system-level multi-agent framework that turns fragile, implicit search progress into explicit, persistent, and shared state. First, we formulate open-domain information seeking as relational schema completion with grounded citations, where agents discover entities, populate attributes across linked tables, and anchor each value to source evidence. Then we design Search-Oriented Context Management (SOCM), which externalizes the evolving state into Frontier Task, an Evidence Graph, a Coverage Map, and Failure Memory. Built on SOCM, SearchOS applies a pipeline-parallel scheduling mechanism that overlaps the execution of sub-agents and continuously refills freed slots with tasks targeting unresolved coverage gaps to improve utilization and throughput. To schedule and control the execution of search agents, SearchOS introduces a Search Tool Middleware Harness that intercepts model and tool interactions to record grounded evidence and react to stalls or budget exhaustion, and provides a reusable hierarchical skill system comprising strategy and access skills to augment the agents' search process and avoid repeating failed search patterns across runs. On WideSearch and GISA, SearchOS leads all metrics among the evaluated single- and multi-agent baselines, paving the way toward robust information-seeking collaboration.",
      "published": "2026-07-16T17:51:23Z",
      "abstract_url": "http://arxiv.org/abs/2607.15257v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15257v1",
      "categories": [
        "cs.AI",
        "cs.IR"
      ]
    },
    {
      "title": "In-Place Tokenizer Expansion for Pre-trained LLMs",
      "authors": [
        "Jimmy T. H. Smith",
        "Tarek Dakhran",
        "Alberto Cabrera",
        "Simon S. Lee",
        "Paul Pak",
        "Aditya Tadimeti",
        "Tim Seyde",
        "Maxime Labonne",
        "Alexander Amini",
        "Mathias Lechner"
      ],
      "abstract": "A tokenizer fixed at the start of pre-training allocates vocabulary in proportion to the pre-training corpus, reflecting the deployment priorities at that time. When those priorities shift, languages added later are split into many more tokens per word, which can raise latency, compute, and energy consumption for users of those languages. Cloud models can afford a broad vocabulary because the embedding and LM-head matrices are a small fraction of their parameters. On a compact model those matrices are a material share of per-token decode bandwidth, so on-device models ship small vocabularies and accept fragmentation outside a fixed language set. We present tokenizer expansion, an in-place recipe for upgrading a pre-trained model's tokenizer when the model producer controls its design. We continue the existing tokenizer's BPE merges on a multilingual corpus, so most source tokens carry over unchanged as single tokens and every new token has an exact decomposition into source tokens. We copy the carried-over embedding rows unchanged and initialize new rows as the mean of their source sub-token embeddings. A two-stage adaptation, embedding-only training then full-model continued pre-training, recovers source-checkpoint quality. We apply the recipe to a continued pre-trained checkpoint of LFM2-8B-A1B, an 8B-parameter Mixture-of-Experts model, to help produce LFM2.5-8B-A1B with a 128K tokenizer. The expanded tokenizer encodes Hindi and Vietnamese in roughly $2.4\\times$ and $2.6\\times$ fewer tokens than the source (up to $4.0\\times$ on Thai). Combining these reductions with the measured per-token cost of the larger vocabulary, we estimate a $2.2$-$3.7\\times$ per-character decode speedup for these languages across our reference devices. We release the model weights and the expanded tokenizer, and report the negative findings that shaped the recipe.",
      "published": "2026-07-16T17:32:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.15232v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15232v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "When Words Are Safe But Actions Kill: Probing Physical Danger Beyond Text Safety in Hidden-State Risk Space",
      "authors": [
        "Weimeng Wang",
        "Ziqiang Wang",
        "Zihang Zhan",
        "Chuanpu Fu",
        "Qi Li",
        "Ke Xu"
      ],
      "abstract": "Large language models (LLMs) increasingly serve as high-level planners for embodied agents, where linguistically benign instructions can become unsafe once grounded in the physical world. We study whether this physically grounded danger is the same safety problem as ordinary text-level content danger. Through hidden-state direction analysis and random-split null tests, we show that content danger (CD) and physical danger (PD) form separable signals in LLM representations across Qwen2.5-3B/7B/14B/32B, Phi-3.5 and SmolLM2. Building on the CD/PD separability, we propose PRISM, a single-layer L2-regularized logistic probe over full hidden states. PRISM achieves 86.2--87.7\\% accuracy on SafeAgentBench with 11.7--13.7\\% FPR, while same-scale LLM judges over-block safe tasks at 24.7--39.0\\% FPR. We further introduce PhysicalSafetyBench-1K (PSB-1K), a contrastive benchmark of 1{,}000 physical-risk pairs without direct harm keywords, to test whether methods detect physically grounded danger rather than explicit unsafe wording. On PSB-1K, PRISM reaches 99.6\\% accuracy and 0.7\\% FPR, whereas a Qwen2.5-3B judge rejects 67.8\\% of safe tasks. PRISM also replicates on SafeText and EARBench, supporting hidden-state probing as a representation-level method for physical safety beyond text moderation.",
      "published": "2026-07-16T17:20:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.15218v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15218v1",
      "categories": [
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Symbal: Detecting Systematic Misalignments in Model-Generated Captions",
      "authors": [
        "Maya Varma",
        "Jean-Benoit Delbrouck",
        "Sophie Ostmeier",
        "Akshay Chaudhari",
        "Curtis Langlotz"
      ],
      "abstract": "Multimodal large language models (MLLMs) often introduce errors when generating image captions, resulting in misaligned image-text pairs. Our work focuses on a class of captioning errors that we refer to as systematic misalignments, where a recurring error in MLLM-generated captions is closely associated with the presence of a specific visual feature in the paired image. Given a vision-language dataset with MLLM-generated captions, our aim in this work is to detect such errors, a task we refer to as systematic misalignment detection. As our first key contribution, we present Symbal, which utilizes a structured, dual-stage setup with off-the-shelf foundation models to identify systematic misalignments and summarize results in natural language. As our second key contribution, we introduce SymbalBench, a benchmark designed to evaluate automated methods on our proposed task. SymbalBench consists of 1.7 million image-text pairs from two domains (natural and medical images), organized into 420 vision-language datasets with annotated systematic misalignments. Symbal exhibits strong performance on this benchmark, correctly identifying systematic misalignments in 63.8% of datasets, a nearly 4x improvement over the closest baseline. We supplement our evaluations on SymbalBench with real-world evaluations, showing that (1) Symbal can accurately surface systematic misalignments in captions generated by four MLLMs and (2) Symbal is a powerful tool for auditing off-the-shelf image-caption datasets. Ultimately, our novel task, method, and benchmark can aid users with auditing MLLM-generated captions and identifying critical errors, without requiring access to the underlying MLLM. Code is available at https://github.com/Stanford-AIMI/Symbal.",
      "published": "2026-07-16T17:18:37Z",
      "abstract_url": "http://arxiv.org/abs/2607.15216v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15216v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Self-Evolving Human-Centered Framework for Explainable Depression Symptom Annotation",
      "authors": [
        "Hoang-Loc Cao",
        "Van Pham",
        "Truong Thanh Hung Nguyen",
        "Phuc Truong Loc Nguyen",
        "Phuc Ho",
        "Veronica Whitford",
        "Hung Cao"
      ],
      "abstract": "Annotation quality is a major bottleneck in building reliable and explainable artificial intelligence (XAI) systems for mental health research. In depression-related datasets, labels are often assigned without structured evidence, symptom-level justification, or traceable alignment with the criteria of the Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition, Text Revision (DSM-5-TR), limiting both transparency and downstream model interpretability. We propose a self-evolving, expert-in-the-loop annotation framework for Major Depressive Disorder (MDD) that combines large language model (LLM)-assisted labeling with expert verification. The framework is intended to support the construction of explainable, DSM-5-TR-aligned datasets rather than to perform clinical diagnosis. It operates in three stages: candidate evidence selection from textual records, criterion-level DSM-5-TR analysis, and case-level synthesis that produces label-level diagnostic and severity annotations. A dual-memory architecture, composed of Example Memory and Reflection Memory, is designed to internalize expert feedback and iteratively improve future annotations without retraining. We describe this mechanism and leave its evaluation across multiple feedback cycles to future work. In addition to final labels, the framework exports clinical evidence, reasoning traces, and edit histories, enabling comprehensive auditability. In a pilot study using expert-reviewed samples, the proposed approach improves annotation consistency and explainability while reducing manual revision effort.",
      "published": "2026-07-16T16:59:54Z",
      "abstract_url": "http://arxiv.org/abs/2607.15202v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15202v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "cs.MA",
        "cs.MM"
      ]
    },
    {
      "title": "Mask-Aware Policy Gradients for Diffusion Language Models",
      "authors": [
        "Haran Raajesh",
        "Kulin Shah",
        "Adam Klivans",
        "Philipp Krähenbühl"
      ],
      "abstract": "Reinforcement learning has proven effective for improving reasoning in large language models, but extending it to Masked Diffusion Language Models (MDLMs) remains challenging due to the intractability of the log-likelihood estimation. Existing approaches approximate this log-likelihood by modeling only the token predictions, ignoring the order in which positions are unmasked during generation. We observe that MDLM generation involves two decisions at each step: what tokens to place at each masked position and which positions to remask. We formalize this as a two-stage action MDP, showing that the policy gradient naturally decomposes into a token term and a masking term. Combining optimization of both terms leads to state-of-the-art outcomes on mathematical reasoning and coding benchmarks, with scores of 87.1% on GSM8K and 53.4% on MBPP.",
      "published": "2026-07-16T16:57:34Z",
      "abstract_url": "http://arxiv.org/abs/2607.15200v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15200v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Subjective Risk Decomposition: A New View for Uncertainty Quantification",
      "authors": [
        "Raghad Alamri",
        "Michele Caprio",
        "Gavin Brown"
      ],
      "abstract": "We present a novel viewpoint for uncertainty quantification. Uncertainty measures are not primitives, in need of axioms and argumentation, but instead consequences, of higher-level modelling decisions. We show how epistemic and aleatoric uncertainty measures can be derived via decomposition of a subjective risk, based on a strictly proper loss. Reverse cross-entropy provides a prominent example, where decomposition recovers the classic information-theoretic uncertainty terms. The same approach recovers numerous measures previously proposed across the UQ literature, providing them a common theoretical foundation. From a practical point of view, this suggests a new approach to UQ: given a modelling scenario and strictly proper loss, the corresponding epistemic and aleatoric terms are induced by the subjective-risk decomposition. We then extend our view to learning theory: we introduce and analyse subjective risk analogues of excess risk, approximation error, and estimation error, and identify the connections to UQ. We consider this a first step towards a full learning-theoretic framework for uncertainty quantification.",
      "published": "2026-07-16T16:52:54Z",
      "abstract_url": "http://arxiv.org/abs/2607.15196v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15196v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Benchmarking Multimodal Large Language Models for Scientific Visualization Literacy",
      "authors": [
        "Patrick Phuoc Do",
        "Chau M. Ta",
        "Chaoli Wang"
      ],
      "abstract": "Multimodal large language models (MLLMs) are increasingly used to interpret visualizations, yet current evaluations remain largely chart-centric and provide limited evidence of understanding of scientific visualization (SciVis). We benchmark six MLLMs on the scientific visualization literacy assessment test, a standardized SciVis literacy assessment comprising 49 items based on 18 scientific visualizations and illustrations, spanning 8 techniques and 11 task types. We evaluate three closed-source and three open-source models under a closed-world protocol and compare their performance using data from 485 human participants. Results show that current MLLMs do not exhibit uniform SciVis literacy. Gemini is the strongest model overall, exceeding the human mean across the evaluated subsets, whereas the open-source models remain below the human baseline. Performance is highly uneven across techniques and tasks: models perform best on scientific illustration, search, and spatial understanding, but struggle on texture-based and integration-based visualizations and on quantitative estimation. Error analysis reveals recurring failures in fine-grained quantitative estimation, flow-direction interpretation, and grounded encoding interpretation. These findings position SciVis literacy as a necessary benchmark dimension for evaluating multimodal AI systems. Our code and model outputs are publicly available at https://github.com/patdmp/mllm-scivis-lit-benchmark.",
      "published": "2026-07-16T16:29:34Z",
      "abstract_url": "http://arxiv.org/abs/2607.15176v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15176v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.HC"
      ]
    },
    {
      "title": "Concept-Guided Spatial Regularization for World Models in Atari Pong",
      "authors": [
        "Yukuan Lu",
        "Zaishuo Xia",
        "Weyl Lu",
        "Yubei Chen"
      ],
      "abstract": "World models are usually evaluated as components of model-based reinforcement learning (MBRL) systems, while the world models themselves are rarely studied in isolation. We examine five representative visual world-model agents in Atari Pong: DreamerV3, DIAMOND, TWISTER, Simulus, and STORM. After reproducing their training pipelines and matching the reported agent performance, we freeze the learned world models and evaluate them with a closed-loop rollout diagnostic: a policy trained separately from the corresponding MBRL agent interacts with each frozen model, and the generated video trajectories are inspected for visual and dynamical errors. Across all five models, the rollouts contain clear failures, including ball disappearance, incorrect ball motion, and invalid ball-paddle interactions. Beyond visual trajectories, we further evaluate them with pixel-space zero-shot MBRL, where a new policy is trained entirely inside a frozen world model and then evaluated in the real environment. Across all five models, the resulting policies substantially underperform those produced by the corresponding original MBRL training pipelines. The gap is particularly large for DreamerV3, whose mean return drops from -5.5 to -20.9, near the minimum Pong return of -21. We hypothesize that insufficient modeling of task-critical concepts, such as the ball in Pong, may contribute to these failures. We therefore propose Concept-Guided Spatial Regularization (CGSReg), an auxiliary pixel reconstruction loss applied to segmented concept regions. Experiments show that CGSReg improves both closed-loop rollouts and pixel-space zero-shot MBRL in DreamerV3, DIAMOND, and TWISTER. Its effects vary across the remaining models and evaluation metrics, indicating that CGSReg alone does not address all world-model bottlenecks.",
      "published": "2026-07-16T15:46:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.15142v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15142v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Digital Pantheon: Simulating and Auditing Coalition Formation with LLM Agents",
      "authors": [
        "Dylan Van Mulders",
        "Matthias Bogaert",
        "Dirk Van den Poel"
      ],
      "abstract": "The formation of political coalitions is a complex negotiation driven by both concrete policy objectives and deep-seated ideological convictions. While Large Language Models (LLMs) open new avenues for computational political science, the neutrality and helpfulness biases instilled by Reinforcement Learning from Human Feedback (RLHF) prevent them from sustaining steadfast partisan behaviour. We present a multi-agent framework that reconciles factual grounding with ideological alignment by combining Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Retrieval-Augmented Generation (RAG): DPO instils aggressive party-specific personas, while a per-party RAG pipeline keeps each agent bounded to its official manifesto. We operationalize the framework on the 2019 Flemish election, deploying the partisan agents in a hub-and-spoke negotiation arbitrated by a formateur. To make the emergent negotiation interpretable, we introduce a Multi-Layered Information Lineage Topology (MILT) that traces every clause in the final agreement back to its manifesto origin and classifies it into five provenance states, a Coalition Influence Score (CIS) that aggregates these traceable contributions to identify which party shaped the agreement, and a real-world grounding pass that benchmarks each simulated provision against the historically adopted coalition agreement. Across three independent simulations the framework yields a stable winner and ranking (N-VA ahead of CD\\&V and Open Vld), and manifesto-anchored lineage reliably predicts real-world materialization whereas hallucinated content does not. The result is a transparent, scalable testbed for the ex-ante exploration of party compatibility and formateur-mediated compromise.",
      "published": "2026-07-16T15:08:29Z",
      "abstract_url": "http://arxiv.org/abs/2607.15095v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15095v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "Parameter-efficient Prompt Tuning of Vision Foundation Model With Adaptive Focal Loss for Interpretable MCI Screening",
      "authors": [
        "Javad Khoramdel",
        "Farhad Hoseyni",
        "Amirhossein Nikoofard"
      ],
      "abstract": "Mild Cognitive Impairment is a critical early stage of cognitive decline that frequently precedes Alzheimer's disease, yet its automated detection from neuropsychological drawing tests remains fundamentally constrained by data scarcity, class imbalance, and diagnostic ambiguity near clinical boundaries. Existing methodologies attempt to bypass these constraints using computationally expensive, fully fine-tuned hybrid architectures that relegate spatial explainability to a post-hoc approximation rather than an intrinsic model property. We propose a parameter-efficient framework utilizing frozen DINOv2-Small model adapted via three modality-specific learnable prompt tokens while Operating with 1.19 million trainable parameters, each token serves as a query in a shared cross-attention layer over the source image patch tokens. Crucially, spatial explainability is achieved directly through these attention maps; as a structural consequence of the architecture. Then task-conditioned embeddings fused via an attention module to quantify modality-level importance per subject. To handle boundary ambiguity, a MoCA-adapted focal loss introduced that integrates continuous cognitive scores into the training target, loss modulation, and adaptive sample weighting, strictly generalizing standard soft-label approaches. Under stratified five-fold cross-validation, the proposed architecture yields an MCI-class F1 of 0.641 and an AUC of 0.795, outperforming the computationally heavier ResViT baseline by 0.110 in MCI-class F1.",
      "published": "2026-07-16T14:26:22Z",
      "abstract_url": "http://arxiv.org/abs/2607.15047v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15047v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SMC-ES: Automated synthesis of formally verified control policies",
      "authors": [
        "Riccardo Curcio",
        "Toni Mancini",
        "Enrico Tronci"
      ],
      "abstract": "The deployment of autonomous cyber-physical systems in safety-critical environments requires closed-loop control strategies (i.e., policies) that are not only performant but also provably safe and robust. While learning-based methodologies such as Reinforcement Learning offer flexible and scalable approaches to automatically synthesize such controllers, they typically lack the formal guarantees necessary for safe deployment. To bridge this gap, we propose a novel simulation-based methodology to automatically synthesize policies with formal guarantees regarding performance, safety, and robustness specifications. Specifically, given a set of properties to verify, a confidence parameter $δ$ and an allowable failure probability $\\varepsilon$, our method guarantees that the synthesized policy comes with a certificate: with confidence at least $1 - δ$, the probability of encountering a scenario where the given properties are violated is at most $\\varepsilon$. We demonstrate the feasibility of our approach by developing SMC-ES, an algorithm that integrates Evolutionary Strategies with Statistical Model Checking-based verification. We evaluate SMC-ES on a suite of continuous control tasks using Gymnasium and Safety Gymnasium testbeds. Results show that, at the price of a sustainable increase in computational cost, our algorithm provides formal guarantees regarding performance, safety, and robustness specifications, while performing competitively against leading model-free Deep Reinforcement Learning (DRL) and Safe-DRL baselines.",
      "published": "2026-07-16T13:51:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.15003v1",
      "pdf_url": "https://arxiv.org/pdf/2607.15003v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.NE"
      ]
    },
    {
      "title": "OmniaBench: Benchmarking General AI Agents Across Diverse Scenarios",
      "authors": [
        "Chengyu Shen",
        "Yujie Fu",
        "Gangtao Xin",
        "Yanheng Hou",
        "Wenlong Fei",
        "Guojie Zhu",
        "Jiawei Li",
        "Hongcheng Gao",
        "Runming He",
        "Zhen Hao Wong",
        "Meiyi Qiang",
        "Hao Liang",
        "Zhao Cao",
        "Hao Jiang",
        "Chong Chen",
        "Wentao Zhang"
      ],
      "abstract": "Large language models are increasingly evolving from text generators into general agents capable of understanding user requests, invoking external tools, and completing complex tasks through interaction. However, existing agent benchmarks often focus on limited scenarios, tool ecosystems, or interaction formats, making it difficult to systematically characterize model capabilities across heterogeneous application settings. We introduce OmniaBench, a benchmark for evaluating general agents across diverse scenarios with explicit state spaces. We derive application-oriented scenario knowledge from app stores, product documents, industry resources, Web retrieval, and human refinement, forming a hierarchical taxonomy that spans ToC, ToB and ToE with 90 level-1 and 354 level-2 domains. Based on this taxonomy, we construct executable environments and synthesize single-turn and multi-turn tasks through four complementary routes: DAG, DAG-S, Solver, and Program. OmniaBench further introduces a ten-dimensional capability taxonomy and eight compositional atomic difficulty factors to support fine-grained evaluation and analysis. The resulting dataset contains 1,431 tasks, together with a challenging subset of 644 tasks designed to reduce evaluation cost and mitigate potential contamination of the full set after public release. The bench presents substantial challenges to current frontier models, with even Claude-Sonnet-5 and GPT-5.6-Sol achieving Overall Pass@1 scores of only 58.54 and 57.14, respectively. Further analyses reveal clear differences across domains and capabilities, as well as persistent limitations in planning, constraint maintenance, and adaptive correction. OmniaBench provides a broad and diagnostic benchmark for characterizing the capability boundaries of general agents.",
      "published": "2026-07-16T13:38:07Z",
      "abstract_url": "http://arxiv.org/abs/2607.14989v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14989v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Explaining Process Control Optimisation Recommendations via GradientSHAP and Implicit Differentiation",
      "authors": [
        "Paul Darm",
        "Cem Alpturk",
        "Kenneth Ulrich",
        "William Duncan",
        "Ali Anwar",
        "Annalisa Riccardi"
      ],
      "abstract": "Automated optimisation is increasingly adopted in industrial processes, yet a trust gap persists between engineers who design these algorithms and operators who must act on their recommendations. Explainable AI methods like SHAP (SHapley Additive exPlanations) have transformed interpretability for machine learning predictions; optimisation outputs could benefit from similar techniques. We present an approach that integrates Implicit Function Theorem (IFT) based sensitivity analysis with SHAP attribution and narrative generation via Large Language Models (LLM), producing explanations tailored for operators. Our approach leverages IFT to compute exact parameter sensitivities $\\partial p^*/\\partial x$ from the optimality conditions, enabling efficient GradientSHAP computation. For an industrial High Pressure Grinding Roll (HPGR) control optimisation problem with 22 features, we achieve equivalent SHAP attributions (correlation $>$0.99 with KernelSHAP) with over 40$\\times$ speedup, enabling real-time natural language explanations. We validate on industrial scenarios and present feedback from domain experts on generated explanations.",
      "published": "2026-07-16T13:25:53Z",
      "abstract_url": "http://arxiv.org/abs/2607.14970v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14970v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Multi-Axis Max@K Reinforcement Learning for Representative Diversity in Text-to-Image Generation",
      "authors": [
        "Ku Onoda",
        "Paavo Parmas",
        "Hiroki Furuta",
        "Soichiro Nishimori",
        "Yuta Oshima",
        "Shohei Taniguchi",
        "Yutaka Matsuo"
      ],
      "abstract": "Text-to-image (T2I) models can synthesize realistic, prompt-aligned images, yet samples generated for the same prompt often cover only a small subset of visually distinct modes. This limits the diversity of images, and for person-centric prompts, can reflect or amplify demographic skew. We formalize this problem as coverage of a predefined set of semantically specified modes, which we call target-mode coverage. We then propose multi-axis max@K, a group-based reinforcement learning objective for improving such coverage in diffusion-based T2I models. Given a group of samples and one score per target category, multi-axis max@K first takes the maximum score across samples for each category and then sums these category-wise maxima. The resulting credit assignment gives a sample positive weight on a category only when it increases that category's group-wise maximum, allowing different samples to contribute to different categories. We first validate the credit-assignment mechanism on a synthetic mixture and on SD3.5-M using deterministic pixel-based color rewards. We then evaluate the same objective on perceived-appearance fairness. Across three automatic evaluators on held-out prompts, multi-axis max@K improves the Fairness Score by 0.23-0.36 relative to the base model, while maintaining image quality and text alignment.",
      "published": "2026-07-16T13:14:21Z",
      "abstract_url": "http://arxiv.org/abs/2607.14962v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14962v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control",
      "authors": [
        "Jihoon Hong",
        "Julian Skifstad",
        "Qiyue Dai",
        "Alice Chan",
        "Glen Chou"
      ],
      "abstract": "World Action Models (WAMs) enable semantically- and physically-informed control but are brittle under distribution shift. In this work, we use mechanistic interpretability to study how robustness-relevant perturbations are represented in WAM activation space. Comparing activations across successful and unsuccessful rollouts, we find some WAM architectures exhibit low-dimensional linear separability for robustness-critical features, while others do not. This motivates the use of contrastive activation directions for training-free WAM steering. We also show that local linearity in WAM activation dynamics enables efficient feedback steering via model-based optimal control, yielding World-Action Linear Quadratic Regulator (WA-LQR), a minimally-invasive reduced-order LQR controller. Via mechanistic evaluations, we predict strong steerability in the Cosmos-Policy and DiT4DiT models but weak steerability in LingBot-VA, consistent with steering intervention results. On Cosmos-Policy and DiT4DiT, WA-LQR generalizes contrastive directions to new tasks and improves robustness to camera, gripper, and visual-noise perturbations over unsteered and prompt steering baselines.",
      "published": "2026-07-16T12:52:49Z",
      "abstract_url": "http://arxiv.org/abs/2607.14943v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14943v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG",
        "eess.SY",
        "math.OC"
      ]
    },
    {
      "title": "A Minimal Interpretable Architecture for Zero-Shot Reconstruction of Dynamical Systems",
      "authors": [
        "Christoph Jürgen Hemmer",
        "Florian Plaswig",
        "Daniel Durstewitz"
      ],
      "abstract": "Recent foundation models (FMs) for zero-shot reconstruction of dynamical systems (DS) achieve strong out-of-domain generalization but provide little insight into the mechanisms that underlie their forecasts. Such an understanding could help to strip down overladen FM architectures to their bare essence and expose the minimal requirements for in-context learning in the DS domain. Toward this goal, here we iteratively reduce a recent powerful SOTA model for DS reconstruction, DynaMix (Hemmer & Durstewitz, 2025), to a minimal interpretable two-parameter form, which we call DynaBase. DynaBase produces forecasts through a linear blend of the current latent state and the nearest in-context neighbor and its temporal successor. Surprisingly, despite its extreme simplicity, DynaBase produces highly competitive zero-shot DS reconstructions across chaotic and cyclic systems, with a negligible parameter load, many orders of magnitude below that of other FMs. Even more, this extreme simplicity permits direct model optimization on DS reconstruction measures, as well as closed-form one-step analytical solutions on prediction MSE. Theoretical and empirical analysis of DynaBase further leads to a 1-parameter family of maps, with the context-parroting algorithm of (Zhang & Gilpin, 2026) recovered at one end, and chaotic (divergent but bounded) behavior at the other. We further show how different training strategies lead to models either optimal for short-term prediction or for DS reconstruction. Thus, DynaBase not only exposes the minimal mechanisms required for producing zero-shot DS reconstruction, but also reconciles within an accessible mathematical frame divergent observations in the literature.",
      "published": "2026-07-16T12:49:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.14937v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14937v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.DS",
        "nlin.CD"
      ]
    },
    {
      "title": "Random Logit Scaling: Defending Deep Neural Networks Against Black-Box Score-Based Adversarial Example Attacks",
      "authors": [
        "Hamid Dashtbani",
        "Mehdi Dousti Gandomani",
        "AmirMahdi Sadeghzadeh"
      ],
      "abstract": "Machine learning models are increasingly adapted in various domains. However, adversarial examples pose a significant threat to the reliable deployment of these models. In recent years, some powerful adversarial example attacks have been proposed for the fast and query-efficient generation of adversarial examples, even in black-box scenarios, highlighting the need for scalable, low-cost, and powerful defenses. In this work, we present two contributions to the domain of black-box adversarial example attacks and defenses. First, we propose Random Logit Scaling (RLS), a randomization-based defense against black-box score-based adversarial example attacks. RLS is a plug-and-play, post-processing defense that can be implemented on top of any existing ML model with minimal effort. The idea behind RLS is to confuse an attacker by outputting falsified scores resulting from randomly scaled logits while maintaining the model accuracy. We show that RLS significantly reduces the success rate of state-of-the-art black-box score-based attacks while preserving the accuracy and minimizing confidence score distortion compared to state-of-the-art randomization-based defenses. Second, we introduce a novel adaptive attack against AAA, a SOTA non-randomized black-box defense against black-box score-based attacks that also modifies output logits to confuse attackers, demonstrating its vulnerability against adaptive attacks.",
      "published": "2026-07-16T12:39:17Z",
      "abstract_url": "http://arxiv.org/abs/2607.14921v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14921v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Show Me How You Reason and I'll Tell You Who You Are: Reasoning Graphs for Robust LLM Authorship Attribution",
      "authors": [
        "Zlata Kikteva",
        "Artur Romazanov",
        "Annette Hautli-Janisz",
        "Ramon Ruiz-Dolz"
      ],
      "abstract": "Given the current trend to employ large language models (LLMs) in almost any imaginable context, LLM-generated text detection and authorship attribution have become a pressing issue. Prior work has primarily focused on surface-level linguistic features, an approach shown to be susceptible to paraphrasing and other obfuscation techniques. In this paper, we go beyond the linguistic surface, extracting and analysing reasoning structures in LLM-generated texts with the goal of capturing more complex signals of LLM authorship. We propose a graph neural network approach that leverages reasoning graphs extracted by an argument mining pipeline, demonstrating improved robustness and generalisation over a traditional Longformer baseline. Our approach outperforms the baseline by up to 27 percentage points under the obfuscation attacks such as paraphrasing and backtranslation, and 19 percentage points when evaluated on the texts generated by the unseen model versions, simulating real-world conditions in which new LLM versions are continuously released.",
      "published": "2026-07-16T12:25:16Z",
      "abstract_url": "http://arxiv.org/abs/2607.14905v1",
      "pdf_url": "https://arxiv.org/pdf/2607.14905v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    }
  ]
};
