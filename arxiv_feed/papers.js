const PAPERS_DATA = {
  "last_updated": "2026-04-28 03:53:11 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Personalized Worked Example Generation from Student Code Submissions using Pattern-based Knowledge Components",
      "authors": [
        "Griffin Pitts",
        "Muntasir Hoq",
        "Peter Brusilovsky",
        "Narges Norouzi",
        "Arto Hellas",
        "Juho Leinonen",
        "Bita Akram"
      ],
      "abstract": "Adaptive programming practice often relies on fixed libraries of worked examples and practice problems, which require substantial authoring effort and may not correspond well to the logical errors and partial solutions students produce while writing code. As a result, students may receive learning content that does not directly address the concepts they are working to understand, while instructors must either invest additional effort in expanding content libraries or accept a coarse level of personalization. We present an approach for knowledge-component (KC) guided educational content generation using pattern-based KCs extracted from student code. Given a problem statement and student submissions, our pipeline extracts recurring structural KC patterns from students' code through AST-based analysis and uses them to condition a generative model. In this study, we apply this approach to worked example generation, and compare baseline and KC-conditioned outputs through expert evaluation. Results suggest that KC-conditioned generation improves topical focus and relevance to learners' underlying logical errors, providing evidence that KC-based steering of generative models can support personalized learning at scale.",
      "published": "2026-04-27T17:56:56Z",
      "abstract_url": "http://arxiv.org/abs/2604.24758v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24758v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "cs.CY",
        "cs.ET",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Think from Multiple Thinkers",
      "authors": [
        "Nirmit Joshi",
        "Roey Magen",
        "Nathan Srebro",
        "Nikolaos Tsilivis",
        "Gal Vardi"
      ],
      "abstract": "We study learning with Chain-of-Thought (CoT) supervision from multiple thinkers, all of whom provide correct but possibly systematically different solutions, e.g., step-by-step solutions to math problems written by different thinkers, or step-by-step execution traces of different programs solving the same problem. We consider classes that are computationally easy to learn using CoT supervision from a single thinker, but hard to learn with only end-result supervision, i.e., without CoT (Joshi et al. 2025). We establish that, under cryptographic assumptions, learning can be hard from CoT supervision provided by two or a few different thinkers, in passive data-collection settings. On the other hand, we provide a generic computationally efficient active learning algorithm that learns with a small amount of CoT data per thinker that is completely independent of the target accuracy $\\varepsilon$, a moderate number of thinkers that scales as $\\log \\frac{1}{\\varepsilon}\\log \\log \\frac{1}{\\varepsilon}$, and sufficient passive end-result data that scales as $\\frac{1}{\\varepsilon}\\cdot poly\\log\\frac{1}{\\varepsilon}$.",
      "published": "2026-04-27T17:43:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.24737v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24737v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CC",
        "stat.ML"
      ]
    },
    {
      "title": "Scalable Hyperparameter-Divergent Ensemble Training with Automatic Learning Rate Exploration for Large Models",
      "authors": [
        "Hailing Cheng",
        "Tao Huang",
        "Chen Zhu",
        "Antonio Alonso"
      ],
      "abstract": "Training large neural networks with data-parallel stochastic gradient descent allocates N GPU replicas to compute effectively identical updates -- a practice that leaves the rich space of learning rate configurations entirely unexplored during training. We propose Hyperparameter-Divergent Ensemble Training (HDET), a method that repurposes these replicas for simultaneous learning rate exploration at negligible communication overhead. HDET operates in alternating phases: a fan-out stage in which replicas train independently under a structured, symmetric spread of learning rates, and a converge stage in which parameters are averaged across all replicas via AllReduce every T steps. Building on this ensemble substrate, we further propose an automatic learning rate (auto-LR) controller that treats the relative training loss across replicas as a performance signal, updating the shared base schedule toward higher-performing configurations via a momentum-based gradient-free meta-update. The combined method produces a self-adapting learning rate schedule that improves both optimization quality and generalization without additional hyperparameter sweeps or training budget. Crucially, the framework generalizes beyond learning rate: any scalar hyperparameter that does not alter model architecture -- such as dropout rate, attention scale temperature, or weight-decay coefficient -- can be explored across replicas using the same fan-out/converge protocol, with inter-replica loss differences serving as zero-order hypergradients that guide the search direction. HDET is implemented as a drop-in replacement for PyTorch's OneCycleLR scheduler, requiring no changes to model architecture, optimizer, or data pipeline.",
      "published": "2026-04-27T17:17:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.24708v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24708v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Defective Task Descriptions in LLM-Based Code Generation: Detection and Analysis",
      "authors": [
        "Amal Akli",
        "Mike Papadakis",
        "Maxime Cordy",
        "Yves Le Traon"
      ],
      "abstract": "Large language models are widely used for code generation, yet they rely on an implicit assumption that the task descriptions are sufficiently detailed and well-formed. However, in practice, users may provide defective descriptions, which can have a strong effect on code correctness. To address this issue, we develop SpecValidator, a lightweight classifier based on a small model that has been parameter-efficiently finetuned, to automatically detect task description defects. We evaluate SpecValidator on three types of defects, Lexical Vagueness, Under-Specification and Syntax-Formatting on 3 benchmarks with task descriptions of varying structure and complexity. Our results show that SpecValidator achieves defect detection of F1 = 0.804 and MCC = 0.745, significantly outperforming GPT-5-mini (F1 = 0.469 and MCC = 0.281) and Claude Sonnet 4 (F1 = 0.518 and MCC = 0.359). Perhaps more importantly, our analysis indicates that SpecValidator can generalize to unseen issues and detect unknown Under-Specification defects in the original (real) descriptions of the benchmarks used. Our results also show that the robustness of LLMs in task description defects depends primarily on the type of defect and the characteristics of the task description, rather than the capacity of the model, with Under-Specification defects being the most severe. We further found that benchmarks with richer contextual grounding, such as LiveCodeBench, exhibit substantially greater resilience, highlighting the importance of structured task descriptions for reliable LLM-based code generation.",
      "published": "2026-04-27T17:07:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.24703v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24703v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Green Shielding: A User-Centric Approach Towards Trustworthy AI",
      "authors": [
        "Aaron J. Li",
        "Nicolas Sanchez",
        "Hao Huang",
        "Ruijiang Dong",
        "Jaskaran Bains",
        "Katrin Jaradeh",
        "Zhen Xiang",
        "Bo Li",
        "Feng Liu",
        "Aaron Kornblith",
        "Bin Yu"
      ],
      "abstract": "Large language models (LLMs) are increasingly deployed, yet their outputs can be highly sensitive to routine, non-adversarial variation in how users phrase queries, a gap not well addressed by existing red-teaming efforts. We propose Green Shielding, a user-centric agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts model behavior. We operationalize this agenda through the CUE criteria: benchmarks with authentic Context, reference standards and metrics that capture true Utility, and perturbations that reflect realistic variations in the Elicitation of model behavior. Guided by the PCS framework and developed with practicing physicians, we instantiate Green Shielding in medical diagnosis through HealthCareMagic-Diagnosis (HCM-Dx), a benchmark of patient-authored queries, together with structured reference diagnosis sets and clinically grounded metrics for evaluating differential diagnosis lists. We also study perturbation regimes that capture routine input variation and show that prompt-level factors shift model behavior along clinically meaningful dimensions. Across multiple frontier LLMs, these shifts trace out Pareto-like tradeoffs. In particular, neutralization, which removes common user-level factors while preserving clinical content, increases plausibility and yields more concise, clinician-like differentials, but reduces coverage of highly likely and safety-critical conditions. Together, these results show that interaction choices can systematically shift task-relevant properties of model outputs and support user-facing guidance for safer deployment in high-stakes domains. Although instantiated here in medical diagnosis, the agenda extends naturally to other decision-support settings and agentic AI systems.",
      "published": "2026-04-27T17:04:17Z",
      "abstract_url": "http://arxiv.org/abs/2604.24700v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24700v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Leveraging LLMs for Multi-File DSL Code Generation: An Industrial Case Study",
      "authors": [
        "Sivajeet Chand",
        "Kevin Nguyen",
        "Peter Kuntz",
        "Alexander Pretschner"
      ],
      "abstract": "Large language models (LLMs) perform strongly on general-purpose code generation, yet their applicability to enterprise domain-specific languages (DSLs) remains underexplored, especially for repository-scale change generation spanning multiple files and folder structures from a single natural-language (NL) instruction. We report an industrial case study at BMW that adapts code-oriented LLMs to generate and modify project-root DSL artifacts for an Xtext-based DSL that drives downstream Java/TypeScript code generation. We develop an end-to-end pipeline for dataset construction, multi-file task representation, model adaptation, and evaluation. We encode DSL folder hierarchies as structured, path-preserving JSON, allowing single-response generation at repository scale and learning cross-file dependencies. We evaluate two instruction-tuned code LLMs (Qwen2.5-Coder and DeepSeek-Coder, 7B) under three configurations: baseline prompting, one-shot in-context learning, and parameter-efficient fine-tuning (QLoRA). Beyond standard similarity metrics, we introduce task-specific measures that assess edit correctness and repository structural fidelity. Fine-tuning yields the most significant gains across models and metrics, achieving high exact-match accuracy, substantial edit similarity, and structural fidelity of 1.00 on our held-out set for multi-file outputs. At the same time, one-shot in-context learning provides smaller but consistent improvements over baseline prompting. We further validate practical utility via an expert developer survey and an execution-based check using the existing code generator.",
      "published": "2026-04-27T16:38:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.24678v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24678v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications",
      "authors": [
        "Zhenyu Zhao",
        "Aparna Balagopalan",
        "Adi Agrawal",
        "Dilshoda Yergasheva",
        "Waseem Alshikh",
        "Daniel M. Bikel"
      ],
      "abstract": "Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM.",
      "published": "2026-04-27T16:27:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.24668v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24668v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Benchmarking Source-Sensitive Reasoning in Turkish: Humans and LLMs under Evidential Trust Manipulation",
      "authors": [
        "Sercan Karakaş",
        "Yusuf Şimşek"
      ],
      "abstract": "This paper investigates whether source trustworthiness shapes Turkish evidential morphology and whether large language models (LLMs) track this sensitivity. We study the past-domain contrast between -DI and -mIs in controlled cloze contexts where the information source is overtly external, while only its perceived reliability is manipulated (High-Trust vs. Low-Trust). In a human production experiment, native speakers of Turkish show a robust trust effect: High-Trust contexts yield relatively more -DI, whereas Low-Trust contexts yield relatively more -mIs, with the pattern remaining stable across sensitivity analyses. We then evaluate 10 LLMs in three prompting paradigms (open gap-fill, explicit past-tense gap-fill, and forced-choice A/B selection). LLM behavior is highly model- and prompt-dependent: some models show weak or local trust-consistent shifts, but effects are generally unstable, often reversed, and frequently overshadowed by output-compliance problems and strong base-rate suffix preferences. The results provide new evidence for a trust-/commitment-based account of Turkish evidentiality and reveal a clear human-LLM gap in source-sensitive evidential reasoning.",
      "published": "2026-04-27T16:26:20Z",
      "abstract_url": "http://arxiv.org/abs/2604.24665v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24665v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "AgentWard: A Lifecycle Security Architecture for Autonomous AI Agents",
      "authors": [
        "Yixiang Zhang",
        "Xinhao Deng",
        "Jiaqing Wu",
        "Yue Xiao",
        "Ke Xu",
        "Qi Li"
      ],
      "abstract": "Autonomous AI agents extend large language models into full runtime systems that load skills, ingest external content, maintain memory, plan multi-step actions, and invoke privileged tools. In such systems, security failures rarely remain confined to a single interface; instead, they can propagate across initialization, input processing, memory, decision-making, and execution, often becoming apparent only when harmful effects materialize in the environment. This paper presents AgentWard, a lifecycle-oriented, defense-in-depth architecture that systematically organizes protection across these five stages. AgentWard integrates stage-specific, heterogeneous controls with cross-layer coordination, enabling threats to be intercepted along their propagation paths while safeguarding critical assets. We detail the design rationale and architecture of five coordinated protection layers, and implement a plugin-native prototype on OpenClaw to demonstrate practical feasibility. This perspective provides a concrete blueprint for structuring runtime security controls, managing trust propagation, and enforcing execution containment in autonomous AI agents. Our code is available at https://github.com/FIND-Lab/AgentWard .",
      "published": "2026-04-27T16:22:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.24657v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24657v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "DepthKV: Layer-Dependent KV Cache Pruning for Long-Context LLM Inference",
      "authors": [
        "Zahra Dehghanighobadi",
        "Asja Fischer"
      ],
      "abstract": "Long-context reasoning is a critical capability of large language models (LLMs), enabling applications such as long-document understanding, summarization, and code generation. However, efficient autoregressive inference relies on the key-value (KV) cache, whose memory footprint grows linearly with sequence length, leading to a major memory bottleneck. To mitigate this overhead, KV cache pruning methods discard cached tokens with low attention scores during inference. Most existing methods apply a uniform pruning ratio across layers, implicitly assuming that all layers contribute equally to overall model performance. We show that this assumption is suboptimal, as layers differ significantly in their sensitivity to pruning. We propose DepthKV, a layer-dependent pruning framework that allocates a fixed global KV budget across layers based on their sensitivity, rather than using a uniform allocation. Across multiple models and tasks, DepthKV consistently outperforms uniform pruning at the same global pruning ratio, demonstrating more effective utilization of the KV cache budget through layer-dependent allocation.",
      "published": "2026-04-27T16:15:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.24647v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24647v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology",
      "authors": [
        "Soyeon Kim",
        "Cheongwoong Kang",
        "Myeongjin Lee",
        "Eun-Chul Chang",
        "Jaedeok Lee",
        "Jaesik Choi"
      ],
      "abstract": "The development of practical (multimodal) large language model assistants for Korean weather forecasters is hindered by the absence of a multidimensional, expert-level evaluation framework grounded in authoritative sources. To address this, we introduce K-MetBench, a diagnostic benchmark grounded in national qualification exams. It exposes critical gaps across four dimensions: expert visual reasoning of charts, logical validity via expert-verified rationales, Korean-specific geo-cultural comprehension, and fine-grained domain analysis. Our evaluation of 55 models reveals a profound modality gap in interpreting specialized diagrams and a reasoning gap where models hallucinate logic despite correct predictions. Crucially, Korean models outperform significantly larger global models in local contexts, demonstrating that parameter scaling alone cannot resolve cultural dependencies. K-MetBench serves as a roadmap for developing reliable, culturally aware expert AI agents. The dataset is available at https://huggingface.co/datasets/soyeonbot/K-MetBench .",
      "published": "2026-04-27T16:13:14Z",
      "abstract_url": "http://arxiv.org/abs/2604.24645v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24645v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Cortex-Inspired Continual Learning: Unsupervised Instantiation and Recovery of Functional Task Networks",
      "authors": [
        "Kevin McKee",
        "Thomas Hazy",
        "Yicong Zheng",
        "Zacharie Bugaud",
        "Thomas Miconi"
      ],
      "abstract": "Block-sequential continual learning demands that a single model both protect prior solutions from catastrophic forgetting and efficiently infer at inference time which prior solution matches the current input without task labels. We present Functional Task Networks (FTN), a parameter-isolation method inspired by structural and dynamical motifs found in the mammalian neocortex. Similar to mixture-of-experts, this method uses a high dimensional, self-organizing binary mask over a large population of small but deep networks, inspired by dendritic models of pyramidal neurons. The mask is produced by a three-stage procedure: (1) gradient descent on a continuous mask identifies task-relevant neurons, (2) a smoothing kernel biases the result toward spatial contiguity, (3) and k-winner-take-all binarizes the resulting group at a fixed capacity budget. Like mixture-of-experts, each neuron is an independent deep network, so disjoint masks give exactly disjoint gradient updates, providing structural guarantees against catastrophic forgetting. This three-stage procedure recovers the sub-network of a previously-trained task in a single gradient step, providing unsupervised task segmentation at inference time. We test it on three continual-learning benchmarks: (1) a synthetic multi-task classification/regression generator, (2) MNIST with shuffled class labels (pure concept shift), and (3) Permuted MNIST (domain shift). On all three, FTN with fine grained smoothing (FTN-Slow) results in nearly zero forgetting. FTN with a large kernel and only 2 iterations of smoothing (FTN-Fast) trades off some retention for increased speed. We show that the spatial organization mechanism reduces the effective mask search from the combinatorial top-k subset problem in O(C(H,K)) to the complexity of a near-linear scan in O(H) over compact cortical neighborhoods, which is parallelized by the gradient-based update.",
      "published": "2026-04-27T16:06:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.24637v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24637v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.NC"
      ]
    },
    {
      "title": "Meta-CoT: Enhancing Granularity and Generalization in Image Editing",
      "authors": [
        "Shiyi Zhang",
        "Yiji Cheng",
        "Tiankai Hang",
        "Zijin Yin",
        "Runze He",
        "Yu Xu",
        "Wenxun Dai",
        "Yunlong Lin",
        "Chunyu Wang",
        "Qinglin Lu",
        "Yansong Tang"
      ],
      "abstract": "Unified multi-modal understanding/generative models have shown improved image editing performance by incorporating fine-grained understanding into their Chain-of-Thought (CoT) process. However, a critical question remains underexplored: what forms of CoT and training strategy can jointly enhance both the understanding granularity and generalization? To address this, we propose Meta-CoT, a paradigm that performs a two-level decomposition of any single-image editing operation with two key properties: (1) Decomposability. We observe that any editing intention can be represented as a triplet - (task, target, required understanding ability). Inspired by this, Meta-CoT decomposes both the editing task and the target, generating task-specific CoT and traversing editing operations on all targets. This decomposition enhances the model's understanding granularity of editing operations and guides it to learn each element of the triplet during training, substantially improving the editing capability. (2) Generalizability. In the second decomposition level, we further break down editing tasks into five fundamental meta-tasks. We find that training on these five meta-tasks, together with the other two elements of the triplet, is sufficient to achieve strong generalization across diverse, unseen editing tasks. To further align the model's editing behavior with its CoT reasoning, we introduce the CoT-Editing Consistency Reward, which encourages more accurate and effective utilization of CoT information during editing. Experiments demonstrate that our method achieves an overall 15.8% improvement across 21 editing tasks, and generalizes effectively to unseen editing tasks when trained on only a small set of meta-tasks. Our code, benchmark, and model are released at https://shiyi-zh0408.github.io/projectpages/Meta-CoT/",
      "published": "2026-04-27T15:52:48Z",
      "abstract_url": "http://arxiv.org/abs/2604.24625v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24625v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.MM"
      ]
    },
    {
      "title": "XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation",
      "authors": [
        "Zhuoling Li",
        "Ha Linh Hong Tran Nguyen",
        "Valeria Bladinieres",
        "Maxim Romanovsky"
      ],
      "abstract": "Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our ability to understand how specific pieces of structured knowledge influence the final output. Existing explainability (XAI) methods for RAG systems, designed for text-based retrieval, are limited to interpreting an LLM response through the relational structures among knowledge components, creating a critical gap in transparency and trustworthiness. To address this, we introduce XGRAG, a novel framework that generates causally grounded explanations for GraphRAG systems by employing graph-based perturbation strategies, to quantify the contribution of individual graph components on the model answer. We conduct extensive experiments comparing XGRAG against RAG-Ex, an XAI baseline for standard RAG, and evaluate its robustness across various question types, narrative structures and LLMs. Our results demonstrate a 14.81% improvement in explanation quality over the baseline RAG-Ex across NarrativeQA, FairyTaleQA, and TriviaQA, evaluated by F1-score measuring alignment between generated explanations and original answers. Furthermore, XGRAG explanations exhibit a strong correlation with graph centrality measures, validating its ability to capture graph structure. XGRAG provides a scalable and generalizable approach towards trustworthy AI through transparent, graph-based explanations that enhance the interpretability of RAG systems.",
      "published": "2026-04-27T15:52:20Z",
      "abstract_url": "http://arxiv.org/abs/2604.24623v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24623v1",
      "categories": [
        "cs.AI",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Route Queries to Heads for Attention-based Re-ranking with Large Language Models",
      "authors": [
        "Yuxing Tian",
        "Fengran Mo",
        "Zhiqi Huang",
        "Weixu Zhang",
        "Jian-Yun Nie"
      ],
      "abstract": "Large Language Models (LLMs) have recently been explored as fine-grained zero-shot re-rankers by leveraging attention signals to estimate document relevance. However, existing methods either aggregate attention signals across all heads or rely on a statically selected subset identified by heuristic rules. This solution can be suboptimal because the informative heads can vary across queries or domains. Moreover, naively combining multiple heads can degrade performance due to redundancy or conflicting ranking signals. In this paper, we propose a query-dependent head selection method, RouteHead, for attention-based re-ranking with LLMs. Specifically, we learn a lightweight router that can map each query to an optimal head set, and relevance scores are computed by aggregating attention signals only from these heads. Since query-to-head optimal labels are unavailable, we first construct pseudo labels via an offline search. The router represents each head with a learnable embedding and represents each query using an embedding extracted from the hidden states of the frozen LLM. Then it is trained on the pseudo labels with a sparsity regularizer. Experiments on diverse benchmarks and multiple LLM backbones show that the proposed method consistently outperforms strong baselines.",
      "published": "2026-04-27T15:36:54Z",
      "abstract_url": "http://arxiv.org/abs/2604.24608v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24608v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Skill Retrieval Augmentation for Agentic AI",
      "authors": [
        "Weihang Su",
        "Jianming Long",
        "Qingyao Ai",
        "Yichen Tang",
        "Changyue Wang",
        "Yiteng Tu",
        "Yiqun Liu"
      ],
      "abstract": "As large language models (LLMs) evolve into agentic problem solvers, they increasingly rely on external, reusable skills to handle tasks beyond their native parametric capabilities. In existing agent systems, the dominant strategy for incorporating skills is to explicitly enumerate available skills within the context window. However, this strategy fails to scale: as skill corpora expand, context budgets are consumed rapidly, and the agent becomes markedly less accurate in identifying the right skill. To this end, this paper formulates Skill Retrieval Augmentation (SRA), a new paradigm in which agents dynamically retrieve, incorporate, and apply relevant skills from large external skill corpora on demand. To make this problem measurable, we construct a large-scale skill corpus and introduce SRA-Bench, the first benchmark for decomposed evaluation of the full SRA pipeline, covering skill retrieval, skill incorporation, and end-task execution. SRA-Bench contains 5,400 capability-intensive test instances and 636 manually constructed gold skills, which are mixed with web-collected distractor skills to form a large-scale corpus of 26,262 skills. Extensive experiments show that retrieval-based skill augmentation can substantially improve agent performance, validating the promise of the paradigm. At the same time, we uncover a fundamental gap in skill incorporation: current LLM agents tend to load skills at similar rates, regardless of whether a gold skill is retrieved or whether the task actually requires external capabilities. This shows that the bottleneck in skill augmentation lies not only in retrieval but also in the base model's ability to determine which skill to load and when external loading is actually needed. These findings position SRA as a distinct research problem and establish a foundation for the scalable augmentation of capabilities in future agent systems.",
      "published": "2026-04-27T15:19:59Z",
      "abstract_url": "http://arxiv.org/abs/2604.24594v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24594v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Towards Lawful Autonomous Driving: Deriving Scenario-Aware Driving Requirements from Traffic Laws and Regulations",
      "authors": [
        "Bowen Jian",
        "Rongjie Yu",
        "Hong Wang",
        "Liqiang Wang",
        "Zihang Zou"
      ],
      "abstract": "Driving in compliance with traffic laws and regulations is a basic requirement for human drivers, yet autonomous vehicles (AVs) can violate these requirements in diverse real-world scenarios. To encode law compliance into AV systems, conventional approaches use formal logic languages to explicitly specify behavioral constraints, but this process is labor-intensive, hard to scale, and costly to maintain. With recent advances in artificial intelligence, it is promising to leverage large language models (LLMs) to derive legal requirements from traffic laws and regulations. However, without explicitly grounding and reasoning in structured traffic scenarios, LLMs often retrieve irrelevant provisions or miss applicable ones, yielding imprecise requirements. To address this, we propose a novel pipeline that grounds LLM reasoning in a traffic scenario taxonomy through node-wise anchors that encode hierarchical semantics. On Chinese traffic laws and OnSite dataset (5,897 scenarios), our method improves law-scenario matching by 29.1\\% and increases the accuracy of derived mandatory and prohibitive requirements by 36.9\\% and 38.2\\%, respectively. We further demonstrate real-world applicability by constructing a law-compliance layer for AV navigation and developing an onboard, real-time compliance monitor for in-field testing, providing a solid foundation for future AV development, deployment, and regulatory oversight.",
      "published": "2026-04-27T14:49:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.24562v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24562v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.CY"
      ]
    },
    {
      "title": "Hierarchical Behaviour Spaces",
      "authors": [
        "Michael Tryfan Matthews",
        "Anssi Kanervisto",
        "Jakob Foerster",
        "Pierluca D'Oro",
        "Scott Fujimoto",
        "Mikael Henaff"
      ],
      "abstract": "Recent work in hierarchical reinforcement learning has shown success in scaling to billions of timesteps when learning over a set of predefined option reward functions. We show that, instead of using a single reward function per option, the reward functions can be effectively used to induce a space of behaviours, by letting the controller specify linear combinations over reward functions, allowing a more expressive set of policies to be represented. We call this method Hierarchical Behaviour Spaces (HBS). We evaluate HBS on the NetHack Learning Environment, demonstrating strong performance. We conduct a series of experiments and determine that, perhaps going against conventional wisdom, the benefits of hierarchy in our method come from increased exploration rather than long term reasoning.",
      "published": "2026-04-27T14:47:22Z",
      "abstract_url": "http://arxiv.org/abs/2604.24558v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24558v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "GradMAP: Gradient-Based Multi-Agent Proximal Learning for Grid-Edge Flexibility",
      "authors": [
        "Yihong Zhou",
        "Hongtai Zeng",
        "Thomas Morstyn"
      ],
      "abstract": "Coordinating large populations of grid-edge devices requires learning methods that remain fully decentralised in deployment while still respecting three-phase AC distribution-network physics. This paper proposes gradient-based multi-agent proximal learning (GradMAP) to address this challenge. GradMAP trains independent neural-network policies for each agent without any parameter sharing, and each agent uses only its own local observation for online decision-making without communication. During offline training, GradMAP embeds a differentiable three-phase AC power-flow model in a primal-dual learning loop and uses implicit differentiation to propagate exact network-constraint violations to update the policy parameters. To speed up training, GradMAP reuses expensive environment gradients through a proximal surrogate within a trust region defined in the more direct policy-output (action) space, instead of the probability distribution space used in other works, such as PPO. In case studies with 1,000 agents managing batteries, heat pumps, and controllable generators on the IEEE 123-bus feeder, GradMAP learns decentralised policies that minimise three-phase AC load-flow constraint violations within 15 minutes of training on a single workstation-class NVIDIA RTX PRO 5000 Blackwell 48GB GPU. This is a 3--5x training speed-up over gradient-based self-supervised learning benchmarks and substantially better training efficiency than multi-agent reinforcement-learning benchmarks. In out-of-sample tests, GradMAP also delivers among the lowest operating cost and constraint violations.",
      "published": "2026-04-27T14:43:02Z",
      "abstract_url": "http://arxiv.org/abs/2604.24549v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24549v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator",
      "authors": [
        "Alessio Sordo",
        "Lingxiao Du",
        "Meeka-Hanna Lenisa",
        "Evgeny Bogdanov",
        "Maxim Romanovsky"
      ],
      "abstract": "The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for manual creation. Existing automated benchmarking methods are often limited by relying on pre-existing data, poor scalability, single-domain focus, and lack of multilingual support. We present STELLAR-E - a fully automated system to generate high-quality synthetic datasets of custom size, using minimal human inputs without depending on existing datasets. The system is structured in two stages: (1) We modify the TGRT Self-Instruct framework to create a synthetic data engine that enables controllable, custom synthetic dataset generation, and (2) an evaluation pipeline incorporating statistical and LLM-based metrics to assess the applicability of the synthetic dataset for LLM-based application evaluations. The synthetic datasets reach an average difference of +5.7% in terms of LLM-as-a-judge scores against existing language-specific benchmarks, demonstrating comparable quality for comprehensive assessment of big and small LLMs. While real datasets remain slightly more challenging for LLMs especially for smaller models, this work establishes a scalable and domain-adaptable benchmarking framework that supports fair evaluation of LLM applications, offering a faster alternative to manual approaches and enabling high-efficiency automated quality assurance cycles.",
      "published": "2026-04-27T14:39:41Z",
      "abstract_url": "http://arxiv.org/abs/2604.24544v1",
      "pdf_url": "https://arxiv.org/pdf/2604.24544v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    }
  ]
};
