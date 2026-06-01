const PAPERS_DATA = {
  "last_updated": "2026-06-01 05:02:19 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards",
      "authors": [
        "Nianyi Lin",
        "Jiajie Zhang",
        "Lei Hou",
        "Juanzi Li"
      ],
      "abstract": "Long-context reasoning remains a central challenge for large language models, which often fail to locate and integrate key information in extensive distracting content. Reinforcement learning with verifiable rewards (RLVR) has shown promise for this task, yet existing methods are limited by low-confusability distractors and sparse, outcome-only reward signals that cannot supervise intermediate reasoning steps. To address these issues, we introduce \\textsc{LongTraceRL}. For data construction, we generate multi-hop questions via knowledge graph random walks and leverage search agent trajectories to build \\emph{tiered distractors}: documents the agent read but did not cite (high confusability) and documents that appeared in search results but were never opened (low confusability), producing training contexts that are far more challenging than those built by random sampling or one-shot search. For reward design, we propose a \\emph{rubric reward} that uses the gold entities along each reasoning chain as fine-grained, entity-level process supervision. This rubric reward is applied only to responses with correct final answers (positive-only strategy), distinguishing the reasoning quality among correct responses and preventing reward hacking. Experiments on three reasoning LLMs (4B--30B) across five long-context benchmarks demonstrate that \\textsc{LongTraceRL} consistently outperforms strong baselines and encourages comprehensive, evidence-grounded reasoning. Codes, datasets and models are available at \\href{https://github.com/THU-KEG/LongTraceRL}{https://github.com/THU-KEG/LongTraceRL}.",
      "published": "2026-05-29T17:51:40Z",
      "abstract_url": "http://arxiv.org/abs/2605.31584v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31584v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Positional versus Symbolic Attention Heads: Learning Dynamics, RoPE Geometry, and Length Generalization",
      "authors": [
        "Felipe Urrutia",
        "Juan José Alegría",
        "Cinthia Sanchez Macias",
        "Jorge Salas",
        "Cristian B. Calderon",
        "Cristobal Rojas"
      ],
      "abstract": "Transformer-based language models are widespread in today's society. As such, understanding the mechanisms by which they solve structured tasks and predicting how they may behave in novel scenarios is of great importance for safe deployment. We study the learning dynamics of attention heads in a controlled setting by training a decoder-only Transformer (GPT-J) on two structurally equivalent multi-hop reasoning tasks: a number task requiring positional reasoning and a letter task requiring symbolic reasoning. Using a recently introduced metric that classifies attention-head behavior as positional or symbolic for a given prompt, we show that successful learning is associated with the emergence of pure heads, i.e., heads that express themselves as either positional or symbolic. Despite the tasks' structural equivalence, they impose different mechanistic demands: the number task requires both positional and symbolic heads, whereas the letter task requires only symbolic heads. We then identify the computational roles of these heads, characterize the basic functions they implement, and give theoretical constructions showing how single-layer RoPE-based attention can realize these functions through geometrically interpretable query, key, and value operations. This analysis yields a quantitative separation between positional and symbolic mechanisms in their robustness to longer sequences, formalized through a novel notion of discrepancy. We empirically validate the resulting predictions in both controlled and real-world models, showing that symbolic mechanisms extrapolate more reliably to longer sequences while positional mechanisms face sharper limitations.",
      "published": "2026-05-29T17:22:04Z",
      "abstract_url": "http://arxiv.org/abs/2605.31558v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31558v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "RayDer: Scalable Self-Supervised Novel View Synthesis from Real-World Video",
      "authors": [
        "Ulrich Prestel",
        "Stefan Andreas Baumann",
        "Nick Stracke",
        "Björn Ommer"
      ],
      "abstract": "Self-supervised novel view synthesis (NVS) remains challenging to scale, despite the abundance of video data, largely due to the brittleness of training on realistic videos and the hard-to-predict scaling behavior of multi-network system designs. We introduce RayDer, a unified, feed-forward transformer that consolidates camera estimation, scene reconstruction, and rendering into a single backbone, turning self-supervised NVS into a well-posed single-model scaling problem. A minimal dynamic state, treated as a nuisance factor, absorbs time-varying content and enables stable training on unconstrained real-world video. Importantly, RayDer keeps static-scene NVS as its target task: dynamic content is leveraged purely as scalable supervision, not reconstructed as in dynamic-scene (4D) NVS. Across multiple model sizes and orders of magnitude in data, RayDer exhibits clean power-law scaling with data and compute, and outperforms static-scene data mixtures. On a large number of benchmarks, RayDer achieves strong zero-shot open-set performance competitive with state-of-the-art supervised approaches. Project Page: https://compvis.github.io/rayder",
      "published": "2026-05-29T16:50:27Z",
      "abstract_url": "http://arxiv.org/abs/2605.31535v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31535v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "If LLMs Have Human-Like Attributes, Then So Does Age of Empires II",
      "authors": [
        "Adrian de Wynter"
      ],
      "abstract": "Much research has been carried out on large language models (LLMs) and LLM-powered agentic workflows. However, many works within the field state emergence of, ascribe to, or assume, generalised anthropomorphic attributes to them (e.g., morality or understanding of natural language). Our goal is not to argue in favour or against the existence of these attributes, but to point out that these conclusions could be incorrect. For this we build and train a simple neural network on the videogame Age of Empires II, and note that any entity in a sufficiently-powerful substrate, such as LEGO or the Greater Boston Area, could also present such attributes. Hence, the purported anthropomorphic attributes of LLMs are empirically non-unique: although some properties (e.g., responses to prompts) could remain constant, others, such as the interpretation of their perceived behaviour, might change with the substrate. Thus, any empirically-grounded discussion requires explicit measurement criteria; otherwise the interpretation is left to the representation. We then show that assuming that these attributes exist or not in a system, independent of the substrate and in a generalised way, leads to either circular or uninformative conclusions, regardless of the experimenter's viewpoint on the subject. Finally we propose a 'null' assumption, where one assumes LLM non-uniqueness instead of assuming anthropomorphic attributes to set up an experiment, along with examples of it. We also discuss potential objections to our work, briefly survey the field, and prove that \\textit{Age of Empires II} is functionally- and Turing-complete.",
      "published": "2026-05-29T16:31:31Z",
      "abstract_url": "http://arxiv.org/abs/2605.31514v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31514v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Skill Reuse as Compression in Agentic RL",
      "authors": [
        "Zhikun Xu",
        "Yu Feng",
        "Jacob Dineen",
        "Taiwei Shi",
        "Jieyu Zhao",
        "Ben Zhou"
      ],
      "abstract": "Large language model agents trained with reinforcement learning (RL) often learn brittle, task-specific shortcuts. We hypothesize that agents generalize better when their successful trajectories are structurally compressible, decomposed into a small set of reusable abstract patterns. To formalize this, we introduce ReuseRL, which grounds agentic RL in the Minimum Description Length (MDL) principle. ReuseRL extracts a shared skill dictionary from successful trajectories and augments the RL objective with a segmentation cost, explicitly penalizing idiosyncratic behaviors that encode poorly. We prove a PAC-Bayes generalization bound for this compression penalty. Across ALFWorld, TextWorld-Cooking, and Countdown-Stepwise, ReuseRL improves in- and out-of-distribution success over vanilla GRPO and strong round-length baselines.",
      "published": "2026-05-29T16:28:34Z",
      "abstract_url": "http://arxiv.org/abs/2605.31509v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31509v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "On Efficient Scaling of GNNs via IO-Aware Layers Implementations",
      "authors": [
        "Daria Fomina",
        "Daniil Krasylnikov",
        "Alexey Boykov",
        "Andrey Dolgovyazov",
        "Vyacheslav Zhdanovskiy",
        "Fedor Velikonivtsev"
      ],
      "abstract": "Graph Neural Networks (GNNs) are bottlenecked by sparse, irregular memory access. Popular frameworks such as DGL and PyTorch Geometric support general message passing, but complex layers often materialize edge-wise intermediates, increasing memory traffic and limiting scalability on large graphs. We take an I/O- and arithmetic-intensity--centric view and show that widely used layers fall into three kernel families: SpMM-based convolutions, reduction-based aggregations, and attention-based layers (GATv2/Graph Transformer). For each family, we develop GPU kernels that reduce data movement, improve locality, and remain robust across realistic graphs. We also study graph reordering and find that its impact depends on the kernel mapping: it benefits neighbor-parallel (gather-dominated) kernels more consistently than feature-parallel designs. Empirically, our fused attention kernels reach up to $\\textbf{3.9}\\times$ speedup for Graph Transformer (median $\\textbf{1.6}\\times$), with Tensor Core (block-sparse) variants up to $\\textbf{7.3}\\times$ on locally dense graphs; for GATv2 we reach up to $\\textbf{8.5}\\times$ speedup (median $\\textbf{2.0}\\times$) while reducing peak memory by up to $\\textbf{76}\\times$ (median $\\textbf{6}\\times$). Our degree-aware reduction kernels achieve up to $\\textbf{10}\\times$ speedup (median $\\textbf{2.6}\\times$). For SpMM-based layers, properly cached cuSPARSE achieves up to $\\textbf{8}\\times$ speedup over DGL and outperforms evaluated custom baselines in the majority of evaluations. We release our implementations as drop-in replacements to support reproducible, hardware-aware GNN acceleration.",
      "published": "2026-05-29T16:22:45Z",
      "abstract_url": "http://arxiv.org/abs/2605.31500v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31500v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LinTree: Improving LLM Reasoning with Explicitly Structured Search Histories",
      "authors": [
        "Liwei Kang",
        "Yee Whye Teh",
        "Wee Sun Lee"
      ],
      "abstract": "Large language models (LLMs) often solve reasoning problems by generating intermediate traces that explore and revise partial solutions. From a search perspective, these traces can be viewed as linearized search trees, where the model extends a partial solution, abandons it when it fails, and backtracks to try alternatives. Compared with traditional heuristic-guided search, such a policy has a potential advantage: it conditions on the whole search trace rather than only on the current local state. We first test whether LLMs utilize this advantage by comparing trace-conditioned reasoning policies against best-first search equipped with an LLM heuristic that only observes the current local state. Across three controlled reasoning environments, Blocks World, grid Navigation, and Sokoban, we find that raw access to search history alone is not enough to reliably outperform heuristic search. We then study one possible reason: in LLM reasoning traces, the underlying search tree is only implicitly represented, and when the model backtracks or switches branches, the trace does not explicitly identify which earlier search state is being revisited. We show that adding simple parent pointers to explicitly represent the linearized tree (LinTree) structure improves both task performance and search efficiency relative to implicit reasoning models and LLM-heuristic-guided search. These results suggest that search history becomes most useful when its tree structure is made explicit, motivating more structure-aware representations for LLM reasoning.",
      "published": "2026-05-29T16:13:19Z",
      "abstract_url": "http://arxiv.org/abs/2605.31492v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31492v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization",
      "authors": [
        "Zaid Khan",
        "Justin Chih-Yao Chen",
        "Jaemin Cho",
        "Elias Stengel-Eskin",
        "Mohit Bansal"
      ],
      "abstract": "GPU kernels are the workhorse of modern deep learning, and optimizing them (via evolutionary search or coding agents) usually requires repeated measurement on target hardware. While these measurements provide the ground-truth signal necessary for kernel search, they are costly, because each evaluation of a kernel requires compilation and repeated execution on a GPU. As improvements in LLM inference reduce the cost of writing novel kernels and LLM-driven searches scale to large search budgets, on-device evaluation becomes a bottleneck. To address this, we study how LLMs can serve as selective GPU surrogates for kernel evaluation, by forecasting the performance of proposed kernels. A useful surrogate should be accurate, and it should be selective, by knowing when it could be wrong, and deferring to the GPU. To evaluate surrogates, we measure whether their forecasts are accurate, calibrated, and practically useful for recovering fast kernels under limited GPU-measurement budgets. Next, we study whether reinforcement learning can improve forecast accuracy and confidence calibration. Our experiments demonstrate that LLMs can accurately forecast relative kernel performance, that their utility can be improved through reinforcement learning. Used inside a kernel search, the surrogate lets the search consider several times as many candidates under the same GPU evaluation budget, and that leads to finding faster kernels than an equal-budget baseline. These results suggest that LLMs can play a broader role in kernel optimization, by acting as virtual models of a GPU rather than solely as kernel generators for search.",
      "published": "2026-05-29T15:56:08Z",
      "abstract_url": "http://arxiv.org/abs/2605.31464v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31464v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PithTrain: A Compact and Agent-Native MoE Training System",
      "authors": [
        "Ruihang Lai",
        "Hao Kang",
        "Haozhan Tang",
        "Akaash R. Parthasarathy",
        "Zichun Yu",
        "Junru Shao",
        "Todd C. Mowry",
        "Chenyan Xiong",
        "Tianqi Chen"
      ],
      "abstract": "Mixture-of-Experts (MoE) has become the dominant architecture for frontier language models. To meet this demand, production frameworks have built optimized MoE training stacks over years of engineering effort. Yet evolving these stacks for new architectures and system optimizations remains expensive. With the rise of AI coding agents, they could automate parts of training-framework development and accelerate this evolution. But applying them to these existing frameworks carries hidden costs, invisible to today's throughput-only evaluations. We name this missing dimension agent-task efficiency (ATE): the cost of using coding agents to understand, operate, and extend a framework. Grounded in four agent-native design principles, we build PithTrain, a compact, agent-native MoE training framework. We further introduce ATE-Bench, covering real-world training-framework tasks. Our evaluation shows PithTrain matches the throughput of production frameworks, and on ATE-Bench, PithTrain enables higher agent-task efficiency, with up to 62% fewer Agent Turns and 64% less Active GPU Time.",
      "published": "2026-05-29T15:52:58Z",
      "abstract_url": "http://arxiv.org/abs/2605.31463v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31463v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.DC"
      ]
    },
    {
      "title": "Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information",
      "authors": [
        "Antonio Valerio Miceli-Barone",
        "Vaishak Belle",
        "Shay B. Cohen"
      ],
      "abstract": "In this work we study agents in simulated bargaining scenarios, where a buyer and a seller communicate through a text channel and attempt to negotiate mutually beneficial trades, under different information regimes (complete information, information asymmetry or mutual uncertainty). We evaluate their performance w.r.t. game-theoretical solutions and further investigate their honesty (their tendency to disclose or withhold information or to mislead and deceive) as well as their credulity (their tendency to trust or distrust information provided by the other agent). We study zero-shot LLM agents with simple prompting scaffolding as well as fine-tuned agents, in order to investigate whether optimising the agents to maximise financial profits makes them stronger negotiators but also more dishonest and less trusting. We find that off-the-shelf LLMs all substantially deviate from game-theoretical equilibria, they attempt to lie about their private information but cannot efficiently exploit information asymmetries. Fine-tuning on financial utility makes the agents stronger at achieving better deals but also more dishonest, highlighting the risks that optimising agents for a task can have on their safety. We release our code and a dataset of bargaining scenarios.",
      "published": "2026-05-29T15:40:29Z",
      "abstract_url": "http://arxiv.org/abs/2605.31445v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31445v1",
      "categories": [
        "cs.GT",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "DOA: Training-Free Decoder-Only Attention Policy for Long-Form Simultaneous Translation with SpeechLLMs",
      "authors": [
        "Sara Papi",
        "Luisa Bentivogli"
      ],
      "abstract": "Simultaneous speech-to-text translation (SimulST) generates translations while speech is still unfolding, requiring a streaming policy that decides when to read and when to write. State-of-the-art approaches rely on attention-based encoder-decoder models where cross-attention provides explicit alignment signals. In contrast, Speech Large Language Models (SpeechLLMs) are decoder-only architectures relying solely on self-attention. This raises a central question: whether decoder self-attention contains sufficiently stable alignment signals to guide the streaming policy. Moreover, existing approaches typically rely on training-based adaptations or heuristic wait-$k$ policies and have not been validated in long-form settings. To fill these gaps, we propose Decoder-Only Attention (DOA), a training-free policy that enables long-form simultaneous translation with off-the-shelf SpeechLLMs by deriving a proxy alignment from self-attention. Experiments on Phi4-Multimodal and Qwen3-Omni show that DOA provides an effective alignment signal for supporting streaming decisions, enabling low-latency long-form SimulST with quality close to offline decoding without retraining.",
      "published": "2026-05-29T15:27:26Z",
      "abstract_url": "http://arxiv.org/abs/2605.31432v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31432v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.SD"
      ]
    },
    {
      "title": "Skill Availability and Presentation Granularity in Large-Language-Model Agents: A Controlled SkillsBench Study",
      "authors": [
        "Xiaonan Xu",
        "Wenjing Wu"
      ],
      "abstract": "Skill documents provide procedural knowledge to large-language-model agents at inference time. This article studies whether the presentation granularity of controlled skill knowledge changes downstream task success. The experiment uses a pinned SkillsBench version, a 30-task domain-balanced subset validated by official oracle runs, two reasoning-enabled model configurations, six skill conditions, and five trials per task-condition-model cell. Skill availability is the clearest empirical signal. Relative to no skill, skill conditions increase task-mean pass rate by 26.7 to 36.0 percentage points for GPT-5.5 and by 18.0 to 26.0 percentage points for DeepSeek V4-Flash. The final data contain 1,800 rows, with 900 rows for each model. The task is the inference unit. Five trials are aggregated within each task-condition-model cell before paired contrasts are estimated over 30 tasks. The primary presentation contrasts are smaller and uncertain. Low-abstraction guidance differs from high-abstraction guidance by +0.7 percentage points for GPT-5.5 and -6.7 percentage points for DeepSeek V4-Flash, with both 95% bootstrap confidence intervals crossing zero. Adding one worked example to medium-abstraction guidance differs from the no-example variant by +0.7 and +1.3 percentage points. Mean-reward robustness checks preserve the same substantive conclusion. In this controlled subset, skill availability is associated with higher success than no skill, while the tested presentation-granularity changes yield small, uncertain, and model-dependent effects.",
      "published": "2026-05-29T15:12:24Z",
      "abstract_url": "http://arxiv.org/abs/2605.31408v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31408v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning",
      "authors": [
        "Xudong Zhang",
        "Jian Yang",
        "Shengkai Wang",
        "Jiangpeng Tian",
        "Shaowen Chen",
        "Xian Wei",
        "Ke Li",
        "Xiong You"
      ],
      "abstract": "Large Language Model (LLM)-based navigation systems commonly construct explicit spatial representations (e.g., topological graphs, semantic raster maps) and translate them into textual descriptions as LLMs' inputs. However, the linguistic structures of such text-based spatial representations and the choices of contextual features (e.g., topology, geometry) they contain are often treated as neutral engineering decisions rather than key factors that shape LLMs' behavior. To fill the gap, we propose a dual-interventional framework that disentangles linguistic structures from different contextual cues to evaluate the linguistic inductive bias of LLMs for navigation planning. In the framework, representation intervention varies the linguistic format and the degree of linguistic compression, clarifying when linguistic representations support or inhibit navigation planning. Context intervention, combined with contextual feature combination and conflict probing, explicitly clarifies the preferences and weaknesses of LLMs when processing different contextual cues. Experiments across diverse spatial reasoning tasks and multiple model scales reveal a consistent pattern: topological information is a sturdy shield and the backbone of robust planning; linguistic format is a double-edged sword whose effect depends on model size, task demands, and the compression level; and semantic information is a fatal Achilles' heel -- incorrect semantic cues can systematically derail the planning process. Overall, our study shows that effective text-based spatial representations in LLM-based navigation should preserve topological integrity, calibrate representational compression to model capacity, and ensure semantic correctness, rather than simply adopting a single representation. Our code is publicly available at https://github.com/jonesdong150/LLM-Navigation-Inductive-Bias.",
      "published": "2026-05-29T15:09:25Z",
      "abstract_url": "http://arxiv.org/abs/2605.31404v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31404v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Target-Side Paraphrase Augmentation for Sign Language Translation with Large Language Models",
      "authors": [
        "Pedro Dal Bianco",
        "Jean Paul Nunes Reinhold",
        "Oscar Stanchi",
        "Facundo Quiroga",
        "Franco Ronchetti",
        "Ulisses Brisolara Corrêa"
      ],
      "abstract": "Sign language translation (SLT) remains constrained by limited paired sign-video/text corpora and heavy-tailed target vocabularies. We study target-side augmentation in which GPT-4o generates controlled paraphrase variants of reference sentences while the sign input remains unchanged. A Signformer-style pose-based Transformer is trained under a two-stage schedule: pre-training on the augmented corpus followed by fine-tuning on the original references. We evaluate on three datasets spanning complementary challenges: PHOENIX14T (German Sign Language), with moderate lexical diversity; GSL (Greek Sign Language), with highly ontrolled, repetitive recordings; and LSA-T (Argentinian Sign Language), with severe long-tail sparsity. On PHOENIX14T, augmentation improves BLEU-4 from 9.56 to 10.33. The near-saturated GSL baseline and extremely sparse LSA-T setting reveal the limits of the approach. To our knowledge, this is the first study to apply LLM-generated target-side araphrases and LLM-as-a-Judge evaluation to SLT. The semantic evaluation reveals gains in fidelity that lexical overlap metrics understate.",
      "published": "2026-05-29T14:58:21Z",
      "abstract_url": "http://arxiv.org/abs/2605.31393v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31393v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Scaling Higher-Order Graph Learning with Maximal Clique Complexes",
      "authors": [
        "Antoine Vialle",
        "Aref Einizade",
        "Fragkiskos D. Malliaros",
        "Jhony H. Giraldo"
      ],
      "abstract": "Graph neural networks (GNNs) are limited to modeling pairwise interactions, while higher-order models based on cell complexes achieve greater expressivity but often suffer from poor scalability. We introduce simplified and factored cellular Weisfeiler Leman tests (sCWL and fCWL), which preserve the expressivity of the CWL test while improving computational efficiency. We further introduce the maximal clique complex, enabling scalable CWNs with reduced time and memory complexity while retaining strong empirical performance. To avoid explicit clique enumeration, we propose CliqueWalk, a biased random walk that samples maximal cliques and scales linearly with graph size. These contributions yield a scalable topological learning framework for higher-order graph representation.",
      "published": "2026-05-29T14:42:40Z",
      "abstract_url": "http://arxiv.org/abs/2605.31373v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31373v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning to Adapt: Self-Improving Web Agent via Cognitive-Aware Exploration",
      "authors": [
        "Weile Chen",
        "Bingchen Miao",
        "Qifan Yu",
        "Wendong Bu",
        "Guoming Wang",
        "Wenqiao Zhang",
        "Shengyu Zhang",
        "Juncheng Li",
        "Siliang Tang"
      ],
      "abstract": "Recent advances in Multimodal Large Language Models (MLLMs) have led to promising progress in web agents. However, existing web agents often rely on handcrafted execution pipelines or expensive expert trajectories, limiting their adaptability to complex, dynamic environments. To address these challenges, we propose SCALE (Self-Cognitive-Aware Learning and Exploration), which leverages three adversarial roles, Selector, Predictor, and Judger to autonomously discover the agent's limitations and expand its cognitive boundaries through environmental exploration. Moreover, we propose SCALE-Hop, a graph exploration strategy that facilitates global planning and helps agents avoid local exploration traps. To further support learning, we construct SCALE-20k, a large-scale dataset collected from 19 real-world websites, containing diverse task types and structured demonstrations generated from SCALE's exploration traces. Experimental results show that our approach significantly improves the performance and generalization of multiple MLLMs in various web environments. Our framework offers a scalable and generalizable solution for building truly autonomous and adaptive web agents.",
      "published": "2026-05-29T14:37:27Z",
      "abstract_url": "http://arxiv.org/abs/2605.31365v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31365v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Dreaming Of Others: Latent Teammate Modeling In World Models For Multi-Agent Reinforcement Learning",
      "authors": [
        "Tomas Leroy-Stone"
      ],
      "abstract": "In cooperative multi-agent reinforcement learning (MARL), agents must coordinate with partners whose internal policies and intentions are not directly observable. While world models such as Dreamer have demonstrated strong generalization and sample efficiency in single-agent settings, their application to MARL remains limited by an inability to handle teammate-induced uncertainty. We propose a new perspective: treat teammates as structured, learnable components within the agent's world model. We introduce an architecture that factorizes the latent state of a Dreamer-style recurrent state-space model (RSSM) into environment and teammate components, and learns an auxiliary Theory-of-Mind (ToM) head to infer latent embeddings of partner behavior such as character, intent, and predicted actions from partial trajectories. These teammate latents condition the actor and critic, enabling the agent to imagine and adapt to diverse collaborators. We outline how this approach can support zero-shot and few-shot coordination in partially observable settings and propose a set of benchmarks and evaluation protocols to assess its impact. This work positions world models as not only predictors of environmental dynamics, but as simulators of social behavior, opening new directions for generalizable, human-compatible AI.",
      "published": "2026-05-29T14:34:50Z",
      "abstract_url": "http://arxiv.org/abs/2605.31361v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31361v1",
      "categories": [
        "cs.MA",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "dashi: A Python library for Dataset Shift Characterization to Support Trustworthy AI Development and Deployment",
      "authors": [
        "David Fernández-Narro",
        "Pablo Ferri",
        "Ángel Sánchez-García",
        "Juan M. García-Gómez",
        "Carlos Sáez"
      ],
      "abstract": "The Artificial Intelligence (AI) life cycle requires a thorough understanding of the underlying data dynamics for robust, safe and cost-effective AI development and use. Dataset shifts are defined as changes between train and test data distributions. Whether occurring over time (temporal) or across different sites (multi-source), they can severely degrade model performance and compromise data quality. This is particularly important in health AI, where the safety and fundamental rights of patients can be severely affected by uncontrolled shifts both at training and operational stages. While the theoretical foundations of covariate, prior, and concept shifts are well established, there is a lack of accessible and comprehensive software tools to perform their analysis. We introduce dashi, an open-source Python library designed for the exploration, quantification, and characterization of dataset shifts. dashi provides a dual approach: an unsupervised approach that leverages information geometry and non-parametric statistical manifolds to data variability characterization and analysis (e.g., Information Geometric Temporal plots and Multi-Source Variability metrics like Global Probabilistic Deviation and Source Probabilistic Outlyingness), and a supervised approach that quantifies and characterizes model performance degradation. Both unsupervised and supervised approaches work across user-defined temporal and domain/source batches. We demonstrate the utility of dashi on three simulated and real-world health AI case studies on gestational diabetes mellitus, COVID-19 and emergency medical dispatch. By providing interactive visual analytics and variability metrics, dashi supports trustworthiness of AI life cycle stages enabling robust and safe machine learning pipelines through the assessment of data coherence and AI performance.",
      "published": "2026-05-29T14:33:45Z",
      "abstract_url": "http://arxiv.org/abs/2605.31360v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31360v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Diagnosing Failure Modes of Shared-State Collaboration in Resource-Constrained Visual Agents",
      "authors": [
        "Yunpeng Zhou"
      ],
      "abstract": "Modular visual reasoning systems increasingly rely on shared working memory for multi-step collaboration, yet the failure dynamics of intermediate state evolution in low-capacity regimes remain underexplored. We study failure modes of collaborative reasoning with weak learners (4B--8B models) through the lens of noise accumulation. We introduce CoSee, an auditing framework that formalizes the read-write-verify loop to trace information flow in document visual question answering. Across multi-page, chart, and web-based benchmarks, we find a counter-intuitive degradation: naive shared workspaces often amplify hallucinations rather than resolve them. We identify two dominant failure modes: Noise Reinforcement, where ungrounded notes are reused as evidence, and Policy Collapse, where added context shifts the model toward under-specified, short-form answers. Using cost-accuracy Pareto frontiers, we show that increased compute can correlate negatively with performance without explicit verification. Our findings suggest that for resource-constrained agents, the bottleneck lies not in reasoning depth but in communication fidelity, providing trace-level diagnostics and a mechanistic baseline for reliable modular design.",
      "published": "2026-05-29T14:29:56Z",
      "abstract_url": "http://arxiv.org/abs/2605.31354v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31354v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Inconsistency-Aware Minimization: Improving Generalization with Unlabeled Data",
      "authors": [
        "Hee-Sung Kim",
        "Hyeonseong Kim",
        "Sungyoon Lee"
      ],
      "abstract": "Estimating the generalization gap and developing optimization methods that improve generalization are crucial for deep learning models, for both theoretical understanding and practical applications. Leveraging unlabeled data for these purposes offers significant advantages in real-world scenarios. This paper introduces a novel generalization measure, local inconsistency, derived from an information-geometric perspective on the parameter space of neural networks. A key feature of local inconsistency is that it can be computed without explicit labels. We establish theoretical underpinnings by connecting local inconsistency to the Fisher information matrix and the loss Hessian. Empirically, we demonstrate that local inconsistency correlates with the generalization gap. Based on these findings, we propose Inconsistency-Aware Minimization (IAM), which incorporates local inconsistency into the training objective. We demonstrate that in standard supervised learning settings, IAM enhances generalization, achieving performance comparable to that of existing methods such as Sharpness-Aware Minimization. Furthermore, IAM exhibits efficacy in semi- and self-supervised learning scenarios, where the local inconsistency is computed from unlabeled data.",
      "published": "2026-05-29T13:56:17Z",
      "abstract_url": "http://arxiv.org/abs/2605.31324v1",
      "pdf_url": "https://arxiv.org/pdf/2605.31324v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
