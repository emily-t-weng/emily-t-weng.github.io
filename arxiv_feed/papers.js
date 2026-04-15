const PAPERS_DATA = {
  "last_updated": "2026-04-15 03:28:27 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe",
      "authors": [
        "Yaxuan Li",
        "Yuxin Zuo",
        "Bingxiang He",
        "Jinqian Zhang",
        "Chaojun Xiao",
        "Cheng Qian",
        "Tianyu Yu",
        "Huan-ang Gao",
        "Wenkai Yang",
        "Zhiyuan Liu",
        "Ning Ding"
      ],
      "abstract": "On-policy distillation (OPD) has become a core technique in the post-training of large language models, yet its training dynamics remain poorly understood. This paper provides a systematic investigation of OPD dynamics and mechanisms. We first identify that two conditions govern whether OPD succeeds or fails: (i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training. We validate these findings through weak-to-strong reverse distillation, showing that same-family 1.5B and 7B teachers are distributionally indistinguishable from the student's perspective. Probing into the token-level mechanism, we show that successful OPD is characterized by progressive alignment on high-probability tokens at student-visited states, a small shared token set that concentrates most of the probability mass (97%-99%). We further propose two practical strategies to recover failing OPD: off-policy cold start and teacher-aligned prompt selection. Finally, we show that OPD's apparent free lunch of dense token-level reward comes at a cost, raising the question of whether OPD can scale to long-horizon distillation.",
      "published": "2026-04-14T17:54:28Z",
      "abstract_url": "http://arxiv.org/abs/2604.13016v1",
      "pdf_url": "https://arxiv.org/pdf/2604.13016v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation",
      "authors": [
        "Yecheng Wu",
        "Song Han",
        "Hai Cai"
      ],
      "abstract": "On-policy distillation (OPD) has emerged as an efficient post-training paradigm for large language models. However, standard OPD requires a live teacher inference server throughout training, resulting in substantial infrastructure overhead. In this work, we investigate whether on-policy distillation can be performed offline. A natural approach is to precompute teacher log-probabilities once over SFT rollouts and reuse them during training. In practice, however, this offline variant fails to reliably match the performance of standard OPD. To understand this discrepancy, we identify a previously overlooked condition that is critical for any OPD pipeline, which we term teacher consistency. This condition requires that the same teacher model be used for both supervised fine-tuning and OPD. We show that violating teacher consistency introduces an irreducible gradient bias, causing both offline and online OPD to converge to a suboptimal fixed point regardless of training duration. Building on this insight, we propose Lightning OPD, an offline on-policy distillation framework that enforces teacher consistency by precomputing teacher log-probabilities over SFT rollouts. This design eliminates the need for a live teacher server entirely. We further show that, under teacher consistency, Lightning OPD shares the same optimum as standard OPD, with bounded gradient discrepancy and an implicit regularization effect that helps prevent policy drift. Extensive experiments on mathematical reasoning and code generation demonstrate that Lightning OPD achieves state-of-the-art performance with significantly improved efficiency. Starting from an SFT-initialized Qwen3-8B-Base model, Lightning OPD reaches 69.9% on AIME 2024 in just 30 GPU hours, achieving a 4.0x speedup over standard OPD and substantially lowering the barrier to entry for academic research on LLM post-training.",
      "published": "2026-04-14T17:44:50Z",
      "abstract_url": "http://arxiv.org/abs/2604.13010v1",
      "pdf_url": "https://arxiv.org/pdf/2604.13010v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "One Token Away from Collapse: The Fragility of Instruction-Tuned Helpfulness",
      "authors": [
        "Erfan Baghaei Potraghloo",
        "Seyedarmin Azizi",
        "Souvik Kundu",
        "Massoud Pedram"
      ],
      "abstract": "Instruction-tuned large language models produce helpful, structured responses, but how robust is this helpfulness when trivially constrained? We show that simple lexical constraints (banning a single punctuation character or common word) cause instruction-tuned LLMs to collapse their responses, losing 14--48% of comprehensiveness in pairwise evaluation across three open-weight model families and one closed-weight model (GPT-4o-mini). The baseline response is preferred in 77--100% of 1,920 pairwise comparisons judged by GPT-4o-mini and GPT-4o. Notably, GPT-4o-mini suffers 31% comprehensiveness loss (99% baseline win rate), demonstrating that the fragility extends to commercially deployed closed-weight models, contrary to prior findings on format-level constraints. Through mechanistic analysis, we identify this as a planning failure: two-pass generation (free generation followed by constrained rewriting) recovers 59--96% of response length, and linear probes on prompt representations predict response length with $R^2 = 0.51$--$0.93$ before generation begins, with $R^2$ tracking collapse severity across models. The same probes yield negative $R^2$ on base models, confirming that instruction tuning creates the representational structure encoding the collapse decision. Crucially, base models show no systematic collapse under identical constraints, with effects that are small, noisy, and bidirectional, demonstrating that instruction tuning creates this fragility by coupling task competence to narrow surface-form templates. The effect replicates on MT-Bench across all eight task categories. We further show that standard independent LLM-as-judge evaluation detects only a 3.5% average quality drop where pairwise evaluation reveals 23%, exposing a methodological blind spot in how constrained generation is assessed.",
      "published": "2026-04-14T17:40:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.13006v1",
      "pdf_url": "https://arxiv.org/pdf/2604.13006v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software",
      "authors": [
        "Syed Md Mukit Rashid",
        "Abdullah Al Ishtiaq",
        "Kai Tu",
        "Yilu Dong",
        "Tianwei Wu",
        "Ali Ranjbar",
        "Tianchang Yang",
        "Najrin Sultana",
        "Shagufta Mehnaz",
        "Syed Rafiul Hussain"
      ],
      "abstract": "Logical vulnerabilities in software stem from flaws in program logic rather than memory safety, which can lead to critical security failures. Although existing automated program repair techniques primarily focus on repairing memory corruption vulnerabilities, they struggle with logical vulnerabilities because of their limited semantic understanding of the vulnerable code and its expected behavior. On the other hand, recent successes of large language models (LLMs) in understanding and repairing code are promising. However, no framework currently exists to analyze the capabilities and limitations of such techniques for logical vulnerabilities. This paper aims to systematically evaluate both traditional and LLM-based repair approaches for addressing real-world logical vulnerabilities. To facilitate our assessment, we created the first ever dataset, LogicDS, of 86 logical vulnerabilities with assigned CVEs reflecting tangible security impact. We also developed a systematic framework, LogicEval, to evaluate patches for logical vulnerabilities. Evaluations suggest that compilation and testing failures are primarily driven by prompt sensitivity, loss of code context, and difficulty in patch localization.",
      "published": "2026-04-14T17:26:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.12994v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12994v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Modeling Co-Pilots for Text-to-Model Translation",
      "authors": [
        "Serdar Kadioglu",
        "Karthik Uppuluri",
        "Akash Singirikonda"
      ],
      "abstract": "There is growing interest in leveraging large language models (LLMs) for text-to-model translation and optimization tasks. This paper aims to advance this line of research by introducing \\textsc{Text2Model} and \\textsc{Text2Zinc}. \\textsc{Text2Model} is a suite of co-pilots based on several LLM strategies with varying complexity, along with an online leaderboard. \\textsc{Text2Zinc} is a cross-domain dataset for capturing optimization and satisfaction problems specified in natural language, along with an interactive editor with built-in AI assistant. While there is an emerging literature on using LLMs for translating combinatorial problems into formal models, our work is the first attempt to integrate \\textit{both} satisfaction and optimization problems within a \\textit{unified architecture} and \\textit{dataset}. Moreover, our approach is \\textit{solver-agnostic} unlike existing work that focuses on translation to a solver-specific model. To achieve this, we leverage \\textsc{MiniZinc}'s solver-and-paradigm-agnostic modeling capabilities to formulate combinatorial problems. We conduct comprehensive experiments to compare execution and solution accuracy across several single- and multi-call strategies, including; zero-shot prompting, chain-of-thought reasoning, intermediate representations via knowledge-graphs, grammar-based syntax encoding, and agentic approaches that decompose the model into sequential sub-tasks. Our co-pilot strategies are competitive, and in parts improve, recent research in this domain. Our findings indicate that while LLMs are promising they are not yet a push-button technology for combinatorial modeling. We contribute \\textsc{Text2Model} co-pilots and leaderboard, and \\textsc{Text2Zinc} and interactive editor to open-source to support closing this performance gap.",
      "published": "2026-04-14T16:51:29Z",
      "abstract_url": "http://arxiv.org/abs/2604.12955v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12955v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Distorted or Fabricated? A Survey on Hallucination in Video LLMs",
      "authors": [
        "Yiyang Huang",
        "Yitian Zhang",
        "Yizhou Wang",
        "Mingyuan Zhang",
        "Liang Shi",
        "Huimin Zeng",
        "Yun Fu"
      ],
      "abstract": "Despite significant progress in video-language modeling, hallucinations remain a persistent challenge in Video Large Language Models (Vid-LLMs), referring to outputs that appear plausible yet contradict the content of the input video. This survey presents a comprehensive analysis of hallucinations in Vid-LLMs and introduces a systematic taxonomy that categorizes them into two core types: dynamic distortion and content fabrication, each comprising two subtypes with representative cases. Building on this taxonomy, we review recent advances in the evaluation and mitigation of hallucinations, covering key benchmarks, metrics, and intervention strategies. We further analyze the root causes of dynamic distortion and content fabrication, which often result from limited capacity for temporal representation and insufficient visual grounding. These insights inform several promising directions for future work, including the development of motion-aware visual encoders and the integration of counterfactual learning techniques. This survey consolidates scattered progress to foster a systematic understanding of hallucinations in Vid-LLMs, laying the groundwork for building robust and reliable video-language systems. An up-to-date curated list of related works is maintained at https://github.com/hukcc/Awesome-Video-Hallucination .",
      "published": "2026-04-14T16:37:57Z",
      "abstract_url": "http://arxiv.org/abs/2604.12944v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12944v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "CoDe-R: Refining Decompiler Output with LLMs via Rationale Guidance and Adaptive Inference",
      "authors": [
        "Qiang Zhang",
        "Zhongnian Li"
      ],
      "abstract": "Binary decompilation is a critical reverse engineering task aimed at reconstructing high-level source code from stripped executables. Although Large Language Models (LLMs) have recently shown promise, they often suffer from \"logical hallucinations\" and \"semantic misalignment\" due to the irreversible semantic loss during compilation, resulting in generated code that fails to re-execute. In this study, we propose Cognitive Decompiler Refinement with Robustness (CoDe-R), a lightweight two-stage code refinement framework. The first stage introduces Semantic Cognitive Enhancement (SCE), a Rationale-Guided Semantic Injection strategy that trains the model to recover high-level algorithmic intent alongside code. The second stage introduces a Dynamic Dual-Path Fallback (DDPF) mechanism during inference, which adaptively balances semantic recovery and syntactic stability via a hybrid verification strategy. Evaluation on the HumanEval-Decompile benchmark demonstrates that CoDe-R (using a 1.3B backbone) establishes a new State-of-the-Art (SOTA) in the lightweight regime. Notably, it is the first 1.3B model to exceed an Average Re-executability Rate of 50.00%, significantly outperforming the baseline and effectively bridging the gap between efficient models and expert-level performance. Our code is available at https://github.com/Theaoi/CoDe-R.",
      "published": "2026-04-14T15:58:38Z",
      "abstract_url": "http://arxiv.org/abs/2604.12913v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12913v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design",
      "authors": [
        "Chuyang Xiang",
        "Yichen Wei",
        "Jiale Ma",
        "Handing Wang",
        "Junchi Yan"
      ],
      "abstract": "Large Language Model-based Hyper Heuristic (LHH) has recently emerged as an efficient way for automatic heuristic design. However, most existing LHHs just perform well in optimizing a single function within a pre-defined solver. Their single-layer evolution makes them not effective enough to write a competent complete solver. While some variants incorporate hyperparameter tuning or attempt to generate complex code through iterative local modifications, they still lack a high-level algorithmic modeling, leading to limited exploration efficiency. To address this, we reformulate heuristic design as a Bi-level Optimization problem and propose \\textbf{BEAM} (Bi-level Memory-adaptive Algorithmic Evolution). BEAM's exterior layer evolves high-level algorithmic structures with function placeholders through genetic algorithm (GA), while the interior layer realizes these placeholders via Monte Carlo Tree Search (MCTS). We further introduce an Adaptive Memory module to facilitate complex code generation. To support the evaluation for complex code generation, we point out the limitations of starting LHHs from scratch or from code templates and introduce a Knowledge Augmentation (KA) Pipeline. Experimental results on several optimization problems demonstrate that BEAM significantly outperforms existing LHHs, notably reducing the optimality gap by 37.84\\% on aggregate in CVRP hybrid algorithm design. BEAM also designs a heuristic that outperforms SOTA Maximum Independent Set (MIS) solver KaMIS.",
      "published": "2026-04-14T15:46:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.12898v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12898v1",
      "categories": [
        "cs.AI",
        "math.CO"
      ]
    },
    {
      "title": "AISafetyBenchExplorer: A Metric-Aware Catalogue of AI Safety Benchmarks Reveals Fragmented Measurement and Weak Benchmark Governance",
      "authors": [
        "Abiodun A. Solanke"
      ],
      "abstract": "The rapid expansion of large language model (LLM) safety evaluation has produced a substantial benchmark ecosystem, but not a correspondingly coherent measurement ecosystem. We present AISafetyBenchExplorer, a structured catalogue of 195 AI safety benchmarks released between 2018 and 2026, organized through a multi-sheet schema that records benchmark-level metadata, metric-level definitions, benchmark-paper metadata, and repository activity. This design enables meta-analysis not only of what benchmarks exist, but also of how safety is operationalized, aggregated, and judged across the literature. Using the updated catalogue, we identify a central structural problem: benchmark proliferation has outpaced measurement standardization. The current landscape is dominated by medium-complexity benchmarks (94/195), while only 7 benchmarks occupy the Popular tier. The workbook further reports strong concentration around English-only evaluation (165/195), evaluation-only resources (170/195), stale GitHub repositories (137/195), stale Hugging Face datasets (96/195), and heavy reliance on arXiv preprints among benchmarks with known venue metadata. At the metric level, the catalogue shows that familiar labels such as accuracy, F1 score, safety score, and aggregate benchmark scores often conceal materially different judges, aggregation rules, and threat models. We argue that the field's main failure mode is fragmentation rather than scarcity. Researchers now have many benchmark artifacts, but they often lack a shared measurement language, a principled basis for benchmark selection, and durable stewardship norms for post publication maintenance. AISafetyBenchExplorer addresses this gap by providing a traceable benchmark catalogue, a controlled metadata schema, and a complexity taxonomy that together support more rigorous benchmark discovery, comparison, and meta-evaluation.",
      "published": "2026-04-14T15:26:03Z",
      "abstract_url": "http://arxiv.org/abs/2604.12875v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12875v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Loop Corrections to the Training and Generalization Errors of Random Feature Models",
      "authors": [
        "Taeyoung Kim"
      ],
      "abstract": "We investigate random feature models in which neural networks sampled from a prescribed initialization ensemble are frozen and used as random features, with only the readout weights optimized. Adopting a statistical-physics viewpoint, we study the training, test, and generalization errors beyond the mean-kernel approximation. Since the predictor is a nonlinear functional of the induced random kernel, the ensemble-averaged errors depend not only on the mean kernel but also on higher-order fluctuation statistics. Within an effective field-theoretic framework, these finite-width contributions naturally appear as loop corrections. We derive the loop corrections to the training, test, and generalization errors, obtain their scaling laws, and support the theory with experimental verification.",
      "published": "2026-04-14T14:48:41Z",
      "abstract_url": "http://arxiv.org/abs/2604.12827v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12827v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair",
      "authors": [
        "Jagadeesh Rachapudi",
        "Pranav Singh",
        "Ritali Vatsi",
        "Praful Hambarde",
        "Amit Shukla"
      ],
      "abstract": "Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requiring retraining pipelines, curated retain datasets, and direct intervention by model service providers (MSPs), thereby excluding end users from controlling their own data. We introduce Interactive Machine Unlearning (IMU), a new paradigm in which users can instruct LLMs to forget targeted knowledge through natural language at inference time. To realize IMU, we propose RePAIR, a prompt-aware model repair framework comprising (i) a watchdog model for unlearning intent detection, (ii) a surgeon model for generating repair procedures, and (iii) a patient model whose parameters are updated autonomously. At the core of RePAIR, we develop Steering Through Activation Manipulation with PseudoInverse (STAMP), a training-free, single-sample unlearning method that redirects MLP activations toward a refusal subspace via closed-form pseudoinverse updates. Its low-rank variant reduces computational complexity from O(d^3) to O(r^3 + r^2 * d), enabling efficient on-device unlearning with up to ~3x speedup over training-based baselines. Extensive experiments across harmful knowledge suppression, misinformation correction, and personal data erasure demonstrate that RePAIR achieves near-zero forget scores (Acc_f = 0.00, F-RL = 0.00) while preserving model utility (Acc_r up to 84.47, R-RL up to 0.88), outperforming six state-of-the-art baselines. These results establish RePAIR as an effective and practical framework for user-driven model editing, advancing transparent and on-device control over learned knowledge, with potential extensions to multimodal foundation models.",
      "published": "2026-04-14T14:44:45Z",
      "abstract_url": "http://arxiv.org/abs/2604.12820v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12820v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding",
      "authors": [
        "Hao Yan",
        "Yuliang Liu",
        "Xingchen Liu",
        "Yuyi Zhang",
        "Minghui Liao",
        "Jihao Wu",
        "Wei Chen",
        "Xiang Bai"
      ],
      "abstract": "Existing Multimodal Large Language Models (MLLMs) suffer from significant performance degradation on the long document understanding task as document length increases. This stems from two fundamental challenges: 1) a low Signal-to-Noise Ratio (SNR), with crucial evidence buried in irrelevant pages; and 2) supervision scarcity, as datasets offering only final short answers provide a weak learning signal. In this paper, we address these challenges by proposing a paradigm that requires the model to execute a structured ``\\textbf{Analysis}, \\textbf{Localization} and \\textbf{Reasoning}'' workflow. To instill this capability, we design a two-stage training framework: we first perform Supervised Fine-Tuning on high-quality data generated via an efficient knowledge distillation strategy. Subsequently, we employ an Evidence-aware Group Relative Policy Optimization which jointly optimizes for both evidence localization and answer accuracy. Additionally, we introduce a Evidence-Guided Resolution Allocation strategy to mitigate memory constraints of training on multi-pages documents. Extensive experiments demonstrate that DocSeeker achieves superior performance on both in-domain and out-of-domain tasks. We show it robustly generalizes from short-page training to ultra-long documents and is naturally synergistic with visual Retrieval-Augmented Generation systems, serving as a solid foundation for their implementation.",
      "published": "2026-04-14T14:39:26Z",
      "abstract_url": "http://arxiv.org/abs/2604.12812v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12812v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Algorithmic Analysis of Dense Associative Memory: Finite-Size Guarantees and Adversarial Robustness",
      "authors": [
        "Madhava Gaikwad"
      ],
      "abstract": "Dense Associative Memory (DAM) generalizes Hopfield networks through higher-order interactions and achieves storage capacity that scales as $O(N^{n-1})$ under suitable pattern separation conditions. Existing dynamical analyses primarily study the thermodynamic limit $N\\to\\infty$ with randomly sampled patterns and therefore do not provide finite-size guarantees or explicit convergence rates. We develop an algorithmic analysis of DAM retrieval dynamics that yields finite-$N$ guarantees under explicit, verifiable pattern conditions. Under a separation assumption and a bounded-interference condition at high loading, we prove geometric convergence of asynchronous retrieval dynamics, which implies $O(\\log N)$ convergence time once the trajectory enters the basin of attraction. We further establish adversarial robustness bounds expressed through an explicit margin condition that quantifies the number of corrupted bits tolerable per sweep, and derive capacity guarantees that scale as $Θ(N^{n-1})$ up to polylogarithmic factors in the worst case, while recovering the classical $Θ(N^{n-1})$ scaling for random pattern ensembles. Finally, we show that DAM retrieval dynamics admit a potential-game interpretation that ensures convergence to pure Nash equilibria under asynchronous updates. Complete proofs are provided in the appendices, together with preliminary experiments that illustrate the predicted convergence, robustness, and capacity scaling behavior.",
      "published": "2026-04-14T14:38:46Z",
      "abstract_url": "http://arxiv.org/abs/2604.12811v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12811v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.NE"
      ]
    },
    {
      "title": "VFA: Relieving Vector Operations in Flash Attention with Global Maximum Pre-computation",
      "authors": [
        "Yupeng Sun",
        "Yanzhao Li",
        "Zhiqiang Zou",
        "Bai Du",
        "Zhiyuan Zhang",
        "Hui Dong",
        "Gaoyige Fan",
        "Hui Wang"
      ],
      "abstract": "FlashAttention-style online softmax enables exact attention computation with linear memory by streaming score tiles through on-chip memory and maintaining a running maximum and normalizer. However, as attention kernels approach peak tensor-core/cube-core throughput on modern accelerators, non-matmul components of online softmax -- especially per-tile rowmax and rowsum reductions and rescale chains -- can become vector or SIMD limited and dominate latency. This paper revisits FlashAttention and proposes Vector Relieved Flash Attention (VFA), a hardware-friendly method that reduces rowmax-driven updates of the running maximum while retaining the online-softmax structure. VFA initializes the running maximum via a cheap approximation from key-block representations, reorders key-block traversal to prioritize high-impact sink and local blocks, and freezes the maximum for remaining blocks to avoid repeated reductions and rescaling. We further integrate VFA with block-sparse skipping methods such as BLASST to form Vector Relieved Sparse Attention (VSA), which reduces both block count and per-block overhead. Notably, VFA and VSA completely avoid the conditional rescale operation in the update stage used in FA4.0. Extensive evaluations on benchmarks including MMLU and MATH500, together with attention statistics, verify our design: (i) sink and local reordering stabilizes the running maximum early; (ii) simple Q and K block summaries fail due to intra-block heterogeneity; (iii) m-initialization is required when maxima appear in middle blocks. Overall, VFA and VSA efficiently alleviate online-softmax reduction bottlenecks without performance loss. Compared to the C16V32 baseline, C8V32, C4V32 and C4V16 achieve nearly two times speedup on modern hardware while hitting the vector bottleneck. With upcoming architecture improvements, C4V16 will deliver six times speedup by enhancing exponent capacity.",
      "published": "2026-04-14T14:28:50Z",
      "abstract_url": "http://arxiv.org/abs/2604.12798v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12798v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "OSC: Hardware Efficient W4A4 Quantization via Outlier Separation in Channel Dimension",
      "authors": [
        "Zhiyuan Zhang",
        "Yanzhao Li",
        "Zhiqiang Zou",
        "Bai Du",
        "Yupeng Sun",
        "Hui Dong",
        "Hui Wang"
      ],
      "abstract": "While 4-bit quantization is essential for high-throughput deployment of Large Language Models, activation outliers often lead to significant accuracy degradation due to the restricted dynamic range of low-bit formats. In this paper, we systematically investigate the spatial distribution of outliers and demonstrate a token-persistent structural clustering effect, where high-magnitude outliers consistently occupy fixed channels across tokens. Building on this insight, we propose OSC, a hardware-efficient framework for outlier suppression. During inference, OSC executes a dual-path computation consisting of a low-precision 4-bit General Matrix Multiplication (GEMM) path and a high-precision 16-bit branch GEMM path. Specifically, OSC uses an offline group-wise strategy to identify the channels where outliers are located and then performs structured sub-tensor extraction to coalesce these scattered activation channels into a compact dense tensor online. This mechanism implements outlier protection through regularized and high-throughput GEMM operations, achieving a seamless fit with modern 4-bit micro-scaling hardware. Furthermore, for the inputs of W2 where outlier clustering is less pronounced, we integrate a fallback strategy to FP8. Evaluation on Qwen3-8B and Qwen3-30B restricts the average accuracy drop to 2.19 and 1.12 points, respectively. Notably, OSC is highly hardware-friendly, achieving a peak speedup of 1.78x over the W8A8 GEMM baseline on a modern AI accelerator.",
      "published": "2026-04-14T14:17:59Z",
      "abstract_url": "http://arxiv.org/abs/2604.12782v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12782v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "CLASP: Class-Adaptive Layer Fusion and Dual-Stage Pruning for Multimodal Large Language Models",
      "authors": [
        "Yunkai Dang",
        "Yizhu Jiang",
        "Yifan Jiang",
        "Qi Fan",
        "Yinghuan Shi",
        "Wenbin Li",
        "Yang Gao"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) suffer from substantial computational overhead due to the high redundancy in visual token sequences. Existing approaches typically address this issue using single-layer Vision Transformer (ViT) features and static pruning strategies. However, such fixed configurations are often brittle under diverse instructions. To overcome these limitations, we propose CLASP, a plug-and-play token reduction framework based on class-adaptive layer fusion and dual-stage pruning. Specifically, CLASP first constructs category-specific visual representations through multi-layer vision feature fusion. It then performs dual-stage pruning, allocating the token budget between attention-salient pivot tokens for relevance and redundancy-aware completion tokens for coverage. Through class-adaptive pruning, CLASP enables prompt-conditioned feature fusion and budget allocation, allowing aggressive yet robust visual token reduction. Extensive experiments demonstrate that CLASP consistently outperforms existing methods across a wide range of benchmarks, pruning ratios, and MLLM architectures. Code will be available at https://github.com/Yunkaidang/CLASP.",
      "published": "2026-04-14T14:08:20Z",
      "abstract_url": "http://arxiv.org/abs/2604.12767v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12767v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "GF-Score: Certified Class-Conditional Robustness Evaluation with Fairness Guarantees",
      "authors": [
        "Arya Shah",
        "Kaveri Visavadiya",
        "Manisha Padala"
      ],
      "abstract": "Adversarial robustness is essential for deploying neural networks in safety-critical applications, yet standard evaluation methods either require expensive adversarial attacks or report only a single aggregate score that obscures how robustness is distributed across classes. We introduce the \\emph{GF-Score} (GREAT-Fairness Score), a framework that decomposes the certified GREAT Score into per-class robustness profiles and quantifies their disparity through four metrics grounded in welfare economics: the Robustness Disparity Index (RDI), the Normalized Robustness Gini Coefficient (NRGC), Worst-Case Class Robustness (WCR), and a Fairness-Penalized GREAT Score (FP-GREAT). The framework further eliminates the original method's dependence on adversarial attacks through a self-calibration procedure that tunes the temperature parameter using only clean accuracy correlations. Evaluating 22 models from RobustBench across CIFAR-10 and ImageNet, we find that the decomposition is exact, that per-class scores reveal consistent vulnerability patterns (e.g., ``cat'' is the weakest class in 76\\% of CIFAR-10 models), and that more robust models tend to exhibit greater class-level disparity. These results establish a practical, attack-free auditing pipeline for diagnosing where certified robustness guarantees fail to protect all classes equally. We release our code on \\href{https://github.com/aryashah2k/gf-score}{GitHub}.",
      "published": "2026-04-14T14:03:22Z",
      "abstract_url": "http://arxiv.org/abs/2604.12757v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12757v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Information-Theoretic Optimization for Task-Adapted Compressed Sensing Magnetic Resonance Imaging",
      "authors": [
        "Xinyu Peng",
        "Ziyang Zheng",
        "Wenrui Dai",
        "Duoduo Xue",
        "Shaohui Li",
        "Chenglin Li",
        "Junni Zou",
        "Hongkai Xiong"
      ],
      "abstract": "Task-adapted compressed sensing magnetic resonance imaging (CS-MRI) is emerging to address the specific demands of downstream clinical tasks with significantly fewer k-space measurements than required by Nyquist sampling. However, existing task-adapted CS-MRI methods suffer from the uncertainty problem for medical diagnosis and cannot achieve adaptive sampling in end-to-end optimization with reconstruction or clinical tasks. To address these limitations, we propose the first task-adapted CS-MRI from the information-theoretic perspective to simultaneously achieve probabilistic inference for uncertainty prediction and adapt to arbitrary sampling ratios and versatile clinical applications. Specifically, we formalize the task-adapted CS-MRI optimization problem by maximizing the mutual information between undersampled k-space measurements and clinical tasks to enable probabilistic inference for addressing the uncertainty problem. We leverage amortized optimization and construct tractable variational bounds for mutual information to jointly optimize sampling, reconstruction, and task-inference models, which enables flexible sampling ratio control using a single end-to-end trained model. Furthermore, the proposed framework addresses two kinds of distinct clinical scenarios within a unified approach, i.e., i) joint task and reconstruction, where reconstruction serves as an auxiliary process to enhance task performance; and ii) task implementation with suppressed reconstruction, applicable for privacy protection. Extensive experiments on large-scale MRI datasets demonstrate that the proposed framework achieves highly competitive performance on standard metrics like Dice compared to deterministic counterpart but provides better distribution matching to the ground-truth posterior distribution as measured by the generalized energy distance (GED).",
      "published": "2026-04-14T13:23:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.12709v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12709v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "MISID: A Multimodal Multi-turn Dataset for Complex Intent Recognition in Strategic Deception Games",
      "authors": [
        "Shufang Lin",
        "Muyang Chen",
        "Xiabing Zhou",
        "Rongrong Zhang",
        "Dayou Zhang",
        "Fangxin Wang"
      ],
      "abstract": "Understanding human intent in complex multi-turn interactions remains a fundamental challenge in human-computer interaction and behavioral analysis. While existing intent recognition datasets focus mainly on single utterances or simple dialogues, real-world scenarios often involve sophisticated strategic interactions where participants must maintain complex deceptive narratives over extended periods. To address this gap, we introduce MISID, a comprehensive multimodal, multi-turn, and multi-participant benchmark for intent recognition. Sourced from high-stakes social strategy games, MISID features a fine-grained, two-tier multi-dimensional annotation scheme tailored for long-context discourse analysis and evidence-based causal tracking. Our systematic evaluation of state-of-the-art Multimodal Large Language Models (MLLMs) on MISID reveals critical deficiencies in complex scenarios, including text-prior visual hallucination, impaired cross-modal synergy, and limited capacity in chaining causal cues. Consequently, we propose FRACTAM as a baseline framework. Using a ``Decouple-Anchor-Reason'' paradigm, FRACTAM reduces text bias by extracting pure unimodal factual representations, employs two-stage retrieval for long-range factual anchoring, and constructs explicit cross-modal evidence chains. Extensive experiments demonstrate that FRACTAM enhances mainstream models' performance in complex strategic tasks, improving hidden intent detection and inference while maintaining robust perceptual accuracy. Our dataset is available at https://naislab.cn/datasets/MISID.",
      "published": "2026-04-14T13:07:54Z",
      "abstract_url": "http://arxiv.org/abs/2604.12700v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12700v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "BID-LoRA: A Parameter-Efficient Framework for Continual Learning and Unlearning",
      "authors": [
        "Jagadeesh Rachapudi",
        "Ritali Vatsi",
        "Praful Hambarde",
        "Amit Shukla"
      ],
      "abstract": "Recent advances in deep learning underscore the need for systems that can not only acquire new knowledge through Continual Learning (CL) but also remove outdated, sensitive, or private information through Machine Unlearning (MU). However, while CL methods are well-developed, MU techniques remain in early stages, creating a critical gap for unified frameworks that depend on both capabilities. We find that naively combining existing CL and MU approaches results in knowledge leakage a gradual degradation of foundational knowledge across repeated adaptation cycles. To address this, we formalize Continual Learning Unlearning (CLU) as a unified paradigm with three key goals: (i) precise deletion of unwanted knowledge, (ii) efficient integration of new knowledge while preserving prior information, and (iii) minimizing knowledge leakage across cycles. We propose Bi-Directional Low-Rank Adaptation (BID-LoRA), a novel framework featuring three dedicated adapter pathways-retain, new, and unlearn applied to attention layers, combined with escape unlearning that pushes forget-class embeddings to positions maximally distant from retained knowledge, updating only 5% of parameters. Experiments on CIFAR-100 show that BID-LoRA outperforms CLU baselines across multiple adaptation cycles. We further evaluate on CASIA-Face100, a curated face recognition subset, demonstrating practical applicability to real-world identity management systems where new users must be enrolled and withdrawn users removed.",
      "published": "2026-04-14T12:57:56Z",
      "abstract_url": "http://arxiv.org/abs/2604.12686v1",
      "pdf_url": "https://arxiv.org/pdf/2604.12686v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
