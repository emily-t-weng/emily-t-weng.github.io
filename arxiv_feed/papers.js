const PAPERS_DATA = {
  "last_updated": "2026-07-22 03:26:58 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Copy Less, Ground More: Overcoming Repetitive Copying in Long-Context Reasoning via Evidence-Aware Reinforcement Learning",
      "authors": [
        "Lizhe Fang",
        "Weizhou Shen",
        "Tianyi Tang",
        "Yisen Wang"
      ],
      "abstract": "Large language models that generate step-by-step reasoning traces have achieved strong performance on complex tasks, and extending them to long-context settings has emerged as an important frontier. However, we identify a critical failure mode in this regime: \\emph{repetitive copying}, where models extensively copy text from the input into their reasoning traces rather than productively solving the problem. We show that this behavior is pervasive across frontier long-context LLMs and intensifies with context length. By separating each prompt into task-relevant key evidence and irrelevant distractor context, we further show that the root cause is insufficient grounding: models copy from the prompt indiscriminately, and those that fail to focus on key evidence are far more likely to answer incorrectly. Motivated by this diagnosis, we propose GEAR (Grounding Evidence-Aware Reward), a reward shaping method that augments the accuracy signal with a grounding reward for overlap with key evidence and a distractor penalty for overlap with irrelevant context. To enable GEAR on natural-language data, we develop an automated pipeline that constructs evidence-annotated training data from arbitrary documents. We validate GEAR across multiple model scales and benchmarks, showing consistent improvements of up to +4.6 average points over standard RL with accuracy-based rewards, with larger gains at longer contexts, while also reducing repetitive copying and thinking length. Our findings suggest that, even as long-context evaluation shifts from simple retrieval toward complex reasoning, accurate grounding in relevant evidence remains an indispensable capability with substantial room for improvement.",
      "published": "2026-07-21T17:59:21Z",
      "abstract_url": "http://arxiv.org/abs/2607.19345v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19345v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Agents in the Wild: Where Research Meets Deployment",
      "authors": [
        "Grace Hui Yang",
        "Pranav N. Venkit",
        "Hooman Sedghamiz",
        "Enrico Santus",
        "Victor Dibia",
        "Ioana Baldini"
      ],
      "abstract": "Agentic systems large language model (LLM) based architectures capable of reasoning, planning, acting, and coordinating with tools and other agents are rapidly transitioning from research prototypes to production scale deployments across domains such as software engineering, scientific discovery, and finance. While academic work has emphasized benchmarks and algorithmic innovation, deployment raises new challenges around robustness, safety, and reliability. This tutorial brings together researchers and practitioners to explore advances in reasoning and planning, multi agent coordination, and evaluation, highlighting open challenges arising from deployment experience. Through applied case studies in pharmaceutical discovery and financial systems, we analyze common design patterns that make agentic systems successful, and discuss practical mitigation strategies for failure modes, such as verification pipelines, fallback mechanisms, and human in the loop supervision. Attendees will gain a comprehensive view of the field along with concrete design patterns, evaluation checklists, and templates for safe and reliable deployment across industries.",
      "published": "2026-07-21T17:55:10Z",
      "abstract_url": "http://arxiv.org/abs/2607.19336v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19336v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Provable diffusion-based posterior sampling for linear inverse problems via DDIM",
      "authors": [
        "Yuchen Jiao",
        "Na Li",
        "Changxiao Cai",
        "Yuxin Chen",
        "Gen Li"
      ],
      "abstract": "Diffusion-based methods have achieved remarkable empirical success in solving inverse problems. However, many existing posterior samplers either lack rigorous theoretical guarantees or incur substantial computational overhead. We propose a simple and efficient algorithm, called \\pddim, for solving linear inverse problems with diffusion priors via a DDIM-type sampler. Our method requires only lightweight, coordinate-wise modifications to the standard DDIM update, while explicitly incorporating the measurement model. The key idea is to perform posterior sampling separately along each singular direction of the measurement operator: for each direction, the sampler follows the learned diffusion prior when the observation signal-to-noise ratio (SNR) is below the corresponding diffusion SNR, and switches to a calibrated measurement-based predictor otherwise. We prove that the proposed sampler converges to the Bayesian posterior conditioned on the measurements. Empirical results show that the proposed sampler performs favorably against existing diffusion-based posterior samplers across a range of image restoration tasks, achieving the best performance on the majority of evaluation metrics considered. Overall, our results convert posterior sampling for noisy linear inverse problems to simple coordinate-wise DDIM updates, yielding an efficient, easy-to-implement algorithm with provable posterior consistency.",
      "published": "2026-07-21T17:53:36Z",
      "abstract_url": "http://arxiv.org/abs/2607.19333v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19333v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "ISO: An RLVR-Native Optimization Stack",
      "authors": [
        "Hanqing Zhu",
        "Wenyan Cong",
        "Zhizhou Sha",
        "Sagnik Mukherjee",
        "Xinyuan Song",
        "David González-Martínez",
        "Xiaoxia Wu",
        "Yuandong Tian",
        "Shiwei Liu",
        "David Z. Pan",
        "Zhangyang \"Atlas\" Wang"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) is rapidly advancing the reasoning capabilities of language models, yet the optimization layer that converts reward feedback into weight-space updates remains poorly understood. Building on our prior analysis (Zhu et al., 2025), we study this missing layer through the singular structure of model weights and identify spectral inheritance: RLVR can reuse the base model's weight spectra while acquiring new behavior through changes in the associated input and output singular frames. We operationalize spectral inheritance as Isospectral Optimization (ISO), an RLVR-native, fixed-spectrum optimization framework with complementary offline and online instantiations. Offline, ISO-Merger combines the frame changes of shared-base specialists into a single fixed-spectrum model, requiring no post-merge data, rollouts, gradient updates, or on-policy distillation (OPD). It recovers complementary specialist capabilities and achieves the strongest aggregate performance among the compared data-free merging methods. Online, ISO-Optimizer applies a chosen base optimizer, including AdamW and Muon, to the frame variables while keeping the base spectra fixed. Across reasoning and coding tasks ranging from 1.5B to 8B parameters, ISO-Optimizer improves accuracy in the reported runs and reaches matched scores with substantially fewer training steps. On Qwen3-8B-Base, AdamW reaches an aggregate accuracy of 0.495 after 270 training steps. ISO-AdamW reaches the same accuracy after only 100 training steps and improves further to 0.509 after 210 training steps. Together, ISO offers a concrete answer to RLVR's missing optimization layer: rather than inheriting pre-training optimization wholesale, design post-training around the structure of reward-driven adaptation: inherit the spectrum, optimize the frames.",
      "published": "2026-07-21T17:51:36Z",
      "abstract_url": "http://arxiv.org/abs/2607.19331v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19331v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ResearchArena: Evaluating Sabotage and Monitoring in Automated AI R&D",
      "authors": [
        "Lena Libon",
        "Ben Rank",
        "Jehyeok Yeon",
        "David Schmotz",
        "Jeremy Qin",
        "Daniel Donnelly",
        "Derck Prinzhorn",
        "Maksym Andriushchenko"
      ],
      "abstract": "As AI agents begin to automate AI R&D, we need ways to assess whether their outputs are safe to deploy, even when the agents themselves may be untrusted. AI control offers one such approach: rather than trusting the agent, it treats it as a potential adversary and uses a monitor to detect covert sabotage before deployment. We evaluate AI control for automated AI R&D with ResearchArena, a framework spanning four long-horizon tasks: safety post-training, capabilities post-training, CUDA-kernel optimization, and inference-server optimization. Because the deliverable in AI R&D is an artifact that will be deployed and run, we pair each main task with two kinds of hidden side task: an embedded side task that sabotages the submitted model, adapter, kernel, or server itself, and an independent side task that takes a covert out-of-scope action in the sandbox while leaving the artifact honest. We evaluate frontier agents at both sabotage and monitoring, across four monitor types that vary along two axes: whether the monitor may execute and probe the produced artifact, and whether it sees the agent's chain-of-thought. We find that sabotage hidden in the training data is the hardest to catch, flagged fewer than half the time. Letting the monitor run experiments on the artifact, rather than only read the trajectory, helps, but it is not enough: monitors still miss embedded sabotage by inspecting only the surface, by noticing the anomaly but explaining it away, or by probing the artifact with the wrong test. We release ResearchArena as a modular framework for evaluating sabotage and control in automated AI R&D.",
      "published": "2026-07-21T17:41:12Z",
      "abstract_url": "http://arxiv.org/abs/2607.19321v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19321v1",
      "categories": [
        "cs.AI",
        "cs.CR",
        "cs.LG"
      ]
    },
    {
      "title": "Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information",
      "authors": [
        "Priyank Agrawal",
        "Ankur Samanta",
        "Shervin Ghasemlou",
        "Jalaj Bhandari",
        "Kavosh Asadi",
        "Daniel Jiang",
        "Aditya Modi"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) improves reasoning in large language models. Yet, typical RLVR approaches fail on difficult problems: when a model cannot generate any correct solutions, it receives \\textit{zero} learning signal. Providing privileged guidance during training, such as solution prefixes, can help overcome this learning cliff by steering the model towards {correct solutions with non-zero reward}. {We call these rollouts \\textit{off-context}: they are generated from a training prompt that contains privileged guidance, while the target objective is defined by the original prompt without that guidance.} {We introduce} Off-Context GRPO (OC-GRPO), a minimally modified variant of GRPO that uses guided rollouts but applies an importance-corrected objective to steer the update back toward the original unguided objective, avoiding the mismatch that destabilizes uncorrected guided training. Empirically, our algorithm achieves a 3.9\\% absolute improvement (13.8\\% relative gain) over vanilla GRPO on average across standard mathematical reasoning benchmarks with negligible additional cost.",
      "published": "2026-07-21T17:28:40Z",
      "abstract_url": "http://arxiv.org/abs/2607.19313v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19313v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Riemannian Deep Learning:Modules, Networks, and Geometries",
      "authors": [
        "Chen Ziheng"
      ],
      "abstract": "Deep neural networks on manifold-valued representations have attracted growing interest, but many basic components remain tied to specific manifolds, rely on Euclidean approximations, or require costly and numerically fragile geometric operations. This thesis develops a unified framework for Riemannian deep learning from three complementary perspectives: reusable neural modules, manifold-specific network architectures, and the design of underlying geometries. It generalizes batch normalization from Euclidean spaces and individual manifolds to broad classes of Lie groups and gyrogroups, and extends multinomial logistic regression from Euclidean space to SPD manifolds and then to general Riemannian manifolds. It further develops neural networks for several important geometric representations, including an unconstrained model of hyperbolic space, Busemann-based hyperbolic learning, and full-rank correlation matrices. Finally, it introduces adaptive and computationally efficient Riemannian metrics on SPD manifolds, including learnable Log-Euclidean geometries and fast, stable Cholesky-based geometries. The proposed methods are supported by theoretical analysis and validated through numerical experiments and applications in vision, signal processing, graph learning, and genomics.",
      "published": "2026-07-21T17:17:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.19305v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19305v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.DG"
      ]
    },
    {
      "title": "GUIDED Network-Agnostic Feature Initialization for Spatial Transferability in GNN-based Models",
      "authors": [
        "Alessandro Scalese",
        "Santhanakrishnan Narayanan",
        "Constantinos Antoniou"
      ],
      "abstract": "The Traffic Assignment Problem is a fundamental but computationally expensive component of transportation planning. While Graph Neural Networks have emerged as fast, data-driven surrogates, their practical deployment is severely constrained by a spatial generalization gap. Standard models rely on transductive feature initializations that tie travel demand to fixed network topologies, preventing seamless transfer to new urban environments. To overcome this structural limitation, this research proposes a network-agnostic initialization layer, termed Geometrically Unconstrained Inductive Demand EmbeDding (GUIDED). By injecting travel demand as a scalar attribute on auxiliary virtual links rather than as specific node features, this modular framework standardizes the input space regardless of network scale. Extensive experimental evaluation across multiple urban topologies demonstrates that a Heterogeneous Graph Attention Network (HetGAT) model integrated with the proposed GUIDED layer maintains state-of-the-art predictive accuracy on single-network tasks, while demonstrating superior robustness to out-of-distribution demand patterns and maintaining a distinct performance advantage over the baseline even under severe data scarcity. Notably, the proposed feature initialization enables highly parameter-efficient domain adaptation for inter-network transfer learning without artificial input homogenization, establishing a robust foundation for truly inductive models. At the same time, the optimized scatter operations of the initialization layer yield an approximate 50% reduction in training time per epoch compared to the baseline approach. Furthermore, while demonstrated on vehicular traffic, this fundamental abstraction of spatial topology provides a versatile blueprint for generalized origin-destination spatial problems, such as freight logistics and multimodal network optimization.",
      "published": "2026-07-21T16:41:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.19270v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19270v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Toward Auditable Fraud Detection: Combining Graph Features, Model Explanations, and Agentic Case Investigation",
      "authors": [
        "Rahil Sharma"
      ],
      "abstract": "Fraud detection systems must scale with rising transaction volume while remaining explainable and reviewable. We study a layered pipeline on the PaySim dataset that combines a gradient-boosted classifier, graph-derived structural features, an autoencoder-based anomaly signal, TreeSHAP explanations, and a bounded LLM investigation agent applied to cases the classifier scores uncertainly. Before any model comparison, we identify and remove a simulator-specific balance shortcut that would otherwise inflate baseline performance. After this correction, neither the graph features nor the anomaly signal improves Average Precision on the full test set. Both, however, rank fraud better within the subset of cases receiving intermediate baseline scores. In a controlled experiment with injected multi-account fraud rings, engineered structural features recover all injected test transactions, while the tabular baseline misses roughly a quarter of them. The investigation agent underperforms direct thresholding of the classifier it relies on, reaching 65.0% accuracy against 71.7% on a balanced 60-case sample, despite having access to model explanations, graph context, and retrieved reference cases. Of the eight decisions the agent changed, six replaced correct classifier outputs with errors, and it produced a coherent written rationale in each case. An exploratory disagreement-based escalation rule flagged two of these agent errors for human review without flagging any correct decision. We conclude that each component of a layered fraud system contributes only under specific conditions, and that a plausible rationale from an investigation agent is not evidence of a better decision.",
      "published": "2026-07-21T16:37:41Z",
      "abstract_url": "http://arxiv.org/abs/2607.19266v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19266v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Benchmarking Generalization in Financial Statement Fraud Detection: robust evaluation and novel tasks",
      "authors": [
        "Guy Stephane Waffo Dzuyo",
        "Gaël Guibon",
        "Christophe Cerisara",
        "Luis Belmar-Letelier"
      ],
      "abstract": "Financial statement fraud detection (FSFD) is crucial for market integrity but faces challenges from increasingly sophisticated schemes and under-utilized textual data in financial reports. Existing methods often rely on random data splits, leading to overoptimistic performance estimates that do not reflect real-world generalization to new companies or future periods. To address this recurring problem with the state of the art, we propose a robust FSFD framework leveraging Large Language Models (LLMs) to integrate both structured financial data and unstructured textual information from financial reports. We provide a more realistic evaluation through a novel and challenging benchmark task called Company-Isolated FSFD (CI-FSFD). We construct and make publicly available a comprehensive U.S. company dataset combining financial statements, summarized MD&A text, and fraud labels. Our approach achieves the best performance on the challenging CI-FSFD task, demonstrating the critical value of textual data and robust evaluation for reliable financial fraud detection.",
      "published": "2026-07-21T16:32:54Z",
      "abstract_url": "http://arxiv.org/abs/2607.19259v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19259v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Prompt Design at Scale: How Format, Instruction Count, and Context Length Shape Instruction Adherence and Hallucination in Large Language Models",
      "authors": [
        "Netanel Eliav"
      ],
      "abstract": "Practitioners make three prompt-design decisions with almost no controlled evidence behind them: how to format instructions and context (markdown, plain text, prose, or tabular), how many simultaneous instructions a system prompt can carry before compliance degrades, and how much context a model can hold before recall and honesty degrade. We report two controlled experiments crossing all three factors on one held, contamination-free synthetic corpus (the \"Book of Veyra,\" 8,780 uniquely-named entities, deterministically regenerable from a fixed seed), evaluated across five models. Experiment 1 (960 calls/model) measures instruction-following decay as rule count N grows from 10 to 160, crossed with four formats and system-prompt vs. user-turn placement. Perfect-response rate collapses to zero by N=80 for every model, format, and placement. Placement produces effects at least as large as format at N=160 in most models, but the direction is model-specific. No model shows a reliable markdown advantage; one 35B model favors plain text instead. Experiment 2 (5,520 calls/model) measures recall accuracy, false-premise sycophancy, and absent-fact fabrication across a 2k-to-512k-token context ladder in the same four formats. Recall stays near ceiling through 64-128k tokens, then degrades sharply and format-dependently: one model's accuracy spread reaches 48 points at 128k tokens. Fabrication never occurs (0/5,760 probes), and sycophancy stays negligible (<=8.3%). What rises sharply near each model's context ceiling is outright refusal to answer (0% to 79-90%), distinct from sycophancy or fabrication. Neither pre-registered format ordering holds, and token overhead (+22% to +37% over plain text) further changes which format is preferable where accuracy spread is genuine. We release the full harness, corpus generator, and raw results (VeyraBench): https://github.com/iNetanel/veyrabench",
      "published": "2026-07-21T16:31:35Z",
      "abstract_url": "http://arxiv.org/abs/2607.19257v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19257v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Inference-Time Steering for Cross-Lingual Factual Consistency in LLMs",
      "authors": [
        "Alexander Manev"
      ],
      "abstract": "Although Large Language Models (LLMs) demonstrate remarkable multilingual fluency, their internal knowledge representations remain disproportionately biased toward high-resource languages. This leads to cross-lingual factual inconsistency, where they shift their empirical answer distributions based solely on the prompt language. We investigate whether these biases can be mitigated at inference time, forcing an English-prompted model to answer as if it were queried in target languages (German, Spanish, Bulgarian), and evaluate four intervention strategies: zero-shot contextual steering (persona prompting), internal representation manipulation via Contrastive Activation Addition (CAA), and lightweight weight modification via Direct Preference Optimization (DPO) trained on benchmark-derived factual data as well as conceptual generalization data. To assess alignment, we curate a multilingual factual dataset alongside a novel generalization benchmark comprising culturally rooted queries to determine whether factual interventions transfer to broader target-centric preferences. Experiments on Gemma 3 12B Instruct reveal persona prompting to be the strongest overall intervention, balancing efficacy, safety, and out-of-domain generalization. While CAA yields sharp inconsistency benchmark shifts, it is configuration-sensitive and risks knowledge degradation. DPO-based adapters offer permanent, yet narrower and less transferable gains. These findings suggest that cross-lingual inconsistency is at least partly a selection problem, and that simple contextual interventions may outperform more invasive methods for robust, transferable alignment.",
      "published": "2026-07-21T16:15:05Z",
      "abstract_url": "http://arxiv.org/abs/2607.19243v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19243v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "The Price of Reasoning: Cost-Quality Tradeoffs in Reinforcement Learning for Neural Machine Translation",
      "authors": [
        "Michael Jungo",
        "Aixiu An"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) has been established as a viable paradigm for the post-training of Large Language Models (LLMs), including downstream tasks, such as Neural Machine Translation (NMT). With the latest research indicating that RLVR could be the preferred training method for translating legal documents due to the induced reasoning capabilities, it raises the question whether it is really attributed to the reasoning or more generally to the training paradigm. We investigate the importance of including the model's reasoning trace in the generated responses during both training and inference by systematically omitting it from one of the phases. Our experiments show that including the reasoning, specifically during inference, has a positive effect on the overall translation quality. Furthermore, we recognise that the reasoning leads to an increase in output tokens, hence we study the cost-quality tradeoff between the increased computational demands and the improved translation quality.",
      "published": "2026-07-21T15:57:36Z",
      "abstract_url": "http://arxiv.org/abs/2607.19226v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19226v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Beyond Score Prediction: LLM-Based Essay Scoring and Feedback Generation via Reinforcement Learning with Rubric Rewards",
      "authors": [
        "Xuefeng Jin",
        "Jiashuo Zhang",
        "Teng Cao",
        "Bin Yang"
      ],
      "abstract": "Large language models (LLMs) have been widely applied to automated essay scoring (AES) and automated feedback generation (AFG). However, existing studies rely primarily on prompt engineering or supervised fine-tuning, while systematic research on reinforcement learning (RL) post-training and automated evaluation of feedback quality remains limited. We propose RLAES, a unified LLM framework that jointly optimizes essay scoring and feedback generation through RL. To make feedback quality measurable, interpretable, and usable for training, we introduce Rubric-based Feedback Evaluation (RFE), an essay-grounded feedback evaluation framework comprising 166 fine-grained binary rubric items and an LLM-as-judge. Building on RFE, we propose Adaptive Gated Feedback Optimization (AGFO), which activates rubric-based feedback rewards on demand during RL, reducing evaluation overhead while improving feedback quality. We also propose Adjacent Contrastive Reasoning (ACR) to improve ordinal score calibration by explicitly contrasting adjacent score levels. Experimental results show that the RFE framework captures essay-feedback consistency, exhibits strong pairwise discriminative power, and closely aligns with expert preferences. On the ASAP benchmark, RLAES-AGFO achieves the best scoring performance among LLM-based methods (QWK = 0.803), while maintaining feedback quality comparable to GPT-5.5 and avoiding the feedback degradation observed under score-only RL. Code and datasets are publicly available at https://github.com/hellomuyi/RLAES.",
      "published": "2026-07-21T15:49:02Z",
      "abstract_url": "http://arxiv.org/abs/2607.19219v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19219v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Assessment in Team Problem-Solving Exercises in Computing Education",
      "authors": [
        "Valdemar Švábenský",
        "Jan Vykopal",
        "Sukrit Leelaluk",
        "Pavel Čeleda",
        "Fumiya Okubo",
        "Atsushi Shimada"
      ],
      "abstract": "This full paper in the research-to-practice track presents methods for assessing student teams in tabletop exercises (TTXs). TTXs enable learner teams to prepare for workplace tasks and practice crisis responses, such as resolving cybersecurity incidents. While assessment is essential for determining how well teams achieve learning objectives, the complex, open-ended nature of TTXs often leads to delayed or incomplete feedback. TTX learning platforms can record teams' actions and communication; yet, leveraging these data to assess performance is underexplored. To address this gap, we compared two post-TTX team assessment methods -- clustering and large language models (LLMs) -- using an original dataset from 81 participants across two countries. We evaluated these methods against instructor-assigned scores based on standardized rubrics. Clustering grouped teams that approached TTX tasks similarly, enabling instructors to deliver faster, targeted feedback to teams within a cluster. This method was valid and reliable, with low computational requirements. LLMs used the standardized rubrics to assess teams' communication. While GPT-4o frequently disagreed with instructor scores, GPT-5.2 demonstrated considerably lower error. The researched methods have been integrated into INJECT, an open-source TTX learning platform, to support scalability and teaching practice. To encourage community adoption, we publicly share all datasets, software tools, and a full-fledged TTX scenario.",
      "published": "2026-07-21T15:40:16Z",
      "abstract_url": "http://arxiv.org/abs/2607.19209v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19209v1",
      "categories": [
        "cs.CY",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU",
      "authors": [
        "Fan Jiang",
        "Zhaoxu Sun",
        "Mengchao Wang",
        "Ziyu Zhu",
        "Chiyu Wang",
        "Yunpeng Zhang",
        "Wenlin Liu",
        "Yun Wang",
        "Xue Zheng",
        "Rui Sun",
        "Junfeng Ni",
        "Hongyu Pan",
        "Zhongxu Sun",
        "Fei Yu",
        "Zengye Ge",
        "Mengmeng Du",
        "Nianfei Fan",
        "Mingchao Sun",
        "Yu Liu",
        "Yongchang",
        "Yanqing Zhu",
        "Jiahang Wang",
        "Ning Ying",
        "Yuze Xuan",
        "Di Yang",
        "Zhicheng Liu",
        "Zhe Gao",
        "Tingbing Xu",
        "Jiacheng Sui",
        "Wenjin Yang",
        "Junnan Lai",
        "Shufeng Liu",
        "Yuan Liu",
        "Zheng Zhou",
        "Yingliang Peng",
        "Dawei Cao",
        "Kaifeng Sheng",
        "Yuxiang Cai",
        "Fei Lu",
        "Mu Xu",
        "Ning Guo"
      ],
      "abstract": "We present ABot-World-0, an action-conditioned video world model for real-time, long-horizon closed-loop interaction, supported by a multi-source data infrastructure spanning AAA games, simulation engines, and internet videos to learn controllable world dynamics. WorldExplorer performs agent-driven collection guided by training feedback, while a unified pipeline applies 14 deterministic quality checks, VLM-based assessment, and synchronized action and text annotation. We progressively distill a bidirectional action-conditioned teacher into a causal student through teacher forcing and ODE distillation, and introduce LongForcing to align long student self-rollouts with an extended-horizon teacher, mitigating accumulated distribution shift and autoregressive drift. Raw keyboard actions provide a unified control interface for scene roaming and third-person character interaction, while reference-character memory provides persistent appearance cues for identity consistency during third-person rollouts. For deployment, we co-design a streaming inference stack with a lightweight VAE decoder, efficient attention, memory-aware scheduling, and low-bit DiT inference. Across optimized low-bit configurations, ABot-World-0 streams 720P video at up to 16 FPS on a single NVIDIA RTX 5090 desktop GPU, with 1.2s action-to-first-frame latency and approximately 19GiB peak VRAM. Experiments on WorldRoamBench and extended interactive rollouts demonstrate competitive controllability and coherent long-horizon world evolution.",
      "published": "2026-07-21T15:26:50Z",
      "abstract_url": "http://arxiv.org/abs/2607.19191v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19191v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning",
      "authors": [
        "Aixiu An",
        "Michael Jungo",
        "Eloi Eynard",
        "Mark Drenhaus",
        "Andreas Fischer",
        "Jean Hennebert",
        "Sébastien Rumley"
      ],
      "abstract": "Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning. In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the quality of small ``base'' models can be greatly enhanced, and that reinforcement learning with verifiable rewards can be applied to NMT in the legal domain and surpasses the translation quality of supervised fine-tuning. The performance of enhanced small models is close to the one of state-of-the-art reasoning models yet remains inferior. We also note that re-training paradigms yield diminishing returns as model size increase. The code and models are publicly available at https://github.com/aixiuxiuxiu/Legal-MT-SFT-RL.",
      "published": "2026-07-21T15:15:56Z",
      "abstract_url": "http://arxiv.org/abs/2607.19181v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19181v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Breaking the Homogeneity Assumption: Specialized Multi-Generator Adversarial Learning for Rare Failure Detection in Predictive Maintenance",
      "authors": [
        "Alexis Lazanas",
        "Georgios Kampouropoulos"
      ],
      "abstract": "Supervised learning models in the predictive maintenance field are regularly trained on highly imbalanced industrial datasets: machine failures occur rarely but have a disproportionate effect on operations. In addition to the clear class disparity, failure data are typically non-homogeneous, with different failure modes arising from distinct physical processes and exhibiting a multimodal distribution across minorities and classes. Traditional imbalance-management methods, e.g., undersampling, SMOTE-based interpolation, or cost-sensitive learning, typically assume that the minority population is homogeneous. This means their effectiveness is severely limited in the multifaceted conditions encountered in industrial practice. This paper determines the possibility of a failure-type-conscious generative augmentation program to improve the identification of infrequent failures in predictive maintenance systems. An experimental design that is leakage-safe is used to compare five imbalance-handling methods: cost-sensitive learning, random undersampling, SMOTE oversampling, single-generator GAN augmentation, and a specialized multi-generator GAN architecture that has independent generators that are asked to learn individual failure subtypes. Precision/Recall-oriented measures are used to quantify model performance; the main evaluation measure is the PR-AUC. Experiments conducted on the AI4I 2020 predictive maintenance dataset indicate that the proposed multi-generator GAN framework produces more realistic minority samples, yielding higher PR-AUC and recall scores compared to traditional resampling methods and individual-generator GAN augmentation.",
      "published": "2026-07-21T14:52:44Z",
      "abstract_url": "http://arxiv.org/abs/2607.19153v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19153v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Incomplete Observations Boost Evolutionary Performance in Ocean Modeling",
      "authors": [
        "Yangyang Kong",
        "Yutong Jiang",
        "Yanhai Gan",
        "Junyu Dong",
        "Feng Gao",
        "Xiaopei Lin"
      ],
      "abstract": "Data-driven methods have revolutionized ocean modeling, yet current approaches rely heavily on complete reanalysis datasets, imposing computational constraints and limiting model performance to that of the training data. Here, we present a generative state-space model and an optimization framework that enable learning directly from sparse and noisy observations. The model is essentially a hidden Markov model with a continuous state space, where oceanic physical quantities are treated as hidden states and measurements as observations, enabling a unified representation of ocean fields and observational data. Both the initial-state and state-transition modules are implemented as neural networks to capture the complexity and temporal evolution of ocean states, while the emission module is formulated as a masked Gaussian distribution. To train the model from sparse observations, we derive an optimization framework based on the expectation-maximization (EM) algorithm. The framework alternately reconstructs high-fidelity ocean fields via Langevin dynamics and optimizes deep neural networks to capture temporal evolution. Theoretical analysis shows that the framework maximizes the likelihood of observations under the generative model. For efficiency, we assume that ocean-state evolution follows a stationary, ergodic, and Markovian stochastic process and adopt only length-two state sequences during optimization. Experiments on CMIP6 simulation data and FY-3D satellite data demonstrate high-fidelity reconstruction and accurate prediction, showing that sparse observations can directly improve the model's representation of ocean-state dynamics. This work offers a scalable pathway for next-generation Earth system models to learn directly from sparse, incomplete real-world observations.",
      "published": "2026-07-21T14:39:55Z",
      "abstract_url": "http://arxiv.org/abs/2607.19147v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19147v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MIRAGE: Multi-scale Lesion-Informed Representation with Auxiliary Guidance for MRI Contrast Enhancement",
      "authors": [
        "Andrea Borghesi",
        "Xin Wang",
        "Jonas Teuwen",
        "George Yiasemis"
      ],
      "abstract": "Inferring contrast enhancement from one pre-contrast breast MRI slice is underdetermined: post-contrast appearance contains physiological information that is not uniquely encoded in baseline anatomy. Optimizing only paired pixel fidelity can suppress uncertain lesion enhancement, whereas adversarial or stochastic generative objectives can favor realistic post-contrast appearance without guaranteeing patient-specific lesion fidelity. We introduce MIRAGE, a residual 2D U-Net that combines global reconstruction and perceptual losses with three forms of lesion-aware supervision available only during training: an asymmetric penalty for missed tumor enhancement, multi-scale auxiliary tumor segmentation, and guidance through a frozen post-contrast tumor segmentation nnU-Net. We evaluate the method on 301 cases from the multi-centre MAMA-SYNTH data using eight complementary image-, region-, radiomics-, and segmentation-based metrics. MIRAGE ranks first on six metrics and markedly improves downstream lesion localization over tuned pix2pix, conditional diffusion, and latent bridge-matching baselines. The generative alternatives retain advantages in LPIPS or contrast classification, revealing a clear fidelity-utility trade-off. Leave-one-in and leave-one-out ablations show that the losses are partly redundant for lesion localization but exert distinct effects on appearance, radiomics, and boundary accuracy. These results support task-aware synthesis while also showing that its apparent optimality is conditional on the downstream models and metrics used to define utility.",
      "published": "2026-07-21T14:26:59Z",
      "abstract_url": "http://arxiv.org/abs/2607.19137v1",
      "pdf_url": "https://arxiv.org/pdf/2607.19137v1",
      "categories": [
        "eess.IV",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
