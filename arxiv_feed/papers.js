const PAPERS_DATA = {
  "last_updated": "2026-07-21 03:28:09 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "A Continual Validation, Updating, and Decision-Making Framework for Self-Adaptive Digital Twins via Robust Model Predictive Control: A Case Study in Additive Manufacturing",
      "authors": [
        "Yi-Ping Chen",
        "Ying-Kuan Tsai",
        "Vispi Karkaria",
        "Seul Lee",
        "Daniel Apley",
        "Wei Chen"
      ],
      "abstract": "Digital Twins rely on surrogate models to mirror physical systems in real time, yet these models can degrade as operating conditions evolve, a phenomenon known as concept drift. Maintaining surrogate fidelity under drift, particularly when models must also capture aleatoric uncertainty, remains an open challenge. Existing adaptive frameworks lack principled mechanisms for detecting when updates are needed, for efficiently adapting models from limited streaming data, and for certifying that updates genuinely improve predictive performance. Here we present an adaptive Digital Twin framework that integrates a Fisher score--based multivariate drift detector, Low-Rank Adaptation (LoRA) for parameter-efficient continual learning, and a Mann--Whitney $U$ test for online statistical validation. The framework monitors surrogate-model confidence via Fisher score vectors, triggers targeted fine-tuning of fewer than 1% of model parameters upon drift detection, and statistically certifies predictive improvement before deploying the updated surrogate. Applied to a stochastic linear system and a directed energy deposition additive manufacturing process as case studies, the framework successfully detects distributional shifts with short delays and restores both predictive accuracy and uncertainty quantification under abrupt and incremental drift. These results establish a statistically rigorous and computationally tractable pathway for sustaining the trustworthiness of neural-network--based Digital Twins throughout their operational life cycle.",
      "published": "2026-07-20T17:07:41Z",
      "abstract_url": "http://arxiv.org/abs/2607.18164v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18164v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.ST"
      ]
    },
    {
      "title": "OR Else: A Differentiable Trust Region for Policy Optimization",
      "authors": [
        "Chinmay Rane",
        "Kanishka Tyagi",
        "Michael Manry"
      ],
      "abstract": "PPO and the GRPO baseline studied here use clipped surrogate objectives whose favorable-direction saturation introduces an abrupt change in the scalar objective's derivative. We ask whether Output Reset (OR), a smooth one-sided saturation rule, offers a useful alternative for large language model post-training. PPO-OR and GRPO-OR replace the clipped policy term with an OR squared-margin loss in rollout-relative token log-ratio space; the advantage sign determines the update direction, and a token contributes zero direct OR residual after crossing the favorable margin. We compare PPO-clip with PPO-OR under generalized advantage estimation (GAE), and GRPO with GRPO-OR under group-relative advantages, using \\texttt{Llama-3.2-1B-Instruct} on Anthropic \\texttt{hh-rlhf} with one shared reward model and three seeds per method. Under GAE, PPO-OR has a mean final training-time reward-model score $0.305$ higher than PPO-clip, with a larger observed across-seed spread. Under group-relative advantages, GRPO-OR does not have a higher mean score, but shows a smaller observed spread, a near-zero terminal OR residual, and a declining overshoot fraction, while the matched GRPO clipped-objective trace remains variable. Both group-relative methods exhibit substantially larger rollout-to-current log-ratio displacement than the GAE methods, and OR does not consistently reduce it. Thus, OR changes optimization behavior in both matched comparisons, but the observed reward effect differs between them. At $G=2$, the GRPO-OR diagnostics do not translate into a reward-score gain. Whether larger groups change this outcome remains open. The reported scores are training-time reward-model measurements, not held-out human-preference performance.",
      "published": "2026-07-20T17:07:40Z",
      "abstract_url": "http://arxiv.org/abs/2607.18163v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18163v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Differentiable Logic Gate Networks for Low-Latency EEG Classification on Edge Devices",
      "authors": [
        "Shyamal Y. Dharia",
        "Stephen D. Smith",
        "Camilo E. Valderrama"
      ],
      "abstract": "Real-time EEG classification on edge devices is bottlenecked by the floating-point arithmetic of conventional neural networks. We investigated Differentiable Logic Gate Networks (Diff-Logic) as a hardware-native alternative that compiles models into pure Boolean circuits executable via bitwise CPU operations. Through rigorous iso-parameter experiments across four EEG datasets spanning two classification tasks, binary dementia detection and 3-class emotion recognition, we compared Diff-Logic against matched-capacity Multi-Layer Perceptron (MLP) and Binarized Neural Network (BNN) baselines at four complexity tiers (50k-500k parameters). On dementia screening, Diff-Logic achieved 80.2% Macro F1, outperforming the MLP baseline by 6.8%. On emotion recognition, the MLP retained a moderate performance advantage but incurred a 2.3$\\times$ higher latency and 14$\\times$ larger model size when deployed on a power-constrained (7W) Nvidia Jetson Orin Nano CPU (Single-core). Critically, Diff-Logic inference time remained nearly constant across a 10$\\times$ increase in model scale, achieving a peak speedup of 2.9$\\times$ over MLPs at the largest complexity tier. Our results establish logic-based neural architectures as a practical paradigm for resource-constrained brain-computer interfaces, achieving competitive or superior performance while natively satisfying the latency and memory constraints of portable edge deployment. Code is available on GitHub: https://github.com/Shyamal-Dharia/eeg-difflogic",
      "published": "2026-07-20T16:49:30Z",
      "abstract_url": "http://arxiv.org/abs/2607.18149v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18149v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LLMs and Agentic AI Systems for Smart Grids: A Tutorial on Architectures and Applications",
      "authors": [
        "Daniela Rojas",
        "Abdulwahab Albassam",
        "Aidan G. Leung",
        "Jett Ngo",
        "Ryan Luo",
        "Peter R. Quawas",
        "Junpyung Kim",
        "Kangkai Liang",
        "Mansi Nanavati",
        "Jonathan Mai",
        "Meng-Chi Tsai",
        "Yun-Tong Tsai",
        "Yize Chen",
        "Yuanyuan Shi"
      ],
      "abstract": "Large language models (LLMs) and agentic AI systems have evolved from natural language tasks to using external tools to plan, retrieve, and act in technical domains. In smart grids, recent work applies agentic schemes to forecasting, optimization, and control, wrapping trusted solvers behind language interfaces and orchestrating multi-step workflows. The literature lacks a unified approach to designing and evaluating such systems. LLMs can produce numerically plausible yet physically infeasible outputs, evaluation protocols vary across tasks, and the boundary between what the model should and should not compute is implicit. This paper presents a solver-grounded design principle: a numerical result is reported only when it originates from a trusted tool and passes explicit verification. We review the building blocks of LLM and agentic AI systems for power systems: prompting strategies and agentic architectures. We instantiate the principle in four case studies: wind power forecasting, EV charging scheduling, power flow analysis, and contingency diagnosis, each comparing an LLM-only baseline against its solver-grounded counterpart on identical data and metrics. EVAgent reproduces the CVXPY optimum while reducing LLM-only unmet energy by 7.5-9.5x, and GridDebugAgent repairs 17/39 contingency cases while reducing total violations by 52.3%. We propose a four-group evaluation framework spanning task utility, solver-grounded correctness, faithfulness and safe failure, and cost and latency. A consistent division of labor emerges: the agentic system reliably orchestrates, retrieves, and explains, while trusted tools compute and a verification gate decides what is reported.",
      "published": "2026-07-20T16:45:13Z",
      "abstract_url": "http://arxiv.org/abs/2607.18147v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18147v1",
      "categories": [
        "eess.SY",
        "cs.AI"
      ]
    },
    {
      "title": "Do Language Models Dream of Binding Molecules? Benchmarking LLMs under Spatial Constraints",
      "authors": [
        "Thomas MacDougall",
        "Maksim Kuznetsov",
        "Roman Schutski",
        "Rim Shayakhmetov",
        "Maxim Malkov",
        "Vladimir Aladinskiy",
        "Alex Aliper",
        "Alex Zhavoronkov"
      ],
      "abstract": "Structure-based drug design (SBDD) leverages the 3D structure of protein targets, often complemented by other spatial constraints, to generate candidate binding molecules. While diffusion models have dominated as a leading paradigm for high-quality 3D molecule generation, LLM-based methods are rapidly emerging in molecular design and have shown competitive performance in pocket-conditioned molecular generation. However, their ability to reason about physics and 3D spatial environments is largely underexplored. In this work, we systematically analyze whether current general-purpose LLMs are capable of navigating complex 3D constraints compared to established baselines such as specialized diffusion models. We consider 3D ligand generation conditioned on protein pockets together with ligand- and interaction-derived spatial constraints, including anchor fragments, pharmacophore points, and mandatory pocket-ligand interactions. To enable this evaluation, we introduce 3D-Fit - a token-efficient benchmarking strategy for assessing LLM performance on multi-conditioned spatial molecule generation. Our findings reveal a clear pattern in LLM spatial capabilities: while they still lag behind state-of-the-art approaches, they are promising and can handle multiple spatial constraints simultaneously, enabling scaling to heterogeneous setups.",
      "published": "2026-07-20T16:43:54Z",
      "abstract_url": "http://arxiv.org/abs/2607.18144v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18144v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "SGA: Plug&Play Geometric Verification for Educational Video Synthesis",
      "authors": [
        "Lopez Jhon",
        "Hinojosa Carlos",
        "Ghanem Bernard"
      ],
      "abstract": "Recent work leverages Large Language Models (LLMs) to generate executable code for pedagogical animations using libraries such as Manim. However, ensuring spatial correctness and visual legibility remains challenging, as existing frameworks emphasize pedagogical content while overlooking geometric occlusions. We propose the Symbolic Geometric Agent (SGA), a plug-and-play module for code-centric animation pipelines that intercepts LLM-generated code, performs partial execution to extract symbolic scene graphs, and applies targeted refinement when spatial conflicts are detected. We further introduce the Manim Visual Quality Score (MVQS), a deterministic rendering-free proxy for spatial integrity. Experiments on the MMMC-Code benchmark across four LLM backbones and two agentic pipelines show that SGA achieves a peak MVQS of 73.11 (Code2Video + GPT-5.1), corresponding to a 16.1% relative improvement over the raw baseline, and improves MVQS in 7 of 8 backbone x pipeline configurations.",
      "published": "2026-07-20T16:11:19Z",
      "abstract_url": "http://arxiv.org/abs/2607.18116v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18116v1",
      "categories": [
        "cs.AI",
        "cs.CV",
        "cs.GR",
        "cs.MA",
        "cs.MM"
      ]
    },
    {
      "title": "How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?",
      "authors": [
        "Prakhar Gupta",
        "Terry Jingchen Zhang",
        "Florent Draye",
        "Bernhard Schölkopf",
        "Zhijing Jin"
      ],
      "abstract": "Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven BCT bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out transfer, and causal intervention. The susceptibility is largely installed by alignment tuning rather than pretraining: pretrained base models barely cave to these biases, and their activations carry no cue-specific signal beyond question content. Within aligned models, each bias becomes a single coherent direction that we can both decode and steer along, recovering the unbiased answer across every family we test. The biases stay representationally distinct, however: cross-bias entanglement is model-specific rather than a property of the bias category, and even behaviorally similar biases occupy different directions. The same intervention also serves as a modest debiasing tool, recovering a meaningful share of bias-induced errors while preserving most correct answers across all instruct families. Cue-induced bias is therefore best understood not as a single flaw in LLMs but as a family of distinct, causally active directions that alignment tuning installs.",
      "published": "2026-07-20T16:10:06Z",
      "abstract_url": "http://arxiv.org/abs/2607.18114v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18114v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering",
      "authors": [
        "Sheldon Yu",
        "Tong Yu",
        "Xunyi Jiang",
        "Rohan Surana",
        "Gagan Mundada",
        "Sungchul Kim",
        "Lina Yao",
        "Julian McAuley",
        "Junda Wu"
      ],
      "abstract": "Extended reasoning has become standard for frontier Large Language Models (LLMs), yet the trajectories these models produce remain largely uncontrollable. Existing methods for shaping how a model reasons are prompt based approaches and operate at the input level, offering no fine-grained control over the reasoning process itself. Related work analyzes and discovers latent transition dynamics in the reasoning traces from Large Language Models. Building on this, we statistically characterize these states, and show that failure trajectories get stuck in self-loops, exhausting the token budget without progress toward the final answer. To intervene on these failures, We propose SOPHIA: Steering Of reasoning Processes via Hidden-state Intervention and Activations. We treat each reasoning trace as a sequence of latent states rather than an unstructured texts, and investigate whether inference time interventions can provide fine-grained control over the self-looping reasoning process. We classify every prefix to a latent state, record step level transitions, and use them to construct a bank of steering vectors indexed by state pairs. At inference time, a controller infers the current state and, given a target state, retrieves the corresponding vector and can also detect self-loops online from the transition structure to prevent the model from sinking into a reasoning black hole. Through extensive experiments, our method reliably intervenes on self-loop failures, with steering vectors that generalize to different state pairs. End task accuracy and token efficiency indicate that fine-grained controllability results in better reasoning quality.",
      "published": "2026-07-20T16:01:57Z",
      "abstract_url": "http://arxiv.org/abs/2607.18100v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18100v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "WorldCupArena: Fine-Grained Evaluation of Language Models and Deep-Research Agents on Football Forecasting",
      "authors": [
        "Zhaokai Wang",
        "Tianlin Gui",
        "Jiayuan Rao",
        "Shangzhe Di",
        "Yihong Tang",
        "Dingli Liang"
      ],
      "abstract": "Predicting a football match before kickoff requires more than knowing past results: a model must use changing information and make a clear prediction before the answer is available. We present WorldCupArena, a dynamic benchmark for language models and deep-research agents. The 2026 FIFA World Cup is its first evaluation, and the same process can be reused for future leagues and cups. Before each match, a model either receives a common evidence package or searches for information itself. It predicts the result and score, likely players and events, match statistics, and the outcome of the competition. After the match, these predictions are compared with the recorded result. We report result accuracy, exact-score accuracy, and a scoreline score that gives some credit when a predicted score is close but not exact, together with scores for the other prediction tasks. Across 104 matches and 13 systems, models with similar result accuracy differ more clearly on detailed predictions. Compared with betting-market and human-fan baselines, the best system shows only small gains in result and exact-score accuracy, but a clearer gain in Scoreline. New schedules can be added as they begin, allowing the benchmark to evaluate future models without using outcomes that are already known. Code, prompts, predictions, and evaluation scripts are open sourced at https://github.com/wzk1015/WorldCupArena.",
      "published": "2026-07-20T15:52:48Z",
      "abstract_url": "http://arxiv.org/abs/2607.18084v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18084v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Enhancing Rubric-based RL via Self-Distillation",
      "authors": [
        "Mingxuan Xia",
        "Yuhang Yang",
        "Chao Ye",
        "Shuai Zhu",
        "Shenzhi Yang",
        "Guangcheng Zhu",
        "Yuhang Zhang",
        "Cheng Peng",
        "Haobo Wang",
        "Siqing Wang"
      ],
      "abstract": "Rubric-based RL has recently shown promise in improving LLMs on open-ended tasks. A widely recognized limitation of rubric-based RL is limited exploration: criteria that no rollout manages to satisfy (Unexplored Criteria, UC) receive no optimization signal. Recent methods address this by incorporating rubric information as external guidance during rollout, yet they introduce a train-inference mismatch: the policy is optimized on rollouts produced under external guidance while this guidance is absent at inference time, causing error accumulation through autoregressive decoding. Moreover, these exploration-focused approaches overlook a fundamentally different failure mode that we term Suppressed Criteria (SC) -- criteria that are satisfied by some rollouts yet whose learning signals are lost during optimization because scalar reward aggregation assigns them non-positive aggregate advantages. Our analysis reveals that SC are remarkably prevalent: over 57% of samples exhibit this failure mode throughout training, with an average of 1.8 SC per sample. To simultaneously address both UC and SC without introducing training-inference mismatch, we propose Criterion-Distilled Policy Optimization (CriPO), which enhances rubric-based RL via on-policy self-distillation. For UC, CriPO constructs a criterion-injection self-teacher and computes a localized forward-KL loss to inject missing behaviors into the policy. For SC, CriPO employs a counterfactual self-teacher to locate criterion-relevant tokens in negative-advantage rollouts and flips their token-level advantages to positive values, preserving useful patterns that would otherwise be suppressed. Experiments on medicine and science benchmarks demonstrate that CriPO consistently outperforms rubric-based RL, achieving stronger final performance with approximately $2\\times$ fewer optimization steps.",
      "published": "2026-07-20T15:50:02Z",
      "abstract_url": "http://arxiv.org/abs/2607.18082v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18082v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SelectInfer: Selective Neuron Loading and Computation for On-Device LLMs",
      "authors": [
        "Huzaifa Shaaban Kabakibo",
        "Eric Schniedermeyer",
        "Artem Burchanow",
        "Lin Wang"
      ],
      "abstract": "Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of Natural Language Processing (NLP) tasks, but their high computational and memory demands pose significant challenges for deployment on resource-constrained edge devices. Existing approaches to model compression and optimization often rely on coarse-grained pruning or quantization, which can compromise accuracy or require re-training and fine-tuning. In this work, we introduce SelectInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices through selective neuron loading and computation. By profiling and identifying both task-specific and general-purpose neurons using an offline LLM profiler, SelectInfer implements two key optimizations: selective loading, which reduces memory footprint by selectively loading a subset of neurons that were identified to be most important during the offline stage, and selective computation, which dynamically computes only the most relevant neurons at runtime. Evaluation across multiple datasets shows that SelectInfer achieves significant reductions in memory footprint and computation while preserving task performance, making it a practical step towards enabling LLM deployment on edge devices",
      "published": "2026-07-20T15:48:33Z",
      "abstract_url": "http://arxiv.org/abs/2607.18081v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18081v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Generalised Bellman recurrence and three dualities in sequential decision-making",
      "authors": [
        "Fernando E. Rosas",
        "David Hyland",
        "Daniel Polani"
      ],
      "abstract": "What gives the Bellman equation its form? We show that the recursive properties of optimal value functions follow from three conditions: that the dynamics decomposes through sufficient statistics, that the return decomposes recursively, and that the aggregation of uncertainty is compatible with both. When all three conditions hold on a common state, the Bellman equation arises from their mutual consistency; when one fails, tractability can often be recovered by augmenting the state or by deforming return or dynamics. The same conditions are shown to give rise to three dualities: one between probability and return, one between return and aggregation, and one between aggregation and probability. Our framework reveals these dualities as arising from a single construction, unifying methods developed separately across reinforcement learning, control, and decision theory.",
      "published": "2026-07-20T15:46:19Z",
      "abstract_url": "http://arxiv.org/abs/2607.18077v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18077v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SGN: A Similarity-based Generative Network for Data Generation under Distribution Shift",
      "authors": [
        "Jiaqi Zhu",
        "Xincheng Chen",
        "Yuncheng Wu",
        "Zhaojing Luo",
        "Beng Chin Ooi"
      ],
      "abstract": "Generative models trained on a source domain often produce samples that are poorly aligned with shifted target domains, limiting their effectiveness for target-domain data augmentation. Although target-specific adaptation can reduce this mismatch, it typically requires additional optimization and domain-specific parameters. We propose a Similarity-based Generative Network (SGN), a reusable framework that is trained once on labeled source data and applied to new target domains without parameter updates. SGN learns a latent space structured by label-induced pairwise similarities while preserving reconstructive information through an encoder-decoder architecture. At generation time, a small labeled representative set from the target domain is encoded and combined in the learned latent space, allowing the generated samples to inherit target-specific characteristics while maintaining class consistency. We further analyze the realizability and dimensionality requirements of the proposed similarity structure. Experiments on image and tabular datasets demonstrate the effectiveness of SGN for target-guided data augmentation under source-to-target distribution shifts.",
      "published": "2026-07-20T15:40:51Z",
      "abstract_url": "http://arxiv.org/abs/2607.18072v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18072v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Human Grounded Evaluation of Large Language Models for Optical Network Automation",
      "authors": [
        "Kiarash Rezaei",
        "Omran Ayoub",
        "Paolo Monti",
        "Carlos Natalino"
      ],
      "abstract": "Large language models (LLMs) are increasingly adopted for network automation, yet their output quality and inference cost can vary substantially across LLM families. We present HuGLEN, a stepwise evaluation pipeline that uses an LLM-as-a-judge together with a small set of expert ratings to enable scalable and reproducible comparison of candidate LLMs, and to rank them using a quality efficiency score (QES). We demonstrate HuGLEN for translating outputs from an explainable artificial intelligence (XAI) model for the optical network quality of transmission (QoT) estimation task into operator-friendly explanations. Our results show that a medium-sized LLM (12B parameters) achieves the highest QES, indicating the best trade-off between explanation quality and efficiency. Overall, HuGLEN reduces the human-labeling burden while supporting consistent model selection for operator-facing automation tasks.",
      "published": "2026-07-20T15:36:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.18068v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18068v1",
      "categories": [
        "cs.NI",
        "cs.AI"
      ]
    },
    {
      "title": "Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security",
      "authors": [
        "Devina Jain",
        "David Hartmann",
        "Chuan Li"
      ],
      "abstract": "LLM-based agents process external content, exposing them to prompt injection and multi-turn manipulation. Most safety benchmarks evaluate defenders against fixed attack pools collected before evaluation, single-turn or multi-turn. We present a 21-scenario benchmark for \\emph{adaptive multi-round attacks against memoryless LLM defenders}: an autonomous LLM attacker observes prior defender responses and pivots across rounds, while each defender response is evaluated as a fresh interaction. Holding the 21 scenarios, attackers, defenders, and structured-output scoring fixed, restricting scoring to the first attacker turn yields $0$-$1\\%$ attack success rate (ASR); allowing 15 rounds of adaptive attack yields $5.4$-$14.0\\%$. Pooling three frontier attacker LLMs uncovers $1.4$-$2.2\\times$ as many unique successful attacks as the best single attacker, and the generated attacks have low cosine similarity ($0.02$-$0.14$) to attacks in existing benchmarks. Claude Opus 4.6 and GPT-5.4 are tied in aggregate ($5.4\\%$ each; overlapping $95\\%$ CIs), but their weaknesses differ sharply: on one scenario Opus reaches $60\\%$ ASR ($95\\%$ CI $36$--$80\\%$) while GPT-5.4 and Gemini each stay at $7\\%$ (CI $1$-$30\\%$; the gap is preserved in a higher-$N$ replication). $13$ of $21$ scenarios distinguish at least one defender pair, yet rankings disagree across scenarios (Kendall's $W = 0.19$). We release the benchmark -- 21 evaluation scenarios, 10 public development scenarios, the orchestrator, baseline harnesses, and a multi-attacker CLI -- plus 945 transcripts from the 3$\\times$3 frontier matrix, an attack-replay dataset, and 18{,}422 gpt-oss-20b battles from an open competition's final scoring rounds.",
      "published": "2026-07-20T15:30:38Z",
      "abstract_url": "http://arxiv.org/abs/2607.18063v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18063v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "AdaHome: An Adaptive Smart Home Assistant using Local Small Language Models",
      "authors": [
        "Eu Jin Lim",
        "Zhaoxing Li",
        "Sebastian Stein"
      ],
      "abstract": "Smart home assistants interpret a wide range of user commands, from explicit device control to underspecified and preference dependent requests. While recent systems based on Large Language Models (LLMs) improve this capability, they often rely on heavyweight reasoning pipelines and cloud-based deployment, limiting their efficiency and suitability for resource-constrained environments, and raising privacy concerns. In addition, existing approaches provide limited support for stable long-term personalization. To address these issues, we present AdaHome, an adaptive smart home assistant designed for locally deployed small language models in smart home environments. Rather than applying complex reasoning uniformly, AdaHome introduces an intent-aware planning framework that dynamically routes commands either to straightforward prompt-based or lightweight reasoning-based components. For commands requiring interpretation, we adopt a Chain-of-Draft strategy to enable efficient and stable decision-making. To support personalization, we further propose a preference adaptation mechanism that learns from user feedback over time without requiring prompt augmentation or model retraining. We evaluate AdaHome against representative LLM-based baselines under a unified small model setting. AdaHome achieves substantially higher accuracy on direct commands (86.7%) while reducing latency by up to 3$\\times$. Furthermore, it maintains competitive performance on ambiguous inputs with lower computational cost. In multi-turn scenarios, AdaHome achieves 88% preference consistency, compared to 52.5% for a prompt augmentation baseline.",
      "published": "2026-07-20T15:01:41Z",
      "abstract_url": "http://arxiv.org/abs/2607.18034v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18034v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Natural Language Access to Domain-Specific Metadata: A Reusable Framework for LLM Query Generation",
      "authors": [
        "Blake G. Fitch",
        "Cato Elia Kurtz"
      ],
      "abstract": "Researchers need to answer ad-hoc questions about the contents of domain-specific archives but often lack the expertise to write structured queries on the metadata. We show that when domain vocabulary and semantics are captured in a well-designed Web Ontology Language (OWL) ontology, Large Language Models (LLMs) can generate accurate structured queries zero-shot, without fine-tuning, retrieval augmentation, or multi-agent orchestration. We present the Natural Language Knowledge Graph Query (NLKGQ) system, a framework and development process that enables natural language access to metadata in such archives. The framework includes a web interface that helps researchers pose natural language questions, which a domain-agnostic harness translates to SPARQL via an LLM and executes against a knowledge graph. The development process begins with capturing domain vocabulary and semantics in a formal OWL ontology. Domain-specific code then extracts metadata from archive sources and imports it into a knowledge graph defined by the ontology. Both are designed for reuse across domains. We demonstrate the system on metadata derived from a large-scale neuroimaging research archive, evaluating multiple LLMs and ontology representations. The best configurations achieve 100% accuracy on a competence and regression question set developed with domain experts. An ablation study across eight ontology representations reveals that readable entity names and semantic annotations are the dominant factors in accuracy, more significant than model choice or prompt engineering. We also compare SPARQL to an auto-generated SQL database as query backends, showing that OWL's structural features provide a substantial advantage over SQL DDL for LLM-driven query generation. Our demonstration domain also requires local LLMs on modest institutional hardware to address privacy concerns for human subject data.",
      "published": "2026-07-20T14:59:33Z",
      "abstract_url": "http://arxiv.org/abs/2607.18029v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18029v1",
      "categories": [
        "cs.DB",
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking Heterogeneous LLM Merging: A Weighted Model Averaging Perspective",
      "authors": [
        "Jiahe Fan",
        "Yinghao Hou",
        "Si Chen",
        "Aiyuan Zhang",
        "Hong Xie",
        "Defu Lian"
      ],
      "abstract": "Can large language models with substantially different parameter spaces be merged by direct weighted averaging, without training or semantic alignment? Existing heterogeneous fusion methods typically introduce distillation, adapters, learned latent spaces, routing, or feature alignment, leaving open whether a simpler recipe can work for genuinely different billion-parameter checkpoints. We revisit this counterintuitive question through training-free dimensional adaptation followed by ratio-controlled interpolation. In union-style merging, we expand the smaller model into the larger parameter space; in intersection-style merging, we truncate the larger model into the smaller parameter space. Across Qwen-family model pairs and benchmarks covering mathematical reasoning, code generation, language understanding, commonsense reasoning, knowledge, and instruction following, deterministic expansion largely preserves the source model function, and small-ratio interpolation can improve over strong source checkpoints by transferring complementary capabilities. However, near-balanced interpolation often collapses, and task-level results reveal a seesaw effect in which gains on some capabilities coexist with regressions on others. These results show that simple parameter averaging, when paired with lightweight dimensional adaptation and carefully controlled ratios, is a surprisingly strong baseline for heterogeneous LLM merging, suggesting that the limits of direct weighted fusion may also bound what more complex heterogeneous merging methods can achieve at scale.",
      "published": "2026-07-20T14:58:53Z",
      "abstract_url": "http://arxiv.org/abs/2607.18026v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18026v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models",
      "authors": [
        "Martino M. L. Pulici",
        "Cuong Xuan Chu",
        "Evgeny Kharlamov",
        "Zifeng Ding",
        "Volker Tresp",
        "Yunpu Ma"
      ],
      "abstract": "Large language models achieve strong reasoning performance, but often at prohibitive training cost - a challenge that is especially acute for compact models ($\\leq 4 \\, \\mathrm{B}$ parameters) trained under limited budgets. We introduce MADA-RL, a post-training framework that specializes compact models into generator and critic roles and trains them with a debate-aware learning signal, fine-tuning only a small subset of parameters via LoRA adapters. Our central contribution is a counterfactual critic advantage: a dynamic, role-conditioned baseline that redefines the critic's advantage as its reward minus the generator ensemble's per-instance accuracy. This explicitly optimizes critics to improve over generator consensus rather than to merely reproduce a correct answer, yielding more targeted credit assignment than static mean-reward normalization. At deployment, the specialized agents are composed in a lightweight multi-round protocol. Across five mathematical reasoning benchmarks, MADA-RL raises the accuracy of the DeepSeek-R1-Distill-Qwen-1.5B model from $39.9 \\, \\%$ to $41.9 \\, \\%$ ($+2.0$ points, $p < 0.001$) using $16$ times fewer trainable parameters than fully fine-tuned baselines, placing it on the accuracy-trainable-parameter Pareto front. It approaches, but does not surpass, the strongest baselines (DeepScaleR, STILL-3), which are trained on substantially larger datasets; we analyse this gap and the associated inference-time cost directly. A controlled study isolates the source of MADA-RL's gains: the counterfactual advantage produces the highest critic improvement rate of any model evaluated, indicating that trained critics learn to correct generator errors rather than to imitate them.",
      "published": "2026-07-20T14:38:00Z",
      "abstract_url": "http://arxiv.org/abs/2607.18006v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18006v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "cs.MA"
      ]
    },
    {
      "title": "PAMD: Structured Adaptive Distances for Bisimulation Representations in Visual Reinforcement Learning",
      "authors": [
        "Daegyeong Roh",
        "Juho Bae",
        "Han-Lim Choi"
      ],
      "abstract": "Many visual reinforcement learning (RL) algorithms learn representations by matching latent distances to a behavioral distance induced by reward and transition similarity. In practice, the choice of the latent distance can strongly affect performance: using a fixed, pre-specified global norms (e.g., $\\ell_p$ norms or other hand-designed metrics) may be overly restrictive to capture the behavioral distance. In contrast, unconstrained pairwise distances may admit degenerate solutions that drive the metric loss down without improving the representation. To address this gap, we introduce **PAMD: Pairwise Adaptive Mahalanobis Distance**, which parameterizes a positive-definite, pair-conditioned metric for measuring latent state similarity. PAMD is a simple plug-in for existing bisimulation-based methods, offering a more expressive yet structured alternative to fixed, pre-specified latent distances. We empirically validate our method on visual MuJoCo continuous-control tasks, where final performance of several recent bisimulation-based RL algorithms is substantially improved when equipped with the distance we propose.",
      "published": "2026-07-20T14:36:11Z",
      "abstract_url": "http://arxiv.org/abs/2607.18004v1",
      "pdf_url": "https://arxiv.org/pdf/2607.18004v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
