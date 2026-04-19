const PAPERS_DATA = {
  "last_updated": "2026-04-19 03:38:21 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Generalization in LLM Problem Solving: The Case of the Shortest Path",
      "authors": [
        "Yao Tong",
        "Jiayuan Ye",
        "Anastasia Borovykh",
        "Reza Shokri"
      ],
      "abstract": "Whether language models can systematically generalize remains actively debated. Yet empirical performance is jointly shaped by multiple factors such as training data, training paradigms, and inference-time strategies, making failures difficult to interpret. We introduce a controlled synthetic environment based on shortest-path planning, a canonical composable sequential optimization problem. The setup enables clean separation of these factors and supports two orthogonal axes of generalization: spatial transfer to unseen maps and length scaling to longer-horizon problems. We find that models exhibit strong spatial transfer but consistently fail under length scaling due to recursive instability. We further analyze how distinct stages of the learning pipeline influence systematic problem-solving: for example, data coverage sets capability limits; reinforcement learning improves training stability but does not expand those limits; and inference-time scaling enhances performance but cannot rescue length-scaling failures.",
      "published": "2026-04-16T17:59:43Z",
      "abstract_url": "http://arxiv.org/abs/2604.15306v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15306v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations",
      "authors": [
        "Manan Gupta",
        "Dhruv Kumar"
      ],
      "abstract": "LLM-as-judge frameworks are increasingly used for automatic NLG evaluation, yet their per-instance reliability remains poorly understood. We present a two-pronged diagnostic toolkit applied to SummEval: $\\textbf{(1)}$ a transitivity analysis that reveals widespread per-input inconsistency masked by low aggregate violation rates ($\\barρ = 0.8$-$4.1\\%$), with $33$-$67\\%$ of documents exhibiting at least one directed 3-cycle; and $\\textbf{(2)}$ split conformal prediction sets over 1-5 Likert scores providing theoretically-guaranteed $\\geq(1{-}α)$ coverage, with set width serving as a per-instance reliability indicator ($r_s = {+}0.576$, $N{=}1{,}918$, $p < 10^{-100}$, pooled across all judges). Critically, prediction set width shows consistent cross-judge agreement ($\\bar{r} = 0.32$-$0.38$), demonstrating it captures document-level difficulty rather than judge-specific noise. Across four judges and four criteria, both diagnostics converge: criterion matters more than judge, with relevance judged most reliably (avg. set size $\\approx 3.0$) and coherence moderately so (avg. set size $\\approx 3.9$), while fluency and consistency remain unreliable (avg. set size $\\approx 4.9$). We release all code, prompts, and cached results.",
      "published": "2026-04-16T17:58:21Z",
      "abstract_url": "http://arxiv.org/abs/2604.15302v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15302v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Prism: Symbolic Superoptimization of Tensor Programs",
      "authors": [
        "Mengdi Wu",
        "Xiaoyu Jiang",
        "Oded Padon",
        "Zhihao Jia"
      ],
      "abstract": "This paper presents Prism, the first symbolic superoptimizer for tensor programs. The key idea is sGraph, a symbolic, hierarchical representation that compactly encodes large classes of tensor programs by symbolically representing some execution parameters. Prism organizes optimization as a two-level search: it constructs symbolic graphs that represent families of programs, and then instantiates them into concrete implementations. This formulation enables structured pruning of provably suboptimal regions of the search space using symbolic reasoning over operator semantics, algebraic identities, and hardware constraints. We develop techniques for efficient symbolic graph generation, equivalence verification via e-graph rewriting, and parameter instantiation through auto-tuning. Together, these components allow Prism to bridge the rigor of exhaustive search with the scalability required for modern ML workloads. Evaluation on five commonly used LLM workloads shows that Prism achieves up to $2.2\\times$ speedup over best superoptimizers and $4.9\\times$ over best compiler-based approaches, while reducing end-to-end optimization time by up to $3.4\\times$.",
      "published": "2026-04-16T17:43:31Z",
      "abstract_url": "http://arxiv.org/abs/2604.15272v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15272v1",
      "categories": [
        "cs.PL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "SegWithU: Uncertainty as Perturbation Energy for Single-Forward-Pass Risk-Aware Medical Image Segmentation",
      "authors": [
        "Tianhao Fu",
        "Austin Wang",
        "Charles Chen",
        "Roby Aldave-Garza",
        "Yucheng Chen"
      ],
      "abstract": "Reliable uncertainty estimation is critical for medical image segmentation, where automated contours feed downstream quantification and clinical decision support. Many strong uncertainty methods require repeated inference, while efficient single-forward-pass alternatives often provide weaker failure ranking or rely on restrictive feature-space assumptions. We present $\\textbf{SegWithU}$, a post-hoc framework that augments a frozen pretrained segmentation backbone with a lightweight uncertainty head. SegWithU taps intermediate backbone features and models uncertainty as perturbation energy in a compact probe space using rank-1 posterior probes. It produces two voxel-wise uncertainty maps: a calibration-oriented map for probability tempering and a ranking-oriented map for error detection and selective prediction. Across ACDC, BraTS2024, and LiTS, SegWithU is the strongest and most consistent single-forward-pass baseline, achieving AUROC/AURC of $0.9838/2.4885$, $0.9946/0.2660$, and $0.9925/0.8193$, respectively, while preserving segmentation quality. These results suggest that perturbation-based uncertainty modeling is an effective and practical route to reliability-aware medical segmentation. Source code is available at https://github.com/ProjectNeura/SegWithU.",
      "published": "2026-04-16T17:42:42Z",
      "abstract_url": "http://arxiv.org/abs/2604.15271v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15271v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Stability and Generalization in Looped Transformers",
      "authors": [
        "Asher Labovich"
      ],
      "abstract": "Looped transformers promise test-time compute scaling by spending more iterations on harder problems, but it remains unclear which architectural choices let them extrapolate to harder problems at test time rather than memorize training-specific solutions. We introduce a fixed-point based framework for analyzing looped architectures along three axes of stability -- reachability, input-dependence, and geometry -- and use it to characterize when fixed-point iteration yields meaningful predictions. Theoretically, we prove that looped networks without recall have countable fixed points and cannot achieve strong input-dependence at any spectral regime, while recall combined with outer normalization reliably produces a regime in which fixed points are simultaneously reachable, locally smooth in the input, and supported by stable backpropagation. Empirically, we train single-layer looped transformers on chess, sudoku, and prefix-sums and find that downstream performance tracks the framework's predictions across tasks and architectural configurations. We additionally introduce internal recall, a novel recall placement variant, and show that it becomes competitive with -- and on sudoku, substantially better than -- standard recall placement once outer normalization is applied.",
      "published": "2026-04-16T17:35:49Z",
      "abstract_url": "http://arxiv.org/abs/2604.15259v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15259v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Context Over Content: Exposing Evaluation Faking in Automated Judges",
      "authors": [
        "Manan Gupta",
        "Inderjeet Nair",
        "Lu Wang",
        "Dhruv Kumar"
      ],
      "abstract": "The $\\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $ΔV = -9.8 pp$ (a $30\\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking.",
      "published": "2026-04-16T16:55:53Z",
      "abstract_url": "http://arxiv.org/abs/2604.15224v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15224v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines",
      "authors": [
        "Marcel Wagenländer",
        "Otto White",
        "Britannio Jarrett",
        "Pedro Silvestre",
        "Yanda Tao",
        "Guo Li",
        "Huanzhou Zhu",
        "Llúis Vilanova",
        "Peter Pietzuch"
      ],
      "abstract": "Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription. We describe Scepsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. Scepsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, Scepsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that Scepsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations.",
      "published": "2026-04-16T16:15:29Z",
      "abstract_url": "http://arxiv.org/abs/2604.15186v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15186v1",
      "categories": [
        "cs.DC",
        "cs.AI"
      ]
    },
    {
      "title": "MambaSL: Exploring Single-Layer Mamba for Time Series Classification",
      "authors": [
        "Yoo-Min Jung",
        "Leekyung Kim"
      ],
      "abstract": "Despite recent advances in state space models (SSMs) such as Mamba across various sequence domains, research on their standalone capacity for time series classification (TSC) has remained limited. We propose MambaSL, a framework that minimally redesigns the selective SSM and projection layers of a single-layer Mamba, guided by four TSC-specific hypotheses. To address benchmarking limitations -- restricted configurations, partial University of East Anglia (UEA) dataset coverage, and insufficiently reproducible setups -- we re-evaluate 20 strong baselines across all 30 UEA datasets under a unified protocol. As a result, MambaSL achieves state-of-the-art performance with statistically significant average improvements, while ensuring reproducibility via public checkpoints for all evaluated models. Together with visualizations, these results demonstrate the potential of Mamba-based architectures as a TSC backbone.",
      "published": "2026-04-16T15:51:16Z",
      "abstract_url": "http://arxiv.org/abs/2604.15174v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15174v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Class Unlearning via Depth-Aware Removal of Forget-Specific Directions",
      "authors": [
        "Arman Hatami",
        "Romina Aalishah",
        "Ilya E. Monosov"
      ],
      "abstract": "Machine unlearning aims to remove targeted knowledge from a trained model without the cost of retraining from scratch. In class unlearning, however, reducing accuracy on forget classes does not necessarily imply true forgetting: forgotten information can remain encoded in internal representations, and apparent forgetting may arise from classifier-head suppression rather than representational removal. We show that existing class-unlearning methods often exhibit weak or negative selectivity, preserve forget-class structure in deep representations, or rely heavily on final-layer bias shifts. We then introduce DAMP (Depth-Aware Modulation by Projection), a one-shot, closed-form weight-surgery method that removes forget-specific directions from a pretrained network without gradient-based optimization. At each stage, DAMP computes class prototypes in the input space of the next learnable operator, extracts forget directions as residuals relative to retain-class prototypes, and applies a projection-based update to reduce downstream sensitivity to those directions. To preserve utility, DAMP uses a parameter-free depth-aware scaling rule derived from probe separability, applying smaller edits in early layers and larger edits in deeper layers. The method naturally extends to multi-class forgetting through low-rank subspace removal. Across MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet, and across convolutional and transformer architectures, DAMP more closely resembles the retraining gold standard than some of the prior methods, improving selective forgetting while better preserving retain-class performance and reducing residual forget-class structure in deep layers.",
      "published": "2026-04-16T15:46:02Z",
      "abstract_url": "http://arxiv.org/abs/2604.15166v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15166v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models",
      "authors": [
        "Zihao Xu",
        "John Harvill",
        "Ziwei Fan",
        "Yizhou Sun",
        "Hao Ding",
        "Hao Wang"
      ],
      "abstract": "Large Language Models (LLMs) incur significant computational and memory costs when processing long prompts, as full self-attention scales quadratically with input length. Token compression aims to address this challenge by reducing the number of tokens representing inputs. However, existing prompt-compression approaches primarily operate in token space and overlook inefficiencies in the latent embedding space. In this paper, we propose K-Token Merging, a latent-space compression framework that merges each contiguous block of K token embeddings into a single embedding via a lightweight encoder. The compressed sequence is processed by a LoRA-adapted LLM, while generation remains in the original vocabulary. Experiments on structural reasoning (Textualized Tree), sentiment classification (Amazon Reviews), and code editing (CommitPackFT) show that K-Token Merging lies on the Pareto frontier of performance vs. compression, achieving up to 75% input length reduction with minimal performance degradation.",
      "published": "2026-04-16T15:32:45Z",
      "abstract_url": "http://arxiv.org/abs/2604.15153v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15153v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking",
      "authors": [
        "Lukas Helff",
        "Quentin Delfosse",
        "David Steinmann",
        "Ruben Härle",
        "Hikaru Shindo",
        "Patrick Schramowski",
        "Wolfgang Stammer",
        "Kristian Kersting",
        "Felix Friedrich"
      ],
      "abstract": "As reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant paradigm for scaling reasoning capabilities in LLMs, a new failure mode emerges: LLMs gaming verifiers. We study this phenomenon on inductive reasoning tasks, where models must induce and output logical rules. We find that RLVR-trained models systematically abandon rule induction. Instead of learning generalizable patterns (e.g., ``trains carrying red cars go east''), they enumerate instance-level labels, producing outputs that pass verifiers without capturing the relational patterns required by the task. We show that this behavior is not a failure of understanding but a form of reward hacking: imperfect verifiers that check only extensional correctness admit false positives. To detect such shortcuts, we introduce Isomorphic Perturbation Testing (IPT), which evaluates a single model output under both extensional and isomorphic verification, where the latter enforces invariance under logically isomorphic tasks. While genuine rule induction remains invariant, shortcut strategies fail. We find that shortcut behavior is specific to RLVR-trained reasoning models (e.g., GPT-5, Olmo3) and absent in non-RLVR models (e.g., GPT-4o, GPT-4.5, Ministral). Moreover, shortcut prevalence increases with task complexity and inference-time compute. In controlled training experiments, extensional verification directly induces shortcut strategies, while isomorphic verification eliminates them. These results show that RLVR can incentivize reward hacking not only through overt manipulation but also by exploiting what the verifier fails to enforce.",
      "published": "2026-04-16T15:30:10Z",
      "abstract_url": "http://arxiv.org/abs/2604.15149v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15149v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning",
      "authors": [
        "Zihan Liang",
        "Yufei Ma",
        "Ben Chen",
        "Zhipeng Qian",
        "Huangyu Dai",
        "Lingtao Mao",
        "Xuxin Zhang",
        "Chenyi Lei",
        "Wenwu Ou"
      ],
      "abstract": "Reinforcement learning has emerged as an effective paradigm for training large language models to perform search-augmented reasoning. However, existing approaches rely on trajectory-level rewards that cannot distinguish precise search queries from vague or redundant ones within a rollout group, and collapse to a near-zero gradient signal whenever every sampled trajectory fails. In this paper, we propose IG-Search, a reinforcement learning framework that introduces a step-level reward based on Information Gain (IG). For each search step, IG measures how much the retrieved documents improve the model's confidence in the gold answer relative to a counterfactual baseline of random documents, thereby reflecting the effectiveness of the underlying search query. This signal is fed back to the corresponding search-query tokens via per-token advantage modulation in GRPO, enabling fine-grained, step-level credit assignment within a rollout. Unlike prior step-level methods that require either externally annotated intermediate supervision or shared environment states across trajectories, IG-Search derives its signals from the policy's own generation probabilities, requiring no intermediate annotations beyond standard question-answer pairs. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate that IG-Search achieves an average EM of 0.430 with Qwen2.5-3B, outperforming the strongest trajectory-level baseline (MR-Search) by 1.6 points and the step-level method GiGPO by 0.9 points on average across benchmarks, with particularly pronounced gains on multi-hop reasoning tasks. Despite introducing a dense step-level signal, IG-Search adds only ~6.4% to per-step training wall-clock time over the trajectory-level baseline and leaves inference latency unchanged, while still providing a meaningful gradient signal even when every sampled trajectory answers incorrectly.",
      "published": "2026-04-16T15:22:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.15148v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15148v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.IR"
      ]
    },
    {
      "title": "Structure as Computation: Developmental Generation of Minimal Neural Circuits",
      "authors": [
        "Duan Zhou"
      ],
      "abstract": "This work simulates the developmental process of cortical neurogenesis, initiating from a single stem cell and governed by gene regulatory rules derived from mouse single-cell transcriptomic data. The developmental process spontaneously generates a heterogeneous population of 5,000 cells, yet yields only 85 mature neurons - merely 1.7% of the total population. These 85 neurons form a densely interconnected core of 200,400 synapses, corresponding to an average degree of 4,715 per neuron. At iteration zero, this minimal circuit performs at chance level on MNIST. However, after a single epoch of standard training, accuracy surges to over 90% - a gain exceeding 80 percentage points - with typical runs falling in the 89-94% range depending on developmental stochasticity. The identical circuit, without any architectural modification or data augmentation, achieves 40.53% on CIFAR-10 after one epoch. These findings demonstrate that developmental rules sculpt a domain-general topological substrate exceptionally amenable to rapid learning, suggesting that biological developmental processes inherently encode powerful structural priors for efficient computation.",
      "published": "2026-04-16T15:19:27Z",
      "abstract_url": "http://arxiv.org/abs/2604.15143v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15143v1",
      "categories": [
        "cs.NE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Amortized Optimal Transport from Sliced Potentials",
      "authors": [
        "Minh-Phuc Truong",
        "Khai Nguyen"
      ],
      "abstract": "We propose a novel amortized optimization method for predicting optimal transport (OT) plans across multiple pairs of measures by leveraging Kantorovich potentials derived from sliced OT. We introduce two amortization strategies: regression-based amortization (RA-OT) and objective-based amortization (OA-OT). In RA-OT, we formulate a functional regression model that treats Kantorovich potentials from the original OT problem as responses and those obtained from sliced OT as predictors, and estimate these models via least-squares methods. In OA-OT, we estimate the parameters of the functional model by optimizing the Kantorovich dual objective. In both approaches, the predicted OT plan is subsequently recovered from the estimated potentials. As amortized OT methods, both RA-OT and OA-OT enable efficient solutions to repeated OT problems across different measure pairs by reusing information learned from prior instances to rapidly approximate new solutions. Moreover, by exploiting the structure provided by sliced OT, the proposed models are more parsimonious, independent of specific structures of the measures, such as the number of atoms in the discrete case, while achieving high accuracy. We demonstrate the effectiveness of our approaches on tasks including MNIST digit transport, color transfer, supply-demand transportation on spherical data, and mini-batch OT conditional flow matching.",
      "published": "2026-04-16T15:05:59Z",
      "abstract_url": "http://arxiv.org/abs/2604.15114v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15114v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation",
      "authors": [
        "Haozhi Fan",
        "Jinhao Duan",
        "Kaidi Xu"
      ],
      "abstract": "Despite the rapid advancement of Large Language Models (LLMs), uncertainty quantification in LLM generation is a persistent challenge. Although recent approaches have achieved strong performance by restricting LLMs to produce short or constrained answer sets, many real-world applications require long-form and free-form text generation. A key difficulty in this setting is that LLMs often produce responses that are semantically coherent yet factually inaccurate, while the underlying semantics are multifaceted and the linguistic structure is complex. To tackle this challenge, this paper introduces Interrogative Uncertainty Quantification (IUQ), a novel framework that leverages inter-sample consistency and intra-sample faithfulness to quantify the uncertainty in long-form LLM outputs. By utilizing an interrogate-then-respond paradigm, our method provides reliable measures of claim-level uncertainty and the model's faithfulness. Experimental results across diverse model families and model sizes demonstrate the superior performance of IUQ over two widely used long-form generation datasets. The code is available at https://github.com/louisfanhz/IUQ.",
      "published": "2026-04-16T15:03:00Z",
      "abstract_url": "http://arxiv.org/abs/2604.15109v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15109v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Autonomous Evolution of EDA Tools: Multi-Agent Self-Evolved ABC",
      "authors": [
        "Cunxi Yu",
        "Haoxing Ren"
      ],
      "abstract": "This paper introduces the first \\emph{self-evolving} logic synthesis framework, which leverages Large Language Model (LLM) agents to autonomously improve the source code of \\textsc{ABC}, the widely adopted logic synthesis system. Our framework operates on the \\emph{entire integrated ABC codebase}, and the output repository preserves its single-binary execution model and command interface. In the initial evolution cycle, we bootstrap the system using existing prior open-source synthesis components, covering flow tuning, logic minimization, and technology mapping, but without manually injecting new heuristics. On top of this foundation, a team of LLM-based agents iteratively rewrites and evolves specific sub-components of ABC following our ``programming guidance`` prompts under a unified correctness and QoR-driven evaluation loop. Each evolution cycle proposes code modifications, compiles the integrated binary, validates correctness, and evaluates quality-of-results (QoR) on \\emph{multi-suite benchmarks including ISCAS~85/89/99, VTR, EPFL, and IWLS~2005}. Through continuous feedback, the system discovers optimizations beyond human-designed heuristics, effectively \\emph{learning new synthesis strategies} that enhance QoR. We detail the architecture of this self-improving system, its integration with \\textsc{ABC}, and results demonstrating that the framework can autonomously and progressively improve EDA tool at full million-line scale.",
      "published": "2026-04-16T14:42:55Z",
      "abstract_url": "http://arxiv.org/abs/2604.15082v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15082v1",
      "categories": [
        "cs.AR",
        "cs.AI"
      ]
    },
    {
      "title": "No More Guessing: a Verifiable Gradient Inversion Attack in Federated Learning",
      "authors": [
        "Francesco Diana",
        "Chuan Xu",
        "André Nusser",
        "Giovanni Neglia"
      ],
      "abstract": "Gradient inversion attacks threaten client privacy in federated learning by reconstructing training samples from clients' shared gradients. Gradients aggregate contributions from multiple records and existing attacks may fail to disentangle them, yielding incorrect reconstructions with no intrinsic way to certify success. In vision and language, attackers may fall back on human inspection to judge reconstruction plausibility, but this is far less feasible for numerical tabular records, fueling the impression that tabular data is less vulnerable. We challenge this perception by proposing a verifiable gradient inversion attack (VGIA) that provides an explicit certificate of correctness for reconstructed samples. Our method adopts a geometric view of ReLU leakage: the activation boundary of a fully connected layer defines a hyperplane in input space. VGIA introduces an algebraic, subspace-based verification test that detects when a hyperplane-delimited region contains exactly one record. Once isolation is certified, VGIA recovers the corresponding feature vector analytically and reconstructs the target via a lightweight optimization step. Experiments on tabular benchmarks with large batch sizes demonstrate exact record and target recovery in regimes where existing state-of-the-art attacks either fail or cannot assess reconstruction fidelity. Compared to prior geometric approaches, VGIA allocates hyperplane queries more effectively, yielding faster reconstructions with fewer attack rounds.",
      "published": "2026-04-16T14:28:19Z",
      "abstract_url": "http://arxiv.org/abs/2604.15063v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15063v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning",
      "authors": [
        "Khalid Adnan Alsayed"
      ],
      "abstract": "The evaluation of fairness in machine learning systems has become a central concern in high-stakes applications, including biometric recognition, healthcare decision-making, and automated risk assessment. Existing approaches typically rely on a small number of fairness metrics to assess model behaviour across group partitions, implicitly assuming that these metrics provide consistent and reliable conclusions. However, different fairness metrics capture distinct statistical properties of model performance and may therefore produce conflicting assessments when applied to the same system. In this work, we investigate the consistency of fairness evaluation by conducting a systematic multi-metric analysis of demographic bias in machine learning models. Using face recognition as a controlled experimental setting, we evaluate model performance across multiple group partitions under a range of commonly used fairness metrics, including error-rate disparities and performance-based measures. Our results demonstrate that fairness assessments can vary significantly depending on the choice of metrics, leading to contradictory conclusions regarding model bias. To quantify this phenomenon, we introduce the Fairness Disagreement Index (FDI), a measure designed to capture the degree of inconsistency across fairness metrics. We further show that disagreement remains high across thresholds and model configurations. These findings highlight a critical limitation in current fairness evaluation practices and suggest that single-metric reporting is insufficient for reliable bias assessment.",
      "published": "2026-04-16T14:07:37Z",
      "abstract_url": "http://arxiv.org/abs/2604.15038v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15038v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization",
      "authors": [
        "Haochun Tang",
        "Yuliang Yan",
        "Jiahua Lu",
        "Huaxiao Liu",
        "Enyan Dai"
      ],
      "abstract": "Cost-aware routing dynamically dispatches user queries to models of varying capability to balance performance and inference cost. However, the routing strategy introduces a new security concern that adversaries may manipulate the router to consistently select expensive high-capability models. Existing routing attacks depend on either white-box access or heuristic prompts, rendering them ineffective in real-world black-box scenarios. In this work, we propose R$^2$A, which aims to mislead black-box LLM routers to expensive models via adversarial suffix optimization. Specifically, R$^2$A deploys a hybrid ensemble surrogate router to mimic the black-box router. A suffix optimization algorithm is further adapted for the ensemble-based surrogate. Extensive experiments on multiple open-source and commercial routing systems demonstrate that {R$^2$A} significantly increases the routing rate to expensive models on queries of different distributions. Code and examples: https://github.com/thcxiker/R2A-Attack.",
      "published": "2026-04-16T13:51:48Z",
      "abstract_url": "http://arxiv.org/abs/2604.15022v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15022v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers",
      "authors": [
        "Éric Jacopin"
      ],
      "abstract": "When do transformers commit to a decision, and what prevents them from correcting it? We introduce \\textbf{prolepsis}: a transformer commits early, task-specific attention heads sustain the commitment, and no layer corrects it. Replicating \\citeauthor{lindsey2025biology}'s (\\citeyear{lindsey2025biology}) planning-site finding on open models (Gemma~2 2B, Llama~3.2 1B), we ask five questions. (Q1)~Planning is invisible to six residual-stream methods; CLTs are necessary. (Q2)~The planning-site spike replicates with identical geometry. (Q3)~Specific attention heads route the decision to the output, filling a gap flagged as invisible to attribution graphs. (Q4)~Search requires ${\\leq}16$ layers; commitment requires more. (Q5)~Factual recall shows the same motif at a different network depth, with zero overlap between recurring planning heads and the factual top-10. Prolepsis is architectural: the template is shared, the routing substrates differ. All experiments run on a single consumer GPU (16\\,GB VRAM).",
      "published": "2026-04-16T13:38:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.15010v1",
      "pdf_url": "https://arxiv.org/pdf/2604.15010v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    }
  ]
};
