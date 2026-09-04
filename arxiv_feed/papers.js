const PAPERS_DATA = {
  "last_updated": "2026-09-04 04:09:27 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Compile by Training: Turning Natural-Language Specifications into Local Neural Functions",
      "authors": [
        "Yuntian Deng",
        "Pengyu Nie",
        "Stuart Shieber"
      ],
      "abstract": "Many recurring text functions are easy to describe but difficult to implement with rules, while calling a large remote model for every input introduces repeated cost, latency, and dependency on a provider. We present compile by training, which turns a natural-language specification into a reusable neural function. At compile time, teacher models generate task-specific examples that are used to train a small adapter for a compact interpreter. The resulting function runs without the teachers and can be stored, versioned, and composed like ordinary software. On FuzzyBench-Hard, a subset on which the Program-as-Weights fast compiler produced no exact matches, compile by training reaches 83.6% semantic accuracy. This higher accuracy comes with a higher compile-time cost: roughly a minute rather than seconds for the fast compiler. We deploy the compiler in a public interactive service and demonstrate compiled functions in a multi-site website helper, a language-controlled 3D avatar, and a bidirectional English-Claudish translator.",
      "published": "2026-09-03T17:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2609.04199v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04199v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints",
      "authors": [
        "Haoyaun Zhu",
        "Jie Zhang"
      ],
      "abstract": "Language-model judges now gate training data, score generations, and drive leaderboards. The judge is then a measurement instrument, resting on one rarely stated assumption: the same request, sent to the same model name, reads the same tomorrow. We audited that assumption in two preregistered campaigns with every threshold fixed in advance; neither got past validating its instrument. Across 52,988 audited request attempts, same-window repeat rankings agreed at Spearman 0.400 against a required 0.90, and byte-identical next-day replays agreed at 0.78 against a required 0.99, each time with the execution record at ceiling. Three mechanisms explain the gap: a label-to-meaning mapping that biased readouts as strongly as the signal; candidate gaps seven orders of magnitude below the instrument's own noise floor; and byte-identical inputs returning different rankings, a noise that exact-permutation readouts compound. Neither metric substitution nor sampling repaired it on the tested grid. Preregistered follow-ups bound the problem: waiting did not help on the days sampled (0.805 versus 0.800, replicated over five further days); switching providers did not help (four providers share the floor, medians 0.74 to 0.88, predicted by none of the metadata fields they expose); self-hosting on batch-invariant kernels helped only while the server was quiet; and on constructed errors with known gaps, the readout's separation tracks error type, not size. We distill the evidence into a three-level snapshot-identity ladder, eight design rules, and a reporting checklist; a pilot at roughly 2% of the study's call volume would have exposed both unreachable gates in advance. All results concern externally measured behaviour on shared serving infrastructure. On a shared endpoint, a model name is not a frozen instrument; a preregistered evaluation must measure its instrument before freezing any gate on it.",
      "published": "2026-09-03T17:59:43Z",
      "abstract_url": "http://arxiv.org/abs/2609.04198v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04198v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views",
      "authors": [
        "Joseph Lee",
        "Yidi Huang",
        "Dokyoon Kim",
        "Shu Yang",
        "Li Shen"
      ],
      "abstract": "Gaps remain in our understanding of how large language models (LLMs) acquire knowledge during pre-training. We posit that auxiliary views, reformulations of knowledge, are causally helpful for learning. We design controlled experiments to isolate this. First, we confirm that repetition is necessary for acquisition and clarify that paraphrasing helps only at smaller batch sizes. Second, holding the token budget fixed, allocating tokens from document repetition to auxiliary views improves learning, counterintuitively, even for factual recall. Third, the effectiveness of auxiliary views is not contingent on the strength of the teacher model that generates them. Fourth, we identify forms of knowledge, contextual and foundational, that aid learning in the presence of prior knowledge gaps. Finally, we examine how these effects manifest mechanistically via layer-wise biases and compression. Together, our findings suggest that auxiliary representations of knowledge, which arise naturally in large pre-training corpora, are a key factor in the success of pre-training and offer a plausible explanation for why data diversity matters.",
      "published": "2026-09-03T17:57:02Z",
      "abstract_url": "http://arxiv.org/abs/2609.04180v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04180v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Computationally Feasible Framework for Causal Probabilistic Explanation",
      "authors": [
        "Rafal Urbaniak",
        "Sam Witty",
        "Daniel Waxman",
        "Andy Zane",
        "Poorva Garg",
        "Emily Bunnapradist",
        "Sankaran Vaidyanathan",
        "Jack Feser",
        "Drew Lehe",
        "Eli Bingham"
      ],
      "abstract": "Explaining why a specific outcome occurred, and which inputs deserve the blame or credit, is central to philosophical, scientific, and policy analysis. Existing tools split into two camps. The theory of actual causality (AC) gives principled verdicts, but only for toy-sized models, because computing them requires enumerating counterfactual scenarios. Scalable attribution methods like SHAP (or even causal SHAP) at least partially ignore the causal structure that generated the data, and can give answers that conflict with a careful causal analysis. We close this gap with Probabilistic Causal Impact (PCI). PCI builds on actual causality and on Pearl's notions of probability of necessity and sufficiency, but recasts the question of explainability as an estimation problem on a probabilistic causal model that is easily approximated via Monte Carlo. By specifying a distribution over \"candidate explanations,\" a distribution over counterfactual values, and a scoring function, PCI provides tractable, causally grounded, graded explanations, generalizing AC and Pearl's probability of causation as degenerate cases. We evaluate PCI in synthetic and real-world examples, spanning consistency checks with AC, scaling experiments, complex continuous-valued dynamical systems, and a real-world deployed causal machine learning model trained on millions of datapoints.",
      "published": "2026-09-03T17:55:43Z",
      "abstract_url": "http://arxiv.org/abs/2609.04177v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04177v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking On-Policy Distillation of Large Language Models II: One Training Example",
      "authors": [
        "Zixuan Fu",
        "Bingxiang He",
        "Yuxin Zuo",
        "Haohuan Huang",
        "Jinqian Zhang",
        "Ruhang Xiao",
        "Cheng Qian",
        "Qinyu Luo",
        "Huan-ang Gao",
        "Yudong Wang",
        "Zhiyuan Liu",
        "Ning Ding",
        "Chaojun Xiao"
      ],
      "abstract": "On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher. Existing work has mainly studied its algorithmic behavior, leaving the role of training data unclear. We examine this role at the data-minimal limit by training on a single query. One-shot OPD keeps improving for hundreds of steps and recovers most of full-data OPD's gain across task domains and model families. We explain this result through the states visited during training and the rate at which the student aligns with the teacher. We measure \\emph{state coverage}, the fraction of the states full-data OPD visits that a query set's rollouts reach. A single query already reaches \\(71.5\\%\\), most of it within the first 100 steps. Adding semantically distinct queries raises coverage and validation accuracy together, until 16 queries reach \\(98.9\\%\\) and match full-data training. Yet alignment slows at a similar pace whether OPD trains on one query or the whole dataset, and even a fixed set of states takes hundreds of steps to absorb. OPD is therefore data-overfed but algorithm-starved. Its rollouts quickly expose broad supervision, while the student absorbs that supervision increasingly slowly. The state-coverage result extends to multi-teacher OPD, where 16 semantically diverse queries per domain match full-data MOPD. As a further stress test, content-light templates and off-domain WildChat queries also approach the real-query baseline. Task content and induced state coverage can therefore come apart. We hope these findings direct future work toward the step efficiency of OPD, and prompt a re-examination of the data and the mechanisms behind its recent successes in frontier post-training.",
      "published": "2026-09-03T17:54:38Z",
      "abstract_url": "http://arxiv.org/abs/2609.04172v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04172v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "SENTINEL-RL: Offloading Topological Reasoning from LLM Agents in the Security Operations Center",
      "authors": [
        "Uday Vallabhaneni",
        "Cassie L. Cagwin",
        "David J. Wild"
      ],
      "abstract": "Large language model (LLM) agents are increasingly proposed as autonomous SOC analysts, but two limitations make them unreliable at enterprise scale: a finite context window cannot hold a multi-thousand-host authentication graph, and free-form generation offers no guarantee that a recommended containment action is consistent with the topology it operates on. We present Sentinel-RL, an agentic-SOC architecture that decouples topological reasoning from semantic reasoning: a heterogeneous graph attention encoder summarizes the live authentication subgraph into a fixed-dimensional state, a Proximal Policy Optimization (PPO) policy maps this state to a constrained set of investigative actions, and an LLM agent loop is restricted to consuming the policy's recommendations and producing analyst-readable narratives gated by a critic. We instantiate the system on the LANL Comprehensive, Multi-Source Cyber-Security Events dataset and the Indiana University Quartz HPC cluster, reporting four results: (i) a two-phase CREATE ingestion pattern loads a 24M-edge authentication subgraph into Neo4j in 14.2 minutes on a single 32-core node, roughly 24x faster than the canonical MERGE-based pipeline; (ii) a sliding-window alert engine reliably trips a 25-event/10-second threshold in <=2.5 s across 50 trials; (iii) PPO training over 200 iterations converges to a mean episodic return of 8.74+/-0.31, with held-out precision of 0.91 and recall of 0.87 on labeled red-team events; and (iv) the integrated containment loop completes a full detect-investigate-recommend-human-approve cycle in a median of 6.3 s. We contribute a reusable engineering pattern (the hot-node deadlock workaround), a portable HPC deployment pattern (anchor-node co-location), and an enterprise-readiness analysis covering false-positive economics, reversibility guarantees, audit compliance, and the human-approval boundary.",
      "published": "2026-09-03T17:49:12Z",
      "abstract_url": "http://arxiv.org/abs/2609.04159v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04159v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "A Low-Cost, Open Platform for End-to-End Autonomous Driving on a Miniature Ackermann Vehicle",
      "authors": [
        "Gustavo Claudio Karl Couto",
        "Eric Aislan Antonelo",
        "Gabriel George Zipperer"
      ],
      "abstract": "This paper presents a low-cost, open experimental platform for research in end-to-end autonomous driving with miniature Ackermann vehicles. The platform combines a physical vehicle, a printed urban track, data collection tools, trajectory registration, and a Webots digital twin, enabling controlled experiments that connect simulation-based autonomous-driving methods to real-world execution. As a first baseline, we implement command-conditioned behavior cloning, in which a neural policy receives an on-board camera image and a high-level navigation command and outputs steering and speed. The system is evaluated both on the physical vehicle and in simulation. In real closed-loop experiments, the learned policy follows lanes and executes commanded turns, reaching a mean cross-track error of 6.1 cm with respect to the reference route, close to the 4.7 cm observed in human demonstrations. In the digital twin, camera field of view has a strong effect on performance, reducing the mean cross-track error from 35.6 to 3.3 cm when widened from 58 to 120 degrees. Using the digital twin to generate synthetic driving data and a learned sim-to-real image translator to reduce the appearance gap, we further show that a higher-capacity policy trained on this synthetic data combined with real demonstrations is the only configuration that completes all four track routes in closed loop, whereas the compact baseline and the same network trained on real data alone complete fewer. These results establish the open platform as a practical testbed for sim-to-real studies and provide an initial command-conditioned imitation-learning baseline; we release it to support reproducible research.",
      "published": "2026-09-03T17:40:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.04147v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04147v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "Epistemic Warrant for LLM Recommendations: Characterizing the Basis for Reliance When Ground Truth Is Unavailable",
      "authors": [
        "Shai Vardi",
        "João Sedoc"
      ],
      "abstract": "Large language models are increasingly used to support organizational decisions, yet users often lack a principled basis for assessing whether to rely on a specific recommendation. Existing approaches typically evaluate broad model properties, such as reliability, uncertainty, or robustness, or focus on user trust, rather than the underlying basis for relying on an individual recommendation. Adapting theoretical foundations from epistemology, we introduce epistemic warrant, a decision-level construct that characterizes the stability of a model's preference and the scope over which that preference holds. We operationalize this construct through a four-tier reliance certificate for pairwise recommendations, distinguishing among unstable, context-dependent, locally supported, and broadly supported recommendations. We validate the construct using contemporary methodologies: known-groups tests successfully recover expert-prespecified warrant orderings, and stronger warrants systematically align with independent consensus from crowd workers. Furthermore, we demonstrate that epistemic warrant provides information distinct from verbalized confidence and is not readily explained by decision difficulty. Ultimately, this framework offers a theoretically grounded, implementable approach for characterizing the warrant of individual LLM recommendations when objective ground truth is unavailable.",
      "published": "2026-09-03T17:25:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.04127v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04127v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR",
      "authors": [
        "Boyan Li",
        "Bingsen Chen",
        "Chenghao Yang",
        "Ping Nie",
        "Chen Zhao",
        "Xi Ye"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \\emph{weighted-additive combination} or a \\emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To provide a practical recipe, we find that the OPD validation score is the key signal for when to switch to RL, and that OPD is a better cold start for RL than SFT. Together, our results establish OPD-then-RL as a simple yet strong way to combine the two methods, turning two entangled signals into complementary stages.",
      "published": "2026-09-03T17:14:27Z",
      "abstract_url": "http://arxiv.org/abs/2609.04108v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04108v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training",
      "authors": [
        "Shubham Gandhi",
        "Saurabh Goyal",
        "Kiran Kate",
        "Yara Rizk"
      ],
      "abstract": "Reinforcement Learning from Verifiable Rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-criteria rubrics are a popular way to supply such a reward; they are scored once per trajectory, but a single scalar is a poor signal across tens of steps. We propose DRACO: Distributing Rubric-based Advantage for Credit Optimization. It generates rubrics dynamically during training to track the policy's evolving capability, scores those rubrics once per completed trajectory, and redistributes that judgment over the steps responsible for annotated rubrics to produce differentiated per-step advantages in GRPO. The redistribution is closed-form and does not introduce any trained attribution module. On AppWorld, DRACO gains 15.9 points over the base model and 5.3 points over GRPO trained with a sparse ground-truth reward, despite not using any verifiers itself. On out-of-domain Tau-Bench, it gains 5.3 points over the base model even without a frontier judge, beating both ground-truth-reward training and other rubric-based training settings. The code for DRACO is available at https://github.com/IBM/draco.",
      "published": "2026-09-03T17:02:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.04094v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04094v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "Subspace Inference Enables Efficient Active Reward Learning from Preferences",
      "authors": [
        "Yutai Zhou",
        "Erdem Bıyık"
      ],
      "abstract": "Reinforcement learning from human feedback (RLHF) has emerged as a powerful yet sample-inefficient approach for learning reward models from human preferences, making active learning a critical component in synthesizing informative preference queries. However, effective uncertainty quantification required for active learning remains a key challenge for large neural network reward models. In this paper, we introduce PreferenceEKF, a sample-efficient approach that tracks reward model uncertainty by framing active preference learning as a sequential Bayesian filtering problem. Instead of relying on computationally prohibitive posterior inference over the full neural network parameter space, our method performs sequential inference via an extended Kalman filter within a low-dimensional parameter subspace, continuously updating the reward model posterior as new preference queries arrive. Our approach enables scalable sampling of neural network parameters to efficiently compute acquisition functions for active reward learning. Experiments on the D4RL and V-D4RL benchmarks demonstrate that our approach achieves better sample efficiency, runtime, scalability, and calibration compared to other Bayesian deep learning approaches, and the learned reward models lead to competitive offline reinforcement learning policy performance. This highlights the potential of scalable Bayesian methods for preference-based reward modeling in RLHF. Our code is available at https://github.com/yutaizhou/bnn_pref.",
      "published": "2026-09-03T16:39:55Z",
      "abstract_url": "http://arxiv.org/abs/2609.04066v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04066v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "When Models Edit Too Much: On the Fidelity of Minimal Code Edits",
      "authors": [
        "Tongyao Zhu",
        "Wei Hern Lim",
        "Min-Yen Kan"
      ],
      "abstract": "Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning budget or larger models. We next ask whether minimal editing can be learned directly during post-training. We observe that supervised fine-tuning overfits to seen corruption patterns, whereas reinforcement learning gives the best out-of-domain edit-fidelity and performance-retention trade-off. These results position edit fidelity as a distinct axis of code-repair quality and show that it can be measured and learned.",
      "published": "2026-09-03T16:36:05Z",
      "abstract_url": "http://arxiv.org/abs/2609.04061v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04061v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "IRWOZ 2.0: A Large Language Model-driven Dialogue Dataset for Industrial Robot Conversations",
      "authors": [
        "Chen Li",
        "Dimitrios Chrysostomou"
      ],
      "abstract": "IRWOZ has improved industrial human-robot interaction (HRI) dialogue systems through domain-specific annotations. However, its initial version contains substantial noise in dialogue states and utterances, limiting state-tracking accuracy. We introduce IRWOZ 2.0, which addresses these limitations through large language model (LLM) enhanced generation (Mistral/Claude-3.5) and quality refinements. Our improved dataset expands to 390 dialogues across 4 industrial domains (Assembly, Delivery, Position, Relocation), featuring manual corrections and automated typo removal. Benchmark experiments on dialogue state tracking demonstrate significant improvements, with GPT-2's BLEU-4 score increasing from 0.1651 to 0.5604 compared to original IRWOZ. To support industrial HRI research, we publicly released IRWOZ 2.0 dataset at https://ieee-dataport.org/documents/irwoz-20-large-language-model-driven-dialogue-dataset-industrial-robot-conversations",
      "published": "2026-09-03T16:08:57Z",
      "abstract_url": "http://arxiv.org/abs/2609.04030v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04030v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach",
      "authors": [
        "Giacomo Rizzieri",
        "Saif-Ur-Rehman",
        "Jörg F. Unger",
        "Annika Robens-Radermacher"
      ],
      "abstract": "The geometric morphology of deposited filaments can significantly influence the structural performance and stability of 3D concrete-printed (3DCP) structures. However, most finite element (FEM)-based approaches for buildability assessment represent printed layers as simplified rectangles, potentially limiting predictive accuracy. This study proposes a geometry-informed modelling framework that integrates the deep-learning-based filament shape prediction tool ShapeGen3DCP with a layer-activation FEM approach to investigate the effect of realistic filament geometries on buildability. The framework generates geometry-aware numerical models directly from material and process parameters, eliminating the need for experimental filament characterization or computationally intensive fluid-flow simulations. Validation against experimental data and a parametric study of rectilinear walls demonstrate that extrusion parameters and the resulting filament geometry can significantly influence buildability predictions. Realistic filament representations are particularly important for free-flow deposition, whereas layer-pressing strategies are less sensitive to geometric simplifications. Among the investigated representations, an elliptical approximation provides an effective balance between geometric fidelity and modelling simplicity. When rectangular representations are preferred to enable regular computational meshes for faster simulations, defining their dimensions based on volume conservation improves prediction reliability compared with calibrating them using either the maximum filament width or the interlayer contact width. Overall, the proposed methodology demonstrates the importance of incorporating filament geometry into 3DCP simulations and provides practical guidance for selecting efficient and accurate geometric representations for buildability assessment.",
      "published": "2026-09-03T16:08:19Z",
      "abstract_url": "http://arxiv.org/abs/2609.04028v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04028v1",
      "categories": [
        "cs.CE",
        "cs.AI",
        "cs.LG",
        "math.NA"
      ]
    },
    {
      "title": "Representational alignment yields generalizable safety in language models",
      "authors": [
        "Lingyu Li",
        "Yan Teng",
        "Yingchun Wang",
        "Xia Hu"
      ],
      "abstract": "Aligning large language models (LLMs) is essential for their safe deployment. Current alignment methods mainly optimize observable responses, yet models remain vulnerable when the same harmful intent is recast in unfamiliar or adversarial forms that humans can easily recognize. Prototype theory offers an account of this adaptability. Human concepts are represented around central cases, and new instances are categorized according to their graded typicality relative to these prototypes. Here we show that such categorization of moral concepts is weakly preserved in current LLMs. Across 23 LLMs, models often failed to distinguish opposed moral categories or preserve fine-grained typicality within each category. These deficits persist across parameter sizes and alignment stages. We developed representational similarity optimization, which directly aligns the latent representations in LLMs with the categorization expressed in human moral judgements, without supervising generated responses. In matched experiments using the same 251,334 moral annotations, standard behavioral alignment learned the intended moral judgements at the response level while leaving the categorization structure largely unchanged and increasing vulnerability across adversarial evaluations. Reorganizing moral categorization produced more modest gains in explicit judgements but consistently improved adversarial robustness across model scales on diverse benchmarks and attack strategies. Our findings provide functional support for the view that prototype-based categorization contributes to behavioral adaptability. They also show that transferring this representational principle to LLMs yields generalizable safety under adversarial conditions.",
      "published": "2026-09-03T16:00:16Z",
      "abstract_url": "http://arxiv.org/abs/2609.04022v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04022v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models",
      "authors": [
        "Yalun Wu",
        "Junfeng Fang",
        "Jiawei Wang",
        "Haotian Liu",
        "Qijun Yang",
        "Minghan Yang",
        "Hongcheng Guo",
        "Zhoujun Li",
        "Boyang Wang"
      ],
      "abstract": "Evaluating large language models (LLMs) in safety-critical, physics-governed environments requires more than accuracy-based metrics, because predictions that are numerically close to the ground truth can still violate operational constraints, combine fields in physically inconsistent ways, or fail to produce usable structured outputs. Existing evaluation protocols do not measure these failure modes reliably. We propose FLY-EVAL++, an evidence-driven evaluation protocol that combines deterministic verification of protocol compliance, physical feasibility, and safety constraints with fixed rubric-guided aggregation into interpretable multi-dimensional scores. We instantiate FLY-EVAL++ for Flight Trajectory and Attitude Prediction (FTAP) by extending the PilotBench setting with history-conditioned and multi-step prediction tasks. Across 66 LLMs, safety compliance is the most discriminative dimension of model behavior: models with comparable predictive performance differ by more than 28 points in safety score, and we observe recurrent failures including safety violations under physically plausible predictions and instability in multi-step rollouts. These results show that evaluation in safety-critical domains should measure constraint satisfaction and structured validity explicitly rather than rely on accuracy-centric reporting alone.",
      "published": "2026-09-03T16:00:02Z",
      "abstract_url": "http://arxiv.org/abs/2609.04021v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04021v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "InSituMeasure: Probing Situated Measurement Grounding in Industrial Scenes with Multimodal Large Language Models",
      "authors": [
        "Chao Shen",
        "Xinyuan Li",
        "Yunfan Zhou",
        "Jianguo Yao",
        "Haibing Guan",
        "Zhihai Wang",
        "Xijun Li"
      ],
      "abstract": "For trained operators, gauge reading requires little specialized knowledge, low cognitive effort, and high repeatability. Yet Multimodal Large Language Models (MLLMs) remain unreliable in continuous-valued measurement despite strong results on general multimodal benchmarks. Existing benchmarks expose this weakness but isolate measurement from realistic, knowledge-grounded settings, with limited situated context, specialized instruments, real-world noise, and matched diagnostic annotations, reducing realism and constraining root-cause analysis. We introduce InSituMeasure to evaluate situated measurement grounding. It contains 2,922 real industrial monitoring scenes across eight functional categories of professional engineering instruments, with dense gauge-attribute annotations and noise tags for failure diagnosis. We define metrics for numerical accuracy under predefined tolerances and unit consistency, rejection of fake or unanswerable tasks, and alignment between model failures and annotated error factors. Across 24 state-of-the-art MLLMs, the best model reaches only 25.7\\% joint value-unit accuracy and 51.8\\% confidence-diagnosis F1, revealing a substantial gap between general multimodal competence and reliable situated measurement. Further analysis identifies failures from text-induced shortcuts, overconfident responses, and authentic industrial noise, including mixed disturbances, viewpoint deviation, occlusion, and environmental interference.",
      "published": "2026-09-03T15:54:21Z",
      "abstract_url": "http://arxiv.org/abs/2609.04014v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04014v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening",
      "authors": [
        "Muhammad Ashad Kabir",
        "Sirajam Munira"
      ],
      "abstract": "Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language models (LLMs) for CKD screening under zero-shot and few-shot in-context learning settings and compares them with traditional ML and DL methods. We propose a framework that uses clinically selected tabular features and structured prompt templates to enable LLM-based inference without task-specific training. LLM performance is evaluated across multiple prompt styles, feature configurations, and data settings, and compared with standard ML, DL, and tabular foundation model (TFM) baselines, and existing CKD screening tools. The results show that LLMs can achieve competitive performance using only a small number of examples, often matching or outperforming traditional approaches in low-data settings. However, their performance remains model-dependent and less stable as input complexity increases. In contrast, ML, DL, and TFM models show more consistent improvement with larger training data. Overall, the findings highlight a trade-off between data efficiency and stability, suggesting that LLMs may serve as a flexible complementary approach for CKD screening when labeled data are limited.",
      "published": "2026-09-03T15:53:20Z",
      "abstract_url": "http://arxiv.org/abs/2609.04013v1",
      "pdf_url": "https://arxiv.org/pdf/2609.04013v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Investigating the Ability of Large Language Models to Analyze Recipes for Diabetes",
      "authors": [
        "Revathy Venkataramanan",
        "Aditya Luthra",
        "Venkatesan Nadimuthu",
        "Amit Sheth"
      ],
      "abstract": "Several studies have evaluated the ability of Large Language Models (LLMs) for meal planning, yielding positive outcomes. These models can process natural language inputs and leverage learned knowledge from their pretraining to generate meal plans. In this work, we investigate the ability of LLMs to analyze the suitability of given recipes for diabetes. The primary challenge for LLMs is to retrieve relevant dietary guidelines for diabetes, decompose recipes into ingredients and cooking methods, and apply these guidelines to determine the recipe's suitability. To study these challenges, we employ three kinds of prompts namely, (i) Direct Query Prompt (ii) Context-Guided Prompt, and (iii) Exemplary Context Prompt that incorporate different levels of diabetes dietary guidelines from medical sources. We introduce a benchmark dataset curated for this investigation consisting of 7607 recipes that include 3807 recipes suitable for diabetes and 3800 recipes not suitable for diabetes. Our results demonstrate that most LLMs are cautious in predicting recipes as suitable to prevent detrimental outcomes. Further, the models that can reason using the dietary guidelines performed better in predicting the suitability of recipes for diabetes. Overall, Mistral-7B and Llama 70B showed superior performance to their counterparts.",
      "published": "2026-09-03T15:03:36Z",
      "abstract_url": "http://arxiv.org/abs/2609.03967v1",
      "pdf_url": "https://arxiv.org/pdf/2609.03967v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting",
      "authors": [
        "Tomas Guija-Valiente",
        "Blanca Rodriguez-Gonzalez",
        "Norberto Malpica",
        "Angel Torrado-Carvajal"
      ],
      "abstract": "Medical image inpainting has the potential to improve automated brain MRI analysis by reconstructing healthy tissue within pathological regions. We introduce RARF, a task-agnostic region-aware rectified flow framework for masked data generation. We instantiate the framework for 3D brain MRI inpainting as our submission to the BraTS Inpainting Challenge 2026. RARF restricts the stochastic interpolation process to the inpainting region, while the observed voxels remain fixed and provide patient-specific anatomical context. A three-dimensional neural network receives the partially voided image, with Gaussian noise filling the missing region, together with the inpainting mask and the corresponding timestep. The model is trained using masked flow-matching and reconstruction-consistency objectives, combined with mask-aware preprocessing and data augmentation. During inference, the learned velocity field transports the initial noise toward a plausible reconstruction of the missing tissue, which is then combined with the unchanged observed anatomy. Experiments under the BraTS evaluation protocol show that the proposed approach produces competitive reconstructions while maintaining anatomical consistency. Source code is available at: https://github.com/TomasGuija/rarf.",
      "published": "2026-09-03T14:55:57Z",
      "abstract_url": "http://arxiv.org/abs/2609.03956v1",
      "pdf_url": "https://arxiv.org/pdf/2609.03956v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
