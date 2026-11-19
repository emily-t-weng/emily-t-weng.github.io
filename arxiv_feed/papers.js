const PAPERS_DATA = {
  "last_updated": "2026-11-19 06:59:46 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Knowing What You Know: A Trait Theory of Reviewer Value in Human-AI Cognitive Systems",
      "authors": [
        "Priyanka Sinha",
        "Sugam Pandey"
      ],
      "abstract": "When a capable model and a human reviewer cooperate on the same decision, which human still adds value? We argue this is a question about a single cognitive trait, metacognitive calibration: the alignment between a reviewer's behavioral confidence and the actual correctness of their judgments. We develop a trait theory of reviewer value to make it precise. The theory makes five claims: value is governed by calibration rather than by activity, speed, agreement, or tenure; calibration leaves observable traces in behavioral telemetry; those traces are non-diagnostic on their own, because the fast, low-revision signature of a calibrated expert is indistinguishable from that of a confident novice until it is gated by outcomes; human calibration and a model's self-knowledge are the same construct, so the joint system should defer to whichever agent is better calibrated on each item; and individual calibration is causally identifiable only above the level of the single judgment. We ground each claim in a six-month production window of an AI-assisted accounting system (541621 reviewer events, 483 reviewers), and find that a purely behavioral expertise profile inverts the ground truth, scoring an overconfident cohort (5.2 percent correction survival) higher than genuine experts (99.2 percent); that reviewer quality is flat across tenure, so the trait is not learned on the job; and that the same calibration logic that separates good from bad reviewers, applied to the model, concentrates 92.5 percent of human effort on genuine errors. We close with the boundary the theory must respect: under majority-vote aggregation, individual causal value is unidentifiable.",
      "published": "2026-11-18T18:55:07Z",
      "abstract_url": "http://arxiv.org/abs/2611.24932v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24932v1",
      "categories": [
        "cs.AI",
        "cs.CE",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "Who are you? The Personal Imitation Game",
      "authors": [
        "Muhammad Rizky Wellyanto",
        "Hajara-Yasmin Isa",
        "Kexin Xue",
        "Vishal Ravichandran",
        "Wenpeng Lu",
        "Jerry O. Talton",
        "Ranjitha Kumar"
      ],
      "abstract": "Large Language Models (LLMs) now perform strongly on Alan Turing's 1950 Imitation Game, so we propose a harder challenge, the Personal Imitation Game, which asks whether a machine can replicate a specific individual in conversation rather than a generic human. Testing with 30 participants, we establish a human baseline and report the identity signals interrogators rely on: stylometric features, personality traits, and biographical facts. To demonstrate the framework, we design an LLM-based system that plays the game for one of the paper's authors, drawing on an autoethnographic dataset spanning ten years of personal digital communication. In a 20-person deception study, interrogators choose our system over the author's own friends in 7 of 10 rounds, and the author over our system in 13 of 20; neither rate is distinguishable from chance at this sample size. We consider design improvements, applications of personal AI agents, and implications for identity and privacy.",
      "published": "2026-11-18T18:32:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.24888v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24888v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Auditing black-box clinical prediction models with a large language model through queries alone: a blinded test with planted flaws",
      "authors": [
        "Zhiyu Liu",
        "Xinyu Yang"
      ],
      "abstract": "Large language models (LLMs) are used to explain clinical prediction models, on the promise that they can say what a model computes, find its faults and even reveal new knowledge in it. We tested the three promises where the answers could be checked. From the International Stroke Trial we trained four kinds of model (logistic regression, explainable boosting machine, gradient-boosted trees, neural network) for two outcomes on patients randomised earlier, planted one flaw in each of 24 copies (an under-predicted subgroup, a reversed risk factor or an implausible step) and mixed them with the 8 clean models. GPT-6 Astra audited each model three times, blind, through a tool that only returned the model's predictions; the judging rules were committed before any run and applied by program. Its surrogate models, with a median of 128 terms, reproduced the audited models' log-odds with a mean $R^2$ of 0.976 (0.959 for an additive surrogate of 42 terms fitted by an analyst). It found the planted flaw in 69 of 72 runs (96%; 95% CI 80% to 99%); a script written for the purpose found all 24. It also reported many more problems: 60% of 1,089 checkable reports failed the pre-specified check against the model's average behaviour; post hoc, most of these were real but small steps, and 15% of the checkable reports were not found in the model. Of 601 unexpected relationships it read out of the clean models, 11% held in patients randomised later (random pairs 6%; a regression scan's shorter list of its strongest candidates, 28%); after a check on the training data, 33% against 30%. With query access alone, an LLM can describe a black-box model faithfully and find nearly every planted flaw, as a purpose-written script also did; its reports of problems must be checked on the model and its discoveries on data.",
      "published": "2026-11-18T18:24:54Z",
      "abstract_url": "http://arxiv.org/abs/2611.24877v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24877v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Does the Action Matter? Probing Action-Conditioned Semantic Forecasting as a World Model for Autonomous Driving",
      "authors": [
        "Binay Kumar Singh",
        "Niels Da Vitoria Lobo"
      ],
      "abstract": "World models are increasingly used in autonomous driving to predict how a scene evolves under the ego vehicle's actions, supporting counterfactual reasoning and model-based planning. However, existing evaluations focus on prediction fidelity and planning performance, and rarely verify that the predictions depend on the conditioning action. In this paper, we study action dependence in action-conditioned semantic forecasting, which predicts future semantic segmentation from past segmentations and the ego action. We compare a forecaster with feature-wise and input-level action conditioning against an identical model without action input, and measure action dependence with test-time action interventions scored by standard segmentation metrics. The evaluation is validated in a synthetic environment with known action-dependent dynamics and applied to Cityscapes and BDD100K, both within each dataset and across the two vehicle platforms. Experiments show that the executed action improves mid-term mIoU by at most 2.5 points on Cityscapes and 0.6 points on BDD100K, whereas a wrong action reduces it by up to 23.2 and 7.0 points, below the model without action input. On real data, feature-wise conditioning depends 1.6 times more on the action than input-level conditioning at similar accuracy, whereas both depend on it equally in the synthetic environment, where the history does not determine the action. These results show that an accuracy gain indicates that a driving world model uses its action, but not how strongly its forecasts depend on it. Code is available at https://github.com/binayksingh/does-action-matter.",
      "published": "2026-11-18T18:06:20Z",
      "abstract_url": "http://arxiv.org/abs/2611.24840v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24840v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "A Foundation Model to Measure and Predict Human Behaviour",
      "authors": [
        "Haoran Ye",
        "Tianrui Huo",
        "Hangzhen Li",
        "Xin Zhang",
        "Guojie Song"
      ],
      "abstract": "Large language models (LLMs) are increasingly used to infer psychological attributes, predict individual behaviour, and simulate human populations. However, despite generating plausible responses, they often misrepresent individuals and distort population statistics. Here we introduce Chiron, a foundation model to measure and predict human behaviour. We jointly fine-tune an LLM to represent psychological constructs, predict individual behaviour, and align population response distributions with human statistics. The training and evaluation data span 36 behavioural domains and 78 social science experiments, comprising 288,307 person records, 1,411,750 atomic behavioural decisions, and 99,953 psychometric assessments over 109 psychological constructs. In our evaluations, Chiron's psychological measurements agree more closely with human self-reports and exhibit fewer biases that undermine psychometric validity. It predicts individual behaviour more accurately than frontier LLMs (up to 28.4% relative improvement). In simulations of social science experiments, it more accurately captures the direction and magnitude of human experimental effects (82.2--94.3% lower effect-size estimation error). Together, these results establish Chiron as a unified computational instrument for measuring and predicting human behaviour.",
      "published": "2026-11-18T17:59:08Z",
      "abstract_url": "http://arxiv.org/abs/2611.24830v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24830v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "Soft Mixture of Quantum Experts for Hybrid Quantum-Classical Learning",
      "authors": [
        "Valeria Repetto",
        "Romina D'Aurizio",
        "Piero Farinelli"
      ],
      "abstract": "Quantum Machine Learning (QML) has attracted increasing interest as a possible route to construct feature representations that are difficult to reproduce with classical models. However, near-term QML architectures based on trainable parametrized quantum circuits remain affected by limited hardware resources, noise, and severe optimization difficulties such as barren plateaus. In this work we propose Soft-MoQE, a Soft Mixture of Quantum Experts architecture that replaces variational quantum training with the input-dependent selection and combination of fixed quantum feature maps. The model is composed of a heterogeneous pool of non-trainable quantum circuits, each defining a distinct embedding, and a classical gating network that learns how to weight these embeddings for each data point. We benchmark Soft-MoQE on three PCA-reduced tasks: MNIST parity, MNIST 10-class, and FashionMNIST. Soft-MoQE outperforms the single best expert, the uniform average of all experts and a linear baseline, achieving task-dependent performance comparable to that of a trainable quantum neural network, without optimizing any quantum parameter and with a quantum training cost that is lower by orders of magnitude. The analysis of the class- conditioned expert usage shows that the learned routing combines a few experts for each input, with a selection that depends on the class. These results support the learned routing among fixed quantum circuits as a trainability-aware design principle for hybrid QML.",
      "published": "2026-11-18T17:41:45Z",
      "abstract_url": "http://arxiv.org/abs/2611.24798v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24798v1",
      "categories": [
        "quant-ph",
        "cs.AI"
      ]
    },
    {
      "title": "Bias-Only Training Dynamics in ReLU Networks",
      "authors": [
        "Shreyas Chaudhari",
        "Jose M. F. Moura"
      ],
      "abstract": "Neural networks with randomly initialized, fixed weights and trainable biases are universal approximators, yet the dynamics of bias-only training remain largely unexplored. We study these dynamics in one-hidden-layer ReLU networks through activation codes, the binary patterns of active neurons that partition the input space into convex polyhedral regions. We show that the gradients with respect to the biases depend on each input only through its activation code, and hence the training dynamics can be characterized with these codes. In particular, as long as the activation codes remain fixed, training reduces to a linear least-squares problem with the codes as features. Training can then be understood as a sequence of least-squares problems, with transitions occurring when neurons change their activation states. We show that the fraction of switching neurons vanishes as the network widens, and the training dynamics approach those of linear least squares on the initial activation codes in the infinite-width limit. Our experiments support these theoretical findings, showing that while activation switches can noticeably affect training in narrower networks, their relative frequency and the associated changes in the dynamics diminish as the width increases.",
      "published": "2026-11-18T17:41:18Z",
      "abstract_url": "http://arxiv.org/abs/2611.24797v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24797v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning Trajectories Leave Decodable Traces in Neural Network Weight Space and Internal Representations",
      "authors": [
        "Patrick Krauss",
        "Achim Schilling",
        "Andreas Maier",
        "Thomas Kinfe",
        "Claus Metzner"
      ],
      "abstract": "Neural networks trained on the same task develop related weight-space structures, yet remain distinguishable after learning. Do these differences merely reflect alternative solutions, or do they preserve information about individual learning histories? We tested whether final weights reveal initialization, task order, and earlier trajectory states. Training reorganized the ensemble: initial pairwise distances did not consistently predict final distances. Nevertheless, all 30 trained networks were correctly matched to their initial weights among ten candidates per dataset. Networks sharing an initialization also developed recognizable final hidden representations across different batch schedules, although direct comparisons with initial activity did not recover identity. Learning order remained readable as well. A linear probe decoded the order of two earlier tasks with 100% accuracy on 80 held-out networks after ten epochs of a common final task. Earlier intermediate weight states were partly predictable, but individual initial weights could not be reliably reconstructed. Final weights thus retain selective historical information rather than a complete trajectory record. Perturbed networks recovered classification performance without returning to their unperturbed training paths. This separation persisted across datasets, intervention times, and network layers. Moreover, 2,157 of 2,160 aligned endpoint pairs showed no positive loss barrier at sampled interpolation points, indicating low-loss connections between distinct solutions rather than convergence to one weight vector. Together, these findings show that a shared task constrains networks to related successful solutions without erasing their individual histories. Final weights reflect not only what a network has learned, but also where learning began and how it unfolded.",
      "published": "2026-11-18T17:27:55Z",
      "abstract_url": "http://arxiv.org/abs/2611.24782v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24782v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.NC"
      ]
    },
    {
      "title": "Counterfactual World Models: JEPAs That Reason About \"What If\"",
      "authors": [
        "Randall Balestriero",
        "Yann LeCun",
        "Kerda Varaku"
      ],
      "abstract": "World models operate in a world they cannot fully observe. When a hidden disturbance, such as a current pushing a boat, changes the outcome of an action, a counterfactual question follows: what would have happened under a different action, subject to the same disturbance? Its answer separates bad decisions from bad luck. We prove that world models trained on factual transitions cannot answer it on their own, whatever their data or capacity. For many physical systems, the observed outcome and a simple property of the dynamics suffice: natura non facit saltus, nature makes no leaps. When actions and disturbances act as separate forces, their interaction is second-order in time, so the counterfactual outcome is, to first order, the observed outcome plus the difference between the world model's predictions under the alternative and factual actions. This yields a counterfactual world model with no counterfactual data, extra parameters, or retraining. For transitions of duration $h$, we prove an $O(h^2)$ bound on the counterfactual error relative to the disturbance variance, establish a matching minimax lower bound from unpaired factual data, and characterize failures at discontinuities, coordinate folds, and disturbances that scale the command. We validate these predictions from pixels with JEPA world models on frozen DINOv2 and masked autoencoder (MAE) encoders. On four rendered simulators, our approach improves counterfactual $R^2$ from $38$--$46\\%$ to $86$--$96\\%$ with the same world models; on a fifth, an underwater vehicle, it reaches $90$--$94\\%$ with sixteen times more factual data. Four further frozen encoders, DINOv3, SigLIP~2, CLIP and EVA-02, give $81$--$95\\%$ on the same simulators. In hindsight planning, it finds the action that would have reached the goal in $35$--$69\\%$ of events, against $14$--$23\\%$ for an oracle planner that knows the simulator but not the outcome.",
      "published": "2026-11-18T17:24:00Z",
      "abstract_url": "http://arxiv.org/abs/2611.24778v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24778v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV",
        "cs.RO",
        "stat.ML"
      ]
    },
    {
      "title": "LLM-Guided Cyber-Physical Attack Path Identification in Smart Grids",
      "authors": [
        "Amr Elshazly",
        "Jesse Roberts",
        "Rachad Atat",
        "Muhammad Ismail"
      ],
      "abstract": "Identifying high-impact attack paths in cyber-physical power systems is critical for improving grid resilience. Existing methods typically rely on either structural topology analysis or simulation-based cascading-failure models. Topology-based approaches capture connectivity but lack adversarial reasoning, while simulation-driven methods are primarily suited for offline analysis and are less effective for rapid evaluation of evolving attack scenarios. As a result, identifying realistic attack paths that jointly capture topology, threat semantics, and physical impact remains challenging. This paper presents a physics-grounded framework for identifying cyber-physical attack paths using large language models (LLMs), with cascading-failure-derived impact rankings used as physics-grounded references. The framework integrates system topology, LLM-guided STRIDE and MITRE ATT&CK threat modeling, and three baseline strategies for comparison: topology-driven, threat-driven, and fusion-based. Evaluation is performed using (i) critical-node overlap between LLM-inferred top-K rankings and cascading-failure-based ground truth; (ii) node-level similarity between LLM-inferred and reference paths; and (iii) path-level alignment capturing agreement in complete attack sequences. Results show that structured LLM-based threat modeling substantially improves alignment with physics-grounded references. In the cyber layer, critical-node overlap@3 increases from approximately 33% to as high as 100%, while node-level similarity reaches up to 65% and path-level alignment reaches up to 48%. In addition, the proposed framework reduces runtime by approximately 75--85% compared to the fusion-based baseline in the IEEE 14-bus system. These results suggest that LLM-guided attack path identification is a promising approach for efficient and impact-aware cyber-physical security analysis.",
      "published": "2026-11-18T17:16:56Z",
      "abstract_url": "http://arxiv.org/abs/2611.24765v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24765v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Horizontal Flow Matching for Single-Cell Spatiotemporal Dynamics",
      "authors": [
        "Zihan Wang",
        "Jianzhe Li",
        "Zhenyi Zhang",
        "Peijie Zhou"
      ],
      "abstract": "Learning the spatiotemporal dynamics of cells from spatial transcriptomics requires connecting snapshots of different cells acquired in different coordinate frames. A rotation or translation of a section can appear as cell movement and distort the inferred correspondence. We introduce horizontal flow matching, which learns changes in tissue organization, molecular state, and cell abundance up to shared rigid motion. The dynamics follow a continuity--reaction equation whose action assigns no cost to shared rigid motion, and the resulting horizontal projector removes this component from spatial prediction errors during learning and from predicted velocities during generation. An equivalent endpoint problem jointly estimates correspondence and relative pose in Wasserstein--Fisher--Rao geometry and yields analytic paths whose velocity and growth serve as regression targets. Regressing these targets recovers the horizontal dynamics under ideal population conditions. Our model, HORIZON, uses an equivariant transformer to predict velocity and growth from the current tissue, so that a tissue can be generated continuously from a single initial snapshot. On a simulated tissue observed in perturbed coordinate frames, alignment improves reconstruction and projection reduces rigid drift. On four spatial transcriptomics time courses, HORIZON reconstructs tissue geometry, gene expression, and cell abundance more accurately than existing spatial and unbalanced trajectory methods, and interpolates held-out developmental stages.",
      "published": "2026-11-18T17:16:53Z",
      "abstract_url": "http://arxiv.org/abs/2611.24764v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24764v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.QM"
      ]
    },
    {
      "title": "Elysium: Simulation-Free Unbalanced Schrödinger Bridges",
      "authors": [
        "Zihan Wang",
        "Jianzhe Li",
        "Zhenyi Zhang",
        "Peijie Zhou"
      ],
      "abstract": "Reconstructing cellular dynamics from snapshots is challenging because the destructive nature of single-cell measurements makes tracking the same cells over time impossible. Cellular dynamics are inherently stochastic and involve changes in population size through proliferation and apoptosis. Growth-based methods represent population change as a rate on the density, which does not assign exits and entries to individual cells. Unbalanced Schrödinger bridges model individual events, but neural solvers for killed diffusion bridges simulate trajectories during training. We introduce Elysium, a simulation-free solver that jointly learns state transition and mass change. Using a killed Brownian reference and its time reversal, Elysium constructs direct regression objectives from an optimal endpoint coupling and conditional bridge distributions, without simulating trajectories during training. We show that the minimizers of the expected training objectives recover the optimal unbalanced Schrödinger bridge. Empirically, Elysium reconstructs state transitions, population size, and event distributions on simulated benchmarks with known killed-diffusion ground truth, outperforming existing deterministic and bridge-based solvers. On single-cell time courses, it interpolates cell state distributions at unobserved time points, tracks population expansion and contraction, and captures biphasic drug-response dynamics.",
      "published": "2026-11-18T17:16:41Z",
      "abstract_url": "http://arxiv.org/abs/2611.24763v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24763v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.QM"
      ]
    },
    {
      "title": "How Generative Models Generate: A Concise Guide to the Generative AI Landscape",
      "authors": [
        "Tianhua Chen"
      ],
      "abstract": "The rapidly expanding generative AI landscape brings together methods that differ in architecture, learning objective, mathematical representation and sampling procedure. A concise, selectively organised technical guide can provide an accessible starting point for developing a coherent understanding of this increasingly complex landscape. This article takes a generation-centred approach, connecting how models produce new observations with the modelling and learning principles that make generation possible. We develop a two-level perspective: first, distinguishing three recurring generation routes-direct transformation or decoding, sequential conditional construction, and iterative updating or transport; and second, examining the modelling and learning principles underlying these routes. Representative approaches are introduced through their key components, learning objectives and sampling procedures. We then examine their broader connections, including how different architectures and generation mechanisms can overlap or be combined. By bringing together essential technical explanations within a compact conceptual framework, the article aims to help readers develop a working understanding of unfamiliar approaches, recognise connections among models they have encountered, and navigate different approaches within the broader generative AI landscape.",
      "published": "2026-11-18T16:45:19Z",
      "abstract_url": "http://arxiv.org/abs/2611.24728v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24728v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Openhagen Worlds: Calibrated Environments for Measuring and Training Scientific Discovery Agents",
      "authors": [
        "Alexander Boesgaard Lorup"
      ],
      "abstract": "Agents that do science must choose informative experiments and reason from their results, but most discovery benchmarks cannot separate that skill from recall, unstated scoring rules or shortcut policies. We introduce Openhagen Worlds, a suite of 27 procedurally generated environments (some with finite truth spaces), from classic puzzles of experimental reasoning to simulator-grounded physics and chemistry, in which an agent runs experiments under a budget and commits an answer that a deterministic verifier checks against a hidden truth. Every observation states the complete scoring rule, and each of the 178 difficulty levels is calibrated: it is admitted only if, on 64 development instances, a scripted oracle succeeds on at least 0.75, the strongest plausible non-deductive policy on at most 0.25 and each gated shortcut on at most 0.35. The 92 core levels span all 27 families and are sized to train small models; the 86 frontier rungs extend the 15 discovery worlds with harder questions and with campaigns of three, five and eight or more committed stages on one hidden system. On the same 84 withheld frontier instances, one per family and rung, DeepSeek V4.1 Flash solves 32 at its default reasoning effort with 16,384 tokens per call and 57 at its maximum effort with up to 393,216 tokens per call (oracle: 79); the headroom that remains is largest in eight-stage campaigns. On-policy expert iteration gives a 9B model modest gains: its core-tier success on trained families rises from 0.090 to 0.148, partly through better action format, with no detectable loss on GPQA Diamond. A hosted service plays the core tier's withheld instances.",
      "published": "2026-11-18T16:36:18Z",
      "abstract_url": "http://arxiv.org/abs/2611.24711v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24711v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "When Does Continual Pre-Training Help? A Controlled Study of Dialectal Adaptation for Québec French",
      "authors": [
        "Eeham Khan",
        "Richard Khoury",
        "Leila Kosseim"
      ],
      "abstract": "Continual pre-training (CPT) is widely used to adapt pretrained language models to dialectal and non-standard language varieties, yet it remains unclear when this additional training is necessary, what drives its benefits, and whether aggregate evaluation scores adequately capture its effects. In this paper, we present a controlled study of CPT for Québec French language understanding across nine LLMs spanning different families and scales. Using a budget-matched protocol of 30M tokens per experiment, our results show that the benefit of CPT depends strongly on the starting model's existing competence in the target variety. When CPT is beneficial, the improvements are driven more by corpus composition than by data volume, with literary text accounting for most of the gain. Paired Québec and standard French evaluations further show that CPT primarily improves expression comprehension rather than grammatical competence, and that corpus register determines which regional expressions are acquired. These gains also involve capability-specific trade-offs, including declines in reading comprehension that aggregate scores can conceal. Overall, these findings show that CPT for dialectal adaptation should not be treated as a uniformly beneficial default. Instead, adaptation strategies should begin with a careful assessment of the base model's existing competence, accompanied by source-level data analysis and task-level evaluation.",
      "published": "2026-11-18T16:26:03Z",
      "abstract_url": "http://arxiv.org/abs/2611.24699v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24699v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Depth Before Breadth: Reverse-to-Forward Curriculum with Advantage Decomposition for LLM Agent",
      "authors": [
        "Siyuan Du",
        "Yuxin Liu",
        "Meng Tang",
        "Haoran Li",
        "Yechao Bai",
        "Feiyang Pan",
        "Jia He"
      ],
      "abstract": "Reinforcement learning (RL) has become a common approach for training large language model (LLM) agents on long-horizon tasks. However, learning from sparse terminal rewards makes credit assignment and exploration difficult. We approach these challenges through a policy-induced capability region, the set of states from which the current policy can reliably complete a task. Near its boundary, an agent may need only a few actions to reach a more solvable state, even when the goal remains distant. This reduces long-horizon learning to two coupled subproblems of reaching that state and completing the task from there. We address them with paired advantages and a progressive reverse-to-forward curriculum. The advantages provide separate signals for reaching more solvable states and completing the task from them, while the curriculum expands the capability region backward along successful reference trajectories and then into off-reference states through on-policy exploration. On ALFWorld and WebShop, our method outperforms competitive agentic RL methods at both model scales, achieving success rates of 90.1%/96.1% and 80.2%/82.6% with Qwen2.5-1.5B/7B-Instruct, respectively.",
      "published": "2026-11-18T16:25:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.24697v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24697v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models",
      "authors": [
        "Prakhar Mishra",
        "Bhavya Yajush Awasthi",
        "Tanuja",
        "Nishant"
      ],
      "abstract": "Small language models (SLMs) can run on commodity hardware and support privacy-preserving, on-premises clinical deployment, but their medical explanations are often short and poorly structured compared with large language models. Recent supervised approaches such as Meerkat transfer chain-of-thought reasoning into 7B-8B backbones, yet their training objective does not directly target the style or organization of the generated rationales. Building on the instruction-tuned Mistral-7B backbone, we investigate whether a lightweight reinforcement learning (RL) step can improve explanation structure without human preference labels.We apply Proximal Policy Optimization (PPO) with LoRA adaptation to 2,000 Meerkat-derived medical prompts, guided by a simple self-supervised reward that favors longer, lexically diverse, and well-punctuated responses while penalizing very short ones. Training for about 170 PPO steps on a single NVIDIA A100 GPU yields a stable increase in the average reward. Quantitative and qualitative comparisons show that the RL-tuned model produces longer, more organized, multi-point explanations while preserving the underlying medical content.We discuss the promise and the limitations of self-rewarded RL for shaping the style of medical explanations.",
      "published": "2026-11-18T15:52:24Z",
      "abstract_url": "http://arxiv.org/abs/2611.24650v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24650v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Choosing the Right Representations: Task-Aware Fusion for Multimodal Materials Language Models",
      "authors": [
        "Huu Binh Ta",
        "Alex Morehead",
        "N. Benjamin Erichson",
        "Ferdinando Fioretto",
        "Sandeep Madireddy"
      ],
      "abstract": "Multimodal language models for materials science rely on effective mechanisms for incorporating three-dimensional structural representations. MatterChat addresses this challenge using a large bridge module between a pretrained structural encoder and the LLM, together with a separate contrastive alignment stage. We instead ask whether this bridge can adaptively select and combine representations from multiple structural encoders according to the target property. We find that supervised and generative material encoders provide complementary information for property prediction. We found that, for Zatom-1, this information is not concentrated in the final layer. Instead, the most informative representations occur at different depths depending on the target property. This motivates Task-Conditioned Attention Fusion (TCAF), a lightweight attention-based bridge that uses task-conditioned latent queries to combine CHGNet features with intermediate Zatom-1 representations. The bridge is trained directly on downstream prediction tasks, allowing representation selection and material-language alignment to be learned jointly without separate contrastive pretraining. Across classification and regression tasks, TCAF outperforms MatterChat, with up to 47.8% lower RMSE, while using substantially fewer trainable parameters and halving the overall training time. Further experiments incorporating UMA-S demonstrate TCAF's flexibility across encoder combinations, with the three-encoder configuration achieving the highest macro-average classification accuracy of 89.61%.",
      "published": "2026-11-18T15:48:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.24641v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24641v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Clover-35B-A3B-Preview Technical Report",
      "authors": [
        "DiffuSpace Technical Team"
      ],
      "abstract": "We introduce Clover-35B-A3B-Preview (Clover), a mixture-of-experts diffusion language model with 35 billion total parameters and 3 billion activated parameters, supporting thinking-mode generation. Clover adapts Gated Delta Networks to block diffusion by combining recurrent memory of previously generated context with bidirectional softmax attention inside the current block. Historical memory readout competes with current-block tokens in a shared softmax, with learnable block-size calibration and a separate, zero-initialized right-context convolution branch. Clover leads the seven evaluated agent metrics, including 73.6 on SWE-bench Verified and 51.7 on Terminal-Bench 2.1. On internal agent workloads, it reaches a reported generation speed of 680 tokens/s using one NVIDIA GB300 GPU at BF16 precision and batch size one. We organize the evaluation around model quality, drafting and verification costs, and deployment under constrained memory. We also describe a decision-model variant that generates answers to related questions in one denoising pass. On an internal set of 500 groups, joint prediction raises question accuracy from 90.37% to 93.88% and all-correct group accuracy from 65.4% to 77.0%.",
      "published": "2026-11-18T15:46:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.24636v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24636v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "GateWatch: Low-Overhead LLM Monitoring with Boolean Circuits",
      "authors": [
        "Firas Spanioli",
        "Yaniv David"
      ],
      "abstract": "As large language models (LLMs) and LLM-based agents are increasingly deployed, monitoring their behavior becomes important for detecting unsafe or unintended actions. Activation probes offer an efficient approach, but existing methods face trade-offs between accuracy, interpretability, and computational overhead. Probes on raw activations lack interpretable inputs, whereas sparse autoencoder (SAE) probes use interpretable features but often rely on additive classifiers and large SAE encoders, limiting interpretability and increasing memory overhead. We introduce GateWatch, a compact, interpretable classifier that expresses its entire decision rule as an explicit Boolean logic over a small number of binarized SAE features. We explore three approaches to constructing these circuits under a specified gate budget. For efficient deployment, GateWatchServe computes only selected SAE features, enables streaming of their binary values across layers during inference, and evaluates multiple GateWatch classifiers in parallel using compact lookup tables. Across 24 tasks on Gemma-2-9B, GateWatch achieves 80.49% mean test accuracy using only 3.8 SAE features and 3.5 logic gates per probe on average, just 1.17 percentage points below full 16K-feature SAE probes, while matching or outperforming them on 10 tasks and reducing probe memory by 99.98%. Inside vLLM, 1,000 concurrent multi-layer, multi-token GateWatch classifiers retain 97-98% of baseline throughput and 99.8% of KV-cache capacity.",
      "published": "2026-11-18T15:37:04Z",
      "abstract_url": "http://arxiv.org/abs/2611.24619v1",
      "pdf_url": "https://arxiv.org/pdf/2611.24619v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
