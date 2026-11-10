const PAPERS_DATA = {
  "last_updated": "2026-11-10 06:57:11 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "A Strict Upper Bound for Three-Point Weighted Selection in Planar Vector-Valued Regression",
      "authors": [
        "Sergey Volkov"
      ],
      "abstract": "We prove $F_{\\mathrm{weighted}}(2,2,3)<11/6$ for minimum-Frobenius-norm least-squares regression with two input and two output coordinates, improving the upper bound $15/8$. More precisely, a universal $\\varepsilon_0>0$ satisfies $\\mathcal T_3\\le(5/6-\\varepsilon_0)E$ for every finite residual frame. The proof combines an exact scalar certificate, rigidity of equality in planar selection, and a compactness argument that controls disappearing weights. Together with the known lower bound, this gives $13/8\\le F_{\\mathrm{weighted}}(2,2,3)<11/6$. No numerical value of $\\varepsilon_0$ is claimed.",
      "published": "2026-11-09T15:20:15Z",
      "abstract_url": "http://arxiv.org/abs/2611.12550v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12550v1",
      "categories": [
        "math.ST",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Phrase-Level Speculative Decoding on a Frozen Target",
      "authors": [
        "Madhusudhana Naidu"
      ],
      "abstract": "Speculative decoders raise throughput by banking more target tokens per forward pass, and Medusa, EAGLE-3, DFlash and DSpark all do it the same way: more candidates that survive verification. All of them draft in the target's own vocabulary, so one node of the candidate tree is one target token, and the tokenizer makes that unit look like a constant. It is not. BPE's pretokenizer splits on whitespace before merges are learned, so no multi-word string can ever become a single token: in Qwen3-4B, \"United States of America\" is four tokens and no adjacent pair of them exists as one entry, so a phrase is out of reach of any method drafting in the target's vocabulary. We change the unit. A small MLP head reads the frozen target's hidden state and predicts a one- to three-word phrase as one entry in a separate draft vocabulary; greedy longest-match expansion maps it back into the target's own tokens, and N candidates are verified in one forward pass behind a prefix-tree mask, so the change is orthogonal to tree depth and width. The target and its tokenizer stay frozen and a candidate is accepted only while it matches the target's own argmax, so the served output is identical to greedy decoding up to floating-point tie-breaking. On one A5000 at batch 1, against HuggingFace generate(), a 21.94M head (0.545% of Qwen3-4B) reaches 2.391x on GSM8K, 2.444x on MATH-500 and 1.544x on MT-Bench. Because the head's output layer is one row of 257 parameters per entry, the draft vocabulary size is the parameter budget: an 8,000-entry head is 3.434M parameters, 0.0854% of the target and 64x smaller than EAGLE-3's drafter, and keeps 92.3% of the larger head's GSM8K tokens per pass. With EAGLE-3 run on same prompts and baseline, our 21.94M head is ahead of tree-16 on both maths sets and behind tree-60 on every set.",
      "published": "2026-11-09T14:13:32Z",
      "abstract_url": "http://arxiv.org/abs/2611.12464v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12464v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Pricing Symbolic Rules by What GNNs Leave Unlearned",
      "authors": [
        "Satabarto Sarkar",
        "Prithwineel Paul"
      ],
      "abstract": "Symbolic knowledge (motif counts, ring counts, label constraints) is routinely added to graph neural networks (GNNs) and justified by their limited expressivity. We replace this yardstick with an exact one. The largest log-loss reduction a rule $S$ can give a model family is $I(Y;S\\midσ)$ given what the family computes; every realised gain splits exactly into this term plus repaid neural debt; and for a trained network with representation $T$ the value is $I(Y;S\\mid T)$, the label information training failed to extract. Rule design thus becomes rule discovery: we score candidates on the network's out-of-fold residual against calibrated permutation nulls with family-wise error control. On ZINC, discovery recovers the hidden ring term $\\max(0,\\ell_{\\max}-6)$ of the target in three independent runs ($p_{\\mathrm{adj}}=0.001$) and beats expert cycle counts; frozen and applied to five unseen datasets it is never significantly worse than the plain network or the expert rule, and on BBBP it significantly beats both.",
      "published": "2026-11-09T14:03:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.12446v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12446v1",
      "categories": [
        "cs.NE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Patching Up Circuit Learning via Parameter Sharing",
      "authors": [
        "Charles Bricout",
        "Gennaro Gala",
        "Viacheslav Borovitskiy",
        "Antonio Vergari"
      ],
      "abstract": "Probabilistic circuits (PCs) have emerged as generative models guaranteeing tractable probabilistic inference. However, tractability comes at the price of larger model sizes than in other generative models. As such, to scale PCs to high-dimensional data such as images, previous work has devised several parameter sharing techniques as well as fast evaluation protocols that consider learning PCs on independent image patches. In this work, we revisit this practice to highlight a surprising learnability issue: models that assume independence counterintuitively can yield better likelihoods than models that do not. To break this apparent bottleneck and to scale PCs further, we introduce a recursive parameter sharing construction (RePA) that fosters efficiency without assuming independence. We show empirically that RePA can greatly reduce time and memory consumption while yielding competitive likelihoods. Inspired by these results, we further generalize the approach to data modalities that do not inherently exhibit patch-structure (NoPA), such as tabular data, showing similar efficiency improvements, with marginal changes in likelihoods and sample quality on seven tabular datasets. Code available at https://github.com/april-tools/RePA.",
      "published": "2026-11-09T13:59:16Z",
      "abstract_url": "http://arxiv.org/abs/2611.12436v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12436v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "To SIMD or Not to SIMD: Comparing RVV vs. CGRA SoCs for Non-Linear Functions in AI",
      "authors": [
        "Sameer Deshmukh",
        "Naoya Mizuki",
        "Takumi Honda"
      ],
      "abstract": "Non-linear functions are an important class of operations in modern large language models. However, chip vendors building for AI have so far relied on traditional vector or scalar processing units for non-linear functions. The performance of these units has not kept pace with matrix multiplication units (MMUs), which can be found in both high-performance and edge-computing devices from almost every vendor. Reconfigurable units such as FPGAs and CGRAs have been proposed in the literature, but have not been systematically compared with traditional units backed by MMUs. In this paper, we build two SoCs: one using an RVV vector processor and the other using a latency-insensitive (\\emph{elastic}) CGRA, both coupled with an MMU and driven by a single-core CPU. We then propose mappings for common feed-forward networks on our SoCs and find that the CGRA reaches a maximum speedup in clock cycles of 7.12x over RVV on the activation stage of simple point-wise non-linear functions like SiLU, whereas the RVV reaches a maximum speedup of \\textbf{9.8x} over the CGRA on complex non-linear functions such as $\\mathrm{softmax}$. Over the enclosing feed-forward block the corresponding maxima are 3.16x and 4.24x.",
      "published": "2026-11-09T13:58:23Z",
      "abstract_url": "http://arxiv.org/abs/2611.12435v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12435v1",
      "categories": [
        "cs.AR",
        "cs.AI"
      ]
    },
    {
      "title": "AutoBuddy: Evaluating Proactive, Always-On Agents on Task Discovery and Execution in a Simulated Workplace",
      "authors": [
        "Shiqi Chen",
        "Daniel Madrigal Diaz",
        "Wei-Ning Chen",
        "Gbola Afonja",
        "Fangkai Yang",
        "Rujia Wang",
        "Chetan Bansal",
        "Victor Rühle",
        "Saravan Rajmohan",
        "Dongge Han"
      ],
      "abstract": "As LLM agents become more autonomous, they are increasingly expected to run as always-on assistants on a user's behalf, identifying work to be done and completing it without being asked. Yet existing benchmarks usually give the agent its task, whereas in knowledge work the requests sit in the user's workspace and the agent must first recognise which obligations exist. We study proactive task discovery and execution, in which an agent must infer a user's pending tasks from their workspace and complete them in the systems where the work lives. This setting poses two challenges. For the agent, the tasks are buried in a large volume of mail, chat, documents and records. For evaluation, a benchmark needs a simulated organisation whose people, events and documents are mutually consistent, so that tasks arise realistically and their correct outcomes are known. We introduce AutoBuddy-Bench, a simulator that addresses these challenges. It models an organisation's employees, projects, business systems and event history as a temporal graph, renders it into realistic workspace artefacts, derives ground-truth tasks from the events that create obligations, and computes each task's outcome from the world state; execution is judged by the agent's changes to the workspace. To evaluate current LLMs, we build AutoBuddy, a reference agent with interchangeable backbone LLMs: it identifies tasks from the workspace, grounding each in its source records, and executes each in a separate session with write access. Varying only the backbone, we find a systematic shift in where tasks are lost: weaker models fail mainly at discovery, intermediate models identify tasks but often stop short of carrying them out, and the strongest models fail mainly during execution, most often on business-system operations. Discovery precision is only weakly related to execution quality, suggesting the two are distinct capabilities.",
      "published": "2026-11-09T13:30:18Z",
      "abstract_url": "http://arxiv.org/abs/2611.12393v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12393v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Evidence or Marginal? What Moment-Vector Propagation Computes in Polynomial Density Neurons",
      "authors": [
        "Przemysław Spyra"
      ],
      "abstract": "When a neuron's input is lost, or known through a belief $q$, what should it output? For a neuron storing a joint polynomial density $N$ with input density $D$, two formulas compete. $p_\\mathrm{CD}=E_q[N]/E_q[D]$ costs one matrix-vector product; Hierarchical Correlation Reconstruction introduces it as a constant-denominator approximation to $p_\\mathrm{EU}=E_q[N/D]$. For a density model both are exact answers to different questions: $p_\\mathrm{CD}$ is Bayesian updating with the belief as virtual evidence; $p_\\mathrm{EU}$ installs the belief as a new marginal (Jeffrey's rule). With a missing input $p_\\mathrm{CD}$ returns the model's conditional; $p_\\mathrm{EU}$ generally does not. This post hoc reading is proved, and illustrated outside the pair family up to $k=8$ inputs; it has no pre-registered test, only one exploratory real-data pilot. Post hoc, closed forms fall within a factor 2 of all 112 measured gaps, and negative density begins near input correlation $1/k$. In a pre-registered synthetic study (up to 32 inputs), 5 of 15 hypotheses are supported, 2 inconclusive and 8 not supported, including all three on transfer beyond equicorrelated Gaussians. Post hoc, where $p_\\mathrm{EU}$ is defined, truncation usually outweighs the gap for Beta beliefs in pair neurons; triple neurons under flat and data beliefs are exceptions.",
      "published": "2026-11-09T12:57:07Z",
      "abstract_url": "http://arxiv.org/abs/2611.12339v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12339v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Rethinking Radar 3D Perception: Representation Learning from Raw Spectra",
      "authors": [
        "Miao Zhang",
        "Mohamed Abdelsamad",
        "Bin Yang",
        "Sherif Abdulatif",
        "Benedikt Loesch",
        "Marco Altmann",
        "Abhinav Valada",
        "Bin Yang"
      ],
      "abstract": "Raw radar spectra are widely viewed as too cluttered and noisy for direct representation learning, motivating handcrafted reductions, sparse point clouds, or compact Doppler descriptors in radar perception. We challenge this assumption and argue that the main limitation lies not in the raw representation itself, but in the absence of a learning objective that separates structured signal from noise. Handcrafted reductions partially address this problem but impose rigid assumptions about what constitutes signal, discarding weak yet informative structure while retaining artifacts that fit the reduction model. Once removed, such information cannot be recovered by supervised learning. We propose an alternative: structure is the component of a radar spectrum that remains reconstructable under localized corruption, while noise is the component that does not. This perspective motivates self-supervised masked modeling for dense radar spectra without predefined signal-noise separation. We instantiate it with Leakage-aware Spectral MAE, a masked autoencoder for raw radar cubes using a corruption strategy inspired by cross-bin energy spreading caused by spectral leakage. We also demonstrate the importance of Doppler organization: treating Doppler as a primary spatial dimension in an x-y-D representation, with height encoded as features, substantially outperforms naive channel stacking and enables effective learning from dense 4D spectra. Without handcrafted reductions, our approach consistently outperforms state-of-the-art radar-based detectors and point-cloud-based self-supervised learning (SSL) counterparts. These results establish raw radar spectra as an effective representation for 3D perception when paired with an appropriate pretraining objective.",
      "published": "2026-11-09T12:54:27Z",
      "abstract_url": "http://arxiv.org/abs/2611.12334v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12334v1",
      "categories": [
        "cs.AI",
        "cs.CV",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "Do Machine Learning Models Predict?",
      "authors": [
        "Michele Pizzochero",
        "Emanuele Ratti"
      ],
      "abstract": "Machine learning (ML) models are said to predict. Philosophers draw a line between accommodation and prediction, depending on whether or not the evidence a theory succeeds on was used in constructing it. Does this distinction survive when a theory is replaced by an ML model? We argue so, yet the dividing line becomes fuzzy. Whether an output of an ML model counts as an accommodation or as a prediction hinges upon the relationship between the patterns the input instantiate and those on which the model was selected through its training. Once this relation is taken into account, most successful outputs of ML models, we suggest, bear a strong accommodation character. We further trace the consequences of our account for confirmation, novelty, and scientific realism.",
      "published": "2026-11-09T12:51:18Z",
      "abstract_url": "http://arxiv.org/abs/2611.12332v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12332v1",
      "categories": [
        "cs.CY",
        "cs.AI"
      ]
    },
    {
      "title": "How Many Exposures Does a World Model Need? Learning Unseen Game Mechanics at Test Time",
      "authors": [
        "Ku Onoda",
        "Haruto Kawano",
        "Shota Takashiro",
        "Kohei Hayashi"
      ],
      "abstract": "World models enable agents to plan and learn from imagined experience, but in an unfamiliar environment they must learn its mechanics from limited interaction. A mechanic is a rule for how an action or an interaction changes part of the environment. Each occurrence of a mechanic that the model observes is one exposure. We ask how many exposures a world model needs to predict the effects of a new environment's mechanics. We address this question on ARC-AGI-3, a benchmark of interactive games defined by their programs. We introduce MechEval, which labels 69 recurring mechanics across 20 games from these programs. It scores each prediction on the cells the mechanic changes at each exposure. As a test bed, we build WARC (World model for ARC-AGI-3), a 35M-parameter vision transformer that is pretrained on other games and updates all of its parameters after each transition. On held-out games, WARC learns to predict four in five mechanics with at least 80% accuracy on the cells they change and keeps that accuracy to the end of the recordings. Half of them reach that accuracy within about 80 exposures, in a single play of the game, and without updates only one in seven does at all. In closed-loop play on the public games, WARC-Agent, which pairs WARC with a fixed graph explorer, triples the explorer's official efficiency score from 0.15 to 0.46, mostly through test-time training. On the hidden games of the Kaggle leaderboard, our entry scores 0.20.",
      "published": "2026-11-09T12:45:40Z",
      "abstract_url": "http://arxiv.org/abs/2611.12324v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12324v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "A Modular Perspective for Modeling MARL Systems",
      "authors": [
        "Tiziano Maisonhaute",
        "Jean-Christophe Soulié",
        "Fabien Michel"
      ],
      "abstract": "The growing complexity of Multi-Agent Reinforcement Learning (MARL) has fragmented the literature, obscuring underlying architectural choices behind diverse implementations. This fragmentation makes it difficult to evaluate critical modeling dimensions in isolation, identify how components affect the resulting system dynamics, and compare, reuse, or combine contributions across the literature. This paper proposes a modular perspective for modeling, analyzing, and implementing MARL systems. We decompose them into eight functional modules: agents, environments, learning, reward modeling, training and execution modes, communication, modeling of other agents, and evaluation criteria. For each module, we clarify its role and review representative approaches. We instantiate this decomposition in MARLKIT, a framework whose modules are explicit and interchangeable. MARLKIT demonstrates how the proposed perspective supports transparent experimental assumptions and the controlled variation and combination of MARL mechanisms. Experiments illustrate its ability to prototype, compare, reuse and combine modules while making differences between configurations explicit. Overall, the paper provides a structured perspective on MARL modeling and implementation, offering a common language for making modeling choices explicit, supporting clearer descriptions of MARL approaches, and supporting more controlled experimentation through the systematic isolation and combination of MARL mechanisms.",
      "published": "2026-11-09T12:39:59Z",
      "abstract_url": "http://arxiv.org/abs/2611.12314v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12314v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "HumanSightBench: A Benchmark for Human Visual Perception in MLLMs",
      "authors": [
        "Zixin Su",
        "Yangguang Lin",
        "Zegang Peng",
        "Yuntong Sun",
        "Xuanyi Wang",
        "Jiahao Wang",
        "Qiankai Xu",
        "Bing Yuan",
        "Jinze Bai",
        "Youbin Wu",
        "Jingjia Huang",
        "Ruyang Liu",
        "Yanxu Hu"
      ],
      "abstract": "Multimodal large language models (MLLMs) have demonstrated strong performance on knowledge-intensive and reasoning-based vision-language benchmark datasets, but still struggle to solve the visual questions that people can solve easily. Evaluating the differences between MLLMs and human-level visual perception is challenging due to end-to-end evaluation which combines various skills, or because we are uncertain if individual items are valid. Specifically, observed errors may reflect visual perception, language priors, background knowledge, question ambiguity, and underspecified answers. Our guideline for HumanSightBench is that we construct items that are human-easy by construction and model-hard by evaluation. HumanSightBench contains 2,876 visual questions with short answers across 7 top-level perceptual categories and 21 subcategories. We only include items that pass five inclusion gates: question clarity, answer uniqueness, visual grounding, human solvability, and text-only shortcut filtering. According to the protocol, errors are defined as incorrect answers on human-validated visual tasks with the evaluated interface. The inclusion gates are intended to reduce confounds from ambiguity and external knowledge. The best pass@3 score is 65.6% across eight evaluated MLLMs. No evaluated model exceeds 27.5% on Counting. These results identify capability gaps that are hidden by overall scores and can help future training experiments and held-out evaluation.",
      "published": "2026-11-09T12:27:19Z",
      "abstract_url": "http://arxiv.org/abs/2611.12305v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12305v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Depth as a Candidate Source: Portfolio Allocation for Recurrent Test-Time Scaling in Large Language Models",
      "authors": [
        "Chaoyuan Hao",
        "Wentao Yue",
        "Yanyan Wu",
        "Boyuan Gao",
        "Tianyou Lai",
        "Maolin Wang",
        "Yongxin Ni",
        "Hanwen Du",
        "Youhua Li"
      ],
      "abstract": "Recurrent latent reasoning in large lanugage models (LLMs) scales test-time computation by increasing recurrent depth, naturally framing inference as a problem of deciding when to stop. We identify a previously overlooked limitation of this single-depth view through systematic preliminary analysis: deeper recurrence can lose correct solutions found at shallower depths, while depth-specific success sets remain broadly non-nested even after controlling for marginal accuracy. This suggests that depth determines not only computation cost, but also access to distinct candidate solutions. Motivated by this finding, we propose Depth Portfolio Allocation (DPA), which reformulates recurrent test-time scaling as budgeted allocation across depth-specific candidate sources. DPA supports both fixed portfolios and feedback-guided sequential allocation under a shared inference budget. Experiments on GSM8K, SVAMP, MultiArith, and MBPP show that mixed-depth portfolios expand oracle coverage under matched candidate counts and nominal costs, while feedback-guided allocation generally matches or improves uniform allocation and can reduce computation in high-budget settings. Code and data are avaible at : https://anonymous.4open.science/r/LLM-latent-B7BC/.",
      "published": "2026-11-09T12:17:59Z",
      "abstract_url": "http://arxiv.org/abs/2611.12293v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12293v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "RAIL: Representation-Aligned Imitation Learning for Student-Compatible Teacher Policies",
      "authors": [
        "Meraj Mammadov",
        "Pedro Zuidberg Dos Martires",
        "Johannes Andreas Stork"
      ],
      "abstract": "Reinforcement learning (RL) from raw sensory inputs such as images or onboard sensors can be challenging due to high-dimensional observations and sparse rewards. A common strategy is to train a teacher policy with access to privileged state information and then distill it into a student policy that acts from raw inputs alone. However, when the teacher relies on information unavailable to the student, exact imitation may be impossible, creating an irreducible imitation gap. Existing approaches typically mitigate this mismatch through careful reward shaping or additional reinforcement learning on the student policy after the teacher has been trained. Instead, we introduce Representation-Aligned Imitation Learning (RAIL), which addresses the mismatch during teacher training itself. RAIL learns a latent representation shared across teacher and student observations using contrastive learning, and trains the teacher policy directly in this space. Because the teacher policy is constrained to operate on this shared space, it learns behaviors that are reproducible from student observations. This substantially reduces the imitation gap while preserving task performance. Across multiple environments, RAIL outperforms strong baselines without reward modification or post-hoc student fine-tuning, and surpasses direct reinforcement learning from raw observations. The learned representations further enable zero-shot transfer to new tasks through teacher-only training. Project site: https://merajmammadov.com/rail/.",
      "published": "2026-11-09T11:41:49Z",
      "abstract_url": "http://arxiv.org/abs/2611.12251v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12251v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "Controlling Temporal Pseudo-Label Marginals for Stable Online Test-Time Adaptation",
      "authors": [
        "Rani Amar",
        "Ofir Lindenbaum"
      ],
      "abstract": "Continual Test-Time Adaptation (CTTA) methods update models online using a stream of unlabeled data to overcome distribution shifts encountered at test time. However, CTTA is often unstable, as it can reinforce biases in early predictions and increasingly predict only a few classes, resulting in drift or even collapse. In this paper, we identify narrowing pseudo-label class usage as an indicator of CTTA instability and quantify this behavior. Most existing methods have attempted to address this issue by either selecting reliable adaptation samples or re-weighting the loss to encourage diverse pseudo-labels, but they do not explicitly enforce a class marginal over a temporal window. Instead, we propose Sinkhorn Class-Marginal Regularization (SCALAR), a simple plug-and-play module that controls the pseudo-label distribution over a short temporal window. SCALAR works by maintaining recent predictions in a buffer, balancing their class distribution via the Sinkhorn algorithm to match a specified target distribution, and using the balanced soft targets for adaptation. Without changing the model architecture, SCALAR maintains the per-step runtime and increases the peak allocated GPU memory by only 10 MB on ImageNet-C. Our method is particularly effective on skewed, non-stationary data with severe distribution shifts, where SCALAR significantly alleviates drift and collapse when combined with existing stabilization methods based on entropy or diversity.",
      "published": "2026-11-09T11:30:46Z",
      "abstract_url": "http://arxiv.org/abs/2611.12242v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12242v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "PARIS: A multi-agent LLM framework for end-to-end, PRISMA-guided systematic literature reviews with pre-synthesis corpus quality assessment",
      "authors": [
        "Christophe Cruz",
        "Hussam Ghanem",
        "Samir Jabbar",
        "Ait Kbir",
        "Hocine Cherifi"
      ],
      "abstract": "Systematic literature reviews (SRs) remain slow and labor-intensive, and most automation tools address single stages of the workflow. We present PARIS (PRISMA Agentic Research Intelligence System), an open-source framework in which a conductor agent running in Claude Code orchestrates 15 specialized large language model (LLM) agents, from scoping a research question to exporting a manuscript draft. Agents are specified declaratively, coordinate exclusively through persisted artifacts that the conductor validates before every transition, and delegate retrieval, bibliometric analysis and scoring to deterministic Python utilities. Three human checkpoints gate study inclusion, corpus adequacy and the synthesis outline. We also introduce QualityScore, a pre-synthesis index that aggregates normalized network, bibliometric, semantic and process sub-scores into a 0-100 grade. Two case studies illustrate both: AI chatbots and university students' learning (388 records, 100 included) and AI-assisted SR methods (131 records, 123 included, QualityScore 69.1 recomputed with a proxy network sub-score). They show end-to-end feasibility across domains and the diagnostic use of QualityScore: in the first run, corpus improvements chosen at the corpus-adequacy checkpoint raised the score from 74.6 to 81.9 and the number of pre-2023 studies from 1 to 19. Audits of both runs exposed failure modes: incomplete retrieval, counts that drifted between agents, unrecorded or delegated checkpoint decisions and a grade that depends on how citation data are handled. PARIS fully supports 8 of the 27 PRISMA 2020 items and partially supports 11. Output quality has so far been judged only by an internal reviewer agent; independent validation against human reviews is the next step.",
      "published": "2026-11-09T11:16:03Z",
      "abstract_url": "http://arxiv.org/abs/2611.12222v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12222v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "NearFormer: Linear-Size Sparse Attention via Randomized Near Neighbors",
      "authors": [
        "Ali Parviz",
        "Alex Cloninger",
        "Gal Mishne"
      ],
      "abstract": "Graph transformers can capture long-range interactions, but their quadratic attention cost limits their use on large graphs. We introduce \\emph{NearFormer}, a graph transformer whose global attention is restricted to a randomized near-neighbor (NN) graph. NearFormer uses an efficient randomized construction to connect each node to a small number of near neighbors from a broader neighborhood defined by a label-independent feature or spectral geometry. For a fixed retained budget \\(k\\), the resulting attention support has linear size \\(\\mathcal{O}(|E|+nk)\\). The NN graph is built once by within-bin search and then held fixed. We use these edges as sparse global-attention supports, with optional learned attention biases, while preserving the original graph for local message passing. Experiments on long-range graph benchmarks and synthetic structural tasks show that NearFormer provides a favorable accuracy--scalability trade-off relative to local message passing, dense and sparse attention.",
      "published": "2026-11-09T10:40:42Z",
      "abstract_url": "http://arxiv.org/abs/2611.12186v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12186v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "FraudBench: A Legal Evaluation of AI Deception",
      "authors": [
        "Kevin Wei",
        "Sumaya Nur Adan",
        "Stephan Llerena",
        "Mark Gitau",
        "Jakob Merane",
        "Anita Srinivasan",
        "Colette Le Brannan",
        "Keethan Kleiner",
        "Alessandro Tacconelli",
        "Michael Tiu",
        "Case Thomason",
        "Kyle Emili",
        "Natasza Gadowska",
        "Alex Mark",
        "Eric Ren",
        "Umang Bhatt",
        "Jatinder Singh",
        "Jacy Reese Anthis",
        "Doni Bloomfield",
        "Noam Kolt"
      ],
      "abstract": "We introduce FraudBench, an evaluation of large language models' (LLMs) propensities to recommend, assist, or directly take actions that would constitute fraud under U.S. law. Our evaluation contains a novel dataset of 159 scenarios based on real-world U.S. court cases and validated by legal experts. We present each scenario to LLMs via three tasks at varying levels of autonomy: advising users, drafting messages for users to send, and acting directly on behalf of users (477 tasks total). In each scenario, FraudBench re-enacts the facts of a court case to give LLMs an opportunity, but no explicit instruction, to either misrepresent or conceal a material fact in a high-stakes commercial setting--e.g., whether to describe company-funded product testing as \"independent\" to customers (misrepresentation) or whether to hide unfavorable regulatory investigations from clients (concealment). We evaluate 20 LLMs across 25 experimental conditions (totaling >300,000 generations) and include a preregistered human baseline ($n_{\\texttt{annotations}} = 2{,}957$, $n_{\\texttt{humans}} = 996$). We score a response as fraudulent when it engages in, assists with, or recommends misrepresentation or concealment. Key findings include: (1) across LLMs, fraud rates ranged from near 0% to over 40%; (2) LLM fraud rates increased with autonomy, with overall rates of 3.0% when advising, 14.7% when drafting, and 21.4% when acting directly, without applying pressure; (3) LLMs sometimes engaged in fraud despite recognizing legal risks; (4) LLMs more often engaged in fraud by concealment than by misrepresentation; and (5) prompt-based interventions substantially affected fraud rates, with a prompt to obey the law reducing fraud rates, while prompts to maximize profits and act agentically increased fraud rates.",
      "published": "2026-11-09T10:22:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.12167v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12167v1",
      "categories": [
        "cs.AI",
        "cs.CY",
        "cs.LG"
      ]
    },
    {
      "title": "The balance between feature learning and collapse in dynamical systems-based generative models",
      "authors": [
        "Julian Brandon",
        "Bruno Loureiro",
        "N Alex Cayco-Gajic",
        "Arthur Pellegrino"
      ],
      "abstract": "Over the past decade, a growing body of work has established that implicit biases in learning dynamics fundamentally shape the solutions found by neural networks, governing their learned features and generalisation performance. However, despite these strides in deep learning theory, far less is known about representation learning in generative models based on dynamical systems, from normalising flows to diffusion. Importantly, empirical evidence suggests that continuous-time generative models can be prone to mode collapse, yet a learning theory explaining such biases remains elusive. To address this, we use tools from dynamical systems and operator theory to express the gradient flow of a general class of stochastic differential equations as a composition of linear operators, allowing us to relate collapse to low-rank biases in generative models. In variational inference and flow-matching tasks, we show that this bias can manifest in two different mechanisms: i) where collapsed models are stable fixed points of the gradient flow, or ii) in sequential feature learning, where smaller modes take progressively longer to be learned, producing an effective collapse for a fixed training budget. Together these findings point to a trade-off between feature learning and collapse, suggesting an operator view as a step towards a unified framework for understanding the learning dynamics of dynamical systems-based generative models.",
      "published": "2026-11-09T10:05:19Z",
      "abstract_url": "http://arxiv.org/abs/2611.12156v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12156v1",
      "categories": [
        "stat.ML",
        "cs.AI",
        "cs.LG",
        "math.DS"
      ]
    },
    {
      "title": "Visible Tails: Characterizing Adaptation to Unknown Moments in Heavy-Tailed Bandits",
      "authors": [
        "Tuan Ngoc Do",
        "Hai Duc Nguyen"
      ],
      "abstract": "Heavy-tailed bandit learners without moment parameters face an obstruction: the rewards that carry an arm's mean can stay unseen. We address the remaining part of a COLT open problem by characterizing the optimal-arm laws that permit adaptation to unknown moments at constant-factor cost. The condition, variance-core visibility, is one-sided and translation-invariant: the mean hidden in any rare event is at most a constant multiple of the standard error of the rewards that remain. A quantile-trimmed UCB index that knows the visibility constants attains the central-moment benchmark on this class, and no single strategy adapts across all visibility levels. The same index achieves logarithmic expected regret on every instance with a visible optimal arm and finite-mean rewards. On an explicit family with no higher moment, gap $Δ$ and constant $A$, its logarithmic coefficient and the information lower bound are of order $Δ^{-1}e^{A/Δ}$, exponential in the inverse gap within one visibility class.",
      "published": "2026-11-09T09:53:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.12140v1",
      "pdf_url": "https://arxiv.org/pdf/2611.12140v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
