const PAPERS_DATA = {
  "last_updated": "2026-03-15 03:20:34 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "The Latent Color Subspace: Emergent Order in High-Dimensional Chaos",
      "authors": [
        "Mateusz Pach",
        "Jessica Bader",
        "Quentin Bouniot",
        "Serge Belongie",
        "Zeynep Akata"
      ],
      "abstract": "Text-to-image generation models have advanced rapidly, yet achieving fine-grained control over generated images remains difficult, largely due to limited understanding of how semantic information is encoded. We develop an interpretation of the color representation in the Variational Autoencoder latent space of FLUX.1 [Dev], revealing a structure reflecting Hue, Saturation, and Lightness. We verify our Latent Color Subspace (LCS) interpretation by demonstrating that it can both predict and explicitly control color, introducing a fully training-free method in FLUX based solely on closed-form latent-space manipulation. Code is available at https://github.com/ExplainableML/LCS.",
      "published": "2026-03-12T17:59:48Z",
      "abstract_url": "http://arxiv.org/abs/2603.12261v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12261v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training",
      "authors": [
        "Yixin Liu",
        "Yue Yu",
        "DiJia Su",
        "Sid Wang",
        "Xuewei Wang",
        "Song Jiang",
        "Bo Liu",
        "Arman Cohan",
        "Yuandong Tian",
        "Zhengxing Chen"
      ],
      "abstract": "Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, while reasoning judges have shown better performance on static evaluation benchmarks, their effectiveness in actual policy training has not been systematically examined. Therefore, we conduct a rigorous study to investigate the actual impact of non-reasoning and reasoning judges in reinforcement-learning-based LLM alignment. Our controlled synthetic setting, where a \"gold-standard\" judge (gpt-oss-120b) provides preference annotations to train smaller judges, reveals key differences between non-reasoning and reasoning judges: non-reasoning judges lead to reward hacking easily, while reasoning judges can lead to policies that achieve strong performance when evaluated by the gold-standard judge. Interestingly, we find that the reasoning-judge-trained policies achieve such strong performance by learning to generate highly effective adversarial outputs that can also score well on popular benchmarks such as Arena-Hard by deceiving other LLM-judges. Combined with our further analysis, our study highlights both important findings and room for improvements for applying (reasoning) LLM-judges in non-verifiable LLM post-training.",
      "published": "2026-03-12T17:57:06Z",
      "abstract_url": "http://arxiv.org/abs/2603.12246v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12246v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Separable neural architectures as a primitive for unified predictive and generative intelligence",
      "authors": [
        "Reza T. Batley",
        "Apurba Sarker",
        "Rajib Mostakim",
        "Andrew Klichine",
        "Sourav Saha"
      ],
      "abstract": "Intelligent systems across physics, language and perception often exhibit factorisable structure, yet are typically modelled by monolithic neural architectures that do not explicitly exploit this structure. The separable neural architecture (SNA) addresses this by formalising a representational class that unifies additive, quadratic and tensor-decomposed neural models. By constraining interaction order and tensor rank, SNAs impose a structural inductive bias that factorises high-dimensional mappings into low-arity components. Separability need not be a property of the system itself: it often emerges in the coordinates or representations through which the system is expressed. Crucially, this coordinate-aware formulation reveals a structural analogy between chaotic spatiotemporal dynamics and linguistic autoregression. By treating continuous physical states as smooth, separable embeddings, SNAs enable distributional modelling of chaotic systems. This approach mitigates the nonphysical drift characteristics of deterministic operators whilst remaining applicable to discrete sequences. The compositional versatility of this approach is demonstrated across four domains: autonomous waypoint navigation via reinforcement learning, inverse generation of multifunctional microstructures, distributional modelling of turbulent flow and neural language modelling. These results establish the separable neural architecture as a domain-agnostic primitive for predictive and generative intelligence, capable of unifying both deterministic and distributional representations.",
      "published": "2026-03-12T17:56:54Z",
      "abstract_url": "http://arxiv.org/abs/2603.12244v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12244v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Security Considerations for Artificial Intelligence Agents",
      "authors": [
        "Ninghui Li",
        "Kaiyuan Zhang",
        "Kyle Polley",
        "Jerry Ma"
      ],
      "abstract": "This article, a lightly adapted version of Perplexity's response to NIST/CAISI Request for Information 2025-0035, details our observations and recommendations concerning the security of frontier AI agents. These insights are informed by Perplexity's experience operating general-purpose agentic systems used by millions of users and thousands of enterprises in both controlled and open-world environments. Agent architectures change core assumptions around code-data separation, authority boundaries, and execution predictability, creating new confidentiality, integrity, and availability failure modes. We map principal attack surfaces across tools, connectors, hosting boundaries, and multi-agent coordination, with particular emphasis on indirect prompt injection, confused-deputy behavior, and cascading failures in long-running workflows. We then assess current defenses as a layered stack: input-level and model-level mitigations, sandboxed execution, and deterministic policy enforcement for high-consequence actions. Finally, we identify standards and research gaps, including adaptive security benchmarks, policy models for delegation and privilege control, and guidance for secure multi-agent system design aligned with NIST risk management principles.",
      "published": "2026-03-12T17:49:39Z",
      "abstract_url": "http://arxiv.org/abs/2603.12230v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12230v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR"
      ]
    },
    {
      "title": "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights",
      "authors": [
        "Yulu Gan",
        "Phillip Isola"
      ],
      "abstract": "Pretraining produces a learned parameter vector that is typically treated as a starting point for further iterative adaptation. In this work, we instead view the outcome of pretraining as a distribution over parameter vectors, whose support already contains task-specific experts. We show that in small models such expert solutions occupy a negligible fraction of the volume of this distribution, making their discovery reliant on structured optimization methods such as gradient descent. In contrast, in large, well-pretrained models the density of task-experts increases dramatically, so that diverse, task-improving specialists populate a substantial fraction of the neighborhood around the pretrained weights. Motivated by this perspective, we explore a simple, fully parallel post-training method that samples $N$ parameter perturbations at random, selects the top $K$, and ensembles predictions via majority vote. Despite its simplicity, this approach is competitive with standard post-training methods such as PPO, GRPO, and ES for contemporary large-scale models.",
      "published": "2026-03-12T17:49:30Z",
      "abstract_url": "http://arxiv.org/abs/2603.12228v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12228v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration",
      "authors": [
        "Priyanka Kargupta",
        "Shuhaib Mehri",
        "Dilek Hakkani-Tur",
        "Jiawei Han"
      ],
      "abstract": "Despite interdisciplinary research leading to larger and longer-term impact, most work remains confined to single-domain academic silos. Recent AI-based approaches to scientific discovery show promise for interdisciplinary research, but many prioritize rapidly designing experiments and solutions, bypassing the exploratory, collaborative reasoning processes that drive creative interdisciplinary breakthroughs. As a result, prior efforts largely prioritize automating scientific discovery rather than augmenting the reasoning processes that underlie scientific disruption. We present Idea-Catalyst, a novel framework that systematically identifies interdisciplinary insights to support creative reasoning in both humans and large language models. Starting from an abstract research goal, Idea-Catalyst is designed to assist the brainstorming stage, explicitly avoiding premature anchoring on specific solutions. The framework embodies key metacognitive features of interdisciplinary reasoning: (a) defining and assessing research goals, (b) awareness of a domain's opportunities and unresolved challenges, and (c) strategic exploration of interdisciplinary ideas based on impact potential. Concretely, Idea-Catalyst decomposes an abstract goal (e.g., improving human-AI collaboration) into core target-domain research questions that guide the analysis of progress and open challenges within that domain. These challenges are reformulated as domain-agnostic conceptual problems, enabling retrieval from external disciplines (e.g., Psychology, Sociology) that address analogous issues. By synthesizing and recontextualizing insights from these domains back into the target domain, Idea-Catalyst ranks source domains by their interdisciplinary potential. Empirically, this targeted integration improves average novelty by 21% and insightfulness by 16%, while remaining grounded in the original research problem.",
      "published": "2026-03-12T17:48:34Z",
      "abstract_url": "http://arxiv.org/abs/2603.12226v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12226v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Proof-Carrying Materials: Falsifiable Safety Certificates for Machine-Learned Interatomic Potentials",
      "authors": [
        "Abhinaba Basu",
        "Pavan Chakraborty"
      ],
      "abstract": "Machine-learned interatomic potentials (MLIPs) are deployed for high-throughput materials screening without formal reliability guarantees. We show that a single MLIP used as a stability filter misses 93% of density functional theory (DFT)-stable materials (recall 0.07) on a 25,000-material benchmark. Proof-Carrying Materials (PCM) closes this gap through three stages: adversarial falsification across compositional space, bootstrap envelope refinement with 95% confidence intervals, and Lean 4 formal certification. Auditing CHGNet, TensorNet and MACE reveals architecture-specific blind spots with near-zero pairwise error correlations (r <= 0.13; n = 5,000), confirmed by independent Quantum ESPRESSO validation (20/20 converged; median DFT/CHGNet force ratio 12x). A risk model trained on PCM-discovered features predicts failures on unseen materials (AUC-ROC = 0.938 +/- 0.004) and transfers across architectures (cross-MLIP AUC-ROC ~ 0.70; feature importance r = 0.877). In a thermoelectric screening case study, PCM-audited protocols discover 62 additional stable materials missed by single-MLIP screening - a 25% improvement in discovery yield.",
      "published": "2026-03-12T17:13:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.12183v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12183v1",
      "categories": [
        "cond-mat.mtrl-sci",
        "cs.AI",
        "cs.LG",
        "physics.comp-ph"
      ]
    },
    {
      "title": "A Quantitative Characterization of Forgetting in Post-Training",
      "authors": [
        "Krishnakumar Balasubramanian",
        "Shiva Prasad Kasiviswanathan"
      ],
      "abstract": "Continual post-training of generative models is widely used, yet a principled understanding of when and why forgetting occurs remains limited. We develop theoretical results under a two-mode mixture abstraction (representing old and new tasks), proposed by Chen et al. (2025) (arXiv:2510.18874), and formalize forgetting in two forms: (i) mass forgetting, where the old mixture weight collapses to zero, and (ii) old-component drift, where an already-correct old component shifts during training. For equal-covariance Gaussian modes, we prove that forward-KL objectives trained on data from the new distribution drive the old weight to zero, while reverse-KL objectives converge to the true target (thereby avoiding mass forgetting) and perturb the old mean only through overlap-gated misassignment probabilities controlled by the Bhattacharyya coefficient, yielding drift that decays exponentially with mode separation and a locally well-conditioned geometry with exponential convergence. We further quantify how replay interacts with these objectives. For forward-KL, replay must modify the training distribution to change the population optimum; for reverse-KL, replay leaves the population objective unchanged but prevents finite-batch old-mode starvation through bounded importance weighting. Finally, we analyze three recently proposed near-on-policy post-training methods, SDFT (arxiv:2601.19897), TTT-Discover (arxiv:2601.16175), and OAPL (arxiv:2602.19362), via the same lens and derive explicit conditions under which each retains old mass and exhibits overlap-controlled drift. Overall, our results show that forgetting can by precisely quantified based on the interaction between divergence direction, geometric behavioral overlap, sampling regime, and the visibility of past behavior during training.",
      "published": "2026-03-12T17:00:16Z",
      "abstract_url": "http://arxiv.org/abs/2603.12163v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12163v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.ST",
        "stat.ML"
      ]
    },
    {
      "title": "IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL",
      "authors": [
        "Zhoujun Cheng",
        "Yutao Xie",
        "Yuxiao Qu",
        "Amrith Setlur",
        "Shibo Hao",
        "Varad Pimpalkhute",
        "Tongtong Liang",
        "Feng Yao",
        "Zhengzhong Liu",
        "Eric Xing",
        "Virginia Smith",
        "Ruslan Salakhutdinov",
        "Zhiting Hu",
        "Taylor Killian",
        "Aviral Kumar"
      ],
      "abstract": "While scaling laws guide compute allocation for LLM pre-training, analogous prescriptions for reinforcement learning (RL) post-training of large language models (LLMs) remain poorly understood. We study the compute-optimal allocation of sampling compute for on-policy RL methods in LLMs, framing scaling as a compute-constrained optimization over three resources: parallel rollouts per problem, number of problems per batch, and number of update steps. We find that the compute-optimal number of parallel rollouts per problem increases predictably with compute budget and then saturates. This trend holds across both easy and hard problems, though driven by different mechanisms: solution sharpening on easy problems and coverage expansion on hard problems. We further show that increasing the number of parallel rollouts mitigates interference across problems, while the number of problems per batch primarily affects training stability and can be chosen within a broad range. Validated across base models and data distributions, our results recast RL scaling laws as prescriptive allocation rules and provide practical guidance for compute-efficient LLM RL post-training.",
      "published": "2026-03-12T16:49:21Z",
      "abstract_url": "http://arxiv.org/abs/2603.12151v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12151v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance",
      "authors": [
        "Quanhao Li",
        "Zhen Xing",
        "Rui Wang",
        "Haidong Cao",
        "Qi Dai",
        "Daoguo Dong",
        "Zuxuan Wu"
      ],
      "abstract": "Recent advances in trajectory-controllable video generation have achieved remarkable progress. Previous methods mainly use adapter-based architectures for precise motion control along predefined trajectories. However, all these methods rely on a multi-step denoising process, leading to substantial time redundancy and computational overhead. While existing video distillation methods successfully distill multi-step generators into few-step, directly applying these approaches to trajectory-controllable video generation results in noticeable degradation in both video quality and trajectory accuracy. To bridge this gap, we introduce FlashMotion, a novel training framework designed for few-step trajectory-controllable video generation. We first train a trajectory adapter on a multi-step video generator for precise trajectory control. Then, we distill the generator into a few-step version to accelerate video generation. Finally, we finetune the adapter using a hybrid strategy that combines diffusion and adversarial objectives, aligning it with the few-step generator to produce high-quality, trajectory-accurate videos. For evaluation, we introduce FlashBench, a benchmark for long-sequence trajectory-controllable video generation that measures both video quality and trajectory accuracy across varying numbers of foreground objects. Experiments on two adapter architectures show that FlashMotion surpasses existing video distillation methods and previous multi-step models in both visual quality and trajectory consistency.",
      "published": "2026-03-12T16:45:53Z",
      "abstract_url": "http://arxiv.org/abs/2603.12146v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12146v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.MM"
      ]
    },
    {
      "title": "Automatic Generation of High-Performance RL Environments",
      "authors": [
        "Seth Karten",
        "Rahul Dev Appapogu",
        "Chi Jin"
      ],
      "abstract": "Translating complex reinforcement learning (RL) environments into high-performance implementations has traditionally required months of specialized engineering. We present a reusable recipe - a generic prompt template, hierarchical verification, and iterative agent-assisted repair - that produces semantically equivalent high-performance environments for <$10 in compute cost. We demonstrate three distinct workflows across five environments. Direct translation (no prior performance implementation exists): EmuRust (1.5x PPO speedup via Rust parallelism for a Game Boy emulator) and PokeJAX, the first GPU-parallel Pokemon battle simulator (500M SPS random action, 15.2M SPS PPO; 22,320x over the TypeScript reference). Translation verified against existing performance implementations: throughput parity with MJX (1.04x) and 5x over Brax at matched GPU batch sizes (HalfCheetah JAX); 42x PPO (Puffer Pong). New environment creation: TCGJax, the first deployable JAX Pokemon TCG engine (717K SPS random action, 153K SPS PPO; 6.6x over the Python reference), synthesized from a web-extracted specification. At 200M parameters, the environment overhead drops below 4% of training time. Hierarchical verification (property, interaction, and rollout tests) confirms semantic equivalence for all five environments; cross-backend policy transfer confirms zero sim-to-sim gap for all five environments. TCGJax, synthesized from a private reference absent from public repositories, serves as a contamination control for agent pretraining data concerns. The paper contains sufficient detail - including representative prompts, verification methodology, and complete results - that a coding agent could reproduce the translations directly from the manuscript.",
      "published": "2026-03-12T16:45:47Z",
      "abstract_url": "http://arxiv.org/abs/2603.12145v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12145v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "TopoBench: Benchmarking LLMs on Hard Topological Reasoning",
      "authors": [
        "Mayug Maniparambil",
        "Nils Hoehing",
        "Janak Kapuriya",
        "Arjun Karuvally",
        "Ellen Rushe",
        "Anthony Ventresque",
        "Noel O'Connor",
        "Fergal Reid"
      ],
      "abstract": "Solving topological grid puzzles requires reasoning over global spatial invariants such as connectivity, loop closure, and region symmetry and remains challenging for even the most powerful large language models (LLMs). To study these abilities under controlled settings, we introduce TopoBench, a benchmark of six puzzle families across three difficulty levels. We evaluate strong reasoning LLMs on TopoBench and find that even frontier models solve fewer than one quarter of hard instances, with two families nearly unsolved. To investigate whether these failures stem from reasoning limitations or from difficulty extracting and maintaining spatial constraints, we annotate 750 chain of thought traces with an error taxonomy that surfaces four candidate causal failure modes, then test them with targeted interventions simulating each error type. These interventions show that certain error patterns like premature commitment and constraint forgetting have a direct impact on the ability to solve the puzzle, while repeated reasoning is a benign effect of search. Finally we study mitigation strategies including prompt guidance, cell-aligned grid representations and tool-based constraint checking, finding that the bottleneck lies in extracting constraints from spatial representations and not in reasoning over them. Code and data are available at github.com/mayug/topobench-benchmark.",
      "published": "2026-03-12T16:37:21Z",
      "abstract_url": "http://arxiv.org/abs/2603.12133v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12133v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "SommBench: Assessing Sommelier Expertise of Language Models",
      "authors": [
        "William Brach",
        "Tomas Bedej",
        "Jacob Nielsen",
        "Jacob Pichna",
        "Juraj Bedej",
        "Eemeli Saarensilta",
        "Julie Dupouy",
        "Gianluca Barmina",
        "Andrea Blasi Núñez",
        "Peter Schneider-Kamp",
        "Kristian Košťál",
        "Michal Ries",
        "Lukas Galke Poech"
      ],
      "abstract": "With the rapid advances of large language models, it becomes increasingly important to systematically evaluate their multilingual and multicultural capabilities. Previous cultural evaluation benchmarks focus mainly on basic cultural knowledge that can be encoded in linguistic form. Here, we propose SommBench, a multilingual benchmark to assess sommelier expertise, a domain deeply grounded in the senses of smell and taste. While language models learn about sensory properties exclusively through textual descriptions, SommBench tests whether this textual grounding is sufficient to emulate expert-level sensory judgment. SommBench comprises three main tasks: Wine Theory Question Answering (WTQA), Wine Feature Completion (WFC), and Food-Wine Pairing (FWP). SommBench is available in multiple languages: English, Slovak, Swedish, Finnish, German, Danish, Italian, and Spanish. This helps separate a language model's wine expertise from its language skills. The benchmark datasets were developed in close collaboration with a professional sommelier and native speakers of the respective languages, resulting in 1,024 wine theory question-answering questions, 1,000 wine feature-completion examples, and 1,000 food-wine pairing examples. We provide results for the most popular language models, including closed-weights models such as Gemini 2.5, and open-weights models, such as GPT-OSS and Qwen 3. Our results show that the most capable models perform well on wine theory question answering (up to 97% correct with a closed-weights model), yet feature completion (peaking at 65%) and food-wine pairing show (MCC ranging between 0 and 0.39) turn out to be more challenging. These results position SommBench as an interesting and challenging benchmark for evaluating the sommelier expertise of language models. The benchmark is publicly available at https://github.com/sommify/sommbench.",
      "published": "2026-03-12T16:19:04Z",
      "abstract_url": "http://arxiv.org/abs/2603.12117v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12117v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Taming the Adversary: Stable Minimax Deep Deterministic Policy Gradient via Fractional Objectives",
      "authors": [
        "Taeho Lee",
        "Donghwan Lee"
      ],
      "abstract": "Reinforcement learning (RL) has achieved remarkable success in a wide range of control and decision-making tasks. However, RL agents often exhibit unstable or degraded performance when deployed in environments subject to unexpected external disturbances and model uncertainties. Consequently, ensuring reliable performance under such conditions remains a critical challenge. In this paper, we propose minimax deep deterministic policy gradient (MMDDPG), a framework for learning disturbance-resilient policies in continuous control tasks. The training process is formulated as a minimax optimization problem between a user policy and an adversarial disturbance policy. In this problem, the user learns a robust policy that minimizes the objective function, while the adversary generates disturbances that maximize it. To stabilize this interaction, we introduce a fractional objective that balances task performance and disturbance magnitude. This objective prevents excessively aggressive disturbances and promotes robust learning. Experimental evaluations in MuJoCo environments demonstrate that the proposed MMDDPG achieves significantly improved robustness against both external force perturbations and model parameter variations.",
      "published": "2026-03-12T16:15:06Z",
      "abstract_url": "http://arxiv.org/abs/2603.12110v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12110v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents",
      "authors": [
        "Deyu Zou",
        "Yongqiang Chen",
        "Fan Feng",
        "Mufei Li",
        "Pan Li",
        "Yu Gong",
        "James Cheng"
      ],
      "abstract": "Reinforcement learning (RL) with outcome-based rewards has achieved significant success in training large language model (LLM) agents for complex reasoning tasks. However, in active reasoning where agents need to strategically ask questions to acquire task-relevant information, we find that LLM agents trained with RL often suffer from information self-locking: the agent ceases to ask informative questions and struggles to internalize already-obtained information. To understand the phenomenon, we decompose active reasoning into two core capabilities: Action Selection (AS), which determines the observation stream through queries, and Belief Tracking (BT), which updates the agent's belief based on collected evidence. We show that deficient AS and BT capabilities will limit the information exploration during RL training. Furthermore, insufficient exploration in turn hinders the improvement of AS and BT, creating a feedback loop that locks the agent in a low-information regime. To resolve the issue, we propose a simple yet effective approach that reallocates the learning signal by injecting easy- to-obtain directional critiques to help the agent escape self-locking. Extensive experiments with 7 datasets show that our approach significantly mitigates the information self-locking, bringing up to 60% improvements.",
      "published": "2026-03-12T16:14:14Z",
      "abstract_url": "http://arxiv.org/abs/2603.12109v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12109v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Human-Centred LLM Privacy Audits: Findings and Frictions",
      "authors": [
        "Dimitri Staufer",
        "Kirsten Morehouse",
        "David Hartmann",
        "Bettina Berendt"
      ],
      "abstract": "Large language models (LLMs) learn statistical associations from massive training corpora and user interactions, and deployed systems can surface or infer information about individuals. Yet people lack practical ways to inspect what a model associates with their name. We report interim findings from an ongoing study and introduce LMP2, a browser-based self-audit tool. In two user studies ($N_{total}{=}458$), GPT-4o predicts 11 of 50 features for everyday people with $\\ge$60\\% accuracy, and participants report wanting control over LLM-generated associations despite not considering all outputs privacy violations. To validate our probing method, we evaluate eight LLMs on public figures and non-existent names, observing clear separation between stable name-conditioned associations and model defaults. Our findings also contribute to exposing a broader generative AI evaluation crisis: when outputs are probabilistic, context-dependent, and user-mediated through elicitation, what model--individual associations even include is under-specified and operationalisation relies on crafting probes and metrics that are hard to validate or compare. To move towards reliable, actionable human-centred LLM privacy audits, we identify nine frictions that emerged in our study and offer recommendations for future work and the design of human-centred LLM privacy audits.",
      "published": "2026-03-12T16:01:01Z",
      "abstract_url": "http://arxiv.org/abs/2603.12094v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12094v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "cs.CL",
        "cs.CY"
      ]
    },
    {
      "title": "Resource-Efficient Iterative LLM-Based NAS with Feedback Memory",
      "authors": [
        "Xiaojie Gu",
        "Dmitry Ignatov",
        "Radu Timofte"
      ],
      "abstract": "Neural Architecture Search (NAS) automates network design, but conventional methods demand substantial computational resources. We propose a closed-loop pipeline leveraging large language models (LLMs) to iteratively generate, evaluate, and refine convolutional neural network architectures for image classification on a single consumer-grade GPU without LLM fine-tuning. Central to our approach is a historical feedback memory inspired by Markov chains: a sliding window of $K{=}5$ recent improvement attempts keeps context size constant while providing sufficient signal for iterative learning. Unlike prior LLM optimizers that discard failure trajectories, each history entry is a structured diagnostic triple -- recording the identified problem, suggested modification, and resulting outcome -- treating code execution failures as first-class learning signals. A dual-LLM specialization reduces per-call cognitive load: a Code Generator produces executable PyTorch architectures while a Prompt Improver handles diagnostic reasoning. Since both the LLM and architecture training share limited VRAM, the search implicitly favors compact, hardware-efficient models suited to edge deployment. We evaluate three frozen instruction-tuned LLMs (${\\leq}7$B parameters) across up to 2000 iterations in an unconstrained open code space, using one-epoch proxy accuracy on CIFAR-10, CIFAR-100, and ImageNette as a fast ranking signal. On CIFAR-10, DeepSeek-Coder-6.7B improves from 28.2% to 69.2%, Qwen2.5-7B from 50.0% to 71.5%, and GLM-5 from 43.2% to 62.0%. A full 2000-iteration search completes in ${\\approx}18$ GPU hours on a single RTX~4090, establishing a low-budget, reproducible, and hardware-aware paradigm for LLM-driven NAS without cloud infrastructure.",
      "published": "2026-03-12T16:00:22Z",
      "abstract_url": "http://arxiv.org/abs/2603.12091v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12091v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Multi-Label Temporal Convolutional Framework for Transcription Factor Binding Characterization",
      "authors": [
        "Pietro Demurtas",
        "Ferdinando Zanchetta",
        "Giovanni Perini",
        "Rita Fioresi"
      ],
      "abstract": "Transcription factors (TFs) regulate gene expression through complex and co-operative mechanisms. While many TFs act together, the logic underlying TFs binding and their interactions is not fully understood yet. Most current approaches for TF binding site prediction focus on individual TFs and binary classification tasks, without a full analysis of the possible interactions among various TFs. In this paper we investigate DNA TF binding site recognition as a multi-label classification problem, achieving reliable predictions for multiple TFs on DNA sequences retrieved in public repositories. Our deep learning models are based on Temporal Convolutional Networks (TCNs), which are able to predict multiple TF binding profiles, capturing correlations among TFs andtheir cooperative regulatory mechanisms. Our results suggest that multi-label learning leading to reliable predictive performances can reveal biologically meaningful motifs and co-binding patterns consistent with known TF interactions, while also suggesting novel relationships and cooperation among TFs.",
      "published": "2026-03-12T15:42:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.12073v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12073v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.GN"
      ]
    },
    {
      "title": "Chemical Reaction Networks Learn Better than Spiking Neural Networks",
      "authors": [
        "Sophie Jaffard",
        "Ivo F. Sbalzarini"
      ],
      "abstract": "We mathematically prove that chemical reaction networks without hidden layers can solve tasks for which spiking neural networks require hidden layers. Our proof uses the deterministic mass-action kinetics formulation of chemical reaction networks. Specifically, we prove that a certain reaction network without hidden layers can learn a classification task previously proved to be achievable by a spiking neural network with hidden layers. We provide analytical regret bounds for the global behavior of the network and analyze its asymptotic behavior and Vapnik-Chervonenkis dimension. In a numerical experiment, we confirm the learning capacity of the proposed chemical reaction network for classifying handwritten digits in pixel images, and we show that it solves the task more accurately and efficiently than a spiking neural network with hidden layers. This provides a motivation for machine learning in chemical computers and a mathematical explanation for how biological cells might exhibit more efficient learning behavior within biochemical reaction networks than neuronal networks.",
      "published": "2026-03-12T15:27:33Z",
      "abstract_url": "http://arxiv.org/abs/2603.12060v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12060v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.ST",
        "stat.ML"
      ]
    },
    {
      "title": "Slow-Fast Inference: Training-Free Inference Acceleration via Within-Sentence Support Stability",
      "authors": [
        "Xingyu Xie",
        "Zhaochen Yu",
        "Yue Liao",
        "Tao Wang",
        "Kim-Chuan Toh",
        "Shuicheng Yan"
      ],
      "abstract": "Long-context autoregressive decoding remains expensive because each decoding step must repeatedly process a growing history. We observe a consistent pattern during decoding: within a sentence, and more generally within a short semantically coherent span, the dominant attention support often remains largely stable. Motivated by this observation, we propose Slow-Fast Inference (SFI), a training-free decoding framework that decouples generation into frequent low-cost fast steps and occasional dense-attention slow steps. Fast steps reuse a compact sparse memory for efficient decoding. Slow steps are triggered near semantic boundaries. At slow steps, the model revisits the broader context and uses the Selector to refresh the selected memory for subsequent fast steps. Across the evaluated context lengths, SFI delivers approximately $1.6\\times$--$14.4\\times$ higher decoding throughput while generally maintaining quality on par with the full-KV baseline across long-context and long-CoT settings. Because SFI is training-free and applies directly to existing checkpoints, it offers a practical path to reducing inference cost for contemporary autoregressive reasoning models in long-context, long-horizon, and agentic workloads.",
      "published": "2026-03-12T15:14:48Z",
      "abstract_url": "http://arxiv.org/abs/2603.12038v1",
      "pdf_url": "https://arxiv.org/pdf/2603.12038v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
