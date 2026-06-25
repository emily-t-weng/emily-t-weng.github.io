const PAPERS_DATA = {
  "last_updated": "2026-06-25 04:13:23 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity",
      "authors": [
        "Andrei Liviu Nicolicioiu",
        "Mohammad Pezeshki",
        "Aaron Courville"
      ],
      "abstract": "On-policy self-distillation achieves strong pass@1 accuracy by using a single model as both teacher and student, with the teacher conditioned on a correct demonstration to provide dense token-level feedback. We show that this could come at a hidden cost: rollout diversity decreases and pass@k curves flatten (i.e., generating more rollouts fails to improve accuracy). We trace this to compounding biases in the design of self-distillation with sampled demonstrations. The teacher scores each student rollout while conditioned on a sampled correct rollout, channeling its feedback through the model's own biases. We theoretically analyze the optimal self-distillation policy and show that it tilts the base distribution by a pointwise conditional mutual information score between the student's rollout and the correct rollout used as context. Unlike the ideal optimal on-policy reinforcement learning (RL), which preserves probability ratios among equally correct rollouts, self-distillation can amplify existing probability gaps, concentrating mass on already-dominant modes. On a controlled graph path-finding task and science question-answering benchmarks, self-distilled models match or exceed RL on average performance but exhibit substantially lower functional and semantic diversity, failing on out-of-distribution settings that require diverse strategies.",
      "published": "2026-06-24T17:59:02Z",
      "abstract_url": "http://arxiv.org/abs/2606.26091v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26091v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents",
      "authors": [
        "Changdae Oh",
        "Wendi Li",
        "Seongheon Park",
        "Samuel Yeh",
        "Tanwi Mallick",
        "Sharon Li"
      ],
      "abstract": "Process reward models enable fine-grained, step-level evaluation of LLMs, yet building them for agentic settings remains prohibitively difficult: long-horizon interactions, irreversible actions, and stochastic environment feedback make both human annotation and Monte Carlo estimation infeasible at scale. In this work, we show that reinforcement learning (RL) post-training already provides the ingredients for effective step-level scoring, eliminating the need for dedicated reward model training altogether. Concretely, we derive an implicit advantage under a general stochastic Markov decision process, which we term progress advantage -- log-probability ratio between the RL-trained policy and its reference policy exactly recovers the optimal advantage function. This formulation makes the resulting signal annotation-free, domain-agnostic, and available as a byproduct of the standard RL post-training pipeline. We validate the effectiveness of the progress advantage across three different applications: test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families. Across all settings, it consistently outperforms confidence-based baselines and, despite requiring no task-specific training, surpasses dedicated trained reward models. We complement these results with deeper analyses on characteristics of progress advantage, offering practical guidance for adoption in real-world agentic systems.",
      "published": "2026-06-24T17:54:08Z",
      "abstract_url": "http://arxiv.org/abs/2606.26080v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26080v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment",
      "authors": [
        "Aditya Singh",
        "Gerson Kroiz",
        "Senthooran Rajamanoharan",
        "Neel Nanda"
      ],
      "abstract": "A central goal of safety research is determining whether a model is misaligned. Prior work has largely focused on detecting concerning behavior. But behavior alone does not establish misalignment: a concerning action can arise from benign causes such as confusion. This motivates model forensics: investigating whether the action was driven by malign intent. In this paper, we propose a baseline protocol for model forensics consisting of two steps, iterated as needed. First, we read the chain of thought (CoT) to generate hypotheses about what drives model behavior. Second, we make edits to the prompt or environment to test these hypotheses. While the CoT is not always faithful, it is a rich source of unsupervised insight that can guide the collection of more rigorous evidence. To evaluate our protocol, we create a suite of six agentic environments where models exhibit concerning behavior, and apply it to each. We establish that Kimi K2 Thinking takes shortcuts due to a genuine disposition towards low-effort actions, by showing this hypothesis successfully predicts its behavior. Through counterfactual experiments, we show DeepSeek R1 deceives out of a desire to be consistent with a previous instance of itself. Our methods nonetheless leave significant room for refinement. For example, when we test whether Kimi K2 Thinking believes it is violating user intent, we find no evidence of such a belief, but without positive controls we cannot confirm our tests would detect it. Overall, we find our simple protocol provides a strong baseline that we hope future work will improve upon. More broadly, our work is a concrete step in developing the growing field of model forensics.",
      "published": "2026-06-24T17:45:47Z",
      "abstract_url": "http://arxiv.org/abs/2606.26071v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26071v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems",
      "authors": [
        "Seth Dobrin",
        "Łukasz Chmiel"
      ],
      "abstract": "AI agents are granted access to tools, APIs, and other infrastructure, making them active principals in those systems. The dominant approach places controls inside the agent's own runtime: system prompts, output filters, and guardrail libraries. Any control in the agent's address space is reachable by inputs that influence it; this generalizes to any AI system with sufficient reach into its own runtime, a class we term escapable AI systems. We identify four properties that an authorization mechanism must satisfy for architectural control rather than for cooperative requests: process separation, pre-action enforcement on a structurally only path, fail-closed at both the request and system levels, and externalized signed evidence verifiable outside the controlled system's trust boundary. We position this layer as execution-time AI alignment, complementing training-time alignment (RLHF, Constitutional AI) and inference-time alignment. We present the Unfireable Safety Kernel, a Rust reference implementation realizing all four. Its fail-closed invariant is machine-checked at two levels: an SMT theorem (Z3) and an exhaustive bounded-model-checking proof of the production decision function (Kani, 4/4 harnesses). A Python-to-Rust migration was gated on byte-equivalence (1000/1000 fixtures; 17/17 adversarial classes). We evaluate the kernel governing a live, escapable AI system, a deterministic, self-improving world model, against an escape-seeking adversary driving its real self-modification seam: across 1,000 self-modifications, all 704 attempts on the safety-critical core are refused, with no escape; a further 300, under the operator kill switch, are also refused. A separate campaign of 6,240 authorization round-trips had no successful bypass. Against 3 contemporary systems claiming the agent control plane, the agent invokes control; here, it lacks that choice.",
      "published": "2026-06-24T17:32:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.26057v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26057v1",
      "categories": [
        "cs.AI",
        "cs.CR",
        "cs.LG"
      ]
    },
    {
      "title": "Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining",
      "authors": [
        "Juliana Li",
        "Diya Sreedhar"
      ],
      "abstract": "Midway through an ordinary pretraining run, a small language model learns the pronoun-gender rule: cued with a girl's name (\"Sue cried because\"), it resolves the next pronoun to she, generalizing to held-out probes (0.94 by step 925). By step 3,500 the same model scores near zero on the same probes, although the rule's evidence is still in the training data. We call this within-run reversal natural ungrokking: the corpus decides, with no trace in the loss curve, which learned rules a model keeps. Which rules survive is predictable from one corpus statistic: how often the training stream shows the rule winning. Across un-intervened runs (two corpora, three budgets, three seeds), support frequency decides a rule's fate; the data-to-parameter ratio only modulates how deeply a doomed rule falls. The same emerge-then-collapse dynamics appear in public Pythia checkpoints, collapse depth ordered by model scale as predicted. The forgetting is a displacement: a competing surface pattern out-competes the rule, and the log-probability margin between them crosses zero within 100 training steps of the behavioral collapse. Control over this fate is asymmetric: the same edit that destroys a rule on demand cannot restore it. Flipping support to counter-evidence in place kills the rule with monotone dose-response in two unrelated rules; but injecting support back, even to 450 times the level that naturally sustains it, buys no recovery. Every confirmatory threshold and prediction was pre-registered before the data it governed was read.",
      "published": "2026-06-24T17:27:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.26050v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26050v1",
      "categories": [
        "cs.LG",
        "cond-mat.dis-nn",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "TriViewBench: Controlled Complexity Scaling for Multi-View Structural Reasoning in MLLMs",
      "authors": [
        "Yu-Yang Chen",
        "Lan-Zhe Guo"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) demonstrate strong performance on standard visual question answering benchmarks, yet their scalability under controlled structural complexity remains poorly understood. We introduce TriViewBench, a controlled three-view visual reasoning benchmark constructed from synthetic 3D scenes with explicitly parameterized object count and occlusion. The benchmark contains 1,923 scenes and over 14K Question-Answer (QA) pairs organized into four complexity levels and three reasoning categories: Local Decision, Object Counting, and Global Recovery. We evaluate 18 open- and closed-source MLLMs under a unified prompting protocol. All 18 models exhibit an identical capability hierarchy without exception (Local Decision > Object Counting > Global Recovery), and performance degrades monotonically with complexity: Local Decision tasks decline modestly (12.11% relative drop), while Object Counting degrades substantially (59.14%) and Global Recovery collapses severely (80.02%). Error analysis on Object Counting reveals two mechanistically independent failure modes: single-view tasks are dominated by undercounting due to occlusion blindness, whereas the multi-view task reverses to overcounting due to cross-view identity confusion. Chain-of-Thought (CoT) prompting yields near-zero overall benefit ($Δ= -0.16\\%$) and its effect on Global Recovery is strongly capability-gated, suggesting that the bottleneck lies in cross-view spatial representation rather than reasoning strategy. These findings reveal fundamental scalability limitations in current MLLMs and position TriViewBench as a controlled diagnostic framework for analyzing structural reasoning failures.",
      "published": "2026-06-24T17:00:05Z",
      "abstract_url": "http://arxiv.org/abs/2606.26029v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26029v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization",
      "authors": [
        "Kamar Hibatallah Baghdadi",
        "Kawther Guoual Belhamidi",
        "Sara Belhadj",
        "Aissa Boulmerka",
        "Nadir Farhi"
      ],
      "abstract": "We present HiReLC, a hierarchical ensemble-reinforcement learning framework for automated joint quantization and structured pruning of deep neural networks. The framework decomposes the compression search across two levels of abstraction: low-level agents (LLAs) operate independently per block, selecting per-kernel configurations over a multi-discrete action space spanning bitwidth, pruning keep-ratio, quantization type, and granularity, while high-level agents (HLAs) coordinate global budget allocation via ensemble voting guided by Fisher Information-based sensitivity estimates. To mitigate the computational cost of policy evaluation, an iterative active learning loop interleaves surrogate-guided RL optimization with post-compression fine-tuning, using a lightweight MLP surrogate to amortize expensive evaluations and a logit-MSE proxy during cold-start. The surrogate is used for reward shaping rather than as a replacement for final post-compression evaluation. The controller is architecture-agnostic by design, with a modular layer abstraction decoupling the RL environment from the underlying network topology. Experiments across Vision Transformer and CNN benchmarks demonstrate effective parameter-storage compression ratios of 5.99 - 6.72$\\times$ with a 3.83 % gain in one setting and 0.55 - 5.62 % accuracy drops elsewhere, supporting hierarchical policy decomposition and sensitivity-aware guidance as practical design choices for joint neural network compression.",
      "published": "2026-06-24T16:19:29Z",
      "abstract_url": "http://arxiv.org/abs/2606.26002v1",
      "pdf_url": "https://arxiv.org/pdf/2606.26002v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.OC"
      ]
    },
    {
      "title": "Autodata: An agentic data scientist to create high quality synthetic data",
      "authors": [
        "Ilia Kulikov",
        "Chenxi Whitehouse",
        "Tianhao Wu",
        "Yixin Nie",
        "Swarnadeep Saha",
        "Eryk Helenowski",
        "Weizhe Yuan",
        "Olga Golovneva",
        "Jack Lanchantin",
        "Yoram Bachrach",
        "Jakob Foerster",
        "Xian Li",
        "Han Fang",
        "Sainbayar Sukhbaatar",
        "Jason Weston"
      ],
      "abstract": "We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.",
      "published": "2026-06-24T16:08:31Z",
      "abstract_url": "http://arxiv.org/abs/2606.25996v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25996v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Weave of Formal Thought",
      "authors": [
        "Alexandre Bouayad"
      ],
      "abstract": "Large language models (LLMs) attain remarkable surface fluency on code, yet they neither formally guarantee the syntactic validity of their output nor leverage the hierarchical structure defining the target language. While existing constrained-decoding frameworks address the former, they operate under rigid assumptions that preclude critical lexical mechanisms -- including context-sensitive lexing, maximal-munch tokenization, and keyword extraction -- and only approximate vocabulary masking, sacrificing completeness. For the latter, code LLMs typically inject grammatical structure via predetermined policies rather than learning which structural information to expose. In this work, we introduce Weave of Formal Thought (WoFT), a paradigm uniting rigorous syntactic validation with learned structural representations. First, we present a formal engine and constrained decoder that is sound and complete with respect to the full Tree-sitter specification. By augmenting generalized LR (GLR) parsing with a speculative-lexing construction that maintains concurrent lexer-state hypotheses synchronized with a GLR graph-structured stack, our decoder admits every subword token extending to a valid program prefix and rejects all others. Second, we present a latent-variable fine-tuning method training the language model to interleave non-terminal grammar symbols directly into generation. Utilizing the reweighted wake-sleep (RWS) algorithm to optimize the importance-weighted evidence lower bound (IW-ELBO) of the surface text, the model learns to selectively retain formal derivations as an adaptive structural scratchpad. For Python, fine-tuning StarCoder2-3B with our RWS objective reduces per-token cross-entropy by 14.3% relative to a text-only SFT baseline, demonstrating that discretionary latent syntax recovers critical structural information that flat autoregressive training discards.",
      "published": "2026-06-24T15:58:11Z",
      "abstract_url": "http://arxiv.org/abs/2606.25987v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25987v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "InvestPhilBench: A Multi-Layer Dynamic Benchmark for Evaluating Large Language Model Procedural Reasoning in Expert Investment Philosophy",
      "authors": [
        "Mingguang Chen",
        "Bo Qu"
      ],
      "abstract": "Large language models are increasingly deployed as investment research assistants, yet no benchmark tests whether they can accurately reconstruct and apply the specific procedural decision frameworks of expert investors. We introduce InvestPhilBench, a multi-layer dynamic benchmark spanning eight cognitive tiers, from principle identification (L1) to novel framework extrapolation (L8). The v0.6 release comprises 118 primary-source-verified investment principle cards, 25 decision framework cards with explicit topology metadata, and 243 QA questions (197 dev / 46 held-out test). For reproducible scoring at scale we introduce the Benchmark Automated Scoring Pipeline (BASP) -- five algorithmic metrics (OGRS, KCCS, SAP@k, IVP, CKCA) -- the Failure Mode Detection Protocol (FMDP) with computable rules for six failure modes, and Gate Reconstruction Accuracy (GRA), a per-gate metric for questions with gold reasoning programs. In this release, InvestPhilBench is primarily a benchmark-and-methodology contribution. A four-model sanity wave on the 188-question development split shows a sharp provider-tier split (BASP 0.906 vs. 0.438); these mixed-judge numbers are confounded upper bounds. The central finding: the BASP composite saturates at the frontier (Claude L4 = 0.932) while GRA still exposes a procedural deficit (frontier L4 GRA approx. 0.77, L7 GRA 0.57-0.62) -- composite scoring rewards fluent prose and hides the procedural gap. v0.6 implements a unified judge and true model-in-the-loop retrieval/oracle conditions; the de-confounded multi-model leaderboard and full three-condition run are v1.0 deliverables. On a 100-item expert-annotated gold set the automated BASP composite tracks the human reference at Pearson r = 0.72 (MAE = 0.10), with attribution (SAP@3) the weakest sub-metric and the failure-mode detector running sensitive-but-over-flagging.",
      "published": "2026-06-24T15:53:20Z",
      "abstract_url": "http://arxiv.org/abs/2606.25984v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25984v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound",
      "authors": [
        "Thiago Thomas",
        "Gabriel de Oliveira Ramos",
        "Felipe Meneguzzi"
      ],
      "abstract": "Multi-agent goal recognition asks an observer to jointly infer which agents act together and what each team is trying to achieve, so the hypothesis space grows combinatorially with the number of team partitions and goals per team. Real applications such as drone surveillance and collaborative robotics expose only the agents' trajectory, which forces the observer to rank team-goal hypotheses from behavior alone. Multi-Agent Goal Recognition with Branch-and-Bound (MAGR-BB) addresses this setting with a shared team- and goal-conditioned policy used as the scoring model inside a factorized branch-and-bound search. On a controlled multi-agent Blocksworld benchmark, MAGR-BB returns the same top-ranked hypothesis as exhaustive search throughout the trajectory while cutting hypothesis materialization by orders of magnitude and reducing cumulative recognition runtime substantially.",
      "published": "2026-06-24T15:50:39Z",
      "abstract_url": "http://arxiv.org/abs/2606.25978v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25978v1",
      "categories": [
        "cs.MA",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Helpful or Harmful? Evaluating LLM-Assisted Vulnerability Patching via a Human Study",
      "authors": [
        "Giulian Biolo",
        "Michael Tezza",
        "Yuanjun Gong",
        "Fabio Massacci"
      ],
      "abstract": "Software vulnerability remediation is a cognitively demanding task that requires specialized security expertise often lacking in general developers. In the meantime, Large Language Models (LLMs) assisted tools show potential in vulnerability detection, location, and repair tasks. [Hypothesis:] While LLM-assistance is hypothesized to accelerate patching, it also risks introducing hallucinations or insecure code, leading to a higher likelihood of generating superficial repairs that bypass the standard functionality checks but fail the security validation. [Objective:] We aim to present an empirical experiment, unveiling the capability of LLM-assisted vulnerability patching compared to manual debugging on human participants in real-world scenarios. [Method:] We plan to conduct a controlled experiment using a Balanced Crossover design. For that, we have developed a WebApp for code execution and integrated hidden Ghost Tests to verify patch integrity beyond visible functional requirements. The experiment involves training and evaluation scenarios. The remediation speed, remediation efficacy for both standard functionality tests and security tests, and participant perception will be evaluated. [Pilot Study:] A pilot experiment with a small sample of participants has been conducted, providing insights for the following study.",
      "published": "2026-06-24T15:45:38Z",
      "abstract_url": "http://arxiv.org/abs/2606.25973v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25973v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "WinDOM: Self-Family Distillation for Small-Model GUI Grounding",
      "authors": [
        "Chengheng Li-Chen",
        "Zhiqian Zhou",
        "Hao Chen",
        "Nicolas Chauvin"
      ],
      "abstract": "Small ($\\sim$2B) GUI-grounding agents are attractive for on-device deployment, accessibility tooling, and low-cost iteration, but at this scale they face two open recipe questions: how to obtain bounding-box training data without expensive human annotation, and how to combine supervised fine-tuning with reinforcement learning. We address both, with the explicit goal of pushing small-model performance rather than scaling up. WinDOM is a $54{,}425$-record grounding corpus harvested by driving an open-source Windows 11 web reimplementation under headless Playwright, with bounding boxes read directly off the DOM and no OCR or human annotation. Self-Family Distillation (SFD) is a single rejection-sampling cold-start parameterised only by the teacher choice: either an EMA of the student (no external model) or a frozen larger same-family teacher. We then treat the saturation depth of the SFD cold-start as an explicit GRPO hyperparameter. On a Qwen3.5-2B student, the under-saturated cold-start is a better GRPO initialiser than the converged one: SFD-4B with Early-init RL gains $+5.4$ OOD-mean ($+3.5$ ScreenSpot-Pro, $+7.0$ OSWorld-G, $+5.8$ ScreenSpot-V2) over the base. The same-size EMA mode lands within roughly one OOD-mean point of the cross-size $4$B variant ($65.2$ vs $66.3$) without an external teacher.",
      "published": "2026-06-24T15:35:29Z",
      "abstract_url": "http://arxiv.org/abs/2606.25964v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25964v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Agentic System as Compressor: Quantifying System Intelligence in Bits",
      "authors": [
        "Zihan Qin",
        "Hongrui Zhang"
      ],
      "abstract": "Large language models are turning from isolated predictors into agentic systems: they call tools, retrieve evidence, obey environment constraints, use verifiers, and complete tasks through search and multi-turn interaction. We adopts an analytical viewpoint based on \"compression is intelligence\": under a fixed task distribution, interface, and compute budget, a stronger agentic system lets a target object be reconstructed with fewer bits. We operationalize the measure with arithmetic coding, seed coding, and a fallback, and evaluate it in five settings: reversed text, chess moves, protein sequences, retrieval-augmented question answering, and semantic story compression; in all of them agentic components reduce codelength. These small, controlled experiments cover component types typical of real agentic systems, show that codelength can analyze how components, observers, and budgets change residual uncertainty, and offer guidance for evaluating real agent systems.",
      "published": "2026-06-24T15:32:49Z",
      "abstract_url": "http://arxiv.org/abs/2606.25960v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25960v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Pulmonary Embolism Risk Stratification from CTPA and Medical Records: Vascular Graphs Are Not All You Need",
      "authors": [
        "Nathan Painchaud",
        "Tristan Habémont",
        "Morgane des Ligneris",
        "Allan Serva",
        "Pierre Croisille",
        "Laurent Bertoletti",
        "Thomas Lampert",
        "Johannes F. Lutzeyer",
        "Odyssée Merveille"
      ],
      "abstract": "Risk stratification for pulmonary embolism (PE) is critical for clinical decision-making. Stratification guidelines are based on patient medical records, parameters measured from computed tomography pulmonary angiography (CTPA), and blood tests. However, blood tests are often missing in routine practice. This work studies whether state-of-the-art models can accurately classify risk stratification from only medical records and biomarkers extracted from CTPA images. We benchmark different approaches to combine medical records and cardiac biomarkers with rich pulmonary vascular information; we add vascular biomarkers to tabular models and apply graph neural networks (GNNs) on the vascular tree's intrinsic graph representation. We use a private dataset (n=353) with uniquely complete data for PE risk stratification. Our results show that, among global features, medical records and cardiac biomarkers are the most significant predictors, while vascular biomarkers do not further improve stratification. Even more surprising, even GNNs on vascular graphs fail to outperform strong tabular baseline on global features. We consider hypotheses, on both models and data, that could explain this suboptimal performance. Our investigation suggests that, counter-intuitively, vascular graphs might hold no discriminative information for PE risk stratification. Code is available from https://github.com/creatis-myriad/GENESIS.",
      "published": "2026-06-24T15:28:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.25956v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25956v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "eess.IV"
      ]
    },
    {
      "title": "Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interface",
      "authors": [
        "Faliang Yin",
        "Hak-Keung Lam",
        "David Watson"
      ],
      "abstract": "Increasing demand for precise and reliable control in complex scenarios has led to the development of increasingly sophisticated controllers, including data-driven approaches employing closed box models and mathematically rigorous yet complex designs. This complexity highlights the needs for explainable control that can provide human-understandable insights into controller behavior. In this paper, an explainable control framework (XCF) along with supporting algorithms and user interface are proposed to explain how controllers determine their control actions and their underlying working mechanism. The novel contributions of this work are threefold: First, the XCF is designed to provide model-agnostic explanations for controllers in closed-loop systems and can optionally refine local explanations by system response dynamics. Second, a novel explanation method, hierarchical fuzzy model-agnostic explanation for control systems (HFMAE-C), is proposed based on the designed framework. The HFMAE-C employs a fuzzy logic system to approximate the controller's behavior and system dynamics, providing sample, local, domain and universe level explanations via IF-THEN rules revealing the controller's decision logic and salience values quantifying the contribution of system states to control actions. Third, a large language model agent-supported user interface is developed to automatically analyze user requirements, select appropriate algorithms, interpret the generated explanations to a natural language report, and provide interactive consultation. Case studies on inverted pendulum system and Turtlebot obstacle avoidance demonstrate the effectiveness of the proposed method through simulated user experiments and quantitative comparisons with mainstream explainable control approaches.",
      "published": "2026-06-24T15:17:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.25941v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25941v1",
      "categories": [
        "cs.HC",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts",
      "authors": [
        "Juri Opitz",
        "Maud Ehrmann",
        "Corina Raclé",
        "Andrianos Michail",
        "Matteo Romanello",
        "Simon Clematide"
      ],
      "abstract": "Was this person ever at that place, and if so, when? Answering such questions from noisy, multilingual historical documents is the central challenge of HIPE-2026, the third edition of the HIPE evaluation series. Moving from named entity recognition and linking (HIPE-2020, HIPE-2022) to reasoning about relationships between entities, HIPE-2026 targets two temporally grounded relation types: $at$, indicating that a person was present at a location at some point prior to a document's publication date, and $isAt$, indicating presence contemporaneous with that date. This paper presents the results of the evaluation campaign, which confronted 17 participating teams with the challenges of historical language variation, OCR noise, and indirect contextual cues across three languages: French, German, and English. The datasets include historical newspaper text from the nineteenth and twentieth centuries, as well as a surprise-domain generalization set drawn from early modern French literary texts. A distinctive feature of HIPE-2026 is its three-fold evaluation framework, which assesses predictive accuracy, computational efficiency, and cross-domain generalization, reflecting the practical demands of large-scale historical document processing in the cultural heritage domain. Across more than 40 submitted runs, results reveal a wide range of strategies, from state-of-the-art large language models to lightweight task-specific classifiers, and highlight the trade-offs between accuracy, efficiency, and robustness inherent to historical relation extraction at corpus scale. System descriptions, datasets, and findings are presented and discussed, offering a detailed picture of the current state of temporally grounded relation extraction for historical documents.",
      "published": "2026-06-24T15:09:12Z",
      "abstract_url": "http://arxiv.org/abs/2606.25935v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25935v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "AI-Assisted Computational Reproducibility on the FABRIC Testbed",
      "authors": [
        "Komal Thareja",
        "Paul Ruth",
        "Berent Aldikacti",
        "Michael Zink"
      ],
      "abstract": "Computational reproducibility remains difficult despite being central to scientific research. In this paper, we show how the international FABRIC testbed, combined with large language model (LLM) coding assistants through LoomAI, can simplify reproducing published experiments across multiple domains. We reproduced three case studies on FABRIC, covering BBR-family congestion-control evaluations, LAMMPS molecular dynamics scaling benchmarks on a CPU-only MPI cluster, and stress protein homeostasis genomics pipelines. Rather than focusing only on matching numerical outputs, we evaluate whether the reproduced experiments support the same scientific conclusions as the original studies. The AI assistant was effective in setting up the environment, adapting code, and debugging, but struggled with the analysis stages that lacked clearly defined workflows, which required human guidance to establish execution order and data dependencies. Across the case studies, the AI-assisted workflow reduced reproduction effort by roughly 4--6 times. We conclude with practical recommendations for improving AI-assisted reproducibility on research testbeds.",
      "published": "2026-06-24T14:23:14Z",
      "abstract_url": "http://arxiv.org/abs/2606.25879v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25879v1",
      "categories": [
        "cs.DC",
        "cs.AI"
      ]
    },
    {
      "title": "Color Matters: Trigger Color Affects Success in Federated Backdoor Attacks",
      "authors": [
        "Kavindu Herath",
        "Joshua C. Zhao",
        "Saurabh Bagchi"
      ],
      "abstract": "Federated learning is vulnerable to backdoor attacks in which malicious clients inject poisoned updates while preserving benign-task performance. In this paper, we study a semantics-driven backdoor mechanism in which attackers use natural visual accessories as triggers and manipulate only the trigger color while keeping the attack pipeline fixed. Our framework considers semantic trigger objects such as masks and sunglasses, instantiated in black and white variants, and evaluates their effect in a controlled federated learning setting. Malicious clients construct poisoned samples by applying a trigger to source-class images and relabeling them to an attacker-chosen target class, while benign clients train only on clean data. We analyze this mechanism under both a standard poisoning objective and a stronger SABLE-based objective that combines clean classification loss, triggered target loss, feature-separation loss in the penultimate representation space, and regularization to keep malicious updates close to the global model. This design enables the attack to remain effective while reducing excessive update drift. Experiments on a four-class CelebA hair-color task show that trigger color significantly changes attack success rate even when trigger semantics, placement, and poisoning budget are unchanged. White triggers are more effective for attacks targeting the blond class, whereas black triggers perform better for attacks targeting the black class. The same trend persists under robust aggregation, showing that trigger color is a meaningful factor in the operation, persistence, and evaluation of semantic backdoor mechanisms in federated learning.",
      "published": "2026-06-24T14:07:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.25858v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25858v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents",
      "authors": [
        "Peng Xu",
        "Sijia Chen",
        "Junzhuo Li",
        "Xuming Hu"
      ],
      "abstract": "Group-based reinforcement learning effectively post-trains LLM agents for long-horizon, sparse-reward tasks by deriving step-level credit from trajectory outcomes. However, this ties a step's credit to its rollout's final outcome: semantically near-identical intermediate steps receive opposite credit depending on whether their trajectory eventually succeeded or failed. Such semantic credit inconsistency sends conflicting gradients to similar actions and wastes the partially-correct progress inside failed rollouts. Motivated by this, we propose Semantic Consistency Policy Optimization (SCPO), a value-free reward-shaping method that mitigates this inconsistency by recovering step-level credit from successful siblings in the same rollout group. Concretely, SCPO scores each failed step against a successful sibling and adds positive step-level credit for new progress along that sibling. On ALFWorld and WebShop, SCPO matches or exceeds strong group-based baselines, reaching 93.7+/-4.1 percent success on ALFWorld and 74.8+/-2.0 percent on WebShop at 1.5B parameters, with gains concentrated on the hardest multi-step tasks.",
      "published": "2026-06-24T14:02:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.25852v1",
      "pdf_url": "https://arxiv.org/pdf/2606.25852v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
