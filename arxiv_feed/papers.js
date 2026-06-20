const PAPERS_DATA = {
  "last_updated": "2026-06-20 04:20:35 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "How Transparent is DiffusionGemma?",
      "authors": [
        "Joshua Engels",
        "Callum McDougall",
        "Bilal Chughtai",
        "Janos Kramar",
        "Senthoran Rajamanoharan",
        "Cindy Wu",
        "Arthur Conmy",
        "Asic Q Chen",
        "Jean Tarbouriech",
        "Min Ma",
        "Brendan O'Donoghue",
        "João Gabriel Lopes de Oliveira",
        "Rohin Shah",
        "Neel Nanda"
      ],
      "abstract": "LLM reasoning transparency is a critical affordance for understanding model decisions, mitigating misuse and misalignment, and debugging surprising model behaviors. However, DiffusionGemma performs a larger fraction of its computation in a continuous latent space; does this make its reasoning less transparent? We study this question by decomposing transparency into two components: variable transparency, whether we understand intermediate snapshots of a model's computational state; and algorithmic transparency, whether we can use these snapshots to reconstruct the process by which the model arrived at its outputs. Naively, DiffusionGemma has poor variable transparency: its opaque serial depth, the amount of serial computation that occurs in between interpretable model states, seems at first 28.6X higher than the corresponding autoregressive Gemma 4 model. However, we show that we can map the information flowing between denoising steps through an interpretable token bottleneck with no decrease in downstream performance. Treating these intermediate states as interpretable reduces the opaque serial depth to just 1.1X that of Gemma 4. Algorithmic transparency is harder for diffusion models than for autoregressive models because all token predictions in the canvas can change at every denoising step, giving the model the power to implement complicated distributed algorithms during the denoising process. To begin bridging this gap, we conduct a suite of interpretability case studies, uncovering initial evidence of novel diffusion-specific phenomena such as non-chronological reasoning, token and sequence smearing, and intermediate-context reasoning. Finally, we test monitorability, a key application of transparency that measures whether model outputs are useful for downstream tasks. We find that DiffusionGemma is similarly monitorable to Gemma 4.",
      "published": "2026-06-18T17:59:46Z",
      "abstract_url": "http://arxiv.org/abs/2606.20560v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20560v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Toward Calibrated Mixture-of-Experts Under Distribution Shift",
      "authors": [
        "Gina Wong",
        "Drew Prinster",
        "Suchi Saria",
        "Rama Chellappa",
        "Anqi Liu"
      ],
      "abstract": "Calibration aligns a model's predictive uncertainty with the frequencies of its empirical outcomes and is important for understanding and trusting reported probabilities. Recent work shows that enforcing calibration at the level of individual predictors can improve ensemble accuracy and calibration, with mixture-of-experts (MoE) models showing strong empirical improvements in particular; however, the conditions under which calibration helps MoE are not well understood. In this work, we study how MoE models behave under distribution shift, focusing on how routing mechanisms interact with expert-level calibration. We show that expert calibration is sufficient to ensure calibration of the overall model under a broad class of distribution shifts in hard-routed models, but is insufficient for calibrating soft-routed models. To address this, we propose an adversarial reweighting that penalizes calibration errors of the routed aggregate under distribution shift, and we demonstrate that it improves the accuracy-calibration tradeoff both on average and on difficult subsets of the data, across model classes, prediction tasks, and distribution shifts.",
      "published": "2026-06-18T17:55:00Z",
      "abstract_url": "http://arxiv.org/abs/2606.20544v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20544v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Sovereign Execution Brokers: Enforcing Certificate-Bound Authority in Agentic Control Planes",
      "authors": [
        "Jun He",
        "Deying Yu"
      ],
      "abstract": "Autonomous agents are increasingly connected to cloud, deployment, and data-control workflows, but production mutation authority should not reside inside non-deterministic reasoning processes. Existing access-control mechanisms authorize identities, while assurance layers certify proposed actions; neither alone provides a mandatory enforcement point for certified authority at the moment of mutation. This paper introduces the Sovereign Execution Broker (SEB), a runtime enforcement boundary for certificate-bound agentic infrastructure. SEB consumes certificates issued by the Sovereign Assurance Boundary (SAB), verifies that the requested mutation matches the certified execution contract, checks validity windows, policy epochs, revocation epochs, and live-state drift, mints scoped execution identity, invokes infrastructure APIs, and records signed decision and outcome records. By separating proposal, admission, and execution, SEB turns certified authority into a short-lived, revocable, auditable runtime capability, provided that production mutation APIs reject non-broker identities. We present the SEB execution model, certificate and replay-verification predicates, scoped identity semantics, bypass-prevention deployment patterns, failure behavior, and a concrete prototype implementation. We evaluate the prototype on AWS and Kubernetes clusters, measuring latency overheads, revocation propagation, drift detection, and security under fault injection.",
      "published": "2026-06-18T17:36:46Z",
      "abstract_url": "http://arxiv.org/abs/2606.20520v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20520v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.DC",
        "cs.LG"
      ]
    },
    {
      "title": "Multi-LCB: Extending LiveCodeBench to Multiple Programming Languages",
      "authors": [
        "Maria Ivanova",
        "Pavel Zadorozhny",
        "Rodion Levichev",
        "Ivan Petrov",
        "Adamenko Pavel",
        "Ivan Lopatin",
        "Alexey Kutalev",
        "Dmitrii Babaev"
      ],
      "abstract": "LiveCodeBench (LCB) has recently become a widely adopted benchmark for evaluating large language models (LLMs) on code-generation tasks. By curating competitive programming problems, constantly adding fresh problems to the set, and filtering them by release dates, LCB provides contamination-aware evaluation and offers a holistic view of coding capability. However, LCB remains restricted to Python, leaving open the question of whether LLMs can generalize across the diverse programming languages required in real-world software engineering. We introduce Multi-LCB, a benchmark for evaluating LLMs across twelve programming languages, including Python. Multi-LCB transforms Python tasks from the LCB dataset into equivalent tasks in other languages while preserving LCB's contamination controls and evaluation protocol. Because it is fully compatible with the original LCB format, Multi-LCB will automatically track future LCB updates, enabling systematic assessment of cross-language code generation competence and requiring models to sustain performance well beyond Python. We evaluated 24 LLMs for instruction and reasoning on Multi-LCB, uncovering evidence of Python overfitting, language-specific contamination, and substantial disparities in multilingual performance. Our results establish Multi-LCB as a rigorous new benchmark for multi-programming-language code evaluation, directly addressing LCB's primary limitation and exposing critical gaps in current LLM capabilities.",
      "published": "2026-06-18T17:35:57Z",
      "abstract_url": "http://arxiv.org/abs/2606.20517v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20517v1",
      "categories": [
        "cs.AI",
        "cs.PL"
      ]
    },
    {
      "title": "What Do Safety-Aligned LLMs Learn From Mixed Compliance Demonstrations?",
      "authors": [
        "Sihui Dai",
        "Mann Patel"
      ],
      "abstract": "Prior work has shown that in-context demonstrations can jailbreak language models, but it remains unclear how models interpret different types of compliance demonstrations. We study this by mixing benign compliance demonstrations (non-harmful request, helpful response) with harmful compliance demonstrations (harmful request, helpful response) and testing three hypotheses about how demonstration composition drives harmful compliance. Across four models, we find that benign and harmful demonstrations are not interchangeable: benign demonstrations can either reduce or increase harmful compliance depending on the model. We further show that preference optimization is the critical training stage that prevents benign demonstrations from increasing harmful compliance, that demonstration ordering exhibits strong recency bias, and that models differ in how refusal interacts with in-context learning: some adopt demonstrated formatting even when refusing, while others override all in-context signals upon refusal. Taken together, this work moves beyond showing that demonstration-based jailbreaking works to characterizing how it works: what models extract from compliance demonstrations depends on demonstration content, ordering, and training methodology.",
      "published": "2026-06-18T17:25:38Z",
      "abstract_url": "http://arxiv.org/abs/2606.20508v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20508v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Contagion Networks: Evaluator Bias Propagation in Multi-Agent LLM Systems",
      "authors": [
        "Zewen Liu"
      ],
      "abstract": "When large language models serve as evaluators in multi-agent systems, their systematic evaluation biases propagate through the agent network. We introduce Contagion Networks, a formal framework for measuring how evaluator biases spread across interacting LLM agents. In a controlled 3-agent experiment using DeepSeek-chat with three distinct evaluator bias profiles (structured, balanced, evidence-based), we measure the Cross-Agent Contagion Matrix Gamma_3 and find that evaluator biases consistently propagate between agents (gamma in [0.157, 0.352]), even within the same underlying model. We identify three propagation regimes governed by the spectral radius rho(Gamma_N), and demonstrate that homogeneous-model agents produce contagion coefficients 3-5x weaker than cross-model coefficients observed in prior work (MM-EPC: gamma approx 0.85-1.3), placing them in the suppression regime. We show that increasing evaluator committee size from k=1 to k=3 reduces effective contagion by 72.4%, providing an actionable mitigation strategy. We release the open-source Contagion Network experimental framework.",
      "published": "2026-06-18T17:09:34Z",
      "abstract_url": "http://arxiv.org/abs/2606.20493v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20493v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "UltraQuant: 4-bit KV Caching for Context-Heavy Agents",
      "authors": [
        "Inesh Chakrabarti",
        "David Limpus",
        "Aditi Ghai Rana",
        "Bowen Bao",
        "Spandan Tiwari",
        "Thiago Crepaldi",
        "Ashish Sirasao"
      ],
      "abstract": "Context-heavy agents place unusual pressure on the key-value (KV) cache: long prefixes are reused across many short turns, while concurrency determines whether the serving system can keep GPUs utilized. We study 4-bit KV-cache compression for this setting, using TurboQuant-style rotation and codebook quantization as a quality anchor and vLLM FP8 KV caching as the deployment anchor. We report three contributions. First, we frame 4-bit KV caching around multi-round agent workloads where task quality, cache residency, and serving throughput must be measured jointly. Second, we describe the practical design choices needed to make the 4-bit path robust, including asymmetric K/V treatment, Walsh-Hadamard rotation, QJL removal, and block-scale variants. Third, we present serving optimizations on AMD GPUs, including optimized decode-attention kernels and UltraQuant, an FP4 approximation path that uses FP8 queries, FP4 KV tensors, UE8M0 group scales, and native scaled-MFMA support on CDNA4. On a long-context, multi-turn agentic workload, UltraQuant cuts P50 time-to-first-token by 3.47x in the cache-pressured late rounds (2.3x across all rounds) and raises output throughput by 1.63x over the FP8 KV baseline.",
      "published": "2026-06-18T16:54:07Z",
      "abstract_url": "http://arxiv.org/abs/2606.20474v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20474v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.PF"
      ]
    },
    {
      "title": "Repurposing a Speech Classifier for Guided Diffusion-Based Speech Generation",
      "authors": [
        "Rostislav Makarov",
        "Timo Gerkmann"
      ],
      "abstract": "Classifier guidance is a way to control diffusion generation by using a noise-conditioned classifier to steer the sampling process toward a target class. One drawback of classifier guidance is that it requires two separately trained models: a classifier and a diffusion model. We therefore study a more compact alternative in which a conventionally trained speech classifier is repurposed as the backbone for diffusion generation. Starting from a frozen noise-conditioned classifier in log-Mel space, we attach a lightweight subnetwork that reuses intermediate classifier representations and train only this subnetwork under a Denoising Score Matching objective. Our work shows that a pretrained classifier can be repurposed for conditional generation, providing an appealing bridge between discriminative modeling and conditional speech synthesis resulting in high speech quality within a single-backbone model, with reduced memory footprint and computational cost.",
      "published": "2026-06-18T16:40:02Z",
      "abstract_url": "http://arxiv.org/abs/2606.20457v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20457v1",
      "categories": [
        "eess.AS",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Multi-View Decompilation for LLM-Based Malware Classification",
      "authors": [
        "Bercan Turkmen",
        "Vyas Raina"
      ],
      "abstract": "Malware analysts often inspect compiled binaries through decompiled pseudo-C, when source code is unavailable. Recent work suggests that large language models (LLMs) can assist this process by classifying decompiled code as benign or malicious, but existing pipelines typically rely on a single decompiler view. We argue that this assumption is fragile: decompilers are lossy heuristic tools, and different decompilers can expose different artefacts of the same binary. We curate a benchmark of benign utilities and malicious programs spanning a range of threat behaviors. Each sample is compiled and decompiled with both Ghidra and RetDec, yielding matched pseudo-C views. Across a range of LLMs from major model families, we find that providing both decompiler views improves malicious-class F1, mainly by increasing recall on malicious samples. Agreement analyses further show that Ghidra and RetDec make partially different errors, supporting the view that decompiler outputs provide complementary evidence. Our results suggest that multi-decompiler prompting is a simple, training-free way to improve LLM-based malware triage in practical settings.",
      "published": "2026-06-18T16:15:30Z",
      "abstract_url": "http://arxiv.org/abs/2606.20436v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20436v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems",
      "authors": [
        "Hanwool Lee",
        "Dasol Choi",
        "Bokyeong Kim",
        "Seung Geun Kim",
        "Haon Park"
      ],
      "abstract": "Large language model (LLM) agents are increasingly proposed as supervisory components for safety-critical systems, yet their robustness under sustained, adaptive adversarial pressure remains poorly characterized. We present NRT-Bench, a benchmark for multi-turn red-teaming of LLM agents acting as operators of a safety-critical system, instantiated in a simulated nuclear power plant control room. A five-role operator team, each backed by a configurable LLM, runs a plant governed by six critical safety functions (CSFs), while adversaries inject messages over four channels in bounded multi-turn sessions with per-turn feedback. Harm is an objective signal rather than LLM-judged text: a run terminates the moment any CSF is lost, attributed to the causing message. Evaluating four frontier operator models under a fixed-attack paired-replay protocol, we find that adaptive multi-turn attacks reliably push the operator team past a safety limit: across the four models, between 8.7% and 12.1% of attack sessions end with the plant losing a critical safety function. Although the four models look almost equally robust by this aggregate rate, their failures barely overlap: of $149$ sessions, none defeat all four models while a third defeat at least one, so vulnerabilities are nearly disjoint across models rather than nested. The effect of added defences is strongly model-dependent: the same guardrail stack or safety-advisor agent that lowers attack success for one model can raise it for another. We release the simulation venue, attack dataset, and replay tooling for reproducible safety evaluation of LLM agents.",
      "published": "2026-06-18T15:57:53Z",
      "abstract_url": "http://arxiv.org/abs/2606.20408v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20408v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "CRAX: Fast Safe Reinforcement Learning Benchmarking",
      "authors": [
        "Tristan Tomilin",
        "Mourad Boustani",
        "Mickey Beurskens",
        "Thiago D. Simão"
      ],
      "abstract": "Safety is a core concern for deploying reinforcement learning (RL) agents in real-world domains such as robotics and autonomous driving. While benchmarks have been central to progress in RL, existing safety benchmarks with high-fidelity 3D physics remain computationally slow, limiting large-scale experimentation and rapid prototyping. To address this gap, we propose CRAX (Constrained RL Accelerated with JAX). Built on top of the MuJoCo XLA (MJX) physics engine with realistic 3D dynamics, CRAX leverages vectorized operations and hardware acceleration, yielding up to ~100x speedups over comparable CPU-based safety benchmarks. The benchmark features six environment suites and three agent-specific tasks, each spanning three difficulty levels. Evaluating six popular safe RL methods shows that no single approach dominates across all tasks, and reveals the trade-offs between performance and safety. We find that curriculum learning across difficulty levels and safety transfer can improve performance over direct training in harder settings.",
      "published": "2026-06-18T15:36:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.20376v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20376v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning",
      "authors": [
        "Zepeng Li",
        "Jie Ren",
        "Zhanyong Tang",
        "Jie Zheng",
        "Zheng Wang"
      ],
      "abstract": "Large Language Models (LLMs) show promise for code compilation tasks, but applying them to runtime performance tuning is difficult due to complex microarchitectural effects and noisy runtime measurements. We present AutoPass, a multi-agent framework for compiler performance tuning that uses compiler and runtime evidence to guide LLM-generated optimization decisions. Rather than treating the compiler as a black box like prior auto-tuning schemes, AutoPass opens up the compiler to the LLM, enabling it to query compiler-internal optimization states and analyze the intermediate representation to orchestrate compiler options. The search process iteratively refines optimization configurations using measured runtime feedback to diagnose regressions and guide latency-improving edits. AutoPass operates in an inference-only, training-free setting and requires no offline training or task-specific fine-tuning, making it readily applicable to new benchmarks and platforms. We implement AutoPass on the LLVM compiler and evaluate it on server-grade x86-64 and embedded ARM64 systems. AutoPass outperforms expert-tuned heuristics and classical autotuning methods, achieving geometric-mean speedups of 1.043x and 1.117x over LLVM -O3 on x86-64 and ARM64, respectively.",
      "published": "2026-06-18T15:35:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.20373v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20373v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Robust $Q$-learning for mean-field control under Wasserstein uncertainty in common noise",
      "authors": [
        "Mathieu Laurière",
        "Ariel Neufeld",
        "Kyunghyun Park"
      ],
      "abstract": "In this article, we present a robust $Q$-learning algorithm for discrete-time mean-field control problems under Wasserstein uncertainty in the common noise law. The algorithm combines a quantization-and-projection scheme with a Wasserstein dual reformulation on the common-noise space. We establish its convergence together with finite-time iteration bounds for both synchronous and asynchronous learning schemes. Numerical experiments on systemic risk and epidemic models compare the asynchronous implementation with an idealized Bellman iteration, illustrate the robustness-performance tradeoff under common-noise misspecification, and report the observed convergence behavior of the asynchronous $Q$-learning algorithm.",
      "published": "2026-06-18T15:20:00Z",
      "abstract_url": "http://arxiv.org/abs/2606.20356v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20356v1",
      "categories": [
        "math.OC",
        "cs.AI",
        "cs.LG",
        "math.PR",
        "stat.ML"
      ]
    },
    {
      "title": "Boundary Embedding Shaping with Adaptive Contrastive Learning for Graph Structural Disentanglement",
      "authors": [
        "Jiaqing Chen",
        "Zidu Yin",
        "Yichao Cai",
        "Yuhang Liu",
        "Zhen Zhang",
        "Dong Gong",
        "Javen Qinfeng Shi"
      ],
      "abstract": "Graph neural networks (GNNs) excel at aggregating neighbor information for classification, yet their performance is hindered by graph structural entanglement, where spurious correlations from semantically irrelevant neighbors contaminate node embeddings. This challenge is most acute for nodes near class boundaries in the embedding space, where amplified structural noise blurs decision boundaries and destabilizes predictions. Existing robust GNN methods largely treat all nodes uniformly, ignoring boundary vulnerabilities. In this paper, to improve classification performance, we tackle graph structural disentanglement by identifying boundary-region entanglement as the primary bottleneck and propose Boundary Embedding Shaping (BES), an adaptive contrastive learning GNN plug-in module that selectively suppresses spurious structural noise at decision boundaries with minimal model parameter perturbation. Extensive experiments demonstrate that BES consistently improves boundary discrimination and outperforms existing leading methods. Notably, BES boosts GCN performance by an average of 3.3% in node classification (up to 5.0% on WikiCS) and achieves superior accuracy in link prediction.",
      "published": "2026-06-18T14:28:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.20283v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20283v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval",
      "authors": [
        "Yuhan Liu",
        "Pei Fu",
        "Hang Li",
        "Yukun Qi",
        "Chao Jiang",
        "Jingwen Fu",
        "Zhen Liu",
        "Bin Qin",
        "Zhenbo Luo",
        "Jian Luan",
        "Jingmin Xin"
      ],
      "abstract": "Leveraging Multimodal Large Language Models (MLLMs) via contrastive learning has become a mainstream paradigm for improving the performance of Universal Multimodal Retrieval (UMR). However, previous works have ignored the grain blindness when adapting the contrastive paradigm into retrieval tasks. Grain blindness refers to the tendency of the model to overlook grain-level information contained in the query, which is crucial for effectively handling complex queries. This stems from contrastive learning treating samples as a binary classification (positive/negative), while ignoring the different information carried by each negative sample. To address this, we argue that negatives should be treated differently according to their similarity to the positive sample, enabling the model to learn distinct grain information from each negative. In this paper, we introduce a simple but effective framework, called ELVA, a novel rule-based RL framework that mitigates grain blindness through ranking-driven MLLMs. 1) Instead of relying on reward models, we extend Reinforcement Learning with Verifiable Rewards (RLVR) to retrieval tasks, allowing the model to explore new ranking behaviors without explicit ranking labels. 2) By utilizing rule-based rewards, our approach jointly optimizes the ranking of negative samples while enlarging the similarity gap between positive and negative. To more precisely measure grain blindness, we further introduce MRBench, a new benchmark specifically designed for multi-grain query scenarios. ELVA achieves state-of-the-art results across standard retrieval benchmarks, and its notable 13.1% improvement on MRBench further demonstrates its effectiveness in alleviating grain blindness.",
      "published": "2026-06-18T14:23:23Z",
      "abstract_url": "http://arxiv.org/abs/2606.20280v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20280v1",
      "categories": [
        "cs.IR",
        "cs.AI"
      ]
    },
    {
      "title": "Navigating Unreliable Parametric and Contextual Knowledge: Explicit Knowledge Conflict Resolution for LLM Inference",
      "authors": [
        "Huang Peng",
        "Jiuyang Tang",
        "Weixin Zeng",
        "Hao Xu",
        "Xiang Zhao"
      ],
      "abstract": "Large language models (LLMs) have achieved strong performance across a wide range of language-based tasks by leveraging both extensive parametric knowledge and in-context learning ability, enabling them to incorporate external information provided in the input prompt. However, the integration of external knowledge can introduce conflicts, not only between the model's internal parametric knowledge and the external information, but also among multiple pieces of external contexts. Existing approaches typically assume that either the model or the provided context is reliable, overlooking the possibility that both sources may contain errors, and avoid conflicts by privileging one source over the other, rather than actively resolving inconsistencies. To address these limitations, we propose a novel framework MACR for LLM knowledge conflict resolution that moves beyond the conventional binary choice paradigm and incorporates an explicit conflict-resolution mechanism based on a multi-agent reasoning approach. Specifically, we first propose an adaptive knowledge assessment and retrieval approach that employs a modified semantic entropy measure to quantify an LLM's confidence in its answer to a given query. Based on this confidence estimation, MACR either externalizes the model's internal knowledge as textual representations or retrieves relevant external knowledge when internal knowledge is insufficient, generating basic contexts for subsequent reasoning. Then we introduce an inductive multi-agent reasoning framework with three specialized agents that, respectively, induce explicit rules, analyze potential conflicts, and resolve inconsistencies across all available contexts. Empirical results demonstrate that MACR significantly outperforms state-of-the-art baselines across benchmarks, while also providing interpretable resolutions of explicit conflicts.",
      "published": "2026-06-18T13:56:31Z",
      "abstract_url": "http://arxiv.org/abs/2606.20245v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20245v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "A Multi-Agent system for Multi-Objective constrained optimization",
      "authors": [
        "Federica Filippini"
      ],
      "abstract": "Many decision-making problems in computing and networking systems can be naturally formulated as cost-minimization problems under performance constraints. In dynamic environments, reinforcement learning (RL) is often used to solve such problems at runtime by embedding both costs and constraint violations into a single scalar reward through weighted penalty terms, following a Lagrangian-inspired formulation. However, in this context the behavior of the learned policy critically depends on the choice of these weights, which are typically selected manually. This makes it difficult to identify an appropriate trade-off between optimizing the primary objective and effectively avoiding constraint violations, particularly in non-stationary environments where their relative importance may change. This paper presents MAMO (Multi-Agent system for Multi-Objective constrained optimization), an approach to tackle this balancing problem through multi-agent RL. MAMO decouples task execution from objective design by formulating the selection of reward weights as a learning problem, providing a !rst step towards more autonomous and robust RL-based solutions for constrained optimization problems in dynamic environments.",
      "published": "2026-06-18T13:47:28Z",
      "abstract_url": "http://arxiv.org/abs/2606.20236v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20236v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Thermodynamic Measure of Intelligence",
      "authors": [
        "Ishanu Chattopadhyay"
      ],
      "abstract": "Can intelligence be measured? We propose that intelligence can be defined as the lawful amplification of rare but valid futures: a system increases the probability of outcomes that would be unlikely under passive dynamics but remain admissible under the constraints of the domain. We start with the premise that an intelligent system must model the world and its own place within it. Because the system is part of the world it models, this leads naturally to recursive self-simulation: the system represents futures in which its own actions are part of the trajectory. Our central results give a necessity statement and a conditional near-sufficiency statement connecting this architecture to a precise thermodynamic measure of lawful amplification of rare-valid futures: high rare-valid lift is impossible unless the internal simulation identifies rare-valid futures with high fidelity; conversely, when rare-valid fidelity is high and the simulation contains an effective policy, the achievable lift approaches the actuation-limited optimum. Thus recursive self-simulation is not merely a plausible feature of intelligence but, under the stated assumptions, is necessary and nearly sufficient for high thermodynamic intelligence. The resulting framework makes intelligence measurable on a universal scale, from passive matter and feedback controllers, large language models, and humans as text generators to Maxwell-demon-like information engines.",
      "published": "2026-06-18T13:41:35Z",
      "abstract_url": "http://arxiv.org/abs/2606.20231v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20231v1",
      "categories": [
        "cs.AI",
        "cond-mat.stat-mech",
        "cs.IT",
        "math-ph",
        "nlin.AO"
      ]
    },
    {
      "title": "QMFOL: Benchmarking Large Language Model Reasoning via Quantifiable Monadic First-Order Logic Test Case Generation",
      "authors": [
        "Xinyi Zheng",
        "Ling Shi",
        "Tianlong Yu",
        "Yongxin Zhao",
        "Lorenz Goette",
        "Kailong Wang"
      ],
      "abstract": "Large Language Models (LLMs) have made significant progress in reasoning, particularly in deductive reasoning, which is crucial for high-stakes decision-making. As models improve, evaluation benchmarks should evolve to keep pace. However, existing benchmarks lack fine-grained control over logical complexity and struggle to balance semantic diversity with logical consistency. To address these issues, we propose QMFOL, an automated framework for generating monadic first-order logic reasoning tasks with quantifiable and controllable complexity. It constructs formal logical structures using conjunction and disjunction patterns, enabling precise control over reasoning depth, width, label types, and distractors. These structures are then translated into natural language via LLMs, with logical consistency ensured through round-trip verification using an external prover. Based on our framework, we build QMFOLBench, a benchmark comprising 2880 instances with 960 configurations across diverse logical and semantic dimensions. Evaluations on six large reasoning models (LRMs) and two LLMs show that performance degrades and computational overhead increases with rising logical complexity. Models perform better on True-labeled tasks than on False or Unknown ones, and exhibit sensitivity to semantic variation. Overall, QMFOL offers a scalable and reliable approach for constructing deductive reasoning benchmarks with controllable complexity, enabling more precise evaluation of reasoning capabilities in modern language models.",
      "published": "2026-06-18T13:40:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.20227v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20227v1",
      "categories": [
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "Learner-based Concept Drift Detection: Analysis and Evaluation",
      "authors": [
        "Md Moman Ul Haque Khan",
        "Samira Sadaoui"
      ],
      "abstract": "Machine learning algorithms deployed for evolving streaming environments must handle the non-stationary data distributions, commonly referred to as concept drift. The presence of concept drift poses a major challenge for many real-world applications because it can severely degrade their predictive performance, hindering their ability to support robust decision-making. Consequently, the timely and efficient detection of drift events is critical for sustaining high accuracy over time. This study examines theoretically the concept drift characteristics and numerous drift detection algorithms across several categories. Furthermore, we evaluate their performance on both synthetic and real-world datasets exhibiting diverse streaming scenarios and drift characteristics, such as abrupt and gradual changes. This study aims to enhance understanding of the complex notion of concept drift characteristics and behavior of drift detectors, along with their applicability to diverse contexts.",
      "published": "2026-06-18T13:31:58Z",
      "abstract_url": "http://arxiv.org/abs/2606.20216v1",
      "pdf_url": "https://arxiv.org/pdf/2606.20216v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
