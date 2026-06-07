const PAPERS_DATA = {
  "last_updated": "2026-06-07 04:45:28 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers",
      "authors": [
        "Lizhi Yang",
        "Junheng Li",
        "Nehar Poddar",
        "Yiling Hou",
        "Gio Huh",
        "Robert Griffin",
        "Georgia Gkioxari",
        "Aaron Ames"
      ],
      "abstract": "For a humanoid robot to be deployed in the real world, the choice of command space (i.e., the interface between task planning and whole-body control) is crucial. Existing whole-body controllers typically demand dense kinematic or spatial references that planners struggle to synthesize from task semantics. We instead propose a compact, explicit interface that is intuitive, general, modular, and expressive enough for diverse manipulation skills. To this end, we introduce HANDOFF, a single humanoid whole-body controller that follows this interface and is distilled via multi-teacher KL distillation under a context-conditioned gating scheme into a mixture-of-experts student from three complementary specialists: whole-body motion tracking with safety-filtered data, locomotion, and fall-recovery. On the Unitree G1, HANDOFF matches state-of-the-art velocity tracking and offers one of the largest robust manipulation workspaces. We further demonstrate hardware feasibility through multiple natural-language-driven task roll-outs, powered by a VLM-driven agentic planner with no task-specific data or controller fine-tuning.",
      "published": "2026-06-04T17:59:50Z",
      "abstract_url": "http://arxiv.org/abs/2606.06493v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06493v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Regret Minimization with Adaptive Opponents in Repeated Games",
      "authors": [
        "Mingyang Liu",
        "Asuman Ozdaglar",
        "Tiancheng Yu",
        "Kaiqing Zhang"
      ],
      "abstract": "In this paper, we study regret minimization in repeated games with \\emph{adaptive} opponents who can respond based on histories of play. The standard metric of \\emph{external regret} in online learning is known to fail to capture such adaptivity. To account for players' counterfactual reasoning, we introduce {\\tt Repeated Policy Regret (RP-Regret)}, a game-theoretic metric that measures the difference between the \\emph{realized} and the \\emph{best-in-hindsight} accumulated utility when all players can \\emph{respond} to the history of play. Compared to existing regret notions in this setting, ours is native to repeated game playing, enabling stronger comparators and opponents with fewer constraints, while maintaining the possibility of finding better equilibria when all players minimize it. We first identify necessary conditions for obtaining {\\tt RP-Regret} sublinear in time, on the variation of the player's comparator strategies in the regret definition and on the memories of both the comparator and opponents' strategies. We then study additional conditions and provable algorithms to minimize {\\tt RP-Regret}, which is by definition \\emph{non-convex} in the strategy space. To address this challenge, we propose three algorithms: (i) one based on an optimization oracle, as assumed in some prior work in online non-convex learning; (ii) one that minimizes a convex and \\emph{linearized} surrogate of {\\tt RP-Regret} at each iteration; (iii) one that directly minimizes {\\tt RP-Regret} when opponents change strategies slowly. Furthermore, when all players can run algorithms to minimize the {\\tt RP-Regret} (or its linearized variant), certain subgame perfect equilibria of the repeated game can be learned. We also provide experiments showing that minimizing our regret notions can lead to more cooperative solutions with higher utility in games such as Stag-Hunt.",
      "published": "2026-06-04T17:59:08Z",
      "abstract_url": "http://arxiv.org/abs/2606.06486v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06486v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.GT"
      ]
    },
    {
      "title": "Operation-Guided Progressive Human-to-AI Text Transformation Benchmark for Multi-Granularity AI-Text Detection",
      "authors": [
        "Sondos Mahmoud Bsharat",
        "Jiacheng Liu",
        "Xiaohan Zhao",
        "Tianjun Yao",
        "Xinyi Shang",
        "Yi Tang",
        "Jiacheng Cui",
        "Ahmed Elhagry",
        "Salwa K. Al Khatib",
        "Hao Li",
        "Salman Khan",
        "Zhiqiang Shen"
      ],
      "abstract": "As AI writing assistants become increasingly integrated into real-world drafting and revision workflows, many documents are no longer purely human-written or AI-generated, but instead result from progressive human-AI co-editing. However, existing AI-text detection benchmarks largely focus on final outputs and provide limited understanding of how AI authorship signals emerge, accumulate, or disappear throughout the revision process. We introduce OpAI-Bench, an operation-guided benchmark for studying progressive human-to-AI text transformation across document, sentence, token, and span granularities. Starting from human-written documents, OpAI-Bench constructs nine sequentially revised versions for each sample under predefined AI coverage levels and five representative AI edit operations, covering four domains while preserving complete authorship provenance at multiple granularities. The benchmark supports comprehensive evaluation with 8 document-level detectors, 7 sentence-level detectors, and 2 fine-grained token/span-level detectors. Experiments reveal that AI-text detectability is governed not only by the proportion of AI-edited content, but also by edit operation, domain, and cumulative revision history. Interestingly, we notice that mixed-authorship intermediate versions are often harder to detect than both fully human and heavily AI-edited endpoints, exposing non-monotonic detection patterns missed by existing benchmarks. OpAI-Bench provides a controlled testbed for analyzing whether, when, and how AI-assisted writing becomes detectable under realistic progressive editing scenarios. Our code and benchmark are available at https://github.com/VILA-Lab/OpAI-Bench.",
      "published": "2026-06-04T17:58:05Z",
      "abstract_url": "http://arxiv.org/abs/2606.06481v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06481v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Pretraining Recurrent Networks without Recurrence",
      "authors": [
        "Akarsh Kumar",
        "Phillip Isola"
      ],
      "abstract": "Training recurrent neural networks (RNNs) requires assigning credit across long sequences of computations. Standard backpropagation through time (BPTT) addresses this problem poorly: it is sequential in time, limiting parallelism, and suffers from vanishing or exploding gradients, making long-range associations difficult to learn. We propose Supervised Memory Training (SMT), a method for training nonlinear RNNs that sidesteps recurrent credit propagation entirely by reducing RNN training to supervised learning on one-step memory transition labels $(m_t, x_{t+1}) \\rightarrow m_{t+1}$. SMT acquires these memory labels by training a Transformer-based encoder on a predictive state objective--retaining only information from the past necessary to predict the future. By decoupling what to remember from how to update memory, SMT enables time-parallel RNN training with a stable $O(1)$ length gradient path between any two tokens--without ever unrolling the RNN. We find that SMT outperforms BPTT when pretraining various RNN architectures on tasks like language modeling and pixel sequence modeling. SMT enables nonlinear RNNs to better capture long-range dependencies and train in parallel, potentially unlocking the scaling of models that build temporal abstractions of past experience.",
      "published": "2026-06-04T17:57:33Z",
      "abstract_url": "http://arxiv.org/abs/2606.06479v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06479v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "RREDCoT: Segment-Level Reward Redistribution for Reasoning Models",
      "authors": [
        "Mykyta Ielanskyi",
        "Kajetan Schweighofer",
        "Lukas Aichberger",
        "Sepp Hochreiter"
      ],
      "abstract": "Recent advancements in reasoning language models have been driven by Reinforcement Learning (RL) fine-tuning. Most often, these rely on the Group Relative Policy Optimization (GRPO) algorithm or modifications thereof to steer the models to produce Chain-of-Thought (CoT) traces. The final answer can only be verified, and the reward assigned, after the CoT trace is complete, making it a delayed reward problem. GRPO and its modifications correspond to Monte Carlo methods in standard RL, which are known to suffer from high variance. A possible solution to this problem is the redistribution of rewards through credit assignment, where segments of the CoT trace that are important for arriving at the desirable solution are emphasized by assigning a higher reward. While Monte Carlo sampling can be used to provide an unbiased estimate of intermediate state values, its computational overhead makes it unsuitable for train-time credit assignment in long contexts at high granularity. We introduce RREDCoT (Reward REDistribution for Chain of Thoughts), which utilizes the model itself to approximate the optimal reward redistribution without additional generation. We investigate the advantages of our method compared to MC sampling and several attribution methods. We further analyze several aspects relevant to the construction of the redistribution such as segmentation of CoT traces and state value estimation.",
      "published": "2026-06-04T17:56:31Z",
      "abstract_url": "http://arxiv.org/abs/2606.06475v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06475v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Self-Augmenting Retrieval for Diffusion Language Models",
      "authors": [
        "Paul Jünger",
        "Justin Lovelace",
        "Linxi Zhao",
        "Dongyoung Go",
        "Kilian Q. Weinberger"
      ],
      "abstract": "Discrete diffusion language models generate text by iteratively denoising an entire response in parallel. At each step, they predict tentative tokens for every masked position, committing the confident predictions to the output and discarding the unconfident ones. We show that the discarded tokens are in fact a useful lookahead signal for retrieval-augmented generation: even low-confidence tokens often surface salient entities early in the denoising trajectory, enabling retrieval of stronger evidence before the output is finalized. We exploit this through Self-Augmenting Retrieval for Diffusion Language Models (SARDI), a dynamic RAG framework that uses these lookahead tokens to guide retrieval during denoising. SARDI is training-free, retriever-agnostic, and applicable to any reasoning-capable discrete diffusion language model. Across five multi-hop QA benchmarks, SARDI outperforms current training-free diffusion and autoregressive retrieval baselines at up to $8\\times$ higher throughput.",
      "published": "2026-06-04T17:56:27Z",
      "abstract_url": "http://arxiv.org/abs/2606.06474v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06474v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery",
      "authors": [
        "Shangheng Du",
        "Xiangchao Yan",
        "Jinxin Shi",
        "Zongsheng Cao",
        "Shiyang Feng",
        "Zichen Liang",
        "Boyuan Sun",
        "Tianshuo Peng",
        "Yifan Zhou",
        "Xin Li",
        "Jie Zhou",
        "Liang He",
        "Bo Zhang",
        "Lei Bai"
      ],
      "abstract": "Large language model (LLM) agents are increasingly applied to long-horizon tasks such as scientific discovery and machine learning engineering (MLE), where sustained self-evolution becomes a key capability. However, existing MLE agents suffer from inter-branch information isolation, memoryless search, and lack of hierarchical control, which together hinder long-horizon optimization. We present MLEvolve, an LLM-based self-evolving multi-agent framework for end-to-end machine learning algorithm discovery. By extending tree search to Progressive MCGS, MLEvolve enables cross-branch information flow through graph-based reference edges and gradually shifts the search from broad exploration to focused exploitation with an entropy-inspired progressive schedule. To allow the agent to evolve with accumulated experience, we introduce Retrospective Memory, which combines a cold-start domain knowledge base with a dynamic global memory for task-specific experience retrieval and reuse. For stable long-horizon iteration, we further decouple strategic planning from code generation with adaptive coding modes. Evaluation on MLE-Bench shows that MLEvolve achieves state-of-the-art performance across multiple dimensions including average medal rate and valid submission rate under a 12-hour budget (half the standard runtime). Moreover, MLEvolve also outperforms specialized algorithm discovery methods including AlphaEvolve on mathematical algorithm optimization tasks, demonstrating strong cross-domain generalization. Our code is available at https://github.com/InternScience/MLEvolve.",
      "published": "2026-06-04T17:55:59Z",
      "abstract_url": "http://arxiv.org/abs/2606.06473v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06473v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training",
      "authors": [
        "Senmiao Wang",
        "Tiantian Fang",
        "Haoran Zhang",
        "Yushun Zhang",
        "Kunxiang Zhao",
        "Alex Schwing",
        "Ruoyu Sun"
      ],
      "abstract": "We propose a preconditioning (PC) layer, a weight parameterization via polynomial preconditioner that ensures stable weight conditioning throughout LLM training. The PC module reshapes the singular-value spectrum of weight matrices via low-degree polynomial preconditioning. After training, the preconditioned weights can be merged back into the original architecture, incurring no inference overhead. We demonstrate the advantage of the proposed PC layer over standard transformers in Llama-1B pre-training, for both the AdamW and Muon optimizers. Theoretically, we justify this spectrum-control principle by proving that uniformly bounding each layer's singular values ensures geometric convergence of gradient descent to global minima, for certain deep linear networks. Our code is available at https://github.com/Empath-aln/PC-layer.",
      "published": "2026-06-04T17:55:11Z",
      "abstract_url": "http://arxiv.org/abs/2606.06470v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06470v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "You Only Index Once: Cross-Layer Sparse Attention with Shared Routing",
      "authors": [
        "Yutao Sun",
        "Yanqi Zhang",
        "Li Dong",
        "Jianyong Wang",
        "Furu Wei"
      ],
      "abstract": "Long-context inference in modern LLMs is increasingly constrained by decoding efficiency, especially in reasoning-heavy settings where models generate long intermediate chains of thought. Existing sparse attention methods often face a practical efficiency-quality trade-off. Structured block sparse methods typically provide stronger acceleration but incur noticeable quality loss, while token sparse methods are usually more accurate yet deliver limited end-to-end speedup because top-k routing over the full cache remains expensive. In this work, we propose cross-layer sparse attention (CLSA), which is built on top of KV-sharing architectures such as YOCO. The core idea is to share not only the KV cache across cross-decoder layers, but also the routing index. A single indexer computes token-level top-k selection once and reuses the resulting index across layers, thereby preserving the fine-grained selectivity of token sparse attention while amortizing the routing overhead. The resulting architecture improves all major inference bottlenecks jointly, including pre-filling, KV-cache storage, and long-context decoding. Experiments across short-context and long-context benchmarks show that CLSA is both accurate and efficient, achieving up to 7.6x decoding speedup and 17.1x overall throughput improvement at 128K context. These results suggest a more complete architectural solution for long-context LLMs that jointly advances model quality and inference efficiency.",
      "published": "2026-06-04T17:54:04Z",
      "abstract_url": "http://arxiv.org/abs/2606.06467v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06467v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "In-Context Multiple Instance Learning",
      "authors": [
        "Alexander Möllers",
        "Marvin Sextro",
        "Julius Hense",
        "Gabriel Dernbach",
        "Klaus-Robert Müller"
      ],
      "abstract": "Multiple Instance Learning (MIL) addresses problems where supervision is available at the level of bags of instances and has been successfully applied in fields ranging from computational pathology to satellite imagery. Nevertheless, existing algorithms struggle in the low-label regime that characterizes many real-world applications. Flexible models overfit and rigid ones fail to adapt to the task at hand. We show that pretraining an in-context learner with a Perceiver-style architecture on synthetic data yields a model that can solve new tasks from a handful of labeled bags. At inference time, classification happens in a single forward pass and requires no gradient updates. We propose and investigate different synthetic data generators for bag-structured data and find that they capture complementary inductive biases. A model pretrained on a mixture of these generators inherits their per-task strengths and achieves the best average performance across twelve MIL benchmarks, outperforming supervised baselines that require task-specific training.",
      "published": "2026-06-04T17:50:32Z",
      "abstract_url": "http://arxiv.org/abs/2606.06458v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06458v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents",
      "authors": [
        "Zhuoming Chen",
        "Xinrui Zhong",
        "Qilong Feng",
        "Ranajoy Sadhukhan",
        "Yang Zhou",
        "Michael Qizhe Shieh",
        "Zhihao Jia",
        "Beidi Chen"
      ],
      "abstract": "Sparse attention is becoming increasingly important for serving large language models (LLMs) as generation lengths continue to grow. However, deploying and evaluating new sparse attention algorithms at scale remains highly engineering-intensive, slowing both human researchers and AI agents in exploring the sparse attention design. To address this challenge, we present Vortex, a system that combines a Python-embedded frontend language atop a page-centric tensor abstraction for expressing a broad range of sparse attention algorithms, with an efficient backend tightly integrated into modern LLM serving stacks. Vortex enables rapid prototyping, deployment, and evaluation of sparse attention algorithms, effectively translating their theoretical efficiency gains into real-world throughput improvements. As a result, Vortex substantially accelerates the design and iteration of sparse attention algorithms. First, AI agents use Vortex to automatically generate and refine diverse algorithms, the best reaching up to $3.46\\times$ higher throughput than full attention while preserving accuracy. Second, Vortex extends sparse attention to emerging architectures and very large models that are otherwise hard to experiment with, reaching up to $4.7\\times$ higher throughput on the MLA-based GLM-4.7-Flash and $1.37\\times$ on the 229B-parameter MiniMax-M2.7 on NVIDIA B200 GPUs.",
      "published": "2026-06-04T17:48:17Z",
      "abstract_url": "http://arxiv.org/abs/2606.06453v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06453v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss",
      "authors": [
        "Thomas T. Zhang",
        "Alok Shah",
        "Yifei Zhang",
        "Vincent Zhang",
        "Nikolai Matni",
        "Max Simchowitz"
      ],
      "abstract": "Many modern applications of deep learning involve training a neural network via a one-step prediction loss (e.g., $L^2$ regression, cross-entropy), but deploy the network by rolling out along its own predictions. Key examples include autoregressive language modeling, flow-based generative modeling, and robot policy learning. It is well-documented that these settings induce a phenomenon we call test-time feedback (TTF): the mismatch between the training/validation loss and downstream metrics of interest, such as task success rate and generation quality, which grows with task length. While data curation, architecture, and objective design have been proposed to combat train-test shift in TTF settings, this paper proposes optimization as a new design axis to mitigate error accumulation. Specifically, we introduce a new optimization paradigm called double-preconditioning (DoPr) uniquely tailored to the challenges of TTF. DoPr combines gradient-wise preconditioning, as in Adam and Muon, with activation-wise preconditioning (AP), such as in KFAC. We show that the addition of AP yields a drop-in intervention for increasing downstream model performance across a range of TTF settings. Interestingly, these gains in test-time performance do not consistently accompany improvements in validation loss, opening new questions about how to properly evaluate models trained with one-step supervised objectives.",
      "published": "2026-06-04T17:22:58Z",
      "abstract_url": "http://arxiv.org/abs/2606.06418v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06418v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Unsupervised Skill Discovery for Agentic Data Analysis",
      "authors": [
        "Zhisong Qiu",
        "Kangqi Song",
        "Shengwei Tang",
        "Shuofei Qiao",
        "Lei Liang",
        "Huajun Chen",
        "Shumin Deng"
      ],
      "abstract": "Inference-time skill augmentation provides a lightweight way to improve data-analytic agents by injecting reusable procedural knowledge without updating model parameters. However, discovering effective skills for data analysis remains challenging, as reliable supervision is expensive and success criteria vary across analytical formats. This raises the key question of how to discover reusable data-analysis skills from unlabeled exploration alone. We propose DataCOPE, an unsupervised verifier-guided skill discovery framework for data-analytic agents. DataCOPE derives verifier signals from the exploration trajectories and uses them to characterize relative quality or aggreement among trajectories. It iteratively coordinates a Data-Analytic Agent for trajectory generation, an Unsupervised Verifier for signal extraction, and a Skill Manager for contrastive skill distillation. For report-style analysis, we instantiate the verifier as an Adaptive Checklist Verifier that derives task-specific criteria, scores reports by verifiable coverage, and iteratively refines the checklist. For reasoning-style analysis, we instantiate it as an Answer Agreement Verifier that groups trajectories by answer agreement and uses self-consistency as an auxiliary signal. We evaluate DataCOPE on report-style analysis from Deep Data Research and reasoning-style analysis from DABStep. Across both settings, DataCOPE consistently improves held-out performance over baselines. Averaged across four model settings, DataCOPE improves the mean score by 9.71% and 32.30% on report-style and reasoning-style tasks respectively.",
      "published": "2026-06-04T17:20:47Z",
      "abstract_url": "http://arxiv.org/abs/2606.06416v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06416v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes",
      "authors": [
        "Wenbo Li",
        "Xiaoliang Ju",
        "Zipeng Qin",
        "Rongyao Fang",
        "Hongsheng Li"
      ],
      "abstract": "Indoor scene generation is crucial for robot simulation and modern interior design. However, complex layouts together with scarce 3D scene data make learning-based generation challenging. Existing methods often rely on hand-crafted rules or focus on isolated sub-tasks (e.g., floorplan synthesis or single-room furnishing), producing whole-home scenes that lack global coherence, realism, and simulation readiness. To mitigate these limitations, we propose a unified hierarchical framework that decomposes indoor scene synthesis into controllable stages. First, we curate a large-scale dataset of 300K real residential floorplans to train a large language model for whole-home floorplan generation. With detailed descriptions and a K-D tree-based representation, our method enables fine-grained, controllable whole-home floorplan generation. Building upon the generated whole-home floorplan, we leverage image generation models to draft furniture layouts from multi-level roaming viewpoints, and then generate the layouts of small manipulable objects on different supporting surfaces (e.g., cabinets, desks, and dining tables) for embodied AI simulation. During furniture and object layout generation, a VLM-based refiner iteratively corrects furniture and object placement, and a 3D generative model enables flexible replacement of individual assets. We further attach basic physical attributes and simple surface texture and lighting setups to complete the pipeline for embodied AI use. Experiments and user studies demonstrate that our pipeline produces indoor spaces with greater layout diversity and stronger 3D design appeal, outperforming prior methods on both quantitative and qualitative metrics. Finally, alongside our generation pipeline, we will release the floorplan dataset and 5K fully furnished scenes to the community. Project Page: https://kairos-homeworld.github.io/",
      "published": "2026-06-04T16:58:43Z",
      "abstract_url": "http://arxiv.org/abs/2606.06390v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06390v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "An Infectious Disease Spread Simulation Based on Large Language Model Decision Making",
      "authors": [
        "Yonchanok Khaokaew",
        "Ruochen Kong",
        "Andreas Zufle",
        "Hao Xue",
        "Taylor Anderson",
        "Chandini Raina MacIntyre",
        "Matthew Scotch",
        "Flora D. Salim",
        "David J Heslop"
      ],
      "abstract": "Modelling individual decision-making during infectious disease outbreaks is crucial for understanding behavioural dynamics and informing effective public health interventions. Prior work has shown that large language models can simulate realistic human behaviour by generating agent decisions based on demographic prompts and situational context. We build on this foundation with a spatially grounded, agent-based simulation framework that integrates LLM-generated decisions about self-reported influenza-like illness into a census-based synthetic population of agents. Location is treated as a central feature: agents are assigned to spatial units within cities, capturing the spatial distributions of different demographic groups using real-world census data and enabling geographically diverse behavioural modelling. We implement and compare three decision scenarios, independent reasoning, household influence, and message framing, and simulate self-reporting outcomes in San Francisco and Atlanta. Results reveal that income and education are the dominant drivers of reporting rate variation, with smaller but consistent effects from geography, LLM model choice, and message framing. Our framework generates synthetic data that captures both social and geographic heterogeneity, supporting spatial epidemiological modelling and bias-aware behavioural analysis.",
      "published": "2026-06-04T16:30:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.06360v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06360v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Boosting Brain-to-Image Decoding with TRIBE v2 Data Augmentation",
      "authors": [
        "Yohann Benchetrit",
        "Marlène Careil",
        "Simon Dahan",
        "Hubert Banville",
        "Stéphane d'Ascoli",
        "Jean-Rémi King"
      ],
      "abstract": "Brain decoding is limited by the availability of labeled neural data, and remains challenging in low-data regimes. To address this issue, we investigate whether and when brain decoding can be boosted by augmenting small fMRI datasets with synthetic data generated by a pretrained model of fMRI responses to stimuli. We use TRIBE v2, a large encoding model pretrained on more than 1000 hours of fMRI responses to video, audio and language. For each dataset, we evaluate systematic grids that show how the performance of image decoders varies with the amount of synthetic data used for training. Our results, based on two datasets (the 7T fMRI Natural Scenes Dataset and 3T fMRI BOLD5000), show up to 68% improvement in Top-10 image-retrieval accuracy compared to decoders trained only on real data. Importantly, the proportion of augmented data required to reach a given image decoding performance needs to be adjusted depending on the data source. Surprisingly, image decoders trained exclusively on synthetic fMRI can perform above chance in some settings, suggesting that TRIBE v2 can support zero-shot brain-to-image decoding. Together, these results show how large-scale models of the fMRI responses to sight, sound and language may provide a foundation to improve the data efficiency for image decoding.",
      "published": "2026-06-04T16:18:08Z",
      "abstract_url": "http://arxiv.org/abs/2606.06345v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06345v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "q-bio.NC"
      ]
    },
    {
      "title": "TokenMizer: Graph-Structured Session Memory for Long-Horizon LLM Context Management",
      "authors": [
        "Shweta Mishra"
      ],
      "abstract": "Large language model (LLM) deployments for long-horizon tasks face a fundamental constraint: context windows are finite while productive work sessions are not. When history exceeds the Maximum Effective Context Window (MECW), critical structured information - architectural decisions, task transitions, file histories - is silently discarded. Existing mitigations treat history as flat text, destroying the relational structure that makes sessions resumable. We present TokenMizer, an open-source proxy system that models LLM session history as a typed knowledge graph. The schema defines 14 node types and 7 edge types. A hybrid extraction pipeline populates the graph incrementally, while a three-tier checkpoint system serializes it into compact resume blocks. An 8-layer compression pipeline reduces context overhead, and a semantic cache reduces repeated-query latency. Evaluated on a controlled benchmark of 21 sessions spanning 5 domains, TokenMizer demonstrates significant token economy. It produces resume blocks averaging 78 tokens (range: 42-124) - 2x smaller than evaluated baselines (159-170 tokens) - while achieving higher decision recall (+9-17 percentage points). Crucially, baselines only preserve that a technology was mentioned; TokenMizer preserves the rationale. Across all sessions, TokenMizer achieves mean task recall 51.0%, decision recall 46.6%, and file recall 58.7%. Variance reflects domain heterogeneity: explicit imperative phrasing (software engineering) scores higher than implicit reasoning (research). Ablation studies show fuzzy label matching is the dominant improvement factor (+33 pp task recall). The heuristic compression achieves 47.3% token reduction with zero external dependencies. TokenMizer provides a queryable alternative to text-retention baselines at half the token cost.",
      "published": "2026-06-04T16:12:28Z",
      "abstract_url": "http://arxiv.org/abs/2606.06337v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06337v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Bridging Domain Expertise and Generalization for Performance Estimation",
      "authors": [
        "Shuxuan Li",
        "Zhilin Zhao",
        "Quyu Kong",
        "Wei-Shi Zheng"
      ],
      "abstract": "Performance estimation under distribution shift aims to predict how a model behaves on an unlabeled test set whose distribution differs from the training data, a scenario that requires reliable indicators that can faithfully reflect model behavior without ground-truth labels. Existing approaches rely solely on the outputs of the given model whose biases are amplified once the distribution shifts, weakening the correlation with the true performance. Motivated by this limitation, we propose Fused Reference Alignment Prediction (FRAP), which leverages the complementary strengths of an external foundation model and the base model to construct a more reliable surrogate of the ground-truth labels. FRAP aligns the prediction distribution of the foundation model with that of the base model by applying temperature-scaled calibration that minimizes their divergence. The aligned predictions are fused through confidence-based weighting into a refined reference distribution that integrates robustness from the foundation model and domain-specific expertise from the base model, and performance estimation is obtained by measuring how closely the base model predictions agree with this reference. Extensive experiments across diverse datasets and architectures show that FRAP provides consistent and substantial improvements over representative performance-estimation methods under distribution shift.",
      "published": "2026-06-04T16:10:04Z",
      "abstract_url": "http://arxiv.org/abs/2606.06335v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06335v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability",
      "authors": [
        "Seyed Arshan Dalili",
        "Mehrdad Mahdavi"
      ],
      "abstract": "Sparse Autoencoders (SAEs) are widely used for mechanistic interpretability in large language models, yet their formulation assigns each latent feature a single decoder direction, implicitly assuming features to be one-dimensional. We show that this assumption mismatches with the multi-dimensional structure of model features, provably inducing feature splitting through two distinct mechanisms. Geometrically, reconstructing a feature of intrinsic dimension $d_i \\ge 2$ to error $\\varepsilon$ with single-direction decoders forces a number of atoms that is exponential in $d_i$. From an end-to-end optimization perspective, this splitting is not merely possible but actively preferred. We prove that there exists a continuous path from the true $d_i$-dimensional basis to a strictly lower risk of the $\\ell_1$-regularized SAE objective, whose descent directions drive any trained dictionary into that exponential regime. A single coherent feature is therefore fragmented across many near-collinear latents, producing spurious multiplicity and obscuring the intrinsic geometry. Motivated by this, we introduce Subspace-Aware Sparse Autoencoders (SASA), which replace single-vector decoders with learned decoder subspaces, enforce block sparsity via Top-$s$ group gating, and adapt each group's effective rank with a nuclear-norm regularizer. We then show that once the block size satisfies $r \\ge d_i$, a single group not only can represent the entire feature slice but is the global minimizer of the SASA objective. This consolidation yields a sample complexity polynomial in $d_i$ rather than exponential -- a decisive advantage given that every training activation costs an LLM forward pass. Empirically, on GPT-2 and Mistral-7B, SASA reduces feature splitting and absorption, improves monosemanticity and interpretability, and matches or exceeds standard SAEs while training on roughly half the token budget.",
      "published": "2026-06-04T16:08:25Z",
      "abstract_url": "http://arxiv.org/abs/2606.06333v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06333v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PAMF: Prior-Aware Multimodal Fusion for Incomplete Time Series Data",
      "authors": [
        "Ziwen Kan",
        "Wugeng Zheng",
        "Tianlong Chen",
        "Song Wang"
      ],
      "abstract": "In healthcare, multimodal time series tasks often operate on incomplete observations in practice, for example when ECG segments are lost because electrodes detach or an entire respiratory channel is unavailable during overnight monitoring. Such missingness typically appears in two structurally distinct patterns: within-modality missing, where values are absent within an otherwise observed modality, and modality-level missing, where an entire modality is unavailable. Existing methods typically represent unobserved data implicitly through masks or missing embeddings, without learning instance-specific missing information, and most are designed for only one missingness pattern. A natural approach is to explicitly estimate the missing data; however, existing imputation methods treat missingness uniformly despite their different structural priors, and the imputation process is often isolated from downstream tasks, preventing downstream tasks from guiding imputation toward more informative representations. To address these limitations, we present PAMF, a multimodal time-series framework that explicitly handles different missingness patterns while coupling imputation with downstream prediction through prior-aware flow matching and weight sharing. Specifically, the method initializes the flow-matching source state with type-specific priors to distinguish two missing types. It further connects imputation and classification through architecturally matched encoders with weight sharing, transferring task-relevant representations into the imputation process. Experiments on multiple multimodal healthcare time-series benchmarks show that the proposed method achieves the strongest overall downstream performance across diverse datasets and missing settings compared with existing baselines.",
      "published": "2026-06-04T16:04:21Z",
      "abstract_url": "http://arxiv.org/abs/2606.06328v1",
      "pdf_url": "https://arxiv.org/pdf/2606.06328v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
