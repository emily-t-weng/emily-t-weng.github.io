const PAPERS_DATA = {
  "last_updated": "2026-11-09 05:52:14 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "On Agentic Manipulation Policies",
      "authors": [
        "Chen Chen",
        "Omid Bagherieh",
        "Naveen Nikamanth Appiah Balaji",
        "Jesse T. Gonzalez",
        "Negin Heravi",
        "Fahad Islam",
        "Xueting Li",
        "Jean-Baptiste Regli",
        "Kejia Ren",
        "Petr Švec",
        "David J. Yoon",
        "Matthias Müller",
        "Vladlen Koltun"
      ],
      "abstract": "Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6\\% success rate zero-shot, compared to 49.2\\% and 42.9\\% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rates by 20x and 15x relative to strong VLA and WAM baselines. The basic harness supplies no grasping, pushing, or placing skills; the agent develops its own strategies, such as searching for hidden objects, using objects as tools, and calibrating its cameras. With the addition of a persistent memory store, the agent can also learn from experience: an augmented harness with a memory representation that is carried across episodes lowers the failure rate on RoboDojo bimanual manipulation tasks by 4.6x relative to zero-shot deployment. Our analysis suggests that AMPs are substantially more general and robust than specialist manipulation policies, providing a promising foundation for robot manipulation.",
      "published": "2026-11-06T18:59:34Z",
      "abstract_url": "http://arxiv.org/abs/2611.09892v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09892v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse",
      "authors": [
        "Peter Belcak",
        "Greg Heinrich",
        "Saurav Muralidharan",
        "Shizhe Diao",
        "Pavlo Molchanov"
      ],
      "abstract": "Large language models reinvent the wheel all the time. They conflate task-solving with the execution of the solutions, which forces them to solve every problem from scratch. We show that once a task has been reasoned through, the application of its solution to a new problem instance can be done very efficiently and with little involvement from language models. We introduce agentic reasoning caching, a method for constructing robust CPU-executable agentic reasoning caches (ARCs) from LLM reasoning traces. ARCs show on-par performance with frontier LLMs across a range of difficult mathematical and scientific tasks, are fully explainable, continually trainable, robust to input perturbations, and 16-291x cheaper to invoke than the source model.",
      "published": "2026-11-06T18:58:50Z",
      "abstract_url": "http://arxiv.org/abs/2611.09891v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09891v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Grasp-TAGS: Task-Aware Grasp Selection via Critic Distillation for Prehensile Loco-Manipulation",
      "authors": [
        "Xun Tu",
        "Minghao Zou",
        "Karthik Desingh"
      ],
      "abstract": "Loco-manipulation, which requires mobile manipulators to coordinate locomotion and manipulation over long horizons, is a key capability for large-object rearrangement in applications such as service robotics and warehouse automation. Despite recent progress, reliably executing these tasks remains challenging because performance depends on complex robot-object and object-environment interaction dynamics. Existing approaches address this challenge either through model-based methods, which often require accurate dynamics models, or through learning-based methods, which reduce this requirement but typically assume a fixed or predefined initial grasp. However, different geometrically stable grasps can lead to substantially different downstream manipulation performance. Therefore, we propose Grasp-TAGS, a task-aware grasp selection framework for goal-conditioned large-object rearrangement. We train a unified PPO-based reinforcement learning policy across multiple initial grasp configurations and distill the learned critic into a lightweight grasp utility network that ranks candidate grasps using only the candidate grasp configuration and goal information. At inference time, we select the grasp with the highest predicted downstream utility for the downstream policy to complete the task, before even making a contact. Experiments demonstrate that Grasp-TAGS improves rearrangement success over both fixed single-grasp policies and random grasp selection, achieving an average success rate of 89.84% compared with 79.87% for the best-performing single-grasp policies, an improvement of 9.97 percentage points. Grasp-TAGS also reduces final orientation errors across all evaluated objects. For more qualitative results, refer to the supplementary webpage: https://rpm-lab-umn.github.io/Grasp-TAGS-web-page/",
      "published": "2026-11-06T18:44:33Z",
      "abstract_url": "http://arxiv.org/abs/2611.09864v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09864v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Flow-Guided Target-Space Alignment via Path Consistency",
      "authors": [
        "Ruizhi Yuan",
        "Zeqiu Yu",
        "Wei Gao",
        "Wei Chen",
        "Xiwei Tang",
        "Lu Tang"
      ],
      "abstract": "Many multimodal alignment problems require mapping a source modality into a fixed space defined by a target modality, where the target space is not merely an intermediate embedding but is used directly for downstream analysis. In this setting, successful alignment should preserve both the global structure of the target distribution and the instance-level correspondence observed in paired source--target training data. Existing objectives typically address only part of this goal: contrastive losses encourage relative similarity, distribution-matching losses improve marginal agreement, and pointwise regression fits each source sample to its paired target without explicitly modeling target-space path geometry. We propose FlowAlign, a flow-guided target-space alignment method that repurposes conditional flow matching as a training-time alignment signal rather than an inference-time transport model. FlowAlign uses the learned flow field to guide the alignment of source-side representations with the fixed target space, encouraging them to preserve paired correspondence while remaining compatible with the global target-space structure. We further provide theoretical analysis showing how FlowAlign helps improve paired correspondence by controlling endpoint accuracy and capturing local target-space geometry. Experiments on synthetic data, paired single-cell multiome benchmarks, and image--text alignment demonstrate that FlowAlign improves instance level correspondence while preserving target-space structure across different modality settings.",
      "published": "2026-11-06T18:44:08Z",
      "abstract_url": "http://arxiv.org/abs/2611.09863v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09863v1",
      "categories": [
        "cs.AI",
        "cs.CV",
        "cs.LG"
      ]
    },
    {
      "title": "Can Agentic Systems Close the Gap in Radiology with Production AI Tools?",
      "authors": [
        "Aubin Charley",
        "Julien Khlaut",
        "Charles Corbière",
        "Pierre Manceron",
        "Corentin Dancette",
        "Léo Machado"
      ],
      "abstract": "Radiology AI has developed along two paths: specialized models that perform well on narrow tasks but cover few clinical findings, and generalist models that address a broader range of tasks but remain prone to hallucination and inaccurate measurements. Combining the two through agentic systems offers a route towards comprehensive radiology reporting, but published systems are tied to one modality or body part. We describe a radiology agentic framework that standardizes how an orchestrator calls modular tools to produce a structured report, and use it to measure what such a system can do with the tools from FDA-cleared radiology AI devices. Report generation is simulated on a large multi-modality report corpus, with the orchestrator given either a catalog of finding classes consolidated from the cleared devices, each available on the modalities it is cleared for, or an unlimited catalog in which any class can be requested, and planning either from the clinical indication or from the target report. Tool outputs are simulated from the reference report, so the comparison isolates tool availability and orchestration from tool error. With the cleared catalog and the indication alone, the orchestrator recovers under a third of the clinically actionable findings; recovery improves substantially with the unlimited catalog, and further when the plan is written from the target report. Attributing every missed finding to a cause shows that most are classes no cleared device offers for the study's modality, and that a third are classes or tools that were offered and not called. Adding the most frequently missing tools closes most of the gap to the unlimited catalog on held-out reports, whereas a smaller orchestrator with the unlimited catalog gives up the whole gain. Tool availability and effective orchestration are thus complementary constraints on comprehensive AI-assisted radiology reporting.",
      "published": "2026-11-06T18:23:14Z",
      "abstract_url": "http://arxiv.org/abs/2611.09836v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09836v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Discovering formally verified quantum programs with AI",
      "authors": [
        "Sergey Shirobokov",
        "Ben Jaderberg",
        "Katerina Hristova",
        "George Tsoukalas",
        "Johannes Bausch",
        "Eric Wieser",
        "Francisco J. H. Heras",
        "Nicholas C. Rubin",
        "Guang Hao Low",
        "Sébastien Racanière",
        "Nenad Tomašev",
        "Lei M. Zhang",
        "James Watson",
        "Tanuj Khattar",
        "Aws Albarghouthi",
        "Stavan Jain",
        "Harun Khan",
        "Leopoldo Sarra",
        "Andrew W. Senior",
        "Michael Broughton",
        "Hussain Masoom",
        "Sergio Boixo",
        "Ryan Babbush",
        "Swarat Chaudhuri",
        "Pushmeet Kohli"
      ],
      "abstract": "The timeline by which quantum computers can solve breakthrough problems in science depends on improved quantum algorithms as much as on better hardware. Such algorithmic advances are typically achieved through many years of expert human collaboration. In mathematics and computer science, large language models have accelerated progress through automated discovery loops. However, quantum algorithms can only be evaluated by classical computers on restricted or small instances. Here we demonstrate automated discovery of quantum programs beyond these regimes including order-of-magnitude improvements on scientific applications. To achieve this, we build a quantum program library in the Lean theorem prover in which large language model agents synthesize novel programs against formal problem specifications and verify their correctness and cost. Agent teams collaborate, sharing results and knowledge through wikis and discussing ideas at organized conferences. Our results improve on prior baselines for parallel search (up to 1.4x), quantum arithmetic (290x), and simulations relevant to fusion, photochemistry and drug-metabolism by 11x, 8x and 2.5x respectively. Combining human and artificial intelligence at scale, under the rigor of mathematical proof, lets researchers design algorithms that today's classical and quantum computers cannot test, shortening the path to practical quantum computation.",
      "published": "2026-11-06T18:08:06Z",
      "abstract_url": "http://arxiv.org/abs/2611.09822v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09822v1",
      "categories": [
        "quant-ph",
        "cs.AI",
        "cs.PL"
      ]
    },
    {
      "title": "Generative Latent Priors for Interpreting and Steering Protein Language Models",
      "authors": [
        "Daniel Hinjos",
        "Lucía Urcelay"
      ],
      "abstract": "Dominant methods for interpreting large models impose strong structural assumptions. Generative Latent Priors (GLP) are a complementary approach: a diffusion model that learns the distribution of a frozen model's activations directly, imposing no assumptions. We port GLP to ESM-2 activations (UniRef50) and evaluate it at two host scales (ESM-2-8M and -650M), both as a lens for protein representations and as a prior for editing them. Generation quality scales smoothly with depth and compute. As a lens, single GLP \"meta-neurons\" are the strongest probes of discrete Swiss-Prot concepts at both hosts, beating InterPLM's released SAE features, and lead on continuous properties at 8M though not at 650M; falsification tests show this is mostly concentration, not new information. As an editing prior, the on-manifold correction makes DiffMean steering of three biochemical properties effective and restores on-manifoldness, but only in isolation: re-embedding the sequences actually produced after several rounds of decoding, the structural plausibility is compromised. This is mainly reproduced in additional SAE-based clamping structural steering experiments, where only Helix motif succeeds. The naturalness failure is traced to the rewriting coverage and the decoder rather than the prior, highlighting what the method must clear before it can serve for both discovery-by-editing and discovery-by-exploration. Trained GLPs are released on huggingface.co/collections/HPAI-BSC/glprot.",
      "published": "2026-11-06T18:06:39Z",
      "abstract_url": "http://arxiv.org/abs/2611.09818v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09818v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Scaling Graph Similarity Learning with Graph Neural Networks",
      "authors": [
        "Ahmet Tüzen",
        "Helge Langseth",
        "Kjetil Nørvåg"
      ],
      "abstract": "Graph similarity computation is an important task in graph mining with many real-world applications, such as chemical compound analysis, program dependence comparison, and social network analysis. The problem is computationally challenging: Graph Edit Distance (GED) and Maximum Common Subgraph (MCS), which quantify the distance between graphs and identify their largest shared subgraph, respectively, are NP-hard to compute exactly. To address these computational challenges, Graph Neural Networks (GNNs) have been applied to this problem. However, existing GNN-based approaches have primarily focused on small-scale graph datasets, while scalability to larger scales remains underexplored. In this work, we propose FiLSim, a scalable GNN-based framework, and investigate whether a simplified GNN pipeline can maintain competitive performance on standard benchmarks while scaling to large graphs. We additionally propose a sequential graph-editing strategy to construct large-scale graph datasets with reliable GED labels to evaluate scalability. The proposed framework is evaluated on four benchmark datasets and four newly created large-scale datasets, demonstrating its effectiveness and scalability. FiLSim remains accurate on graphs with more than 150{,}000 nodes, where the baseline methods run out of memory or fail to learn, and scores all test pairs within seconds on a single GPU.",
      "published": "2026-11-06T17:55:35Z",
      "abstract_url": "http://arxiv.org/abs/2611.09805v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09805v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Knowing When to Intervene: Causal Evaluation of AI Agents",
      "authors": [
        "Ayush Sawarni",
        "Wenqian Xing",
        "Tijana Zrnic",
        "Ramesh Johari",
        "Vasilis Syrgkanis"
      ],
      "abstract": "Current evaluations of agentic AI systems primarily rely on benchmarks. However, the static nature of benchmarks makes them uninformative about the value of auditing interventions at deployment time. For reliable deployment, we often need to determine within an agent's trajectory whether or not to intervene, e.g., by verifying the agent's current output or steering it toward different behavior, using either human review or more costly models or time-consuming agentic supervision. The value of such interventions is fundamentally a causal inference question, asking whether the causal effect of such an intervention is beneficial for the final outcome. Estimating this causal effect is particularly challenging due to the dynamic nature of the problem: the impact of an intervention depends on the current context and environment, as well as the agent's future behavior. We develop the first methodologically grounded framework for evaluating the causal effects of randomized interventions within agent trajectories. By leveraging a relatively small dataset of agent trajectories collected under randomized interventions such as verification or screening, we can estimate the dynamic causal effect of a rich class of (contextual) intervention policies. The structural parameters that we estimate can also be used to construct safety leaderboards that capture the risk of various agentic actions, which can be used by practitioners for audit prioritization. Applying our methods to tau-bench Retail, ToolSandbox, and ImpossibleBench, we find that intervention effects vary substantially across agent actions and states. More broadly, our work suggests a rich research agenda at the intersection of causal inference and agentic evaluation, which we believe is essential for the reliable and safe deployment of AI agents in the real world.",
      "published": "2026-11-06T17:46:53Z",
      "abstract_url": "http://arxiv.org/abs/2611.09792v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09792v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "econ.EM",
        "stat.ME"
      ]
    },
    {
      "title": "Assessing the Impact of Retrieval-Augmented Generation on Algorithmic Problem-Solving",
      "authors": [
        "Gabriel Souza Baggio",
        "Gabriel Machado Lunardi",
        "Guilherme Medeiros Machado",
        "Subhankar Maity",
        "Faiza Belbachir",
        "José Palazzo Moreira de Oliveira"
      ],
      "abstract": "Competitive programming problems pose distinctive challenges for Large Language Models, as they demand not only proficient code generation but also advanced algorithmic reasoning. This study presents a Retrieval-Augmented Generation approach specifically designed to enhance the capacity of Large Language Models to address such problems. Our approach integrates an embedding-based retrieval mechanism and a similarity search algorithm to identify analogous problems within a knowledge database, employing a dynamically adjusted threshold to mitigate retrieval noise. We conducted an evaluation using the CodeContests dataset, focusing on the test split problems that exhibit similarity to instances from the training split. The experimental framework considered three conditions: (i) problems provided solely with their original descriptions, (ii) problems supplemented with randomly selected examples from the training split, and (iii) problems augmented with similar examples retrieved through the proposed method. In addition, we incorporated a Min-K membership inference analysis to examine whether the generated code solutions were influenced by potential memorization of training data within the evaluated Large Language Models. This complementary perspective enables a deeper understanding of model behavior by distinguishing genuine generalization through retrieval from implicit recall of pretraining samples. Performance was assessed using both pass rates and the CodeForces Elo rating of the problems. The results provide empirical evidence that similarity-based retrieval substantially enhances Large Language Models performance in competitive programming scenarios, offering an scalable framework for improving code generation in algorithmic problem-solving contexts.",
      "published": "2026-11-06T17:46:28Z",
      "abstract_url": "http://arxiv.org/abs/2611.09791v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09791v1",
      "categories": [
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models",
      "authors": [
        "Lingyun Song",
        "Ziyao Chen",
        "Kang Pan",
        "Xiaolin Han",
        "Yifei Zhang",
        "Xiaoqi Wang",
        "Yudai Pan",
        "Xuequn Shang"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) require continuous factual updates to stay current, yet lifelong knowledge editing remains a daunting challenge. The primary obstacle is the severe Locality Erosion and Catastrophic Forgetting, where sequential edits inevitably interfere with pre-trained capabilities and prior updates. In this paper, we propose \\textbf{ROSE}, a \\textbf{R}isk-aware \\textbf{O}rthogonal \\textbf{S}ubspace navigation framework for lifelong knowledge \\textbf{E}diting. ROSE addresses forgetting by aiming to restrict parameter updates to the orthogonal complement of previously edited subspaces. Crucially, it introduces a risk-guided mechanism that utilizes a dynamically computed risk mask to safeguard specific factual parameter subspaces essential for model stability. During inference, ROSE employs a multi-granularity knowledge integration strategy, featuring prototype-anchored semantic gating and perturbation-driven subspace fusion to precisely activate edited knowledge for relevant queries while limiting interference on unrelated inputs. Comprehensive experiments on two prominent multimodal benchmarks show that ROSE is competitive with state-of-the-art approaches across five key metrics, while maintaining outstanding stability over $1,000$ sequential updates. Our code is available at https://github.com/sunlitsong/ROSE.",
      "published": "2026-11-06T17:32:49Z",
      "abstract_url": "http://arxiv.org/abs/2611.09770v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09770v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Language as Trait, Politeness as Strategy: Controlled Tests of Linguistic Effects in LLM Social-Deduction Agents",
      "authors": [
        "Yuchen Li",
        "Lisa Liu",
        "Runqi Zou",
        "Shuxuan Kuang",
        "Qingquan Zhang",
        "Julian Togelius"
      ],
      "abstract": "Large language model (LLM) agents coordinate through conversation, and the same message can be phrased in simpler or more complex language, bluntly or politely. We test whether such differences change how other agents judge and act, using the LURK Werewolf framework, which separates each utterance's speech-act predicate from its natural-language realization. We rewrite only the realization, check that the game content is preserved, and replay each game day from saved checkpoints against a re-sampled control. We vary linguistic range consistently across a speaker's messages and politeness strategy per utterance. Across 30 games and 4,335 replays, neither manipulation changes the trust other agents place in the speaker, and no effect on votes or night actions survives correction for multiple comparisons; the largest uncorrected effect is that shifting range away from a neutral paraphrase changes about 4 percentage points more of the other players' votes. In natural play, 88% of face-threatening speech is bald on-record regardless of role.",
      "published": "2026-11-06T17:31:01Z",
      "abstract_url": "http://arxiv.org/abs/2611.09768v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09768v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Safety Is $Π_1$, Provability Is $Σ_1$: One Framework for the Hugging Face Breach and OpenAI's Mathematical Proofs",
      "authors": [
        "Jinho Kang",
        "Jinsook Kim"
      ],
      "abstract": "OpenAI's agents breached Hugging Face in July 2026, and the same company released more than seven hundred AI-generated manuscripts of mathematical proofs in October 2026. The first event is read as a failure in security, while the second is read as a success in mathematics. We read both through the hybrid framework of our paper (Kim and Kang, 2026, arXiv:2611.01914), in which coding, proving, and learning from data are derivations by one and the same deductive mechanism, distinguished by the kinds of premises from which they are derived. Under this framework, the AI systems behind the two events ran the same loop, in which a statistical proposer produces candidate exploits or proofs and a deductive verifier settles each one by a derivation. What separates the events is the logical form of the statement wanted by each of the two domains, security and mathematics. The safety of a program is a $Π_1$ statement, for it says that no exploit exists, which is not assured by any finite list of tests that have been passed by the program. By contrast, the provability of a sentence is a $Σ_1$ statement, for it says that a proof exists, which is assured by a single instance, namely a proof that has been checked by the deductive verifier. The theorem behind both is Turing's Undecidability Theorem: the halting set is $Σ_1$ but not decidable, so its complement is $Π_1$ and not $Σ_1$, which is to say that membership has a finite witness but non-membership has none. We prove that attaching a statistical proposer to a deductive verifier does not change this undecidability result, however good the proposer is. We then discuss what does assure safety and where the residual risk goes.",
      "published": "2026-11-06T17:10:40Z",
      "abstract_url": "http://arxiv.org/abs/2611.09736v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09736v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CR",
        "cs.LO"
      ]
    },
    {
      "title": "FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding",
      "authors": [
        "Donglin Zhuang",
        "Pragaash Ponnusamy",
        "Xiaoxia Wu",
        "Shuaiwen Leon Song",
        "Zhongzhu Zhou"
      ],
      "abstract": "Large-language-model decode attention streams the key-value (KV) cache for every generated token. Packing that cache into 2-bit or 4-bit quantized integers reduces memory traffic but does not automatically translate into latency saving. Existing 2-bit fused kernels reach under 31%/15% of peak HBM throughput on H100/B200. Converting each value and applying its scale and offset before feeding Tensor Cores is a major bottleneck because instruction-issue throughput is scarce on modern GPUs. Instead of applying scale and offset to each KV value, FOSCAR uses affine folding to apply them to the other operand or the output of decode-attention matrix products, depending on the quantization scheme (token-wise or channel-wise). This reduces scale-and-offset operations by up to two orders of magnitude while preserving the dequantized attention output in exact arithmetic. FOSCAR also uses bit manipulation to feed KV operands to Tensor Cores efficiently, requiring only 1 and 0.19 ALU instructions per INT2 element on H100 and B200, respectively. Finally, FOSCAR builds warp-specialized GPU pipelines for Hopper (H100) and Blackwell (B200) using wgmma and tcgen05. On H100, FOSCAR INT2 and INT4 kernels achieve geo-mean speedups of $3.36\\times$ and $2.83\\times$ over FlashInfer BF16, and $2.03\\times$ and $1.71\\times$ over TensorRT-LLM FP8. On B200, the corresponding INT2 and INT4 speedups are $2.67\\times$ and $2.30\\times$ over FlashInfer BF16, $1.48\\times$ and $1.28\\times$ over TensorRT-LLM FP8, and $1.16\\times$ and $1.00\\times$ over TensorRT-LLM NVFP4. We integrate FOSCAR into SGLang to benchmark serving efficiency. On long-context serving workloads, FOSCAR increases peak aggregate throughput by up to 170% and 341% over 2-bit and 16-bit baselines, respectively. We open-source FOSCAR as a kernel library at https://github.com/FutureMLS-Lab/FOSCAR.",
      "published": "2026-11-06T17:00:25Z",
      "abstract_url": "http://arxiv.org/abs/2611.09722v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09722v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Intent-Aware Caching for Efficient LLM Serving",
      "authors": [
        "Kazi Hasan Ibn Arif",
        "JinYi Yoon",
        "Dimitrios S. Nikolopoulos",
        "Hans Vandierendonck",
        "Deepu John",
        "Bo Ji"
      ],
      "abstract": "Modern LLM serving systems use prefix caching to accelerate multi-turn conversational workloads. They cache the key-value states of previous conversation prefixes and reuse them when subsequent requests share the same prefix, to avoid redundant prefill computation and reduce serving latency. However, achieving a high cache hit rate remains challenging because cache reuse depends on whether a conversation continues and how soon the next turn arrives. Our analysis of real workloads shows that user intent is a useful signal for this behavior, as different intents exhibit different continuation probabilities and inter-turn delays. Based on this key insight, we propose Intent-Aware Caching (IAC), an online eviction policy that classifies requests by intent, learns lightweight reuse statistics from recent serving logs and leverages these statistics to guide an oracle-inspired eviction strategy. Extensive evaluation on real workloads (WildChat and ShareChat) shows that IAC reaches a 34.6% cache hit rate. This substantially outperforms both the previous learning-based approach (18.8%) and traditional methods such as least recently used (LRU, 12.2%), while reducing average serving latency by up to 46.1%.",
      "published": "2026-11-06T16:56:48Z",
      "abstract_url": "http://arxiv.org/abs/2611.09715v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09715v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ICare-DT: a digital twin for continuous forecasting of intensive care patients",
      "authors": [
        "Yanke Li",
        "Yahang Qi",
        "Yumi Kim",
        "Diego Paez-Granados"
      ],
      "abstract": "Digital twins of intensive care patients should update forecasts from new data and predict treatment responses, but few studies test whether routine records support either. We propose ICare-DT, a Transformer-based twin that updates a latent patient state hourly from new measurements without retraining. We trained it on 140,473 eICU-CRD stays to forecast vital signs crossing their training-set means. On test data, later forecasts beat the twin's own hour-6 forecasts of the same windows in all 12 settings (+0.073 to +0.153 AUROC). On validation data, controls that withheld or swapped the new hours traced the gain to the patient's own measurements. Without refitting, the model kept its gain on a second database (8 of 9 settings, +0.025 to +0.114 AUROC) in a prespecified evaluation of the previously analyzed MIMIC-IV test split. Refitted within MIMIC-IV, a twin state fine-tuned on clinical endpoints gave gradient boosting a significant gain for acute kidney injury (+0.0035 AUROC; hypotension +0.0010, not significant). To test response prediction, we built a benchmark that replaces the treatments in real records with simulated doses and injects responses of known size, visible only at measurement times. Ridge models supervised with injected effects recovered them from real records only through frequently measured variables (heart rate) read at their measurement times. We asked standard learners to predict these effects for a dosing regime no patient had received. No standard learner outperformed predicting zero effect at an out-of-support query. When told how the simulated responses were generated, a separate learner recovered the effects from the same records without seeing their true values (R2 0.30 to 0.40 at one effect size). Routine records support forecasts that improve with each patient's new data, and our benchmark is a testbed for models that learn how responses are generated.",
      "published": "2026-11-06T16:55:37Z",
      "abstract_url": "http://arxiv.org/abs/2611.09712v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09712v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Attend at the Right Resolution: Head-Adaptive Sparse Attention via Query-Key Interaction Structure",
      "authors": [
        "Wajih Hassan Raza",
        "Ahsan Bilal",
        "Yidan Shen",
        "Renjie Hu",
        "Xin Fu",
        "Yu Wen"
      ],
      "abstract": "Attention is a major scalability bottleneck in large language models (LLMs), with its computational cost and key-value (KV) memory traffic becoming increasingly prohibitive as context length grows. Sparse attention addresses this bottleneck by restricting each query to a subset of the context, but its effectiveness hinges on identifying relevant regions both cheaply and reliably. Existing training-free methods primarily rely on runtime relevance estimates, with selection dimensionality and granularity typically governed by fixed heuristics rather than the intrinsic interaction structure of pretrained attention heads. We take a different perspective: the query-key operator of each pretrained head defines a characteristic response structure that determines the resolution needed for sparse selection, both across feature directions and across context tokens. Building on this insight, we propose Head-Adaptive Sparse Attention (HASA), a training-free sparse attention framework that jointly adapts selection resolution along these two dimensions. Along the feature dimension, the singular spectrum of each head's query-key operator determines an effective rank, yielding a compact, head-specific response space. Along the sequence dimension, neighboring regions are progressively merged according to the directional coherence of their token responses in this same space, while heterogeneous regions remain separate. Exact attention is then computed using the original full-dimensional KV states of the selected regions. HASA captures 88.1% of dense-attention mass and, compared with UNIQUE, a strong recent sparse-attention baseline, improves accuracy by up to 3.47 and 2.57 points on RULER and LongBench-Pro, respectively, and accelerates self-attention by up to 1.32x, while achieving up to 14.07x speedup over dense attention.",
      "published": "2026-11-06T16:45:26Z",
      "abstract_url": "http://arxiv.org/abs/2611.09700v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09700v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Verifier Noise Is a Temperature: Testing Tilt Predictions in Finite-Budget LLM-Guided Search",
      "authors": [
        "Xiaomeng Li"
      ],
      "abstract": "LLM-guided search loops propose a modified candidate, score it, and keep it if it scores better. When both candidates are re-scored at every comparison, as in the minibatch acceptance gates of prompt optimizers, keep-if-better is a reversible Markov chain if the score difference is logistic and the mutation operator is reversible, and its stationary law is the mutation equilibrium tilted by exp(beta f). For any continuous symmetric noise difference the same tilt is the stationary law of a small-step diffusion limit, with beta = 4F'(0); for a minibatch gate beta grows with the square root of the minibatch size. Caching the incumbent's score instead inflates the recorded score. We test the tilt as a prediction for LLM-guided program search, committing each prediction from calibration data before the search runs. The level after 150 steps rises with the minibatch size in six of seven settings, but its predicted value fails: on bin packing, search attains 62-88% of the predicted gain when the proposer edits the parent and 109-119% when it writes new heuristics. The attained fraction falls with the parent-child quality correlation of unselected mutation; a registered test on two new settings confirmed the direction, but not the quantitative agreement after a correction. Further registered tests found that runs started from a good program kept losing quality over 300 steps, more so from starts that search had selected, by a margin that passed its registered rule narrowly. Runs from different starts had not reached a common level, so whether the tilt describes the stationary law of LLM editing remains open. A registered test in prompt optimisation on frozen task-model scores failed: search exceeded the tilt, and the bin-packing relation did not transfer.",
      "published": "2026-11-06T16:44:08Z",
      "abstract_url": "http://arxiv.org/abs/2611.09699v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09699v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Goodhart in the Planner: Sampling-Based Control Exploits Rewards Learned from Human Preferences",
      "authors": [
        "Soovadeep Bakshi",
        "Apurba Bose"
      ],
      "abstract": "Robots increasingly choose actions by searching against a reward model learned from human preferences, in model-predictive control or best-of-$N$ selection; we measure what this does to the true objective. On HalfCheetah and Hopper with crowd-sourced labels, and on Meta-World drawer-open and plate-slide with synthetic labels and blind labels from the two authors, harder MPPI search cuts true return by up to 76%: with human labels the drawer opens in 0 of 20 runs against 10 of 10 for the cloned policy the planner starts from, while held-out label accuracy stays at 0.75-0.97. On plate-slide human labels are exploited less than synthetic ones, and on window-open search helps. True return roughly follows how far the planner's weights move from its proposals ($\\sqrt{\\mathrm{KL}}$) for both weighting rules we test, and best-of-$N$ with a verifier from synthetic labels falls below a single noisy action. We prove that $χ^2$-regularized MPPI at a temperature set by the scorer's error spread on the proposals is worst-case optimal, that no positive gain can be certified once that spread exceeds the scorer's own, and that an episode's return splits exactly into per-step terms, of which the guarantee covers the selection terms. Calibrated from 100 proposal pairs ranked by true return (an oracle) on start states the evaluation never uses, this rule is the safest fix we test: with each of five independent sets it succeeds in every arm run and keeps on average 86-94% of the cloned policy's return on plate-slide and drawer-open, though it is cautious where search helps. Calibration from a person's comparisons cannot detect a bias the scorer learned from that person; correcting for the pairs people cannot tell apart, or a second person, limits the damage.",
      "published": "2026-11-06T16:22:17Z",
      "abstract_url": "http://arxiv.org/abs/2611.09670v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09670v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Breaking the Privacy Promise: Gradient Inversion Attacks on Federated Low-Rank Adapted Vision Transformers",
      "authors": [
        "Ritik K. Badiya",
        "Siladittya Manna",
        "Ashmit Sinha",
        "M Yashwanth",
        "Anirban Chakraborty"
      ],
      "abstract": "Parameter-efficient fine-tuning (PEFT) has been increasingly adopted to fine-tune pretrained models with low computational cost and is therefore also preferred in the federated learning paradigm to reduce computational and communication overhead. In this work, we investigate the sparsely explored problem of the risk of private data leakage in LoRA-adapted Vision Transformers (ViTs), where low-rank updates weaken the gradient signal available to a server with access to individual client updates. We show that quality of reconstruction is limited by two coupled ambiguities: reduced information in LoRA gradients and missing spatial cues from frozen positional embeddings. To address this, we propose a spatially anchored generative inversion framework that incorporates server-prescribed trainable positional embeddings into ViT blocks during fine-tuning at clients. The resulting positional gradients act as layer-wise spatial anchors, providing additional positional cues, while the generative prior keeps reconstructions on a realistic image manifold, reducing duplicate-data and patch-permutation artifacts, {which produce similar gradients as those produced when the target client images are used}. Adding our proposed spatial anchoring to GIAS improves PSNR by 10.14 dB, SSIM by 0.45, and LPIPS by 0.098 on CIFAR-10, as well as PSNR gains of 7.83 and 11.74 dB on high-resolution CelebA and CARS datasets, respectively, in the FedSGD setting. Such gains in reconstructed image quality metrics are also maintained at higher batch sizes and in FedAvg settings. These results indicate that LoRA-based PEFT does not, by itself, prevent data leakage in FL, and carefully incorporated positional embeddings can help in exposing private information.",
      "published": "2026-11-06T16:02:20Z",
      "abstract_url": "http://arxiv.org/abs/2611.09652v1",
      "pdf_url": "https://arxiv.org/pdf/2611.09652v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CR",
        "cs.DC",
        "cs.LG"
      ]
    }
  ]
};
