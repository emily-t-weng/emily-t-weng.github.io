const PAPERS_DATA = {
  "last_updated": "2026-06-10 04:24:09 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "A Unifying Lens on Supervised Fine-Tuning Through Target Distribution Design",
      "authors": [
        "Tong Xie",
        "Yuanhao Ban",
        "Yunqi Hong",
        "Sohyun An",
        "Yihang Chen",
        "Cho-Jui Hsieh"
      ],
      "abstract": "Supervised fine-tuning (SFT) typically maximizes the likelihood of every token in a demonstrated trajectory. However, an observed token can be non-unique, noisy, or misaligned with the model prior. Strictly fitting toward this one-hot target may be suboptimal, especially when the pretrained model encodes a rich knowledge prior. In this work, we reinterpret SFT as target distribution design: instead of studying only the loss objective, we analyze the token-level target that the loss drives the model to match. We introduce the Q-target framework, which decomposes SFT supervision into two explicit choices: (1) how strongly to rely on the observed token, and (2) how to allocate the remaining probability mass over alternatives. This perspective unifies many existing SFT variants as implicit choices of the target distribution Q. Building on this view, we propose Target-SFT which constructs the training objective directly from the desired target distribution. This method consistently outperforms across the ten reasoning dataset-model settings evaluated, showing the effectiveness of this target-based approach. Overall, our formulation reveals a more fundamental design principle for SFT training and opens a broader search space for SFT objectives.",
      "published": "2026-06-09T17:59:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.11189v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11189v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "EEVEE: Towards Test-time Prompt Learning in the Real World for Self-Improving Agents",
      "authors": [
        "Weixian Xu",
        "Shilong Liu",
        "Mengdi Wang"
      ],
      "abstract": "In this paper, we propose EEVEE, the first multi-dataset test-time prompt learning framework for LLM agents, enabling test-time prompt learning under real-world task streams. Existing methods are largely designed for single-dataset settings, while real-world applications require models to handle heterogeneous input streams drawn from multiple datasets, domains, and task distributions, limiting their practical applicability. To mitigate cross-dataset interference, EEVEE introduces a router that partitions incoming inputs into task clusters and assigns them to suitable prompt configurations. This design is optimized via a router-prompt co-evolution strategy, which employs interleaved router and prompt learning phases to address their mutual dependency. Experiments across multiple datasets demonstrate that the framework improves robustness under heterogeneous data streams while maintaining single-benchmark learning capability and efficiency. Specifically, EEVEE improves average multi-benchmark scores by 10.38 and 24.32 points over Qwen3-4B-Instruct and DeepSeek-V3.2, surpassing SOTA methods GEPA and ACE by up to 37.2% and 48.2%.",
      "published": "2026-06-09T17:57:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.11182v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11182v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "The Role of Feedback Alignment in Self-Distillation",
      "authors": [
        "Semih Kara",
        "Oğuzhan Ersoy"
      ],
      "abstract": "Conditioning a language model on additional context, such as feedback on a previous attempt, typically improves its response. Self-distillation trains the model to retain this improvement when the context is not present. The method works by matching the model's output distribution under two settings: a student that sees only the question, and a self-teacher that also sees the context. What the model learns therefore depends on what context the self-teacher receives, yet the design of this context remains largely unexplored. We study context design for self-distillation by training a solver on feedback from a frozen critic. We compare three conditions: (i) a binary reward (GRPO), (ii) the reference solution, and (iii) a step-by-step critique aligned to the solver's reasoning trace. Step-aligned critique yields the largest gains, outperforming GRPO by 16.11 points and reference-solution-conditioned self-distillation by 5.27 points (Avg@12). Per-token advantage analysis reveals why: step-aligned feedback targets only the tokens where reasoning fails, leaving correct behavior intact. Conditioning on the reference solution, by contrast, pressures the model to change its behavior at every token (even correct steps) because an alternative derivation inevitably differs in phrasing and approach. This suggests that structural alignment between feedback and the solver's reasoning is a key driver of self-distillation effectiveness.",
      "published": "2026-06-09T17:50:09Z",
      "abstract_url": "http://arxiv.org/abs/2606.11173v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11173v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Flaws in the LLM Automation Narrative",
      "authors": [
        "George Perrett",
        "Javae Elliott",
        "Jennifer Hill",
        "Marc Scott"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly described as performing at the level of human experts on knowledge economy tasks. These claims are primarily based on how LLMs perform on benchmarking tasks that measure average performance across standardized datasets. Primary limitations of many benchmarking tasks are that they often measure performance based on content directly included in LLM training data, and they frequently do not assess the reliability of LLM performance or the magnitude of LLM errors. However, in high stakes contexts, these qualities are critically important. Through a novel LLM benchmarking task that requires writing computer code to complete a data analysis task, we compare the performance of a frontier LLM against submissions from human experts and explicitly measure the variance of responses and the magnitude of errors. Our study reveals that the human experts perform better on average on a range of metrics and demonstrate less variability in performance. Our results provide evidence that LLMs do not consistently perform at the level of human experts and demonstrate the importance of measuring variance and assessing error magnitude in LLM benchmark evaluations.",
      "published": "2026-06-09T17:46:10Z",
      "abstract_url": "http://arxiv.org/abs/2606.11166v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11166v1",
      "categories": [
        "stat.OT",
        "cs.AI"
      ]
    },
    {
      "title": "ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models",
      "authors": [
        "Wenhao Liu",
        "Hao Shi",
        "Yunhe Li",
        "Weizhi Fei",
        "Xiangyuan Wang",
        "Mengzhe Ruan",
        "Hanxu Hou",
        "Peisong Wang",
        "Linqi Song",
        "Shuang Qiu"
      ],
      "abstract": "Long chain-of-thought (CoT) trajectories in large language model (LLM) reasoning cause severe inference bottlenecks due to rapid key-value (KV) cache growth. Current decoding-time compression methods mitigate this issue via token eviction, but typically assume a uniform budget distribution across all layers and heads. In contrast, existing non-uniform budget allocation methods are predominantly designed for the static prompt prefill phase, and they do not capture the stepwise context demands of autoregressive reasoning. To bridge this gap, we propose ReasonAlloc, a training-free framework that recasts decoding-time KV compression as a hierarchical budget allocation problem. ReasonAlloc operates at two complementary levels: an offline layer-wise preallocation strategy captures an architecture-driven demand pattern which we call ``\\textit{Reasoning Wave}'', while an online head-wise strategy reallocates resources during decoding to information-rich heads based on real-time utility. Evaluations on mathematical reasoning benchmarks (MATH-500, AIME~2024) using DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B, and AceReason-14B show that ReasonAlloc outperforms uniform-budget R-KV, SnapKV, and Pyramid-RKV (a baseline enforcing a static, monotonically decreasing layer budget), with the largest gains at small budgets (128-512 tokens). ReasonAlloc is plug-and-play with existing token-eviction policies and introduces negligible inference-time overhead.",
      "published": "2026-06-09T17:44:23Z",
      "abstract_url": "http://arxiv.org/abs/2606.11164v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11164v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "ABC-Bench: An Agentic Bio-Capabilities Benchmark for Biosecurity",
      "authors": [
        "Andrew Bo Liu",
        "Samira Nedungadi",
        "Bryce Cai",
        "Alex Kleinman",
        "Harmon Bhasin",
        "Seth Donoughe"
      ],
      "abstract": "Large language models (LLMs) are rapidly acquiring capabilities relevant to biological research, from literature synthesis to interpretation of experimental data. Increasingly, LLM agents can also perform in silico biology tasks that previously required experienced human biologists. These emerging AI capabilities offer new opportunities for scientific discovery and biomedical advances, but they also shift the landscape of biosecurity risks. To address this, we introduce the Agentic Bio-Capabilities Benchmark (ABC-Bench), a suite of tasks to measure agentic biosecurity-relevant capabilities. ABC-Bench evaluates LLM agents on both benign and dual-use biology tasks: writing code to operate liquid handling robots, designing DNA fragments for in vitro assembly, and evading DNA synthesis screening. These tasks require a combination of biology and software expertise. All tested LLM agents outperformed the median expert human baseliner on all three tasks. Agents performed highly on tasks drawing on published knowledge and well-documented protocols, and more weakly on a task requiring novel bioinformatics reasoning. In three wet-lab validation experiments, we found that OpenAI's o4-mini-high produced scripts that, when run on an OpenTrons liquid handling robot, successfully assembled DNA with expected sequences.",
      "published": "2026-06-09T17:35:37Z",
      "abstract_url": "http://arxiv.org/abs/2606.11150v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11150v1",
      "categories": [
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Data assimilation for subsurface flow using latent diffusion model parameterization: performance of ensemble-Kalman and Monte Carlo techniques",
      "authors": [
        "Guido Di Federico",
        "Wenchao Teng",
        "Louis J. Durlofsky"
      ],
      "abstract": "Data assimilation (DA) in subsurface flow entails calibrating model parameters to match observed data, typically at wells, while preserving geological realism. Latent diffusion models (LDMs) provide efficient mappings from high-dimensional geological model space to a low-dimensional latent variable, reducing the dimensionality of the inverse problem while maintaining plausibility in posterior geomodels. However, the high nonlinearity in the LDM mapping may degrade the performance of Kalman-gain-based ensemble updates. We present a systematic comparison of DA algorithms applied to large-scale 3D channelized geomodels with hierarchical geological uncertainty. We compare model-space and latent-space DA using the ensemble smoother with multiple data assimilation (ESMDA), and demonstrate a key trade-off: model-space updates achieve significant uncertainty reduction but produce geologically unrealistic posterior models, while latent-space updates preserve realism but exhibit limited uncertainty reduction. Motivated by this, we explore rigorous Markov chain Monte Carlo (MCMC) and Sequential Monte Carlo (SMC) algorithms in the 3D-LDM latent space. To accommodate their high computational demands, we develop a fast surrogate flow model that approximates well-rate responses. MCMC and SMC are evaluated against ESMDA across three synthetic test cases, with DA performed in the LDM latent space. All models maintain geological realism due to the LDM parameterization. MCMC and SMC are consistent with one another and achieve lower data mismatch and more uncertainty reduction than latent-space ESMDA. Our overall results demonstrate that ensemble Kalman methods may provide overestimated posterior uncertainty with highly nonlinear parameterizations, while rigorous Monte Carlo sampling, enabled by fast surrogate models, can provide a more reliable alternative.",
      "published": "2026-06-09T17:29:47Z",
      "abstract_url": "http://arxiv.org/abs/2606.11140v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11140v1",
      "categories": [
        "physics.geo-ph",
        "cs.AI",
        "cs.LG",
        "stat.AP",
        "stat.ML"
      ]
    },
    {
      "title": "TRACE: A Unified Rollout Budget Allocation Framework for Efficient Agentic Reinforcement Learning",
      "authors": [
        "Heming Zou",
        "Qi Wang",
        "Yun Qu",
        "Yuhang Jiang",
        "Lizhou Cai",
        "Yixiu Mao",
        "Ru Peng",
        "Xin Xu",
        "Weijie Liu",
        "Kai Yang",
        "Saiyong Yang",
        "Xiangyang Ji"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) is a promising approach for enhancing reasoning and agentic behavior in large language models. However, rollout-intensive policy optimization is often limited by insufficient reward contrast, arising when overly simple or complex prompts generate low-variance feedback and when outcome-only rewards assign the same terminal assessment to every decision in a multi-turn rollout. Past efforts have focused on allocating available rollout resources to promising prompts, yet they only leverage sample informativeness at the prompt level and neglect variation in prefix-level informativeness across turns within the same rollout. This work targets multi-turn agentic RL by modeling each ReAct-style thought-action-observation turn as a semantically distinct node, allowing budget allocation to extend from prompt roots to turn-level prefixes with further continuations, which naturally forms tree-structured rollouts. We introduce Tree Rollout Allocation for Contrastive Exploration (TRACE), a unified rollout allocation framework that enhances reward contrast within a fixed sampling budget. Technically, TRACE allocates rollout budget to both prompt roots and intermediate prefixes that are most likely to yield mixed terminal rewards. A shared generalizable predictor estimates conditional success probability at these anchors from prefix histories to guide this allocation. The resulting adaptive tree structure enriches outcome-only feedback and amplifies the policy-update signal. Empirically, TRACE achieves competitive performance and efficiency gains on typical agentic benchmarks, e.g., improving Qwen3-14B Multi-Hop QA average accuracy by 2.8 points over competitive baselines at equal sampling cost.",
      "published": "2026-06-09T17:16:03Z",
      "abstract_url": "http://arxiv.org/abs/2606.11119v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11119v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Towards Autonomous Accelerator Design: FPGA Accelerator Generation with SECDA",
      "authors": [
        "Vinamra Sharma",
        "Xingjian Fu",
        "Jude Haris",
        "José Cano"
      ],
      "abstract": "Designing FPGA-based accelerators for modern artificial intelligence workloads requires exploring a large and complex hardware design space that involves architectural parameters, data flow strategies, and memory hierarchies, making the process very time consuming. While existing methodologies such as SECDA enable rapid hardware-software co-design through SystemC simulation and FPGA execution, identifying efficient accelerator configurations remains a largely manual process requiring extensive domain knowledge. SECDA-DSE is a framework that integrates Large Language Models (LLMs) into the SECDA ecosystem to guide design space exploration (DSE) of FPGA-based accelerators. It combines a structured DSE Explorer for generating candidate architectures with an LLM Stack that performs reasoning-guided exploration using retrieval-augmented generation and chain-of-thought prompting, coupled with a feedback loop for iterative and reinforced refinement. Building on our previous work introducing SECDA-DSE, this paper extends its evaluation by generating three accelerator designs, including element-wise vector multiplication, 2D convolution, and matrix transpose, and performing end-to-end execution on FPGA hardware. The results show that SECDA-DSE can generate SECDA-compliant accelerator designs that are successfully synthesized and executed on FPGA hardware. Furthermore, the generated designs capture kernel-specific trade-offs between compute parallelism and data movement, highlighting the potential of LLM-guided exploration to adapt architectural configurations across diverse workloads while reducing exploration time and the need for extensive human expertise.",
      "published": "2026-06-09T17:14:44Z",
      "abstract_url": "http://arxiv.org/abs/2606.11117v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11117v1",
      "categories": [
        "cs.AR",
        "cs.AI",
        "cs.PF"
      ]
    },
    {
      "title": "Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning",
      "authors": [
        "Zhiyuan Zhou",
        "Andy Peng",
        "Charles Xu",
        "Qiyang Li",
        "Tobias Springenberg",
        "Kevin Frans",
        "Sergey Levine"
      ],
      "abstract": "Expressive continuous control policies, such as diffusion and flow models, form the backbone of recent advances in scaling imitation learning for simulated and real robot control. While they are known to scale stably in the supervised imitation learning setting, incorporating them into reinforcement learning (RL) pipelines for policy improvement has proven more difficult. It often requires specialized training objectives or backpropagating through denoising processes, which cause well-known issues with stability and affect scalability. In this paper we study the question of whether simple policy improvement schemes at test time alone, leaving stable supervised policy training intact, can be a competitive alternative which sidesteps these issues. To this end, we propose QGF (Q-Guided Flow), an RL algorithm that performs policy optimization entirely at test time. QGF works by pre-training both a reference flow policy (via a standard behavioral cloning objective) and a value function critic and, at test time, using the value gradient to guide the reference policy to generate higher-value actions without any additional policy learning. Empirically, QGF outperforms prior test-time RL methods on single-task and goal-conditioned offline RL benchmarks with high-dimensional action spaces, and is competitive with state-of-the-art training-time algorithms while being much cheaper to run. Moreover, it exhibits favorable scaling with model size by avoiding the instability of actor-critic training, offering a practical and effective alternative RL algorithm with expressive policies.",
      "published": "2026-06-09T16:45:57Z",
      "abstract_url": "http://arxiv.org/abs/2606.11087v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11087v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Unifying Local Communications and Local Updates for LLM Pretraining",
      "authors": [
        "Pietro Cagnasso",
        "Eugene Belilovsky",
        "Edouard Oyallon"
      ],
      "abstract": "Communication-efficient pre-training of LLMs is increasingly important as training draws on compute distributed across clusters, data centers, and lower-bandwidth links. Many practical methods reduce communication frequency but still rely on synchronous All-Reduce operations that maintain identical model states and tie progress to global collectives. This can become a bottleneck when bandwidth or worker speed is heterogeneous. We introduce GASLoC, a novel decentralized pre-training algorithm that generalizes the notion of communication acceleration to the recently popular \"outer optimizer\" to allow a practical gossip-based training framework that is compatible with adaptive optimizers, allows for local optimizer steps, and can utilize sparse randomized peer communication. Empirically, on a number of standard LLM training tasks, we demonstrate that GASLoC outperforms state-of-the-art decentralized algorithms in single step per communication setting for a number of topologies and, unlike existing decentralized methods in the LLM setting, it allows to obtain performance competitive with DiLoCo when utilizing multiple local steps. In the heterogeneous bandwidth setting we demonstrate the advantage of GASLoC showing that it can significantly outperform DiLoCo.",
      "published": "2026-06-09T16:40:54Z",
      "abstract_url": "http://arxiv.org/abs/2606.11081v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11081v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Modeling Complex Behaviors: Multi-Personality Composition and Dynamic Switching in Vision-Language Models",
      "authors": [
        "Peiqi Jia",
        "Haonan Jia",
        "Ziqi Miao",
        "Linkang Du",
        "Yuntao Wang",
        "Zhou Su"
      ],
      "abstract": "With the widespread deployment of Multimodal Large Language Models (MLLMs) in social interaction, understanding and controlling their behavior under complex personality conditions is essential. This paper introduces explicit personality conditioning and establishes a systematic evaluation framework encompassing single-personality induction, multi-personality induction, and personality switching. Experiments show that personality induction improves image captioning performance but can impair performance on tasks requiring precise reasoning, such as visual question answering (VQA). Balancing and residual effects are observed during multi-trait composition and dynamic switching, indicating that model behavior is co-modulated by both previous and current personality constraints. Existing prompt-based personality induction methods show limited transferability to multimodal settings. Our work reveals the dynamic and complex nature of personality modeling in MLLMs and underscores the need for robust, tailored methods for personality induction and evaluation. The code will be released when the paper is accepted.",
      "published": "2026-06-09T16:34:37Z",
      "abstract_url": "http://arxiv.org/abs/2606.11074v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11074v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "T1-Bench: Benchmarking Multi-Scenario Agents in Real-World Domains",
      "authors": [
        "Genta Indra Winata",
        "Amartya Chakraborty",
        "Yuzhen Lin",
        "Swasthi P Rao",
        "Shikhhar Siingh",
        "Houhan Lu",
        "Nadia Bathaee",
        "Sriharsha Hatwar",
        "Paresh Dashore",
        "Anmol Jain",
        "Kshitij Tayal",
        "Xiuzhu Lin",
        "Anirban Das",
        "Sambit Sahu",
        "Shi-Xiong Zhang"
      ],
      "abstract": "Recent advances in reasoning and tool-calling capabilities of large language models (LLMs) have enabled increasingly capable agentic systems. However, existing benchmarks remain limited in task complexity, realism, and domain diversity, and often fail to capture interactions that span multiple domains, limiting their ability to evaluate agents in realistic multi-step settings that require sustained reasoning and coordination. To address these limitations, we introduce T1-Bench, a high-fidelity, comprehensive benchmark for evaluating agentic systems in realistic customer-facing, multi-domain environments, featuring interleaved scenarios that require structured reasoning across multi-turn user-assistant interactions and substantially increasing both compositional complexity and evaluative rigor across 25 domains of varying difficulty. We evaluate T1-Bench using 12 proprietary and open-weight models, providing a reproducible and standardized framework for assessing agent behavior, tool utilization, and conversational quality in complex, multi-step environments. We further complement automatic evaluation with human judgments to strengthen the assessment of qualitative performance. Overall, T1-Bench substantially advances prior benchmarks by increasing task complexity, interaction depth, and domain coverage in simulated multi-domain environments. To facilitate future research on agentic systems, we will publicly release data and evaluation code as open source.",
      "published": "2026-06-09T16:32:14Z",
      "abstract_url": "http://arxiv.org/abs/2606.11070v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11070v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "CIAware-Bench: Benchmarking Control Intervention Awareness Across Frontier LLMs",
      "authors": [
        "Joachim Schaeffer",
        "Thomas Jiralerspong",
        "Alexander Panfilov",
        "Guillaume Lajoie",
        "Jonas Geiping",
        "Yoshua Bengio",
        "Roland S. Zimmermann"
      ],
      "abstract": "AI control protocols oversee untrusted models by monitoring their actions and modifying potentially unsafe steps, often using a trusted model. This partially tampers with the untrusted model's trajectory. If the trusted model detects such an intervention, it may infer properties of the monitor and adapt to evade control. We introduce \\textbf{CIAware-Bench}, a benchmark for measuring \\textbf{c}ontrol \\textbf{i}ntervention (CI) awareness across frontier models. CIAware-Bench tests whether models can distinguish their own trajectories from those modified by a control intervention. The benchmark is comprised of a suite of four task domains (essay writing, BigCodeBench, Bash Arena, and SHADE-Arena), while varying trajectory watermarking, side-task presence, and the control protocol. Evaluating eleven frontier models, we find low to moderate CI awareness under default settings (up to 0.87; random chance balanced binary classification accuracy is 0.5) with substantial variation across task domains and model pairs. Detection is generally easier across model families, suggesting that models exploit provider-specific differences in style or post-training. Overall, CI awareness is not a fixed model-level property, and should be measured for each new model release and deployment scenario. We release CIAware-Bench to track CI awareness and inform control protocols whose interventions are harder to detect.",
      "published": "2026-06-09T16:24:16Z",
      "abstract_url": "http://arxiv.org/abs/2606.11063v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11063v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "What Fits (Into Few Tokens) Doesn't Overfit: Compression and Generalization in ML Research Agents",
      "authors": [
        "Martin Andres Bertran",
        "Aaron Roth",
        "Zhiwei Steven Wu"
      ],
      "abstract": "Reusing a held-out benchmark adaptively should, in principle, invite overfitting. Yet benchmark-driven machine learning (ML) has produced surprisingly little overfitting in practice. An attractive hypothesis is that successful ML strategies are highly compressible. We study this in the setting of LLM-driven research agents, where the hypothesis becomes directly testable via two complementary information bottlenecks. In \\emph{output compression}, an exploration agent adaptively searches for high-performance models using a validation set, and we test whether a fresh ``reproducer agent'' can reproduce its performance given only an extremely short prompt and the training data. In \\emph{input compression}, the explorer receives only one-bit feedback indicating whether each submitted model improves on the running best. Across 8 datasets spanning tabular classification, vision, language modeling, diffusion modeling, and reward modeling, we find that these bottlenecks have little effect on performance: short prompts and compressible feedback are sufficient to reproduce and find high-performance models. The hypothesis is falsifiable: when we deliberately induce validation-set overfitting, the results fail to reproduce with short prompts. Taken together, our results support a description-length explanation for the lack of overfitting in benchmark-driven ML: successful strategies occupy a low-complexity region of strategy space.",
      "published": "2026-06-09T16:12:55Z",
      "abstract_url": "http://arxiv.org/abs/2606.11045v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11045v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "AuRA: Internalizing Audio Understanding into LLMs as LoRA",
      "authors": [
        "Bo Cheng",
        "Lei Shi",
        "Zhanyu Ma",
        "Yuan Wu",
        "Jun Xu",
        "Jiuchong Gao",
        "Jinghua Hao",
        "Renqing He"
      ],
      "abstract": "Recent efforts to extend large language models (LLMs) to speech inputs typically rely on cascaded ASR-LLM pipelines, end-to-end speech-language models, or bridge/distillation-based adaptation. While these routes respectively reuse strong pretrained components, enable native speech-language interaction, or offer lightweight adaptation, they often suffer from transcript-interface latency, costly multimodal training, or sequential speech-language coupling. To address these limitations, we present AuRA, a method that distills audio encoding capability into the LLM. Specifically, AuRA feeds the same speech input to an ASR encoder (as a teacher) and a LoRA-adapted LLM (as a student) through a lightweight audio embedding layer, and uses layer-wise distillation to align the student's hidden states with corresponding teacher representations, thereby internalizing speech representations into lightweight LLM-side adaptations. Compared with cascaded and serial bridge methods, AuRA enables tighter speech-language joint modeling and efficient parallel end-to-end inference, while also reusing pretrained speech and language models rather than requiring large-scale multimodal training. On multiple speech-language benchmarks, AuRA consistently outperforms cascaded systems, speech-to-LLM adaptation baselines, and large-scale speech-language and multimodal models in both effectiveness and efficiency.",
      "published": "2026-06-09T16:05:23Z",
      "abstract_url": "http://arxiv.org/abs/2606.11033v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11033v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Superficial Beliefs in LLM Decision-Making",
      "authors": [
        "Gabriel Freedman",
        "Francesca Toni"
      ],
      "abstract": "We ask whether large language models (LLMs) merely imitate rationales when choosing between two options, or whether their choices reflect a systematic underlying decision structure. Using synthetic binary decision settings in which models choose between profiles defined by graded attributes, we compare the attribute a model says mattered most with the attribute that best explains its choice under a behavioural model fit to prior decisions. The behavioural model predicts held-out choices well, showing that model behaviour is systematically related to the visible attributes rather than being random. However, direct self-reports and a separate score-based judge recover the behaviourally inferred driver only partially. The resulting picture is neither one of arbitrary behaviour nor one of fully articulated belief - outputs are structured enough to support prediction, but explicit reasons track the recovered driver only imperfectly. This qualitative pattern persists across prompt-order and sampling perturbations, alternative behavioural models, targeted occlusion analyses, and structurally varied decision settings. We interpret this as evidence for ``superficial belief'' in LLM decision-making: models behave as if guided by probabilistic local priorities over attributes, while having only limited verbal access to the attributes that drive their decisions.",
      "published": "2026-06-09T15:54:35Z",
      "abstract_url": "http://arxiv.org/abs/2606.11016v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11016v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Structure from Reasoning, Numbers from Search: On-Premise Open LLMs as Structural Priors for Coupled MIMO Controller Tuning",
      "authors": [
        "Jiaxuan Chen",
        "Haonan Li",
        "Yang Shu"
      ],
      "abstract": "Tuning controllers for strongly coupled multi-input multi-output (MIMO) industrial processes is hard: decentralized classical auto-tuning ignores loop interaction, and local numerical optimization from natural initializations stalls in the resulting non-convex cost landscape. We ask whether on-premise open-source large language models (LLMs), which keep data on-site and need no plant model, can help. On a single-loop CSTR, classical relay-feedback tuning (IAE 0.106, near the 0.102 optimum) beats an LLM tuner (0.162): for simple loops the LLM adds nothing. The picture inverts on a strongly coupled quadruple-tank with conflicting set-points, scored by a penalized cost J = IAE + lambda*TV(u) that rewards tracking without chattering actuators. There, naive relay tuning (J ~ 28.6) and naive LLM tuning (29.7) are no better than open loop (22.7), and a local optimizer from balanced starts fails in 10/10 runs. A scaffolded open LLM instead reasons about the coupling, proposes the counter-intuitive asymmetric structure, and reaches J ~ 16.9 +/- 0.2 from any start; refining it with a classical optimizer attains the smooth global optimum (J ~ 12.0, 10/10 vs. 0/10), which even applies a non-obvious negative integral correction decentralized tuning cannot. A global optimizer (differential evolution) also reaches this optimum, so the LLM is not the only route; its advantage is sample efficiency and interpretability: a usable controller in 18 evaluations (where the global optimizer is worse than open loop) plus a stated rationale. This edge grows with dimension, reaching ~6x fewer evaluations on a 3x3 plant. The behaviour generalizes across four open models, and on a benign plant the LLM offers no advantage, sharpening the boundary. We contribute a reproducible benchmark delimiting when open LLMs help in control tuning: not as optimizers, but as a sample-efficient, interpretable structural prior.",
      "published": "2026-06-09T15:53:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.11015v1",
      "pdf_url": "https://arxiv.org/pdf/2606.11015v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Null-Space Constrained Low-Rank Adaptation for Response-Specified Large Language Model Unlearning",
      "authors": [
        "Bocheng Ju",
        "Jianhua Wang",
        "Chengliang Liu",
        "Xiaolin Chang"
      ],
      "abstract": "Large language model unlearning aims to suppress designated undesirable knowledge while preserving benign capabilities. Many unlearning objectives focus on suppressing undesired answers, while recent target-guided variants specify replacement behavior but still leave update locality largely unconstrained. This paper introduces \\emph{Null-Space Constrained Response-Specified Unlearning} (NSRU), a projection-constrained low-rank framework for controlled LLM unlearning. NSRU uses an explicitly structured safe target response to specify the desired behavior for each forget query, while suppressing the original undesired content. To localize adaptation, NSRU estimates per-module retain subspaces from benign hidden representations and uses an orthogonal-projected low-rank parameterization to confine LoRA updates to the null space of the retain subspace. The resulting objective jointly optimizes safe-target learning, undesired-response suppression, and retention preservation under this constrained parameterization. We provide a local first-order analysis showing that the projected update reduces retain-side perturbations while preserving editable directions for shaping forget-query behavior. Experiments on TOFU show that NSRU effectively suppresses extractable forget-set knowledge while improving retain QA performance, model utility, and safe-target alignment over representative baselines. On WMDP, NSRU keeps hazardous-domain accuracy near the random-choice region while preserving broad and domain-adjacent MMLU utility. Ablation studies support the complementary roles of safe-target supervision, undesired-response suppression, retention loss, and null-space projected updates, while sensitivity and robustness analyses indicate stable behavior across the tested hyperparameter and prompt variations.",
      "published": "2026-06-09T15:26:36Z",
      "abstract_url": "http://arxiv.org/abs/2606.10989v1",
      "pdf_url": "https://arxiv.org/pdf/2606.10989v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning",
      "authors": [
        "Renjie Mao",
        "Xiangxin Zhou",
        "Lvfang Tao",
        "Yixin Ding",
        "Yu Shi",
        "Yongguang Lin",
        "Yuheng Wu",
        "Honglin Zhu",
        "Qian Qiu",
        "Wenxi Zhu"
      ],
      "abstract": "Reinforcement learning with verifiable rewards (RLVR) has become standard for improving LLM reasoning. However, existing PPO-style trust-region mechanisms remain position-agnostic by enforcing uniform thresholds across all tokens independently. This pointwise treatment conflicts with autoregressive generation in two critical ways. First, uniform thresholds ignore autoregressive asymmetry. Early-stage deviations produce compounding sequence-level drift, causing static thresholds to under-regulate early divergence and excessively constrain late-stage exploration. Second, evaluating token-level divergence in isolation overlooks cumulative prefix drift, granting the same divergence allowance regardless of how far the conditioning history has already deviated from the rollout policy. To address this limitation, we propose CPPO (Cumulative Prefix-divergence Policy Optimization), a token-level masking rule that aligns updates with a finite-horizon policy-improvement bound via two coupled mechanisms. First, a position-weighted threshold imposes stricter limits at early positions whose effects persist longer, relaxing constraints for late-stage tokens. Second, a cumulative prefix budget tracks historical deviations, dynamically restricting further token-level deviation to prevent compounding errors along the prefix. Empirically, CPPO enhances training stability and significantly improves reasoning accuracy across various model scales.",
      "published": "2026-06-09T15:09:29Z",
      "abstract_url": "http://arxiv.org/abs/2606.10968v1",
      "pdf_url": "https://arxiv.org/pdf/2606.10968v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
