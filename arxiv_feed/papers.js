const PAPERS_DATA = {
  "last_updated": "2026-10-08 05:42:54 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Decoupling Exploration from Optimization in RLVR",
      "authors": [
        "Saif Punjwani",
        "Micah Goldblum"
      ],
      "abstract": "Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model's knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alternating between exploration and optimization. This decoupling allows us to aggressively scale exploration without degrading the student policy. Across seven mathematical reasoning benchmarks and two model families, ExpDis outperforms DAPO at the same wall-clock budget. Moreover, we observe improved pass@$k$ scaling, indicating that ExpDis produces models that generate more diverse correct solutions.",
      "published": "2026-10-07T17:59:26Z",
      "abstract_url": "http://arxiv.org/abs/2610.10536v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10536v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "SciExam for ENSO: Can AI Agents Build Climate Models?",
      "authors": [
        "Yinling Zhang",
        "Langchen Liu",
        "Dongbin Xiu",
        "Xueyan Zou",
        "Xu Kuang",
        "Mengdi Wang",
        "Shilong Liu"
      ],
      "abstract": "Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reconstruction and forecasting. The simplified forms of the stronger models are each compatible with one of the two competing explanations of ENSO's warm-cold asymmetry, an open debate that the task never mentions. Controlled runs of the top system under varied information suggest that its scores do not come from recalling the dated observational record and that the information it receives shapes how it builds its model. SciExam for ENSO can thus evaluate agent research where no answer is known, and the results suggest that agents can already build competitive models whose structures bear on questions that scientists still debate.",
      "published": "2026-10-07T17:52:52Z",
      "abstract_url": "http://arxiv.org/abs/2610.10513v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10513v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "physics.ao-ph"
      ]
    },
    {
      "title": "RECAST: Learning to Compute the Right Context through Adaptive Evidence Routing",
      "authors": [
        "Yilun Hao",
        "Krishna Sayana",
        "Isabella Ye",
        "James S Ren",
        "Sukhdeep Sodhi",
        "Craig Boutilier",
        "Chuchu Fan"
      ],
      "abstract": "Large language models are increasingly applied to tasks grounded in long, heterogeneous information sources. Conventional Retrieval-Augmented Generation (RAG) relies on fixed similarity-based retrieval, while agentic variants adapt queries and tool use but remain largely retrieval-centric. However, in many tasks, the evidence required for a solution is not explicitly present in any single source item. Instead, it must be derived through filtering, aggregation, or computation across multiple source items. In this work, we introduce RECAST (Routing Evidence through Computation, Access, and Synthesized Tools), a learned framework that formulates evidence construction as a sequential decision process over heterogeneous retrieval and computation operations, allowing evidence to be actively derived rather than merely retrieved. A lightweight RouterLM iteratively selects and formulates primitive operations or specifies customized operations for a frozen CompilerLM to translate into executable code. Once it judges the evidence sufficient, RouterLM passes the accepted evidence to a frozen AnswerLM to produce the final solution. We train RouterLM with supervised fine-tuning (SFT) followed by group relative policy optimization (GRPO). Across six heterogeneous benchmark families, RECAST achieves a mean success rate of 75.6%, outperforming the strongest large-model baseline by 15.9%. Moreover, training enables the Qwen3.5-9B RouterLM to outperform a training-free Gemini 3.5 Flash RouterLM by 5.0%. On three held-out benchmarks, RECAST improves over the strongest baseline by 15.0% on average, demonstrating strong zero-shot generalization across tasks and heterogeneous source representations.",
      "published": "2026-10-07T17:51:08Z",
      "abstract_url": "http://arxiv.org/abs/2610.10507v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10507v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Validity Without Ground Truth: What Stated-Preference Economics Offers the Evaluation of Language Models",
      "authors": [
        "Daniel Robert Kling Alexander",
        "Catherine Louise Kling"
      ],
      "abstract": "Many of the questions now put to large language models have no correct answer to score against: what a policy is worth, which option a user should choose, how to weigh competing values. Stated-preference economics has faced this problem for decades. It judges survey responses without knowing the true value, through a framework of validity and related concepts: content, construct, and criterion validity, reliability, incentive compatibility, and consequentiality. We argue that this framework is a general method for evaluating language models, and we set out what each concept means for LLM evaluation. We demonstrate the approach using a published water-quality stated preference economic valuation survey (Vossler et al. 2023) administered to six models. In this economic application, the validity tests take the form of predictions from economic theory: demand should slope down, and willingness to pay should respond to the scope of the good and to income. The tests separate the models sharply. Two older models fail the most basic test at a household income level of \\$75,000, and the two newest pass every test of theoretical validity we can score, but diverge on convergent validity. Passing validity tests shows that a model's answers are coherent, not that they are correct.",
      "published": "2026-10-07T17:51:07Z",
      "abstract_url": "http://arxiv.org/abs/2610.10506v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10506v1",
      "categories": [
        "cs.AI",
        "cs.CL",
        "econ.GN"
      ]
    },
    {
      "title": "Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts",
      "authors": [
        "Hejian Sang",
        "Zhengze Zhou",
        "Shayan Mohajer Hamidi",
        "Xiaomin Li",
        "Rohit Jain",
        "Alborz Geramifard"
      ],
      "abstract": "Multi-teacher on-policy distillation (MOPD) is used in two settings. In common-domain composition, several teachers score each student rollout from one prompt domain and their signals form a single target; in routed-domain distillation, prompts from different domains are assigned to the corresponding specialist. Both settings usually transfer each teacher's endpoint policy, which mixes what post-training changed with preferences inherited from the teacher's base. We introduce $Δ$-MOPD, which transfers each teacher's teacher-minus-base logit shift re-anchored at the student's frozen initialization, and compare it with endpoint supervision in both settings while holding teacher selection fixed. We first expose the mechanism that impedes endpoint transfer: inherited base pull can exceed the post-training shift. Removing it reduces the teacher-term norm ratio and target--student KL. Across our experiments, the results suggest that shift targets are particularly useful when teacher signals are combined at a state. With three composed teachers, $Δ$-MOPD exceeds endpoint composition by $4.11$ Math and $1.95$ five-benchmark points; with two, it matches endpoint accuracy. Under phased routing, it achieves higher mean performance in both phase orders and reduces the observed order gap from $10.50$ to $6.42$ points. Under interleaved routing, where each update involves one teacher, the two targets perform comparably. The phased results provide supporting evidence that the benefit may extend to signals accumulated across training phases. Target construction is thus an independent design axis in MOPD, complementary to teacher selection.",
      "published": "2026-10-07T17:27:46Z",
      "abstract_url": "http://arxiv.org/abs/2610.10460v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10460v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs",
      "authors": [
        "Linghao Meng",
        "Feng He",
        "Xuan Yang",
        "Junyuan Mao",
        "Pinze Ren",
        "Deqing Mu",
        "Hesen Yang",
        "Qiankun Li"
      ],
      "abstract": "Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning trajectory. We further find that properties of the hallucinated prompt contain substantial predictive signal for successful recovery, with a lightweight predictor achieving an AUROC of 0.847. These findings provide a behavioral view of post-hallucination reasoning, characterizing how LLMs resolve erroneous context and when successful recovery is likely to occur.",
      "published": "2026-10-07T17:25:23Z",
      "abstract_url": "http://arxiv.org/abs/2610.10455v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10455v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Good Self-Teacher Meets the Student Where They Are: Joint On-Policy Learning and Teaching",
      "authors": [
        "Randy Ardywibowo",
        "Arnav Dalal",
        "Jiantao Jiao"
      ],
      "abstract": "Reinforcement Learning (RL) from outcome rewards suffers from sparse supervision, particularly on difficult, long-horizon tasks where successful trajectories are rare and costly to generate. On-Policy Distillation (OPD) offers an attractive alternative by providing dense token-level supervision from a stronger teacher along the student's own generations. Self-distillation methods further remove the need for a separate teacher model by conditioning the same policy on privileged information to serve as its own teacher. However, privileged conditioning alone does not guarantee that the resulting distillation update improves the student. Indeed, privileged information can lead the teacher to solve tasks through shortcuts unavailable to the student, producing supervision poorly matched to the student's current behavior. Consequently, even a higher-performing teacher can provide guidance that degrades student performance. To address this, we analyze how the choice of privileged teacher affects the student's update. We derive a necessary and sufficient condition for the teacher's local distillation update to be a positive multiple of the student's reward gradient. Our analysis suggests that the teacher should not only perform well on the task, but also provide guidance suited to the student's current capabilities. This characterization motivates a practical teacher-training surrogate that combines outcome rewards with token-level Kullback-Leibler (KL) regularization toward the student. Based on this result, we propose Joint On-Policy Learning and Teaching (JOLT), which jointly trains a single policy in two roles: a privileged teacher using a KL-regularized objective, and an unprivileged student using dense on-policy distillation. Across mathematical reasoning, coding, tool use, and terminal use, JOLT improves training efficiency and performance, with further gains from student rewards.",
      "published": "2026-10-07T17:19:19Z",
      "abstract_url": "http://arxiv.org/abs/2610.10447v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10447v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Q-Learning with Scalar Adjoint Matching",
      "authors": [
        "Yonghoon Dong",
        "Minsung Yoon",
        "Jaehyuk Kim",
        "Jungwoo Park",
        "Changyeon Kim",
        "Jinwoo Shin"
      ],
      "abstract": "Flow policies capture rich and diverse action distributions, and fine-tuning them with off-policy RL to improve beyond the demonstrations has drawn growing interest. However, fine-tuning a flow policy against a learned value function is not trivial, because the policy generates its action over many flow steps. Adjoint matching offers a principled way to update the flow model itself by propagating value information from the final action back to each flow step, but it requires a vector--Jacobian product through the policy at every step, a cost that grows with the number of flow steps and the policy size. We observe that the batch-averaged velocity Jacobian of pretrained flow policies concentrates on its diagonal. Motivated by this finding, we derive a closed-form scalar adjoint that scales the value gradient at the final action by the flow time, eliminating the per-step vector--Jacobian products. We further find that controlling the critic's value at policy-generated actions is particularly important under the scalar adjoint. Based on these findings, we propose Q-learning with Scalar Adjoint Matching (SQAM), which combines the scalar adjoint with a value penalty at those actions. SQAM's gains concentrate on the four hardest OGBench domains, where its success rate exceeds that of the strongest baseline in each domain by 18 to 35 percentage points. To test whether SQAM extends to large pretrained policies, we also fine-tune a vision-language-action policy on a real bimanual robot. SQAM improves over supervised fine-tuning on all three tasks.",
      "published": "2026-10-07T17:13:35Z",
      "abstract_url": "http://arxiv.org/abs/2610.10437v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10437v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.RO"
      ]
    },
    {
      "title": "Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds",
      "authors": [
        "Yunxiao Zhao",
        "Changxiao Cai"
      ],
      "abstract": "Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection costs by state occupancies and exactly equals the expected number of decoding rounds. Unlike prior surrogate objectives, EDR introduces no auxiliary hyperparameters. We then derive an exact temporal-difference gradient that supports unbiased stochastic optimization from target-model rollouts. The same framework also yields an exact offline evaluator for round counts, enabling paired drafter comparisons on shared target rollouts without running speculative decoding. Finetuning two state-of-the- art drafters, DSpark and DFly, with EDR consistently improves mean accepted length and outperforms existing training objectives across nine benchmarks spanning math reasoning, code generation, and chat.",
      "published": "2026-10-07T16:56:29Z",
      "abstract_url": "http://arxiv.org/abs/2610.10411v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10411v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "stat.ML"
      ]
    },
    {
      "title": "SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions",
      "authors": [
        "Yizhen Xie",
        "Mengyang Liu"
      ],
      "abstract": "As option markets grow and AI advances, agentic systems for option trading are gaining increasing attention. Language-model-based agents can reason over contextual information such as news, but option trading presents a particularly challenging decision problem: a single stock can have thousands of contracts, and the agent must decide both which contracts to trade and how to combine them. Existing approaches often sidestep this complexity by restricting the policy to a fixed strategy structure, such as a straddle, limiting their ability to switch strategies as market conditions change. We present SOTA (Stock Options Trading Agents), an agentic trading framework for structured option-strategy selection. SOTA abstracts the large option universe into strategy-level decisions while deterministic resolvers handle portfolio implementation. We develop SOTA by post-training Qwen3.8-27B with supervised fine-tuning followed by reinforcement learning. SOTA is evaluated on options on nine large-cap U.S. equities and SPY against rule-based and machine-learning strategy selectors in the same trading environment. Over a six-month out-of-sample period, SOTA earns an 18.3% total return with a Sharpe ratio of 1.60 and a maximum drawdown of 8.96%. We also document an asymmetric role of news: news improves frontier-teacher trajectories, but retaining news during reinforcement learning reduces out-of-sample return from 18.3% to -2.7%.",
      "published": "2026-10-07T16:55:05Z",
      "abstract_url": "http://arxiv.org/abs/2610.10407v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10407v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "q-fin.PM",
        "q-fin.TR"
      ]
    },
    {
      "title": "Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models",
      "authors": [
        "Maverick Morales",
        "Tomáš Dominik",
        "Vermut Gao",
        "Katrina Shirey",
        "Paulius Rimkevičius",
        "Aaron Schurger",
        "Uri Maoz"
      ],
      "abstract": "Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respond truthfully, falsely, or without regard for truth. Across all three models, truth-directed responding elicited fewer reasoning tokens than both lie-directed and truth-indifferent responding. These findings show that explicitly prompted untruthful response policies can produce robust group-level differences in test-time reasoning-token use. While not yet establishing reasoning-token count as a detector of spontaneous deception or general misalignment, our results are a proof of concept that it can serve as a simple, content-independent candidate signal for differentiating untruthful from truthful model behavior when raw reasoning traces are unavailable or unreliable. Future work should test instance-level detection rates, out-of-distribution generalization, learned deceptive policies, hidden objectives, and robustness under adversarial pressure.",
      "published": "2026-10-07T16:53:11Z",
      "abstract_url": "http://arxiv.org/abs/2610.10405v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10405v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "TaoD2C-Bench: Benchmarking MLLMs for Industrial UI Code Generation Beyond Visual Fidelity",
      "authors": [
        "Chengwei Shi",
        "Yunnong Chen",
        "Tingting Zhou",
        "Qiang Lu",
        "Shiyu Yue",
        "Xinyuan Hu",
        "Jianfang Ru",
        "Liuqing Chen"
      ],
      "abstract": "A key challenge for multimodal large language models (MLLMs) is moving beyond visual recognition to constraint-aware cross-modal reasoning. This involves combining visual cues with information from other modalities to understand elements' relationships under domain-specific rules. This challenge is acutely evident in industrial design-to-code (D2C), which converts user interface (UI) designs into code and requires MLLMs to connect design images with disorganized layer metadata, infer component and layout implementation requirements, and realize them in code under target-library constraints. However, these capabilities remain insufficiently evaluated in realistic industrial settings. To fill this gap, we present TaoD2C-Bench, a benchmark for evaluating MLLMs' ability to generate UI code that satisfies implementation requirements in industrial applications. The TaoD2C dataset consists of 2,861 production designs from 17 commercial platforms with 97,652 expert annotations across four categories: Component, Group, Alignment, and Position. These annotations distinguish required constraints from permitted implementation choices. TaoD2C-Bench defines three tasks: end-to-end UI code generation, requirement inference, and requirement realization. Evaluating eight MLLMs reveals substantial gaps in generating UI code that satisfies implementation requirements, alongside distinct performance profiles in inference and realization. We further show that MLLMs' visual reconstruction ability does not necessarily imply an ability to generate code that meets these requirements. We release TaoD2C to support research on industrial UI code generation.",
      "published": "2026-10-07T16:38:10Z",
      "abstract_url": "http://arxiv.org/abs/2610.10374v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10374v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning",
      "authors": [
        "Junkai Chen",
        "Yuhao He",
        "Qianshan Wei",
        "Junxiang You",
        "Jingwen Shao",
        "Junkai Lin",
        "Zhongkai Yue",
        "Xiaotian Ye",
        "Zhengbo Jiao",
        "Jiali Cheng",
        "Zhijie Deng",
        "Kening Zheng",
        "Ruiqi Liu",
        "Hadi Amiri",
        "Yi Yu",
        "Zhenan Sun",
        "Qi Li",
        "Ka-Ho Chow",
        "Sijia Liu",
        "Liang Wang",
        "Jiaqi Li",
        "Shu Wu"
      ],
      "abstract": "As multimodal large language models (MLLMs) become more capable and widely deployed, concerns about privacy and safety have become increasingly pressing. Machine unlearning offers one approach to addressing these concerns by removing designated information from trained models while preserving unrelated capabilities. However, fragmented implementations and evaluation protocols, incomplete robustness testing, and limited understanding of metric reliability make progress in MLLM unlearning difficult to assess systematically. We introduce Open-MMUnlearning, an open-source, extensible framework that integrates target-model preparation, multimodal data processing, unlearning, and evaluation through shared interfaces and structured configurations. The framework supports five benchmarks spanning privacy, safety, and copyright, eight MLLMs from four model families, and twelve unlearning methods. Its evaluation suite jointly assesses forgetting effectiveness, retained utility, and robustness to model interventions, adversarial inputs, and membership inference attacks. Using a common evaluation protocol, we compare ten representative unlearning methods. In this comparison, GD and MIP-Editor tie for the highest overall score: GD achieves the highest Forget Quality, while MIP-Editor preserves more Model Utility. We further introduce a metric meta-evaluation protocol that tests faithfulness using models with controlled exposure to target knowledge and robustness under quantization and relearning. Among the thirteen evaluated metrics, BLEU achieves the highest aggregate reliability score. KS-Test attains the highest faithfulness AUC but performs less well on robustness. Together, the framework and these findings support reproducible comparison of MLLM unlearning methods and systematic assessment of evaluation reliability.",
      "published": "2026-10-07T16:30:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.10358v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10358v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SLDR: Defending Against Malicious Fine-tuning via Selective Layers Recovery and Dynamic Routing",
      "authors": [
        "Hui Zhang",
        "Yachao Yuan",
        "Jiayun Wang",
        "Yuanzhuo Li",
        "Hongtao Wang",
        "Yali Yuan"
      ],
      "abstract": "Fine-tuning-as-a-service enables users to adapt aligned large language models (LLMs) to specialized tasks, but malicious fine-tuning can erode refusal behavior while preserving task performance on legitimate inputs. We revisit recent layer-wise safety diagnostics and find that safety sensitivity is signed: scaling different layers can strengthen refusal, weaken it, or have little effect. Motivated by this observation, we propose SLDR, a post-fine-tuning defense based on Selective Layers Recovery and Dynamic Routing. SLDR trains a LoRA recovery adapter only on the layers with the maximum and minimum sensitivity scores in the signed spectrum, and uses representation-based dynamic routing inference to activate the adapter only for malicious queries. Across four model architectures, five downstream tasks, and four harmful benchmarks, SLDR substantially reduces harmful outputs while preserving downstream utility. On Llama3.1/SST2, SLDR reduces the average harmful score from 11.54 to 0.08 while maintaining downstream accuracy, and the harmful score remains near zero under poisoning ratios up to 0.9. The code is available at https://github.com/Stardust457/SLDR.",
      "published": "2026-10-07T16:23:45Z",
      "abstract_url": "http://arxiv.org/abs/2610.10345v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10345v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "LLM-Assisted Generation of Transparent, Open-Source Multiphysics Models of Electrochemical Devices",
      "authors": [
        "Sebastian Castro",
        "Maya F. Schuchert",
        "Spencer A. McCluskey",
        "Eric W. Lees",
        "Justin C. Bui"
      ],
      "abstract": "Multiphysics continuum models are powerful tools for studying electrochemical devices, enabling in silico reactor design and resolution of local pH, potential, and concentration fields that govern device performance but are difficult to measure experimentally. However, constructing such models requires substantial numerical expertise or reliance on proprietary software. Here, we show that frontier large language model agents can remove this implementation burden while keeping the underlying physics under researcher control. Using one-dimensional electrochemical CO2 reduction to CO in a porous gas diffusion electrode as a test case, we develop a machine-readable, human-specified modeling harness containing governing equations, parameters, numerical methods, logical build stages, and human-verifiable checkpoints. From this specification, the agent reproducibly constructs complete multiphysics models in open-source Julia. Independently built models, including fully autonomous agent-built models, agree with an equivalent COMSOL implementation to within 0.7% of the peak CO partial current density, and with one another to within 0.04%. Systematically planted errors demonstrate the importance of explicit specifications for reproducibility and reveal the agent's capabilities and limitations in debugging model physics. This framework establishes a more transparent approach to multiphysics modeling in which physical descriptions and governing equations, rather than specialized code, become the primary inputs for computational model development.",
      "published": "2026-10-07T16:11:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.10320v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10320v1",
      "categories": [
        "physics.chem-ph",
        "cs.AI",
        "physics.comp-ph"
      ]
    },
    {
      "title": "Fault-tolerant foundation models",
      "authors": [
        "Trevor McCourt",
        "Ila R. Fiete",
        "Isaac L. Chuang"
      ],
      "abstract": "Emerging computer hardware often trades reliability for energy efficiency; here we show that large-language models (LLMs) can be trained to tolerate this unreliability, and that rather than degrading, their error resilience actually increases as they grow. Modified neural scaling laws inferred from 40,000 GPU-hours of training runs on simulated faulty digital hardware quantify this trend and suggest that models learn to compute within \"good\" error-correcting codes, whose relative overhead remains finite no matter how large the model gets. This finding leads us to conjecture that appropriately trained LLMs may be formally fault-tolerant; if true, running AI inference on low energy, faulty hardware may be a path to substantial energy savings over the status quo.",
      "published": "2026-10-07T16:05:48Z",
      "abstract_url": "http://arxiv.org/abs/2610.10311v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10311v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.AR"
      ]
    },
    {
      "title": "SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning",
      "authors": [
        "Mingyan Liu",
        "Min Huang"
      ],
      "abstract": "We study whether latent sequence compression of prompt prefixes preserves the capabilities that large language models rely on during inference. We introduce SemanticFold, a compression scheme that folds prefix hidden states at learned boundaries, and evaluate it across five model scales: Qwen3-1.7B, Qwen3-8B, SmolLM2-1.7B, Pythia-1.4B, and Pythia-6.9B. We use a fixed-target protocol: a frozen prefix is executed natively or compressed, and both arms teacher-force identical continuation tokens. This design rules out target-selection explanations for likelihood changes. We examine five endpoint families: fixed-target negative log-likelihood, finite-label reasoning accuracy, linear probe accessibility, open-ended generation, and systems-level memory and latency. We find that compression moves these endpoints non-monotonically and that they do not share a single compression threshold. On Qwen3-1.7B at compression ratio R=1.7, compressed-minus-native mean NLL decreases by 0.135 under paired bootstrap with 10000 draws. On SmolLM2 at R=1.2, the mean change is 0.013 higher than native. On both Pythia checkpoints, NLL is effectively unchanged. An NLL decomposition separating sequence shortening from the learned residual transform shows that the favorable Qwen likelihood is attributable primarily to residual adaptation rather than to shortening alone. MLP-only, which applies the transform without shortening, achieves 0.082 lower NLL than Full SemanticFold. Linear probe accuracy and macro AUC change by less than 0.03 in absolute value across conditions, with confidence intervals crossing zero. We conclude that preservation under latent compression has no single scalar certificate: language-model fit, decodability, and reasoning behavior answer different questions and can move in different directions under the same compression operation.",
      "published": "2026-10-07T16:00:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.10304v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10304v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "AI Safety Considerations for Agents With Limited Time to Act",
      "authors": [
        "Leo Zeitler",
        "Jack Richings",
        "Victoria Nockles"
      ],
      "abstract": "In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent.",
      "published": "2026-10-07T15:49:06Z",
      "abstract_url": "http://arxiv.org/abs/2610.10285v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10285v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Logarithmic Regret via Passive Change Detection in Piecewise-Stationary Self-Tuning Regulation",
      "authors": [
        "A. Ch. Madhusudanarao",
        "Rahul Singh"
      ],
      "abstract": "We study minimum-variance control of an unknown autoregressive system with exogenous inputs and coefficients that change at unknown times. Under bounded independent disturbances, fixed detection gaps, stability and feasibility conditions, and sufficient time between changes, we prove \\(O((C+1)\\log((T+1)/δ))\\) regret with probability at least \\(1-δ\\), where \\(T\\) is the horizon and \\(C\\) the number of changes. Unlike switching bandits, where unselected arms can change unobserved, admissible plant changes provide information during exploitation: the correct feasible controller leaves only the disturbance in the output, whereas a detectable change raises output energy under the old controller. PIECE-CD explores initially and after alarms, then uses gated recursive least squares for control. Its energy test compares windowed output power with a threshold above the noise floor; the extension to unstable controller mismatches also monitors the reference controller's input proposal. We control false alarms across the horizon and prove logarithmic detection delay. Inputs are clipped to prescribed bounds. Logarithmic regret also holds under an explicit condition ensuring that clipping becomes inactive after a finite burn-in. Under the stated feasibility conditions, the extended detector covers destabilizing changes with detectable excess energy over a fixed window.",
      "published": "2026-10-07T15:31:27Z",
      "abstract_url": "http://arxiv.org/abs/2610.10250v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10250v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Stationary Bias and Extrapolation in Nonlinear Two-Timescale Stochastic Approximation",
      "authors": [
        "A. Ch. Madhusudanarao",
        "Rahul Singh"
      ],
      "abstract": "Constant-step stochastic approximation generally has a nonzero stationary mean error that persists under time averaging. This paper studies that error for nonlinear two-timescale recursions driven by an exogenous finite-state Markov chain. Under stated smoothness assumptions and conditions on the stationary distribution, we derive a first-order bias expansion whose error bound remains uniform as the slow step size becomes much smaller than the fast step size. Fast-manifold coordinates keep the associated covariance equation regular in this limit. For fast step $η$ and slow step $\\varepsilon$, the expansion reveals a mixed contribution $\\varepsilon^2/η$ alongside terms linear in each step size. This dependence matters for bias reduction: along power-law step-size paths, the bias exponents need not be integers, so Richardson--Romberg extrapolation requires weights matched to the path. An exactly solvable nonlinear Markov example verifies the coefficients. We verify localization for temporal-difference learning and compare finite-run extrapolation at equal update budgets. For finite runs, we bound the initialization error of tail averages on both timescales under an additional coupling assumption. In the special case of additive independent noise, signed third-moment cancellation yields a sharper remainder.",
      "published": "2026-10-07T15:28:13Z",
      "abstract_url": "http://arxiv.org/abs/2610.10246v1",
      "pdf_url": "https://arxiv.org/pdf/2610.10246v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
