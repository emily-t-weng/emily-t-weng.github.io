const PAPERS_DATA = {
  "last_updated": "2026-04-06 03:29:19 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Enhancing Robustness of Federated Learning via Server Learning",
      "authors": [
        "Van Sy Mai",
        "Kushal Chakrabarti",
        "Richard J. La",
        "Dipankar Maity"
      ],
      "abstract": "This paper explores the use of server learning for enhancing the robustness of federated learning against malicious attacks even when clients' training data are not independent and identically distributed. We propose a heuristic algorithm that uses server learning and client update filtering in combination with geometric median aggregation. We demonstrate via experiments that this approach can achieve significant improvement in model accuracy even when the fraction of malicious clients is high, even more than $50\\%$ in some cases, and the dataset utilized by the server is small and could be synthetic with its distribution not necessarily close to that of the clients' aggregated data.",
      "published": "2026-04-03T17:51:29Z",
      "abstract_url": "http://arxiv.org/abs/2604.03226v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03226v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "PR3DICTR: A modular AI framework for medical 3D image-based detection and outcome prediction",
      "authors": [
        "Daniel C. MacRae",
        "Luuk van der Hoek",
        "Robert van der Wal",
        "Suzanne P. M. de Vette",
        "Hendrike Neh",
        "Baoqiang Ma",
        "Peter M. A. van Ooijen",
        "Lisanne V. van Dijk"
      ],
      "abstract": "Three-dimensional medical image data and computer-aided decision making, particularly using deep learning, are becoming increasingly important in the medical field. To aid in these developments we introduce PR3DICTR: Platform for Research in 3D Image Classification and sTandardised tRaining. Built using community-standard distributions (PyTorch and MONAI), PR3DICTR provides an open-access, flexible and convenient framework for prediction model development, with an explicit focus on classification using three-dimensional medical image data. By combining modular design principles and standardization, it aims to alleviate developmental burden whilst retaining adjustability. It provides users with a wealth of pre-established functionality, for instance in model architecture design options, hyper-parameter solutions and training methodologies, but still gives users the opportunity and freedom to ``plug in'' their own solutions or modules. PR3DICTR can be applied to any binary or event-based three-dimensional classification task and can work with as little as two lines of code.",
      "published": "2026-04-03T17:25:17Z",
      "abstract_url": "http://arxiv.org/abs/2604.03203v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03203v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Gradient Boosting within a Single Attention Layer",
      "authors": [
        "Saleh Sargolzaei"
      ],
      "abstract": "Transformer attention computes a single softmax-weighted average over values -- a one-pass estimate that cannot correct its own errors. We introduce \\emph{gradient-boosted attention}, which applies the principle of gradient boosting \\emph{within} a single attention layer: a second attention pass, with its own learned projections, attends to the prediction error of the first and applies a gated correction. Under a squared reconstruction objective, the construction maps onto Friedman's gradient boosting machine, with each attention pass as a base learner and the per-dimension gate as the shrinkage parameter. We show that a single Hopfield-style update erases all query information orthogonal to the stored-pattern subspace, and that further iteration under local contraction can collapse distinct queries in the same region to the same fixed point. We also show that separate projections for the correction pass can recover residual information inaccessible to the shared-projection approach of Tukey's twicing. On a 10M-token subset of WikiText-103, gradient-boosted attention achieves a test perplexity of $67.9$ compared to $72.2$ for standard attention, $69.6$ for Twicing Attention, and $69.0$ for a parameter-matched wider baseline, with two rounds capturing most of the benefit.",
      "published": "2026-04-03T17:06:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.03190v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03190v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Reflective Context Learning: Studying the Optimization Primitives of Context Space",
      "authors": [
        "Nikita Vassilyev",
        "William Berrios",
        "Ruowang Zhang",
        "Bo Han",
        "Douwe Kiela",
        "Shikib Mehri"
      ],
      "abstract": "Generally capable agents must learn from experience in ways that generalize across tasks and environments. The fundamental problems of learning, including credit assignment, overfitting, forgetting, local optima, and high-variance learning signals, persist whether the learned object lies in parameter space or context space. While these challenges are well understood in classical machine learning optimization, they remain underexplored in context space, leading current methods to be fragmented and ad hoc. We present Reflective Context Learning (RCL), a unified framework for agents that learn through repeated interaction, reflection on behavior and failure modes, and iterative updates to context. In RCL, reflection converts trajectories and current context into a directional update signal analogous to gradients, while mutation applies that signal to improve future behavior in context space. We recast recent context-optimization approaches as instances of this shared learning problem and systematically extend them with classical optimization primitives, including batching, improved credit-assignment signal, auxiliary losses, failure replay, and grouped rollouts for variance reduction. On AppWorld, BrowseComp+, and RewardBench2, these primitives improve over strong baselines, with their relative importance shifting across task regimes. We further analyze robustness to initialization, the effects of batch size, sampling and curriculum strategy, optimizer-state variants, and the impact of allocating stronger or weaker models to different optimization components. Our results suggest that learning through context updates should be treated not as a set of isolated algorithms, but as an optimization problem whose mechanisms can be studied systematically and improved through transferable principles.",
      "published": "2026-04-03T17:05:45Z",
      "abstract_url": "http://arxiv.org/abs/2604.03189v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03189v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models",
      "authors": [
        "Gengwei Zhang",
        "Jie Peng",
        "Zhen Tan",
        "Mufan Qiu",
        "Hossein Nourkhiz Mahjoub",
        "Vaishnav Tadiparthi",
        "Kwonjoon Lee",
        "Yanyong Zhang",
        "Tianlong Chen"
      ],
      "abstract": "The recent success of reinforcement learning (RL) in large reasoning models has inspired the growing adoption of RL for post-training Multimodal Large Language Models (MLLMs) to enhance their visual reasoning capabilities. Although many studies have reported improved performance, it remains unclear whether RL training truly enables models to learn from visual information. In this work, we propose the Hallucination-as-Cue Framework, an analytical framework designed to investigate the effects of RL-based post-training on multimodal reasoning models from the perspective of model hallucination. Specifically, we introduce hallucination-inductive, modality-specific corruptions that remove or replace essential information required to derive correct answers, thereby forcing the model to reason by hallucination. By applying these corruptions during both training and evaluation, our framework provides a unique perspective for diagnosing RL training dynamics and understanding the intrinsic properties of datasets. Through extensive experiments and analyses across multiple multimodal reasoning benchmarks, we reveal that the role of model hallucination for RL-training is more significant than previously recognized. For instance, we find that RL post-training under purely hallucination-inductive settings can still significantly improve models' reasoning performance, and in some cases even outperform standard training. These findings challenge prevailing assumptions about MLLM reasoning training and motivate the development of more modality-aware RL-based training designs.",
      "published": "2026-04-03T16:56:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.03179v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03179v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Beyond the Parameters: A Technical Survey of Contextual Enrichment in Large Language Models: From In-Context Prompting to Causal Retrieval-Augmented Generation",
      "authors": [
        "Prakhar Bansal",
        "Shivangi Agarwal"
      ],
      "abstract": "Large language models (LLMs) encode vast world knowledge in their parameters, yet they remain fundamentally limited by static knowledge, finite context windows, and weakly structured causal reasoning. This survey provides a unified account of augmentation strategies along a single axis: the degree of structured context supplied at inference time. We cover in-context learning and prompt engineering, Retrieval-Augmented Generation (RAG), GraphRAG, and CausalRAG. Beyond conceptual comparison, we provide a transparent literature-screening protocol, a claim-audit framework, and a structured cross-paper evidence synthesis that distinguishes higher-confidence findings from emerging results. The paper concludes with a deployment-oriented decision framework and concrete research priorities for trustworthy retrieval-augmented NLP.",
      "published": "2026-04-03T16:49:09Z",
      "abstract_url": "http://arxiv.org/abs/2604.03174v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03174v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control",
      "authors": [
        "Lihao Sun",
        "Lewen Yan",
        "Xiaoya Lu",
        "Andrew Lee",
        "Jie Zhang",
        "Jing Shao"
      ],
      "abstract": "We present a method to identify a valence-arousal (VA) subspace within large language model representations. From 211k emotion-labeled texts, we derive emotion steering vectors, then learn VA axes as linear combinations of their top PCA components via ridge regression on the model's self-reported valence-arousal scores. The resulting VA subspace exhibits circular geometry consistent with established models of human emotion perception. Projections along our recovered VA subspace correlate with human-crowdsourced VA ratings across 44k lexical items. Furthermore, steering generation along these axes produces monotonic shifts in the corresponding affective dimensions of model outputs. Steering along these directions also induces near-monotonic bidirectional control over refusal and sycophancy: increasing arousal decreases refusal and increases sycophancy, and vice versa. These effects replicate across Llama-3.1-8B, Qwen3-8B, and Qwen3-14B, demonstrating cross-architecture generality. We provide a mechanistic account for these effects and prior emotionally-framed controls: refusal-associated tokens (\"I can't,\" \"sorry\") occupy low-arousal, negative-valence regions, so VA steering directly modulates their emission probability.",
      "published": "2026-04-03T16:08:05Z",
      "abstract_url": "http://arxiv.org/abs/2604.03147v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03147v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "A Systematic Security Evaluation of OpenClaw and Its Variants",
      "authors": [
        "Yuhang Wang",
        "Haichang Gao",
        "Zhenxing Niu",
        "Zhaoxiang Liu",
        "Wenjing Zhang",
        "Xiang Wang",
        "Shiguo Lian"
      ],
      "abstract": "Tool-augmented AI agents substantially extend the practical capabilities of large language models, but they also introduce security risks that cannot be identified through model-only evaluation. In this paper, we present a systematic security assessment of six representative OpenClaw-series agent frameworks, namely OpenClaw, AutoClaw, QClaw, KimiClaw, MaxClaw, and ArkClaw, under multiple backbone models. To support this study, we construct a benchmark of 205 test cases covering representative attack behaviors across the full agent execution lifecycle, enabling unified evaluation of risk exposure at both the framework and model levels. Our results show that all evaluated agents exhibit substantial security vulnerabilities, and that agentized systems are significantly riskier than their underlying models used in isolation. In particular, reconnaissance and discovery behaviors emerge as the most common weaknesses, while different frameworks expose distinct high-risk profiles, including credential leakage, lateral movement, privilege escalation, and resource development. These findings indicate that the security of modern agent systems is shaped not only by the safety properties of the backbone model, but also by the coupling among model capability, tool use, multi-step planning, and runtime orchestration. We further show that once an agent is granted execution capability and persistent runtime context, weaknesses arising in early stages can be amplified into concrete system-level failures. Overall, our study highlights the need to move beyond prompt-level safeguards toward lifecycle-wide security governance for intelligent agent frameworks.",
      "published": "2026-04-03T15:52:36Z",
      "abstract_url": "http://arxiv.org/abs/2604.03131v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03131v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Co-Evolution of Policy and Internal Reward for Language Agents",
      "authors": [
        "Xinyu Wang",
        "Hanwei Wu",
        "Jingwei Song",
        "Shuyuan Zhang",
        "Jiayi Zhang",
        "Fanqi Kong",
        "Tung Sum Thomas Kwok",
        "Xiao-Wen Chang",
        "Yuyu Luo",
        "Chenglin Wu",
        "Bang Liu"
      ],
      "abstract": "Large language model (LLM) agents learn by interacting with environments, but long-horizon training remains fundamentally bottlenecked by sparse and delayed rewards. Existing methods typically address this challenge through post-hoc credit assignment or external reward models, which provide limited guidance at inference time and often separate reward improvement from policy improvement. We propose Self-Guide, a self-generated internal reward for language agents that supports both inference-time guidance and training-time supervision. Specifically, the agent uses Self-Guide as a short self-guidance signal to steer the next action during inference, and converts the same signal into step-level internal reward for denser policy optimization during training. This creates a co-evolving loop: better policy produces better guidance, and better guidance further improves policy as internal reward. Across three agent benchmarks, inference-time self-guidance already yields clear gains, while jointly evolving policy and internal reward with GRPO brings further improvements (8\\%) over baselines trained solely with environment reward. Overall, our results suggest that language agents can improve not only by collecting more experience, but also by learning to generate and refine their own internal reward during acting and learning.",
      "published": "2026-04-03T15:21:11Z",
      "abstract_url": "http://arxiv.org/abs/2604.03098v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03098v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?",
      "authors": [
        "Qianshan Wei",
        "Yishan Yang",
        "Siyi Wang",
        "Jinglin Chen",
        "Binyu Wang",
        "Jiaming Wang",
        "Shuang Chen",
        "Zechen Li",
        "Yang Shi",
        "Yuqi Tang",
        "Weining Wang",
        "Yi Yu",
        "Chaoyou Fu",
        "Qi Li",
        "Yi-Fan Zhang"
      ],
      "abstract": "Multimodal Large Language Models (MLLMs) are evolving from passive observers into active agents, solving problems through Visual Expansion (invoking visual tools) and Knowledge Expansion (open-web search). However, existing evaluations fall short: they lack flexible tool integration, test visual and search tools separately, and evaluate primarily by final answers. Consequently, they cannot verify if tools were actually invoked, applied correctly, or used efficiently. To address this, we introduce Agentic-MME, a process-verified benchmark for Multimodal Agentic Capabilities. It contains 418 real-world tasks across 6 domains and 3 difficulty levels to evaluate capability synergy, featuring over 2,000 stepwise checkpoints that average 10+ person-hours of manual annotation per task. Each task includes a unified evaluation framework supporting sandboxed code and APIs, alongside a human reference trajectory annotated with stepwise checkpoints along dual-axis: S-axis and V-axis. To enable true process-level verification, we audit fine-grained intermediate states rather than just final answers, and quantify efficiency via an overthinking metric relative to human trajectories. Experimental results show the best model, Gemini3-pro, achieves 56.3% overall accuracy, which falls significantly to 23.0% on Level-3 tasks, underscoring the difficulty of real-world multimodal agentic problem solving.",
      "published": "2026-04-03T13:02:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.03016v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03016v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "R2-Write: Reflection and Revision for Open-Ended Writing with Deep Reasoning",
      "authors": [
        "Wanlong Liu",
        "Bo Zhang",
        "Chenliang Li",
        "Shaopeng Lai",
        "Yuning Wu",
        "Xuanyu Lei",
        "Ming Yan"
      ],
      "abstract": "While deep reasoning with long chain-of-thought has dramatically improved large language models in verifiable domains like mathematics, its effectiveness for open-ended tasks such as writing remains unexplored. In this paper, we conduct a systematic investigation revealing that existing mainstream reasoning models achieve limited gains on open-ended writing tasks. Our further analysis shows that these models lack deep reflection and revision patterns in open-ended writing, resulting in substantially smaller improvements compared to mathematical reasoning tasks. To address this limitation, we introduce R2-Write: an automated framework that synthesizes high-quality thinking trajectories enriched with explicit reflection and revision patterns through iterative writer-judge interaction. To prevent redundant reflections, we design a process reward mechanism that supervises reflection quality during reinforcement learning, improving both performance and token efficiency. Extensive experiments across multiple creative writing and deep-research benchmarks demonstrate significant improvements, validating that explicitly incorporating reflection and revision patterns unlocks deep reasoning capabilities for open-ended writing tasks.",
      "published": "2026-04-03T12:43:26Z",
      "abstract_url": "http://arxiv.org/abs/2604.03004v1",
      "pdf_url": "https://arxiv.org/pdf/2604.03004v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "FedSQ: Optimized Weight Averaging via Fixed Gating",
      "authors": [
        "Cristian Pérez-Corral",
        "Jose I. Mestre",
        "Alberto Fernández-Hernández",
        "Manuel F. Dolz",
        "José Duato",
        "Enrique S. Quintana-Ortí"
      ],
      "abstract": "Federated learning (FL) enables collaborative training across organizations without sharing raw data, but it is hindered by statistical heterogeneity (non-i.i.d.\\ client data) and by instability of naive weight averaging under client drift. In many cross-silo deployments, FL is warm-started from a strong pretrained backbone (e.g., ImageNet-1K) and then adapted to local domains. Motivated by recent evidence that ReLU-like gating regimes (structural knowledge) stabilize earlier than the remaining parameter values (quantitative knowledge), we propose FedSQ (Federated Structural-Quantitative learning), a transfer-initialized neural federated procedure based on a DualCopy, piecewise-linear view of deep networks. FedSQ freezes a structural copy of the pretrained model to induce fixed binary gating masks during federated fine-tuning, while only a quantitative copy is optimized locally and aggregated across rounds. Fixing the gating reduces learning to within-regime affine refinements, which stabilizes aggregation under heterogeneous partitions. Experiments on two convolutional neural network backbones under i.i.d.\\ and Dirichlet splits show that FedSQ improves robustness and can reduce rounds-to-best validation performance relative to standard baselines while preserving accuracy in the transfer setting.",
      "published": "2026-04-03T11:54:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.02990v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02990v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.DC"
      ]
    },
    {
      "title": "Mitigating Reward Hacking in RLHF via Advantage Sign Robustness",
      "authors": [
        "Shinnosuke Ono",
        "Johannes Ackermann",
        "Soichiro Nishimori",
        "Takashi Ishida",
        "Masashi Sugiyama"
      ],
      "abstract": "Reward models (RMs) used in reinforcement learning from human feedback (RLHF) are vulnerable to reward hacking: as the policy maximizes a learned proxy reward, true quality plateaus or degrades. We make the assumption that reward hacking is often caused by flipped advantage signs: instead of reducing the likelihood of a bad response, a flipped sign causes the update to increase it. By considering an adversarial perturbation in the RM parameter space, we can derive a certified sign-preservation radius, which is the smallest perturbation that can flip the advantage sign during policy optimization. Based on this formulation, we propose Sign-Certified Policy Optimization (SignCert-PO), down-weighting non-robust completions in the policy gradient update. Unlike prior approaches that require multiple RMs or access to the RM training data, SignCert-PO is lightweight and operates purely at the policy optimization stage using only the RM parameters and on-policy completions. On TL;DR summarization and AlpacaFarm benchmarks, SignCert-PO consistently achieves a better win rate than baselines and reduces reward hacking.",
      "published": "2026-04-03T11:45:16Z",
      "abstract_url": "http://arxiv.org/abs/2604.02986v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02986v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Prompt Compression in the Wild: Measuring Latency, Rate Adherence, and Quality for Faster LLM Inference",
      "authors": [
        "Cornelius Kummer",
        "Lena Jurkschat",
        "Michael Färber",
        "Sahar Vahdati"
      ],
      "abstract": "With the wide adoption of language models for IR -- and specifically RAG systems -- the latency of the underlying LLM becomes a crucial bottleneck, since the long contexts of retrieved passages lead large prompts and therefore, compute increase. Prompt compression, which reduces the size of input prompts while aiming to preserve performance on downstream tasks, has established itself as a cost-effective and low-latency method for accelerating inference in large language models. However, its usefulness depends on whether the additional preprocessing time during generation is offset by faster decoding. We present the first systematic, large-scale study of this trade-off, with thousands of runs and 30,000 queries across several open-source LLMs and three GPU classes. Our evaluation separates compression overhead from decoding latency while tracking output quality and memory usage. LLMLingua achieves up to 18% end-to-end speed-ups, when prompt length, compression ratio, and hardware capacity are well matched, with response quality remaining statistically unchanged across summarization, code generation, and question answering tasks. Outside this operating window, however, the compression step dominates and cancels out the gains. We also show that effective compression can reduce memory usage enough to offload workloads from data center GPUs to commodity cards, with only a 0.3s increase in latency. Our open-source profiler predicts the latency break-even point for each model-hardware setup, providing practical guidance on when prompt compression delivers real-world benefits.",
      "published": "2026-04-03T11:41:53Z",
      "abstract_url": "http://arxiv.org/abs/2604.02985v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02985v1",
      "categories": [
        "cs.IR",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking",
      "authors": [
        "Ka Yiu Lee",
        "Yuxuan Huang",
        "Zhiyuan He",
        "Huichi Zhou",
        "Weilin Luo",
        "Kun Shao",
        "Meng Fang",
        "Jun Wang"
      ],
      "abstract": "Recent agentic search systems have made substantial progress by emphasising deep, multi-step reasoning. However, this focus often overlooks the challenges of wide-scale information synthesis, where agents must aggregate large volumes of heterogeneous evidence across many sources. As a result, most existing large language model agent systems face severe limitations in data-intensive settings, including context saturation, cascading error propagation, and high end-to-end latency. To address these challenges, we present \\framework, a hierarchical framework based on principle of near-decomposability, containing a strategic \\textit{Host}, multiple \\textit{Managers} and parallel \\textit{Workers}. By leveraging aggregation and reflection mechanisms at the Manager layer, our framework enforces strict context isolation to prevent saturation and error propagation. Simultaneously, the parallelism in worker layer accelerates the speed of overall task execution, mitigating the significant latency. Our evaluation on two complementary benchmarks demonstrates both efficiency ($ 3-5 \\times$ speed-up) and effectiveness, achieving a $8.4\\%$ success rate on WideSearch-en and $52.9\\%$ accuracy on BrowseComp-zh. The code is released at https://github.com/agent-on-the-fly/InfoSeeker",
      "published": "2026-04-03T11:19:17Z",
      "abstract_url": "http://arxiv.org/abs/2604.02971v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02971v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation",
      "authors": [
        "Yilin Xiao",
        "Jin Chen",
        "Qinggang Zhang",
        "Yujing Zhang",
        "Chuang Zhou",
        "Longhao Yang",
        "Lingfei Ren",
        "Xin Yang",
        "Xiao Huang"
      ],
      "abstract": "Graph-based Retrieval-Augmented Generation (GraphRAG) enhances the reasoning capabilities of Large Language Models (LLMs) by grounding their responses in structured knowledge graphs. Leveraging community detection and relation filtering techniques, GraphRAG systems demonstrate inherent resistance to traditional RAG attacks, such as text poisoning and prompt injection. However, in this paper, we find that the security of GraphRAG systems fundamentally relies on the topological integrity of the underlying graph, which can be undermined by implicitly corrupting the logical connections, without altering surface-level text semantics. To exploit this vulnerability, we propose \\textsc{LogicPoison}, a novel attack framework that targets logical reasoning rather than injecting false contents. Specifically, \\textsc{LogicPoison} employs a type-preserving entity swapping mechanism to perturb both global logic hubs for disrupting overall graph connectivity and query-specific reasoning bridges for severing essential multi-hop inference paths. This approach effectively reroutes valid reasoning into dead ends while maintaining surface-level textual plausibility. Comprehensive experiments across multiple benchmarks demonstrate that \\textsc{LogicPoison} successfully bypasses GraphRAG's defenses, significantly degrading performance and outperforming state-of-the-art baselines in both effectiveness and stealth. Our code is available at \\textcolor{blue}https://github.com/Jord8061/logicPoison.",
      "published": "2026-04-03T10:42:07Z",
      "abstract_url": "http://arxiv.org/abs/2604.02954v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02954v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "How Annotation Trains Annotators: Competence Development in Social Influence Recognition",
      "authors": [
        "Maciej Markiewicz",
        "Beata Bajcar",
        "Wiktoria Mieleszczenko-Kowszewicz",
        "Aleksander Szczęsny",
        "Tomasz Adamczyk",
        "Grzegorz Chodak",
        "Karolina Ostrowska",
        "Aleksandra Sawczuk",
        "Jolanta Babiak",
        "Jagoda Szklarczyk",
        "Przemysław Kazienko"
      ],
      "abstract": "Human data annotation, especially when involving experts, is often treated as an objective reference. However, many annotation tasks are inherently subjective, and annotators' judgments may evolve over time. This study investigates changes in the quality of annotators' work from a competence perspective during a process of social influence recognition. The study involved 25 annotators from five different groups, including both experts and non-experts, who annotated a dataset of 1,021 dialogues with 20 social influence techniques, along with intentions, reactions, and consequences. An initial subset of 150 texts was annotated twice - before and after the main annotation process - to enable comparison. To measure competence shifts, we combined qualitative and quantitative analyses of the annotated data, semi-structured interviews with annotators, self-assessment surveys, and Large Language Model training and evaluation on the comparison dataset. The results indicate a significant increase in annotators' self-perceived competence and confidence. Moreover, observed changes in data quality suggest that the annotation process may enhance annotator competence and that this effect is more pronounced in expert groups. The observed shifts in annotator competence have a visible impact on the performance of LLMs trained on their annotated data.",
      "published": "2026-04-03T10:32:57Z",
      "abstract_url": "http://arxiv.org/abs/2604.02951v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02951v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Learning from Synthetic Data via Provenance-Based Input Gradient Guidance",
      "authors": [
        "Koshiro Nagano",
        "Ryo Fujii",
        "Ryo Hachiuma",
        "Fumiaki Sato",
        "Taiki Sekii",
        "Hideo Saito"
      ],
      "abstract": "Learning methods using synthetic data have attracted attention as an effective approach for increasing the diversity of training data while reducing collection costs, thereby improving the robustness of model discrimination. However, many existing methods improve robustness only indirectly through the diversification of training samples and do not explicitly teach the model which regions in the input space truly contribute to discrimination; consequently, the model may learn spurious correlations caused by synthesis biases and artifacts. Motivated by this limitation, this paper proposes a learning framework that uses provenance information obtained during the training data synthesis process, indicating whether each region in the input space originates from the target object, as an auxiliary supervisory signal to promote the acquisition of representations focused on target regions. Specifically, input gradients are decomposed based on information about target and non-target regions during synthesis, and input gradient guidance is introduced to suppress gradients over non-target regions. This suppresses the model's reliance on non-target regions and directly promotes the learning of discriminative representations for target regions. Experiments demonstrate the effectiveness and generality of the proposed method across multiple tasks and modalities, including weakly supervised object localization, spatio-temporal action localization, and image classification.",
      "published": "2026-04-03T10:28:58Z",
      "abstract_url": "http://arxiv.org/abs/2604.02946v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02946v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Council Mode: Mitigating Hallucination and Bias in LLMs via Multi-Agent Consensus",
      "authors": [
        "Shuai Wu",
        "Xue Li",
        "Yanna Feng",
        "Yufang Li",
        "Zhijun Wang"
      ],
      "abstract": "Large Language Models (LLMs), particularly those employing Mixture-of-Experts (MoE) architectures, have achieved remarkable capabilities across diverse natural language processing tasks. However, these models frequently suffer from hallucinations -- generating plausible but factually incorrect content -- and exhibit systematic biases that are amplified by uneven expert activation during inference. In this paper, we propose the Council Mode, a novel multi-agent consensus framework that addresses these limitations by dispatching queries to multiple heterogeneous frontier LLMs in parallel and synthesizing their outputs through a dedicated consensus model. The Council pipeline operates in three phases: (1) an intelligent triage classifier that routes queries based on complexity, (2) parallel expert generation across architecturally diverse models, and (3) a structured consensus synthesis that explicitly identifies agreement, disagreement, and unique findings before producing the final response. We implement and evaluate this architecture within an open-source AI workspace. Our comprehensive evaluation across multiple benchmarks demonstrates that the Council Mode achieves a 35.9% relative reduction in hallucination rates on the HaluEval benchmark and a 7.8-point improvement on TruthfulQA compared to the best-performing individual model, while maintaining significantly lower bias variance across domains. We provide the mathematical formulation of the consensus mechanism, detail the system architecture, and present extensive empirical results with ablation studies.",
      "published": "2026-04-03T09:40:43Z",
      "abstract_url": "http://arxiv.org/abs/2604.02923v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02923v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Split and Conquer Partial Deepfake Speech",
      "authors": [
        "Inbal Rimon",
        "Oren Gal",
        "Haim Permuter"
      ],
      "abstract": "Partial deepfake speech detection requires identifying manipulated regions that may occur within short temporal portions of an otherwise bona fide utterance, making the task particularly challenging for conventional utterance-level classifiers. We propose a split-and-conquer framework that decomposes the problem into two stages: boundary detection and segment-level classification. A dedicated boundary detector first identifies temporal transition points, allowing the audio signal to be divided into segments that are expected to contain acoustically consistent content. Each resulting segment is then evaluated independently to determine whether it corresponds to bona fide or fake speech. This formulation simplifies the learning objective by explicitly separating temporal localization from authenticity assessment, allowing each component to focus on a well-defined task. To further improve robustness, we introduce a reflection-based multi-length training strategy that converts variable-duration segments into several fixed input lengths, producing diverse feature-space representations. Each stage is trained using multiple configurations with different feature extractors and augmentation strategies, and their complementary predictions are fused to obtain improved final models. Experiments on the PartialSpoof benchmark demonstrate state-of-the-art performance across multiple temporal resolutions as well as at the utterance level, with substantial improvements in the accurate detection and localization of spoofed regions. In addition, the proposed method achieves state-of-the-art performance on the Half-Truth dataset, further confirming the robustness and generalization capability of the framework.",
      "published": "2026-04-03T09:33:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.02913v1",
      "pdf_url": "https://arxiv.org/pdf/2604.02913v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
