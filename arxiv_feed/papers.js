const PAPERS_DATA = {
  "last_updated": "2026-08-12 02:30:04 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "ConVAWG: A Retrieval-Grounded Framework for Controlled Synthetic Dialogue Generation in Violence Against Women and Girls",
      "authors": [
        "Chen Lyu",
        "Xingwei Tan",
        "Simon Cullen",
        "Shelley Wilson",
        "Lois Arthurs",
        "Arshad Jhumka",
        "Gabriele Pergola"
      ],
      "abstract": "Synthetic dialogue generation offers a way to study conversational dynamics in sensitive domains where real data are difficult to access, release, or annotate. The underlying abuse may occur online or offline: threats and coercion can appear directly in messages, while behaviours such as surveillance, isolation, stalking, and physical violence may be planned, disclosed, or referred to conversationally. Privacy and legal constraints make it difficult the release of large-scale real conversation datasets; existing work has mostly focused on sentence-level toxicity of online abuses, leaving a gap in modelling abuse as a relational and temporally unfolding phenomenon. In this work, we focus on modelling Violence Against Women and Girls (VAWG) scenarios as multi-turn dialogues. We introduce ConVAWG, a retrieval-grounded framework for generating CPS-aligned synthetic VAWG chat dialogues. ConVAWG builds scenarios from persona seeds, demographic patterns reported by the UK Office for National Statistics, official crime definitions, and retrieved Domestic Homicide Review cases; converts them into hierarchical event timelines; generates multi-scene role-play dialogues; and applies targeted activation-steered toxicity control to appropriate utterances. We release over 6,000 multi-turn dialogue events across 200 scenarios with rich scenario-, event-, and turn-level metadata. Extensive human evaluation, LLM-as-Judge assessment, ablations, and downstream tasks show strong dialogue quality and domain fidelity.",
      "published": "2026-08-11T17:57:34Z",
      "abstract_url": "http://arxiv.org/abs/2608.11200v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11200v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "How to Verify Consistency of Probabilistic Claims",
      "authors": [
        "Orr Paradise",
        "Oliver Richardson",
        "Yoshua Bengio",
        "Shafi Goldwasser"
      ],
      "abstract": "When a probabilistic predictor answers many conditional-probability queries, are its answers self-consistent, and can this be verified in polynomial time? This problem is of interest for AI safety, where safety is derived from honesty about probabilistic predictions of unwanted outcomes potentially caused by an AI action. We construct an interactive PCP as follows. Let a predictive model be specified by a probability circuit P and a circuit Q which outputs confidence in predictions. Together, P and Q implicitly specify exponentially many probabilistic claims. We show a protocol in which a polynomial-time verifier can verify the approximate consistency of (P,Q). The verifier is given the pair of circuits (P,Q), which it evaluates at only a few points; alongside them it is given a proof oracle, an encoding of a witnessing probability distribution allegedly consistent with the predictions of (P,Q), which it reads at a few locations while interacting with a single untrusted prover. En route, we must ensure the existence of a sparse witnessing distribution consistent with the model's predictions. To do so, we first consider witness distributions for the consistency of explicit probabilistic claims, rather than claims specified by a predictor: say m claims, each of the form Pr[Y = 1 | X = x] = p, over n Boolean variables. Building on work initiated by Nilsson (Artif. Intell., 1986), we place l_2-approximate probabilistic consistency of explicit claims in NP, with certificates of length O(mn + log B) in the input bit-precision B; we further show how a small additive completeness-soundness gap removes the dependence on B. Together these results provide a complexity-theoretic foundation for certifying the self-consistency of probabilistic predictors. We view our interactive PCP as a first step toward training predictive models to prove their own consistency.",
      "published": "2026-08-11T17:41:39Z",
      "abstract_url": "http://arxiv.org/abs/2608.11181v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11181v1",
      "categories": [
        "cs.CC",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Attention-Path Fragility as an Uncertainty Signal in Large Language Models",
      "authors": [
        "Minsoo Kim",
        "Sungyoung Ji",
        "Kisung Moon",
        "Ilyong Yoon"
      ],
      "abstract": "We propose that a model's uncertainty about a token is reflected not only in the breadth of its output distribution but also in whether a confident prediction is \\emph{fragile} under perturbation of its attention pathways. We instantiate this as ASMI (Attention-Subnetwork Mutual Information), a training-free estimator that masks attention heads and measures the BALD mutual information among the resulting subnetworks, with a semantic-agreement kernel to discount surface-form disagreement. The signal is not a restatement of output confidence: on grounded QA an out-of-fold test shows it adds error-predictive information beyond single-pass confidence and entropy, concentrated in \\emph{confident-but-fragile} predictions, where acting on it roughly halves the retained error of a confidence filter. The distinctness is regime-graded, so ASMI predicts its own domain of applicability, strong where answers are routed through provided context and bounded by design where they are recalled from parametric knowledge. Sem-ASMI reads the signal from a single greedy response, without the stochastic generations the strongest baselines require, and ties or beats Semantic Entropy on ten of the twelve grounded benchmark-backbone settings. Across the same twelve settings, the best ASMI variant, typically the adaptive one reusing the ten samples already drawn for the baselines, ties or leads the strongest baseline in eight, significantly in three under a paired test. On parametric QA all variants revert to or below the zero-cost MSP baseline, exactly as predicted, and the estimates are near-deterministic across reruns. A head-level analysis shows that what tracks this boundary is not the presence of head-level fragility but whether that fragility couples to errors.",
      "published": "2026-08-11T16:59:02Z",
      "abstract_url": "http://arxiv.org/abs/2608.11138v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11138v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Two-stage Odd Residual Flows for Mean-Preserving Probabilistic Time Series Forecasting",
      "authors": [
        "Kiran Madhusudhanan",
        "Christian Klötergens",
        "Lars Schmidt-Thieme",
        "Vijaya Krishna Yalavarthi"
      ],
      "abstract": "Probabilistic forecasting plays an essential role in risk-sensitive decision-making, particularly in long-horizon settings. However, existing approaches often face a fundamental trade-off between distributional flexibility and accurate mean prediction. Traditional parametric methods, such as Mean Variance Estimation (MVE), can suffer from degraded point accuracy when trained under joint Negative Log-Likelihood (NLL) objectives, while modern-flexible generative models, including Normalizing Flows and Diffusion Models, typically rely on costly Monte Carlo sampling and may yield suboptimal mean estimates. To address this limitation, we propose Two-stage Odd Residual Flows (TORF), a framework that decouples mean forecasting from uncertainty estimation. In the first stage, a pre-trained deterministic model is used to produce an accurate mean prediction. In the second stage, a Restricted Normalizing Flow, with strictly odd functions learns flexible residual distributions around the point forecast, guaranteeing mean preservation from the first stage without sampling. Experiments show that TORF achieves state-of-the-art deterministic accuracy (NMAE) while providing strong density estimation performance (CRPS) on short and long-horizon forecasting.",
      "published": "2026-08-11T16:22:47Z",
      "abstract_url": "http://arxiv.org/abs/2608.11114v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11114v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding",
      "authors": [
        "Kushal Chakrabarti"
      ],
      "abstract": "Agentic coding READMEs like CLAUDE.md grow without bound in real repositories, stopping only when the repository retires or someone rewrites the file wholesale. We trace this to imperfect recall: appending an instruction is always cheap, but once an instruction's rationale is gone, deleting it without risking a correctness regression costs O(2^|D|) in a prompt of |D| instructions. We name the resulting divergence catastrophic remembering, the inverse of catastrophic forgetting around which continual learning is organized. First, we characterize this phenomenon across 247,694 instruction lifetimes in 1,867 repositories: agentic prompts grow without bound, more than tripling over their lifetime (+226%), gaining +4.9 net instructions every commit; further, the older an instruction gets, the less likely it is to be deleted (log-hazard -0.032/commit). Then, we show that prompt comments can halt the growth: inverting IFEval yields verifiable worlds whose optimal prompts are known, and there comments encoding latent reasoning remove 99.3% of excess instructions (+211.3% to +1.4%). Finally, applying the same inversion to WildIFEval, we show that prompt comments can improve real-world agentic instruction-following by up to 23.1%. If English is the new code, why don't we have comments yet?",
      "published": "2026-08-11T16:00:55Z",
      "abstract_url": "http://arxiv.org/abs/2608.11095v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11095v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "Multiclass Sentiment Analysis for Identifying Political Viewpoints",
      "authors": [
        "Girma Yohannis Bade",
        "Olga Kolesnikova",
        "Jose Luis Oropeza",
        "Grigori Sidorov"
      ],
      "abstract": "The rapid growth of social media has created vast amounts of political discourse, which provides valuable opportunities to analyze public opinions and identify different political perspectives. Sentiment Analysis (SA) is a core task in Natural Language Processing (NLP) that allows the computational study of attitudes and opinions in textual data, and has become increasingly important for understanding political discourse. In this work, we investigate multiclass sentiment analysis of political view- points on social media, that is to automatically discriminate multiple sentiment classes over political issues and figures. To solve this task we design and evaluate two machine-learning approaches based on XGBoost and BERT. We train and evaluate the models on a labeled dataset of political social media posts using standard classification metrics. The experimental results show that the XGBoost model reaches an F1-score of 0.2835 and the BERT- based model reaches an F1-score of 0.2806 on the test set. These results demonstrate the challenge of classifying complex and contextualized political discourse sentiment and provide a baseline for future research in multiclass political sentiment analysis.",
      "published": "2026-08-11T15:21:23Z",
      "abstract_url": "http://arxiv.org/abs/2608.11049v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11049v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "V-FiLLM: Verified Financial LLM Reasoning Benchmark",
      "authors": [
        "Alicia Larsen",
        "Victoire Laurent",
        "Aulia Kharis Rakhamsari",
        "Lara Turgut",
        "Nino Antulov-Fantulin"
      ],
      "abstract": "While existing benchmarks have made substantial progress in evaluating LLMs across STEM domains, financial reasoning over structured data remains comparatively less explored. We introduce V-FiLLM, a framework that generates financial reasoning benchmarks from executable computation trees grounded in real tables, yielding items whose answers are correct by construction. Trees are evaluated symbolically to obtain ground truth and rendered into natural-language questions, removing any model from the labeling loop, so items can be generated at arbitrary scale without annotation cost and without inheriting a generator's error rate. V-FiLLM exposes four independently controllable axes of difficulty including computation depth, expression breadth, financial concept complexity, and context size. By evaluating on open-source models, we find that accuracy falls up to 51% as reasoning depth increases, and up to 47% points under adversarial numerical perturbations, highlighting remaining challenges in robust financial reasoning over tables. We further show that lightweight LoRA fine-tuning on verified chain-of-thought traces improves accuracy from 81.1% to 85.6% on held-out problems and outperforms the base model by 5% points on FinQA (Chen et al., 2022a), s), suggesting that targeted, low-cost adaptation is a promising direction for compositional reasoning in financial QA.",
      "published": "2026-08-11T15:18:47Z",
      "abstract_url": "http://arxiv.org/abs/2608.11047v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11047v1",
      "categories": [
        "cs.AI",
        "cs.CE",
        "cs.LG"
      ]
    },
    {
      "title": "Workflow Cards: Structured Summaries of Workflow Executions Using Provenance Data",
      "authors": [
        "Nicola Giuseppe Marchioro",
        "Gabriele Padovani",
        "Amal Gueroudji",
        "Rafael Ferreira da Silva",
        "Wesley Brewer",
        "Valentine Anantharaj",
        "Sandro Fiore",
        "Renan Souza"
      ],
      "abstract": "Model Cards and Data Cards have demonstrated the value of structured, human-readable documentation for machine learning artifacts, capturing their context, parameters, limitations, and intended use. However, these practices remain focused on static artifacts (the datasets and trained models themselves) while overlooking the workflow executions that produce, transform, and evaluate them. Such executions hold critical details about data preparation, parameter choice, runtime behavior, resource use, and intermediate transformations, precisely where bias, performance variation, and reproducibility gaps tend to originate. To close this gap, we introduce Workflow Cards: structured summaries that condense the machine-readable provenance data of a workflow execution into a form both humans and large language models (LLMs) can read and analyze. This paper has two main parts. First, it defines a Workflow Card template informed by a representative set of provenance questions that surface from the execution-level data missing from Model and Data Cards. Second, it evaluates how effectively LLMs use Workflow Cards to understand workflow executions compared with querying provenance databases through a schema-based interface. Results show that Workflow Cards provide execution-level information absent from existing card types, such as Model Cards and Data Cards, thereby filling an important documentation gap; and that Workflow Cards nearly double answer quality compared with schema-based querying, consistently across LLM-as-a-Judge and human assessments.",
      "published": "2026-08-11T15:02:11Z",
      "abstract_url": "http://arxiv.org/abs/2608.11022v1",
      "pdf_url": "https://arxiv.org/pdf/2608.11022v1",
      "categories": [
        "cs.DC",
        "cs.AI"
      ]
    },
    {
      "title": "ReLTEx: Reliable LLM-based Taxonomy Expansion",
      "authors": [
        "Zeinab Ghamlouch",
        "Mehwish Alam"
      ],
      "abstract": "Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in generating semantically relevant concepts and relations, making them promising tools for taxonomy enrichment. However, directly relying on LLM-generated expansions often leads to noisy, redundant, or hierarchically inconsistent structures, limiting their reliability for automated taxonomy expansion. In this paper, we present ReLTEx, a framework for reliable LLM-based taxonomy expansion. ReLTEx combines LLM-driven candidate generation with structure-aware validation and recursive expansion control to improve the consistency and quality of generated taxonomies by reducing hallucinations. We evaluate the proposed framework using benchmark taxonomies under a masked taxonomy expansion setting and compare multiple validation strategies. Experimental results, supported by both adapted evaluation metrics and human evaluation, demonstrate that ReLTEx produces more reliable and semantically coherent taxonomy expansions.",
      "published": "2026-08-11T14:29:58Z",
      "abstract_url": "http://arxiv.org/abs/2608.10970v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10970v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "CARE: Confidence-Aware Reasoning for Reliable Medical VQA",
      "authors": [
        "Yuetian Du",
        "Yucheng Wang",
        "Zhenyuan Chen",
        "Luyuan Chen",
        "Rongyu Zhang",
        "Jinjian Zhang",
        "Wei Zhou",
        "Zhijie Xu",
        "Ming Kong",
        "Zhan Zhou",
        "Jie Liu",
        "Qiang Zhu"
      ],
      "abstract": "Reinforcement Fine-Tuning (RFT) has enabled medical Multimodal Large Language Models (MLLMs) to produce Chain-of-Thought (CoT) reasoning for visual question answering, yet these models suffer from $\\textit{confidence miscalibration}$---a systematic gap between expressed certainty and actual diagnostic accuracy that undermines clinical trust. We propose $\\textbf{CARE}$, a $\\textbf{C}$onfidence-$\\textbf{A}$ware medical $\\textbf{RE}$asoning framework that jointly optimizes accuracy and calibration through a dual-stage pipeline. First, a scalable Medical-CoT synthesis provides structured cold-start data for Supervised Fine-Tuning. Second, Group Relative Policy Optimization (GRPO) with a novel $\\textbf{Confidence-Aware Reward (CAR)}$ mechanism ties the model's confidence to diagnostic correctness within the reward signal. Across three Medical VQA benchmarks, $\\textbf{CARE}$ achieves the highest diagnostic accuracy while obtaining the lowest Expected Calibration Error and Hallucination Rate, establishing a foundation for trustworthy clinical decision support. Our code is available at https://github.com/anotherbricki/CARE.",
      "published": "2026-08-11T14:28:52Z",
      "abstract_url": "http://arxiv.org/abs/2608.10964v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10964v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Evidence-Grounded Trustworthy Multimodal Reasoning and Evaluation Benchmark in Complex Urban Scenes",
      "authors": [
        "Zhaoyang Wei",
        "Bowen Jiang",
        "Xumeng Han",
        "Jiashu Li",
        "Xuehui Yu",
        "Yuling Liu",
        "Guorong Li",
        "Zhenjun Han",
        "Jianbin Jiao"
      ],
      "abstract": "While Multimodal Large Language Models (MLLMs) demonstrate impressive performance in benign scenarios, their cognitive reliability deteriorates significantly in complex scenes under adverse conditions. In these settings, models often rely on implicit inference without sufficient visual evidence, leading to a disconnect between perception and reasoning. Meanwhile, existing outcome-oriented benchmarks evaluate only final predictions and fail to diagnose failures in the underlying reasoning process. To address this gap, the authors propose AD2-Bench, which introduces a Hierarchical Visual Diagnosis framework that decomposes reasoning into a structured Chain of Evidence (CoE). This fine-grained diagnosis reveals that robust multimodal reasoning fundamentally depends on accurate evidence acquisition. Building on this perspective, the authors formulate reasoning from a probabilistic viewpoint and identify two primary causes of reasoning failure: Spatial Ambiguity, where models fail to distinguish target objects from background clutter, resulting in localization errors; and Semantic Uncertainty, where degraded visual features lead to incorrect semantic interpretation, resulting in understanding errors. To overcome these evidence deficiencies, they further propose Evidence-grounded Visual Reasoning (EGVOR), which replaces implicit reasoning with the explicit generation of Evidence Atoms - structured spatial-semantic triplets that enforce tight alignment between localization and semantic understanding. The model is trained through a hierarchical curriculum that progresses from reflective supervision construction to reinforcement learning, where reducing reasoning variance is explicitly rewarded. Extensive experiments demonstrate that EGVOR substantially improves reasoning stability under adverse conditions, providing a more robust framework for trustworthy multimodal cognition.",
      "published": "2026-08-11T14:23:07Z",
      "abstract_url": "http://arxiv.org/abs/2608.10954v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10954v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Temporally Grounded Compositional Camera Motion Understanding via Geometric Knowledge Distillation",
      "authors": [
        "Dazhao Du",
        "Shiyan Du",
        "Jian Liu",
        "Yongjian Yu",
        "Bohai Gu",
        "Tao Han",
        "Hualuo Liu",
        "Eric Liu",
        "Yujia Zhang",
        "Xi Chen",
        "Song Guo"
      ],
      "abstract": "Understanding camera motion is fundamental to video perception, with applications in spatial intelligence and controllable video generation. Multimodal large language models (MLLMs) provide a natural interface for this task, but existing work typically assigns one or more labels to an entire clip. Such clip-level recognition overlooks two defining properties of real camera motion: it can change within a shot, and multiple movements can occur simultaneously. We therefore formulate camera-motion understanding as temporally grounded, compositional recognition, which requires a model to localize motion-consistent intervals and identify every movement active within each interval. We introduce CamChoreo, a benchmark of 4,229 real single-shot clips with expert-annotated temporal segments. Its annotations use a compact vocabulary of 20 direction-aware labels, and nearly half of the segments contain compound camera motion, with multiple movement primitives active simultaneously. Recognizing such fine-grained, compositional motion is hard for current MLLMs, whose visual encoders emphasize semantic content rather than the geometric evidence on which camera motion depends. Directly injecting features from a frozen 3D foundation model addresses this gap, but requires running the expensive geometry model on every input; we refer to this baseline as CamInject. We instead propose CamDistill, which distills the same geometric knowledge into lightweight camera tokens during training and removes the 3D model at inference. CamDistill matches the accuracy of direct feature injection without running the 3D teacher at inference. Together, CamChoreo and CamDistill advance camera-motion understanding from clip-level labeling to temporally grounded, compositional recognition. Project page: https://ddz16.github.io/cammotion.github.io/.",
      "published": "2026-08-11T14:00:02Z",
      "abstract_url": "http://arxiv.org/abs/2608.10932v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10932v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?",
      "authors": [
        "Xiaohongshu Inc"
      ],
      "abstract": "Large language model (LLM) agents are increasingly deployed as personal assistants. Existing evaluations, however, mostly use short, self-contained requests in static environments. Everyday life assistance is different. A task runs for weeks rather than minutes. The world keeps changing while the agent is not being prompted. Many constraints are never stated outright. An agent that merely answers the request in front of it will fail at such a task. What is needed instead is an agent that stays proactive and consistent. It decides on its own when to act, when to ask, and when to stay silent. It notices changes that nobody announced. It keeps one plan coherent from the first day to the last. No current benchmark measures this. We introduce VibeLifeBench, a benchmark of 200 long-horizon tasks across ten everyday-life domains. Each task is a scripted multi-week timeline in a simulated world of 22 mock services. The world advances on its own clock, and many of its changes are silent, so only an agent that re-inspects the world discovers them. Every task is graded by fine-grained, weighted checks that read only what the agent actually left behind, covering the end state, the timeliness of its actions, and whether it upheld the implicit constraints. We evaluate seven frontier models. All of them score low, which shows how far current agents are from assisting with real life. We will open-source all tasks, environments, and the evaluation framework.",
      "published": "2026-08-11T12:52:38Z",
      "abstract_url": "http://arxiv.org/abs/2608.10875v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10875v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "TACTICL: Task-Aware Compression of Tabular ICL Models",
      "authors": [
        "Mykhailo Koshil",
        "Matthias Feurer",
        "Katharina Eggensperger"
      ],
      "abstract": "The strong performance of foundation models for tabular tasks comes at substantial inference costs. Distilling models into task-specific architectures reduces model size and computational demands but also sacrifices in-context adaptability. Here we introduce TACTICL, an automated task-aware compression framework for tabular in-context learning models that jointly prunes transformer layers and replaces them with lightweight adapters trained on downstream tasks, thus blending in-context with in-weight learning. We study TACTICL on 47 benchmark datasets and show that we can substitute up to 85% of layers without substantial performance drop on a given downstream task. We further show that TACTICL maintains robustness to data shifts, leaving its in-context ability intact. Overall, TACTICL provides a robust framework for exploiting the depth-wise redundancy of tabular foundation models by combining task-specific adaptation and structured compression. We provide the code at: https://github.com/Hebog/tfm_compression",
      "published": "2026-08-11T12:03:37Z",
      "abstract_url": "http://arxiv.org/abs/2608.10837v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10837v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Reference-Free Post-Training of Open Large Language Models for Multilingual Machine Translation",
      "authors": [
        "Chris Han",
        "Pengzhi Gao",
        "Pei Fu",
        "Jian Luan"
      ],
      "abstract": "We study reference-free post-training for multilingual machine translation with open large language models. Starting from the supervised-finetuned MiLMMT-46-v0.1 models, we apply Group Relative Policy Optimization (GRPO) with a reward that averages two reference-free quality estimation models and is gated by language identification. We then linearly interpolate the supervised fine-tuning (SFT) and reinforcement learning (RL) model checkpoints to obtain MiLMMT-46-v1.0. Across 46 languages, the resulting models consistently improve translation quality over their SFT counterparts, outperform strong recent open baselines, including Seed-X, HY-MT2, and TranslateGemma, and achieve leading reference-free scores against evaluated proprietary systems such as Google Translate, Gemini 3 Pro, and GPT-5. We further investigate on-policy distillation and find that it reaches, but does not surpass, the quality frontier achieved by RL with checkpoint interpolation. We release the models and code to facilitate future research.",
      "published": "2026-08-11T11:30:38Z",
      "abstract_url": "http://arxiv.org/abs/2608.10812v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10812v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "BPG: Balancing Plasticity and Generalization for Domain Incremental Learning",
      "authors": [
        "Qiang Wang",
        "Songlin Dong",
        "Shaokun Wang",
        "Jizhou Han",
        "Xiang Song",
        "Chenhao Ding",
        "Yuhang He",
        "Yihong Gong"
      ],
      "abstract": "Deep neural networks excel in various tasks but struggle to generalize across evolving data distributions, leading to significant performance degradation under domain shifts. Domain incremental learning (DIL) addresses this challenge by enabling models to continuously adapt while retaining prior knowledge. Among existing DIL approaches, the parameter-isolation paradigm achieves state-of-the-art performance. However, these methods often adopt a one-size-fits-all approach to adapt to new domains, resulting in either insufficient learning capacity or redundant parameters. In this work, we propose BPG, a unified framework that addresses both challenges through two complementary components: BPG-Adapter, which dynamically determines each domain's adapter hidden dimension based on domain-specific feature separability, and BPG-Inference, a soft domain mixture strategy that integrates multiple domain-specific models at test time, mitigating domain ID misselection. Experimental results on DomainNet, CDDB, and CORe50 demonstrate that BPG consistently outperforms uniform adapter-based approaches and hard domain selection strategies, achieving state-of-the-art average accuracy while reducing forgetting to as low as 0.22% on DomainNet.",
      "published": "2026-08-11T11:22:27Z",
      "abstract_url": "http://arxiv.org/abs/2608.10804v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10804v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Beyond Fixed Luminance: Towards Panchromatic and Orthochromatic Image Colorization",
      "authors": [
        "Swarnim Maheshwari",
        "Syed Imam Ali",
        "Vineeth N. Balasubramanian"
      ],
      "abstract": "Most image colorization systems operate in $Lab$ space by predicting chroma ($ab$) while preserving an input-derived luminance channel ($L$). While effective on standard benchmarks, this fixed-luminance design restricts brightness changes and becomes unreliable when grayscale formation deviates from natural-image luminance, as in historical orthochromatic photography. We propose a luminance-agnostic colorization framework that formulates colorization as full-RGB image editing using a foundation image-editing model. To bridge modern panchromatic and historical orthochromatic conditions, we introduce a mixed grayscale objective that trains the model under both standard luminance grayscale and a red-insensitive grayscale formation. Experiments on COCO, ImageNet, and a multi-instance benchmark show that our method is competitive on standard grayscale inputs and substantially more robust under orthochromatic inputs, with qualitative comparisons and a human study indicating fewer visible color artifacts.",
      "published": "2026-08-11T11:12:24Z",
      "abstract_url": "http://arxiv.org/abs/2608.10798v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10798v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "ChemWorld: Programmable Chemical Worlds for Controlled and Replayable Agent Experimentation",
      "authors": [
        "Jiangjie Qiu",
        "Yijun Li",
        "Xiaonan Wang"
      ],
      "abstract": "Autonomous chemistry increasingly depends on environments in which agents can repeatedly act, observe, and adapt.Physical laboratories provide essential real-material evidence but are costly to repeat and difficult to use for tightly matched interventions, whereas most digital environments keep the underlying experimental world largely fixed. We introduce ChemWorld, a programmable chemical environment in which reusable process and observation components are compiled into executable worlds. ChemWorld separates the public experimental contract available to an agent from evaluator-owned chemical and material laws. Researchers can therefore vary world composition and operating conditions, or change a single hidden law while holding the public task and interaction conditions fixed. Transactional execution records operations, failures, resource changes, and state transitions, allowing complete environment-action trajectories to be replayed exactly and audited. Full-census qualification covered the reference registry, 52 generated compositions, and module, interface, compilation, and invalid-action tests. Eight deterministic experimental cases demonstrated shared lifecycle semantics, failure recovery, and exact replay, while six parent-child world-fork pairs isolated the effects of single private-law interventions under matched public conditions. An independent agent also completed a full lifecycle in a non-reference world through the same public interface. Within the declared component and model domain, ChemWorld provides a controlled and replayable substrate for studying experimentation across systematically varied chemical worlds, complementary to physical-laboratory evidence and calibration.",
      "published": "2026-08-11T10:53:39Z",
      "abstract_url": "http://arxiv.org/abs/2608.10792v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10792v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Rule of Thumb: Explaining Artificial Intelligence Systems using Partial Information",
      "authors": [
        "Kaivalya Rawal",
        "Daria Onitiu",
        "Brent Mittelstadt",
        "Sandra Wachter",
        "Chris Russell"
      ],
      "abstract": "Explainable Artificial Intelligence (XAI) seeks to explain how an Artificial Intelligence (AI) system arrived at a particular decision. We propose ''Rule of Thumb'' (RoT) explanations, a new approach to XAI based upon a novel formulation that identifies the most relevant features for predicting the behaviour of an AI system, for a particular datapoint. We show how RoT is well-suited to enable XAI in: (a) zero-shot classification using large language models (LLMs), (b) auditing of opaque AI systems without model access, and (c) the use of AI in scientific discovery. Additionally, RoT meets specific requirements from leading AI regulations, provides a familiar interface and visualisations for XAI practitioners, is model-agnostic, and is substantially faster than alternatives. Code available at: https://github.com/KaiRawal/Rule-of-Thumb-Explaining-Artificial-Intelligence-Systems-using-Partial-Information",
      "published": "2026-08-11T10:23:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.10766v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10766v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "stat.ML"
      ]
    },
    {
      "title": "Conversational Orchestration for Organic 6G",
      "authors": [
        "Masoud Shokrnezhad",
        "Tarik Taleb"
      ],
      "abstract": "The Organic 6G vision of a network of networks spanning an edge-cloud continuum complemented by non-terrestrial resources requires, to realize its promise, service provisioning that is simple to operate, scalable across independently administered domains, and agile under domain churn (i.e., domains dynamically joining and leaving). Despite advances in cross-domain orchestration, many proposals rely on heavy integration fabrics, multi-layer coordinators, and deep telemetry pipelines that hinder deployability and amplify coordination overhead. We propose a lightweight, decentralized conversational orchestration framework based on Large Language Model (LLM)-driven domain agents. Each domain remains autonomous: an agent observes local state via tools, reasons in a closed loop, and exchanges summaries with neighboring agents over an Agent-to-Agent (A2A) overlay aligned with data-plane coupling. Fast feasible placement is enabled by periodic, routing-like dissemination of reachability advertisements (latency, bottleneck bandwidth, and compute capacity), while safe re-optimization, scaling, and migration are handled through event-driven requests and negotiation. To meet real-time constraints, we deploy a compact reasoning model trained with verifier-based self-verification and periodically refined online via shadow updates. Simulations show manageable, near-linear control-plane overhead as domains scale and during domain joins, and robust decision quality, including recovery after objective changes. We close by outlining future research directions for principled, secure, and uncertainty-aware agentic orchestration in Organic 6G.",
      "published": "2026-08-11T09:32:52Z",
      "abstract_url": "http://arxiv.org/abs/2608.10714v1",
      "pdf_url": "https://arxiv.org/pdf/2608.10714v1",
      "categories": [
        "cs.NI",
        "cs.AI",
        "cs.DC",
        "cs.ET",
        "cs.MA"
      ]
    }
  ]
};
