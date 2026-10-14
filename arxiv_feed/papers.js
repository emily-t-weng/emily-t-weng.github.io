const PAPERS_DATA = {
  "last_updated": "2026-10-14 04:33:41 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "A Global View of How LLMs Value Human Time",
      "authors": [
        "Yifan He",
        "Sofie Goethals",
        "Manon Reusens",
        "David Martens"
      ],
      "abstract": "Large language models (LLMs) increasingly act as personal assistants; however, little is known about how they capture national-level preferences on a global scale. Given that \"time\" is a universal and representative concept, we examine LLMs' time preferences by varying only users' country and measuring the compensation LLMs deem appropriate for a 60-minute appointment delay. We then compare LLMs' responses with GDP per capita on Purchasing Power Parity (GDP_PPP), Mean Income, and a literature-based index (Patience Score). Our results across 193 countries and 5 LLMs demonstrate that people's time across the world is evaluated in noticeably different ways among LLMs, as well as varying levels of robustness under prompt variations. LLMs consistently recommend the highest compensation for Europe, but mostly the lowest for Africa. Based on GDP_PPP, LLMs underestimate the value of time for users living in Asia and South America. These findings indicate that geographical patterns are embedded in LLMs and may translate in different treatment in agentic settings.",
      "published": "2026-10-13T15:08:38Z",
      "abstract_url": "http://arxiv.org/abs/2610.18515v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18515v1",
      "categories": [
        "cs.CY",
        "cs.AI"
      ]
    },
    {
      "title": "The Smart Buildings Challenge: Learning Foundation Models for Real-World Optimization at Scale",
      "authors": [
        "Judah Goldfeder",
        "Ayush Rai",
        "Philippe M. Wyder",
        "Na Li",
        "Victoria Dean",
        "Donna Vakalis",
        "Jiong Lin",
        "Bing Dong",
        "Michael Rossetti",
        "Nathan Kutz",
        "Sohei Arisaka",
        "John Sipple",
        "Hod Lipson",
        "Yoshua Bengio"
      ],
      "abstract": "Buildings cause nearly 40\\% of carbon emissions in the US. Of total building emissions, HVAC systems are responsible for approximately 40-60\\% and account for 15\\% of global energy consumption. Despite advances in artificial intelligence, most buildings still operate as inefficiently as they did decades ago. Building optimization, perhaps more so than any other optimization problem, has enormous potential for Machine Learning (ML) driven impact on climate change, in both the short and long term. One of the biggest obstacles for building optimization is the lack of policy transfer. Most building control systems, whether simulation or learning based, don't generalize: they only work well for the building they were developed for, but not for other buildings. To solve this problem, we propose a challenge for the development of building foundation models that easily adapt to new buildings with minimal configuration and thereby enable building optimization to scale across the globe. Participants will build control policies on a set of simulated buildings, and have their policies evaluated on an unseen set of buildings. The buildings vary in size, layout, weather, material properties, and HVAC system. Successful outcomes from this initiative promise a massive leap forward in global sustainability efforts and demonstrate the great potential of ML to combat climate change.",
      "published": "2026-10-13T15:06:08Z",
      "abstract_url": "http://arxiv.org/abs/2610.18511v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18511v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "EgoOmniWorld: Unifying Egocentric Understanding via Ego-Action-World Interactions",
      "authors": [
        "Songruo Chen",
        "Yuqian Fu",
        "Yu Li",
        "Mingkang Dong",
        "Sifan Zhou",
        "Bin Ren",
        "Xu Zheng",
        "Tianwen Qian",
        "Mohamed Elhoseiny",
        "Yanwei Fu"
      ],
      "abstract": "Egocentric video understanding spans diverse capabilities, modalities, and evaluation formats, yet these are often studied in isolation despite their tight coupling in real-world interaction. We propose the Ego-Action-World interaction loop as a unifying principle: the wearer perceives the world, forms goals, acts, changes the world, and uses feedback to guide subsequent behavior. Based on this view, we introduce EgoOmniWorld, a benchmark organized around four complementary interaction structures: Node captures ego, action, and world states; Edge captures directed relations among them; Chain captures linked dependencies; and Loop captures feedback-closed interactions. We develop an interaction-first, evidence-grounded data pipeline that extracts structured interactions from egocentric videos, grounds them in visual and auditory evidence, and generates questions according to their interaction structure. Required modalities and single- or multi-turn formats follow naturally from the supporting evidence and dependencies. EgoOmniWorld contains 24,166 QA groups, including 20,284 training and 3,882 test samples. Experiments on representative open- and closed-source multimodal large language models reveal substantial and uneven limitations across interaction structures. Models trained on EgoOmniWorld improve in-domain interaction understanding and generalize to existing egocentric benchmarks. EgoOmniWorld provides a unified foundation for evaluating and advancing egocentric video understanding. Data, code, and models will be released.",
      "published": "2026-10-13T14:53:16Z",
      "abstract_url": "http://arxiv.org/abs/2610.18499v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18499v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Task-evoked eye movements as candidate digital biomarkers of depression severity in adolescents and young adults",
      "authors": [
        "Youxiang Sun",
        "Yujie Nie",
        "Zhuoru Wu",
        "Mengyuan Yu",
        "Wenkong Wang",
        "Xiang Li",
        "Yumei Wang",
        "Xin Ma",
        "Lizhou Fan"
      ],
      "abstract": "Objective digital measures of depression severity could complement conventional symptom assessments and facilitate more timely clinical evaluation. Here, we investigated whether task-evoked eye movements provide interpretable behavioral markers of depression severity in adolescents and young adults. In this single-center, cross-sectional study, 155 participants aged 14--20 years completed emotional-spatial free-viewing and memory-guided recall tasks and were stratified by 17-item Hamilton Depression Rating Scale scores. We examined 19 prespecified eye-movement features across six task-informed domains using covariate-adjusted ordinal-trend models. Greater depression severity was associated with lower gaze-allocation entropy (standardized $β=-0.599$), fewer inter-area switches ($β=-0.526$), longer first target-entry latency ($β=0.572$), and lower target dwell proportion ($β=-0.678$; all false-discovery-rate-adjusted $q<0.001$). Emotion-specific attentional-bias measures were not associated with severity, whereas associations involving fixation dispersion weakened after adjustment for anxiety symptoms. In a complementary machine-learning analysis incorporating 44 candidate eye-movement features and demographic covariates, a three-stage cascade classifier achieved the highest four-class stratification performance (accuracy, 0.665; macro-F1, 0.670; area under the receiver operating characteristic curve, 0.869), outperforming single-stage and two-stage approaches. Together, these findings identify altered visual exploration and memory-guided gaze control as candidate digital biomarkers of depression severity, rather than generalized negative attentional bias. These results support further prospective evaluation of task-evoked eye tracking as a complementary approach for timely severity assessment and clinical decision support.",
      "published": "2026-10-13T14:51:53Z",
      "abstract_url": "http://arxiv.org/abs/2610.18494v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18494v1",
      "categories": [
        "eess.SP",
        "cs.AI"
      ]
    },
    {
      "title": "ReTerm: Recovery Training for Self-Improving Terminal Agents",
      "authors": [
        "Wei Xie",
        "Kaituo Feng",
        "Shiming Fu",
        "Yong Liu",
        "Xiangyu Yue",
        "Shuangjia Zheng"
      ],
      "abstract": "Terminal agents have shown strong abilities in complex coding tasks. However, they often struggle to recover from errors during long-running execution (e.g., command errors, faulty implementations), eventually leading to task failures. To this end, we present ReTerm, a self-improving terminal agent trained by learning how to recover from its own failed terminal tasks. To support its training, we construct a customized data pipeline and produce ReTerm-10K, a high-quality dataset that provides effective recovery demonstrations for SFT cold start. Furthermore, we propose ReTerm-GRPO, an online RL algorithm that forks from the agent's own failure states to generate multiple independent self-recovery attempts. By learning from whether these attempts successfully complete the task, the agent self-improves through its own recovery experience. Built on Qwen3.5-9B, our ReTerm-9B achieves 34.8% Pass@1 on Terminal-Bench 2.0, improving over the base model by 14.6 points. Further analysis shows that our method achieves consistently higher recovery rates across diverse error categories compared with standard GRPO training. All code, data and model are released at https://github.com/twothsnd/ReTerm.",
      "published": "2026-10-13T14:44:24Z",
      "abstract_url": "http://arxiv.org/abs/2610.18484v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18484v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "Readout Scaling Shapes Sparse Feature Recovery",
      "authors": [
        "Brain Xiao",
        "Sarah Wilson"
      ],
      "abstract": "Readout scaling changes what a sparse autoencoder (SAE) learns, but the benefit of removing it depends on the data. We use readout-canonicalized decision-space (RC-DS) coordinates to remove singular-value gains while preserving retained decision information. At a fixed readout condition number, changing the generating distribution reverses which training coordinates recover known factors best. In our controlled coordinate-matching study, recovery is highest when training restores the space in which the factors were generated. When factors are generated before readout scaling, RC-DS increases mean sparse-factor recovery from 0.506 to 0.913 at condition number 100, compared with centered-logit training under the same budget. An exact wrapper reproduces RC-DS training to numerical precision, isolating training geometry rather than additional information. In neural policies, RC-DS improves mean logit reconstruction under the tested budgets. On three fixed language-model checkpoints, RC-DS has higher mean gain-free reconstruction in all 18 tested task-sparsity settings, while centered-logit SAEs reconstruct output logits better in 17. Small-label-head controls show that longer training reduces the reconstruction gap from 0.0198 to 0.0016. Together, these results show how readout scaling, the data distribution, and training shape sparse learning. RC-DS provides an information-preserving method for studying and controlling the readout's contribution. Separate label-detection and steering tests distinguish these reconstruction effects from task-specific feature usefulness.",
      "published": "2026-10-13T14:28:22Z",
      "abstract_url": "http://arxiv.org/abs/2610.18462v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18462v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Learning Perturbation-Specific Representations through Conditional Energy Prior for Cellular Response Prediction",
      "authors": [
        "Yuchen Yan",
        "Yatao Bian"
      ],
      "abstract": "AI-based cellular perturbation prediction is essential to accelerate drug discovery and therapeutic identification. However, perturbation-specific signals are generally obscured by cellular complexity, making it difficult for existing methods to correctly capture perturbation effects. To address this problem, we propose PerturbEnergy, which introduces a conditional energy-based prior over perturbation representations. It assigns lower energy (higher probability) to representations compatible with the given perturbation and higher energy to others, thereby explicitly tilting the latent space toward perturbation-preferred regions. Besides, PerturbEnergy can be integrated into autoencoder architectures as we find its Evidence Lower Bound (ELBO) is equivalent to the ELBO of the complete autoencoder. We further prove that optimizing PerturbEnergy increases the lower bound on the conditional mutual information between representation and perturbation. Extensive experiments on four benchmark datasets demonstrate the effectiveness of our PerturbEnergy: it improves perturbation recognition especially in latent space with improvement of up to 83.2\\%; It also achieves state-of-the-art performance for predicting top-50/100/200 differentially expressed gene(DEG) response and remaining competitive on highly variable gene(HVG) prediction. The code is provided: https://github.com/BlueWhaleLab/PerturbEnergy",
      "published": "2026-10-13T14:24:25Z",
      "abstract_url": "http://arxiv.org/abs/2610.18456v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18456v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "How Well Do LLMs Perform in Peer Review? Evidence from Finance and Accounting",
      "authors": [
        "Hai Lu",
        "Yichen Luo",
        "Luping Yu"
      ],
      "abstract": "Referees increasingly use large language models (LLMs), while finance and accounting journals disagree about whether to allow it. We ask which part of a referee's job the models can do. GPT-4o, whose training data end before the 2024 conference season, referees the papers presented at the 2024 AFA and NBER finance meetings, and we compare its verdicts with publication in the top-three finance journals over the next two years. Papers the model scores one standard deviation higher are 44\\% more likely to be published, net of author reputation. The result is not driven by look-ahead bias, as the model does not recall the outcomes. The information comes from its judgements of correctness and exposition. Its judgements of originality and significance carry none, and a referee guide that stresses importance does not help. The split between correctness and contribution holds for accounting papers and for a newer model, GPT-5. LLMs can check whether research is right, but not whether it matters. The technical check can go to the model, while the judgement of contribution stays with people.",
      "published": "2026-10-13T14:11:35Z",
      "abstract_url": "http://arxiv.org/abs/2610.18435v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18435v1",
      "categories": [
        "q-fin.GN",
        "cs.AI"
      ]
    },
    {
      "title": "FedGraph: Defending Federated Large Language Model Fine-Tuning Against Backdoor Attacks via Graph-Based Aggregation",
      "authors": [
        "Xi Chen",
        "Chunyi Zhou",
        "Rui Zeng",
        "Xiaogang Xu",
        "Zhe Liu",
        "Shouling Ji"
      ],
      "abstract": "Federated fine-tuning of large language models (LLMs) allows multiple clients to collaboratively adapt models without sharing raw data, offering strong privacy benefits. However, this setting is highly vulnerable to backdoor attacks, in which malicious clients manipulate their updates to induce targeted behaviors while maintaining normal performance on benign inputs. In this work, we make a key observation specific to federated LLM fine-tuning: malicious clients tend to exhibit strong consistency in their model updates, whereas benign clients produce more diverse and dispersed updates. This structural difference is not well captured by existing defense methods, which typically rely on direct anomaly detection in high-dimensional parameter space and therefore struggle in the presence of entangled LLM updates. Motivated by this observation, we propose FedGraph, a graph-based aggregation framework that explicitly models relationships among client updates. By representing clients as nodes in a dynamically constructed graph and extracting structural features that characterize their relative positions and interactions, FedGraph derives compact client representations that effectively separate malicious and benign behaviors. These representations are then used to identify and suppress malicious contributions during aggregation. Extensive experiments demonstrate that FedGraph significantly improves backdoor defense. It consistently reduces attack success rates to below 10% while achieving high detection accuracy (95.5% on average) and a low false positive rate (2.33%), substantially outperforming existing federated learning defenses.",
      "published": "2026-10-13T14:08:49Z",
      "abstract_url": "http://arxiv.org/abs/2610.18430v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18430v1",
      "categories": [
        "cs.CR",
        "cs.AI"
      ]
    },
    {
      "title": "Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages",
      "authors": [
        "Ramchand Kumaresan",
        "Madhusudhanan G",
        "Kaushik Kezendirane"
      ],
      "abstract": "We trained Agaram 2.5B, a 2.5-billion-parameter Tamil-English language model, from scratch on about 50 billion tokens (58% Tamil, the rest English, math, code and parallel text) on one node of eight H100 GPUs in about six days, with a new tokenizer that, unlike our first model's, keeps line breaks. Supervised fine-tuning on 134,428 conversations, whose replies an open-weight teacher model wrote, and three short targeted rounds turned it into a chat model that answers in Tamil script, romanized Tamil (Tanglish) and English and can call tools: on 25 tool prompts whose schemas appear in training it picks the right tool 23 times and gives exactly the expected arguments 13 times. The released model translates better than Sarvam-1, a 2B Indic base model, in all six FLORES and IN22 directions (FLORES chrF 54.3 into Tamil and 55.2 into English, against 49.2 and 44.2), and nearly doubles its base model on native-Tamil IndicIFEval (19.3% from 10.0%). It is weak on knowledge: reading-comprehension and knowledge benchmarks sit near chance, and on 50 held-out everyday questions written by one native speaker, one LLM judge rated 46% of its replies usable (22-24 of 50 over three passes; 95% interval for 23 of 50, 33-60%), with nearly every failure a wrong fact. Our two further SFT experiments changed answer format but not the knowledge measures. We also describe our first model, Agaram 1.5B, whose failures shaped the choices here; how the judge's scores varied between passes on identical replies; and, item by item, what the targeted rounds trained against and where frontier-model text entered the data. The project ran on compute credits and our own funds, about USD 13,300 across both models; the weights are released (licence to be announced after legal review), with GGUF files for llama.cpp, at https://huggingface.co/murai-labs/agaram-2.5b-instruct; code at https://github.com/Murai-Labs/Agaram2.5b",
      "published": "2026-10-13T14:00:05Z",
      "abstract_url": "http://arxiv.org/abs/2610.18420v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18420v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Do Factuality Methods Make LLMs More Factual? Reproducibility and Fair Comparison",
      "authors": [
        "Pratuat Amatya",
        "Vinay Setty",
        "V Venktesh",
        "Avishek Anand"
      ],
      "abstract": "Large Language Models (LLMs) are aligned to human preferences, after pre-training and supervised fine-tuning steps. However, standard RL alignment steps such as RLHF do not focus on factuality of LLMs, hence they may generate non-factual information, which could result in unintended consequences. In response, a number of factuality alignment approaches have been proposed in the literature which claim to improve factuality of LLMs on their own benchmarks. However, these methods have not been reproduced independently and are also not comparable since each uses a different base model, training data, evaluation set, judge, and metric. In this paper, we present the first independent reproduction of five factuality alignment methods, which span on-policy RL (PPO, GRPO, and Dr.~GRPO) and offline preference optimization (DPO and KTO), and compare them on a common benchmark with common evaluation metrics. We found that none of these methods reproduces in full, and on the common benchmark none of the methods significantly outperforms the base model. On closer inspection, the rewards optimize for format, length, or brevity rather than factuality and relevance, whereas the choice between on-policy and offline training matters little. Furthermore, the evaluation metrics and benchmarks are also part of the problem since they do not measure relevance of text to the prompt. Since there is no ground truth associated with the prompts, the reliance on LLM judges and retrieval for verification also makes the evaluation fragile. We release our code, data, and checkpoints.",
      "published": "2026-10-13T13:54:51Z",
      "abstract_url": "http://arxiv.org/abs/2610.18415v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18415v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Not All Drift Is a Mechanism: Tracing Prompt Adaptation in LLMs",
      "authors": [
        "Mikhail Aleksandrov",
        "Nail Bashirov",
        "Maksim Ivanov",
        "Andrew Tsukanov",
        "Timofey Karyagin",
        "Alexander Miasnikov",
        "Gleb Molodtsov",
        "Valentin Samokhin",
        "Aleksandr Beznosikov"
      ],
      "abstract": "Prompt-based adaptation offers several ways to specialize a frozen language model, ranging from natural-language prompt optimization to learned continuous prompts. These methods are usually compared through downstream performance, although similar outputs need not arise from similar internal computations. We study how discrete prompt optimization, prompt tuning, and prefix tuning alter the processing of dense and mixture-of-experts language models. Our analysis shows that the location of the adaptation predicts how task information is carried to the output: optimized text primarily changes response construction, input-level prompts write their influence into content representations, and layer-wise prefixes maintain direct control through attention. In MoE models, these differences further induce distinct expert-routing patterns. At the same time, common representation-level measurements respond strongly to generic context changes and do not reliably distinguish useful adaptations from matched placebos. We argue that understanding prompt adaptation requires separating representation drift from the part of that drift that is aligned with, and causally used by, the model's task computation. This perspective provides practical guidance for selecting, evaluating, and interpreting prompt adaptation methods.",
      "published": "2026-10-13T13:52:52Z",
      "abstract_url": "http://arxiv.org/abs/2610.18412v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18412v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PRISM: Calibrating Black-Box Judges with Counterfactual Probes",
      "authors": [
        "Kakei Yamamoto",
        "So Kuroki"
      ],
      "abstract": "Large language models are increasingly used as judges to reduce reliance on costly human evaluation, yet their scores can systematically disagree with human judgments. We ask whether responses to an explicitly imposed evaluative bias can help correct a judge's implicit scoring biases. We introduce PRISM (Probe-based Residual Inference of Scoring Misalignment), which holds the evaluated content fixed and measures score changes when attributes are emphasized or de-emphasized. A local sensitivity model motivates representing these responses as a surrogate Jacobian. Ridge regression on human labels learns a correction to the baseline score. The resulting scoring protocol is rubric-free: it uses overall scores without requesting criterion-level ratings. In the main Jev and Sol comparisons on SummEval summarization and HANNA story evaluation, PRISM and its learned-baseline variant significantly improve agreement with human rankings over uncalibrated scores while remaining competitive with rubric-based methods. A controlled SummEval comparison yields similar agreement for rubric-derived quality probes and style probes, supporting calibration without reproducing rubric criteria. The learned coefficients reveal shared and opposing correction directions across judges.",
      "published": "2026-10-13T13:39:50Z",
      "abstract_url": "http://arxiv.org/abs/2610.18389v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18389v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Choosing LLMs for Strategic Agent Interactions",
      "authors": [
        "Arastun Mammadli",
        "Herbert Woisetschläger",
        "Jawad Fayaz",
        "Shiqiang Wang"
      ],
      "abstract": "Game-theoretic benchmarks claim that large language model (LLM) strategic decision-making varies across game types, interaction structures, and capability dimensions. However, they only evaluate strategic performance across individual fixed models. In parallel, contemporary LLM routers pick models under quality-cost trade-offs, but with no notion of a strategic structure. They do not account for an opponent that might be unknown, and that can adapt over time and shift the feedback signal. To address this, we extend the game-theoretic evaluations and define strategic selection as the meta game layered on top of the base game. In each round, a strategic selector fields an LLM from a portfolio of models to navigate a game payoff matrix that is latent and learned online. We find that the evaluation signal is game-specific, and that extracting a reliable signal requires a much higher number of seed runs $N$ compared to traditional benchmarking setups. Evaluating across 5 games and 10 LLMs, we show a model's strategic competence varies significantly depending on the game setup and the opponent faced. Building on this, we implement a suite of selector methods and find that approaches that rely on query embeddings or LLM self-reflection do not consistently capture the underlying strategic signal. A lightweight multi-armed bandit algorithm performs more reliably across games, though it still falls short of the best fixed model. Finally, we show that simple non-stationary and adversarial bandit variants fail to outperform a stationary bandit baseline, pointing to a gap in how current methods handle a shifting opponent in strategic settings.",
      "published": "2026-10-13T13:21:27Z",
      "abstract_url": "http://arxiv.org/abs/2610.18354v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18354v1",
      "categories": [
        "cs.AI",
        "cs.IR",
        "cs.MA"
      ]
    },
    {
      "title": "Lightweight but Linguistically Fair? Quantisation Robustness across Bavarian, German, and English",
      "authors": [
        "Jophin John",
        "Michael Hoffmann",
        "Piroska Lendvai",
        "Florent Dufour"
      ],
      "abstract": "Post-training quantisation reduces the memory cost of large language models, but it remains unclear whether compression preserves language skills of regional varieties as reliably as their standard-language counterparts. We study this question for Bavarian, Standard German, and English using BAVGROUND, a multiple-choice benchmark, where 206 source questions are aligned across all three languages. Across 14 open-weight 7B--10B instruction models, we compare BF16 with 8-bit GPTQ and two 4-bit settings, GPTQ and AWQ. Eight-bit GPTQ remains close to BF16 and largely preserves the relative behaviour of the three languages, while 4-bit quantisation was found to be much less stable. The two 4-bit methods reduce average accuracy by about 1.4--1.5 percentage points and change 11.9--16.0% of BF16 decisions, with changes concentrated on examples where the BF16 model is already uncertain. This instability does not translate into a consistent Bavarian disadvantage. Although Bavarian, a low-resource dialect, has lower baseline accuracy than German and English, the language most affected at 4 bits varies across models and quantisation methods. Calibration further shapes these cross-language effects: the relative Bavarian--German effect changes with calibration language, while changing the calibration source has little average effect on GPTQ-4 but can affect AWQ-4 much more strongly. Free-form evaluation on five models further shows that 4-bit compression can degrade generated responses even in settings where multiple-choice accuracy changes only modestly, while 8-bit GPTQ remains comparatively stable. Overall, our results show that quantisation robustness for regional language varieties depends on the model, quantisation method, calibration data, and evaluation setting, and cannot be measured by aggregate multilingual accuracy alone.",
      "published": "2026-10-13T13:15:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.18343v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18343v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "Efficient Safety Guardrails through Latent Dual-Hypothesis Reasoning with On-Demand Trace Reconstruction",
      "authors": [
        "Md Asiful Islam",
        "Fahmida Alam",
        "Mihai Surdeanu"
      ],
      "abstract": "Guardrail approaches that rely on reasoning-based large language models (LLM) improve safety decisions, but come with substantial inference overhead due to the explicit reasoning. We introduce LR-Guard (Latent Reasoning Guardrail), an efficient and explainable guardrail that jointly learns structured latent reasoning and on-demand reasoning reconstruction. Our method distills a frozen teacher's analysis of safe and unsafe prompt interpretations and their comparison into compact latent representations. Joint training optimizes these representations for both safety prediction and reconstruction of the teacher's explicit reasoning traces, enabling human-readable explanations without requiring intermediate text generation for every decision. Experiments show that LR-Guard outperforms its explicit-reasoning teacher in most in-domain and out-of-domain settings on safety classification and explanation extraction, while achieving a 19.4x inference speedup at batch size one. Further, reconstructed reasoning traces from the latent information exhibit high semantic similarity to the teacher's explicit reasoning.",
      "published": "2026-10-13T12:49:47Z",
      "abstract_url": "http://arxiv.org/abs/2610.18304v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18304v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Output Discipline Sets the Floor: A Falsification-Guided, Model-Agnostic Agent Protocol for Auditable Hypothesis-Driven Decision Making",
      "authors": [
        "Jing Liu",
        "Yongxing Qi",
        "Ping Yao",
        "Ting Wu"
      ],
      "abstract": "What binds large language models on auditable, hypothesis-driven decision tasks, where every candidate can be checked against the evidence? On the public TCMEval-SDT benchmark for Traditional Chinese Medicine syndrome differentiation (200 cases), the binding constraint is output discipline and reasoning structure, not model capability alone: two years of model scaling move zero-shot scores by only -0.03 to +0.08, while output-discipline failures (non-terminating reasoning, format non-compliance, large run-to-run spread) persist even in frontier models. We test this with a model-agnostic agent framework formalizing decision making as a five-stage closed loop, enforced by a candidate x evidence matrix falsification protocol: every candidate is judged against every evidence item, at least half of the candidates must be falsified before a conclusion is licensed, and survivors are emitted in a machine-verifiable single-line format. Across five base models, the effect is sign-conditional on zero-shot output discipline: it rescues models whose discipline fails (DeepSeek-V4-Flash: +0.267/+0.271, entirely termination rescue), reduces a high-variance model's run-to-run standard deviation from 0.218 to 0.007 (Kimi K3), and significantly costs already-disciplined models (-0.087 to -0.120 on budget-matched Qwen3-4B/32B). Ablations attribute the cost to the candidate-bounded format layer and show the effect is independent of injected knowledge content (protocol-knowledge decoupling). The protocol also transfers to misinformation-robust mathematical reasoning, detecting injected false premises at 89.8% recall with a 6.8% false-positive rate (MCC 0.54; a flag-everything baseline scores 0) and restoring accuracy to the no-misinformation ceiling. Where zero-shot output discipline is weak, the reasoning protocol sets the floor; base-model capability always sets the ceiling.",
      "published": "2026-10-13T12:46:44Z",
      "abstract_url": "http://arxiv.org/abs/2610.18298v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18298v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Which Optimizer Leaks Most? Optimizer Implicit Bias as a Training-Data Privacy Controller",
      "authors": [
        "Sagi Habani",
        "Yehonathan Refael",
        "Ofir Lindenbaum"
      ],
      "abstract": "Releasing a neural network checkpoint can leak information about its training data, and existing reconstruction results rely on gradient-descent training. We ask whether the choice of optimizer affects the ability to recover the training data. Extending work connecting Karush--Kuhn--Tucker conditions to training-data reconstruction, we show that the implicit biases of popular optimizers (SGD, Adam, Muon) leave very different fingerprints of the training data. We argue that the key distinction is the \\emph{weight-specificity} of these implicit biases, namely how much continuous information about the final weights the optimizer-dependent target of the stationarity conditions encodes. The limiting stationarity conditions for models trained with SGD and Muon retain enough weight-specific information to reconstruct training examples in practice. For Adam in the $\\ell_\\infty$ implicit-bias regime, the optimizer-dependent part of these conditions encodes only a discrete pattern of maximal coordinates and their signs, and at nondegenerate KKT points the conditions are too few to identify the training set to first order. Experiments on MNIST and CIFAR-10 support these claims: models trained with Muon can yield stronger reconstructions at much lower training cost than models trained with SGD, although this advantage depends on the training regime and architecture, and models trained with Adam appear consistently more resistant to the evaluated attacks.",
      "published": "2026-10-13T12:42:00Z",
      "abstract_url": "http://arxiv.org/abs/2610.18294v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18294v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "LiFUSE: Frequency-Spatial Unified Self-Supervised Pretraining for Liver Disease Analysis on Non-contrast CT",
      "authors": [
        "Hao Zhang",
        "Yi Huang",
        "Wei Liu",
        "Yuan Gao",
        "Zeju Li",
        "Xu Han",
        "Yu Shi",
        "Ling Zhang",
        "Ke Yan"
      ],
      "abstract": "Given the growing global burden of liver tumors and diffuse liver diseases, non-contrast computed tomography (NCCT) is increasingly used for opportunistic screening because of its low cost and favorable safety profile. However, downstream tasks such as tumor detection and fine-grained disease diagnosis remain challenging due to NCCT's inherently low tissue contrast, lack of hemodynamic enhancement, and complex anatomical background. These limitations make subtle, heterogeneous, and multifocal hepatic lesions difficult to identify. Although large-scale self-supervised pretraining can learn robust representations from unlabeled NCCT scans, conventional voxel-wise reconstruction does not fully exploit the complementary information available in NCCT. Under weak NCCT contrast, low-level intensity patterns become less informative for pathology recognition, limiting the capture of lesion-specific discriminative features. In this paper, we propose the Liver Frequency-Spatial Unified Self-Supervised Pretraining (LiFUSE). The frequency branch applies random band-stop masking and spectrum reconstruction to encourage the model to capture subtle differences between lesions and surrounding tissues, while the spatial branch performs hybrid-granularity masked volumetric reconstruction to accommodate lesions of diverse sizes and morphologies. Finally, a cross-domain consistency scheme couples the spatial and frequency branches, yielding more discriminative representations for hepatic structures and lesions. Extensive experiments on three clinically relevant tasks across internal cohorts and a public target-domain transfer setting show that LiFUSE consistently outperforms strong baselines. These results suggest that unified frequency-spatial learning can improve liver disease analysis on NCCT, highlighting its potential for large-scale screening.",
      "published": "2026-10-13T12:08:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.18243v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18243v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Beyond Answer Sharing: TrueMAD for Targeted Peer Engagement in Multi-Agent Debate",
      "authors": [
        "Dezheng Bao",
        "Sihang Du",
        "Renhong Huang",
        "Zhizhang Yuan",
        "Yifei Sun",
        "Yang Yang"
      ],
      "abstract": "Multi-agent debate (MAD) aims to improve large language model reasoning by exposing agents to alternative solutions and enabling iterative revision. However, standard broadcast-style MAD makes peer answers visible without requiring agents to engage with a particular peer claim. As a result, reliable reasoning may be suppressed by conflicting or majority responses, limiting achievable performance, while erroneous reasoning may persist and degrade final predictions without actionable feedback. We propose TrueMAD, a backbone-agnostic test-time plug-in that introduces one-to-one, free-form peer advice between answer broadcasting and self-revision. Each source agent responds to a target's current reasoning through support, rebuttal, refinement, or further questions, while the target retains control over how to revise. The same mechanism can augment standard MAD and its improved variants without parameter updates. Across AIME, GPQA-Diamond, and TruthfulQA with Qwen3-30B-A3B, TrueMAD improves agent-level and strict-majority accuracy over MAD by 3.62 and 3.52 percentage points on average, while TrueDMAD improves DMAD by 4.72 and 4.06 points, respectively. Analyses at the utterance, agent-transition, and group levels further show more peer-specific deliberation, more correction-oriented revisions, and more favorable consensus dynamics.",
      "published": "2026-10-13T12:08:38Z",
      "abstract_url": "http://arxiv.org/abs/2610.18240v1",
      "pdf_url": "https://arxiv.org/pdf/2610.18240v1",
      "categories": [
        "cs.AI"
      ]
    }
  ]
};
