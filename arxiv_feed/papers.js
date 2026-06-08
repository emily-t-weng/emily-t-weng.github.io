const PAPERS_DATA = {
  "last_updated": "2026-06-08 04:50:19 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "How reliable are LLMs when it comes to playing dice?",
      "authors": [
        "Luca Avena",
        "Gianmarco Bet",
        "Bernardo Busoni"
      ],
      "abstract": "We investigate the probabilistic reasoning capabilities of large language models through a controlled benchmarking study on discrete probability problems. We constructed two datasets, respectively a set of standard exercises and a set of counterintuitive exercises, designed to trigger heuristic reasoning, and evaluated 8 state-of-the-art models, each tested with and without Chain-of-Thought prompting. Models achieve an average accuracy of 0.96 on standard problems but only 0.59 on counterintuitive ones. We further provide empirical evidence of token bias: performance drops by over 20% when canonical formulations are replaced by disguised variants. Embedding misleading suggestions in the prompt reduces performance by up to 34%, with no model proving immune. Taken together, the reported findings suggest that current LLMs are not yet genuine probabilistic reasoners, despite their success in advanced mathematical problems.",
      "published": "2026-06-05T17:59:42Z",
      "abstract_url": "http://arxiv.org/abs/2606.07515v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07515v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.HC",
        "math.PR"
      ]
    },
    {
      "title": "Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning",
      "authors": [
        "Fatema Siddika",
        "Md Anwar Hossen",
        "Tanwi Mallick",
        "Ali Jannesari"
      ],
      "abstract": "Continual learning in Large Language Models (LLMs) is hindered by the plasticity-stability dilemma, where acquiring new capabilities often leads to catastrophic forgetting of previous knowledge. Existing methods typically treat parameters uniformly, failing to distinguish between specific task knowledge and shared capabilities. We introduce Mixture of Sparse Experts for Task Agnostic Continual Learning (SETA), a framework that resolves the plasticity-stability conflict through adaptive sparse subspace decomposition into task-specific expert modules. Unlike standard updates, where tasks compete for the same parameters, SETA separates knowledge into unique experts, designed to isolate task-specific patterns, and shared experts, responsible for capturing common features. This structure is maintained through adaptive elastic anchoring and a routing-aware regularization that jointly protect shared knowledge at both the weight and routing levels and enable a unified gating network to automatically retrieve the correct expert combination during inference. Extensive experiments across diverse domain-specific benchmarks demonstrate that SETA achieves competitive or superior overall performance relative to state-of-the-art continual learning baselines, with particularly strong retention of early-task knowledge and improved backward transfer on LLaMA-2 7B and Qwen3-4B.",
      "published": "2026-06-05T17:53:52Z",
      "abstract_url": "http://arxiv.org/abs/2606.07500v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07500v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Twelve quick tips for designing AI-driven HPC workflows",
      "authors": [
        "Jamie J. Alnasir"
      ],
      "abstract": "High-performance computing (HPC) clusters remain the backbone of large-scale scientific computation, traditionally executing deterministic, linear pipelines optimised for predictable performance. However, the pervasive integration of artificial intelligence (AI) and foundation models into scientific research has introduced a fundamentally new computational paradigm. AI-driven workflows are characteristically iterative, data-driven, and probabilistic, introducing unique challenges regarding data gravity, heterogeneous resource management, and complex workflow orchestration. This guide provides twelve practical tips designed to help researchers design efficient, scalable, and reproducible AI-driven HPC workflows. By addressing critical system-level bottlenecks - such as containerisation for environment portability, strategic deployment of job arrays, explicit feedback loop mechanics, and I/O optimisation for small files - this article offers a framework for transitioning from rigid execution pipelines to adaptive, intelligent computational environments. While these architectural principles are broadly applicable across distributed environments, they are particularly tailored to the resource-intensive throughput demands of modern computational biology.",
      "published": "2026-06-05T17:46:32Z",
      "abstract_url": "http://arxiv.org/abs/2606.07491v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07491v1",
      "categories": [
        "cs.DC",
        "cs.AI",
        "cs.LG",
        "cs.SE"
      ]
    },
    {
      "title": "Graph Neural Network leveraging Higher-order Class Label Connectivity for Heterophilous Graphs",
      "authors": [
        "Takuto Takahashi",
        "Itsuki Nakayama",
        "Takahiro Mitani",
        "Ryosuke Kikuchi",
        "Yuya Sasaki",
        "Makoto Onizuka"
      ],
      "abstract": "Node classification in graph neural networks (GNNs) has been widely applied in various fields of graph analysis. GNNs achieve high-accuracy node classification in homophilous graphs, where nodes with the same class label tend to be connected. However, their performance remains limited in heterophilous graphs, where nodes with different class labels are more likely to be connected. In particular, current GNNs derived from graph convolutional networks cannot capture higher-order class label connectivity, which is frequently observed in real-world heterophilous graphs. To address this issue, we propose a novel classifier, Label Context Classifier (LCC), designed to capture higher-order class label connectivity in directed graphs. LCC estimates the class label of a target node by leveraging label context embeddings that are generated through four distinct types of walks. In addition, our approach allows the integration of LCC and any GNN by adaptively learning their importance. Experimental results demonstrate that GNNs integrated with LCC outperform SOTA methods and the label context embeddings improve the node classification performance in heterophilous directed graphs.",
      "published": "2026-06-05T17:28:19Z",
      "abstract_url": "http://arxiv.org/abs/2606.07475v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07475v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment",
      "authors": [
        "Sweta Mahajan",
        "Sukrut Rao",
        "Jiahao Xie",
        "Alexander Koller",
        "Bernt Schiele"
      ],
      "abstract": "Vision-language models such as CLIP are highly useful for diverse tasks due to their shared image-text embedding space. Despite this, the image and text embeddings are often poorly aligned, affecting downstream performance. Recent work has shown that this can be attributed to an information imbalance: images contain more information than their captions describe. In this work, we propose TEVI, a framework that uses captions as a signal for what to retain from image embeddings. Specifically, we use sparse autoencoders to disentangle image embeddings and train a masking module to selectively reconstruct the embedding based on a given caption. In a controlled setup with synthetic captions, we show that TEVI is effective at preserving caption-described attributes while discarding others. By applying TEVI to CLIP models trained on natural images, we further achieve improved retrieval performance across coarse-grained short-caption (MS COCO, Flickr) and fine-grained long-caption (IIW, DOCCI) benchmarks, with stronger gains on richer captions, and improved robustness on the RoCOCO benchmark.",
      "published": "2026-06-05T16:54:40Z",
      "abstract_url": "http://arxiv.org/abs/2606.07451v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07451v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "Watch, Remember, Reason: Human-View Video Understanding with MLLMs",
      "authors": [
        "Jiahao Meng",
        "Yue Tan",
        "Qi Xu",
        "Kuan Gao",
        "Weisong Liu",
        "Yanwei Li",
        "Jason Li",
        "Lingdong Kong",
        "Haochen Wang",
        "Qianyu Zhou",
        "Jiangning Zhang",
        "Guangliang Cheng",
        "Yunhai Tong",
        "Lu Qi",
        "Minghsuan Yang"
      ],
      "abstract": "Video understanding is being rapidly transformed by multimodal large language models (MLLMs), as research moves from short clips to long, multimodal, and knowledge-intensive video scenarios. These scenarios require models to handle sparse evidence, long-range dependencies, multimodal alignment, and reliable inference under limited computational budgets. This work presents a human-view perspective on LLM-based video understanding, organized around three functional abilities: watching, remembering, and reasoning. Rather than treating video tasks as isolated benchmarks, this view provides a unified structure for analyzing how video MLLMs acquire evidence, preserve context, and produce grounded outputs. We introduce a formulation that characterizes video understanding systems by their perceptual representations, memory states, reasoning traces, and final predictions. Based on this formulation, we identify challenges in spatio-temporal perception, efficient long-video processing, memory modeling, streaming understanding, and faithful reasoning. Representative methods are organized by their roles in video MLLM systems. Watching covers fine-grained, comprehensive, audio-visual, and efficient perception. Remembering includes offline and streaming memory, while reasoning covers text-only reasoning and thinking with videos. We further examine application domains such as egocentric, sports, instructional, medical, and narrative videos, and cover training datasets and evaluation benchmarks across task types, supervision formats, modalities, and capability dimensions. Finally, we outline open problems and future directions for scalable, memory-aware, and evidence-grounded video intelligence. Related works will be continuously traced at https://github.com/marinero4972/Awesome-HumanView-VideoUnderstanding.",
      "published": "2026-06-05T16:29:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.07433v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07433v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.MM"
      ]
    },
    {
      "title": "The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs",
      "authors": [
        "Yang Zhang",
        "Xiao Fei",
        "Amr Mohamed",
        "Sarah Almeida Carneiro",
        "Mersin Konomi",
        "Mingmeng Geng",
        "Ahmed Asaad",
        "Guokan Shang",
        "Michalis Vazirgiannis"
      ],
      "abstract": "Large language models are increasingly used to answer culturally grounded questions across languages, yet it remains unclear whether local cultural knowledge is better accessed through English or the local language. Existing evaluations face two key limitations: many rely on parallel template-based questions that may not reflect how cultural knowledge naturally appears, and raw accuracy conflates general language proficiency with language-conditioned knowledge access. We address these issues with a controlled framework built on real-world cultural questions collected from regional benchmarks and local sources. By crossing question type (culture-agnostic vs. culture-specific) with query language (English vs. local language), and estimating ability with a shared 1PL item response theory model, we separate proficiency from localized knowledge access. Across 13 locales and roughly 80 models, we find a consistent English advantage on culture-agnostic questions, indicating stronger English proficiency. However, after accounting for this proficiency gap, local languages show a positive knowledge-access advantage in nearly all locale-model settings. This advantage is often masked in raw accuracy but becomes more visible for frontier, regionally aligned, or language-adapted models. Our results suggest that weaker local-language performance does not necessarily imply weaker cultural knowledge; rather, local cultural knowledge may be more accessible through the local language but hidden by limited language proficiency.",
      "published": "2026-06-05T16:16:59Z",
      "abstract_url": "http://arxiv.org/abs/2606.07422v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07422v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "A Comprehensive Anatomy of Human and DeepSeek-R1 LLM Mathematical Reasoning",
      "authors": [
        "Yuxiang Chen",
        "Jun Wang"
      ],
      "abstract": "The emergence of \"Aha moments\" in large language models, particularly DeepSeek-R1-0120, has raised the question of whether these systems genuinely reason or merely imitate the appearance of reasoning. We conduct a comprehensive empirical comparison between model and human reasoning across all 30 problems from AIME 2025, exhaustively annotating 10,247 reasoning steps into five functional categories: Analysis, Inference, Branch, Backtrace, and Reflection. We find a clear structural difference. Human solutions maintain a compact alternation between analysis and deduction, whereas DeepSeek-R1 frequently revisits intermediate results, performs shallow and often unnecessary verification, and loops through local checks without meaningful logical progress. We describe this as topological mimicry: reproducing the surface form of reasoning without its functional role. Despite this, we identify two signals of genuine reasoning. First, successful traces exhibit stable use of branching and backtracking, while failed traces either underuse or overuse exploratory actions. Second, reflection is only effective when placed within deductive inference; reflections trapped in analysis loops focus on local numerical details while missing global logical errors. These findings suggest that current long-CoT models may be rewarded more for the appearance of reasoning than for genuine deductive progress. We discuss directions for improving evaluation and training, including measuring cross-trace stability, penalising \"spinning-wheel\" traces, encouraging deeper logical correction, and reallocating inference-time compute toward deduction and backtracking. Overall, reasoning quality depends not simply on how much reflection occurs, but on whether reflection appears consistently and at the appropriate logical scale.",
      "published": "2026-06-05T15:57:42Z",
      "abstract_url": "http://arxiv.org/abs/2606.07410v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07410v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Online Pandora's Box for Contextual LLM Cascading",
      "authors": [
        "Alexandre Belloni",
        "Yan Chen",
        "Yehua Wei"
      ],
      "abstract": "Motivated by Large Language Model (LLM) cascading, we propose an online contextual Pandora's Box model for adaptively querying and selecting LLM APIs. In each period, a decision-maker observes a request context and faces a two-phase decision problem. In the query phase, the decision-maker sequentially queries APIs, where each query reveals a generated output and the decision-maker incurs an (output-dependent) cost. In the selection phase, the decision-maker selects one of the generated outputs to deploy and observes only the downstream reward of the deployed output. This output-mediated feedback structure differs from classical online contextual Pandora's Box models, in which opening a box directly reveals its reward. Rather than estimating the full conditional output and cost distributions of each API, we directly model the reservation index and develop a learning approach for the query phase. Specifically, we impose a parametric structure on the contextual reservation index functions induced by the classical Weitzman's policy. Our policy combines generalized method of moments (GMM) type estimation of these reservation indices with UCB-style confidence bounds for both these indices and the shared output-level reward evaluator. Under regularity conditions, we prove that the resulting policy achieves dimension-dependent $\\widetilde O(\\sqrt T)$ cumulative regret over a horizon of $T$ periods.",
      "published": "2026-06-05T15:29:17Z",
      "abstract_url": "http://arxiv.org/abs/2606.07392v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07392v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "econ.EM",
        "stat.ML"
      ]
    },
    {
      "title": "Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests",
      "authors": [
        "Thanawat Lodkaew",
        "Johannes Ackermann",
        "Soichiro Nishimori",
        "Nontawat Charoenphakdee",
        "Masashi Sugiyama",
        "Takashi Ishida"
      ],
      "abstract": "A growing failure mode in agent evaluation and training is that models can achieve high evaluation scores by exploiting shortcuts instead of solving the intended task, producing deceptive performance. This makes evaluation scores unreliable as measures of true task-solving ability. We propose CapCode, a framework for constructing coding datasets with randomized tests whose best achievable non-cheating performance is deliberately capped below one. This capped-performance design gives evaluation scores a clearer interpretation: scores substantially above the cap are implausible and therefore provide evidence of cheating. To prevent cheating, we propose CapReward, a reward design based on the CapCode principle to discourage optimization beyond the cap. Experiments across multiple datasets show that CapCode detects cheating while preserving performance ranking of models, and CapReward reduces cheating behavior, yielding models that better follow the intended task specification.",
      "published": "2026-06-05T15:20:37Z",
      "abstract_url": "http://arxiv.org/abs/2606.07379v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07379v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL",
        "stat.ME"
      ]
    },
    {
      "title": "A robust PPG foundation model using multimodal physiological supervision",
      "authors": [
        "Eloy Geenjaar",
        "Vince Calhoun",
        "Scott Daly",
        "Gouthaman KV",
        "Lie Lu",
        "Trisha Mittal",
        "Daniel P. Darcy"
      ],
      "abstract": "Photoplethysmography (PPG), a non-invasive measure of changes in blood volume, is widely used in both wearable devices and clinical settings. Recent PPG foundation models either use open-source ICU datasets with pretraining paradigms that require curated data and thus complicate generalization to field-like data, or use closed-source field-like PPG data. In contrast, we propose a PPG foundation model that does not require high-quality or field-like pretraining data, and instead leverages accompanying electrocardiogram and respiratory signals in ICU datasets to select contrastive samples during pretraining. Our approach allows the model to retain and learn from noisy PPG segments, improving robustness at inference. Our model, pretrained on 3x fewer subjects than existing state-of-the-art approaches, achieves performance improvements on 14 out of 15 diverse downstream tasks, including field-like daily activity and heart rate prediction. Our results demonstrate that multimodal supervision can integrate complementary physiological information to improve the robustness of PPG foundation models and enhance their generalization to consumer-grade data.",
      "published": "2026-06-05T15:08:50Z",
      "abstract_url": "http://arxiv.org/abs/2606.07365v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07365v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SleepExplain: Explainable Non-Rapid Eye Movement and Rapid Eye Movement Sleep Stage Classification from EEG Signal",
      "authors": [
        "Rafsan Jany",
        "Md. Hamjajul Ashmafee",
        "Iqram Hussain",
        "Md Azam Hossain"
      ],
      "abstract": "Classification of sleep stages is one of the most important diagnostic approaches for a variety of sleep-related disorders. Electroencephalography (EEG) is regarded as a powerful tool for examining the association between neurological effects and sleep phases since it correctly identifies sleep-related neurological alterations. During Non-Rapid Eye Movement (NREM) and Rapid Eye Movement (REM) sleep phases, a number of nerve and bodily functions are affected and therefore hold an important role both in their functionalities. This work aims to classify NREM and REM sleep stages from sleep EEG data and present a noble SleepExplain model, an explainable NREM and REM sleep stage classification to explain its predictions. In this work, sleep stages were classified using Random Forest, XGBoost, and Gradient Boosting ensemble classification models. Overall, we obtained an accuracy of 92.54% (Random Forest), 94.25% (Gradient Boosting), and 94.30% (XGBoost). For explainable classification model, we utilized a game theoretic approach, SHAP (SHapley Addictive exPlanations) to offer a convincing explanation for the prediction.",
      "published": "2026-06-05T15:00:33Z",
      "abstract_url": "http://arxiv.org/abs/2606.07351v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07351v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration",
      "authors": [
        "Haoran Xu",
        "Lei Zhang",
        "Iadh Ounis",
        "Xianbin Wang"
      ],
      "abstract": "Byzantine collaboration among large-language-model agents requires a finality-control primitive: given delivered stochastic, structured natural-language proposals, the protocol must decide whether the round supports a commit, what kind of commit, or a typed safe abort. Naive aggregation hides this choice behind a single verdict; classical Byzantine fault tolerance hides it behind byte-identity that LLM proposals do not satisfy. We introduce Hierarchical Certified Semantic Commitment (H-CSC), a BFT-inspired protocol that converts embedding-derived finality signals over verdict-conditioned proposal groups into one of three typed outcomes: a semantic_commit (a 2f+1 within-verdict semantic core backs the verdict, emitting a parameter-bound digest over the quantised aggregate), a verdict_commit (strong verdict margin but dispersed semantic rationale, emitting a verdict-level certificate without claiming a semantic aggregate), or an explicit abort with a typed reason. The contribution is typed finality, not raw commit accuracy. On a controlled semantic-poisoning diagnostic (BCS_v1, 120 episodes), H-CSC commits with low angular deviation on BFT-feasible buckets (0.31 to 2.04 degrees) and aborts 100% of beyond-BFT rounds (n<3f+1) as intended. On a real LLM-agent claim-verification benchmark (MVR-50, 50 tasks) under paired static and rushing Byzantine attacks, H-CSC commits 0.90/0.92 with honest-reference-invalid rates of 0.02/0.00, statistically matching a strong certificate-emitting verdict-only baseline. Unlike that baseline, H-CSC also emits an embedding-backed semantic_commit digest on 74%/72% of rounds, supplying typed provenance. A strict-semantic ablation commits only 0.54/0.48, showing the verdict-level fallback is necessary for coverage (+0.36/+0.44) at the same <=0.04 safety floor; a 100-task cross-model check across four LLMs preserves invalid_hmaj within 0.00 to 0.03.",
      "published": "2026-06-05T14:35:58Z",
      "abstract_url": "http://arxiv.org/abs/2606.07316v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07316v1",
      "categories": [
        "cs.MA",
        "cs.AI",
        "cs.DC"
      ]
    },
    {
      "title": "Where Rectified Flows Leak: Characterising Membership Signals Along the Interpolation Path",
      "authors": [
        "Thomas Sesmat",
        "Gabriel Meseguer-Brocal",
        "Geoffroy Peeters"
      ],
      "abstract": "Understanding what generative models retain from training data remains challenging, with implications for copyright and privacy. Beyond verbatim reproduction, models can encode subtler traces of their training data that never surface in their outputs yet remain exploitable. We study this regime for Rectified Flows, which are increasingly used in deployed generative systems. We analyse the interpolation path $X_λ= (1-λ)X_0 + λX_1$ that defines the Rectified Flow training. We show that a gap exists between the reconstruction of train and test data that follows a bell-shaped curve over $λ$, wich accumulates during training, while the validation metrics remain stable. The signal has a maximum whose location we derive in closed form under Gaussian assumptions. We validate these predictions on both audio and images and show that the bell-shaped structure is universal, while the peak prediction holds when our assumptions are satisfied. As a proof of concept, we exploit this specific $λ$-resolved structure to perform a Membership Inference Attack, distinguishing members of the training set from non-members.",
      "published": "2026-06-05T13:46:37Z",
      "abstract_url": "http://arxiv.org/abs/2606.07271v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07271v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.SD"
      ]
    },
    {
      "title": "When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations",
      "authors": [
        "Mahdi Alkaeed"
      ],
      "abstract": "Large Language Models (LLMs) are increasingly used in healthcare for tasks such as clinical question answering, diagnosis support, and report summarization. Despite their promise, these models remain highly sensitive to subtle prompt perturbations, both lexical and syntactic, posing serious risks in safety-critical clinical applications. In this study, we conduct a systematic sensitivity analysis to evaluate the robustness of both general-purpose (e.g., GPT-3.5, Llama3) and medical-specific LLMs (e.g., ClinicalBERT, BioLlama3, BioBERT) using the MedMCQA benchmark. We categorize perturbations into natural and adversarial types and examine their effect on model consistency, accuracy, and reliability in clinical reasoning tasks. Our findings reveal that medical LLMs are not intrinsically safe. Even minor variations in phrasing can alter clinical advice, and targeted adversarial prompts can provoke harmful outputs. In high-stakes settings like healthcare, such unpredictability is unacceptable-models that change diagnoses due to reworded inputs or hallucinate medications when slightly rephrased cannot be reliably trusted by clinicians. While models tend to show resilience to simple lexical substitutions or paraphrasing, they often break down under syntactic reordering or misleading contextual cues. This fragility is evident across both general-purpose and domain-specific LLMs. Notably, adversarial manipulations can lead to clinically dangerous outputs, such as recommending incorrect dosages or omitting critical findings.",
      "published": "2026-06-05T13:07:11Z",
      "abstract_url": "http://arxiv.org/abs/2606.07237v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07237v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "DEFINED: A Data-Efficient Computational Framework for Fine-Grained Creativity Assessment in Debate Scenarios",
      "authors": [
        "Tongzhou Yu",
        "Mingjia Li",
        "Hong Qian",
        "Wenkai Wang",
        "Zongbao Zhang",
        "Yaoyu Jiang",
        "Xiangfeng Wang",
        "Aimin Zhou",
        "Jiajun Guo"
      ],
      "abstract": "Human creativity has emerged as a critical competency in the era of large language models. Assessing creativity in complex, open-ended environments is a grand challenge in data mining, currently hindered by a reliance on standardized simple tasks and the scarcity of fine-grained expert data. As an ecologically valid assessment context, debate reflects multiple dimensions of creativity, encompassing both divergent thinking and convergent thinking. Moreover, debate is a data-rich domain, with a large volume of publicly accessible materials. Current mainstream automated scoring methods are poorly suited to complex settings such as debate, and therefore still rely on costly human evaluation. To this end, this paper proposes DEFINED, a data-efficient computational framework for fine-grained creativity assessment in debate scenarios. DEFINED operationalizes debate creativity through a hierarchical eight-dimensional metric system, implemented via a pre-trained autoregressive language model with a hierarchical scoring head that supports both fine-grained and coarse-grained evaluation. Statements and their associated expert scores were obtained from authentic debate competitions, and a constrained data augmentation strategy was employed to address the elite bias inherent in the original data. DEFINED adopts a mixed-granularity training strategy enabling robust learning from limited fine-grained supervision annotated by trained graduate experts. To rigorously validate ecological validity beyond synthetic benchmarks, we incorporate an empirical study with debate-naive participants, utilizing these authentic data to serve as a qualitative case study for mid-to-low proficiency populations. Across our evaluation protocol, our scoring model achieves accurate and stable scoring, outperforming prompt-based large language model evaluators and existing debate scoring methods.",
      "published": "2026-06-05T12:42:56Z",
      "abstract_url": "http://arxiv.org/abs/2606.07226v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07226v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "RETROSPECT: RETROsynthesis via Sequential Prediction, and Chemically Transformed-ranking",
      "authors": [
        "Raja Sekhar Pappala",
        "Shreyas Vinaya Sathyanarayana",
        "Ronit Kumar Choudhary",
        "Arjun Verma",
        "Deepak Warrier"
      ],
      "abstract": "Single-step retrosynthesis needs both accurate first-ranked suggestions and candidate lists that are rich enough for downstream selection. We study this as a proposal-selection decomposition. Our system, RETROSPECT, combines a single Transformer proposal model, which we call the ChemAlign Transformer, with a LambdaMART reranker over structural, reaction-template, upstream-score, and optional DFT-derived descriptors. The generator is trained with hybrid root-aligned and random SMILES augmentation, Pre-LayerNorm, tied embeddings, exponential moving average weights, and a differentiable atom-balance auxiliary loss. On the full USPTO-50K test set of 5,007 reactions, the generator reaches 55.00% top-1 and 86.18% top-10 exact-match accuracy with 99.86% top-1 validity. On the merged candidate-pool benchmark used for reranking, which contains 5,007 test products and about 111 candidates per product, a LambdaMART model trained on the structural feature set reaches 59.4% top-1 with 0.7171 mean reciprocal rank. Feature ablations show that upstream proposal score and template-frequency statistics provide most of the reranking signal, while DFT and reaction-center DFT features provide smaller and less consistent gains. These results support a modular view of retrosynthesis: stronger single-model proposal and learned candidate selection are complementary, and the proposal model can serve as a drop-in component for ensemble systems such as RetroChimera (Maziarz et al., 2024)",
      "published": "2026-06-05T11:45:36Z",
      "abstract_url": "http://arxiv.org/abs/2606.07181v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07181v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.MN"
      ]
    },
    {
      "title": "Textual Supervision Enhances Geospatial Representations in Vision-Language Models",
      "authors": [
        "Marcelo Sartori Locatelli",
        "Fernando Tonucci",
        "Jea Kwon",
        "Luiz Felipe Vecchietti",
        "Bryan Nathanael Wijaya",
        "Cheng Yaw Low",
        "Virgilio Almeida",
        "Meeyoung Cha"
      ],
      "abstract": "Geospatial understanding is a critical yet underexplored dimension in the development of machine learning systems for tasks such as image geolocation and spatial reasoning. In this work, we analyze the geospatial representations acquired by three model families: vision-only architectures (e.g., ViT), vision-language models (e.g., CLIP), and large-scale multimodal foundation models (e.g., LLaVA, Qwen, and Gemma). By evaluating across image clusters, including people, landmarks, and everyday objects, grouped based on the degree of localizability, we reveal systematic gaps in spatial accuracy and show that textual supervision enhances the learning of geospatial representations. Our findings suggest the role of language as an effective complementary modality for encoding spatial context and multimodal learning as a key direction for advancing geospatial AI.",
      "published": "2026-06-05T11:40:13Z",
      "abstract_url": "http://arxiv.org/abs/2606.07172v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07172v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "REMEDI: A Benchmark for Retention and Unlearning Evaluation in Multi-label Clinical Disease Inference",
      "authors": [
        "Anurag Sharma",
        "Sai Teja Chunchu",
        "Prasenjit Mitra",
        "Sandipan Sikdar",
        "Koustav Rudra"
      ],
      "abstract": "Language models trained for clinical disease inference are trained on patient data, which may include sensitive and private information, and data owners may request the removal of their data from a trained model due to privacy or copyright concerns. However, exactly unlearning patient-specific data is intractable, and retraining with minor data removal is resource-intensive. While there exists several machine unlearning methods that can be used, their utility is generally restricted to non-medical domains. Moreover, the existing benchmarks for evaluating such unlearning methods primarily utilize synthetically curated datasets, which are not truly representative of real-world systems. Hence, the effectiveness of these unlearning methods in the medical domain is largely unclear. To this end, we introduce REMEDI, an extensive benchmark for machine unlearning tailored to multi-label and multiclass clinical disease inference, where label correlations, longitudinal structure, and safety constraints make unlearning particularly challenging. Unlike the existing benchmarks, REMEDI considers: (1) a relevant application domain (medical), (2) comprehensive unlearning setups involving diverse sets of forget instances, (3) challenging unlearning scenarios including multi-label and multi-class classification tasks, and (4) evaluation metrics involving performance both in terms of utility and extent of unlearning achieved. REMEDI is developed using the MIMIC-III clinical database that contains comprehensive clinical data of patients. Experiments with existing unlearning methods indicate that there exists a trade-off between utility and unlearning performance. They are also largely unsuited to multi-label classification tasks. To facilitate reproducibility, we make our benchmark publicly available.",
      "published": "2026-06-05T10:51:20Z",
      "abstract_url": "http://arxiv.org/abs/2606.07141v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07141v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "OffQ: Taming Structured Outliers in LLM Quantization by Offsetting",
      "authors": [
        "Haoqi Wang",
        "Lorenz K. Mueller",
        "Jiawei Zhuang",
        "Mathieu Salzmann",
        "Lukas Cavigelli"
      ],
      "abstract": "Low-bit quantization has been widely adopted to accelerate the inference of large language models (LLMs) by significantly reducing computational cost and memory usage. However, activation outliers pose a major challenge to effective quantization, often leading to notable performance degradation. In this paper, we introduce OffQ, a method designed to mitigate activation outliers in low-bit quantization through a novel offsetting mechanism. Specifically, OffQ first identifies a low-dimensional outlier subspace in the activations using a proposed top-1 PCA, and then concentrates high-magnitude activations into 1 channel via rotation. OffQ then absorbs this concentrated outlier channel by converting its magnitude into a shared offset, thereby reducing the standard deviation of the activations. This offsetting strategy enables effective W4A4KV4 quantization of LLMs using deployment-friendly uniform-grid and uniform-precision quantization. Extensive experiments across diverse LLM architectures and benchmarks demonstrate that OffQ outperforms state-of-the-art baselines, consistently improving model accuracy while preserving low-bit efficiency.",
      "published": "2026-06-05T10:11:34Z",
      "abstract_url": "http://arxiv.org/abs/2606.07116v1",
      "pdf_url": "https://arxiv.org/pdf/2606.07116v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    }
  ]
};
