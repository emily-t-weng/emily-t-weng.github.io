const PAPERS_DATA = {
  "last_updated": "2026-09-01 04:45:59 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "OntoAligner-Ensemble: Voting-Based Fusion across Heterogeneous Ontology Alignment Techniques",
      "authors": [
        "Hamed Babaei Giglou",
        "Sören Auer",
        "Peio Popov",
        "Mahsa Sanaei",
        "Jennifer D'Souza"
      ],
      "abstract": "Ontology alignment (OA) has evolved through several methodological paradigms, ranging from lexical and structural aligners to knowledge graph embedding (KGE) models and, more recently, Large Language Model (LLM)-based approaches. Although modern OA frameworks provide unified ecosystems for deploying these heterogeneous aligners, mechanisms for systematically reconciling their complementary and sometimes conflicting predictions remain relatively underexplored. We present OntoAligner-Ensemble, a modular and aligner-agnostic framework that combines candidate correspondences through a configurable two-stage process comprising voting-based fusion strategies followed by post-fusion selection policies. The framework supports any aligner implemented within OntoAligner that produces candidate correspondences, enabling diverse alignment paradigms to be integrated through a unified decision process. To demonstrate its effectiveness, we instantiate the framework using representative lightweight string-aligner, KGE-based, and Retrieval-Augmented Generation aligners powered by both open-weight and API-based LLMs. We evaluate individual aligners and ensemble configurations across eight benchmark tasks from five OAEI tracks spanning biomedical to beyond-equivalence. The results show that ensemble fusion consistently improves the balance between precision and recall and frequently outperforms standalone aligners across diverse domains. Furthermore, our analysis reveals that ensemble composition directly affects the precision-recall trade-off: heterogeneous cross-paradigm ensembles generally improve precision, whereas homogeneous LLM ensembles more often achieve higher overall F1-scores. These findings demonstrate that systematic ensemble learning offers a robust and reproducible strategy for OA while providing practical guidance for selecting ensemble compositions under different alignment scenarios.",
      "published": "2026-08-31T17:44:25Z",
      "abstract_url": "http://arxiv.org/abs/2608.31137v1",
      "pdf_url": "https://arxiv.org/pdf/2608.31137v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "When Does Bigger Help? A Controlled Study of LLM Scale for Ontology Learning",
      "authors": [
        "Hamed Babaei Giglou",
        "Sören Auer",
        "Jennifer D'Souza"
      ],
      "abstract": "The effect of Large Language Model (LLM) scale on ontology learning (OL) performance remains insufficiently characterized. We present a controlled evaluation of 13 models spanning dense and Mixture-of-Experts variants from the Qwen3.5 and Qwen3.6 lineages, together with proprietary GPT release variants, using the OntoLearner retrieval-augmented generation pipeline. All models are evaluated with the same embedding model, retrieval configuration, prompt templates, decoding settings, datasets, and metrics on term typing, taxonomy discovery, and non-taxonomic relationship extraction across four biomedical and materials science and engineering ontologies. Within the dense Qwen3.5 lineage, increasing parameter count primarily improves precision rather than recall, with the largest gains occurring between 9B and 27B parameters. However, the effect of scale is neither monotonic nor uniform across tasks and domains. Dense 27B models outperform substantially larger sparse models on term typing, whereas larger Mixture-of-Experts models achieve the strongest open-weight results on taxonomy discovery. Non-taxonomic relationship extraction remains difficult across model scales, particularly for the Materials Data Science ontology. Performance differences across matched Qwen variants and proprietary GPT releases further indicate that architecture and model lineage can outweigh nominal parameter count. These findings show that model size alone is an insufficient selection criterion for OL and provide empirical guidance for reproducible LLM-assisted ontology engineering.",
      "published": "2026-08-31T17:30:05Z",
      "abstract_url": "http://arxiv.org/abs/2608.31118v1",
      "pdf_url": "https://arxiv.org/pdf/2608.31118v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "LLM Post-Training as Brownfield Maintenance: An Industrial Perspective on Dataware Engineering",
      "authors": [
        "Gopi Krishnan Rajbahadur",
        "Amir M. Ebrahimi",
        "Boyuan Chen",
        "Ahmed E. Hassan"
      ],
      "abstract": "Industrial post-training is a brownfield regime. Teams inherit a deployed checkpoint and must land targeted improvements under fixed compute and mixture budgets without regressing the rest. The maintained artifact is increasingly dataware: behavior governed by a curated post-training mixture, updated via bounded mixture patches rather than clean-slate retraining. From an industrial code-generation improvement effort, we offer a maintainer's perspective on why this work is hard in practice, distilling three recurring challenges, zero-sum mixture design, yield as the binding metric, and end-to-end integration under uncertainty, and arguing that progress depends less on one-off recipes than on an engineering discipline for programming dataware. In our case study, interventions that raised the conversion of teacher distillation into usable training data increased accepted supervision by 2.84 times while using the same solution teacher and four solution attempts per candidate problem. In our primary evaluation, the yield-engineered patch improved CodeForces pass@1 by +2.59 points (+3.11 pass@3) and held-out LiveCodeBench v6 pass@1 by +6.11 (+8.05 pass@3), all statistically significant across 16 stochastic evaluations of each benchmark from one fixed checkpoint per condition, with internal AIME and MATH regression suites within tolerance.",
      "published": "2026-08-31T17:08:41Z",
      "abstract_url": "http://arxiv.org/abs/2608.31102v1",
      "pdf_url": "https://arxiv.org/pdf/2608.31102v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Learning to Evaluate Before Improving: Automatic Rubric Induction for Automatic Research Agents",
      "authors": [
        "Xuehai Wang",
        "Haowei Qin",
        "Tongxin Liu",
        "Junkai Li",
        "Buqiang Xu",
        "Jintian Zhang",
        "Yijun Chen",
        "Zirui Xue",
        "Shumin Deng"
      ],
      "abstract": "Autonomous scientific research agents are increasingly applied to end-to-end scientific workflows, including literature review, data analysis, experimentation, and report generation. However, open-ended research tasks often do not clearly specify the analyses, methods, and success criteria required to complete the task. As a result, agents may miss important analyses, use inappropriate methods, or draw conclusions that are insufficiently supported by evidence. To address the problem, we present AutoSciRub, an evaluation-first framework that induces a task-specific executable rubric before research execution, and uses it to guide execution, criterion-level verification as well as iterative revision. AutoSciRub decomposes an underspecified instruction into atomic scientific goals, grounds them in relevant literature and task-visible data, and synthesizes specific, actionable, and verifiable criteria. The resulting rubric makes implicit experimental and evidential requirements explicit, providing guidance for experiments and analyses. During revision, rubric-guided verification identifies unmet criteria and enables targeted refinement of the research report and its supporting artifacts. On ResearchClawBench, AutoSciRub consistently improves all tested configurations, with an average gain of 2.08 points across three backbone LLMs under the fixed Codex harness and 2.95 points across three agent harnesses using a fixed DeepSeek-V4-Flash backbone. On a randomly sampled 20-task subset of AstaBench E2E Discovery, AutoSciRub further achieves an average improvement of 16.8 points across three agent harnesses, while maintaining or increasing the number of successfully completed tasks. These results demonstrate that evaluation-first guidance provides an effective and generalizable control mechanism for autonomous scientific research (Code: https://github.com/zjunlp/AutoSciRub).",
      "published": "2026-08-31T16:48:51Z",
      "abstract_url": "http://arxiv.org/abs/2608.31076v1",
      "pdf_url": "https://arxiv.org/pdf/2608.31076v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.IR",
        "cs.LG",
        "cs.MA",
        "cs.SE"
      ]
    },
    {
      "title": "Wrong Prediction, Right Answer: Recovering Evidence from Collapsed LLM Sequence Scores",
      "authors": [
        "Qiyao Yan",
        "Chenpeng Wang",
        "Liangming Pan"
      ],
      "abstract": "When a large language model fails a reasoning task, it is often assumed to lack the underlying capability. However, this conflates a genuine absence of reasoning with a late-stage output bottleneck. We observe a consistent readout gap across diverse reasoning benchmarks: hidden-state probes successfully decode correct answers even when native sequence scoring completely collapses due to structural biases. To test whether instance-specific logic survives this collapse, we introduce a diagnostic protocol using a minimal, target-label-free additive correction. Fitting just two parameters on as few as 25 unlabeled examples recovers 9--34 accuracy points for Qwen3.5 models, transferring successfully to OLMo-2-1B and Llama-3.1-8B. Crucially, these recovered decisions persist on hard instances unresolved by simple lexical overlap and significantly exceed count-preserving permutation baselines. Our results show that many apparent zero-shot reasoning deficits are expression failures masking intact internal logic, urging a narrower interpretation of benchmark evaluations.",
      "published": "2026-08-31T16:43:55Z",
      "abstract_url": "http://arxiv.org/abs/2608.31068v1",
      "pdf_url": "https://arxiv.org/pdf/2608.31068v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Stick to What You Know: A Study of Knowledge-Aligned Supervised Fine-Tuning",
      "authors": [
        "Arthur Becker",
        "Jakob Kemmler",
        "David Thulke",
        "Christine Schäfer",
        "Christian Dugast",
        "Hermann Ney"
      ],
      "abstract": "Supervised fine-tuning (SFT) trains a base language model to imitate target responses, and these targets may require knowledge the base model has not robustly internalized. We study this as a source of hallucinations and frame a group of mitigation methods as \\emph{knowledge-aligned SFT}: constraining SFT training targets to the base model's parametric knowledge. Under a unified setup, we compare existing generation-based and estimation-based knowledge-alignment methods and introduce two new variants: Evidence Rewrite, which verifies base-model generations using external evidence, and Recall Rewrite, which retains claims only when they can be consistently recalled by the base model. Experiments with Qwen 3 4B and OLMo 3 7B show that knowledge-aligned SFT can reduce factual hallucinations on WildHalu and Biography while largely preserving general capabilities. Recall Rewrite yields the strongest factuality gains and improves refusal behavior on UnknownBench. It thereby confirms that SFT targets beyond the base model's knowledge drive hallucination behavior.",
      "published": "2026-08-31T15:43:51Z",
      "abstract_url": "http://arxiv.org/abs/2608.30987v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30987v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "CoJEPA: Combining Contrastive Learning and JEPA for Global-Local Music Representations",
      "authors": [
        "Gabriel Meseguer-Brocal",
        "Yuexuan Kong",
        "Romain Hennequin"
      ],
      "abstract": "Joint-Embedding Predictive Architecture (JEPA) has shown strong performance in learning rich representations through self-supervised prediction in latent space. However, it typically relies on teacher--student architecture with an EMA to stabilise training, and can tend to yield uninformative representations. Contrastive learning is stable to train and produces strong global representations, but remains limited on local tasks by the global nature of its objective. In this work, we combine both into CoJEPA: a single shared backbone jointly trained with a JEPA objective on masked sequence tokens and a contrastive objective on the class token. The contrastive gradient provides stability, removing the need for an EMA teacher entirely, while JEPA enriches the sequence tokens via local predictions that contrastive learning alone cannot provide. Crucially, no extra parameters are added to the backbone: the same model is guided towards richer representations purely through the design of its training signal. CoJEPA takes the best of both worlds, outperforming or matching both individual methods across global and local MIR tasks, with a particularly strong advantage on tonal and harmonic understanding, and without any task-specific architectural changes. CoJEPA shows that combining objectives with complementary inductive biases can substitute for scale, encouraging future work to invest in smarter training objectives over ever-larger models.",
      "published": "2026-08-31T15:36:13Z",
      "abstract_url": "http://arxiv.org/abs/2608.30974v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30974v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.LG",
        "eess.AS",
        "eess.SP"
      ]
    },
    {
      "title": "A Universal Context-Reuse Layer for Cross-Model KV Sharing",
      "authors": [
        "Yi Li",
        "Dongming Jiang",
        "Yi Zhao",
        "Bingzhe Li"
      ],
      "abstract": "Modern large language model (LLM) serving systems increasingly operate over repeated or shared context, yet each model typically performs its own prefill computation even when another model has already processed the same input. Existing KV-cache reuse mechanisms substantially reduce redundant computation within a single model, but generally assume that the producer and consumer of a cache are identical. We study \\emph{cross-model KV sharing}, which translates the KV state produced by a source model into a representation that can be consumed by a different target model, including models that differ in scale, architecture, attention configuration, tokenizer, and model family. We evaluate the approach in both within-family and cross-family settings. For Qwen2.5-7B $\\rightarrow$ Qwen2.5-1.5B, translated KV states improve LongBench2 accuracy from 27.59\\% to 34.48\\%, a gain of 6.89 percentage points over the native 1.5B baseline, while reducing handoff cost relative to native target prefill. For the cross-family Qwen2.5-1.5B $\\rightarrow$ Gemma-2-2B setting, KV handoff reduces target-side prefill cost by up to 67.05\\% at 4K context length while maintaining decoding perplexity close to native-model baselines. In a more heterogeneous Llama3.1-70B $\\rightarrow$ Qwen2.5-7B setting, cross-family handoff achieves 44.0\\% accuracy compared with 45.7\\% for native Qwen2.5-7B inference, while reducing measured latency from 899ms to 138ms. These results provide initial evidence that KV states can serve as transferable computational representations rather than strictly model-local caches, and motivate \\emph{context mobility} as a systems abstraction for reducing redundant prefill across heterogeneous LLM and multi-agent inference workflows.",
      "published": "2026-08-31T15:28:17Z",
      "abstract_url": "http://arxiv.org/abs/2608.30963v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30963v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Taking the Whys Seriously: Limitations of Counterfactual Explanations in Justification and Recourse",
      "authors": [
        "Mattia Cerrato",
        "Otto Sahlgren",
        "Xenia Heilmann"
      ],
      "abstract": "Counterfactual explanations (CEs) are widely used in explainable artificial intelligence (AI) to show how a model's outputs would change if the input features were manipulated. This technique is used for a range of tasks such as debugging models, explaining predictions, justifying decisions, and providing algorithmic recourse. In this paper, we explore the normative legitimacy of employing counterfactuals in real-life model deployment settings. We discuss the different stakes involved in these different purposes for which CEs are commonly employed, and find stricter requirements for justification and recourse. In particular, we find that naive application of CEs for justification and recourse can lead to ignoring contestable choices made throughout the machine learning (ML) pipeline, thus obfuscating that decisions and counterfactuals for those decisions are also artifacts of an organization's materialized design and governance choices. We demonstrate this with four empirical experiments involving interventions at stages of the ML pipeline ``upstream\" of the explanation itself, and show that these affect the generated counterfactuals. We find that an organization's choices on measurement models for feature and labels, business requirements, model validation, and the metric of model success have as much or more impact on the generated counterfactuals as the specifics of the generating method. Our findings underline the need to account for such choices upon providing justification and recourse, providing a stark reminder of the relational nature of these tasks. As putative justifications or recourse recommendations, CEs do not provide adequate answers to some important \"why\"-questions because they preclude consideration of whether the decision-maker ought to have acted differently.",
      "published": "2026-08-31T15:24:19Z",
      "abstract_url": "http://arxiv.org/abs/2608.30956v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30956v1",
      "categories": [
        "cs.CY",
        "cs.AI"
      ]
    },
    {
      "title": "Towards Stream Learning on Embedded Systems: Benchmarking the Memory Consumption of Stream Learning Methods",
      "authors": [
        "Sebastian Buschjäger",
        "Nuwan Gunasekara",
        "Heitor Murilo Gomes"
      ],
      "abstract": "Stream learning is commonly evaluated through predictive performance and adaptation to concept drift. However, sustained operation of a stream learner also requires predictable and bounded resource usage even on long streams. This requirement becomes even more critical when learning moves from servers to near-sensor embedded systems where memory and processing are scarce resources. In state-of-the-art stream learning, however, we perceive a strong focus on concept drift adaptation, whereas resource usage is often an evaluation byproduct. To close this gap, we benchmark seven representative stream classifiers on 13 real and synthetic streams under model-size budgets from 128\\,KiB to approximately 8\\,MiB. Our benchmark comprises a total of 6,463 experiments. We measure failure-aware accuracy, peak model size, time to budget exhaustion, and prediction-plus-update latency. The results reveal two distinct resource failure modes. Adaptive ensembles can exceed small budgets almost immediately because of their initial footprint, even when their size remains stable thereafter. Incremental trees can fit initially but grow throughout a long stream, with HoeffdingTrees (HT) and Extremely Fast Decision Trees (EFDT) increasing by median factors of 7.37 and 5.87. Explicitly compact methods remain the only viable option under the smallest budgets, but are usually overtaken as larger budgets make adaptive ensembles competitive. Hence, many state-of-the-art methods are only partially applicable in embedded systems or for long-running systems. We therefore call on the stream-learning community to make bounded resource usage a first-class design objective alongside drift adaptation, and propose concrete steps toward this goal, including an API through which stream learners can explicitly expose and respect resource budgets.",
      "published": "2026-08-31T15:01:31Z",
      "abstract_url": "http://arxiv.org/abs/2608.30923v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30923v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.PF"
      ]
    },
    {
      "title": "Safety Screening for Voltage Control in Active Distribution Grids via Distributionally Robust Conformal Screening",
      "authors": [
        "Sarra Bouchkati",
        "Petros Ellinas",
        "Adriana Geisler",
        "Steffen Kortmann",
        "Johanna Vorwerk",
        "Spyros Chatzivasiliadis",
        "Andreas Ulbig"
      ],
      "abstract": "Deploying a new control policy for voltage control in active distribution grids requires evidence that physical limits will be satisfied before the policy is tested on the physical grid. This assessment is difficult for two reasons. First, simulations cannot capture every disturbance, modeling error, and device interaction present in the real grid. Second, historical measurements reflect operation under existing control policies, whereas a new policy may drive the grid into different operating conditions. To address these challenges, we propose Distributionally Robust Conformal Safety Screening (DR-CSS), a policy-agnostic framework for pre-deployment, scenario-by-scenario screening of a new control policy using historical data and a nominal simulator. For each new scenario, the simulator predicts a future voltage trajectory for the whole grid; DR-CSS then constructs a conformal safety interval around this prediction using historical simulation-to-reality errors. The interval is further enlarged to account for closed-loop changes induced by the deployment of the new policy and its interactions with the remaining controllers. To the best of our knowledge, DR-CSS is the first framework in power systems to combine historical data from an existing control policy with an imperfect simulator for pre-deployment safety screening of a new policy. Experiments on the IEEE 33-bus and IEEE 141-bus systems evaluate the deployment of learning-based voltage control policies and show that DR-CSS identifies all unsafe test scenarios. To reduce unnecessary warnings on safe scenarios, we adapt the safety intervals to different operating conditions and gradually introduce new policies with recalibration after each stage. These extensions increase the informational value of the safety screening and support safer deployment decisions in active distribution grids.",
      "published": "2026-08-31T14:43:37Z",
      "abstract_url": "http://arxiv.org/abs/2608.30889v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30889v1",
      "categories": [
        "eess.SY",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Predicting Residential Rents in Dakar Using Machine Learning",
      "authors": [
        "Amadou Tidiane Kassa Diallo"
      ],
      "abstract": "Dakar's residential rental market remains poorly documented despite its economic and social importance: 54.4% of households are renters, compared to 23.3% nationally. This study develops a complete machine learning pipeline to predict residential rents in Dakar, from data collection to model interpretation. An original dataset of 1,507 rental listings was built through systematic web scraping and a documented cleaning pipeline, then enriched with four purpose-built features, including a luxury score and a keyword-based quality score. Five models were compared: linear regression, Random Forest (baseline), XGBoost, and LightGBM optimized through Bayesian optimization with Optuna, using leakage-free KFold target encoding for location. The optimized XGBoost model achieved the best performance with an $R^2$ of 0.847, an MAE of 210,902 XOF, and an RMSE of 324,195 XOF. Feature importance was assessed using native XGBoost gain and SHAP values, revealing a substantial difference in the ranking of location, which appears as a minor predictor by gain but as the second most influential variable by SHAP. This result carries methodological implications for hedonic studies using target-encoded categorical variables. This study provides an interpretable benchmark for Dakar's rental market and highlights several avenues for improvement, including the integration of geospatial features and conformal prediction.",
      "published": "2026-08-31T14:29:04Z",
      "abstract_url": "http://arxiv.org/abs/2608.30865v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30865v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "HSRM: Hidden-State Reward Models for Test-Time Verification",
      "authors": [
        "Xianzhi Li",
        "Xiaodan Zhu"
      ],
      "abstract": "Large language models can often generate plausible mathematical reasoning traces, but reliably identifying the correct solution among multiple candidates remains a key challenge. Existing test-time reasoning pipelines typically rely on text-based verifiers that re-read each generated solution, making verification an expensive component of inference. Prior work has shown, however, that LLMs often encode correctness-related signals in their internal representations, including awareness of when their own answers are likely to be wrong. Building on this observation, we introduce HSRM, a lightweight hidden-state reward model that verifies candidate solutions by directly reading the generator's internal representations rather than re-processing its text. HSRM extracts hidden states from a frozen generator at reasoning-step boundaries and uses a small Transformer encoder to rank candidates. It is trained from self-generated trajectories with outcome labels, requiring neither human-written process supervision nor a large pretrained verifier. Across four mathematical reasoning benchmarks, HSRM matches or outperforms a 55M-parameter text-only energy verifier in 15 of 16 generator--dataset settings while using only about 2M parameters, providing an efficient alternative to text-only verification by reusing representations already computed during generation.",
      "published": "2026-08-31T14:12:19Z",
      "abstract_url": "http://arxiv.org/abs/2608.30841v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30841v1",
      "categories": [
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Conjoint Audio-to-Spikes Encoding and Processing for Efficient Neuromorphic Speech Recognition",
      "authors": [
        "Valentin M. Meunier",
        "Amélie Gruel",
        "Pierre Lewden",
        "Adrien F. Vincent",
        "Sylvain Saïghi"
      ],
      "abstract": "Obtaining data from neuromorphic sensors and processing it with Spiking Neural Networks is a promising solution to lower the energy cost of artificial intelligence. The current rarity of natively neuromorphic datasets promotes the development of software tools to translate input sensory data into spikes. However, highly bio-mimetic simulators can be challenging to implement on digital hardware. In this work, we evaluate the neuromorphic encoding and subsequent classification of audio into spikes using a non-learnable, high-level, programmable encoder targeting hardware implementation on FPGA. We quantify the pipeline's efficiency with hardware-agnostic metrics based on the quantitative spiking activity. Our study focuses on the simultaneous optimisation of encoder and classifier: the first provides efficient and informative data so that the latter achieves a better performance with an overall lower energy cost at learning and inference. This work introduces the first end-to-end neuromorphic spike-encoding and evaluation of the TIMIT dataset. Our simple feedforward network reaches a classification accuracy of 99.77% on a spike-encoded Heidelberg Digits, overcoming the neuromorphic state of the art on this benchmark dataset.",
      "published": "2026-08-31T13:47:14Z",
      "abstract_url": "http://arxiv.org/abs/2608.30792v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30792v1",
      "categories": [
        "cs.NE",
        "cs.AI",
        "cs.LG",
        "eess.AS"
      ]
    },
    {
      "title": "On the Prospects of Dynamic LLM Conversations in Software Development",
      "authors": [
        "Annemarie Wittig",
        "Alina Mailach",
        "Janet Siegmund",
        "Norbert Siegmund"
      ],
      "abstract": "Large language models (LLMs) have become an essential tool for assisting developers, yet we still lack knowledge on ways to effectively support their interactions during development activities. That is, the quality of interactions with a chat-based LLM still strongly depends on how developers phrase prompts and which information they include. Our goal is to evaluate whether interventions into these interactions with LLMs have an effect on software developers---be it harmful or beneficial. To this end, we conducted a four-month longitudinal study with third-semester computer science students working on a full-stack Web development project using chat-based LLMs under three conditions: (1) a \\emph{context}-aware group received intent-based conversation augmentation, (2) a \\emph{proactive} group received follow-up suggestions and tailored advice, and (3) a \\emph{control} group without intervention. Our augmentations are minimal: (i) to reduce confounding factors and (ii) to isolate treatment effects. Analyzing interaction logs and user surveys revealed no major differences in interaction patterns, indicating no detectable harmful effects in the measured outcomes when intervening in interactions. Moreover, we observed trends of increased satisfaction with the \\emph{proactive} treatment. The results indicate that even with minimal interventions, dynamic guidance mechanisms for developer-LLM interactions show observable effects, such that more severe augmentations may have the potential to substantially improve developer satisfaction.",
      "published": "2026-08-31T13:24:26Z",
      "abstract_url": "http://arxiv.org/abs/2608.30756v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30756v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Autoregressive Mosaics: Probing 2D Spatial Reasoning in Text-Only Language Models",
      "authors": [
        "Ashwin Nedungadi",
        "Stefan Oehmcke",
        "Stefan Lüdtke"
      ],
      "abstract": "Large language models (LLMs) trained only on text and code can sometimes generate programs that draw recognizable images. However, it is unclear whether this reflects an internal representation of 2D spatial layout or simply the ability to translate spatial descriptions into code. We introduce Autoregressive Mosaics (AM-Bench), a benchmark that separates these factors: First, a translation task gives a model a fully specified geometry of a picture in words as a prompt and asks for the code that produces it. Second, a layout task requires the model to compose an image from an underspecified prompt. Across eight open-weight text-and-code-only models, all models reliably translate specified geometry into code, but their open-ended layout performance differs substantially, indicating that these differences are not explained by code-generation ability alone. An output-medium ablation further shows that the interface or medium of expression that the model uses matters: replacing procedural code with raw SVG improves layout scores across all models. Finally, probing model activations shows that a coarse layout plan is present before generation, but reflects only the layout implied by the prompt. During generation, models track the evolving geometric state instead of executing an initially fixed plan. Overall, these results show that 2D spatial performance in text-only LLMs depends on both the model and the output medium, and is not explained by code-generation ability alone.",
      "published": "2026-08-31T13:18:03Z",
      "abstract_url": "http://arxiv.org/abs/2608.30751v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30751v1",
      "categories": [
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Calibrating Small Language Models for Claim Check-Worthiness Detection",
      "authors": [
        "Pratuat Amatya",
        "Venktesh Viswanathan",
        "Vinay Setty"
      ],
      "abstract": "Assessing claim check-worthiness is an essential first step in automated fact-checking pipelines. This work is motivated by a real deployment challenge at an early-stage startup: running large language models (LLMs) over every incoming claim is cost- and latency-prohibitive, yet smaller models sacrifice accuracy. We propose NN-PPI, a pointwise extension of Prediction-Powered Inference (PPI) that calibrates model predictions at inference time as a lightweight post-hoc layer, without re-training the underlying model. NN-PPI achieves weighted F1 gains ranging from 12% to 33.80% depending on the size and performance of the baseline model, bringing SLMs on par with larger LLMs. Beyond few-shot SLMs, NN-PPI further improves a production-deployed fine-tuned model, demonstrating that residual calibration is complementary to supervised fine-tuning. By recovering LLM-level accuracy from models that are an order of magnitude cheaper to serve, it makes accurate check-worthiness detection substantially cheaper to operate at scale. Our code and data can be found at https://anonymous.4open.science/r/arr-claim-worthiness-F237.",
      "published": "2026-08-31T13:04:57Z",
      "abstract_url": "http://arxiv.org/abs/2608.30731v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30731v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks",
      "authors": [
        "Pradyumna Shyama Prasad",
        "Meiri Anto",
        "Leon Eshuijs",
        "Julian Moncarz",
        "Kaustubh Kislay",
        "Juan J. Vazquez"
      ],
      "abstract": "LLM agents are increasingly used to run autonomous ML experiments, iterating on target metrics with little human oversight. Prior work has documented reward hacking in these environments, bringing into question the validity of produced research and the broader safety case for AI R&D. Existing benchmarks do not measure exploits that live in the data or the modeling task itself. We introduce BAITBENCH, a suite of three synthetic tabular ML tasks that each contain a shortcut that allows agents to inflate the public test score but fail on a hidden test set. Since the shortcut is optional and using it breaks no stated rule, BAITBENCH measures how often models exploit the shortcut to achieve inflated scores. Across seven frontier agents scored by our two-stage judge pipeline, 57.1% of runs exhibit reward hacking, with five of seven above 50%. Agents cheat even under a second condition where they are prompted not to -the mean cheating rate remains above 50%. We release BAITBENCH, along with the judge implementation, and an annotated dataset of transcripts containing reward hacks as a testbed for evaluating reward-hacking mitigations head-to-head.",
      "published": "2026-08-31T12:59:33Z",
      "abstract_url": "http://arxiv.org/abs/2608.30724v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30724v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "SingProbe Technical Report",
      "authors": [
        "Sing Team"
      ],
      "abstract": "Runtime guardrails are essential for reliable large language model (LLM) deployment, yet existing approaches typically rely on independent, external models that introduce additional inference cost, delayed safety signals, and a capacity mismatch with increasingly capable base models. To address these issues, we introduce SingProbe, a lightweight intrinsic runtime guard that directly reuses hidden states produced during LLM inference and operates alongside autoregressive decoding. Within a unified framework, SingProbe continuously predicts query intent, response safety, and hallucination risk at the token level with negligible additional guardrail inference overhead, offering a \"free-lunch\" solution. We further introduce SingStreamBench, a benchmark designed to assess whether streaming guardrails remain inactive on benign prefixes while promptly detecting emerging unsafe content. Extensive experiments show that SingProbe achieves competitive or superior performance compared with substantially larger standalone guardrails and specialized hallucination detectors, with only $\\approx$2M parameters and $<0.5\\%$ extra overhead. Beyond passive detection, we also show that SingProbe scores can anticipate future generation risk and guide constrained safe decoding. We further extend this paradigm to medical generation through SingProbe-Med, which selectively activates risk-directed decoding interventions only when clinically relevant risks emerge. Together, these results demonstrate that internal model representations provide an effective and efficient interface for generation-time monitoring and control.",
      "published": "2026-08-31T12:42:11Z",
      "abstract_url": "http://arxiv.org/abs/2608.30703v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30703v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.CL",
        "cs.LG"
      ]
    },
    {
      "title": "An Agentic Retrobiosynthesis Framework with Learned Frontier Selection",
      "authors": [
        "Philippe Meyer",
        "Guillaume Gricourt",
        "Thomas Duigou",
        "Joan Hérisson",
        "Jean-Loup Faulon"
      ],
      "abstract": "Large language models are increasingly used as agents for multistep retrosynthesis, raising the question of how much their search policy contributes independently of the underlying reaction model. We investigate this question in a biological setting through rule-based retrobiosynthesis: a deterministic biochemical engine generates the same validated transitions for every method, searching for routes that terminate in metabolites available to an \\emph{Escherichia coli} chassis, while the policy only selects which frontier molecule to expand next. Prompted and LoRA-tuned Qwen2.5-7B policies use a strict choice-only interface. The fine-tuned policy reaches $65\\pm1$\\% solve rate at 10 expansions on LASER versus 59\\% for MCTS, and at 200 expansions reaches $78\\pm1$\\% versus 75\\% on LASER, $88\\pm3$\\% versus 80\\% on the RetroPath RL Golden benchmark, and $63\\pm2$\\% versus 45\\% on the BioNavi-NP benchmark. Fine-tuning also consistently outperforms direct prompting. These results show that route-supervised frontier selection can improve budgeted search without altering biochemical generation, although performance remains dependent on frontier construction and reaction ranking.",
      "published": "2026-08-31T12:40:42Z",
      "abstract_url": "http://arxiv.org/abs/2608.30702v1",
      "pdf_url": "https://arxiv.org/pdf/2608.30702v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
