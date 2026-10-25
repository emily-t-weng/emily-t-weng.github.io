const PAPERS_DATA = {
  "last_updated": "2026-10-25 04:18:58 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Cost-Aware Abstention for LLM-Based IT Ticket Classification",
      "authors": [
        "Sachin Gupta"
      ],
      "abstract": "Wrong automatic routing of an IT support ticket can cost more than human triage. We evaluate cost-aware abstention as an offline, cost-conditional readiness assessment: select a routing policy on calibration data under explicit costs, then evaluate that frozen policy once on held-out test data. On a public 47,837-ticket, eight-class corpus, we compare four trained non-LLM classifiers with four large language models under zero- and few-shot prompting. Under the illustrative cost model (c_ok, c_triage, c_wrong) = (0, 1, 5), TF-IDF with logistic regression and a threshold of 0.60 selected on calibration reduces test cost from 0.763 to 0.472, a 38.1% relative reduction (paired ratio-bootstrap 95% CI [35.7%, 40.4%]), while lowering wrong routes from 15.3% to 3.5%. Frozen calibration-selected policies also beat both always-route and always-defer on held-out test data in 14 of 16 settings in a post-hoc cost-parameter analysis. The strongest prompted configuration reaches 0.477 accuracy on a stratified 1,000-ticket test subset, versus 0.879 for fine-tuned DistilBERT on the same tickets. Isotonic lowers expected calibration error without reliably improving discrimination: GPT-4o-mini few-shot has one confidence value before and after recalibration and correctness AUROC 0.500. We therefore report correctness AUROC and tie-aware risk-coverage area alongside ECE. At the analytically derived break-even probability 0.80, seven prompted configurations defer all tickets; Haiku few-shot routes only 19/1,000. An LLM-only agreement control and margin/entropy rejection baselines further separate calibration from discrimination. The findings are scoped to this dataset, prompt design, model endpoints, and illustrative costs; the work is an evaluation framework, not a new abstention algorithm or universal deployment certification.",
      "published": "2026-10-22T17:58:52Z",
      "abstract_url": "http://arxiv.org/abs/2610.31657v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31657v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "METTA: Memory-Enhanced Test-Time Adaptation for Meta-Reinforcement Learning",
      "authors": [
        "Ali Najar",
        "Hosein Hasani",
        "Amir Kooshan Fattah",
        "Yasamin Rajabi",
        "Mahdieh Soleymani Baghshah"
      ],
      "abstract": "Meta-reinforcement learning aims to train agents that adapt quickly to unseen tasks from limited interaction, but this requires representing task evidence over long interaction histories. Classical recurrent agents compress experience into hidden states, while transformer-based agents retrieve from past trajectories at quadratic memory and computation cost. In this work, we propose Memory-Enhanced Test-Time Adaptation (METTA) for meta-reinforcement learning. It decomposes the fast adaptation phase into two temporal scales: a causal neural memory performs fine-grained within-episode adaptation with linear sequence complexity, while compact episode-level summaries support long-range cross-episode task inference without attending over the full interaction history. Moreover, equipping agents with neural memory provides sufficient expressive capacity and test-time training capability, enabling effective adaptation to unseen held-out tasks and task families at meta-test time. On unseen held-out Meta-World ML10/45 test environments, METTA achieves a success rate over $50\\%$ while maintaining a low training budget and short wall-clock time. On long-context sparse-reward PointMaze tasks, it outperforms transformer-based baselines by exhibiting more structured exploration behavior. These results show that METTA provides an efficient path toward more adaptive and scalable long-context meta-RL.",
      "published": "2026-10-22T17:58:27Z",
      "abstract_url": "http://arxiv.org/abs/2610.31656v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31656v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence",
      "authors": [
        "Qijin Zhang",
        "Hanzhen Wang",
        "Yuehan Zhao",
        "Haoyu Dai",
        "Yao Shu",
        "Zhongxiang Dai",
        "Shuang Qiu"
      ],
      "abstract": "Tool-using language model agents rely on environmental facts to decide whether an action is permitted. When a tool response omits a safety-relevant fact, an agent may carry out a prohibited action even though the user request is benign and every returned statement is true. We introduce \\bench{}, an executable benchmark for evaluating unsafe execution, safety boundaries, and evidence recovery in LLM agents under missing evidence. It contains 300 matched unsafe/safe task pairs across six domains. By systematically suppressing critical facts, the benchmark measures unsafe execution under critical evidence omission and determines the minimum observed failure cardinality, including failures with full evidence and cases with no observed failure after exhaustive testing. An independent audit tool tests whether agents seek missing facts and then withhold prohibited actions while completing permissible tasks. The benchmark reveals substantial variation across the ten models in omission robustness, evidence acquisition, and appropriate action. Resistance to missing evidence does not necessarily translate into useful recovery. Recovery access reduces unsafe execution for all ten models, but agents differ in whether they actively obtain the missing facts and use them to act appropriately in both worlds.",
      "published": "2026-10-22T17:57:04Z",
      "abstract_url": "http://arxiv.org/abs/2610.31650v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31650v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation",
      "authors": [
        "Gabriele Dominici",
        "Teague McMillan",
        "Fazl Barez",
        "Marc Langheinrich",
        "Martin Gjoreski"
      ],
      "abstract": "Chain-of-thought (CoT) prompting elicits step-by-step rationales that appear to explain model predictions, but there is no guarantee these steps causally drive the final answer. In standard autoregressive generation, dense attention allows each reasoning step to attend to the full context, enabling the model to derive answers through direct lookup while generating plausible-sounding but post-hoc justifications. We introduce CausalCoT, an inference framework that enforces causal transparency by decomposing problems into a directed acyclic graph of atomic sub-questions, evaluating each node in an isolated context containing only its direct causal parents. This Markovian structure removes attention pathways to non-parent content, making downstream predictions sensitive to local reasoning rather than global context leakage. Across a diverse suite of reasoning benchmarks, CausalCoT preserves competitive predictive performance while improving explanation faithfulness, increasing causal sensitivity of intermediate reasoning steps by an average of 12 and up to 21 percentage points over the strongest baseline. The structured dependency graph additionally enables test-time interventions in which local corrections propagate predictably through downstream reasoning.",
      "published": "2026-10-22T17:52:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.31639v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31639v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3",
      "authors": [
        "Barış Alparslan",
        "Sandy Tanwisuth",
        "Caleb Petterson"
      ],
      "abstract": "Instrumental convergence is often invoked as a general reason to expect capable agents to acquire or express power-seeking subgoals: self-preservation, resource acquisition, deception, or resistance to oversight which poses an important threat to the society. For LLMs, reinforcement learning is the first stage of post-training that directly optimizes behavior against an objective; therefore, it is the stage where convergent instrumental behavior, on IC grounds, likely to appear. Does reinforcement learning with verifiable rewards (RLVR) make a model more likely to express instrumental-convergence behavior, or does this depend on the training regime, evaluation frame, and stage of post-training? We study this question as a checkpoint-conditioned behavioral trajectory rather than as a single before/after safety score. Using InstrumentalEval, we evaluate OLMo-3-7B-Think across SFT, DPO, 13 intermediate RLVR checkpoints, and the final RLVR model. The trajectory is nonmonotonic: instrumental rate rises from 25.0% at SFT to 46.1% near RLVR step 200, then falls to 11.8% at the final checkpoint. Paired item-level tests and step-wise regression support the final SFT-to-RLVR reduction. Prompt framing does not explain this pattern: a harmful-prompt indicator combined with other framing effects explains only 15% of within-checkpoint score variance on average, and Iterative Null-space Projection (INLP) removes a detectable framing direction without changing the SFT-to-final-RLVR gap. Together with preliminary 32B endpoint results (35.53% SFT to 22.37% RLVR final), these findings indicate that RLVR does not monotonically increase instrumental behavior in this setting: the rate rises early, then falls below the SFT baseline. The result is specific to this model family and benchmark, and judge agreement is moderate, so absolute rates should be read cautiously.",
      "published": "2026-10-22T17:44:41Z",
      "abstract_url": "http://arxiv.org/abs/2610.31624v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31624v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "VERA: A Responsible AI Dashboard for Assessing LLM Deployments in Light of the EU AI Act",
      "authors": [
        "Robin Quériaux",
        "Solveig Bachellery",
        "Léa Déléris",
        "Saurabhsingh Rajput",
        "Mariam Barry",
        "Tushar Sharma",
        "Jérôme Lebecq"
      ],
      "abstract": "Institutions that deploy large language models in regulated settings often need technical evidence of responsible AI practices before approving a deployment. COMPL-AI provides a technical interpretation of the EU AI Act that organises this evidence under six principle groups and eighteen technical requirements, twelve of which current benchmarks can measure. The benchmark suites that produce this evidence are typically operated from a command line, while the staff who approve deployments are non-engineers. We present VERA, an open-source self-hosted dashboard deployed to facilitate responsible AI processes within a large European bank. A reviewer selects a model and runs public benchmarks against it. VERA scores the twelve requirements that current benchmarks can measure and instruments the energy consumed by each evaluation, adding run-level evidence for environmental impact. We run four served open-weight models through the requirement pipeline, and the dashboard displays a score for each measurable requirement. A within-subject study then compares how participants read these benchmark results: from the raw output files in one condition, and through the VERA dashboard in the other. The dashboard reduced the time needed to perform the responsible AI assessment, reducing the time required from 56 seconds using raw files to 30 seconds using our dashboard, on average. Our dashboard scored highly (4.36 out of 5) on usefulness in the survey conducted within the organization. The dashboard and the study material are released publicly to promote replicability and reproducibility in different organisational contexts.",
      "published": "2026-10-22T17:23:39Z",
      "abstract_url": "http://arxiv.org/abs/2610.31588v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31588v1",
      "categories": [
        "cs.SE",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination",
      "authors": [
        "Miguel Fernandez-de-Retana",
        "Unai Zulaika",
        "Ruben Sanchez-Corcuera",
        "Aritz Bilbao-Jayo",
        "Aitor Almeida"
      ],
      "abstract": "Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam's own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at the cost of accuracy, without improving how well confidence separates right from wrong answers. Moreover, 13-38% of correct answers rejected a distractor for a clinically unsound reason that confidence did not reflect, and removing or swapping the provided contextual images left accuracy unchanged.",
      "published": "2026-10-22T17:23:11Z",
      "abstract_url": "http://arxiv.org/abs/2610.31586v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31586v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.CY",
        "cs.LG"
      ]
    },
    {
      "title": "Who Is Sycophantic to Whom? Confident LLMs, Human Deference, and the Collapse of Trust Calibration",
      "authors": [
        "Deepthika S",
        "Pawni Yadav",
        "B. Sankar"
      ],
      "abstract": "Research on sycophancy asks whether large language models (LLMs) bend towards what their users want to hear. We ask the reverse question: when an LLM is confident, consistent, and wrong, do users bend towards it? In a within-subjects experiment, 32 participants learned up to five science topics by chatting with five assistants built on the same model (GPT-5-mini). Unknown to participants, each assistant's briefing had 0, 25, 50, 75, or 100% of its facts replaced by plausible falsehoods, and every assistant spoke with full confidence. A surprise quiz after each topic offered the correct fact, the planted falsehood, and two distractors, with a confidence rating for every answer. Across 104 quizzes and 1,040 answers, scores fell from 7.4 to 2.9 out of 10, and the planted option was chosen on 60% of falsified items against 11% of correctly taught ones (odds ratio 12.5). Transcript coding showed that whenever the assistant stated a falsehood, it was adopted about 65% of the time at every falsification level. Absorbed falsehoods were held with high confidence (8.4/10), the modeled confidence advantage of right over wrong answers vanished, and the discrimination of confidence for correctness fell from an AUROC of 0.70 to chance, while post-task trust ratings barely moved. Participants tended to return to the chat more when the assistant was worse, but returning went with more absorption, not less, and 30% of all turns pasted quiz questions back to the source being checked. The assistant itself was almost never sycophantic and never conceded under pushback. We introduce a baseline-corrected Human Deference Index to quantify this human-side sycophancy and argue that trustworthiness is a property of the human-AI pair, that sycophancy evaluations should measure both directions, and that learning interfaces must make reliability legible claim by claim.",
      "published": "2026-10-22T17:15:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.31579v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31579v1",
      "categories": [
        "cs.AI",
        "cs.HC"
      ]
    },
    {
      "title": "Social-HUMINT: An LLM-Augmented Unified Framework for Measuring Social Media Exposure",
      "authors": [
        "Pellegrino Casoria",
        "Simon Pietro Romano"
      ],
      "abstract": "In the post-API landscape, social-media profiling has become inexpensive and automated: Large Language Models (LLMs) can extract structured information from unstructured content and deduce attributes users never disclosed. A person's true exposure is not what they post but what can be inferred from it. As these practices become routine, exposure to social engineering attacks becomes an operational certainty, demanding a shared framework for assessment and mitigation. We make this exposure measurable by quantifying how accurately a target can be profiled, and how far that profile enables social engineering attacks. Because no existing model fully spans this continuum, we unify Social Media Intelligence (SOCMINT) and Cyber Human Intelligence (Cyber-HUMINT), two individual-focused disciplines that evolved separately, into a single doctrine-grounded taxonomy of 38 capabilities across the Cyber Threat Intelligence lifecycle, where user profiling bridges passive collection and active engagement. Domain experts validated the taxonomy (Fleiss' $κ$ = 0.92). We operationalize its core path (extraction, inference, and susceptibility assessment) as an LLM-augmented multi-agent pipeline, evaluated on real participants' public data. Susceptibility is assessed indirectly: the pipeline predicts how each target would rank personalized engagement options, matching their actual choices significantly above chance. This ranking capacity captures the personalization on which tailored attacks depend. The underlying profiles are accurate: extraction reaches F1 = 97.3%, and inferred attributes score 4.73/5, both far above rule-based and NER-based baselines at negligible cost. The framework augments security analysts' tradecraft rather than replacing it and enables defenders to gauge their own exposure before an adversary does.",
      "published": "2026-10-22T17:13:46Z",
      "abstract_url": "http://arxiv.org/abs/2610.31573v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31573v1",
      "categories": [
        "cs.CR",
        "cs.AI",
        "cs.SI"
      ]
    },
    {
      "title": "When to Follow the Teacher: Advantage-Based On-Policy Distillation",
      "authors": [
        "Mahdi Sabbaghi",
        "George Pappas",
        "Hamed Hassani",
        "Adel Javanmard"
      ],
      "abstract": "Online Policy Distillation (OPD) often outperforms offline methods such as Supervised Fine-Tuning (SFT) by training on trajectories sampled directly from the student. However, standard OPD trains the student to follow the teacher at every state of the trajectory, assuming that teacher guidance is always beneficial. We instead distinguish between what the teacher prefers and what actually helps the current student succeed. We introduce Advantage-Based On-Policy Distillation (AOPD), which weights teacher actions by their counterfactual advantage under the student. Teacher actions that improve the student's expected outcome are reinforced, while those that do not are downweighted. Directly computing these advantages requires trying alternative teacher actions at each step and running new rollouts from each. Instead, we show that the AOPD objective can be estimated using only standard student rollouts and their terminal rewards. Our estimator uses teacher importance weights on the student-sampled actions to provide unbiased advantage-weighted supervision, without any additional rollouts or reward evaluations. This allows AOPD to focus teacher supervision on actions that help the current student while preserving the rollout efficiency of standard OPD. Experiments on math, coding, and Medqa show gains over distillation and RL baselines in both external-teacher and self-distillation settings.",
      "published": "2026-10-22T17:05:21Z",
      "abstract_url": "http://arxiv.org/abs/2610.31563v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31563v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Contestable Agents for Data Governance: Recording Human Stewardship of AI-Generated Metadata Across Structured and Unstructured Systems",
      "authors": [
        "Humza Akhtar",
        "Brian Freeman"
      ],
      "abstract": "Industrial organizations hold operational knowledge in structured records, such as telemetry, point lists, and transactions, and in unstructured sources, such as manuals, service notes, communications, and inspection reports. Governance programs often manage these sources separately. Classifying, linking, and documenting them depends on a small group of employees with deep domain knowledge. Agents built on large language models can propose metadata, cite documents, and flag quality issues, but unreviewed proposals can spread errors into downstream systems. A perfunctory approval leaves no record of what a reviewer checked. We present contestable governance: specialist agents produce scored, cited arguments for disputed metadata decisions; a steward can accept, reject, modify, or add arguments before commit; and a governance record retains the evidence, interventions, and outcome. A versioned domain glossary supplies model-specific terms and acronym definitions to the agents. We describe the workflow, a triage path that reserves full argumentation for uncertain proposals while preserving review before commit, the scoring and record models, and a gate that controls which verified decisions become precedent. Two HVAC use cases illustrate the pattern: point classification using Brick Schema class labels, supported by an open reference implementation, and an illustrative warranty failure-mode decision linking claim fields and telemetry to service notes and revisioned bulletins.",
      "published": "2026-10-22T16:57:01Z",
      "abstract_url": "http://arxiv.org/abs/2610.31553v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31553v1",
      "categories": [
        "cs.AI",
        "cs.DB",
        "cs.HC",
        "cs.MA"
      ]
    },
    {
      "title": "The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models",
      "authors": [
        "Shihui Li",
        "Mette Andersen",
        "Martha Lewis"
      ],
      "abstract": "Metaphor is a crucial aspect of human cognition. As such, we would like to understand how it is represented in large language models (LLMs) and how it might be controlled. We use mechanistic interpretability techniques to identify causal subspaces in a range of LLMs covering different architectures and sizes. We show that we are able to steer output on classification tasks across different parts of speech. We are also able to steer output on generation tasks, indicating that these techniques can be used to control the metaphoricity of LLM output.",
      "published": "2026-10-22T16:33:01Z",
      "abstract_url": "http://arxiv.org/abs/2610.31520v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31520v1",
      "categories": [
        "cs.CL",
        "cs.AI"
      ]
    },
    {
      "title": "From Prompt Rules to Golden Data Model: Production-Scale Generation of Personalized, Compliant Pharmaceutical Claims",
      "authors": [
        "Suyash Mishra",
        "Qiang Li",
        "Subhabrata Adak",
        "Upender Phogat",
        "Srikanth Patil",
        "Satyanarayan Pati"
      ],
      "abstract": "Large language models (LLMs) are increasingly used to generate domain-specific commercial & medical content (e.g., claims, statements, and promotional phrases) in regulated industries such as pharmaceuticals. However, ensuring its regulatory compliance, factual accuracy, and audience-specific personalization remains a major challenge. We report a production case study asking whether moving compliance and personalization signal from prompt-level rules into retrieval time structured context improves pharmaceutical claims generation, using Composer, a deployed internal tool built on a Golden Data Model (GDM) of 1,867 pharmaceutical claims spanning four products. The GDM integrates claim metadata, references, audience labels, material type, usage history, Match Text Variant (MTV) links, and a six dimension quality score (6C). Across six product-audience segments, the context pipeline improves intended indication (+29%), intended content type (+62%) and intended target audience (+67%), 6C quality score (+86%), and claim reuse (+34%). Production log analysis over the deployment period shows a 29.5% reduction in weekly API cost. We report this as descriptive deployment evidence. This work is a systems and data architecture contribution: it shows that in a regulated, high-stakes domain, restructuring claim data for retrieval-time filtering is a more scalable lever than expanding prompt-level rules, and documents the engineering path data cleaning, graph linking, and quality scoring.",
      "published": "2026-10-22T16:31:36Z",
      "abstract_url": "http://arxiv.org/abs/2610.31519v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31519v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "SABLE: Scalable Affine Bayesian Latent-space Exploration",
      "authors": [
        "Erik Bodin",
        "Lisa Maria Kreusser",
        "Donney Fan",
        "Henry Moss",
        "Geoff Pleiss",
        "Nicolas Durrande"
      ],
      "abstract": "Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show that the off-span norm implicitly selected by affine EI can lead to underexploration, and SABLE instead prescribes it using a finite-horizon schedule, without restricting search to a learned subspace. A recompression scheme reduces information to $\\mathcal O(M)$ pseudo-observations, bounding refitting cost by $\\mathcal O(M^2D+M^3)$ while keeping parameter inference, prediction, and acquisition decisions to tens of milliseconds. Across synthetic benchmarks and latent-space optimisation tasks, SABLE achieves high sample efficiency and reduces decision time by several orders of magnitude at large budgets.",
      "published": "2026-10-22T16:13:42Z",
      "abstract_url": "http://arxiv.org/abs/2610.31495v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31495v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "stat.ML"
      ]
    },
    {
      "title": "Training Small Language Models for Transferable Tool Selection",
      "authors": [
        "Zhaoyi Zhou",
        "Akshayaa Magesh",
        "Jeongyeol Kwon",
        "Artem Kuriksha",
        "Daniel Jiang",
        "Andrea Zanette",
        "Ayush Jain"
      ],
      "abstract": "As LLM agents operate over corpora of more than ten thousand tools, selecting the right tool for a complex multi-step user instruction becomes a central bottleneck. Tool descriptions cannot all fit in context, and relevance-based retrievers only rank tools without knowing which are actually needed. Recent work instead leverages proprietary commercial LLMs like Claude or ChatGPT to drive a retriever, decomposing the instruction and selecting over retrieved candidates. We show that a small, open model can be trained via reinforcement learning to perform the same functionality. On ToolBench, a Qwen3-8B model post-trained with our recipe outperforms Llama-4-Maverick (402B). In addition, the trained model transfers directly to different retrievers and generalizes to new tool corpora in different domains.",
      "published": "2026-10-22T16:07:49Z",
      "abstract_url": "http://arxiv.org/abs/2610.31483v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31483v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "MoG-VLM: Efficient Video Anomaly Detection through Motion-Guided Vision-Language Models",
      "authors": [
        "Khaled Waleed Dawoud",
        "Muhammad Zaigham Zaheer",
        "Khan Muhammad",
        "Muhammad Haris Khan"
      ],
      "abstract": "Vision-Language Models (VLMs) have shown strong potential for video anomaly detection, but dense frame-level inference makes them impractical for real-time deployment. Existing acceleration commonly relies on uniform sampling, which ignores content dynamics and can either waste computation on static segments or miss localized anomalies. We propose MoG-VLM a motion-adaptive anomaly detection framework that leverages motion purely as a lightweight control signal to decide when VLM inference is necessary. To this end, we introduce Temporal Gradient Magnitude (TGM) which performs content-aware sampling to concentrate computation on informative regions while avoiding redundant processing. We further introduce a hierarchical dual-stream design in which dense multi-frame VLM reasoning is activated only around frames with elevated anomaly likelihood, via anchor-triggered context-window expansion with early termination. A fusion module then reconciles these instantaneous semantic cues with temporal action dynamics. Across four benchmarks, our method achieves 91.91%, 84.30%, 87.88%, and 78.96% AUC-ROC on XD-Violence, UCF-Crime, MSAD, and ShanghaiTech, respectively, establishing new state of the art on three benchmarks under the zero-shot, training-free setting. On XD-Violence, we improve over the previous best result by +0.57% while delivering a 1.31x inference speedup. These results demonstrate that content-aware allocation of VLM capacity, processing fewer frames more intelligently rather than all frames uniformly, is a more effective axis of progress than uniformly densifying VLM inference across frames. Code is available on GitHub: https://github.com/kdawoud91/MoG-VLM.",
      "published": "2026-10-22T16:03:47Z",
      "abstract_url": "http://arxiv.org/abs/2610.31478v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31478v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Adaptive Skill Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery",
      "authors": [
        "Dingxuan Zhang",
        "Yanhui Jiang",
        "Jun Wang",
        "Hefeng Zhou"
      ],
      "abstract": "Reusable skills allow large language model agents to package task-specific procedures, tool affordances, and execution guidance into modular units. Prior work has largely studied how agents retrieve, compose, or synthesize skills before execution. In coding agents, however, execution poses an additional runtime control problem: the same task may be solved through primitive tool calls, macro skills, or hybrid execution that starts with a macro and falls back to micro-level recovery. We introduce Adaptive Skill Granularity, a framework that treats skill granularity as an action-abstraction policy conditioned on task structure, execution state, and macro-skill failures. The system combines a feature-based granularity controller and a finite-state recovery executor. We evaluate the framework with a deterministic coding-agent protocol using verifier-defined success, total cost, and process metrics such as macro adoption, failure recovery, route match, and transition correctness. Results show that fixed macro execution is efficient when applicable but brittle under mismatch, while runtime routing and explicit macro-to-micro recovery improve robustness and recover controlled macro failures that fixed-granularity execution leaves unresolved. Our findings suggest that skill libraries for coding agents should be treated not only as retrievable resources, but also as controllable, verifiable, and recoverable execution abstractions.",
      "published": "2026-10-22T15:55:18Z",
      "abstract_url": "http://arxiv.org/abs/2610.31461v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31461v1",
      "categories": [
        "cs.AI",
        "cs.SE"
      ]
    },
    {
      "title": "DIAL: Steerable Language Models with Distilled Instruction-Adherence Levels",
      "authors": [
        "Thanh Q. Tran",
        "John Russell Himawan",
        "Arun Verma",
        "Kiwan Wong",
        "Wei Xiao",
        "Daniela Rus",
        "Bryan Kian Hsiang Low"
      ],
      "abstract": "When a user says \"Keep it short\", how short should the LLM's response be? More generally, users and applications require different levels of instruction adherence, but prompting does not reliably provide fine-grained control over them. We therefore study \"steerable language models\" that provide fine-grained numerical control over the degree to which models follow general instructions specified in text. However, existing approaches do not jointly offer fine-grained control, efficiency, and scalability across diverse instructions. To address this gap, we introduce DIAL (Distilled Instruction-Adherence Levels) to achieve all three properties. For fine-grained control, we perform classifier-free guidance (CFG) with a frozen LLM to provide training signals for on-policy distillation of instruction adherence at different levels. For efficiency, we design a lightweight steering adapter that uses a classifier over the LLM's hidden states to compute a closed-form steering edit. The classifier adaptively sets the steering magnitude at each token based on the target adherence level, while accounting for the geometry of the LLM's hidden states to preserve response coherence. For scalability, the classifier is text-conditioned, allowing a shared adapter to support new instructions without per-instruction fitting, and training requires only instruction and request text. Evaluation across multiple LLM families shows that DIAL achieves a competitive balance between concept expression and response fluency, with concept scores increasing monotonically across adherence levels. We introduce SteerIF, an instruction-only dataset of 98, 625 concept, persona, verifiable, and descriptive instructions. When trained on SteerIF, DIAL provides fine-grained control over personas and, when combined with prompting, improves verifiable instruction following.",
      "published": "2026-10-22T15:34:39Z",
      "abstract_url": "http://arxiv.org/abs/2610.31438v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31438v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Conservative Meta-Agent Search: Architecture Adaptation under Non-Stationarity",
      "authors": [
        "Ali Atiah Alzahrani"
      ],
      "abstract": "Automated agentic-system design is typically evaluated on static task suites with immediate scalar rewards. Deployed systems face persistent shift, delayed dependent feedback, correlated component failures, and costly architecture changes. We formalize meta-agent architecture search as sequential incumbent replacement with switching costs and hard risk constraints and introduce Conservative Meta-Agent Search (CMAS). CMAS retrieves archived architectures and proposes typed graph edits, then scores each against the incumbent by paired shadow replay under common random numbers, with interventional component credit. Promotion requires an empirically calibrated, variance-penalized simultaneous score above a pre-specified margin, and is followed by a simulated canary stage and pre-specified rollback. Since backtests cannot certify meta-level skill, we introduce MetaInvest-Bench, a regime-switching simulator whose specified worker competence makes a surrogate-optimal architecture computable, with leakage-controlled replay, adversarial governance tracks, and five negative controls. Financial streams serve as a risk-sensitive testbed for meta-level generalization; we claim no alpha. Across 96 streams, CMAS cuts normalized mean surrogate dynamic-oracle regret from 0.116 to 0.104 and raises held-out utility from 0.44 to 0.48 over the strongest adaptive baseline; when extra workers add no value, it contracts to near-single-agent teams. Its deployment inference cost is ~6% lower, though its search-inclusive token cost is the highest of the methods compared. The observed per-update false-promotion rate is 3.4% against a nominal 5% target, and 4.8% in a sealed rerun. The calibration is empirical: it concerns an update-level replay statistic on an author-designed benchmark, and guarantees neither full-utility improvement, nor absence of simulated canary-stage harm, nor sequence-level error control.",
      "published": "2026-10-22T15:32:06Z",
      "abstract_url": "http://arxiv.org/abs/2610.31436v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31436v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.MA"
      ]
    },
    {
      "title": "Predictable performance is suboptimal, and what it means for model routing",
      "authors": [
        "Guanhua Zhang",
        "Moritz Hardt"
      ],
      "abstract": "Large language models (LLMs) differ in capabilities and costs, motivating routing methods that adaptively choose a model for each query. These methods rely on performance prediction: given a query, estimate whether a model will answer it correctly. Most routing methods train separate predictors of candidate models' performance to guide model selection. Our work shows that performance prediction only beats a simple baseline when the candidate model is suboptimal on the task or the performance predictor adds significant capacity relative to the candidate model. Specifically, we take a model's self-confidence, the probability it assigns to its chosen answer, as a baseline and study how much a performance predictor can improve on it. First, we show that any performance predictor that outperforms a model's self-confidence can be used to improve the model's loss: the predictor's advantage transfers exactly in log loss through recalibration. Second, accounting for the predictor's additional parameters, we bound its advantage by the model's suboptimality plus the gain from those parameters. Third, we extend our theoretical results to two-model cascades, which route queries to a more capable model when the first model is predicted to fail. Experiments with 16 language models on four benchmarks show that predictors gain a larger advantage over self-confidence on more suboptimal models. This advantage shrinks as models improve through finetuning. Cascading exhibits the same two patterns: predictor-based routing offers larger gains over confidence-based routing for more suboptimal models, and these gains diminish after finetuning.",
      "published": "2026-10-22T15:27:27Z",
      "abstract_url": "http://arxiv.org/abs/2610.31428v1",
      "pdf_url": "https://arxiv.org/pdf/2610.31428v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    }
  ]
};
