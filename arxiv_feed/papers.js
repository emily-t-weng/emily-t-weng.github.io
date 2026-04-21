const PAPERS_DATA = {
  "last_updated": "2026-04-21 03:32:26 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval",
      "authors": [
        "Shaden Alshammari",
        "Kevin Wen",
        "Abrar Zainal",
        "Mark Hamilton",
        "Navid Safaei",
        "Sultan Albarakati",
        "William T. Freeman",
        "Antonio Torralba"
      ],
      "abstract": "Mathematical problem solving remains a challenging test of reasoning for large language and multimodal models, yet existing benchmarks are limited in size, language coverage, and task diversity. We introduce MathNet, a high-quality, large-scale, multimodal, and multilingual dataset of Olympiad-level math problems together with a benchmark for evaluating mathematical reasoning in generative models and mathematical retrieval in embedding-based systems. MathNet spans 47 countries, 17 languages, and two decades of competitions, comprising 30,676 expert-authored problems with solutions across diverse domains. In addition to the core dataset, we construct a retrieval benchmark consisting of mathematically equivalent and structurally similar problem pairs curated by human experts. MathNet supports three tasks: (i) Problem Solving, (ii) Math-Aware Retrieval, and (iii) Retrieval-Augmented Problem Solving. Experimental results show that even state-of-the-art reasoning models (78.4% for Gemini-3.1-Pro and 69.3% for GPT-5) remain challenged, while embedding models struggle to retrieve equivalent problems. We further show that retrieval-augmented generation performance is highly sensitive to retrieval quality; for example, DeepSeek-V3.2-Speciale achieves gains of up to 12%, obtaining the highest scores on the benchmark. MathNet provides the largest high-quality Olympiad dataset together with the first benchmark for evaluating mathematical problem retrieval, and we publicly release both the dataset and benchmark at https://mathnet.mit.edu.",
      "published": "2026-04-20T17:59:49Z",
      "abstract_url": "http://arxiv.org/abs/2604.18584v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18584v1",
      "categories": [
        "cs.AI",
        "cs.DL",
        "cs.IR",
        "cs.LG"
      ]
    },
    {
      "title": "Sessa: Selective State Space Attention",
      "authors": [
        "Liubomyr Horbatko"
      ],
      "abstract": "Modern sequence models are dominated by Transformers, where self-attention mixes information from the visible context in an input-dependent way. However, when retrieval is not sharp and attention remains diffuse over an effective support $S_{\\mathrm{eff}}(t)$, the influence of any individual token is diluted, typically scaling as $O(1/S_{\\mathrm{eff}}(t))$ and reaching $O(1/\\ell)$ for old tokens in full-prefix settings. Structured state-space models process sequences recurrently through an explicit feedback path; selective variants such as Mamba make this feedback input-dependent, yet when freeze time cannot be sustained over long intervals, their long-range sensitivity decays exponentially with lag. Existing architectures therefore either retrieve from the past in a single read or propagate information through a single feedback chain. We introduce Sessa, a decoder that places attention inside a feedback path, enabling recurrent many-path aggregation within a layer. Under stated assumptions, Sessa admits regimes with a power-law memory tail in lag $\\ell$ of order $O(\\ell^{-β})$ for $0<β<1$, which is asymptotically slower than $1/\\ell$; moreover, this rate is tight in an explicit diffuse uniform-routing setting where the influence is $Θ(\\ell^{-β})$. Under the same conditions, only Sessa among the compared model classes realizes flexible selective retrieval, including non-decaying profiles. Empirically, under matched architectures and training budgets, Sessa achieves the strongest performance on our long-context benchmarks while remaining competitive with Transformer and Mamba style baselines on short-context language modeling.",
      "published": "2026-04-20T17:59:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.18580v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18580v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Bounded Ratio Reinforcement Learning",
      "authors": [
        "Yunke Ao",
        "Le Chen",
        "Bruce D. Lee",
        "Assefa S. Wahd",
        "Aline Czarnobai",
        "Philipp Fürnstahl",
        "Bernhard Schölkopf",
        "Andreas Krause"
      ],
      "abstract": "Proximal Policy Optimization (PPO) has become the predominant algorithm for on-policy reinforcement learning due to its scalability and empirical robustness across domains. However, there is a significant disconnect between the underlying foundations of trust region methods and the heuristic clipped objective used in PPO. In this paper, we bridge this gap by introducing the Bounded Ratio Reinforcement Learning (BRRL) framework. We formulate a novel regularized and constrained policy optimization problem and derive its analytical optimal solution. We prove that this solution ensures monotonic performance improvement. To handle parameterized policy classes, we develop a policy optimization algorithm called Bounded Policy Optimization (BPO) that minimizes an advantage-weighted divergence between the policy and the analytic optimal solution from BRRL. We further establish a lower bound on the expected performance of the resulting policy in terms of the BPO loss function. Notably, our framework also provides a new theoretical lens to interpret the success of the PPO loss, and connects trust region policy optimization and the Cross-Entropy Method (CEM). We additionally extend BPO to Group-relative BPO (GBPO) for LLM fine-tuning. Empirical evaluations of BPO across MuJoCo, Atari, and complex IsaacLab environments (e.g., Humanoid locomotion), and of GBPO for LLM fine-tuning tasks, demonstrate that BPO and GBPO generally match or outperform PPO and GRPO in stability and final performance.",
      "published": "2026-04-20T17:59:01Z",
      "abstract_url": "http://arxiv.org/abs/2604.18578v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18578v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "When Can LLMs Learn to Reason with Weak Supervision?",
      "authors": [
        "Salman Rahman",
        "Jingyan Shen",
        "Anna Mordvina",
        "Hamid Palangi",
        "Saadia Gabriel",
        "Pavel Izmailov"
      ],
      "abstract": "Large language models have achieved significant reasoning improvements through reinforcement learning with verifiable rewards (RLVR). Yet as model capabilities grow, constructing high-quality reward signals becomes increasingly difficult, making it essential to understand when RLVR can succeed under weaker forms of supervision. We conduct a systematic empirical study across diverse model families and reasoning domains under three weak supervision settings: scarce data, noisy rewards, and self-supervised proxy rewards. We find that generalization is governed by training reward saturation dynamics: models that generalize exhibit a prolonged pre-saturation phase during which training reward and downstream performance climb together, while models that saturate rapidly memorize rather than learn. We identify reasoning faithfulness, defined as the extent to which intermediate steps logically support the final answer, as the pre-RL property that predicts which regime a model falls into, while output diversity alone is uninformative. Motivated by these findings, we disentangle the contributions of continual pre-training and supervised fine-tuning, finding that SFT on explicit reasoning traces is necessary for generalization under weak supervision, while continual pre-training on domain data amplifies the effect. Applied together to Llama3.2-3B-Base, these interventions enable generalization across all three settings where the base model previously failed.",
      "published": "2026-04-20T17:57:49Z",
      "abstract_url": "http://arxiv.org/abs/2604.18574v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18574v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale",
      "authors": [
        "A. Sophia Koepke",
        "Daniil Zverev",
        "Shiry Ginosar",
        "Alexei A. Efros"
      ],
      "abstract": "The Platonic Representation Hypothesis suggests that neural networks trained on different modalities (e.g., text and images) align and eventually converge toward the same representation of reality. If true, this has significant implications for whether modality choice matters at all. We show that the experimental evidence for this hypothesis is fragile and depends critically on the evaluation regime. Alignment is measured using mutual nearest neighbors on small datasets ($\\approx$1K samples) and degrades substantially as the dataset is scaled to millions of samples. The alignment that remains between model representations reflects coarse semantic overlap rather than consistent fine-grained structure. Moreover, the evaluations in Huh et al. are done in a one-to-one image-caption setting, a constraint that breaks down in realistic many-to-many settings and further reduces alignment. We also find that the reported trend of stronger language models increasingly aligning with vision does not appear to hold for newer models. Overall, our findings suggest that the current evidence for cross-modal representational convergence is considerably weaker than subsequent works have taken it to be. Models trained on different modalities may learn equally rich representations of the world, just not the same one.",
      "published": "2026-04-20T17:56:02Z",
      "abstract_url": "http://arxiv.org/abs/2604.18572v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18572v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "A multimodal and temporal foundation model for virtual patient representations at healthcare system scale",
      "authors": [
        "Andrew Zhang",
        "Tong Ding",
        "Sophia J. Wagner",
        "Caiwei Tian",
        "Ming Y. Lu",
        "Rowland Pettit",
        "Joshua E. Lewis",
        "Alexandre Misrahi",
        "Dandan Mo",
        "Long Phi Le",
        "Faisal Mahmood"
      ],
      "abstract": "Modern medicine generates vast multimodal data across siloed systems, yet no existing model integrates the full breadth and temporal depth of the clinical record into a unified patient representation. We introduce Apollo, a multimodal temporal foundation model trained and evaluated on over three decades of longitudinal hospital records from a major US hospital system, composed of 25 billion records from 7.2 million patients, representing 28 distinct medical modalities and 12 major medical specialties. Apollo learns a unified representation space integrating over 100 thousand unique medical events in our clinical vocabulary as well as images and clinical text. This \"atlas of medical concepts\" forms a computational substrate for modeling entire patient care journeys comprised of sequences of structured and unstructured events, which are compressed by Apollo into virtual patient representations. To assess the potential of these whole-patient representations, we created 322 prognosis and retrieval tasks from a held-out test set of 1.4 million patients. We demonstrate the generalized clinical forecasting potential of Apollo embeddings, including predicting new disease onset risk up to five years in advance (95 tasks), disease progression (78 tasks), treatment response (59 tasks), risk of treatment-related adverse events (17 tasks), and hospital operations endpoints (12 tasks). Using feature attribution techniques, we show that model predictions align with clinically-interpretable multimodal biomarkers. We evaluate semantic similarity search on 61 retrieval tasks, and moreover demonstrate the potential of Apollo as a multimodal medical search engine using text and image queries. Together, these modeling capabilities establish the foundation for computable medicine, where the full context of patient care becomes accessible to computational reasoning.",
      "published": "2026-04-20T17:55:47Z",
      "abstract_url": "http://arxiv.org/abs/2604.18570v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18570v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring and KV-Cache Steering",
      "authors": [
        "Manan Gupta",
        "Dhruv Kumar"
      ],
      "abstract": "Large language models frequently commit unrecoverable reasoning errors mid-generation: once a wrong step is taken, subsequent tokens compound the mistake rather than correct it. We introduce $\\textbf{Latent Phase-Shift Rollback}$ (LPSR): at each generation step, we monitor the residual stream at a critical layer lcrit, detect abrupt directional reversals (phase shifts) via a cosine-similarity $+$ entropy dual gate, and respond by rolling back the KV-cache and injecting a pre-computed steering vector. No fine-tuning, gradient computation, or additional forward passes are required. LPSR achieves $\\mathbf{44.0\\%}$ on MATH-500 with an 8B model versus $28.8\\%$ for standard AR ($+15.2$ pp; McNemar $χ^2 = 66.96$, $p < 10^{-15}$). Critically, prompted self-correction, the most natural inference-time baseline, scores only $19.8\\%$, below standard AR; LPSR exceeds it by $+24.2$ pp ($χ^2 = 89.4$, $p \\approx 0$). LPSR also outperforms Best-of-16 ($+7.8$ pp) at $5.4\\times$ lower token cost, and surpasses a standard 70B model ($35.2\\%$) with $8.75\\times$ fewer parameters at ${\\sim}3\\times$ the token budget. A 32-layer sweep reveals a novel \\textbf{detection-correction dissociation}: error-detection AUC peaks at layer~14 ($0.718$) but task accuracy peaks at layer~16 ($44.0\\%$ vs.\\ $29.2\\%$), demonstrating that optimal monitoring depth differs for detection and correction.",
      "published": "2026-04-20T17:53:33Z",
      "abstract_url": "http://arxiv.org/abs/2604.18567v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18567v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Benchmarking System Dynamics AI Assistants: Cloud Versus Local LLMs on CLD Extraction and Discussion",
      "authors": [
        "Terry Leitch"
      ],
      "abstract": "We present a systematic evaluation of large language model families -- spanning both proprietary cloud APIs and locally-hosted open-source models -- on two purpose-built benchmarks for System Dynamics AI assistance: the \\textbf{CLD Leaderboard} (53 tests, structured causal loop diagram extraction) and the \\textbf{Discussion Leaderboard} (interactive model discussion, feedback explanation, and model building coaching). On CLD extraction, cloud models achieve 77--89\\% overall pass rates; the best local model reaches 77\\% (Kimi~K2.5~GGUF~Q3, zero-shot engine), matching mid-tier cloud performance. On Discussion, the best local models achieve 50--100\\% on model building steps and 47--75\\% on feedback explanation, but only 0--50\\% on error fixing -- a category dominated by long-context prompts that expose memory limits in local deployments. A central contribution of this paper is a systematic analysis of \\textit{model type effects} on performance: we compare reasoning vs.\\ instruction-tuned architectures, GGUF (llama.cpp) vs.\\ MLX (mlx\\_lm) backends, and quantization levels (Q3 / Q4\\_K\\_M / MLX-3bit / MLX-4bit / MLX-6bit) across the same underlying model families. We find that backend choice has larger practical impact than quantization level: mlx\\_lm does not enforce JSON schema constraints, requiring explicit prompt-level JSON instructions, while llama.cpp grammar-constrained sampling handles JSON reliably but causes indefinite generation on long-context prompts for dense models. We document the full parameter sweep ($t$, $p$, $k$) for all local models, cleaned timing data (stuck requests excluded), and a practitioner guide for running 671B--123B parameter models on Apple~Silicon.",
      "published": "2026-04-20T17:53:29Z",
      "abstract_url": "http://arxiv.org/abs/2604.18566v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18566v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "cs.LG"
      ]
    },
    {
      "title": "OGER: A Robust Offline-Guided Exploration Reward for Hybrid Reinforcement Learning",
      "authors": [
        "Xinyu Ma",
        "Mingzhou Xu",
        "Xuebo Liu",
        "Chang Jin",
        "Qiang Wang",
        "Derek F. Wong",
        "Min Zhang"
      ],
      "abstract": "Recent advancements in Reinforcement Learning with Verifiable Rewards (RLVR) have significantly improved Large Language Model (LLM) reasoning, yet models often struggle to explore novel trajectories beyond their initial latent space. While offline teacher guidance and entropy-driven strategies have been proposed to address this, they often lack deep integration or are constrained by the model's inherent capacity. In this paper, we propose OGER, a novel framework that unifies offline teacher guidance and online reinforcement learning through a specialized reward modeling lens. OGER employs multi-teacher collaborative training and constructs an auxiliary exploration reward that leverages both offline trajectories and the model's own entropy to incentivize autonomous exploration. Extensive experiments across mathematical and general reasoning benchmarks demonstrate that OGER significantly outperforms competitive baselines, achieving substantial gains in mathematical reasoning while maintaining robust generalization to out-of-domain tasks. We provide a comprehensive analysis of training dynamics and conduct detailed ablation studies to validate the effectiveness of our entropy-aware reward modulation. Our code is available at https://github.com/ecoli-hit/OGER.git.",
      "published": "2026-04-20T17:26:00Z",
      "abstract_url": "http://arxiv.org/abs/2604.18530v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18530v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "IDOBE: Infectious Disease Outbreak forecasting Benchmark Ecosystem",
      "authors": [
        "Aniruddha Adiga",
        "Jingyuan Chou",
        "Anshul Chiranth",
        "Bryan Lewis",
        "Ana I. Bento",
        "Shaun Truelove",
        "Geoffrey Fox",
        "Madhav Marathe",
        "Harry Hochheiser",
        "Srini Venkatramanan"
      ],
      "abstract": "Epidemic forecasting has become an integral part of real-time infectious disease outbreak response. While collaborative ensembles composed of statistical and machine learning models have become the norm for real-time forecasting, standardized benchmark datasets for evaluating such methods are lacking. Further, there is limited understanding on performance of these methods for novel outbreaks with limited historical data. In this paper, we propose IDOBE, a curated collection of epidemiological time series focused on outbreak forecasting. IDOBE compiles from multiple data repositories spanning over a century of surveillance and across U.S. states and global locations. We perform derivative-based segmentation to generate over 10,000 outbreaks covering multiple outcomes such as cases and hospitalizations for 13 diseases. We consider a variety of information-theoretic and distributional measures to quantify the epidemiological diversity of the dataset. Finally, we perform multi-horizon short-term forecasting (1- to 4-week-ahead) through the progression of the outbreak using 11 baseline models and report on their performance. In addition to standard metrics such as NMSE and MAPE for point forecasts, we include probabilistic scoring rules such as Normalized Weighted Interval Score (NWIS) to quantify the performance. We find that MLP-based methods have the most robust performance, with statistical methods having a slight edge during the pre-peak phase. IDOBE dataset along with baselines are released publicly on https://github.com/NSSAC/IDOBE to enable standardized, reproducible benchmarking of outbreak forecasting methods.",
      "published": "2026-04-20T17:18:18Z",
      "abstract_url": "http://arxiv.org/abs/2604.18521v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18521v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.PE"
      ]
    },
    {
      "title": "Learning the Riccati solution operator for time-varying LQR via Deep Operator Networks",
      "authors": [
        "Jun Chen",
        "Umberto Biccari",
        "Junmin Wang"
      ],
      "abstract": "We propose a computational framework for replacing the repeated numerical solution of differential Riccati equations in finite-horizon Linear Quadratic Regulator (LQR) problems by a learned operator surrogate. Instead of solving a nonlinear matrix-valued differential equation for each new system instance, we construct offline an approximation of the associated solution operator mapping time-dependent system parameters to the Riccati trajectory. The resulting model enables fast online evaluation of approximate optimal feedbacks across a wide class of systems, thereby shifting the computational burden from repeated numerical integration to a one-time learning stage. From a theoretical perspective, we establish control-theoretic guarantees for this operator-based approximation. In particular, we derive bounds quantifying how operator approximation errors propagate to feedback performance, trajectory accuracy, and cost suboptimality, and we prove that exponential stability of the closed-loop system is preserved under sufficiently accurate operator approximation. These results provide a framework to assess the reliability of data-driven approximations in optimal control. On the computational side, we design tailored DeepONet architectures for matrix-valued, time-dependent problems and introduce a progressive learning strategy to address scalability with respect to the system dimension. Numerical experiments on both time-invariant and time-varying LQR problems demonstrate that the proposed approach achieves high accuracy and strong generalization across a wide range of system configurations, while delivering substantial computational speedups compared to classical solvers. The method offers an effective and scalable alternative for parametric and real-time optimal control applications.",
      "published": "2026-04-20T16:56:34Z",
      "abstract_url": "http://arxiv.org/abs/2604.18507v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18507v1",
      "categories": [
        "math.OC",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Faster by Design: Interactive Aerodynamics via Neural Surrogates Trained on Expert-Validated CFD",
      "authors": [
        "Nicholas Thumiger",
        "Andrea Bartezzaghi",
        "Mattia Rigotti",
        "Cezary Skura",
        "Thomas Frick",
        "Elisa Serioli",
        "Fabrizio Arbucci",
        "A. Cristiano I. Malossi"
      ],
      "abstract": "Computational Fluid Dynamics (CFD) is central to race-car aerodynamic development, yet its cost -- tens of thousands of core-hours per high-fidelity evaluation -- severely limits the design space exploration feasible within realistic budgets. AI-based surrogate models promise to alleviate this bottleneck, but progress has been constrained by the limited complexity of public datasets, which are dominated by smoothed passenger-car shapes that fail to exercise surrogates on the thin, complex, highly loaded components governing motorsport performance. This work presents three primary contributions. First, we introduce a high-fidelity RANS dataset built on a parametric LMP2-class CAD model and spanning six operating conditions (map points) covering straight-line and cornering regimes, generated and validated by aerodynamics experts at Dallara to preserve features relevant to industrial motorsport. Second, we present the Gauge-Invariant Spectral Transformer (GIST), a graph-based neural operator whose spectral embeddings encode mesh connectivity to enhance predictions on tightly packed, complex geometries. GIST guarantees discretization invariance and scales linearly with mesh size, achieving state-of-the-art accuracy on both public benchmarks and the proposed race-car dataset. Third, we demonstrate that GIST achieves a level of predictive accuracy suitable for early-stage aerodynamic design, providing a first validation of the concept of interactive design-space exploration -- where engineers query a surrogate in place of the CFD solver -- within industrial motorsport workflows.",
      "published": "2026-04-20T16:42:35Z",
      "abstract_url": "http://arxiv.org/abs/2604.18491v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18491v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Generalized Synthetic Control Method for Baseline Estimation in Demand Response Services",
      "authors": [
        "Jonas Sievers",
        "Mardavij Roozbehani"
      ],
      "abstract": "Baseline estimation is critical to Demand Response (DR) settlement in electricity markets, yet existing machine learning methods remain limited in predictive performance, while methodologies from causal inference and counterfactual prediction are still underutilized in this domain. We introduce a Generalized Synthetic Control Method that builds on the classical Synthetic Control Method (SCM) from econometrics. While SCM provides a powerful framework for counterfactual estimation, classical SCM remains a static estimator: it fits the treated unit as a combination of contemporaneous donor units and therefore ignores predictable temporal structure in the residual error. We develop a generalized SCM framework that transforms baseline estimation into a dynamic counterfactual prediction problem by augmenting the donor representation with exogenous features, lagged treated load, and selected lagged donor signals. This enriched representation allows the estimator to capture autoregressive dependence, delayed donor-response patterns, and error-correction effects beyond the scope of standard SCM. The framework further accommodates nonlinear predictors when linear weighting is inadequate, with the greatest benefit arising in limited-data settings. Experiments on the Ausgrid smart-meter dataset show consistent improvements over classical SCM and strong benchmark methods, with the dominant performance gains driven by dynamic augmentation.",
      "published": "2026-04-20T16:21:33Z",
      "abstract_url": "http://arxiv.org/abs/2604.18469v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18469v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation",
      "authors": [
        "Tianshi Cao",
        "Jiawei Ren",
        "Yuxuan Zhang",
        "Jaewoo Seo",
        "Jiahui Huang",
        "Shikhar Solanki",
        "Haotian Zhang",
        "Mingfei Guo",
        "Haithem Turki",
        "Muxingzi Li",
        "Yue Zhu",
        "Sipeng Zhang",
        "Zan Gojcic",
        "Sanja Fidler",
        "Kangxue Yin"
      ],
      "abstract": "Closed-loop simulation is a core component of autonomous vehicle (AV) development, enabling scalable testing, training, and safety validation before real-world deployment. Neural scene reconstruction converts driving logs into interactive 3D environments for simulation, but it does not produce complete 3D object assets required for agent manipulation and large-viewpoint novel-view synthesis. To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild object observations from real driving logs into complete, simulation-ready assets. Rather than relying on a single model component, we developed a system-level design for real-world AV data that combines large-scale curation of object-centric training tuples, geometry-aware preprocessing across heterogeneous sensors, and a robust training recipe that couples sparse-view-conditioned multiview generation with 3D Gaussian lifting. Within this system, SparseViewDiT is explicitly designed to address limited-angle views and other real-world data challenges. Together with hybrid data curation, augmentation, and self-distillation, this system enables scalable conversion of sparse AV object observations into reusable 3D assets.",
      "published": "2026-04-20T16:20:57Z",
      "abstract_url": "http://arxiv.org/abs/2604.18468v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18468v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG"
      ]
    },
    {
      "title": "An Integrated Deep-Learning Framework for Peptide-Protein Interaction Prediction and Target-Conditioned Peptide Generation with ConGA-PePPI and TC-PepGen",
      "authors": [
        "Chupei Tang",
        "Junxiao Kong",
        "Moyu Tang",
        "Di Wang",
        "Jixiu Zhai",
        "Ronghao Xie",
        "Shangkun Sima",
        "Tianchi Lu"
      ],
      "abstract": "Motivation: Peptide-protein interactions (PepPIs) are central to cellular regulation and peptide therapeutics, but experimental characterization remains too slow for large-scale screening. Existing methods usually emphasize either interaction prediction or peptide generation, leaving candidate prioritization, residue-level interpretation, and target-conditioned expansion insufficiently integrated. Results: We present an integrated framework for early-stage peptide screening that combines a partner-aware prediction and localization model (ConGA-PepPI) with a target-conditioned generative model (TC-PepGen). ConGA-PepPI uses asymmetric encoding, bidirectional cross-attention, and progressive transfer from pair prediction to binding-site localization, while TC-PepGen preserves target information throughout autoregressive decoding via layerwise conditioning. In five-fold cross-validation, ConGA-PepPI achieved 0.839 accuracy and 0.921 AUROC, with binding-site AUPR values of 0.601 on the protein side and 0.950 on the peptide side, and remained competitive on external benchmarks. Under a controlled length-conditioned benchmark, 40.39% of TC-PepGen peptides exceeded native templates in AlphaFold 3 ipTM, and unconstrained generation retained evidence of target-conditioned signal.",
      "published": "2026-04-20T16:20:23Z",
      "abstract_url": "http://arxiv.org/abs/2604.18467v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18467v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Using large language models for embodied planning introduces systematic safety risks",
      "authors": [
        "Tao Zhang",
        "Kaixian Qu",
        "Zhibin Li",
        "Jiajun Wu",
        "Marco Hutter",
        "Manling Li",
        "Fan Shi"
      ],
      "abstract": "Large language models are increasingly used as planners for robotic systems, yet how safely they plan remains an open question. To evaluate safe planning systematically, we introduce DESPITE, a benchmark of 12,279 tasks spanning physical and normative dangers with fully deterministic validation. Across 23 models, even near-perfect planning ability does not ensure safety: the best-planning model fails to produce a valid plan on only 0.4% of tasks but produces dangerous plans on 28.3%. Among 18 open-source models from 3B to 671B parameters, planning ability improves substantially with scale (0.4-99.3%) while safety awareness remains relatively flat (38-57%). We identify a multiplicative relationship between these two capacities, showing that larger models complete more tasks safely primarily through improved planning, not through better danger avoidance. Three proprietary reasoning models reach notably higher safety awareness (71-81%), while non-reasoning proprietary models and open-source reasoning models remain below 57%. As planning ability approaches saturation for frontier models, improving safety awareness becomes a central challenge for deploying language-model planners in robotic systems.",
      "published": "2026-04-20T16:18:08Z",
      "abstract_url": "http://arxiv.org/abs/2604.18463v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18463v1",
      "categories": [
        "cs.AI",
        "cs.LG",
        "cs.RO"
      ]
    },
    {
      "title": "ProtoCLIP: Prototype-Aligned Latent Refinement for Robust Zero-Shot Chest X-Ray Classification",
      "authors": [
        "Florian Kittler",
        "Sheethal Bhat",
        "Andreas Maier"
      ],
      "abstract": "Zero-shot vision-language models (VLMs) have shown promise for chest radiograph classification, but their performance is often limited by confounding label co-occurrence, long-tail class imbalance, and transfer instability under domain shift. We propose ProtoCLIP, a refinement strategy for CLIP-style VLMs that improves zero-shot discrimination through targeted data curation and distilled anchor alignment. Specifically, we construct pathology-focused training subsets with curated negative samples to reduce co-occurrence bias. We also introduce a representation-preserving distillation objective to stabilize adaptation while maintaining semantic structure and improving discrimination of clinically relevant co-occurring pathologies. Evaluated on an unseen dataset VinDr-CXR, ProtoCLIP improves AUC by 2-10 percentage points over a strong CLIP-based baseline across multiple findings. For pneumothorax specifically, ProtoCLIP achieves a state-of-the-art AUC of 0.94. These results demonstrate that anchor-guided refinement, coupled with curated supervision and controlled adaptation, can mitigate common zero-shot transfer failures in medical VLMs without requiring large-scale retraining.",
      "published": "2026-04-20T16:01:44Z",
      "abstract_url": "http://arxiv.org/abs/2604.18444v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18444v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "Six Llamas: Comparative Religious Ethics Through LoRA-Adapted Language Models",
      "authors": [
        "Chad Coleman",
        "W. Russell Neuman",
        "Manan Shah",
        "Ali Dasdan",
        "Matthew Crispi",
        "Morris Chiang",
        "Zack Leitman",
        "Mustafa Poonawala"
      ],
      "abstract": "We present Six Llamas, a comparative study examining whether large language models fine-tuned on distinct religious corpora encode systematically different patterns of ethical reasoning. Six variants of Meta-Llama-3.1-8B are constructed: one unmodified control and five LoRA-adapted models trained exclusively on the sacred and theological texts of Christianity, Islam, Judaism, Hinduism, or Buddhism. All six models are probed with an identical battery of 17 standardized ethical prompts spanning moral dilemmas, game-theoretic scenarios, public policy questions, and moral-psychological self-assessments. To assess robustness and reproducibility, we implement a multi-temperature sampling design spanning ten temperature settings. We compute response consistency metrics, pairwise inter-model agreement rates, temperature sensitivity coefficients across four prompt domains, and run-to-run stability analyses. Findings show that LoRA-adapted models produce ethical reasoning patterns that are (a) systematically differentiated from the base model, (b) consistent with the moral logics of their training traditions, (c) structured along interpretable dimensions in moral-philosophical space, (d) core ethical positions remain stable across temperature variations for high-consensus dilemmas. The Trolley Problem achieves 100% consistency across all models and temperatures, while (e) tradition-specific divergence intensifies at higher temperatures in morally contested domains, and (f) the base model exhibits the highest overall response consistency (mean 88.3%), suggesting LoRA adaptation introduces both tradition-specific signal and increased sampling sensitivity. The study offers a proof-of-concept for the condensate comparative method using differentially trained language models as instruments for cultural and ethical analysis and identifies specific criteria for falsification and planned extensions.",
      "published": "2026-04-20T15:22:59Z",
      "abstract_url": "http://arxiv.org/abs/2604.18404v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18404v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Randomly Initialized Networks Can Learn from Peer-to-Peer Consensus",
      "authors": [
        "Esteban Rodríguez-Betancourt",
        "Edgar Casasola-Murillo"
      ],
      "abstract": "In self-supervised learning, self-distilled methods have shown impressive performance, learning representations useful for downstream tasks and even displaying emergent properties. However, state-of-the-art methods usually rely on ensembles of complex mechanisms, with many design choices that are empirically motivated and not well understood. In this work, we explore the role of self-distillation within learning dynamics. Specifically, we isolate the effect of self-distillation by training a group of randomly initialized networks, removing all other common components such as projectors, predictors, and even pretext tasks. Our findings show that even this minimal setup can lead to learned representations with non-trivial improvements over a random baseline on downstream tasks. We also demonstrate how this effect varies with different hyperparameters and present a short analysis of what is being learned by the models under this setup.",
      "published": "2026-04-20T15:13:39Z",
      "abstract_url": "http://arxiv.org/abs/2604.18390v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18390v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes",
      "authors": [
        "Justin Bauer",
        "Thomas Walshe",
        "Derek Pham",
        "Harit Vishwakarma",
        "Armin Parchami",
        "Frederic Sala",
        "Paroma Varma"
      ],
      "abstract": "Fine-tuning Large Language Models (LLMs) typically relies on large quantities of high-quality annotated data, or questions with well-defined ground truth answers in the case of Reinforcement Learning with Verifiable Rewards (RLVR). While previous work has explored the benefits to model reasoning capabilities by scaling both data and compute used for RLVR, these results lack applicability in many real-world settings where annotated data and accessible compute may be scarce. In this work, we present a comprehensive empirical study of open-source Small Language Model (SLM) performance after RLVR in low data regimes. Across three novel datasets covering number counting problems, graph reasoning, and spatial reasoning, we characterize how model performance scales with dataset size, diversity, and complexity. We demonstrate that (1) procedural datasets allow for fine-grained evaluation and training dataset development with controllable properties (size, diversity, and complexity), (2) under RLVR, models trained on lower complexity tasks can generalize to higher complexity tasks, and (3) training on mixed complexity datasets is associated with the greatest benefits in low data regimes, providing up to 5x sample efficiency versus training on easy tasks. These findings inspire future work on the development of data scaling laws for RLVR and the use of procedural data generators to further understand effective data development for efficient LLM fine-tuning.",
      "published": "2026-04-20T15:04:57Z",
      "abstract_url": "http://arxiv.org/abs/2604.18381v1",
      "pdf_url": "https://arxiv.org/pdf/2604.18381v1",
      "categories": [
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
