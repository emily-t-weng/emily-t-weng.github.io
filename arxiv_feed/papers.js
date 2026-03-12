const PAPERS_DATA = {
  "last_updated": "2026-03-12 02:45:48 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation",
      "authors": [
        "Tao Zhong",
        "Yixun Hu",
        "Dongzhe Zheng",
        "Aditya Sood",
        "Christine Allen-Blanchette"
      ],
      "abstract": "We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/",
      "published": "2026-03-11T17:59:42Z",
      "abstract_url": "http://arxiv.org/abs/2603.11045v1",
      "pdf_url": "https://arxiv.org/pdf/2603.11045v1",
      "categories": [
        "cs.LG",
        "cond-mat.mtrl-sci",
        "cs.AI",
        "cs.CV",
        "physics.ins-det"
      ]
    },
    {
      "title": "V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation",
      "authors": [
        "Yan-Bo Lin",
        "Jonah Casebeer",
        "Long Mai",
        "Aniruddha Mahapatra",
        "Gedas Bertasius",
        "Nicholas J. Bryan"
      ],
      "abstract": "Generating music that temporally aligns with video events is challenging for existing text-to-music models, which lack fine-grained temporal control. We introduce V2M-Zero, a zero-pair video-to-music generation approach that outputs time-aligned music for video. Our method is motivated by a key observation: temporal synchronization requires matching when and how much change occurs, not what changes. While musical and visual events differ semantically, they exhibit shared temporal structure that can be captured independently within each modality. We capture this structure through event curves computed from intra-modal similarity using pretrained music and video encoders. By measuring temporal change within each modality independently, these curves provide comparable representations across modalities. This enables a simple training strategy: fine-tune a text-to-music model on music-event curves, then substitute video-event curves at inference without cross-modal training or paired data. Across OES-Pub, MovieGenBench-Music, and AIST++, V2M-Zero achieves substantial gains over paired-data baselines: 5-21% higher audio quality, 13-15% better semantic alignment, 21-52% improved temporal synchronization, and 28% higher beat alignment on dance videos. We find similar results via a large crowd-source subjective listening test. Overall, our results validate that temporal alignment through within-modality features, rather than paired cross-modal supervision, is effective for video-to-music generation. Results are available at https://genjib.github.io/v2m_zero/",
      "published": "2026-03-11T17:59:40Z",
      "abstract_url": "http://arxiv.org/abs/2603.11042v1",
      "pdf_url": "https://arxiv.org/pdf/2603.11042v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG",
        "cs.MM",
        "cs.SD"
      ]
    },
    {
      "title": "Artificial Intelligence as a Catalyst for Innovation in Software Engineering",
      "authors": [
        "Carlos Alberto Fernández-y-Fernández",
        "Jorge R. Aguilar-Cisneros"
      ],
      "abstract": "The rapid evolution and inherent complexity of modern software requirements demand highly flexible and responsive development methodologies. While Agile frameworks have become the industry standard for prioritizing iteration, collaboration, and adaptability, software development teams continue to face persistent challenges in managing constantly evolving requirements and maintaining product quality under tight deadlines. This article explores the intersection of Artificial Intelligence (AI) and Software Engineering (SE), to analyze how AI serves as a powerful catalyst for enhancing agility and fostering innovation. The research combines a comprehensive review of existing literature with an empirical study, utilizing a survey directed at Software Engineering professionals to assess the perception, adoption, and impact of AI-driven tools. Key findings reveal that the integration of AI (specifically through Machine Learning (ML) and Natural Language Processing (NLP) )facilitates the automation of tedious tasks, from requirement management to code generation and testing . This paper demonstrates that AI not only optimizes current Agile practices but also introduces new capabilities essential for sustaining quality, speed, and innovation in the future landscape of software development.",
      "published": "2026-03-11T17:20:30Z",
      "abstract_url": "http://arxiv.org/abs/2603.10994v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10994v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control",
      "authors": [
        "Yaswanth Chittepu",
        "Ativ Joshi",
        "Rajarshi Bhattacharjee",
        "Scott Niekum"
      ],
      "abstract": "Safe Reinforcement Learning from Human Feedback (RLHF) typically enforces safety through expected cost constraints, but the expectation captures only a single statistic of the cost distribution and fails to account for distributional uncertainty, particularly under heavy tails or rare catastrophic events. This limitation is problematic when robustness and risk sensitivity are critical. Stochastic dominance offers a principled alternative by comparing entire cost distributions rather than just their averages, enabling direct control over tail risks and potential out-of-distribution failures that expectation-based constraints may overlook. In this work, we propose Risk-sensitive Alignment via Dominance (RAD), a novel alignment framework that replaces scalar expected cost constraints with First-Order Stochastic Dominance (FSD) constraints. We operationalize this constraint by comparing the target policy's cost distribution to that of a reference policy within an Optimal Transport (OT) framework, using entropic regularization and Sinkhorn iterations to obtain a differentiable and computationally efficient objective for stable end-to-end optimization. Furthermore, we introduce quantile-weighted FSD constraints and show that weighted FSD universally controls a broad class of Spectral Risk Measures (SRMs), so that improvements under weighted dominance imply guaranteed improvements in the corresponding spectral risk. This provides a principled mechanism for tuning a model's risk profile via the quantile weighting function. Empirical results demonstrate that RAD improves harmlessness over baselines while remaining competitive in helpfulness, and exhibits greater robustness on out-of-distribution harmlessness evaluations.",
      "published": "2026-03-11T16:24:20Z",
      "abstract_url": "http://arxiv.org/abs/2603.10938v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10938v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Historical Consensus: Preventing Posterior Collapse via Iterative Selection of Gaussian Mixture Priors",
      "authors": [
        "Zegu Zhang",
        "Jian Zhang"
      ],
      "abstract": "Variational autoencoders (VAEs) frequently suffer from posterior collapse, where latent variables become uninformative and the approximate posterior degenerates to the prior. Recent work has characterized this phenomenon as a phase transition governed by the spectral properties of the data covariance matrix. In this paper, we propose a fundamentally different approach: instead of avoiding collapse through architectural constraints or hyperparameter tuning, we eliminate the possibility of collapse altogether by leveraging the multiplicity of Gaussian mixture model (GMM) clusterings. We introduce Historical Consensus Training, an iterative selection procedure that progressively refines a set of candidate GMM priors through alternating optimization and selection. The key insight is that models trained to satisfy multiple distinct clustering constraints develop a historical barrier -- a region in parameter space that remains stable even when subsequently trained with a single objective. We prove that this barrier excludes the collapsed solution, and demonstrate through extensive experiments on synthetic and real-world datasets that our method achieves non-collapsed representations regardless of decoder variance or regularization strength. Our approach requires no explicit stability conditions (e.g., $σ^{\\prime 2} < λ_{\\max}$) and works with arbitrary neural architectures. The code is available at https://github.com/tsegoochang/historical-consensus-vae.",
      "published": "2026-03-11T16:19:07Z",
      "abstract_url": "http://arxiv.org/abs/2603.10935v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10935v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CV"
      ]
    },
    {
      "title": "When Fine-Tuning Fails and when it Generalises: Role of Data Diversity and Mixed Training in LLM-based TTS",
      "authors": [
        "Anupam Purwar",
        "Aditya Choudhary"
      ],
      "abstract": "Large language models are increasingly adopted as semantic backbones for neural text-to-speech systems. However, frozen LLM representations are insufficient for modeling speaker specific acoustic and perceptual characteristics. Our experiments involving fine tuning of the Language Model backbone of TTS show promise in improving the voice consistency and Signal to Noise ratio SNR in voice cloning task. Across multiple speakers LoRA finetuning consistently outperforms the non-finetuned base Qwen-0.5B model across three complementary dimensions of speech quality. First, perceptual quality improves significantly with DNS-MOS gains of up to 0.42 points for speakers whose training data exhibits sufficient acoustic variability. Second, speaker fidelity improves for all evaluated speakers with consistent increases in voice similarity indicating that LoRA effectively adapts speaker identity representations without degrading linguistic modeling. Third, signal level quality improves in most cases with signal to noise ratio increasing by as much as 34 percent. Crucially these improvements are strongly governed by the characteristics of the training data. Speakers with high variability in acoustic energy and perceptual quality achieve simultaneous gains in DNS-MOS voice similarity and SNR. Overall this work establishes that LoRA finetuning is not merely a parameter efficient optimization technique but an effective mechanism for better speaker level adaptation in compact LLM-based TTS systems. When supported by sufficiently diverse training data LoRA adapted Qwen-0.5B consistently surpasses its frozen base model in perceptual quality speaker similarity with low latency using GGUF model hosted in quantized form.",
      "published": "2026-03-11T15:48:11Z",
      "abstract_url": "http://arxiv.org/abs/2603.10904v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10904v1",
      "categories": [
        "cs.SD",
        "cs.AI",
        "cs.ET"
      ]
    },
    {
      "title": "LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation",
      "authors": [
        "Jinwoo Ahn",
        "Ingyu Seong",
        "Akhil Kedia",
        "Junhan Kim",
        "Hyemi Jang",
        "Kangwook Lee",
        "Yongkweon Jeon"
      ],
      "abstract": "Transformer-based large language models (LLMs) rely on key-value (KV) caching to avoid redundant computation during autoregressive inference. While this mechanism greatly improves efficiency, the cache size grows linearly with the input sequence length, quickly becoming a bottleneck for long-context tasks. Existing solutions mitigate this problem by evicting prompt KV that are deemed unimportant, guided by estimated importance scores. Notably, a recent line of work proposes to improve eviction quality by \"glimpsing into the future\", in which a draft generator produces a surrogate future response approximating the target model's true response, and this surrogate is subsequently used to estimate the importance of cached KV more accurately. However, these approaches rely on computationally expensive draft generation, which introduces substantial prefilling overhead and limits their practicality in real-world deployment. To address this challenge, we propose LookaheadKV, a lightweight eviction framework that leverages the strength of surrogate future response without requiring explicit draft generation. LookaheadKV augments transformer layers with parameter-efficient modules trained to predict true importance scores with high accuracy. Our design ensures negligible runtime overhead comparable to existing inexpensive heuristics, while achieving accuracy superior to more costly approximation methods. Extensive experiments on long-context understanding benchmarks, across a wide range of models, demonstrate that our method not only outperforms recent competitive baselines in various long-context understanding tasks, but also reduces the eviction cost by up to 14.5x, leading to significantly faster time-to-first-token. Our code is available at https://github.com/SamsungLabs/LookaheadKV.",
      "published": "2026-03-11T15:44:32Z",
      "abstract_url": "http://arxiv.org/abs/2603.10899v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10899v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification",
      "authors": [
        "Yichi Zhu",
        "Kan Ling",
        "Xu Liu",
        "Hengrun Zhang",
        "Huiqun Yu",
        "Guisheng Fan"
      ],
      "abstract": "Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification.",
      "published": "2026-03-11T15:35:55Z",
      "abstract_url": "http://arxiv.org/abs/2603.10891v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10891v1",
      "categories": [
        "cs.AI",
        "cs.IR"
      ]
    },
    {
      "title": "Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models",
      "authors": [
        "Yixiu Mao",
        "Yun Qu",
        "Qi Wang",
        "Heming Zou",
        "Xiangyang Ji"
      ],
      "abstract": "Reinforcement learning (RL) finetuning has become a key technique for enhancing the reasoning abilities of large language models (LLMs). However, its effectiveness critically depends on the selection of training data. Recent advances underscore the importance of online prompt selection methods, which typically concentrate training on partially solved or moderately challenging examples under the current policy, thereby yielding more effective model updates. While significantly accelerating RL finetuning in terms of training steps, they also incur substantial computational overhead by requiring extensive LLM rollouts over large candidate batches to identify informative samples, an expense that can outweigh the finetuning process itself. To address this challenge, this work proposes Dynamics-Predictive Sampling (DPS), which online predicts and selects informative prompts by inferring their learning dynamics prior to costly rollouts. Specifically, we introduce a new perspective by modeling each prompt's solving progress during RL finetuning as a dynamical system, where the extent of solving is represented as the state and the transition is characterized by a hidden Markov model. Using historical rollout reward signals, we perform online Bayesian inference to estimate evolving state distributions, and the inference outcome provides a predictive prior for efficient prompt selection without rollout-intensive filtering. Empirical results across diverse reasoning tasks, including mathematics, planning, and visual geometry, demonstrate that DPS substantially reduces redundant rollouts, accelerates the training process, and achieves superior reasoning performance.",
      "published": "2026-03-11T15:31:14Z",
      "abstract_url": "http://arxiv.org/abs/2603.10887v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10887v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Continuous Diffusion Transformers for Designing Synthetic Regulatory Elements",
      "authors": [
        "Jonathan Liu",
        "Kia Ghods"
      ],
      "abstract": "We present a parameter-efficient Diffusion Transformer (DiT) for generating 200bp cell-type-specific regulatory DNA sequences. By replacing the U-Net backbone of DNA-Diffusion with a transformer denoiser equipped with a 2D CNN input encoder, our model matches the U-Net's best validation loss in 13 epochs (60$\\times$ fewer) and converges 39% lower, while reducing memorization from 5.3% to 1.7% of generated sequences aligning to training data via BLAT. Ablations show the CNN encoder is essential: without it, validation loss increases 70% regardless of positional embedding choice. We further apply DDPO finetuning using Enformer as a reward model, achieving a 38$\\times$ improvement in predicted regulatory activity. Cross-validation against DRAKES on an independent prediction task confirms that improvements reflect genuine regulatory signal rather than reward model overfitting.",
      "published": "2026-03-11T15:30:38Z",
      "abstract_url": "http://arxiv.org/abs/2603.10885v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10885v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "q-bio.GN"
      ]
    },
    {
      "title": "$V_{0.5}$: Generalist Value Model as a Prior for Sparse RL Rollouts",
      "authors": [
        "Yi-Kai Zhang",
        "Yueqing Sun",
        "Hongyan Hao",
        "Qi Gu",
        "Xunliang Cai",
        "De-Chuan Zhan",
        "Han-Jia Ye"
      ],
      "abstract": "In Reinforcement Learning with Verifiable Rewards (RLVR), constructing a robust advantage baseline is critical for policy gradients, effectively guiding the policy model to reinforce desired behaviors. Recent research has introduced Generalist Value Models (such as $V_0$), which achieve pre-trained value estimation by explicitly encoding model capabilities in-context, eliminating the need to synchronously update the value model alongside the policy model. In this paper, we propose $V_{0.5}$, which adaptively fuses the baseline predicted by such value model (acting as a prior) with the empirical mean derived from sparse rollouts. This constructs a robust baseline that balances computational efficiency with extremely low variance. Specifically, we introduce a real-time statistical testing and dynamic budget allocation. This balances the high variance caused by sparse sampling against the systematic bias (or hallucinations) inherent in the value model's prior. By constructing a hypothesis test to evaluate the prior's reliability in real-time, the system dynamically allocates additional rollout budget on demand. This mechanism minimizes the baseline estimator's Mean Squared Error (MSE), guaranteeing stable policy gradients, even under extreme sparsity with a group size of 4. Extensive evaluations across six mathematical reasoning benchmarks demonstrate that $V_{0.5}$ significantly outperforms GRPO and DAPO, achieving faster convergence and over some 10% performance improvement.",
      "published": "2026-03-11T14:57:41Z",
      "abstract_url": "http://arxiv.org/abs/2603.10848v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10848v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis",
      "authors": [
        "Yujie Zheng",
        "Zhuo Li",
        "Shengtao Zhang",
        "Hanjing Wang",
        "Junjie Sheng",
        "Jiaqian Wang",
        "Junchi Yan",
        "Weinan Zhang",
        "Ying Wen",
        "Bo Tang",
        "Muning Wen"
      ],
      "abstract": "Deploying Large Language Models to data-scarce programming domains poses significant challenges, particularly for kernel synthesis on emerging Domain-Specific Architectures where a \"Data Wall\" limits available training data. While models excel on data-rich platforms like CUDA, they suffer catastrophic performance drops on data-scarce ecosystems such as NPU programming. To overcome this cold-start barrier without expensive fine-tuning, we introduce EvoKernel, a self-evolving agentic framework that automates the lifecycle of kernel synthesis from initial drafting to continual refining. EvoKernel addresses this by formulating the synthesis process as a memory-based reinforcement learning task. Through a novel value-driven retrieval mechanism, it learns stage-specific Q-values that prioritize experiences based on their contribution to the current objective, whether bootstrapping a feasible draft or iteratively refining latency. Furthermore, by enabling cross-task memory sharing, the agent generalizes insights from simple to complex operators. By building an NPU variant of KernelBench and evaluating on it, EvoKernel improves frontier models' correctness from 11.0% to 83.0% and achieves a median speedup of 3.60x over initial drafts through iterative refinement. This demonstrates that value-guided experience accumulation allows general-purpose models to master the kernel synthesis task on niche hardware ecosystems. Our official page is available at https://evokernel.zhuo.li.",
      "published": "2026-03-11T14:57:06Z",
      "abstract_url": "http://arxiv.org/abs/2603.10846v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10846v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "Speaker Verification with Speech-Aware LLMs: Evaluation and Augmentation",
      "authors": [
        "Thomas Thebaud",
        "Yuzhe Wang",
        "Laureano Moro-Velazquez",
        "Jesus Villalba-Lopez",
        "Najim Dehak"
      ],
      "abstract": "Speech-aware large language models (LLMs) can accept speech inputs, yet their training objectives largely emphasize linguistic content or specific fields such as emotions or the speaker's gender, leaving it unclear whether they encode speaker identity. First, we propose a model-agnostic scoring protocol that produces continuous verification scores for both API-only and open-weight models, using confidence scores or log-likelihood ratios from the Yes/No token probabilities. Using this protocol, we benchmark recent speech-aware LLMs and observe weak speaker discrimination (EERs above 20% on VoxCeleb1). Second, we introduce a lightweight augmentation that equips an LLM with ASV capability by injecting frozen ECAPA-TDNN speaker embeddings through a learned projection and training only LoRA adapters. On TinyLLaMA-1.1B, the resulting ECAPA-LLM achieves 1.03% EER on VoxCeleb1-E, approaching a dedicated speaker verification system while preserving a natural-language interface.",
      "published": "2026-03-11T14:34:25Z",
      "abstract_url": "http://arxiv.org/abs/2603.10827v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10827v1",
      "categories": [
        "cs.SD",
        "cs.AI"
      ]
    },
    {
      "title": "Protein Counterfactuals via Diffusion-Guided Latent Optimization",
      "authors": [
        "Weronika Kłos",
        "Sidney Bender",
        "Lukas Kades"
      ],
      "abstract": "Deep learning models can predict protein properties with unprecedented accuracy but rarely offer mechanistic insight or actionable guidance for engineering improved variants. When a model flags an antibody as unstable, the protein engineer is left without recourse: which mutations would rescue stability while preserving function? We introduce Manifold-Constrained Counterfactual Optimization for Proteins (MCCOP), a framework that computes minimal, biologically plausible sequence edits that flip a model's prediction to a desired target state. MCCOP operates in a continuous joint sequence-structure latent space and employs a pretrained diffusion model as a manifold prior, balancing three objectives: validity (achieving the target property), proximity (minimizing mutations), and plausibility (producing foldable proteins). We evaluate MCCOP on three protein engineering tasks - GFP fluorescence rescue, thermodynamic stability enhancement, and E3 ligase activity recovery - and show that it generates sparser, more plausible counterfactuals than both discrete and continuous baselines. The recovered mutations align with known biophysical mechanisms, including chromophore packing and hydrophobic core consolidation, establishing MCCOP as a tool for both model interpretation and hypothesis-driven protein design. Our code is publicly available at github.com/weroks/mccop.",
      "published": "2026-03-11T14:19:52Z",
      "abstract_url": "http://arxiv.org/abs/2603.10811v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10811v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Nurture-First Agent Development: Building Domain-Expert AI Agents Through Conversational Knowledge Crystallization",
      "authors": [
        "Linghao Zhang"
      ],
      "abstract": "The emergence of large language model (LLM)-based agent frameworks has shifted the primary challenge in building domain-expert AI agents from raw capability to effective encoding of domain expertise. Two dominant paradigms -- code-first development, which embeds expertise in deterministic pipelines, and prompt-first development, which captures expertise in static system prompts -- both treat agent construction as a discrete engineering phase preceding deployment. We argue that this sequential assumption creates a fundamental mismatch with the nature of domain expertise, which is substantially tacit, deeply personal, and continuously evolving. We propose Nurture-First Development (NFD), a paradigm in which agents are initialized with minimal scaffolding and progressively grown through structured conversational interaction with domain practitioners. The central mechanism is the Knowledge Crystallization Cycle, whereby fragmented knowledge embedded in operational dialogue is periodically consolidated into structured, reusable knowledge assets. We formalize NFD through: (1) a Three-Layer Cognitive Architecture organizing agent knowledge by volatility and personalization degree; (2) the Knowledge Crystallization Cycle with formal definitions of crystallization operations and efficiency metrics; and (3) an operational framework comprising a Dual-Workspace Pattern and Spiral Development Model. We illustrate the paradigm through a detailed case study on building a financial research agent for U.S. equity analysis and discuss the conditions, limitations, and broader implications of NFD for human-agent co-evolution.",
      "published": "2026-03-11T14:14:53Z",
      "abstract_url": "http://arxiv.org/abs/2603.10808v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10808v1",
      "categories": [
        "cs.AI",
        "cs.HC",
        "cs.SE"
      ]
    },
    {
      "title": "Risk-Adjusted Harm Scoring for Automated Red Teaming for LLMs in Financial Services",
      "authors": [
        "Fabrizio Dimino",
        "Bhaskarjit Sarmah",
        "Stefano Pasquali"
      ],
      "abstract": "The rapid adoption of large language models (LLMs) in financial services introduces new operational, regulatory, and security risks. Yet most red-teaming benchmarks remain domain-agnostic and fail to capture failure modes specific to regulated BFSI settings, where harmful behavior can be elicited through legally or professionally plausible framing. We propose a risk-aware evaluation framework for LLM security failures in Banking, Financial Services, and Insurance (BFSI), combining a domain-specific taxonomy of financial harms, an automated multi-round red-teaming pipeline, and an ensemble-based judging protocol. We introduce the Risk-Adjusted Harm Score (RAHS), a risk-sensitive metric that goes beyond success rates by quantifying the operational severity of disclosures, accounting for mitigation signals, and leveraging inter-judge agreement. Across diverse models, we find that higher decoding stochasticity and sustained adaptive interaction not only increase jailbreak success, but also drive systematic escalation toward more severe and operationally actionable financial disclosures. These results expose limitations of single-turn, domain-agnostic security evaluation and motivate risk-sensitive assessment under prolonged adversarial pressure for real-world BFSI deployment.",
      "published": "2026-03-11T14:14:13Z",
      "abstract_url": "http://arxiv.org/abs/2603.10807v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10807v1",
      "categories": [
        "q-fin.CP",
        "cs.AI",
        "cs.CY"
      ]
    },
    {
      "title": "Towards Intelligent Spectrum Management: Spectrum Demand Estimation Using Graph Neural Networks",
      "authors": [
        "Mohamad Alkadamani",
        "Amir Ghasemi",
        "Halim Yanikomeroglu"
      ],
      "abstract": "The growing demand for wireless connectivity, combined with limited spectrum resources, calls for more efficient spectrum management. Spectrum sharing is a promising approach; however, regulators need accurate methods to characterize demand dynamics and guide allocation decisions. This paper builds and validates a spectrum demand proxy from public deployment records and uses a graph attention network in a hierarchical, multi-resolution setup (HR-GAT) to estimate spectrum demand at fine spatial scales. The model captures both neighborhood effects and cross-scale patterns, reducing spatial autocorrelation and improving generalization. Evaluated across five Canadian cities and against eight competitive baselines, HR-GAT reduces median RMSE by roughly 21% relative to the best alternative and lowers residual spatial bias. The resulting demand maps are regulator-accessible and support spectrum sharing and spectrum allocation in wireless networks.",
      "published": "2026-03-11T14:11:44Z",
      "abstract_url": "http://arxiv.org/abs/2603.10802v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10802v1",
      "categories": [
        "cs.NI",
        "cs.AI",
        "cs.LG",
        "eess.SY"
      ]
    },
    {
      "title": "AI-Enhanced Spatial Cellular Traffic Demand Prediction with Contextual Clustering and Error Correction for 5G/6G Planning",
      "authors": [
        "Mohamad Alkadamani",
        "Colin Brown",
        "Halim Yanikomeroglu"
      ],
      "abstract": "Accurate spatial prediction of cellular traffic demand is essential for 5G NR capacity planning, network densification, and data-driven 6G planning. Although machine learning can fuse heterogeneous geospatial and socio-economic layers to estimate fine-grained demand maps, spatial autocorrelation can cause neighborhood leakage under naive train/test splits, inflating accuracy and weakening planning reliability. This paper presents an AI-driven framework that reduces leakage and improves spatial generalization via a context-aware two-stage splitting strategy with residual spatial error correction. Experiments using crowdsourced usage indicators across five major Canadian cities show consistent mean absolute error (MAE) reductions relative to location-only clustering, supporting more reliable bandwidth provisioning and evidence-based spectrum planning and sharing assessments.",
      "published": "2026-03-11T14:11:37Z",
      "abstract_url": "http://arxiv.org/abs/2603.10800v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10800v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "eess.SY"
      ]
    },
    {
      "title": "Taking Shortcuts for Categorical VQA Using Super Neurons",
      "authors": [
        "Pierre Musacchio",
        "Jaeyi Jeong",
        "Dahun Kim",
        "Jaesik Park"
      ],
      "abstract": "Sparse Attention Vectors (SAVs) have emerged as an excellent training-free alternative to supervised finetuning or low-rank adaptation to improve the performance of Vision Language Models (VLMs). At their heart, SAVs select a few accurate attention heads for a task of interest and use them as classifiers, rather than relying on the model's prediction. In a similar spirit, we find that directly probing the raw activations of the VLM, in the form of scalar values, is sufficient to yield accurate classifiers on diverse visually grounded downstream tasks. Shifting focus from attention vectors to scalar activations dramatically increases the search space for accurate parameters, allowing us to find more discriminative neurons immediately from the first generated token. We call such activations Super Neurons (SNs). In this probing setting, we discover that enough SNs appear in the shallower layers of the large language model to allow for extreme early exiting from the first layer of the model at the first generated token. Compared to the original network, SNs robustly improve the classification performance while achieving a speedup of up to 5.10x.",
      "published": "2026-03-11T13:54:45Z",
      "abstract_url": "http://arxiv.org/abs/2603.10781v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10781v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "Deep Randomized Distributed Function Computation (DeepRDFC): Neural Distributed Channel Simulation",
      "authors": [
        "Didrik Bergström",
        "Onur Günlü"
      ],
      "abstract": "The randomized distributed function computation (RDFC) framework, which unifies many cutting-edge distributed computation and learning applications, is considered. An autoencoder (AE) architecture is proposed to minimize the total variation distance between the probability distribution simulated by the AE outputs and an unknown target distribution, using only data samples. We illustrate significantly high RDFC performance with communication load gains from our AEs compared to data compression methods. Our designs establish deep learning-based RDFC methods and aim to facilitate the use of RDFC methods, especially when the amount of common randomness is limited and strong function computation guarantees are required.",
      "published": "2026-03-11T13:24:49Z",
      "abstract_url": "http://arxiv.org/abs/2603.10750v1",
      "pdf_url": "https://arxiv.org/pdf/2603.10750v1",
      "categories": [
        "cs.IT",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
