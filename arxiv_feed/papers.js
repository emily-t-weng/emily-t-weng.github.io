const PAPERS_DATA = {
  "last_updated": "2026-10-29 05:26:38 UTC",
  "query": "cat:cs.AI AND (all:\"large language model\" OR all:\"machine learning\")",
  "papers": [
    {
      "title": "Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction",
      "authors": [
        "Prakhar Kaushik",
        "Soumava Paul",
        "Alan Yuille"
      ],
      "abstract": "Novel view synthesis and 3D reconstruction have advanced rapidly with radiance fields and splatting, yet their memory and computational demands limit their use in resource-constrained settings. We introduce Hexels: novel planar hexagonal surface elements with a first-order spatial color field for efficient 3D scene and 2D image representation. Hexels exploit locally planar regions with smoothly varying appearance using fewer, larger primitives. Their hexagonal support provides an area- and boundary-efficient domain, while the local color field captures appearance variation within each primitive. Across 34 scenes, Hexels improve PSNR over memory-matched 2DGS by 4.53-7.38 dB across datasets while requiring lower training VRAM. We further ablate Hexels by showing that per primitive, Hexels outperform 2DGS by upto 4.74 dB while using lower training VRAM and having 2.31 times faster FPS. Even in unconstrained settings, Hexels offers complementary quality-memory trade-offs. For efficient 3D surface reconstruction, Hexels reduce error by 28.9% compared with memory-matched 2DGS across 15 DTU scans. On 6 Tanks and Temples surface scenes, Hexels achieves the highest mean F1 among compared methods (34.38%, versus 31.72% for 2DGS). We further also show that Hexels, as a general primitive, improves 2D image representation over Image-GS. Hexels push the Pareto frontier on the quality-efficiency trade-off of surface-aligned primitives.",
      "published": "2026-10-27T17:59:59Z",
      "abstract_url": "http://arxiv.org/abs/2610.37668v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37668v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.GR",
        "cs.LG"
      ]
    },
    {
      "title": "Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops",
      "authors": [
        "Aakash Sarkar",
        "Ananya Mukherjee",
        "Motoshige Sato"
      ],
      "abstract": "Relations compose through the entity they share - Alice is Bob's mother and Bob is Carol's father, so Alice is Carol's grandmother - and the order matters, since a father's mother is not a mother's father; verifying a composed relation means closing the loop. We show that if each entity's coordinate frame may be rotated independently (per-entity frame freedom), three things follow. In the pure-gauge sector, relations that share no cycle cannot interact at all. The only gauge-invariant score of a triangle is a class function of its holonomy $H = T_{ij} T_{jk} T_{ki}$ - a Wilson loop. And the only equivariant update of an edge is $U = Ψ(H)\\, T_{ij}$ with $Ψ$ conjugation-equivariant. Edge attention by Wilson loops is therefore not one design among many but the unique form the symmetry permits. Three communities hold pieces of this form: the commuting-operator embeddings of knowledge graphs are its abelian (commuting) sector; AlphaFold's triangle update and the Edge Transformer have its shape - an edge updated through the third vertex - without its constraints; and lattice QCD builds it equivariantly on lattices. We instantiate the equivariant sub-family on general relational graphs as holonomy attention. With no training and composition alone, it infers unobserved $S_4$ relations on sparse graphs at 0.88 with a fifth of the edges corrupted, at or slightly below specialist synchronization algorithms, where every commutative architecture is capped at 0.083; its holonomy score flags corrupted triangles at AUC 0.99 where commuting scores are capped at 0.76 by a second selection rule, and it separates clusters without labels.",
      "published": "2026-10-27T17:58:48Z",
      "abstract_url": "http://arxiv.org/abs/2610.37656v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37656v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "physics.comp-ph"
      ]
    },
    {
      "title": "KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry",
      "authors": [
        "Wenqing Peng",
        "Michael Boy",
        "Zhi-Song Liu"
      ],
      "abstract": "Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the \\texttt{diffrax} library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of the 25 rate coefficients a flow-tube-like design can identify, and noisy synthetic estimation confirms the predicted uncertainties of the nine identifiable rate coefficients to within about 30\\% while revealing clusters of fast reactions of which only ratios are constrained. In the second, a neural-network correction of an emission inventory is trained offline on per-scenario 4D-Var retrievals and then fine-tuned online through the chemical model, which reduces the error of the biogenic correction by a factor of 2.4--5 and recovers information on low-temperature emissions that the individual retrievals lose. KPPax makes KPP mechanisms directly usable for parameter estimation and hybrid physics--machine-learning modelling.",
      "published": "2026-10-27T17:58:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.37655v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37655v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Can LLMs Uncover Hidden Physical Laws through Free Exploration?",
      "authors": [
        "Zhen Yang",
        "Wen Wang",
        "Xander Xu",
        "Ying-Cong Chen"
      ],
      "abstract": "Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explorers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run without human intervention. Among 12 LLMs, GPT-6-Astra achieves a discovery rate of 16.7%, while the other 11 score 0%. The exploration records reveal uninvestigated anomalies and only partial identification of the altered physical laws.",
      "published": "2026-10-27T17:57:00Z",
      "abstract_url": "http://arxiv.org/abs/2610.37649v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37649v1",
      "categories": [
        "cs.CL",
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control",
      "authors": [
        "Athanasios Glentis",
        "Dawei Li",
        "Mingyi Hong"
      ],
      "abstract": "Weight decay and explicit weight-norm constraints improve optimizer performance in language model pretraining, yet a unified understanding of why such control helps and how much is needed remains incomplete. We analyze norm control in Muon over the entire training trajectory using the \\emph{accumulated effective learning rate}, $C_T=\\sum_{t<T}η_t/\\|W_t\\|_F$, where $η_t$ is the nominal learning rate and $η_t/\\|W_t\\|_F$ is the effective learning rate relative to the weight norm. For bounded update directions, accumulation over any training interval bounds relative weight displacement. We show that vanilla Muon can accumulate only logarithmically, potentially limiting exploration and helping explain its slower empirical progress, whereas MuonW and MuonH guarantee at least linear accumulation under the stated conditions. Going beyond bounded-norm control, we show that sufficiently slow norm growth guarantees faster accumulation than the logarithmic worst case, opening up less restrictive optimizer designs. Based on this insight, we propose MuonGS, which uses Gram--Schmidt (GS) projection to remove the radial component of each Muon update without fixing the weight norm. With bounded update directions and a constant nominal learning rate, we prove that the weight norm grows at most proportionally to $\\sqrt{t}$ after $t$ steps. Consequently, MuonGS guarantees $C_T=Ω(\\sqrt{T})$, with accumulation sustained even in later training. Further, for smooth scale-invariant objectives we establish its $O(T^{-1/2})$ stationarity guarantee. In pretraining experiments with 160M- and 400M-parameter dense models as well as a 135M active- and 1B total- parameter MoE model, MuonGS achieves lower validation perplexity than MuonH at long training horizons. In particular, for the dense model it achieves an estimated token-equivalent speedup up to about $15\\%$ at matched validation loss to MuonH.",
      "published": "2026-10-27T17:56:40Z",
      "abstract_url": "http://arxiv.org/abs/2610.37647v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37647v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "math.OC"
      ]
    },
    {
      "title": "VideoLVR: Latent Visual Reasoning over Grounded Video Evidence",
      "authors": [
        "Shilong Li",
        "Xingyuan Bu",
        "Hao Shi",
        "Yun Xing",
        "Weixin Feng",
        "Qiangpeng Yang",
        "Shilei Wen"
      ],
      "abstract": "Chain-of-thought reasoning has substantially advanced video understanding in multimodal large language models (MLLMs). However, language-based reasoning traces do not provide a reliable mechanism for models to revisit relevant visual evidence, leading to visual forgetting and impairing their ability to reason over video content. Existing approaches maintain visual grounding during reasoning in two main ways: they either refer to evidence using textual coordinates, which specify symbolic locations rather than the visual content itself, or reinsert cropped regions or frames into the multimodal context, preserving visual details at the cost of substantial context expansion. To address these limitations, we introduce VideoLVR, which constructs latent visual evidence from local regions at relevant video moments, thereby providing visual support for intermediate reasoning steps. To supervise latent visual evidence generation, we introduce a visual token prediction objective that formulates visual token reconstruction as prediction over a vocabulary derived from the input video, thereby unifying visual evidence reconstruction and text generation within the same token prediction framework. This shared prediction framework enables MLLMs to reason over interleaved text tokens and latent visual states while integrating evidence from different moments. We further extend VideoLVR to reinforcement learning by treating sampled visual tokens as actions alongside text tokens and jointly optimizing the policy using final-answer rewards. By efficiently revisiting task-relevant visual information during reasoning, VideoLVR improves video understanding without substantially expanding the multimodal context. Experiments on multiple video benchmarks demonstrate its effectiveness, especially on tasks requiring fine-grained visual evidence.",
      "published": "2026-10-27T17:41:56Z",
      "abstract_url": "http://arxiv.org/abs/2610.37613v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37613v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting",
      "authors": [
        "Joseph Natter",
        "Long Vu",
        "Yifan Zhang",
        "Bobin Deng",
        "Kun Suo",
        "Rui Wu"
      ],
      "abstract": "Traffic forecasting requires modeling complex temporal dynamics and the spatial dependencies imposed by road topology. Supervised graph-based models achieve strong performance, but are often trained for a particular dataset and sensor graph, making deployment to new or evolving networks costly. Prior-Data Fitted Networks (PFNs) offer a zero-shot alternative by learning in-context inference from synthetic task priors, which provide an explicit mechanism for encoding domain-specific assumptions. Existing forecasting PFNs, however, do not explicitly represent the graph-structured dependencies governing traffic dynamics. To combine graph-aware traffic modeling with zero-shot adaptation, we introduce the Graph-Aware Spatiotemporal Prior-Data Fitted Network (GAS-PFN). GAS-PFN incorporates sampled road topology into its synthetic task prior through random-walk diffusion, teaching the model to learn graph-coupled dynamics during pretraining. Its co-designed inference architecture uses two mask-conditioned passes through a shared Transformer backbone to separately model sensor-specific temporal dynamics and spatiotemporal interactions among neighboring sensors before integrating their representations for prediction. This design enables GAS-PFN to use target-network structure and context without parameter updates. Across six benchmarks at the 90-minute horizon, GAS-PFN reduces MSE by 34.0% on average relative to TimePFN, its closest PFN-based predecessor. It also reduces MSE by 27.7% on average relative to four recent industry-developed general-purpose zero-shot forecasters. In this comparison, all models receive 90 minutes of contiguous recent history, and any additional context is drawn from each dataset's chronological training split. These results indicate that aligning PFN priors and inference architectures with graph structure improves in-context learning for zero-shot traffic forecasting.",
      "published": "2026-10-27T17:31:26Z",
      "abstract_url": "http://arxiv.org/abs/2610.37598v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37598v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.NE"
      ]
    },
    {
      "title": "Understanding the limits of latent world models",
      "authors": [
        "Rodrigo Gonzalez Laiz",
        "Steffen Schneider"
      ],
      "abstract": "World models are widely proposed as core components for enabling agents to reason over long temporal and spatial horizons. Yet current evaluations often emphasize task performance while providing limited insight into whether learned world models acquire predictive latent dynamics that reflect the underlying structure of the environment. Here we introduce the concept of faithful latent world models as an equivariance condition between the environment's transition map and the learned latent dynamics. We propose latent rollout consistency (LRC) as a metric to evaluate action-conditioned latent prediction. Strikingly, we find that across the studied state of the art latent world models, none solves the task by actually faithfully modeling the dynamics of the environment.",
      "published": "2026-10-27T17:25:50Z",
      "abstract_url": "http://arxiv.org/abs/2610.37588v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37588v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents",
      "authors": [
        "Pranjal Aggarwal",
        "Sean Welleck"
      ],
      "abstract": "AI agents have made substantial progress in domains where they have a non-trivial amount of time to deliberate between actions, such as mathematics, software engineering, and performing tasks on desktop applications. It is less clear how such models fare in settings that demand real-time interaction, such as playing a video game or operating a physical device. In these settings, the environment state may evolve as the agent is reasoning, or even as the underlying model is performing a forward pass. To gain insight into the real-time interaction capabilities of vision-language agents, we introduce Gym-Anything-RT. Our key idea is to construct visual minigames that demand a portfolio of capabilities required for interactive real-time settings. Gym-Anything-RT uses an agentic pipeline to automatically generate such environments based on a taxonomy of six capabilities required for real-time interaction, including the ability to perceive the environment and act under an evolving environment state. A key feature of Gym-Anything-RT is that it allows for controlled comparison of real-time interaction capabilities: we can run the same minigame in real time, or pause or slow down the environment to relax the requirement for real-time actions. We use Gym-Anything-RT to construct 100 real-time minigames and evaluate AI agents on them. We find large capability gaps in real-time interaction. For example, GPT-6 Astra (xHigh) achieves a pass rate of only 9.0% on the real-time minigames, which rises to 75.0% when the environment is paused. We also identify intriguing behaviors by the most powerful frontier agents: for instance, when given the ability to write code, these agents write code that bypasses the need to perceive and act in real time. We publicly release Gym-Anything-RT and the generated minigames to spur further research on AI agents in real-time interactive settings.",
      "published": "2026-10-27T17:13:05Z",
      "abstract_url": "http://arxiv.org/abs/2610.37567v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37567v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance",
      "authors": [
        "Walter Endrizzi",
        "Flavio Ragni",
        "Raffaele Marchesi",
        "Stefano Bovo",
        "Monica Moroni",
        "Giuseppe Jurman",
        "Venet Osmani"
      ],
      "abstract": "Background. Selective prediction systems improve the safety of clinical artificial intelligence by deferring ambiguous cases to human experts while automating high-confidence decisions. However, this safety mechanism relies entirely on the premise that uncertainty scores accurately reflect clinical ambiguity. In prognostic tasks characterized by severe class imbalance, where the clinical outcome of interest is a minority, this assumption breaks down. Standard uncertainty measures cease to capture clinically aligned ambiguity; instead, they become a proxy for class membership, systematically penalizing minority patients regardless of their individual predictive clarity. Results. Across 56 synthetic conditions and three neurodegenerative cohorts spanning severe class imbalance to near-balanced conditions (multiple sclerosis, Parkinson's Disease, Alzheimer's Disease), our framework consistently maintained bounded, balanced risk across trajectories. In contrast, global metrics either collapsed minority Sensitivity or protected the minority at the expense of majority patients. Applying CCR with standard entropy fails to protect the minority because it breaks within-class monotonicity when tau=0.5. Combining CCR with threshold-aware metrics restores strict monotonicity, correctly identifies ambiguous patients, and safely mitigates minority exclusion. Conclusions. Under clinical class imbalance, standard uncertainty quantification measures dataset sparsity rather than clinical ambiguity, creating a systemic illusion of safety. This bias is a correctable structural flaw. Clinical pipelines deploying selective prediction under imbalance should adopt threshold-aligned uncertainty measures and class-stratified evaluation. This reduces class-specific risk imbalance, supporting safer selective prediction under the evaluated conditions without silently excluding the most vulnerable patients.",
      "published": "2026-10-27T17:08:31Z",
      "abstract_url": "http://arxiv.org/abs/2610.37561v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37561v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis",
      "authors": [
        "Akshay Krishnan",
        "Matias Turkulainen",
        "Jamie Wynn",
        "Zawar Qureshi",
        "Mohamed Sayed",
        "Guillermo Garcia-Hernando",
        "Sara Vicente",
        "James Hays",
        "Michael Firman",
        "Daniyar Turmukhambetov"
      ],
      "abstract": "Multi-view or video diffusion models have recently emerged as state-of-the-art approaches for novel-view synthesis from sparse image collections. While they can produce high quality images, they tend to hallucinate unrealistic spatial layouts when synthesizing views outside the input views (scene extrapolation). We propose to address this challenge for outdoor scenes by using satellite images as an additional information source to condition the multi-view generative model. We introduce a new satellite-conditioned multi-view diffusion model that combines conditioning from a satellite image and street-level input images when generating novel-views. Our cross-attention based conditioning allows us to retain pretrained multi-view diffusion priors to generate seamless, realistic, novel views from a handful of source views, while also adhering to the scene layout as seen from satellite images. Our model does not use 3D scene reconstruction, and only relies on posed street and satellite views. Our experiments on outdoor datasets show that our model is better than both satellite-conditioned or street-view conditioned baselines on novel-view synthesis, especially on challenging cases where target views have very little overlap with source views. Project page: https://nianticspatial.github.io/groundcontrol",
      "published": "2026-10-27T16:56:45Z",
      "abstract_url": "http://arxiv.org/abs/2610.37545v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37545v1",
      "categories": [
        "cs.CV",
        "cs.AI",
        "cs.LG"
      ]
    },
    {
      "title": "The Structure Gap: When More Data Cannot Replace Structured Reasoning",
      "authors": [
        "Kheer Sagar Patel",
        "Mojtaba Soltanalian"
      ],
      "abstract": "A persistent belief in modern AI is that sufficiently large learners trained on sufficiently much data can make explicit domain knowledge optional. This paper examines a limitation of that belief. We define structure-encoded problems: tasks whose input-output map is determined, or sharply constrained, by known structure such as a solver, symmetry group, graph, physical law, measurement operator, grammar, combinatorial search space, or optimization objective. For such tasks, the central question is not expressivity--a dense neural network may contain the desired map--but finite-data identification, extrapolation, and certification. We introduce the structure gap: the finite-resource gap between a method that receives the governing structure and a learner that receives only examples. The strongest useful form of the claim is asymptotic rather than absolute: structure-agnostic learning may approach a structured solver as coverage grows, but every finite dataset can leave residual uncertainty that the structure resolves by construction. We prove this exactly for a Gaussian inverse problem: the known-operator Wiener solver attains Bayes risk, while the dense empirical learner in the correct linear class has persistent expected excess risk. We also prove a cell-coverage lower bound and an unobserved-orbit non-identifiability result. Controlled experiments in known-operator inversion, sparse recovery, and graph reachability with size extrapolation instantiate the mechanisms. When validated structure is available, it can constrain the hypothesis space before learning begins, leaving learning to address components the structure does not specify.",
      "published": "2026-10-27T16:45:22Z",
      "abstract_url": "http://arxiv.org/abs/2610.37523v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37523v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Orthogonal and Parametrized Multi-Attribute Steering of LLMs",
      "authors": [
        "Remy Ogasawara",
        "Cullen Anderson",
        "Narmeen Fatimah Oozeer",
        "Foad Namjoo",
        "Amir Abdullah",
        "Jens Kristian Refsgaard Schou",
        "Jeff M. Phillips"
      ],
      "abstract": "Activation steering offers an effective and efficient way to modify an LLM's behavior at inference time. However, it remains unclear how to apply steering such that the signal is consistently strong and the steering strength has a universally interpretable meaning. We address this first by identifying the most steerable interval for each steering vector and rescaling the steering range accordingly. Building on this, we introduce a more interpretable method for steering, called parameterized steering which instead of pushing activations in a direction, it sets them at the desired trait strength. Next, with multi-trait steering, we find that naively combining vectors causes them to interfere with one another. In response, we develop a method to orthogonalize the vectors; it decouples these traits, allowing a model to exhibit one behavior without inducing the others, thus enabling multiple traits to be steered simultaneously.",
      "published": "2026-10-27T16:39:53Z",
      "abstract_url": "http://arxiv.org/abs/2610.37517v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37517v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Follow the Gradient: Tracing Safety Degradation Back to Training Data",
      "authors": [
        "Saket Reddy",
        "Amro Abbas",
        "Anshuman Suri",
        "Ari Morcos",
        "David Schwab",
        "Pratyush Maini"
      ],
      "abstract": "Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned models. We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Training on SafeTulu reduces GCG attack success rate by up to 22% across Qwen, Llama, and Olmo, outperforming both LLM-as-a-judge and prior gradient-based filtering, while largely preserving or even improving capability. Finally, we analyzed the removed data and identify possibly under-studied sources of safety degradation, such as tool-execution failures and narrative arithmetic. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets.",
      "published": "2026-10-27T16:08:08Z",
      "abstract_url": "http://arxiv.org/abs/2610.37481v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37481v1",
      "categories": [
        "cs.LG",
        "cs.AI",
        "cs.CL"
      ]
    },
    {
      "title": "PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions",
      "authors": [
        "Tongkun Guan",
        "Zhibo Yang",
        "Mingkun Yang",
        "Haocheng Wang",
        "Wei Shen",
        "Xiaokang Yang"
      ],
      "abstract": "Fine-grained visual perception remains a bottleneck during post-training of multimodal large language models (MLLMs), as perceptual errors can constrain gains from reasoning-focused optimization. Although image-caption pre-training improves perception, its influence can weaken as subsequent stages change data mixtures and objectives. Continuing caption supervision during post-training is challenging because subjective, information-dense descriptions complicate verifiable reward design. We present PerceptRL, which strengthens perception at two complementary levels. Perception RL uses a multi-agent pipeline to extract verified errors from dense image captions and convert the corresponding visual facts into QA pairs with fixed answer labels and rule-based rewards. Chain QA links correlated errors, while multiple-choice QA combines unrelated errors. A specialized route targets high-error domains with tailored verification, and a domain-agnostic general route covers errors beyond these domains. Perception Agentic RL addresses fine-grained and dense cases unresolved by single-turn Perception RL through multi-turn inspection with visual tools. We observe coordinate drift even in models trained to use tools, where incorrect crops or annotations distort evidence and propagate errors. Coordinate Injection corrects spatial operations in cold-start trajectories, while a grounding-aware process reward supervises intermediate spatial fidelity alongside final-answer correctness. Training proceeds through shared cold-start SFT, single-turn Perception RL, and multi-turn Perception Agentic RL. Across seven benchmarks under a shared tool-use protocol, PerceptRL achieves 71.6% average accuracy, surpassing the base model with the same tools and the strongest evaluated frontier VLM by 15.7 and 5.2 percentage points, respectively.",
      "published": "2026-10-27T16:04:10Z",
      "abstract_url": "http://arxiv.org/abs/2610.37471v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37471v1",
      "categories": [
        "cs.CV",
        "cs.AI"
      ]
    },
    {
      "title": "Concordat: Typed Sessions for LLM Agents",
      "authors": [
        "Milad Sikaroudi"
      ],
      "abstract": "Agents built on large language models are increasingly allowed to take consequential actions: cancelling bookings, issuing refunds, changing accounts. The rules that govern those actions, such as \"obtain the customer's explicit confirmation before changing anything\", are usually stated in the prompt and followed only as reliably as the model chooses to follow them. We present Concordat, a Python library that treats an agent as an object whose interface is a finite-state protocol, in the spirit of typestate and session types, and enforces that protocol on the model at run time. On every turn the model is offered only the tools that are legal in the current protocol state, calls outside that set are rejected before they have side effects, decisions reserved for humans can only be requested, and token and tool-call budgets are charged and split linearly across handoffs. Protocols are checked when they are defined, and a subclass of an agent is rejected if it could bypass its parent's protocol, a behavioural form of Liskov substitution. We evaluate a Concordat agent against the standard agent of tau-bench on its 50 airline tasks, adding simulated customers who pressure the agent to skip confirmation. The standard agent made changes without the required \"yes\" in 13 of 50 tasks with gpt-4.1 and in 29 of 128 task-runs with Claude Haiku 5.5; the Concordat agent made none, with no significant difference in task success. tau-bench's own reward did not detect most of these violations. Concordat is open source.",
      "published": "2026-10-27T16:00:26Z",
      "abstract_url": "http://arxiv.org/abs/2610.37461v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37461v1",
      "categories": [
        "cs.AI",
        "cs.MA"
      ]
    },
    {
      "title": "FeatureArc: Benchmarking Coding Agents on Evolving Features",
      "authors": [
        "Yunlong Wang",
        "Alyssa Wu",
        "Cuiyun Gao",
        "Jia Feng",
        "Songyuanyi Lu",
        "Zhichao Hu"
      ],
      "abstract": "A software feature usually evolves over time: developers extend its first version as new use cases and requirements arrive, and each extension builds on code written earlier. Most feature benchmarks evaluate a single request on a fixed codebase. Feature evolution requires evaluating successive extensions on an agent's own earlier code and checking whether the feature remains correct. We introduce FeatureArc to evaluate coding agents on evolving features, together with a pipeline that constructs its tasks from repository histories. The benchmark combines dependencies between successive stages of the same feature, explicit requirements and interface contracts, and executable checks across four programming languages. The pipeline validates fail-to-pass (F2P) tests for new behavior and pass-to-pass (P2P) tests for existing behavior on the reference code before and after each stage. We evaluated 10 coding agents on 73 manually reviewed tasks of 2 to 10 stages. Each agent works in one persistent session and workspace without external test feedback or rollback. The evaluation reveals a gap between first-stage and whole-feature success: pooled success falls from 62.7% at the first stage to 28.5% when the whole feature is tested on the final code. Trajectory analysis identifies inherited defects as a common failure mechanism, present in 72.9% of failed later stages. These earlier errors often remain undetected: in stages following a failure, agents leave the earlier error unnoticed about 95% of the time, and their self-tests rarely revisit earlier stages. Evolving a feature autonomously, without human checks between stages, therefore remains a substantial challenge for current coding agents. At Tencent, the pipeline extends internal evaluations to this setting and guides the development of Hunyuan, Tencent's family of large language models.",
      "published": "2026-10-27T15:45:50Z",
      "abstract_url": "http://arxiv.org/abs/2610.37434v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37434v1",
      "categories": [
        "cs.SE",
        "cs.AI"
      ]
    },
    {
      "title": "Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training",
      "authors": [
        "Faeze Moradi Kalarde",
        "Armin Toroghi",
        "Nam Ho Koh",
        "Doo Hwan Hwang",
        "Seungjin Kim",
        "Kee-Eung Kim",
        "Scott Sanner"
      ],
      "abstract": "Group Relative Policy Optimization (GRPO) has demonstrated strong performance in LLM post-training. However, it allocates a uniform rollout budget to every training question, ignoring variations in question difficulty. As a result, highly difficult questions may receive too few rollouts to generate informative positive reward signals, limiting their contribution to training, while moderately difficult questions often receive more rollouts than necessary. In this work, we formulate the problem of difficulty-aware rollout budget allocation to maximize training utility under the same total rollout budget as standard GRPO. We propose AdaRoll-GRPO, which reformulates rollout allocation as a sequential decision-making problem and leverages a Bayes-adaptive Markov decision process (MDP) to learn an adaptive allocation policy. The learned policy is computed offline and applied during GRPO training without introducing additional online optimization overhead. Extensive experiments on multiple reasoning datasets and LLM architectures demonstrate that AdaRoll-GRPO consistently improves GRPO training performance under a fixed rollout budget, highlighting the effectiveness of adaptive rollout allocation for GRPO in LLM post-training.",
      "published": "2026-10-27T15:41:57Z",
      "abstract_url": "http://arxiv.org/abs/2610.37429v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37429v1",
      "categories": [
        "cs.LG",
        "cs.AI"
      ]
    },
    {
      "title": "Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers",
      "authors": [
        "Zongyuan Sun",
        "Shuai Zhao",
        "Yingjun Du",
        "Ke Shu",
        "Ziyue Zhu",
        "Haoran Luo"
      ],
      "abstract": "Large language models (LLMs) can answer many factual questions correctly, but a correct answer is not necessarily a supported one. For an LLM agent, a knowledge graph (KG) plays two roles: as a source of answers, it supplies facts that the model lacks; as a source of evidence, it supplies the supporting triples that make an answer traceable and auditable. However, existing KG agents treat the KG only as a source of answers: their training data carries answers but not evidence, and their training rewards check the answer but not the retrieved evidence, so agents often answer correctly without retrieving the supporting triples. We propose FoG, a framework that trains KG agents to retrieve the evidence behind their answers. FoG consists of evidence-grounded synthesis, a KG interaction agent, and an evidence-coverage process reward (ECPR): the synthesis recursively hides explicit entities behind verifiable sub-questions, so that every question carries its supporting triples by construction; the agent records the triples returned by the KG; and ECPR credits each supporting triple to the turn that first retrieves it. Experiments on three KGQA benchmarks demonstrate that FoG enables 3B--4B open-source agents to outperform substantially larger baselines in answer accuracy and evidence coverage, with zero-shot cross-KG transfer.",
      "published": "2026-10-27T15:20:37Z",
      "abstract_url": "http://arxiv.org/abs/2610.37405v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37405v1",
      "categories": [
        "cs.AI"
      ]
    },
    {
      "title": "Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling",
      "authors": [
        "Sinjae Kang",
        "Dong Kyu Shin",
        "Jaehong Kim",
        "Kamin Lee",
        "Hyoeun Kim",
        "Kaixin Wang",
        "Li Zhao",
        "Seonghyeon Ye",
        "Kimin Lee"
      ],
      "abstract": "Robot policy evaluation often relies on a binary success signal for task completion. In real-world deployment, however, desired execution behaviors are also defined by quality attributes such as execution speed and motion smoothness, which need to be dynamically controlled according to task requirements. To address this, we introduce Quality-Controllable Policy (QCP), a robot policy that enables inference-time control over both individual quality attributes and their combinations. To extend the controllable quality range beyond initial demonstrations, we further propose an iterative hindsight relabeling loop: QCP collects online rollouts under target quality labels, adds successful trajectories to the dataset, relabels the accumulated trajectories in hindsight according to their measured quality, and fine-tunes the policy. Experiments on Robomimic and RoboEval demonstrate that QCP consistently improves both task execution quality and success rates across iterations, producing successful rollouts that exceed the quality of the initial demonstrations. Finally, we validate QCP on real-world manipulation tasks, where controlling speed increases throughput in conveyor operations and controlling transport stability improves success when carrying wine glasses, demonstrating that quality-controllable policies yield practical gains in physical deployment.",
      "published": "2026-10-27T15:15:14Z",
      "abstract_url": "http://arxiv.org/abs/2610.37393v1",
      "pdf_url": "https://arxiv.org/pdf/2610.37393v1",
      "categories": [
        "cs.RO",
        "cs.AI",
        "cs.LG"
      ]
    }
  ]
};
