# h-space reading notes (PDF page numbers)
## 2210.10960 Asyrp (Kwon, Jeong, Uh, ICLR 2023) confirmed
- h-space = 8th layer, U-Net middle block, 8x8x512 for 256px (p18 Fig14), not influenced by skip connection.
- Delta h_t = f_t(h_t) small implicit network trained with CLIP directional loss + recon, 100 samples per attribute (p26).
- Homogeneity: qualitative only (Fig 6 p8: one Delta h optimised for one image gives same attribute on others; eps-space distorts). Mean over N=20 samples near identical (Fig 10, p8-9). Global time-invariant mean also similar (p9).
- p26: "roughly homogeneous across samples and timesteps... not completely independent to the conditions especially on unseen-domain" -> they keep per-sample f_t.
- Linearity: Fig 7 p8 qualitative; negative scales untrained still work. Linear combination Fig 8 p9.
- Random direction of same magnitude -> small change in h, distortion in eps (Fig 9 p9).
- Timesteps: edit only in [T, t_edit], t_edit where LPIPS(x,P_t)=0.33 (p5). t_edit values T=1000 scale: CelebA 515 (young) to 321 (Frida Kahlo); church 371-301; AFHQ 430-302 (Table 2 p21). Too high t_edit insufficient change; too low excessive (p18). Bigger visual change (lower CLIP sim src/tgt) needs longer interval (p18).
- Unseen domain = attribute not in training data (Pixar, Neanderthal) not new subject. User study 80 participants, CelebA in-domain overall 94.92% prefer Asyrp vs DiffusionCLIP, unseen 63.13% (Table 1 p8). S_dir smiling 0.921 etc (Table 4 p27).
- Each model per dataset: no cross-dataset direction transfer test.
## 2303.11073 Haas et al (FG 2024) confirmed
- PCA of h_{T:1} across n samples per timestep, direction = concat of j-th PC across timesteps (p3). Unconditional DDPM CelebA-HQ 256 (p3).
- Jacobian d eps/d h_t SVD per image via power iteration (p4-5); directions found at timestep t applied across all timesteps (p4).
- Image-specific directions transfer to other images qualitatively, same magnitude gamma (Fig 5 p5). No number.
- Supervised: mean difference of h_{T:1} over positive/negative (eq 11 p6). Smile direction with h converges with few labels; >200 needed with x_T (Fig 8a p7).
- Table I p8: CLIP zero-shot change, 100 samples x 10 seeds. Original smile edit: smile 0.26, gender 0.31 (entangled). Glasses edit: glasses 0.32, age 0.68, gender 0.66. Disentangled glasses: glasses 0.38, age 0.13, gender 0.07. Hat disentangled 0.44 vs orig 0.41.
- Random directions same norm: minor change small scale, rapid degradation at larger scale (p4).
- Breaks: churches/bedrooms less interpretable PCA directions (p7, SM H p19).
## 2307.12868 Park, Kwon et al NeurIPS 2023 confirmed
- Pullback of Euclidean metric on h (bottleneck) to x_t via Jacobian J_x = dh/dx; SVD gives local latent basis v in x_t; edit x_t once at one timestep (p3-4, Fig 2). Unsupervised; SD 2.1 with empty prompt or prompt (p6, p15).
- Homogeneity measured: geodesic (Grassmann) distance between tangent spaces of 15 samples, most similar at t=T, diverge toward 0 (Fig 7 p8). Parallel transport of a basis vector to other samples works at t=T, fails at 0.6T (Fig 8 p8).
- Dataset: tangent spaces across timesteps more similar for CelebA-HQ than ImageNet (Fig 9 p9). Numbers only in plot (axis to ~10).
- SD: 50 MS-COCO captions; negative correlation CLIP prompt similarity vs tangent distance; R2 declines over timesteps; prompt influence large until ~0.7T then decreases (Fig 10 p9, p10).
- Edit at T = coarse (age, species); 0.6T = fine (nose colour, expression) (p7). PSD low->high freq (Fig 6 p7).
- Breaks: beard direction converts female to male (Fig 11a p10); SD abrupt changes (Fig 11b p10).
## 2311.17216 Li et al Self-discovering (CVPR 2024) confirmed
- One vector c per concept, added to flattened SD v1.4 middle-block output at every step, h <- h + c; trained by denoising loss: images generated with y+ ("a photo of a woman"), U-Net given y- ("a photo of a person") + c (eq 2 p4). 1K synthesized images, 10K steps (p6). Single vector for all timesteps citing Asyrp (p4).
- Transfer to new prompts MEASURED: vectors trained on "person" only, tested on professions. Deviation ratio (0 balanced, 1 all one gender) Table 1 p6: CEO 0.92 SD -> 0.06; Analyst 0.70 -> 0.02; Secretary 0.64 -> 0.36 (weakest); Winobias avg 0.68 -> 0.17 gender; race 0.56 -> 0.23. Extended prompts ("successful") gender+ 0.70 -> 0.23.
- Composition Table 8 p15: gender+age composed avg 0.12 vs single 0.17.
- Safety I2P Table 2 p7: 0.41 -> 0.27 inappropriate fraction; with SLD 0.20 -> 0.16. Responsible text Table 3 p7.
- Cross-subject qualitative: "running" learned on dog images extends to cats, other animals, humans (Fig 8 p8); skateboarding, jumping, eating per animal (Fig 14 p19). No number.
- Linearity: lambda 0.2 to 0.8 smooth (Fig 6 p7) BUT interpolation figures use SDEdit-style average x_t=(x_t^y + x_t^{c,y})/2 to preserve structure (p15) because "manipulating a single attribute precisely using a linear vector is challenging".
- Breaks: composing more safety vectors (1..7) lowers I2P but FID up to ~40 and CLIP drops; large magnitude shifts away from prompt (Fig 10 p15). Negative scaling of "sexual" 0.2975 worse than learned anti-sexual 0.2169 (Table 5 p13). CelebA-learned vectors leak centred-face layout (Fig 15 p19). Abstract concepts like "hate" cost quality (p7).
## 2409.02374 LOCO Edit (Chen, Zhang, ... Qu, NeurIPS 2024) confirmed. Not h-space: Jacobian of posterior-mean predictor x_t -> x0_hat, single-step edit at t in [0.5,0.7] (p8-9).
- Table 1 p13: human-rated Transfer Success Rate (direction from one CelebA-HQ image moved to 3 others, both signs, 90 cases, p38): Pullback (Park) 0.14, d eps/d x 0.24, NoiseCLR 0.66, Asyrp 0.58, LOCO 0.91. Local edit success Asyrp 0.47, LOCO 0.80.
- p38: Jacobian d eps/d h (h-space) "hardly notable editing results"; p29 h-space restricted to U-Net and masks cannot be applied on h.
- Linearity window: Jacobian low rank (rank ratio <0.1) for t in [0.2,0.7]; high rank t<0.2 and t>0.9; linear approx error small at t=0.7 lambda=40, much larger at t=0.0 (p7, p11).
- Transfer robust to gender, shape, moderate position, dataset differences; onto image lacking the part (eyes) gives almost no edit (Fig 9 p25). Directions centred because datasets centred (p25).
## 2402.18206 Parihar et al Balancing Act (CVPR 2024) confirmed
- Linear attribute classifier on h_t (time-conditioned), trained on 2K labelled h vectors; guidance edits h directly: h~_t = h_t - gamma grad_h L(batch attr distribution, reference) (p5). Batch-level distribution matching.
- SD v1.5: classifier trained on h from 10K images of "a photo of a male/female"; applied to profession prompts (Table 6 p8): FD (fairness discrepancy, lower better) doctor 0.355 random -> 0.015; firefighter 0.235 -> 0.093; generic gender 0.317 -> 0.024. FID ~70 unchanged.
- Classifier trained on CelebA-HQ h-space ineffective on SD "due to a significant domain shift" (p8): cross-domain failure.
- Race on doctor 0.356->0.191, firefighter 0.423->0.186, age 0.488->0.194 (Table 11 p13).
- Timing: early steps close to noise, classifier poor; guidance effective only later (t< tau) but by then features formed, forcing a preset attribute distorts samples (p4). Batch size ablation FD 0.108 (bs2) to ~0.05 (bs 50+) (Table 5 p8).
