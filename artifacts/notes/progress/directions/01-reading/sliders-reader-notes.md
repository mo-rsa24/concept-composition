# Slider cluster notes (scratch)

## 2311.12092 Concept Sliders (Gandikota, ECCV 2024), arXiv v2 Nov 2023, read from paper
- Direction: LoRA on SDXL/SD1.4, trained so eps_theta*(x,c_t) = eps(x,c_t)+eta*sum_p[eps(x,(c+,p))-eps(x,(c-,p))] (Eq 7-8, p4); preservation prompts p (race names when editing age). Image-pair variant Eq 9 p4. Scale alpha at inference.
- Steer: Table 1 p5: age dCLIP 3.93 LPIPS 0.06; hair 5.59; sky 1.56; rusty 7.60. Eye-size slider eye area x2.75 (p5, Table 2 deye 1.75). Fig 14 p12: CLIP (normalised min-max, no units) vs scale -3..3 for 15 sliders: antonym sliders rise ~monotone, dirtyroom/cooked peak ~+2 then dip at +3; hard-to-negate (glasses, beard...) flat for negative scales then steep rise 0->3. LPIPS grows with |scale|, dirty room 0.55 at +3.
- Transfer: "person" slider tested on "doctor","man","woman","barista" (p5) but no separate number; StyleGAN FFHQ face neurons used on "diverse image generation" (p2, qualitative). Repair slider preferred by 80.39% raters over 250 pairs (p14) across varied subjects.
- Compose: up to 50 sliders qualitative (Fig 7 p6, Figs 18-19 p13-14); no metric of interference between stacked sliders.
- Breaks: interference 0.10 (10% of 2500 images change race/gender) even with disentanglement, 0.36 without (Table 3 p8); SDEdit trade-off reduces edit strength (p8); clipped negative range (p12).
- Relational/count: fix-hands slider: hand issues (incl. wrong finger count) 62% SDXL -> 22% with slider, 150 images (p14). Nothing on object count or binding.

## 2502.01639 SliderSpace (Gandikota, 2025; arXiv v1 Feb 2025; venue not printed in v1), read from paper
- Direction: unsupervised. ~5000 samples of prompt (p3), x0-prediction per timestep, CLIP embed, PCA -> v_i; one LoRA (rank 1 default, 40 dirs, p12) per component trained with L = sum 1-cos(dphi_i, v_i) (Eq 5 p4). SDXL-DMD 4-step main.
- Steer: no per-slider strength curve. Measures diversity: DreamSim diversity higher, CLIP alignment similar (Fig 4 p6, bars); art FID 19.12 vs Concept Sliders 32.86 vs LLM prompts 38.84 vs generic 76.08 (Fig 5 p7); COCO FID 12.12 vs DMD 15.52 vs SDXL 11.72 (Table 3 p6); user study win 72.4/66.0/68.1 vs DMD (Table 1 p5).
- Transfer: FaceNet-encoder SliderSpace on "person" transfers to "police","athlete" and out-of-domain "dog" (Sec 5.4 p6-7, Fig 8 p8) qualitative only.
- Compose: random sparse 3 of 32 sliders per image (p5); no interference measure.
- Breaks: encoder bias, 2h A100 per concept, art dirs not 1-1 with artists (p7). Mode-collapsed concepts need undistilled base or LLM-expanded prompts (p4).
- Relational: none.

## 2403.17064 Baumann et al. Attribute Control (arXiv v2 Mar 2025; venue not printed, CVPR 2025 from memory = inferred), read from paper
- Direction: vector dE added to the subject noun's tokenwise CLIP embedding (Eq 3 p4). Simple: difference of embeddings for "old person" vs "person" averaged over prompts (Eq 2 p4). Learned: distil noise-space difference eps(x|P+)-eps(x|P) into dE by Eq 4 p4 with lambda varied randomly, over many images/prompts. Applied after first 20% of steps.
- Steer: Fig 6 p5 KDE of CLIP(I,"expensive car")-CLIP(I,"car"), 20 samples per lambda in {0,1,2,3}: peak ~-4.2 at 0, ~-2.8 at 1, ~-2.3 at 2, ~-1.5 at 3 (eye-read). Table 1 p6: subject-specificity ratio 3.35 (prompt adjectives 4.14, AdapEdit 6.92); dId 0.40 vs Concept Sliders 0.45; LPIPS 0.10 vs 0.20 (normalised by attribute change). Ablations: CLIP-difference method subject-specificity 2.38, dId 1.20, LPIPS 0.58; no lambda variation 2.23/0.55/0.31 (Table A p12).
- Transfer: trained on "man","woman","person" -> "child","mother","father","doctor","barista","firefighter","John","Jane" works (A.3 p13, Figs D,E p14-15) qualitative only. Trained single-subject, works per subject in two-subject prompts (p6, Fig 9 p7, subject-specificity metric). Cross-model zero-shot SDXL -> SDXL Turbo, SD1.5, LDM3D, aMUSEd, PixArt-alpha (p8 Fig 13-14) qualitative.
- Compose: additive vectors; Fig 11a p7, 2D grids Figs H,I p18-19 qualitative; "interact according to world knowledge".
- Breaks: CLIP token space not globally smooth, phase transitions car->frog (Fig 2 p3); colour leakage when pushed too far (Fig Ba p12-13); spatial relation "man: on the right" works only very rarely (p13); inherits base attribute mixing between subjects (p8); slight correlation of man/woman age (Fig F text p15ish).
- Relational: spatial relation tried, works "(very) rarely" (p13). No count.

## 2409.16535 Prompt Sliders (Sridhar & Vasconcelos, 2024, ECCV'24 workshop on unlearning/model editing per search), read from paper
- Direction: one learned token embedding S* (768 numbers, 3.1KB) scaled by alpha, trained with Concept Sliders' guided target (Eq 5 p6), alpha sampled 0..3. Visual variant with image pairs (Eq 6 p7).
- Steer: Table 1 p8: SDXL CLIP-s 30.00 vs base 28.90 vs LoRA slider 28.52 (100 prompts, 5 per concept, p7). No strength curve.
- Transfer: across prompts qualitative (Fig 5 p9: dragon, city street, cat, child); claims "generalize much better across different prompts" than Concept Sliders (p8). Cross-model SDXL->SD1.5: CLIP 23.97 transferred vs 23.93 trained native vs 21.36 base (Table 1 p8).
- Compose: tokens appended to prompt; Fig 9 p11, 6 concepts on "a photo of a person", qualitative.
- Breaks: "static concepts that do not vary with alpha" (Fig 4 bottom row, p7); at high alpha concept overpowers image, pose changes for curly hair (Fig 10 p10-11); Concept Sliders fail without careful tuning (p8).
- Relational: none.

## 2509.18831 Text Slider (Chiu, Fang, Chen; Academia Sinica; arXiv Sep 2025, venue not printed), read from paper
- Direction: rank-4 LoRA inside the CLIP and OpenCLIP text encoders, trained so encoder(c_t) matches tau(c_t)+sum_q[tau(c+,q)-tau(c-,q)] (Eq 2-3 p4), MSE on token and pooled embeddings; no diffusion backprop. 550 s training. Enabled after t=800 (SDEdit, p4).
- Steer: Table 1 p5 single positive scale, 1000 images of "image of a person, photorealistic" per attribute (p4): SDXL dCLIP age 2.904, smile 3.832, curly 1.285, chubby 1.292 vs Concept Slider 2.012/0.901/0.323/0.394 and Attribute Control 2.668/2.661/0.419/0.866. Fig A8 p18: dCLIP vs scale (Text Slider 0..0.4, others 0..4) near-linear rise, no saturation in range; age ~4.2 at 0.4 for Text Slider vs ~3.4 at 4 for Concept Slider (eye-read).
- Transfer: across architectures (SD1.5, SD3, FLUX schnell, AnimateDiff) qualitative (Fig 7 p7); SD1.5 numbers Table 1 p5. Across subjects: quantitative eval only on one base prompt; categories (person, hair, car, style, scene) each have their own slider. Not tested.
- Compose: sequential stacking qualitative (Fig 6 p7).
- Breaks: rank ablation Fig 10 p8: higher rank -> abrupt dCLIP drops past a scale; usable range age 0-0.4 rank 4, 0-0.2 rank 8, 0-0.1 rank 16. Large scales cause "catastrophic forgetting", unnatural expressions, distortions (App A p11). Training-free embedding-arithmetic variant less responsive (p12).
- Relational: none.

## 2508.19195 All-in-One Slider (Ye, Zhu et al., arXiv v3 Mar 2026, venue not printed), read from paper
- Direction: one top-k sparse autoencoder on SDXL text-encoder hidden states (layers 10/28 of the two encoders), trained on 52,000 face prompts (52 attributes x 1000) (p4-5); direction = W_dec(lambda*ENC(x_A)) of the attribute word's embedding, added to prompt embedding (Eq 6 p5). Unsupervised per attribute.
- Steer: Fig 11 p8: Qwen score QS (avg of edit fidelity, identity, coherence, 1-5 scale) vs lambda 0.15..0.35: flat ~4.0-4.25 to 0.25, then old drops to ~2.9 at 0.30, smile ~3.85; identity IS (ArcFace cos) falls steadily, e.g. smile 0.72 -> 0.49 (eye-read). Appendix Fig 14 p13: landmark L2 change vs normalised strength, linearity R^2 0.973 (ours) vs 0.966 Concept Slider vs 0.962 AttControl.
- Transfer: attributes not in training (races, celebrity identities) qualitative (Fig 7 p7, Fig 1). Subjects: faces/people only; style SAE separate (40 photo styles, Fig 9 p7). Not tested on other subject classes.
- Compose: Table 1 p6: pairs QS/IS Old+Smile 4.2124/0.6882 vs CSlider 4.1503/0.4993 vs AttCtrl 3.6667/0.3755; Old+Makeup 4.4281/0.6277 vs 3.8006/0.5224 vs 4.0555/0.6347.
- Breaks: QS cliff for "old" above lambda 0.25 (p8): "aging more deeply entangled with identity". Multi-subject requires extra fine-tuned pooling module (Fig 10 p8).
- Relational: none (subject-specific attribute assignment to man vs woman via extra module, p7-8).

## 2609.05234 Measured Sliders (Chen, Wei, Yin; Univ Sydney; arXiv v1 Sep 2026, venue not printed), read from paper
- Direction: LoRA branches (rank 4) per measurement, trained so the decoded-image measurement (closed-form pixel statistics: brightness, saturation, warmth, contrast, tint, left-right lighting) moves by target m_k while all other measurements stay still (Eq 6 p7). Observability test SNR_k = decoded natural range / preview RMSE > 1 before training (Eq 3 p6). Calibration maps raw alpha to natural-range units (Eq 8-10 p7-8).
- Steer: Table 1 p8, 553 GenEval prompts, one seed, 5 levels: lighting Spearman rho 0.995, 98.9% monotone sweeps, selectivity 1.761 (SDXL). Baselines on SDXL: Concept Sliders rho 0.122 / 33.5% monotone; Attribute Control -0.112 / 18.4%; learned AttrControl 0.117/24.6%; FreeSliders -0.035/26.2%; Text Slider -0.077/21.3%. FLUX matched: text-difference target rho 0.111 / 9.6% monotone vs measurement target 0.998 / 99.8%, effect x295 (p9).
- Transfer: across 553 GenEval prompts (varied subjects incl. counting/two-object prompts) measured as above; appearance on 6 unseen contents (Table 2 p11).
- Compose: zero-shot, no joint training: all requested directions correct 96.7% pairs, 86.1% triples (p10-11 Table 2); under strict criterion (magnitude, drift <=0.5, DINO>=0.8) only 45.2% pairs and 15.1% triples pass (p13).
- Breaks: sharpness SNR 0.419 -> rho -0.232, predicted before training (Fig 3 p10). Oversized targets drop DINO to 0.35-0.39 (p12).
- Relational: none; measurements are pixel statistics only.

## Venues confirmed from Measured Sliders refs p14: SliderSpace ICCV 2025; Baumann CVPR 2025; Text Slider WACV 2026; Concept Sliders ECCV 2024.
## Not opened: FreeSliders 2511.00103 (saturation detection), SliderEdit 2511.09715, CompSlider; capped at 7 for depth.
