scope: /home/molef/PhD/poe_repair_min-dog-and-rain (root MASTER_PLAN.md; plans/10 has none of its own)
source: 3-understanding-compositionality/plans/10-what-poe-loses-when-it-composes/today-the-toy-failure-in-six-checkpoints.md, with the write-up's section 8
why: show the cat-dog fusion on shapes where every answer is a formula: the overlay comes from the product (cp3, cp4), gives way when each word's pictures include the pair (cp5), and how much a sampler adds (cp6), before phase 2 tries remedies

mode: --code, reader types by hand
playground: /home/molef/playground/deep-learning/poe-structural-toy/ (notebook.ipynb + poe_toy.py, kernel poe-structural-toy)
source note [my read]: the plan numbers two checkpoints "7" (the one-page record, and the shape range)

rung: a picture is 8 x 16 x 3 numbers (piece 1, child a)
notation: pic[row, col, channel], row 0 at top; white (1,1,1), red (1,0,0), blue (0,0,1)
examples: 2 x 4 toy with one blue pixel at [1, 3]; 8 x 16 white picture, 384 numbers, red at [2, 5]

rung: a shape is the pixels inside a boundary (piece 1, child b)
notation: pixel centre at (row + 0.5, col + 0.5); circle (x-col)^2 + (y-ROW)^2 <= R^2; square |x-col| <= HALF and |y-ROW| <= HALF; ROW 4, R 2.6, HALF 2, columns left 3 / centre 8 / right 13
examples: 3 x 3 toy, radius 1 at (1.5, 1.5) gives a plus of 5; centre circle 24 px covers centre square 16 px, 8 px ring circle-only; without +0.5 the circle is 21 px and off-centre
asked [yours]: wanted to see the masks as actual figures, not printed 0/1 grids

rung: the seven, laid out and saved (piece 1, child c)
notation: draw(*objs) with objs (shape, col); RED (1,0,0), BLUE (0,0,1); training_set() keys "circle left" ... "pair"
examples: coloured pixel counts 24, 24, 24, 16, 16, 16, 40; pair masks share 0 pixels; left/right circles touch the picture edge

rung: one word's model, sampled (piece 2, top)
notation: p(x | circle) = 1/3 sum_k N(x; circle_k, sigma^2 I); sample_mixture(pics, shares, sigma, n, rng) -> (samples, idx)
examples: green channel of pixel [4, 8], bells at 1 (left), 0 (centre), 1 (right); seed 0, 200 draws split 57 / 64 / 79; green means 1.004, -0.004, 1.005, spreads ~0.1; sample values range -0.40 to 1.47 so display clips
source note [my read]: plan cp2's by-hand line says draw the bells for the RED channel; red is 1 in both red and white, so the bells coincide. The green (or blue) channel is the one that separates them
asked [yours]: what's up with the coloured speckle on the white (answered: per-channel noise, clip at 1 keeps only downward draws, 1 in 8 white pixels stay pure white, measured 0.125)
asked [yours]: do we need the noisy row, is the noise a mathematical requirement (answered: one word works at sigma 0; the product needs overlap, at sigma 0 it is zero everywhere, 0/0; d^2 48 same place vs 80 apart, weights e^-1200 vs e^-2000; cp5 is the exception, shared picture)
clicked [my read]: the reader reads the noisy row as decoration; the point that the noise is what makes two words overlap is the one to carry into checkpoint 3
asked [yours]: the noise on/off toy needs show and tell, a name taken apart, the consequence and the derivation; "I don't know what you mean by each word"; "I need to be able to see"
wants to see move [yours]: noise on vs off, side by side, on one pixel
examples: blue of pixel [4, 8], centre circle 0, centre square 1; raw product peak 2.2e-10 at 0.5; scaled product bell at 0.5, sd 0.0707 = sigma/sqrt 2; d^2 table diag 48, off-diag 80; shares 0.33 on the three same-place pairs, 0.00 elsewhere
notation: "word" = the caption ("circle"), its model = the rule built from the pictures captioned with it; support = where a distribution is non-zero
asked [yours]: are there updates from the walk the markdowns need (proposed 4: cp2 red->green channel, duplicate 7, geometry + cp1 counts in the setup, cp3 d^2 48/80 and the noise-off column; waiting on apply)

rung: the exact product, sampled (piece 3, top; piece 2's readout and dot map skipped, still owed for cp2)
notation: exact_product(pics_a, shares_a, pics_b, shares_b, sigma, n, rng) -> (samples, j, k); w_jk = shares_a[j] shares_b[k] e^{-d^2/(4 sigma^2)}; output (A_j + B_k)/2 + noise sigma/sqrt 2
examples: seed 0, 200 of 200 overlays, per place 57/64/79; centre overlays at [4,8] mean (0.485, -0.003, 0.489), spread 0.062 ~ sigma/sqrt2 0.0707; ring pixels (0.999, 0.503, 0.503) pink; weights dropped -> 79 of 200 overlays

applied [yours]: a-d. plan cp2 by-hand -> green channel; record page moved to 9; setup table geometry row; cp1 counts 24/16/40; cp3 by-hand d^2 48/80 shares 1/3; write-up one-pixel section gains "with no noise at all, there is no product" + fig-noise-on-off (script 3-understanding-compositionality/artifacts/notes/what-poe-loses-when-it-composes/fig_noise_on_off.py, card row in its README)

rung: adding two words makes no overlay (piece 4)
notation: exact_sum(pics_a, shares_a, pics_b, shares_b, sigma, n, rng) = sample_mixture over all six with shares/2
examples: seed 0, 88 circles 112 squares, per picture 31 26 31 33 34 45; 0 clean pictures with red and blue; pixel [4,8] near red 26, blue 34, white 140, purple 0; one-pixel sum height at 0.5 is 1.5e-5 vs 1.99 at 0; adding pictures gives (1,0,1) magenta at [4,8] and 2 on white
route [out]: thread, why the product never chooses circle left + square right, emitted, not launched
route [parked]: the "why not apart" thread prompt now lives in the plan, checkpoint 3, as a side-question task; rewritten to rebuild from the geometry row rather than assume the playground survives
integrated [yours]: piece 4 folded into the plan (cp4 by-hand gains the one-pixel sum vs product with fig-sum-vs-product; check gains 88/112 and 26/34/140/0; "watch for: sum adds probabilities, not images") and write-up §8.2 "What the sum does" (six bells, the absent averaging term, the figure); script fig_sum_vs_product.py with a card row
feedback [yours]: inline <img> HTML did not render in the chat panel; use markdown image syntax in replies

rung: the caption dial (piece 5), asked to elaborate [yours]
notation: s = pair's share of each word's pictures; each word = (1-s)/3 per single + s on the pair; 4 x 4 = 16 pairings
examples: d^2 pair x pair 0, circle L x pair 32, overlays 48, pair x square R 48, apart 80+; at s 0.05, sigma 0.1 the next-best pairing is e^-798 behind; 200 of 200 apart at 0.01/0.05/0.2/0.5, 200 of 200 overlays at none; pair share at s 0.01: 1.0 (sigma 0.5), 0.97 (sigma 1), 0.005 (sigma 2); crossover s* ~ e^(-48/(8 sigma^2))/sqrt3, 1.4e-3 at sigma 1, ~0.15 at sigma 2; one-sided (pair in circle only) share = s, measured 15/200 at 0.05, 112/200 at 0.5
applied [yours]: cp5 by-hand gains d^2 0/32/48/80 and the 9s^2/(1-s)^2 e^1200 ratio with fig-caption-dial-pairings; cp5 check gains 200/200 counts; cp5 "watch for" one-sided share s; write-up §9.2 gains "why the dial is a switch" + Figure 14; captions for Figures 12 and 13; appendix rows 12-14; script fig_caption_dial_pairings.py + card row. sigma curve not added: Figure 10 already shows it
route [parked]: SDXL's caption dial (s_cat, s_dog on Re-LAION-2B-en) as route-structural step 7, its own prompt, reusing caption_wetness.py; eye-check bar 40 of 50 written before scoring

rung: the samplers against the exact answer (piece 6, top): the noisy rule, its score, error A
notation: v = abar sigma^2 + 1 - abar; noisy bump N(sqrt(abar) a, v); score -(x - sqrt(abar) a)/v; summed noisy scores aim at a product whose pairing weights use sigma_eff^2 = v/abar = sigma^2 + (1-abar)/abar
examples: one pixel at abar 0.5: centre 0.354 both, variance 0.5025 exact vs 0.2525 summed; s 0.05 pair share under summed scores 1.000 at abar 0.9, 0.997 at 0.5, 0.015 at 0.1 (overlays 0.49), 0.003 at 0.01; exact noisy product 1.0 throughout; unscaled-centres misuse: mean score 0.42 (abar 0.9) and 0.67 (abar 0.1) against ~0
applied [yours]: cp6 by-hand gains the one-pixel variances, sigma_eff and the 1-in-20 table with fig-error-a; cp6 "watch for" sqrt(abar) centres; write-up §3.3 gains "on the toy, error A has a closed form" + Figure 15; appendix row 15; script fig_error_a.py + card row

rung: DDIM on the guided sum, as SDXL runs PoE (piece 6, child a of 4; "next" on the last piece opened its children)
notation: eps = -sqrt(1-abar) score; guided eps = eps_u + w(eps_c - eps_u) + w(eps_s - eps_u); unconditional = all 7 pictures at 1/7; DDIM x0hat = (x - sqrt(1-abar) eps)/sqrt(abar), x_prev = sqrt(abar_prev) x0hat + sqrt(1-abar_prev) eps; cosine schedule as the rough run; readout = nearest of 16 library pictures (3 circles, 3 squares, pair, 9 averages: 3 overlay, 6 "pale, apart")
examples: seed 0, 200 samples, T 200 (stable at 1000): no pair w 1: overlay 122, pale-apart 78 (exact plain product: 200 overlay); w 7.5: 120 / 80; pair 1/20 w 1: overlay 115, pale-apart 68, apart 17 (exact: 200 apart); w 7.5: 115 / 85 / 0 apart. T 50 at w 7.5 gives single shapes (123 circle, 77 square): step-size artefact, the broken run
applied [yours]: cp6 build names 200 DDIM steps; check counts "any other picture"; "Measured, first column" table (pair 1/20 w 7.5 is 116/84 from the saved script, 115/85 in chat from an earlier ordering) + fig-ddim-guided + the p_u division caveat; "watch for" T 50 overshoot; setup readout row names the 16 kinds; script fig_ddim_guided.py writes the figure and counts json; card row
