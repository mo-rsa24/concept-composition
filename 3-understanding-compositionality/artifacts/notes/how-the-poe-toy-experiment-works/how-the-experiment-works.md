# How the product-of-experts toy experiment works

This note explains the three pieces the experiment compares: the ideal expert, the exact product, and the samplers. Each section gives the formula first and the code behind it second. The experiment runs on 2D points, so every picture can be read directly.

[The table the experiment produces](../../results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained.png) has one row per way the two experts' supports relate and 14 columns: the left group uses ideal experts, the right group uses trained ResNets.

## Words this uses

**Expert**: a probability distribution over points $x \in \mathbb R^2$, built from one dataset of $N = 800$ points.

**Noise level $t$**: an integer from 0 to 99. Level 99 is almost pure noise and level 0 is almost clean. The code writes $t = -1$ for "no noise at all" on an ideal expert.

**Score**: the gradient of the log density, $s(x) = \nabla_x \log q(x)$. It points toward where the density is higher.

**PoE (product of experts)**: the distribution $p(x) \propto q_A(x)\,q_B(x)$. It is large only where both experts put mass.

## 1. The ideal expert

**What it is.** The ideal expert is the dataset's own points, each smoothed by a small Gaussian. It is what a perfect diffusion model would learn from those 800 points, written as a formula instead of trained.

**Its density at the clean end** is a mixture with one Gaussian on each data point $x_i$:

$$
q(x) = \frac1N \sum_{i=1}^{N} \mathcal N\!\big(x;\ x_i,\ \sigma^2 I\big), \qquad \sigma = 0.05 .
$$

**Its density at noise level $t$** is the same mixture with every centre shrunk and every Gaussian widened:

$$
q_t(x) = \frac1N \sum_{i=1}^{N} \mathcal N\!\big(x;\ \sqrt{\bar\alpha_t}\,x_i,\ v_t I\big), \qquad v_t = (1-\bar\alpha_t) + \bar\alpha_t\,\sigma^2 .
$$

This is exact, not an approximation. If a clean point is drawn from the smoothed data and then noised by $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$ with $\varepsilon \sim \mathcal N(0, I)$, a sum of Gaussians stays a sum of Gaussians, with the centres and variance above.

**The noise schedule** is the cosine schedule over $T = 100$ steps:

$$
\bar\alpha(u) = \cos^2\!\Big(\frac{u/T + s}{1+s}\cdot\frac\pi2\Big), \quad s = 0.008, \qquad
\beta_t = 1 - \frac{\bar\alpha_{t+1}}{\bar\alpha_t}, \qquad \alpha_t = 1-\beta_t, \qquad \bar\alpha_t = \prod_{s \le t}\alpha_s .
$$

**Its log density**, which is what the code computes, uses a log-sum-exp over the data points:

$$
\log q_t(x) = \operatorname{logsumexp}_i\!\Big(-\frac{\lVert x - \sqrt{\bar\alpha_t}\,x_i\rVert^2}{2 v_t}\Big) - \log N - \log(2\pi v_t) .
$$

The last term is $-\tfrac D2 \log(2\pi v_t)$ with $D = 2$ dimensions.

**Its score** is a weighted pull toward the scaled data points:

$$
s_t(x) = \nabla_x \log q_t(x) = \sum_i r_i(x)\,\frac{\sqrt{\bar\alpha_t}\,x_i - x}{v_t}, \qquad
r_i(x) = \operatorname{softmax}_i\!\Big(-\frac{\lVert x - \sqrt{\bar\alpha_t}\,x_i\rVert^2}{2 v_t}\Big).
$$

The code does not use this closed form. It differentiates $\log q_t$ with autograd, which gives the same vector.

**What it cannot do.** It memorises the 800 points. A trained network may blur or invent between them. That difference is the network's error, and the ideal expert has none.

**Code.** `logq` in `3-understanding-compositionality/toy/rebuild/ideal.py`, and `cosine_beta_schedule` in `3-understanding-compositionality/toy/rebuild/model.py`.

## 2. The exact product

**Definition.** The exact product of expert A and expert B at the clean end is

$$
p(x) = \frac{q_A(x)\,q_B(x)}{Z}, \qquad Z = \int q_A(x)\,q_B(x)\,dx .
$$

This is PoE as a distribution. Every sampler in the table is an attempt to draw from it.

**How it is computed.** The code evaluates $\log q_A + \log q_B$ on a $241 \times 241$ grid over $[-1.1, 1.1]^2$, subtracts the largest value so the exponential cannot overflow, and normalises so the grid cells sum to probability 1:

$$
\tilde p_j = \exp\!\big(\ell_j - \max_k \ell_k\big), \qquad p_j = \frac{\tilde p_j}{\sum_k \tilde p_k \cdot \mathrm{CELL}} .
$$

Draws from the exact product pick a grid cell with probability $p_j\,\mathrm{CELL}$ and add a small uniform jitter inside the cell.

**Why two sets of points give blends.** Two Gaussians multiply into one Gaussian halfway between them:

$$
\mathcal N(x; a, \sigma^2 I)\;\mathcal N(x; b, \sigma^2 I)
= \mathcal N\!\big(a - b;\ 0,\ 2\sigma^2 I\big)\;\mathcal N\!\Big(x;\ \tfrac{a+b}{2},\ \tfrac{\sigma^2}{2} I\Big).
$$

Expanding the product of two mixtures gives one term for every pair of an A point and a B point:

$$
p(x) \propto \sum_{i,j} \underbrace{\exp\!\Big(-\frac{\lVert a_i - b_j\rVert^2}{4\sigma^2}\Big)}_{\text{weight of the pair}}\;
\mathcal N\!\Big(x;\ \tfrac{a_i + b_j}{2},\ \tfrac{\sigma^2}{2} I\Big).
$$

A pair gets weight only when its two points are close, and its mass sits at their midpoint. That is why same-coordinate experts produce blends, why disjoint supports leave only a thin ridge in the gap, and why a point that both experts contain wins by a wide margin ($d = 0$, weight 1).

**With a background.** Dividing by a background distribution $p_b$ removes the part both experts share:

$$
p(x) \propto \frac{q_A(x)\,q_B(x)}{p_b(x)}, \qquad \nabla\log p = s_b + (s_A - s_b) + (s_B - s_b).
$$

The run in the background figure used this form with three different backgrounds. The main table uses no background.

**Total variation** measures how far a set of draws is from the exact product. Both are binned on a $40 \times 40$ grid, and

$$
\mathrm{TV} = \tfrac12 \sum_{a,b} \big|\,\mathrm{pb}_{ab} - \mathrm{sb}_{ab}\,\big|,
$$

where $\mathrm{pb}_{ab}$ is the exact probability in bin $(a,b)$ and $\mathrm{sb}_{ab}$ is the fraction of draws in it. 0 means identical and 1 means no overlap.

**Code.** `exact_grid`, `grid_density`, `perfect_samples` and `tv` in `3-understanding-compositionality/toy/rebuild/ideal.py`.

## 3. The samplers

**What every sampler is given.** At each noise level $t$ the target is the product of the two experts' noised densities, and its score is the sum of their scores:

$$
\pi_t(x) \propto q_{A,t}(x)\,q_{B,t}(x), \qquad s_t(x) = s_{A,t}(x) + s_{B,t}(x).
$$

**Why this is not the exact product.** The exact product is built at the clean end. Noising that product does not give $\pi_t$, because a product of noised densities is not the noised product:

$$
\big(q_A\,q_B\big)_t \;\ne\; q_{A,t}\,q_{B,t} \quad\text{for } t > 0 .
$$

The two agree only at the clean end. A sampler that follows $s_t$ all the way down therefore walks along a different path of distributions from the one the exact product would need. The gap between the true score of the noised product and $s_{A,t} + s_{B,t}$ is the interaction term. This is the point of Du et al.'s paper, and the ideal experts show it with no network involved.

**Reverse diffusion.** One step moves a sample along the added scores and adds a little noise:

$$
x_{t-1} = \frac{x_t + \beta_t\, s_t(x_t)}{\sqrt{\alpha_t}} + \sqrt{\tilde\beta_t}\;z, \qquad
\tilde\beta_t = \beta_t\,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}, \qquad z \sim \mathcal N(0, I).
$$

At $t = 0$ the noise term is dropped. One pass of 100 steps produces a draw.

**Annealed MCMC (ULA, MALA, UHA).** At every noise level, the sampler takes $k = 10$ moves aimed at that level's target $\pi_t$, then drops to the next level. Starting points are standard normal noise.

*ULA, unadjusted Langevin:*

$$
x \leftarrow x + \varepsilon\, s_t(x) + \sqrt{2\varepsilon}\;z .
$$

*MALA, the same proposal with an accept or reject step.* The proposal is $x' \sim \mathcal N\big(x + \varepsilon\,s_t(x),\ 2\varepsilon I\big)$. It is accepted with probability

$$
\min\!\Bigg(1,\ \frac{\pi_t(x')\;\mathcal N\big(x;\ x' + \varepsilon s_t(x'),\ 2\varepsilon I\big)}{\pi_t(x)\;\mathcal N\big(x';\ x + \varepsilon s_t(x),\ 2\varepsilon I\big)}\Bigg),
$$

decided separately for each sample.

*UHA, underdamped Hamiltonian, no accept step.* Each move first refreshes a momentum $v$ partly toward fresh noise, then takes a few leapfrog steps along the score:

$$
v \leftarrow \gamma\,v + \sqrt{1-\gamma^2}\;M^{1/2}\xi, \qquad
v \leftarrow v + \tfrac h2\, s_t(x), \quad x \leftarrow x + h\,M^{-1} v, \quad v \leftarrow v + \tfrac h2\, s_t(x),
$$

with $\xi \sim \mathcal N(0, I)$, damping $\gamma = 0.5$, step $h = 0.03$ and 3 leapfrog steps.

**Settings in the table.** These are the settings of Du et al.'s notebook, used for both groups: ULA and MALA with step 0.001 at all 100 levels, UHA with $h = 0.03$, $\gamma = 0.5$ and 3 leapfrog steps, 10 moves per level, 1500 draws per cell. For the ideal group, reverse diffusion uses `reverse_diffusion` on the exact scores. For the trained group, it uses the network's own sampling loop.

**Code.** `reverse_diffusion`, `ula`, `mala` and `uha` in `3-understanding-compositionality/toy/rebuild/samplers.py`, and the three sampler classes in `3-understanding-compositionality/toy/anneal_samplers.py`.

## 4. What changes with trained experts

**The expert** is a ResNet $f_\theta(x, t)$ wrapped as an energy, so its noise prediction is a true gradient:

$$
E_\theta(x,t) = \lVert f_\theta(x,t) - x\rVert^2, \qquad \hat\varepsilon(x,t) = \nabla_x E_\theta(x,t), \qquad s_t(x) = -\frac{\hat\varepsilon(x,t)}{\sqrt{1-\bar\alpha_t}} .
$$

**It is trained** for 15,001 steps (Adam, learning rate $10^{-3}$, batch 1000, EMA 0.999) to predict the noise that was added:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon, \qquad
\mathcal L = \frac1D\,\lVert \hat\varepsilon(x_t, t) - \varepsilon\rVert^2, \qquad t \sim \mathrm{Uniform}\{0,\dots,99\}.
$$

**Its exact product** is the energy at the clean end, exponentiated and normalised on the same grid as in section 2: $p(x) \propto e^{-(E_A + E_B)}$.

**Code.** `3-understanding-compositionality/toy/rebuild/model.py`, and `train_one.py` in the results folder for the loop that trained the eight networks.

## 5. What each gap in the table measures

**Network error** is the gap between the exact product of the ideal experts and the exact product of the trained experts, in the same row.

**Sampler error** is the gap between a sampler's draws and the exact product of its own group.

A sampler's gap to the ideal exact product mixes both, so a comparison should always stay inside one group.

## 6. What the table shows, read by eye

Nothing below is measured. It is a reading of [the table](../../results/the-exact-toy/what-the-product-of-two-experts-draws-and-what-samplers-lose/interaction-table-ideal-and-trained.png), and the total variation numbers for each cell are still to be computed.

- **Same coordinates:** both groups give the two midpoint blends and every sampler agrees with the exact product.
- **Separate coordinates:** both groups give a grid of clusters, one per red x1 mode and blue x2 mode. The samplers keep it.
- **Disjoint supports:** both groups give a thin line in the gap between the bars.
- **Joint point added:** the exact product of both groups puts most of its mass on the joint point. Every sampler spreads its draws over three clusters of similar size. The likely cause is that the chains do not move mass between clusters, which has not been tested here.
- **Partial overlap:** the exact products and all the samplers look alike in both groups.
- **Ring times bar (Du et al.'s Figure 1):** the exact product is two modes, at the top and bottom of the ring where the bar crosses it. Reverse diffusion draws the whole oval of the ring, with the top and bottom heaviest, as in their figure. UHA draws two clean modes, like their HMC. In the trained group the learned ring has unequal mode heights, so the exact product of the trained experts is lopsided, with the top mode larger than the bottom. That is network error, and it is the only row where it shows by eye.

Two rows map onto the paper's Figure A3, which shows two distributions with no overlap. Its top row is the same-coordinates row and its bottom row is the disjoint-supports row. The paper states only that the approach "fails to generate the product of two distribution when they are disjoint with respect to each other".
