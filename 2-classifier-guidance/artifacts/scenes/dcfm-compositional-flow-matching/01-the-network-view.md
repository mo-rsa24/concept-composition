# DCFM's network view: each stage of the overview, opened

Four figures, each opening one part of [the overview](00-the-overview.md), in the clean cheat-sheet register: thin outlines, a soft glow
on the part the figure is about, faded grey for what is frozen. The colours follow [the shared notation](NOTATION.md): blue inputs, green
experts, red outputs, orange for I_t and the penalty's gradient, grey for frozen. A network whose architecture the paper does not state
is drawn as a box that says so, never with invented layers. All four are drawn by [the network-view script](make_network_view.py) from
[the numbers file](dcfm_numbers.json); each has an SVG beside its PNG.

<a href="figures/00-overview.png"><img src="figures/00-overview.png" width="200" alt="The overview: six stages from data silos to output, with the penalty opened in an inset."></a>

## Stage I: one silo trains alone

<a href="figures/01a-stage-one-local-training.png"><img src="figures/01a-stage-one-local-training.png" width="200" alt="A real MNIST 4 slides from noise at t = 0 to the image at t = 1; a table of the four masks of Eq. 8; x_t, t and the masked label enter a box labelled network not stated in the paper; its velocity is compared with x_1 minus x_0 in Eq. 9."></a>

This opens the overview's "Local experts" stage (Eq. 8 to 9, p. 5). The top strip is the straight path x_t = (1 − t) x_0 + t x_1, with t
running from 0 (noise) to 1 (image). The network sees the half-made picture, the time and a label that is whole, cut to one attribute,
or empty, with probabilities π_full, π_marg/2, π_marg/2 and π_uncond; the paper does not state those values. Its output is compared with
the straight line from that noise to that image, x_1 − x_0. The one-attribute labels matter later: the sum rule needs v(x_t, 4) and
v(x_t, red) on their own, and a network that never saw a cut label would not have learned them.

## The sum rule: velocities added tip to tail

<a href="figures/01b-sum-rule-tip-to-tail.png"><img src="figures/01b-sum-rule-tip-to-tail.png" width="200" alt="Two panels of the toy at one point: left, all four cells present, the velocities added tip to tail land on the missing cell's own velocity; right, one silo with the cell missing, the sum points elsewhere, gap 2.38."></a>

This opens "Composed sampling" (Eq. 5, p. 3), drawn as the guidance family's vector sum on the 2-D toy at one point, t = 0.5. Thin green
arrows are the unconditional velocity and the two differences, added tip to tail; the thick dashed arrow is their sum v̂; the dotted
arrow is the velocity the missing cell (2, 1) would have. When all four cells are in the data, the sum lands on it exactly
(1.57, 1.38), gap 0.00. On a silo that never held (2, 1), the same rule gives (0.24, −0.59), a gap of 2.38. The rule is only as good as
the single-attribute velocities it adds, and those come from data where "right" only ever appeared at the bottom.

## The penalty: one network called four times

<a href="figures/01c-the-penalty.png"><img src="figures/01c-the-penalty.png" width="200" alt="Two panels. Left, the local penalty: the same expert called with the joint label, each single attribute and no label; the outputs combine into I_t, and orange gradient arrows return to all four calls. Right, the peer penalty for silo 2 with routing r = (1, 2, 1): two calls are silo 2's own and receive gradient, two are silo 1's frozen expert marked StopGrad."></a>

This opens "Retraining with the penalty" and the overview's inset (Eq. 10, p. 5, and Eq. 12 to 13, p. 5). Left: the local penalty calls
the silo's own expert four times on the same half-made picture, forms I_t = J − (M_1 + M_2 − U), and sends the gradient of λ_CI |I_t|²
back into all four calls: +2I_t into the joint and unconditional calls, −2I_t into the two single-attribute calls. So the joint and the
marginals all move toward agreement. Right: the peer penalty of DCFM-A routes some calls to a frozen copy of another silo's expert. Those
calls get no gradient (StopGrad, Eq. 13), so silo 2 must make its own joint prediction agree with a sum its peer helps build. How the
routing r is drawn is not stated; at least one entry must be the silo itself.

## What crosses between silos, and when

<a href="figures/01d-what-crosses-between-silos.png"><img src="figures/01d-what-crosses-between-silos.png" width="200" alt="Two rows of four steps. DCFM-A: silos train alone, all-gather their experts, each retrains its own against frozen copies R = 1 to 3 times, ending with N experts and RN transfers per silo. DCFM-B: silos train alone, upload to a trusted server, the server distils one student from the frozen teachers, and sends it back, 2 transfers per silo."></a>

This opens "Model exchange" and the DCFM-B card (Eq. 14 against Eq. 15 to 17, pp. 5 to 6; Table 3, p. 9). In DCFM-A every silo sends its
expert to every other, retrains its own against frozen copies, and repeats R = 1 to 3 times (about 2 on average), which costs RN
transfers per silo and leaves N experts that are composed by routing. In DCFM-B every silo uploads its expert once to a trusted server.
The server makes half-drawn pictures by running each frozen teacher from noise (Eq. 15), trains one student to copy them with its own
penalty (Eq. 16 to 17), and sends the student back: 2 transfers per silo. In both, images never leave a silo; models do, during
training. Paper 1's setting allows neither: its experts are finished and only their samples are used.

## What this view cannot show

The figures show where each term comes from and where the gradient goes, with no network sizes, because the paper gives none for Colored
MNIST or robot planning. Its X-ray model is a latent flow model on an unnamed pretrained autoencoder at 256 by 256 (p. 8 to 9). The
numbers on figures 01b and the overview's inset are the 2-D toy's, not MNIST's.
