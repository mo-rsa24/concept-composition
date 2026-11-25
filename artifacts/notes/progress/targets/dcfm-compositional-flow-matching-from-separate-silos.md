# DCFM: compositional flow matching from separate data silos

DCFM (Morshed and Boddeti, arXiv 2606.10153, ICML 2026, 13 pages, [the PDF](pdfs/2606.10153.pdf)) is the published paper nearest
Paper 1's setting. Several clients each hold their own data, no client may share raw images, and the goal is to sample attribute
combinations that no single client ever saw. It comes from the group behind CoInD and carries CoInD's conditional-independence penalty
across the clients. This note says what DCFM does, where its setting differs from Paper 1's, what it prints, and whether it changes
Paper 1's targets or [its evaluation plan](paper-1-evaluation-plan.md). Every page cited was read in full.

## Table of contents

- [What it does](#what-it-does)
- [Where its setting differs from Paper 1's](#where-its-setting-differs-from-paper-1s)
- [What it prints](#what-it-prints)
- [Chest X-rays](#chest-x-rays)
- [Does it change Paper 1's targets](#does-it-change-paper-1s-targets)
- [Does it change the evaluation plan](#does-it-change-the-evaluation-plan)
- [What DCFM leaves open](#what-dcfm-leaves-open)

## What it does

**Composition rule.** Each image carries an attribute vector, such as a digit and a colour, or 14 disease labels on one X-ray. A
combination nobody saw is sampled with the flow version of classifier-free composition: the unconditional velocity plus, for each
attribute, a weighted difference between that attribute's conditional velocity and the unconditional one (p3, Eq. 5). Appendix A
derives this from the score version, which assumes the attributes are conditionally independent given the image (p12, p13).

**Training in two stages.** In Stage I each client trains a flow model on its own data, sometimes with masked labels so it learns each
attribute's marginal, and with a local conditional-independence penalty (p5, Eqs. 8 to 11). The clients then exchange models. DCFM-A
retrains each client's expert with a penalty that matches its joint velocity to the composition of its peers' frozen velocities,
attribute by attribute, a "global" conditional-independence constraint (p5, Eqs. 12 to 14). DCFM-B instead distils all experts into
one student on a trusted server, trained on samples the experts generate, with the same penalty (p6, Eqs. 15 to 17).

**Why it says plain composition fails.** On a two-Gaussian toy with one combination missing, a federated model, Decentralized Diffusion
Models and Diffusion Federated Dataset all fail to place the missing mode. A local penalty helps only when the clients' data look alike
(p4, Figure 3). The paper's account is that independently trained experts learn spurious correlations and velocity fields that do not
fit together (p2, contributions).

## Where its setting differs from Paper 1's

| | DCFM | Paper 1 |
|---|---|---|
| What is composed | attributes of one object or one image: a digit and its colour, diseases on one X-ray, a start and an end quadrant | separate objects in one picture |
| What may pass between data holders | the trained models, during training: DCFM-A exchanges them 1 to 3 times, and in DCFM-B every holder uploads its model to a central server and receives the distilled model back, a cost of 2 model transfers per holder (p9, Table 3) | nothing: two finished experts and samples drawn from them |
| What is trained after the experts | the experts themselves again (DCFM-A) or a whole new generator (DCFM-B) | a small helper trained on the experts' samples |
| Assumption about the attributes | conditionally independent given the image, and enforced by a penalty | the two objects affect each other in one picture, and the plan measures exactly those effects |

The last row is the sharpest difference. DCFM's penalty pushes the joint velocity toward the sum of single-attribute velocities, which
removes any effect that needs both attributes at once. Paper 1's evaluation plan measures such effects: one object hiding or shading
another, and one X-ray finding changing how another appears. Whether DCFM's penalty suppresses those effects is untested in the paper,
and this note infers it from Eq. 7 and Eq. 12 rather than from a result.

## What it prints

Every table cell is in [the numbers file](numbers.csv) as rows `DCFM-T1-001` to `DCFM-T4-056`, read by Claude and read again blind by
GPT-6 Astra through codex, with all 56 cells agreeing.

**Colored MNIST** (p6, p7, Table 1). Ten digits by ten colours, 100 combinations, spread over 10 clients so that half the combinations
appear on no client. The non-IID split draws from a Dirichlet distribution with alpha 0.1. FID is printed for combinations some client
saw (known) and combinations no client saw (novel); lower is better. The number of samples behind each FID is not stated.

| Method | IID, known | IID, novel | Non-IID, known | Non-IID, novel |
|---|---:|---:|---:|---:|
| Federated flow | 9.41 | 20.83 | 15.02 | 20.02 |
| Decentralized Diffusion Models | 8.19 | 25.19 | 7.46 | 17.98 |
| Diffusion Federated Dataset | 8.81 | 29.71 | 7.02 | 38.27 |
| DCFM-A | 8.53 | 11.41 | 7.33 | 9.29 |
| DCFM-B | 9.32 | 12.24 | 8.49 | 9.15 |

The three baselines with the local penalty added sit between, at 16.38 to 31.36 on novel combinations. Figure 6 (p7) plots precision
against recall on novel combinations, as points only.

**Robot planning** (p7, p8, Table 2). OGBench cube-single-play, two clients, three quarters of the 16 start and end combinations seen,
the four diagonal moves held out. Success rate in %, mean and spread over 3 groups. On held-out moves in the non-IID split, DCFM-A
reaches 53.0 ± 2.94 and DCFM-B 54.67 ± 2.49, against 29.67 ± 2.87 for the best baseline (Decentralized Diffusion Models with the local
penalty).

**Cost** (p9, Tables 3 and 4). Each DCFM-A training step costs about four forward and two backward passes against one each for the
baselines. The average novel-combination FID of each cost group is 25.33 for the baselines, 20.24 with the local penalty, 10.35 for
DCFM-A and 10.69 for DCFM-B.

## Chest X-rays

DCFM trains latent flow models at 256 by 256 pixels on NIH Chest X-ray 14, split over 4 clients, IID and non-IID (p8, p9). With 14
labels, most of the 16,384 label combinations never occur: about 99% of training images hold 3 or fewer diseases, and at most 9 occur
together, in 2 images. The paper keeps the 54 combinations that make up at least 0.1% of the data (p8).

It reports two numbers per method: FID, not split into known and novel because rare combinations have too few real images, and
"utility". Utility trains a ResNet-18 on synthetic images only, tests it on real ones, and averages over the 54 combinations the recall
for images that hold exactly that combination (p8, Eq. 18). Both numbers appear only as points in Figure 9 (p8), with no table, so none
is in the numbers file. The text says DCFM's FID is similar to the other methods' and its utility higher (p9). Figure 8 shows generated
Effusion with Mass and Atelectasis with Nodule images, with no measure of anatomy or of how one finding changes another.

## Does it change Paper 1's targets

No. Paper 1's one scoreboard target is CLEVR with location-conditioned experts, and DCFM runs no CLEVR. Its Colored MNIST puts two
attributes on one digit, which the owner placed among context datasets for Paper 1 (board question 35). Its chest X-ray numbers exist
only in a plot, so [the limits page](limits.md#chest-x-ray) still has no printed rival number for X-rays, and a plot point cannot set a
target until a table or the authors' data confirms it.

## Does it change the evaluation plan

It adds two points to the plan's X-ray section, both accepted by the owner (board question 107):

1. **NIH Chest X-ray 14 becomes the candidate dataset.** DCFM shows it can be split into silos and that effusion appears in its
   generated combinations (p8, Figure 8). Whether the 54 combinations DCFM keeps include effusion with cardiomegaly is not printed, and
   tuberculosis and silicosis are not named on any DCFM page.
2. **DCFM's utility is reported beside measure B.** Measure B reads both findings with a classifier trained on real images. Utility
   runs the other way, training on generated images and testing on real ones, so reporting both lets Paper 1's X-ray numbers sit beside
   DCFM's Figure 9 on its own measure.

The CLEVR measures are unchanged: DCFM runs no CLEVR and composes no separate objects.

DCFM cannot run as a row inside Paper 1's setting, because it retrains the experts after they exchange models. As a reference row it
would show what training-time exchange buys, and the plan's measure C would test whether its independence penalty removes the effusion
hiding the heart border. That costs a full DCFM training on X-rays and is not proposed until the X-ray data and experts exist.

## What DCFM leaves open

Each limit below is stated by the paper or shown by its own results, with the page. Readings marked "by eye" are this note's, made
from the figures at full size by a reader without radiology training.

**What forcing independence costs is never measured.** Every DCFM penalty trains the joint velocity to equal the unconditional velocity
plus one difference per attribute (p3, Eq. 7; p5, Eq. 12; p6, Eq. 16). Any part of an image that needs two attributes at once is pushed
out. The paper counts that as a gain: DCFM "is able to reduce the entanglement between some diseases due to training data correlations"
(p9). It never measures what the penalty removes: no shadow of one thing on another, no occlusion, no X-ray finding that changes how
another looks. Its X-ray data is exactly where such dependence lives, since findings that touch the heart's border erase it.

**It composes attributes of one thing, on three small datasets.** Colored MNIST across 10 clients, two attributes (p6, p7); OGBench
cube-single-play across 2 clients, two attributes (p7, p8); NIH Chest X-ray 14 across 4 clients, 14 binary labels, 54 combinations kept
(p8). Every dataset puts the attributes on one digit, one trajectory or one X-ray. None has two separate objects, natural images or text
prompts, and nothing runs above 256 by 256 pixels.

**The experts never stay finished.** DCFM-A exchanges models and retrains every expert, in 1 to 3 rounds (p5, p9). DCFM-B trains a new
generator on a trusted server from the experts' samples (p6). Composing finished experts with no retraining is covered only by the
baselines Decentralized Diffusion Models and Diffusion Federated Dataset, which route between experts at sampling and fail on novel
combinations (p4, Figure 3; p7, Table 1).

**It needs every attribute value seen somewhere, and enough combinations seen overall.** Each attribute value must appear on some client,
and the seen combinations must reach a minimum share of all combinations (p3, Definition 3.1). The runs sit at half the combinations seen
(Colored MNIST) and three quarters (robot planning). How DCFM behaves below that minimum, or when a value appears on no client, is not
tested.

**Several things are not measured.** The number of samples behind each FID is not stated (p7). Colored MNIST FIDs carry no spread or seed
count (p7, Table 1). X-ray FID is not split into seen and unseen combinations, because rare combinations have too few real images (p8).
X-ray results exist only as plot points (p8, Figure 9), with no per-combination breakdown. Whether both diseases are drawn is read only
through a classifier trained on generated images (p8, Eq. 18), with no direct check of each image and no anatomy check. Cost is given in
forward and backward passes rather than GPU-hours (p9).

**Its own results show where it is weaker.** On combinations some client saw, DCFM does not hold the best FID: Decentralized Diffusion
Models has 8.19 against DCFM-A's 8.53 (IID), and Diffusion Federated Dataset 7.02 against 7.33 (non-IID) (p7, Table 1). DCFM-B is
worse than DCFM-A on seen combinations and has lower recall on unseen ones, which the authors put down to distillation (p7, Figure 6;
p9). On robot planning's seen moves the best baseline is level with DCFM within the printed spread (p7, Table 2). On X-rays, every
method's utility sits below 0.25 on the plot's axis, so a classifier trained on DCFM's images finds the exact disease combination for
under a quarter of real cases. Four of the five methods, DCFM-A and DCFM-B among them, score a lower FID on the non-IID split than
the IID one, which the text does not explain (p8, Figure 9, read from the plot).

**What fails in its own figures.** In the toy with one combination missing, DCFM-B's non-IID samples reach the missing mode but leave a
trail running toward a neighbouring mode (p4, Figure 3, bottom right). Figure 5 shows one sample per method for each unseen pair: Diffusion
Federated Dataset's red 4 is malformed, and DCFM-A's 4 is open at the top (p6). Figure 8 shows four X-rays per variant, two disease
pairs, with no finding marked. By eye, the lower DCFM-B Effusion+Mass image has rib-like lines running below the diaphragm and a bent
spine (p8). No figure shows a failure on purpose.

**The cost grows with the number of attributes.** The local penalty needs one term per attribute, so the paper approximates it by
pairs, at about four forward passes per step for DCFM-A (p9, Table 4).
