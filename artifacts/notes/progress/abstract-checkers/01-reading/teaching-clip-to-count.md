# Teaching CLIP to Count to Ten

arXiv 2302.12066 (v1), read from the full PDF (23 pages, appendix included). No venue is printed on the PDF.

## What it does
The paper finetunes a pretrained CLIP or BASIC model so that its similarity score depends on the number of objects named in a caption (p. 1, p. 2). Each training image with a verified count is compared against its true caption and a copy with the number word swapped (p. 5). The finetuned CLIP-L/14 text encoder then conditions a newly trained Imagen model, which generates requested counts more often than an Imagen model trained on public CLIP-L/14 (p. 8, p. 9, Table 5).

## What the classifier or loss reads
**Input to the count loss.** The loss reads three final embeddings per example: the image, the true caption and the caption with its number swapped (p. 5). The paper feeds these through "CLIP's text and image encoders" (p. 5). Image resolution for CLIP and BASIC is not printed.

**Clean or noisy.** The counting model reads clean images only. Its training set is described as "clean and diverse images" (p. 4), and CountBench holds real web images (p. 5, p. 6). No noisy image, noisy latent or decoded diffusion estimate is passed to the counting model at any point in the paper.

**At generation.** Imagen uses only the counting model's text embeddings (p. 8, p. 15). No use of the CLIP image encoder during sampling is described.

**Models.** The method is run on CLIP-B/32, CLIP-L/14 and BASIC-S (p. 5).

## What it trained on
**Starting point.** Pretrained CLIP or BASIC weights are finetuned on two sets: a large general web image-text dataset and a filtered counting set $C$ (p. 5). The general dataset is unnamed and internal (p. 5, p. 6).

**Filtering, in order.**
1. Keep only captions containing a spelled number from "two" to "ten" (p. 4).
2. Run an off-the-shelf object detector, cited as MobileNetV3, and count detections per object (p. 4, reference on p. 12).
3. Assume the caption refers to the most frequently detected object, and keep the example only if the caption's number equals that object's detected count (p. 4).

**Balancing.** Raw "ten" examples are about 1000 times rarer than "two" (p. 5). The paper samples about 37,000 examples for each of "two" to "six" and uses every passing example for "seven" to "ten", from about 7,000 for "seven" down to about 1,500 for "ten" (p. 5).

**Counterfactual captions.** For each pair $(i_k, t_k)$ in $C$, the number in $t_k$ is replaced with a different random number to make $t_k^{CF}$, so "five dogs" can become "eight dogs" (p. 5). The two captions differ by that one word (p. 4).

**Loss as printed (p. 5, Eq. 1 and 2).**

$$L = L_{CLIP} + \lambda L_{count}$$

$$L_{count} = -\frac{1}{N}\sum_{k=1}^{N}\log\frac{\exp(ei_k \cdot et_k)}{\exp(ei_k \cdot et_k) + \exp(ei_k \cdot et_k^{CF})}$$

$ei_k$ is the image embedding, $et_k$ the true caption embedding and $et_k^{CF}$ the swapped caption embedding. $L_{CLIP}$ runs on all samples and $L_{count}$ runs only on samples from $C$ (p. 5). No temperature is printed in Eq. 2.

**Weight and length.** $\lambda = 1$, reached by a linear warm-up over the first 10,000 steps (p. 5). Training lasts 20,000 steps with a cosine schedule from learning rate $5\times10^{-6}$ and batch size 32,768 (p. 5). A fraction $p = 1/32$ of each batch comes from $C$ (p. 5, p. 7), which is 1,024 examples per batch (inferred). CountBench accuracy for $\lambda$ = 0.1, 1, 5 and 10 is 69.44%, 75.93%, 73.15% and 72.59% (p. 9, Table 4).

## How its gradient enters
No classifier gradient enters any sampler in this paper. The gradient of Eq. 2 updates the CLIP or BASIC weights during finetuning (p. 5). The paper prints no sampler update equation.

**Route into generation.** The finetuned CLIP-L/14 enters through the text encoder: one Imagen model is trained on the textual embeddings of pretrained CLIP-L/14 and another on the counting-aware CLIP-L/14 (p. 8). Each Imagen model is trained for 500,000 steps with batch size 512 on 64 TPUv4 chips (p. 15). Images are sampled with DDPM (p. 15). Only the 64 x 64 base model is retrained, and existing super-resolution models take the output to 1024 x 1024 (p. 15).

## Unseen concepts
This paper does not test held-out concepts.

CountBench images are held out from training: the benchmark comes from "datasets which have no overlap with our training set" (p. 5). That split is on images. The paper prints no category list for $C$ or for CountBench and no split of object categories into trained and untrained (p. 4 to 7). "Zero-shot" here means choosing among nine captions by similarity (p. 6). The statement that the method extends counting "to unrestricted objects" (p. 3) has no category-level test behind it.

## Word leakage and confounds
**The number word is in the input by design.** The text encoder reads the whole caption, including the object noun and the number (p. 5). The test also supplies the object noun, because only the number word changes among the nine candidate captions (p. 6).

**Frequency shortcut.** The paper names this one: on unbalanced data the loss can fall by "classifying frequent numbers as the correct caption" regardless of the image (p. 5). Balancing addresses it for "two" to "six". The counts "seven" to "ten" stay at 1,500 to 7,000 examples against 37,000 (p. 5), so some of this prior can remain (inferred).

**Layout shortcut.** Higher counts are more often simple 2D collections or grids, in both training data and CountBench (p. 5, p. 9, p. 16). The paper attributes its low generation error at "nine" to grid-like training images (p. 15), and lists "images oriented in grids typical to other counts" as a failure (p. 11).

**Written digits.** Baseline CLIP retrieves an image showing the digit 9 for "nine hearts" (p. 1, Fig. 1). The relevancy maps in Figs. 6 and 7 are the paper's check that the model attends to the counted objects, and they are qualitative (p. 8, p. 9).

**Two kinds of object.** CountBench images hold "instances of a particular object" (p. 5), and each test keeps the noun fixed (p. 6). The accuracy at count two therefore measures two of one kind. The paper never tests whether the model can tell "a cat and a dog" from two of one animal or from a fused animal (inferred from p. 5 and p. 6). DrawBench includes prompts with two numbers such as "Three cats and one dog sitting on the grass", scored correct only when both counts match, but those prompts are not reported separately (p. 15).

**Generation comparison changes more than the encoder.** The appendix adds 3% counting-set samples to the batch "while training our Imagen model" (p. 15), and p. 2 says the model "incorporates our counting training set and finetuned CLIP text encoder". The paper does not say the baseline Imagen received those samples. If it did not, the 1.5 to 2 times gain in Table 5 mixes the encoder change with the extra training data (inferred).

## Numbers

| Metric (scale, direction) | This method | Best baseline | Dataset | Model | Page, table |
|---|---|---|---|---|---|
| Count accuracy, % of images given the right number from 9 options, higher is better | 75.93 | 32.94 internal baseline, 31.67 official | CountBench, 540 images, 60 per count | CLIP-B/32 | p. 7, Table 1 |
| Mean absolute count error in objects, lower is better | 0.49 | 1.44 internal baseline, 1.53 official | CountBench | CLIP-B/32 | p. 7, Table 1 |
| Count accuracy, % | 69.02 | 22.75 internal baseline, 17.97 public | CountBench | BASIC-S | p. 7, Table 1 |
| Count accuracy with filtered data but no $L_{count}$, % | 44.26 | 75.93 full method | CountBench | CLIP-B/32 | p. 7, Table 1 |
| Count accuracy with $L_{count}$ on unverified numbered captions, % | 49.81 | 75.93 full method | CountBench | CLIP-B/32 | p. 7, Table 1 |
| ImageNet zero-shot accuracy, %, higher is better | 64.06 | 64.97 internal baseline, 62.93 official | ImageNet | CLIP-B/32 | p. 7, Table 2 |
| **Count two**, % of 60 images classified "two" | 97 | 78 public CLIP | CountBench | CLIP-L/14 | p. 6, Fig. 5 |
| Generation accuracy, % of images with the right count, judged by hand | 40.35 | 24.12 | DrawBench counting, 19 prompts, 12 images each | Imagen | p. 9, Table 5; p. 15 |
| Generation accuracy, % | 68.83 | 34.33 | CIFAR-10 class names, counts 2 to 6 | Imagen | p. 9, Table 5 |
| Generation accuracy, % | 50.18 | 20.00 | CIFAR-10 class names, counts 2 to 10 | Imagen | p. 9, Table 5 |
| **Count two generation**, mean absolute error in objects, lower is better | 0.12 | 0.23 | CIFAR-10 class names | Imagen | p. 16, Table 7 |

**CountBench accuracy by count, CLIP-L/14, read from the diagonal of Fig. 5 (p. 6).**

| True count | two | three | four | five | six | seven | eight | nine | ten | Mean of the nine |
|---|---|---|---|---|---|---|---|---|---|---|
| Public CLIP, % | 78 | 65 | 52 | 22 | 23 | 15 | 23 | 18 | 23 | 35.4 (inferred) |
| Counting CLIP, % | 97 | 90 | 92 | 87 | 90 | 72 | 63 | 92 | 68 | 83.4 (inferred) |

Count two is the easiest count for both models and has the smallest gain, 19 percentage points (inferred from Fig. 5). Each cell rests on 60 images (p. 5). Table 1 prints no CLIP-L/14 row, so the means above are computed from the figure.

**Two inconsistencies in the printed generation numbers.** The per-count errors in Table 7 average 3.32 for the baseline and 1.10 for this method, which match the 2-to-10 column of Table 6 (3.32 and 1.09) (p. 16). Their averages over counts two to six are 1.40 and 0.34, while the 2-to-6 column of Table 6 prints 0.78 and 0.38 (p. 16) (inferred arithmetic). The appendix states 1296 generated images (p. 15), while 19 DrawBench prompts plus 90 CIFAR-10 prompts at 12 images each make 1308 (inferred arithmetic).

## What to copy, what to avoid
- Copy the one-word hard negative for a count or plurality checker: the same example paired with its true description and a minimally changed one. Dropping $L_{count}$ from this recipe lowers CLIP-B/32 from 75.93% to 44.26% (p. 7) (inferred).
- Copy the detector check on training labels. Unverified number captions reach 49.81% where verified ones reach 75.93% (p. 7). Our presence labels on every pair made pictures worse, so their label quality should be measured before another training round (inferred).
- Avoid treating 97% at count two as evidence for our task. It counts two of one kind on clean images, and our failure is two different subjects fused in a noisy latent (inferred).
- Avoid assuming a clean-image CLIP counter can steer early steps. This model never reads noise, and our CLIP checker on decoded early images read the pair at chance (balanced accuracy 0.50 at step 0) (inferred).
- Avoid this paper's route into generation for a frozen SDXL. Its gain comes after retraining Imagen for 500,000 steps with a new text encoder (p. 15), and it gives no evidence for swapping an encoder into a frozen model (inferred).
- Balance the checker's labels and vary layouts across counts, since this paper's higher counts carry a grid-layout cue (p. 5, p. 15, p. 16) that a checker reading attention maps could also pick up (inferred).

## Closer work it cites
- **Learning to Count Objects in Natural Images for Visual Question Answering** (Zhang, Hare and Prügel-Bennett), arXiv 1802.05766, cited on p. 3, reference on p. 14. It is described as a layer that infers the number of objects from normalised attention weights, which is close to a count checker that reads attention maps.
- **TallyQA: Answering Complex Counting Questions** (Acharya, Kafle and Kanan), no arXiv id printed, cited on p. 3, reference on p. 11. It trains a counting model for visual question answering using relation networks between foreground and background regions.
