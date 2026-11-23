# The network view: what NoisyCLIP adds to CLIP

These ten figures open the sentence-score push into the network it is, with real sizes from CLIP ViT-L/14's public config. CLIP's own insides are already drawn on [the CLIP ladder](../../../../../../../goal-setting/learning/papers/deep-learning/diffusion-models/compositional-diffusion-models/scorers/clip-read-through-a-small-network.md), so each of its figures is linked at the point it helps and only what NoisyCLIP adds is drawn here.

**Reading the colours.** Lilac is text, green is image, pale blue is a matching pair, orange is the push and grey is frozen or fixed. The linked CLIP figures keep their own key, where orange marks the shared space and the loss.

**No result on this project's pictures exists yet.** These figures show how the method is wired. Whether it reads or steers any of the five properties is what steps W2 and W3 will measure.

## The whole picture

**1. The same two encoders do three jobs**

<a href="nc1-one-model-three-uses.png"><img src="nc1-one-model-three-uses.png" width="200" alt="Three lanes separated by dotted lines: training, reading and pushing, each through the same text and image encoders."></a>

The model is trained once on a contrastive loss, then read and pushed with its weights held. Open this first when you lose track of which job a later figure belongs to.

**2. The read and push path, with every size**

<a href="nc2-the-whole-read-and-push-path.png"><img src="nc2-the-whole-read-and-push-path.png" width="200" alt="A latent passes a fixed map into the image encoder, meets a right and a wrong sentence at two product nodes, and an orange lane takes the gradient at the latent and subtracts it from the base noise prediction."></a>

The orange lane ends at $x_t$ because that is where the gradient is read. The result then joins $\epsilon_{\text{base}}$, so nothing is added to the latent directly.

**Linked from the CLIP ladder.** [The whole of CLIP](../../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv1-the-whole-of-clip.png) shows the two towers on ViT-B/32, and [the module tree](https://claude.ai/artifact/4mePVaMV9NuxENRkJphieQ) opens every layer.

## The path into the image encoder

**3. Four latent channels become three colours**

<a href="nc3-four-latent-channels-to-rgb.png"><img src="nc3-four-latent-channels-to-rgb.png" width="200" alt="Four grey noise tiles, a three by four weight table with biases, then red, green and blue channels and their colour image."></a>

Each colour is a fixed weighted sum of the four channels, with no VAE decoder. The tiles are synthetic Gaussian noise pushed through the real weights.

**4. The clamp stops the gradient**

<a href="nc4-clamping-kills-the-gradient.png"><img src="nc4-clamping-kills-the-gradient.png" width="200" alt="Left, clamp output against map output with slope one inside 0 to 255. Right, bars of the share of clamped pixels per colour."></a>

The left plot has the map's value on x and the clamped value on y, both in RGB units. The right plot shows the share of pixels clamped on unit-variance noise, computed from the weights with independent channels: 26.8% of red, 11.4% of green and 7.0% of blue.

**5. What is tuned and what is frozen**

<a href="nc5-what-is-tuned-and-what-is-frozen.png"><img src="nc5-what-is-tuned-and-what-is-frozen.png" width="200" alt="The CLIP frame holds a faded text encoder, a saturated image encoder and two faded projections, with the image encoder opened into its blocks at the right."></a>

The published weights change the image encoder. Both projection heads and the first text layer equal base CLIP exactly, and the first vision layer differs by 6.1% of its norm ([the checkpoint comparison](../../../../../artifacts/notes/progress/noise-text-critic/01-reading/tools/checkpoint-compare.txt)). Only those tensors were compared. The released `--only_vision` flag would leave the image projection trainable, while the published weights show it unchanged.

**Linked from the CLIP ladder.** [The picture's path](../../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv6-the-pictures-path.png) walks the patch cut and tokens, and [one query asking the keys](../../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv3b-one-query-asks-the-keys.png) opens attention.

## Training on noisy latents

**6. Training pulls each latent toward its caption**

<a href="nc6-training-on-noisy-latents.png"><img src="nc6-training-on-noisy-latents.png" width="200" alt="Captions and noisy latents enter a faded text encoder and a saturated image encoder, then a similarity matrix with a pale blue diagonal, a contrastive loss label and an attract and repel panel."></a>

The matrix cells are labelled by their product and hold no values. [The CLIP ladder's four pictures against four captions](../../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv7-four-pictures-against-four-captions.png) and [its two softmaxes](../../../../../../../goal-setting/learning/deep-learning/generative-models/02-stable-diffusion-and-clip/artifacts/figures/clip/network-view/cv7b-the-two-softmaxes-and-the-loss.png) show the same loss with real scores.

**7. Which noise levels it practised on**

<a href="nc7-trained-noise-levels-and-pushed-steps.png"><img src="nc7-trained-noise-levels-and-pushed-steps.png" width="200" alt="Noise scale and signal scale against sampler step 0 to 49, with steps 0 to 15 shaded orange and steps 20 to 29 shaded green."></a>

The x axis is the sampler step of 50 and the y axis is the scale from 0 to 1, computed from SDXL's scheduler config. The paper trains on latents 20 to 29 (p. 6), and this loop reads steps 0 to 15, which the model did not train on. W3 pushes only on the steps in that range that W2 can read.

## The score and the push

**8. Right minus wrong**

<a href="nc8-right-minus-wrong.png"><img src="nc8-right-minus-wrong.png" width="200" alt="An image strip meets two sentence columns, and a circle sketch shows the image vector with the two sentence vectors and the dashed line between them."></a>

The two sentences use the same words in a different order. The score difference depends only on the gap between their two vectors, so a text encoder that gave them the same vector would give the push nothing to follow. The circle is a two-dimensional sketch of 768-number vectors.

**9. The gradient walks back to the latent**

<a href="nc9-the-gradient-back-to-the-latent.png"><img src="nc9-the-gradient-back-to-the-latent.png" width="200" alt="Ten forward stages from the latent to the score difference, an orange backward row beneath them, and two boxes naming where the released map blocks the gradient."></a>

The released map blocks the gradient twice, at any clamped colour and at `.byte()`. The push needs a copy of the map without `.byte()`.

**10. How the push joins CO3**

<a href="nc10-how-the-push-joins-co3.png"><img src="nc10-how-the-push-joins-co3.png" width="200" alt="The frozen U-Net gives unconditional and conditional predictions, the push is subtracted from both, and the guided prediction feeds CO3's next step."></a>

Our existing [push code](../../../../scripts/method_search/co3_push.py) subtracts the push from both entries of CO3's main prediction, which lowers the guided prediction by exactly the push.

<details>
<summary>How these were made, and the family checklist</summary>

**How they were made.** `make_network_view.py` draws each figure from `../numbers/numbers.json` and refuses to write one where a line crosses a box or a label overlaps another. The noise thumbnails are synthetic. No SDXL image, no NoisyCLIP weight and no GPU were used.

| Family | Must show | Where |
|---|---|---|
| contrastive matrix | real scores, the temperature, each diagonal cross-entropy, the mean of two directions | layout in figure 6, real values in the linked CLIP figures cv7 and cv7b |
| transformer | blocks with attention, MLP, residual add and norm | figure 5's opened encoder, and the linked cv3 |
| attention | queries, keys, values, one weight row | linked cv3b |
| embedding, tokenizer | ids, a row's numbers, a word that splits | linked cv2 and cv2b |
| pooling | which token is read out | figure 5 reads the summary token, and the linked cv5b compares poolings |
| latent space | the sizes, diffusion on the latent | figures 2 and 3 |
| noise schedule | the signal curve with the steps marked | figure 7 |
| guidance as a vector sum | the base prediction plus the scaled push | figures 2 and 10, and illustration 4 |

</details>
