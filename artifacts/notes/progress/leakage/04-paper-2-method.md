# Paper 2's method, chosen from where CO3 leaks (step Z4)

Step Z4 chooses Paper 2's method from what [Z2](02-what-predicts-borrowing/design.md) and Z3 find and what the choosing test on board question 80 finds. This file holds the candidates and the owner's constraint now. Each candidate's evidence, cost and the rival it must beat are filled in once Z2 and Z3 have results.

## The owner's constraint

**Prefer methods that leave the diffusion model's weights untouched.** The owner said on 2026-11-24 that classifier guidance is the current best fit because no diffusion model has to be trained, and that it is not a commitment. A candidate that trains the diffusion model, even as a LoRA, needs a reason the others cannot meet. Every candidate is judged on one starting noise per seed following its own trajectory: restarting seeds or keeping the best of several tries is selection, which the owner ruled out as the method on 2026-11-18, and appears only as a baseline.

## The bar and the rival

[Loop 4's leakage rivals page](../targets/leakage-rivals.md), built on this loop's mechanism table, sets them. No paper prints a borrowed-part rate on SDXL, so the bar is CO3 alone on the same pictures, by M1's rule: a lead of at least 5 faithful pairs of 96 and more than 1.96 standard errors. Bounded Attention is the one rival with released SDXL code (`run_xl.py` in its repository), so it is both Z3's default fix and the rival Z4 must beat; its row must say where its boxes came from, and boxes read from finished renders make it an upper bound. Be Decisive's repository holds only a README, so its numbers are printed context. Self-Cross Guidance is released without an SDXL pipeline; its GPT-4o not-a-mixture score is the closest published measure to borrowing.

## The candidates

| Candidate | What it does | Trains the diffusion model? | Evidence so far | Waits on |
|---|---|---|---|---|
| Attention bounds | keeps each subject's attention inside its own region during CO3's early steps ([Bounded Attention's note](01-reading/bounded-attention.md)) | no | needs one box per subject; Loop 6's M9 box reader failed 11 of 24 reads; Be Decisive supplies boxes on SDXL at 77 seconds and 36 GB per image ([its note](01-reading/be-decisive.md)) | Z2 pointing at attention, Z3's result |
| The checker chooses CO3's settings | the readout checker picks CO3's beta at each corrected step, with no gradient ([the choosing test](../score-space/11-the-checker-chooses-on-co3.md)) | no | Loop 10's verdict favours choosing over pushing | board question 80 |
| A checker on the failure signal | a small checker reads only the early reading Z2 finds and pushes, or sets CO3's own settings, on one trajectory per seed ([the idea note](ideas/a-checker-on-the-failure-signal.md)) | no | the same idea worked on digits in Loop 8; on CO3 it needs Z2 to find a reading that predicts borrowing | Z2 |
| Regions from each subject's own push | each subject's prediction acts only in its own region, with the regions read from that subject's signed $d_t$ map on CO3's own path; proposed by Loop 10's independent audit (`../score-space/12-independent-audit.md`) as the lead training-free, one-trajectory direction | no | Loop 6's M9 gave each subject its own region inside CO3: with right boxes and bootstrapping, 8 clear pairs of 8 but every picture damaged; its attention box reader failed 11 of 24 reads ([M9's review](../method-search/04-bind-each-body/review.md)). Loop 10 notes that Bounded Attention as Z3's default would reproduce a rival rather than give Paper 2 a method of its own | Z2's $d_t$ reading and window |
| A text-side fix | edits the prompt embedding before sampling ([A Cat Is A Cat's note](01-reading/a-cat-is-a-cat.md), [Follow the Flow's note](01-reading/follow-the-flow.md)) | no | 9.14 against 8.21 seconds per image on SD1.4; untested on SDXL's two encoders | Z2 pointing at the text encoder |
| An identity reward LoRA | trains a LoRA on a reward read from the finished picture ([the directions map](../../../ideas/which-directions-to-pursue/README.md)) | yes, a LoRA | the only line that has fixed identity is the correction, which a LoRA already learns | the owner's constraint above |
| A combination | any two of the above | depends | none yet | Z2 and Z3 |
