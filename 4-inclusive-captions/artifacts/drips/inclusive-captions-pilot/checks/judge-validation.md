# Does the judge agree with a person

Four wordings of the judge's third question, each asked of the same 48 generated cat-and-dog
images, each combined with the three questions that were already right, each scored against
[the eye's calls](eye-labels-first-48.json). The eye kept 37 of 48. The bar, set before the
numbers came back, was agreement on 44.

| Wording | Keep when | Agrees on | Keeps |
|---|---|---|---|
| "ordinary cat or dog, with no features of the other species mixed into it" | yes | 20 of 48 | 9 |
| "contains an animal that is a mix of a cat and a dog" | no | 44 of 48 | 39 |
| **"Is the cat in this picture a normal cat, and the dog a normal dog?"** | yes | **45 of 48** | 40 |
| "List every animal in this picture, one per line, naming its species" | the list names one cat and one dog | 21 of 48 | 14 |

**The one adopted** is the third. The first fails because the question names the failure and the
model goes and finds it: on a picture of an ordinary ginger cat beside an ordinary brown dog it
answered "No, the image shows a cat with a dog's head." The fourth fails because a free-form list
names things the question did not ask about and the one-cat-one-dog test rarely matches.

**Which way it errs.** All three of the winner's disagreements run the same direction: tiles 11,
38 and 42, each a marginal picture a person rejected and the judge kept. It rejects none that a
person keeps. So the pool it builds carries a few images near the line and loses none of the good
ones, which is the error worth having when the alternative is throwing away usable training data.

**What the other three questions do.** The count said two animals on 44 of 48 and caught both the
three-animal frame and the single animal. The one-cat-one-dog question caught the two-cat frames.
The visibility question said yes on 39. Those three were kept unchanged.

Raw answers: `calibration.json` beside the generated images.
