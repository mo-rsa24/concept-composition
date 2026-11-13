# Does any caption say a cat and a dog, and does the picture show both?

## caption-check.md

**What it is.** Four tables of training captions that name a cat word and a dog word, each row a clickable picture, its caption and an empty column for your call (B both, O one, N neither).

**How it was made.** One search over all 128 files of Re-LAION-2B-en kept every caption naming a cat word (cat, cats, kitten, kittens) and a dog word (dog, dogs, puppy, puppies): 1,102,469 captions. `3-understanding-compositionality/scripts/census/cat_and_dog_captions.py` then counted the forms, drew 60 captions at random in each of four classes, fetched the pictures (155 of 240 still load) and wrote the page.

## counts.json

**What it is.** The size of the set, the median caption length (14 words), the share of each caption form, and what joins the two animal words when they sit next to each other.

## sample.json

**What it is.** The 240 sampled captions with their picture links, drawn with seed 20261103, so the page can be rebuilt on any machine.

## images/

**What it is.** The 155 fetched pictures as 360 pixel thumbnails, named by a hash of the picture link.
