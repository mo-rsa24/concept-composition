# How many COCO photos the inclusive-captions pilot can actually train on

The pilot's design assumed 300 cat-only, 300 dog-only and 200 both-animal photos. The effect it
measures is sized by the both-animal count, so that number had to be real before anything was
trained. [The pool script](../../../scripts/census/inclusive_captions_pool.py) reads
`instances_train2017.json`, sorts every image by which of cat and dog it holds, and applies the
same 0.25 box-fraction floor the compose scorer uses, so a training photo and a scored render
agree about what counts as an animal being present.

## counts.json

| Set | In COCO train2017 | After the size filter | Wanted |
|---|---|---|---|
| cat-only | 3,906 | 3,335 | 300 |
| dog-only | 4,177 | 2,708 | 300 |
| both animals | 208 (169 with exactly one of each) | **131** | 200 |
| neither animal | 108,975 | 108,975 | 200 |

**The answer.** 131, not 200. The single-animal pools are ten times what the pilot needs; the
both-animal pool is two thirds of it. 208 photos hold both animals, and 77 of them have at least
one animal smaller than a quarter of the frame, which teaches a condition almost nothing about
the other animal.

**What it changes.** Under the design's 25% cat-captioning of both-animal photos, the share of
"a photo of a cat" training draws that hold a dog falls from about 15% to about 11%. Raising the
single-name shares to 40% each brings it back to about 16% and costs nothing, because the blank
rate stays at 10% in every arm and that is the cell the one-variable comparison depends on.

**The lever, if 131 is not enough.** The size filter is what cut 208 to 131. Loosening it to 0.15
recovers photos at the cost of training on dogs that occupy a tenth of the frame; the filter's
value is a flag on the script, and any change to it belongs in the write-up.

## both_sheet_00.png, 01, 02

`both_sheet_00.png` is not shown because it failed the subject check

**What it shows.** All 131 both-animal photos, 48 to a page, each labelled with its tile number
for the audit, each animal's box as a fraction of the image's longer side, and how much the two
boxes overlap. They are what the design hoped for: domestic photographs of a cat and a dog in one
frame, and no hybrids, which is what makes the mass this set adds co-occurrence mass.

**What the audit is for.** COCO's annotations miss animals and sometimes label drawings or
statues. 131 tiles is small enough to look at every one, and the numbered tiles are how a photo
gets struck off.

## manifest.json

On the cluster at `/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions_pool/`. It names
every chosen file with its box measurements, drawn with a fixed seed, and it is what the trainer
reads, so an arm is reproducible from it.

## Cross-references

- The mention of **the pool card**, for the photos the photo stage trains on, in [the route for classifier-guided PoE](../../../../2-classifier-guidance/plans/route-classifier-guided-poe-from-toy-to-animals.md#what-we-know-so-far) (relevance match).

<!-- cluster:start -->

## On the cluster

The other 3 files of this result (8.5 MB) sit at `/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/`. [The file list](on-the-cluster.tsv) gives each file's size and sha256 checksum.

| Run | Files | Size | Kept on the laptop | Cluster folder |
|---|---|---|---|---|
| `(files at this level)` | 3 | 8.5 MB | no picture | `/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/` |

Bring one run back to the laptop from this folder:

```bash
rsync -a mscluster84:/datasets/mmolefe/poe_repair_min/artifacts-archive/4-inclusive-captions/artifacts/results/how-many-photos-the-inclusive-captions-pilot-can-train-on/<run>/ <run>/
```

<!-- cluster:end -->
