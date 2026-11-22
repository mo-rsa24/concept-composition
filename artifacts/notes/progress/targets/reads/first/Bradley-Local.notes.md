# Bradley-Local: first-read notes

PDF: arXiv 2509.16447v3, 36 pages. Page numbers are PDF pages.

## Numbers that could not be read

None. The dashes in Table 1 (p3) mark untested configurations and have no row.

The text says Exp. 2 fails to place objects beyond "∼3" (p26). That value is approximate and has no row.

## Plot-only curves

These figures print no result values, so they have no rows: Figure 2 scatter (p3), Figure 4 word-influence curves (p8), Figure 5 bars and heatmaps (p10), Figure 11 (p26), Figure 12 (p27), Figures 14 to 16 (p33, p34) and Figure 17 bars and heatmaps (p36).

## Protocol facts a comparison depends on

- CLEVR at 128x128, 500,000 training samples per object count (p22).
- The model is an EDM2 edm2-img64-xs re-implementation with 124M trainable weights, sampled with no guidance (p23).
- Success at K means K objects at least 50% of the time and at least K-2 objects at least 90% of the time (p23).
- Location Kmax uses a ResNet-18 classifier with 99.74% test accuracy, over 1024 samples per configuration (p23). The classifier counts total objects, so Kmax may overestimate objects at correct locations (p26).
- Color Kmax is counted by hand over 64 samples and uses a lower 25% threshold (p29). Color and location Kmax values are therefore not comparable.
- The SDXL analysis uses 24 prompts with 10 seeds each and 15 inference steps (p9, p32).
- The MDM analysis uses 12 prompts with 10 seeds each at t = 999 (p35).

## Hardware and training time

Training uses a batch size of 2048 over 128 x 2^20 samples (p23). The paper states no hardware or wall-clock time.

## Contradictions inside the paper

- Two printed Kmax values fail the 50% rule on p23 under either reading of "K objects". Exp.3 trained on 1-5 has Kmax 6, with 367 of 1024 images at exactly 6 (35.8%) and 412 at 6 or more (40.2%). Exp.3L trained on 1-3 has Kmax 9, with 228 at exactly 9 (22.3%) and 442 at 9 or more (43.2%). Both are in Table 2 (p28) and Table 1 (p3).
- Four more rows pass only if "K objects" means at least K: Exp.2 trained on 1-3 (37.8% exact), Exp.2 trained on 1-5 (20.3% exact), Exp.2L trained on 1-3 (34.3% exact), and the Single label row of Table 3, which repeats the first of these.
- The Table 2 caption (p28) says columns count objects at conditioned locations with K = Kmax. Rows hold images with more objects than K, such as 317 images with 10 objects at Kmax 9. Page 26 says the classifier counts total objects.
- The Table 2 caption describes a "Len" column. The printed column is Kmax.
- Table 1 prints Kmax 1 for Exp.3 trained on 1 object (p3). Table 2 has no row for that configuration (p28).
- The classifier counts 1 to 10 objects (p23), and p26 says testing goes up to K = 12. Tables 2 and 3 stop at 10, so a Kmax of 10 cannot be told apart from a higher one.
- Page 9 says feature space is more disentangled than pixel space (ratio 1.42), and the stated feature-space range starts at 1.4.
- Figure 2 defines its x-axis as Kmax minus the largest count trained on (p3, p26). The Single label row gives 2 - 3 = -1 (Table 3, p28), and the axis ticks start at 0.
- The Table 3 caption says "Rand num labels", and its rows say "Rand labels" (p28).
