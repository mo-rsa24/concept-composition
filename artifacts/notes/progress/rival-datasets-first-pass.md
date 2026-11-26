# What each rival benchmarks on: first pass

Step C3 of the [two-paper route](../../../plans/route-two-papers.md) verifies this table and turns it into
the benchmark spec. It was read from each paper's web page on 2026-11-21. Parts of each page came through an
automated summary, so every number needs checking against the paper before it is used.

| Paper | arXiv | Datasets and prompt sets | Models composed | Metrics | Samples per number | Code |
|---|---|---|---|---|---|---|
| Liu et al., Composable Diffusion | 2206.01714 | CLEVR 128 px, 30,000 images; Relational CLEVR 128 px, 50,000; FFHQ | One conditional model per dataset | Classifier accuracy, Clean-FID | 5,000 images | Project page |
| Du et al., Reduce, Reuse, Recycle | 2302.11552 | 2D densities; CLEVR cubes 64 px, 200,000 images; ImageNet 128 px | CLEVR: one model composed with itself | Accuracy for 1 to 5 cubes; IS, FID | 50,000 for ImageNet | Project page |
| Skreta et al., SuperDiff | 2412.17762 | CIFAR-10 split into two disjoint sets; SD v1-4, 20 concept pairs; proteins; molecules | CIFAR: two separately trained models | FID, IS, CLIP, ImageReward, TIFA | 20 images per pair | github.com/necludov/super-diffusion |
| Skreta et al., Feynman-Kac Correctors | 2503.02819 | ImageNet-512 with EDM2-XS; LJ-13; docking targets | One model, or a product of two targets | CLIP, ImageReward; Wasserstein distances | 10,000 images per cell | github.com/martaskrt/fkc-diffusion |
| Bar-Tal et al., MultiDiffusion | 2302.08113 | Panoramas; COCO subset with 2 to 4 objects | One pretrained SD v2 | FID, CLIP; region IoU | 2,000 panoramas; 1,000 regions | Project page |
| Bradley et al., Projective Composition | 2502.04549 | CLEVR 128 px, object counts 1 to 5, composed to 9; SDXL | Separate models or conditionings | Objects in correct locations | 100 per method | No code released |
| Bradley et al., Local Mechanisms | 2509.16447 | CLEVR by location and by colour; SDXL; Matryoshka | One conditional model | Largest object count that works | 1,024 per configuration | Not stated |
| Gaudi et al., CoInD | 2503.01145 | Colored MNIST, Shapes3D, CelebA | One conditional model | Conformity score, JSD | Not stated | github.com/sachit3022/compositional-generation |
| Dutta et al., CO3 | 2509.25940 | Attend-and-Excite; T2I-CompBench Complex; RareBench | No training; SDXL and PixArt-Σ | BLIP-VQA, ImageReward, DINOv2, human rating | Not stated | github.com/debottam-dutta7/co3 |
| Hu et al., ToMe | 2411.07132 | T2I-CompBench colour, texture, shape, 300 prompts each; 50 object-binding prompts | No training; SDXL | BLIP-VQA, ImageReward | 300 prompts per subset | github.com/hutaihang/ToMe |
| Chefer et al., Attend-and-Excite | 2301.13826 | 276 two-subject prompts | No training; Stable Diffusion | Text-text and CLIP similarity, user study | 64 seeds per prompt | Project page |

## Not read yet

The PDFs of Compositional Sculpting (2309.16115), Catastrophic Compositional Generation (2606.23920) and
Test-Time Compositional Generalization (2605.07078) were too large to load. Diffusion Domain Expansion
(2605.23275) was seen only through a search summary and is read first.

## Corrections to names

arXiv 2505.03203 is PiCo. arXiv 2510.23515 is FreeFuse. The printed title of CO3's paper at 2509.25940v2 is
"Steer Away from Mode Collisions: Improving Composition in Diffusion Models".
