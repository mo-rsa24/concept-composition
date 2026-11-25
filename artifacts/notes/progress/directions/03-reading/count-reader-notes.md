# H3 count dial: notes (per-N curves, PDF pages)
SDXL exact acc by N=1..10 (GeckoNum 2406.14774 p19 Fig A2, human): 62 55 40 20 10 11 4 5 3 2; '10+' column 13..80 (overshoot)
SD3 same: 74 72 50 46 24 12 10 10 8 5
GeckoNum Fig3 p7 (small numbers 1-4): numeric-simple SDXL 60 SD3 82; 2-additive SDXL 14 SD3 42; 3-additive 18/41; 2-additive-color 21/50
CountCluster 2508.10710 p7 Fig4 (by eye, CountGD, N=2..10): ours 93 74 93 63 63 37 37 26 15; naive SDXL 62 47 45 30 23 24 15 2 4; CountGen 82 68 63 58 48 44 26 19 6. Table1 p6: SDXL 27.88 -> 55.75 acc, MAE 5.045 -> 0.821
YOLO-Count 2508.00728 p7 Fig5 (by eye, MAE, Q=25,50,75,100): LargeGen vanilla 20 29.5 36 45.5, YOLO-Count 13 16 17.5 17; LargeGen-New vanilla 22 27 38 42, YOLO-Count 15.5 17 19.5 21
T2ICountBench 2503.06884 p7 Fig1 (best of 4): FLUX1.1 .75/.22/.06, SD3.5LT .59/.13/.02; p11 Table3 'a times b': FLUX .38->.16, SD3.5 .29->.18
D2D 2510.19278 p7 Table1: D2D-Multi SDXL 2.44, SDXL-Turbo 2.12, +D2D 9.81, +ReNO 5.31; CoCoCount SDXL 24.88; p25 Table22 SD3 CoCoCount 54.38 Multi(subset) 17.00; p24 Table19 MAE by total count D2D-Multi base 4.2/6.0/9.1/17.6; 78% over-generations D2D-Large
ISAC 2505.20935 p12 Table3 multi-instance N=2..5 SDXL 90 71 49 32 -> 96 89 71 47; SD1.4 94 74 28 22 -> 100 90 51 40; p14 Table5 Flux.1-dev 97 89 82 66, Flux.2-dev 100 93 81 75; p11 Table2 SD3.5-M multi-class k=2..5 62 23 12 3 -> 98 51 40 20
NUMINA 2604.08546 p7 Fig6 Wan2.1-1.3B N=2..8: base 68.7 44.5 38.6 31.7 18.6 15.2 11.3; NUMINA 71.0 60.7 41.6 40.7 30.3 26.3 20.7
