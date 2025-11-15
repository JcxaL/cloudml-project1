| Backend | Host | Arch | Precision | Batch | Runs | Images/s (mean±std) |
|---|---|---|---|---|---|---|
| cpu | Alvins-MacBook | mobilenet_v2 | fp32 | 16 | 6 | 1.74 ± 0.19 |
| cpu | Alvins-MacBook | mobilenet_v2 | fp32 | 32 | 3 | 2.79 ± 1.02 |
| cpu | Alvins-MacBook | mobilenet_v2 | fp32 | 64 | 3 | 8.5 ± 0.1 |
| cpu | Alvins-MacBook | resnet50 | fp32 | 16 | 11 | 4.77 ± 0.48 |
| cpu | Alvins-MacBook | resnet50 | fp32 | 32 | 6 | 6.02 ± 0.57 |
| cpu | Alvins-MacBook | vgg16 | fp32 | 16 | 6 | 3.86 ± 0.11 |
| cuda | p1-gpu-l4 | mobilenet_v2 | fp32 | 128 | 4 | 359.5 ± 7.82 |
| cuda | p1-gpu-l4 | resnet50 | amp | 256 | 7 | 382.52 ± 2.12 |
| cuda | p1-gpu-l4 | resnet50 | fp32 | 128 | 7 | 205.64 ± 1.29 |
| cuda | p1-gpu-l4 | vgg16 | fp32 | 64 | 4 | 148.26 ± 0.88 |
| mps | Alvins-MacBook | mobilenet_v2 | fp32 | 16 | 3 | 140.14 ± 1.83 |
| mps | Alvins-MacBook | mobilenet_v2 | fp32 | 32 | 3 | 183.09 ± 1.56 |
| mps | Alvins-MacBook | mobilenet_v2 | fp32 | 64 | 3 | 175.45 ± 2.06 |
| mps | Alvins-MacBook | resnet50 | fp32 | 16 | 3 | 97.71 ± 1.1 |
| mps | Alvins-MacBook | resnet50 | fp32 | 32 | 3 | 102.56 ± 1.55 |
| mps | Alvins-MacBook | vgg16 | fp32 | 16 | 3 | 79.14 ± 0.9 |
