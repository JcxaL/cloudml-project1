| Env | Model | Precision | Batch | Images/s (mean±std) | Attained GFLOP/s | AI |
|---|---|---|---|---|---|---|
| cpu | mobilenet_v2 | fp32 | 16 | 1.74 ± 0.19 | 2.09 | 0.02 |
| cpu | mobilenet_v2 | fp32 | 32 | 2.79 ± 1.02 | 3.35 | 0.03 |
| cpu | mobilenet_v2 | fp32 | 64 | 8.5 ± 0.1 | 10.23 | 0.1 |
| cpu | resnet50 | fp32 | 16 | 4.77 ± 0.48 | 77.95 | 0.74 |
| cpu | resnet50 | fp32 | 32 | 6.02 ± 0.57 | 98.51 | 0.94 |
| cpu | vgg16 | fp32 | 16 | 3.86 ± 0.11 | 238.94 | 2.28 |
| mps | mobilenet_v2 | fp32 | 16 | 140.14 ± 1.83 | 168.62 | 1.83 |
| mps | mobilenet_v2 | fp32 | 32 | 183.09 ± 1.56 | 220.3 | 2.39 |
| mps | mobilenet_v2 | fp32 | 64 | 175.45 ± 2.06 | 211.11 | 2.29 |
| mps | resnet50 | fp32 | 16 | 97.71 ± 1.1 | 1598.31 | 17.37 |
| mps | resnet50 | fp32 | 32 | 102.56 ± 1.55 | 1677.55 | 18.23 |
| mps | vgg16 | fp32 | 16 | 79.14 ± 0.9 | 4901.75 | 53.28 |
