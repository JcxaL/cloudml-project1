# Environment Notes

## Local RTX 4090
- OS / Kernel: Ubuntu 24.04.3 LTS on WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- Python / PyTorch: Python 3.13 / torch 2.9.1+cu128
- CUDA / Driver / cuDNN: CUDA 12.8 / Driver 581.80 / cuDNN 9.10.2.21
- GPU / Memory: NVIDIA GeForce RTX 4090 Laptop GPU (Compute 8.9) / 16 GB
- Peak compute / bandwidth: FP32: 73.1 TFLOPS, AMP/FP16: 146.2 TFLOPS / 1008 GB/s
- Command lines: see readme.md section 5 (RTX 4090 steps)
- Extras (gVNIC, container ids, storage notes): Running on WSL2; venv at env/venv4090 

## Local CPU or Apple Silicon
- OS / Kernel: macOS 26.1 (Darwin 25.1.0)
- Python / PyTorch: Python 3.11.6 / torch 2.9.0
- CPU / RAM: Apple M3 Max / 64 GB unified memory
- Peak compute / bandwidth: 
- Perf tooling (perf, Instruments, etc.): /usr/bin/time -l, Instruments→Metal System Trace (planned)
- Command lines: `/usr/bin/time -l python code/run_train.py --data data/imagenet-mini --arch resnet50 --batch-size 8 --warmup-iters 1 --iters 2 --workers 0 --precision fp32 --backend mps`

## GCP GPU VM
- Project / Region / Zone: 
- Machine type / vCPU / RAM: 
- GPU type / count / memory: 
- Boot disk image: 
- Driver / CUDA / cuDNN: 
- Peak compute / bandwidth: 
- Networking (gVNIC / placement group / storage type): 
- Provisioning notes (quota, capacity errors, GPU chase observations): 
- Command lines: 
