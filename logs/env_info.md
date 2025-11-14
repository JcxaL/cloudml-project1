# Environment Notes

## Local RTX 4090
- OS / Kernel: 
- Python / PyTorch: 
- CUDA / Driver / cuDNN: 
- GPU / Memory: 
- Command lines: 
- Extras (gVNIC, container ids, storage notes): 

## Local CPU or Apple Silicon
- OS / Kernel: macOS 26.1 (Darwin 25.1.0)
- Python / PyTorch: Python 3.11.6 / torch 2.9.0
- CPU / RAM: Apple M3 Max / 64 GB unified memory
- Perf tooling (perf, Instruments, etc.): /usr/bin/time -l, Instruments→Metal System Trace (planned)
- Command lines: `/usr/bin/time -l python code/run_train.py --data data/imagenet-mini --arch resnet50 --batch-size 8 --warmup-iters 1 --iters 2 --workers 0 --precision fp32 --backend mps`

## GCP GPU VM
- Project / Region / Zone: 
- Machine type / vCPU / RAM: 
- GPU type / count / memory: 
- Boot disk image: 
- Driver / CUDA / cuDNN: 
- Networking (gVNIC / placement group / storage type): 
- Provisioning notes (quota, capacity errors, GPU chase observations): 
- Command lines: 
