# Environment Notes

## Local RTX 4090
- OS / Kernel: 
- Python / PyTorch: 
- CUDA / Driver / cuDNN: 
- GPU / Memory: 
- Peak compute / bandwidth: 
- Command lines: 
- Extras (gVNIC, container ids, storage notes): 

## Local CPU or Apple Silicon
- OS / Kernel: macOS 26.1 (Darwin 25.1.0)
- Python / PyTorch: Python 3.11.6 / torch 2.9.0
- CPU / RAM: Apple M3 Max / 64 GB unified memory
- Peak compute / bandwidth: 
- Perf tooling (perf, Instruments, etc.): /usr/bin/time -l, Instruments→Metal System Trace (planned)
- Command lines: `/usr/bin/time -l python code/run_train.py --data data/imagenet-mini --arch resnet50 --batch-size 8 --warmup-iters 1 --iters 2 --workers 0 --precision fp32 --backend mps`

## GCP GPU VM
- Project / Region / Zone: velvety-argon-472522-f9 / us-central1 / us-central1-b
- Machine type / vCPU / RAM: g2-standard-4 / 4 vCPU / 16 GB RAM
- GPU type / count / memory: NVIDIA L4 (Ada) / 1 / 24 GB
- Boot disk image: pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20251107 (200 GB balanced PD)
- Driver / CUDA / cuDNN: Driver 570.195.03, CUDA 12.8 (cuDNN from DLVM image)
- Peak compute / bandwidth: ~30.3 TFLOP/s FP32 (242 TFLOP/s FP16/BF16 tensor) / ~300 GB/s (per peaks.json)
- Networking (gVNIC / placement group / storage type): default VPC NIC (10.128.0.4), premium tier ephemeral external IP, no gVNIC / placement group
- Provisioning notes (quota, capacity errors, GPU chase observations): T4 capacity exhausted across all tested US zones; used `scripts/create_gcp_t4.sh` to iterate through L4 zones and landed on us-central1-b (g2-standard-4). Script logs document the chase.
- Command lines: `RUNS=3 WORKERS=4 ./scripts/run_cuda_matrix.sh` (with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `WORKERS=4`) and Nsight via `RUNS=3 WORKERS=4 ./scripts/run_cuda_nsight.sh`
