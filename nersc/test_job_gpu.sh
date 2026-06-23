#!/bin/bash
# GPU smoke test — gauge queue wait + confirm a GPU is visible. Charges the GPU
# allocation (m5340_g). Uses `shared` QOS with a single GPU (1 of 4 on the node),
# the lowest-footprint GPU request, on the default 40 GB A100 pool.
#
#   sbatch nersc/test_job_gpu.sh
#   squeue --me --start      # predicted start time while pending
#   squeue --me              # PD -> R
#
# Swap --qos=shared for --qos=debug (<=30 min, fast turnaround) if you'd rather
# test scheduling latency on the debug pool. `nvidia-smi` needs no conda env.
#SBATCH --job-name=tc-gpu-smoketest
#SBATCH --account=m5340_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "host = $(hostname)   job = ${SLURM_JOB_ID}   date = $(date)"
echo "=== queue wait (submit vs start) ==="
sacct -j "${SLURM_JOB_ID}" --format=Submit,Start,Elapsed,State -P -n | head -1
echo "=== GPU visible? ==="
srun -n 1 nvidia-smi -L
nvidia-smi
echo "=== done ==="
