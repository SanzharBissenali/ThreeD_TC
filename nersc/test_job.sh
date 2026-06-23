#!/bin/bash
# Phase 1: throwaway smoke test to gauge the login->queue->node->output loop and
# its turnaround time. Uses `debug` QOS (fast scheduling, <=30 min, <=2 nodes)
# and the base `python` module, so it does NOT depend on the conda env existing.
#
#   sbatch nersc/test_job.sh
#   # then watch: squeue --me   (or: sqs)
#
# EDIT --account below to your repo. Find it with:
#   sacctmgr -nP show assoc user=$USER format=account
#SBATCH --job-name=tc-smoketest
#SBATCH --account=m5340_g
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "=== node check ==="
echo "host        = $(hostname)"
echo "date        = $(date)"
echo "job id      = ${SLURM_JOB_ID}   nodes=${SLURM_JOB_NUM_NODES}"
echo "cores online= $(nproc)"
free -g | sed -n '1,2p'

echo "=== queue wait (submit vs start) ==="
sacct -j "${SLURM_JOB_ID}" --format=Submit,Start,Elapsed,State -P -n | head -1

echo "=== scipy sanity on the compute node ==="
module load python
srun -n 1 -c 256 python - <<'PY'
import time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
n = 4_000_000
d = sp.diags(np.random.default_rng(0).random(n))
t = time.time()
ev = spla.eigsh(d, k=2, which="SA", tol=1e-6, return_eigenvectors=False)
print(f"eigsh ok: {ev}  ({time.time()-t:.1f}s, n={n})")
PY
echo "=== done ==="
