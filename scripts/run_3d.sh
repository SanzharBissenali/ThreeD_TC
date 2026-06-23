#!/bin/bash
# 3D toric-code NQS training launcher (bosonic or fermionic).
# Copy and edit per experiment, then run from the repo root:
#   bash scripts/run_3d.sh

# --- system ---
L=2                  # linear size (Lx=Ly=Lz); N = 3*L^3 qubits (PBC)
BC=PBC               # PBC or OBC
MODEL=fermionic      # bosonic or fermionic (decorated plaquettes)

# --- physics ---
HX=0.2               # X field
HY=0.0               # Y field (nonzero unsupported: real ansatz only)
HZ=0.2               # Z field
J=1.0                # toric code coupling

# --- architecture ---
ARCH=ToricCNN_full   # ToricCNN (fully-sym) or ToricCNN_full (approx-sym)
HIDDEN=8             # invariant-block channel width

# --- training ---
N_ITER=200
DT=0.02              # SR/TDVP learning rate
DIAG_SHIFT=2e-3      # QGT regulariser; raise if unstable

# --- sampling ---
N_SAMPLES=4096
N_CHAINS=16
N_DISCARD=8

# --- output / logging ---
OUT_DIR=outputs
# drop --no_wandb to enable Weights & Biases logging
WANDB_FLAG="--no_wandb"

mkdir -p "$OUT_DIR"

.venv/bin/python -m Three_TC.train \
  --L "$L" --bc "$BC" --model "$MODEL" \
  --hx "$HX" --hy "$HY" --hz "$HZ" --J "$J" \
  --arch "$ARCH" --hidden "$HIDDEN" \
  --n_iter "$N_ITER" --dt "$DT" --diag_shift "$DIAG_SHIFT" \
  --n_samples "$N_SAMPLES" --n_chains "$N_CHAINS" --n_discard "$N_DISCARD" \
  --out_dir "$OUT_DIR" \
  $WANDB_FLAG
