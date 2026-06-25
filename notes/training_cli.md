# Training CLI reference (`Three_TC/train.py`)

Every architecture + training hyperparameter is a CLI flag. **Omit any flag** and
it falls back to `TRAIN_DEFAULTS` / `builders.DEFAULTS` (the `argparse.SUPPRESS`
design), so you only list the knobs you're actually sweeping.

## Full flag set

| Group | Flag | Notes |
|---|---|---|
| System | `--L` (required), `--bc` | `PBC` \| `OBC` |
| | `--model` | `bosonic` \| `fermionic` |
| Hamiltonian | `--hx --hy --hz --J` | fields + coupling |
| | `--hz_preset` | `hard`\|`mid`\|`easy` — sets `hz` AND exact `E0` (delta FOM) |
| | `--exact_E0` | manual `E_exact` at a custom `hz` (alt to preset) |
| Architecture | `--arch` | `ToricCNN` \| `ToricCNN_full` \| `VanillaCNN` \| `VanillaWilsonCNN` |
| | `--hidden` | `ToricCNN`: invariant hidden width |
| | `--noninv_channels` | `ToricCNN_full`/`VanillaWilsonCNN`: noninv channels |
| | `--n_noninv` | `ToricCNN_full`/`VanillaWilsonCNN`: # noninv layers |
| | `--inv_hidden` | post-Wilson hidden widths, e.g. `--inv_hidden 4 4` (final 1-ch appended) |
| | `--kernel_size` | `VanillaCNN`/`VanillaWilsonCNN` conv kernel |
| | `--vanilla_depth` | `VanillaCNN` only: # hidden conv layers |
| | `--noninv_random` | `VanillaWilsonCNN`: random-init noninv (default = identity warm start) |
| Training | `--n_iter` | # VMC/SR steps |
| | `--dt` | (initial) learning rate |
| | `--lr_min` | cosine-decay lr → this over `n_iter`; set `== dt` for constant lr |
| | `--diag_shift` | SR regularization |
| | `--qgt` | `dense` \| `onthefly` \| `auto` (use `dense` on GPU) |
| | `--seed` | |
| Sampling | `--n_samples` | total MC samples |
| | `--n_chains` | # Metropolis chains (GPU auto-bumps if unset) |
| | `--n_sweeps` | sweeps between recorded samples (default `2N` = 48 at L=2) |
| | `--n_discard` | discarded warmup samples per chain |
| | `--chunk_size` | grad chunking (memory) |
| Output | `--name --out_dir` | auto name = `{model}_{arch}_L{L}_hx{hx}_hz{hz}` |
| | `--wandb_project --wandb_entity --wandb_group` | W&B routing |
| | `--no_wandb` | disable W&B |

## Colab cell template (every knob as a variable)

```python
# ---- system ----
L            = 2
BC           = "PBC"          # PBC | OBC
MODEL        = "bosonic"      # bosonic | fermionic

# ---- Hamiltonian ----
HX, HY, HZ   = 0.2, 0.0, 0.2
J            = 1.0
HZ_PRESET    = None           # None | "hard"|"mid"|"easy"
EXACT_E0     = None

# ---- architecture ----
ARCH         = "ToricCNN_full"   # ToricCNN | ToricCNN_full | VanillaCNN | VanillaWilsonCNN
NONINV_CH    = 4
N_NONINV     = 2
INV_HIDDEN   = [4, 4]
HIDDEN       = 8
KERNEL_SIZE  = 3
VANILLA_DEPTH= 2
NONINV_RANDOM= False

# ---- training ----
N_ITER       = 200
DT           = 7e-3
LR_MIN       = 7e-4           # == DT for constant lr
DIAG_SHIFT   = 5e-3
QGT          = "dense"
SEED         = 0

# ---- sampling ----
N_SAMPLES    = 4096
N_CHAINS     = 16
N_SWEEPS     = 96             # default 2N
N_DISCARD    = 8

# ---- output / wandb ----
NAME         = "run1"
OUT_DIR      = "outputs"
WANDB        = True
WANDB_GROUP  = "capacity_sweep"

# ---- assemble flags ----
flags  = f"--L {L} --bc {BC} --model {MODEL} --hx {HX} --hy {HY} --hz {HZ} --J {J}"
flags += f" --arch {ARCH} --hidden {HIDDEN} --kernel_size {KERNEL_SIZE} --vanilla_depth {VANILLA_DEPTH}"
flags += f" --noninv_channels {NONINV_CH} --n_noninv {N_NONINV} --inv_hidden {' '.join(map(str, INV_HIDDEN))}"
flags += f" --n_iter {N_ITER} --dt {DT} --lr_min {LR_MIN} --diag_shift {DIAG_SHIFT} --qgt {QGT} --seed {SEED}"
flags += f" --n_samples {N_SAMPLES} --n_chains {N_CHAINS} --n_sweeps {N_SWEEPS} --n_discard {N_DISCARD}"
flags += f" --out_dir {OUT_DIR} --name {NAME}"
if HZ_PRESET:    flags += f" --hz_preset {HZ_PRESET}"
if EXACT_E0 is not None: flags += f" --exact_E0 {EXACT_E0}"
if NONINV_RANDOM: flags += " --noninv_random"
flags += f" --wandb_group {WANDB_GROUP}" if WANDB else " --no_wandb"

!python -u -m Three_TC.train {flags}
```

## Gotchas

- Don't mix `--hz` and `--hz_preset` — the preset overrides `hz` and also sets
  `exact_E0`.
- `--inv_hidden 4 4` → invariant block `[4, 4, 1]`; `--inv_hidden` empty → `[1]`.
- Keep `n_samples / n_chains` ≳ a few hundred per chain so `R_hat` / `tau_corr`
  are meaningful (e.g. 8192 samples / 512 chains = only 16 per chain — too few).
- For **slowing symmetry breaking**, `diag_shift` goes **up** (more conservative
  SR steps), not down. The 2D paper's tiny `5e-5` only worked because 2D sampling
  never stalled.
