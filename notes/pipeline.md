# NQS training & validation pipeline (bosonic + fermionic 3D toric code)

The canonical description of how we train and score NQS ansätze for the 3D
toric code at scale. Two models (bosonic, fermionic), two architectures
(fully-symmetric, approximately-symmetric), one shared pipeline.

---

## 1. Two tracks

```
        EXACT REFERENCE (Colab)                     NQS TRAINING (local CPU or Colab GPU)
  ┌───────────────────────────────┐         ┌──────────────────────────────────────────┐
  │ Three_TC/tests/colab_exact_    │         │ Three_TC/validation.py                    │
  │   diag.py                      │  JSON   │   run_validation(configs, regimes,        │
  │   matrix-free eigsh @ L=2 PBC  │ ──────► │     fermionic=…)                          │
  │   bosonic  OR  fermionic=True  │ outputs/│   → train each ansatz @ each h_z regime   │
  │   → expectation-value JSON     │         │   → score vs the loaded reference JSON     │
  └───────────────────────────────┘         │   → outputs/validation_L2_<model>.json     │
                                            └──────────────────────────────────────────┘
```

The exact track produces a **JSON of expectation values only** (no state
vector). The NQS track loads it and never re-diagonalises. Fidelity is therefore
out of scope; goodness is energy error + observable deviations + V-score.

---

## 2. Module map

| File | Role |
|---|---|
| `Three_TC/model/geometry.py` | `ThreeD_ToricCodeGeometry` — `N`, `vertex_all` (6-tuples), `plaq_all` (4-tuples×3), `bonds`. |
| `Three_TC/model/hamiltonian.py` | `create_hamiltonian` (bosonic) and **`create_hamiltonian_fermionic`** (decorated B̃_p). NetKet operators for VMC. |
| `Three_TC/model/fermionic_decoration.py` | `fermionic_plaquettes(geo)` → `(z_edges, x_edges, coef)` triples; `verify_xz_commutation`; `dressed_string`. |
| `Three_TC/model/networks.py` | `ToricCNN` (fully-sym), `ToricCNN_full` (approx-sym), `compute_edges_3D`. Same ansätze serve both models. |
| `model/exact_diag.py` | shared matrix-free `hamiltonian_linop` (`xz_stabs=` for fermionic) + `expect_x/z/xz_string`. Used by the notebook sweeps. |
| `Three_TC/tests/colab_exact_diag.py` | self-contained Colab ED. `PARAMS["fermionic"]` toggles the decorated model. Emits the reference JSON (with a `"model"` field). |
| **`Three_TC/builders.py`** | **single source of truth**: `config` → geometry / Hamiltonian / ansatz / sampler / `MCState` (`build_state`) + the shared `run_loop`. Used by both `train.py` and `validation.py`. |
| **`Three_TC/train.py`** | config-driven training pipeline (3D analogue of 2D `main.py`). `train(config)` + a CLI (`python -m Three_TC.train …`). Writes weights + run JSON; logs to W&B. |
| `Three_TC/validation.py` | the scoring harness (below); builds via `builders`. |
| `Three_TC/utils/wandb_logger.py` | `init_run/log_step/finish_run` (W&B; `finish_run(..., observables=)` logs precomputed values). |
| `simulation/custom_sampler.py` | `WeightedRule` + `MultiRule` vertex-cluster sampler (the topological-phase fix). |
| `utils/io.py` | `save_model`/`load_model` (flax `.mpack`), reused as-is. |
| `scripts/run_3d.sh` | editable bash launcher for `Three_TC.train` (mirrors the 2D `run_example.sh`). |

---

## 2b. Training a single run (`Three_TC/train.py`)

Config in → trained model + artifacts out, mirroring the inherited 2D `main.py`
but 3D- and W&B-aware. `train(config)` (importable) and a CLI share one path:

```python
from Three_TC.train import train
res = train({"L": 2, "model": "fermionic", "arch": "ToricCNN_full",
             "hx": 0.2, "hz": 0.2, "n_iter": 200, "wandb": True})
```
```bash
python -m Three_TC.train --L 2 --model fermionic --arch ToricCNN_full \
    --hx 0.2 --hz 0.2 --n_iter 200            # or: bash scripts/run_3d.sh
```

Config keys: system (`L, bc, model`), Hamiltonian (`hx, hy, hz, J`), architecture
(`arch, hidden`), training (`n_iter, dt, diag_shift, seed`), sampling
(`n_samples, n_chains, n_discard, chunk_size`), output (`name, out_dir, wandb*`).
Defaults live in `builders.DEFAULTS` + `train.TRAIN_DEFAULTS`; `dtype` is derived
(`complex` iff `hy≠0`).

Outputs, all named `{out_dir}/{name}` (auto `name` =
`{model}_{arch}_L{L}_hx{hx}_hz{hz}`):
- `{name}.mpack` — model weights (`utils.io.save_model`; reload with `load_model`).
- `{name}.json` — config + final observables (`nqs_observables`) + per-step energy curve.
- W&B (if `wandb=True`): per-step curve via `log_step`, final summary + weights
  artifact via `finish_run`.

`build_state` / `run_loop` are imported from `builders`, so **the trained model
is exactly what `validation.py` scores** — no drift between train and validate.

---

## 3. `Three_TC/validation.py` — the harness

Functions, in dependency order:

- `load_reference(json_path)` / `find_reference(outputs_dir, hz, hx, model)`
  — read a `colab_exact_diag.py` JSON into the comparison scalars
  (`E0, gap, A_v_mean, B_p_mean, sx_mean, sz_mean`). `find_reference` matches by
  the JSON's `(model, hx, hz)` **content** (robust to filename float formatting);
  a JSON with no `"model"` field is treated as bosonic.
- `build_model(config, geo)` — instantiate `ToricCNN` or `ToricCNN_full` from
  `{"arch", "hidden"}`.
- `build_sampler(hi, geo)` — `WeightedRule(LocalRule, MultiRule(vertex_clusters))`.
- `_mean_operators(hi, geo, xz_stabs=None)` — mean ⟨A_v⟩, ⟨B⟩, ⟨σ_x⟩, ⟨σ_z⟩ as
  single summed operators (one `vs.expect` each, with a proper error bar).
  `xz_stabs=None` → bosonic B_p; else the decorated B̃_p.
- `nqs_metrics(vs, Ham, geo, ref, xz_stabs=None)` — the metric row (below).
- `train_one(config, hz, ref, *, fermionic=False, …)` — build geom + Hamiltonian
  (bosonic vs `create_hamiltonian_fermionic`) + sampler + `MCState`, run
  `n_iter` VMC/SR steps, return one row.
- `run_validation(configs, regimes, *, fermionic=False, …)` — loop
  configs × regimes for one model, resolve references up front, write
  `outputs/validation_L2_<model>.json`, return the rows.

**Bosonic vs fermionic differ in exactly two lines:** the Hamiltonian builder and
the B observable. Everything else is shared because the vertex star `A_v` — and
hence the Wilson-product symmetry the ansatz enforces — is identical in both.

---

## 4. Metrics (per model × architecture × config × h_z regime)

| Metric | Definition | Reads | Notes |
|---|---|---|---|
| `eps_E` | (E_NQS − E₀)/|E₀| | reference E₀ | variational ⇒ one-sided, monotone |
| `Vscore` | N·Var(H)/⟨H⟩² | NQS only | → 0 for an eigenstate; scales past L=2 |
| `dA, dB` | |⟨A_v⟩−exact|, |⟨B⟩−exact| | reference means | absolute (not %); B is B̃_p when fermionic |
| `dMx, dMz` | |⟨σ_x⟩−exact|, |⟨σ_z⟩−exact| | reference means | dMz is the **discriminator** (see §6) |
| `pull_*` | deviation / MC error | NQS error_of_mean | distinguishes real gaps from MC noise |
| cost | `n_params`, `runtime_s`, `n_iter` | NQS | accuracy is meaningless without it |

Use **absolute** deviation for A/B/σ (they sit near ±1 or cross 0 — percent is
misleading); use **relative** error only for E₀.

---

## 5. How to run (current: L=2 PBC, hx=0.2)

1. **Regimes.** From the bosonic ED sweep `rec_3d`, the transition h_z is the peak
   of `∂⟨σ_z⟩/∂h_z`. Use 3 regimes: deep-topo (~0.10), transition, deep-trivial (~0.70).
2. **Exact references (Colab).** For each regime h_z, run `colab_exact_diag.py`
   with `hx=0.2` — once `fermionic=False`, once `fermionic=True`. Drop the 6 JSONs
   into `outputs/`.
3. **Train + score (local CPU is fine at L=2).** Run the notebook driver cell, or:
   ```python
   import Three_TC.validation as val
   rows = []
   for fermionic in (False, True):
       rows += val.run_validation(configs, regimes, fermionic=fermionic,
                                  outputs_dir="outputs", hx=0.2, n_iter=200, n_samples=4096)
   ```
4. **Read** the metric table, the accuracy-vs-cost Pareto, and the claim panel
   (Δ⟨σ_z⟩ / Δ⟨A_v⟩, fully-sym vs approx-sym, per model).

---

## 6. The headline claim (verified)

`ToricCNN` (fully-symmetric) is **exactly** invariant under the global spin flip
x→−x: its inputs are the B_p Wilson products, even under the flip, so
`log ψ(x)=log ψ(−x)` to machine precision (checked: diff = 0.0). This pins
**⟨σ_z⟩=0** and **⟨A_v⟩=1** at *all* parameters. Under the h_z sweep the exact
⟨σ_z⟩ grows and ⟨A_v⟩ drops, so `ToricCNN` has a *structural* error floor that
`ToricCNN_full` (non-invariant block breaks the flip symmetry) can close.
**Δ⟨σ_z⟩ and Δ⟨A_v⟩ are the discriminators — not σ_x** (σ_x is not pinned;
⟨σ_x⟩≈0.96 for `ToricCNN`). The claim is identical for the fermionic model
because the decoration leaves A_v untouched.

---

## 7. How to extend

- **New architecture:** add a branch in `build_model`; it must take a flat
  spin vector and return scalar `log ψ`. To stay A_v-symmetric, route inputs
  through the Wilson 4-product (see `ToricCNN`).
- **New observable:** add a summed operator in `_mean_operators` and a deviation
  in `nqs_metrics`; mirror in `colab_exact_diag.py` so the reference carries it.
- **Sign-problem regime (h_y≠0):** complex dtype throughout; the ansatz's
  `_normalised_sigmoid` and identity init need complex variants; ED loses its
  Perron–Frobenius positivity (fidelity-by-sign argument breaks).
- **Scale past L=2:** ED reference dies (L=3 PBC = 2⁸¹). Drop `eps_E`/`dA`/`dMz`
  (no reference) and rely on `Vscore`, ⟨A_v⟩/⟨B_p⟩ saturation, and cross-seed
  consistency. The harness already computes V-score for exactly this reason.

See `notes/3D_extension_plan.md` (steps 6–7) and the qubit-reduction note
(`arXiv:2505.10403`, HNF rotation) for the scaling path.
