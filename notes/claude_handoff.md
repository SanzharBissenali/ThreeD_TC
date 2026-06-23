# Handoff — read this first

This file is the canonical "where the project is right now" document for any
future Claude session picking up the work. Read in this order:

1. **This file** (quick orientation).
2. `notes/3D_extension_plan.md` (the implementation plan + lit scout).
3. `notes/progress_log.md` (per-checkpoint progress detail).
4. `notes/vmc_internals.md` (NetKet/MCMC reference notes).

Then explore the actual code; everything is wired up to work.

---

## Research direction (do not lose sight of)

User is extending Kufel et al.'s 2D approximately-symmetric NQS architecture
([arXiv:2405.17541](https://arxiv.org/abs/2405.17541)) to the **3D toric code**.
The contribution is the architecture class, *not* "first NQS for 3D TC" —
that's been done (Luo et al. 2021, autoregressive transformer). The novelty
lives in:

1. Sign-problem regimes (h_y or other non-stoquastic perturbations).
2. Mixed-field 3D phase diagrams (under-mapped versus 2D).
3. Possible extensions to X-cube fracton model and (longer term) fermionic 3D
   toric code — both far less explored numerically.

Strategic positioning: build the bosonic 3D toric code baseline cleanly,
then push into novel-physics regimes. The fermionic 3D TC is a follow-up
project, not the current one.

---

## What's built and working

```
Three_TC/
├── model/
│   ├── geometry.py        ThreeD_ToricCodeGeometry (PBC + OBC)
│   │                      vertex_all (6-tuples), plaq_all (4-tuples × 3
│   │                      orientations), bonds, edges_3D-able
│   ├── hamiltonian.py     create_hamiltonian (reused from 2D code verbatim)
│   └── networks.py        Two CNN blocks + two composed models + helpers.
│                          See "Architecture" section below.
├── tests/
│   ├── test_geometry.py
│   ├── test_hamiltonian.py
│   ├── test_tiny_MLP.py        Minimal MLP — works at h=0, scales badly
│   ├── test_symm_CNN.py        ToricCNN  — Wilson + CNN_inv ×2
│   ├── test_full_CNN.py        ToricCNN_full — full architecture
│   ├── test_exact_diag.py      Exact diag via netket (CPU)
│   └── colab_exact_diag.py     Pure scipy, matrix-free; for Colab
└── utils/
    └── wandb_logger.py    init_run / log_step / finish_run
```

### Architecture, condensed

```
                                ┌── CNN_noninvariant_3D (identity-init,
                                │   normalised sigmoid)
                                ▼
spin x ∈ {±1}^N ─── gather ─── (B, 3, L, L, L) ── scatter ── (B, N)
                   via edges_3D                  via inv_perm
                                                            │
                                                            ▼
                                                       Wilson 4-product
                                                            │
                                                            ▼
                                            (B, 3, L, L, L) ── CNN_invariant_3D
                                                                ×2, ELU
                                                            │
                                                            ▼
                                                        mean → log ψ
```

Key facts to remember:
- **N_plaq = N = 3 L³** in 3D PBC (a real geometric coincidence, not a bug).
- `plaq_all` is in natural `(c, ix, iy, iz)` order, so its reshape into
  `(3, L, L, L)` is direct.
- `arr_coord` (and hence the spin sample `x`) is in `lexsort(z, y, x)` order;
  needs `edges_3D` permutation to reach `(3, L, L, L)` form.
- `CNN_noninvariant_3D` is identity-initialised. At step 0 the full model
  *exactly* reproduces `ToricCNN` (verified to ~1e-8).

### Three crucial code patterns

1. **Identity initializer** (`networks.py:identity_initializer_3D`): sets the
   centre kernel position to `eye(C_in, C_out)`, all other positions to 0.
2. **Normalised sigmoid** (`networks.py:_normalised_sigmoid`): maps ±1 → ±1
   exactly, so the identity property survives the activation.
3. **Wilson 4-product**: unchanged from 2D paper. Still A_v invariant in 3D
   because A_v flips 6 edges, each plaquette intersects those 6 in 0 or 2.

---

## What's verified at this point

| Check | Result |
|---|---|
| L=2 PBC, h=0, TinyMLP             | E → −32 ✓ |
| L=3 PBC, h=0, TinyMLP             | E → −108 (after `diag_shift=1e-3`) ✓ |
| L=4 PBC, h=0, TinyMLP             | E → −256 ✓ |
| L=2 PBC, h=0, ToricCNN            | E → −32 ✓ |
| L=2 PBC, h=0, ToricCNN_full       | E → −32 ✓ |
| Vertex-flip symmetry (Wilson)     | log ψ unchanged to ~1e-7 ✓ |
| Translation symmetry (CNN, PBC)   | log ψ unchanged to ~1e-7 ✓ |
| `ToricCNN_full` ≡ `ToricCNN` at init | difference ~1e-8 ✓ |

---

## What's *not* yet verified (the open question)

The headline experiment: at h ≠ 0, does `ToricCNN_full` beat `ToricCNN`?

Predicted, and now CONFIRMED by a direct ψ(x)=ψ(−x) check (2026-06, see
`Three_TC/validation.py`):
- **ToricCNN** (sym-only) has hard architectural constraints under perturbation:
  - `⟨σ_z⟩ = 0` on every qubit — **exactly**, at all parameters. The B_p-only
    inputs are even under the global spin flip x → −x (each plaquette has even
    weight), so `log ψ(x) = log ψ(−x)` to machine precision; any odd-in-x
    observable (the σ_z magnetization) is therefore pinned to 0.
  - `⟨A_v⟩ = 1` on every vertex (Wilson 4-product enforces A_v invariance).
  - NOTE: an earlier draft said `⟨σ_x⟩ = 0`; that was a slip. σ_x is NOT pinned
    (verified ⟨σ_x⟩ ≈ 0.96 for ToricCNN). The pinned magnetization is σ_z.
- **ToricCNN_full** can break both: the non-invariant block, once it deviates
  from identity, breaks ψ(x)=ψ(−x) (verified: the diff jumps from 1e-7 to ~1).

The experiment to run: L=2 PBC, `hx=0.2`, sweep `h_z` (exact diag via
`colab_exact_diag.py` for ground truth). Run both architectures, compare:
1. E vs E_exact (relative error; variational ⇒ one-sided)
2. **⟨σ_z⟩ vs exact** — the key discriminator (ToricCNN floored at 0; under h_z
   the exact value grows, so the gap is structural, not a training failure)
3. **⟨A_v⟩ vs exact** — ToricCNN floored at 1; exact drops under h_z

That's the moment "the architecture started doing real physics."

---

## Important conventions / gotchas

- **Geometry index order**: `plaq_all` is `(c, ix, iy, iz)`, natural reshape.
  But `arr_coord` is `lexsort(z, y, x)`, so spin samples need `edges_3D` to
  match. Don't blindly reshape `x` — gather through `edges_3D`.
- **Flax static fields must be hashable**. `plaq_all` and `edges_3D` are
  passed as nested tuples. See `tests/test_full_CNN.py` for the
  `_to_tuple()` recipe.
- **2D code has a latent PBC bug**: `(coord + shift) % self.Lx` is applied to
  both x and y coordinates. For square lattices Ly=Lx this is harmless; for
  Ly≠Lx it would mis-wrap. Forced square in `utils/config.py:145`. Don't
  carry this into 3D code — wrap per-axis (`% Lx`, `% Ly`, `% Lz`).
- **Single-spin Metropolis fails as wavefunction sharpens**. The custom
  sampler `WeightedRule + MultiRule(np.array(geo.vertex_all))` from
  `simulation/custom_sampler.py` is the fix; trivially generalises to 3D.
- **`exact_E0` reference**: only the 2D code's energies are available
  cheaply. For 3D: only L=2 PBC has reachable exact diag (N=24, 2²⁴=16M).
  L=3 PBC (N=81) is beyond Lanczos forever.
- **GPU OOM on Colab when building Hamiltonian sparsely via NetKet**.
  Workaround: `colab_exact_diag.py` is matrix-free (scipy LinearOperator),
  uses ~1 GB peak. Force CPU at the top.

---

## What's NOT built yet (next steps, prioritised)

1. **The headline comparison run**: L=2 PBC, hx=0.1, both architectures,
   exact diag reference. The current `test_symm_CNN.py` and `test_full_CNN.py`
   are mostly ready but need:
   - Load `exact_E0` from `colab_exact_diag.py`'s JSON
   - Log `⟨σ_x⟩` and `⟨A_v⟩` in `finish_run` extras (most important)
   - Guard `gap_to_target` for when `exact_E0` is None
   - Tags must reflect actual perturbation, not "h=0"
2. **Perturbation sweep** at L=2 PBC: `hx ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3}`,
   both architectures, plot energy gap and ⟨σ_x⟩ vs hx.
3. **Scale to L=3 PBC** under same perturbations (no exact ref; use
   ⟨A_v⟩, ⟨B_p⟩, Var(H), cross-seed checks).
4. **Sign-problem regime**: `hy ≠ 0`, complex dtype. The architecture's
   `_normalised_sigmoid` is real; you'll need a complex version, and the
   identity initialiser will need a complex eye matrix. Bigger change than
   it looks.
5. **Observables module for 3D**: 1D Wilson loops (electric strings) and
   closed-surface Wilson operators (m-loop BFFM order parameter). The 2D
   `simulation/observables.py` has the BFFM construction for surfaces;
   generalise to 3D.

Step 4 (sign problem) is where the paper-shaped result lives. Don't rush
there; build through steps 1–3 first.

---

## User preferences observed in this conversation

- Wants to **understand**, not just copy code. Prefers stepwise explanations
  and quizzes over code dumps. Asks "why?" frequently.
- Comfortable with physics; learning JAX/Flax from scratch.
- Prefers concise answers over thorough ones; called out verbosity a few
  times.
- Will edit files independently — don't assume my versions of test scripts
  are current. Read before commenting.
- Plans to run experiments on Colab (L4/T4/A100). Mind GPU OOM.
- Has wandb set up (entity:
  `models-california-institute-of-technology-caltech`,
  project: `approx-sym-3D-TC`).

---

## Session addendum — geometry walkthrough (2026-06-16)

Pedagogical session: user re-walked `geometry.py` line-by-line. Concrete
artifacts and lessons that don't appear above:

- **Float-key dict pitfall, fixed.** `_coord_to_idx` is now keyed on
  `tuple((2 * coord).round().astype(int))`, not on raw floats. Reason:
  `(... + ½) % L` on floats can produce `1.9999...` vs `2.0`. The integer
  2×-coord trick is exact and hashable. Used in both branches of
  `_setup_lattice` and inside `_mapping3Dto1D`.
- **Bonds: perpendicular pairs only, enumerated per vertex.** New
  `_generate_bonds` iterates over vertices, collects 6 incident edges as
  `(axis, qubit_idx)`, and yields the C(6,2)−3 = 12 perpendicular pairs.
  Counts: `len(bonds) == 12·Lx·Ly·Lz` for PBC. The 2D "+½,+½" trick does
  not cleanly de-duplicate in 3D — don't try to port it.
- **Per-axis PBC mod confirmed.** Vertex and plaquette enumerators wrap
  `% Lx, % Ly, % Lz` separately (handoff gotcha #3 was the warning;
  current code is correct).
- **`_separate_vertex_stabilizers`**: the `len(lst) == 4` literal from 2D
  is now `== 6`. Still only fires for OBC; PBC always returns all-bulk.
- **`select_subset`, `qubit_select`, `select_bulk`**: still 2D, will crash
  if called from 3D code. Not blocking ED tests; rewrite when observables
  module is ported.

### New tiny test files (additive — don't replace the existing ones)

```
Three_TC/tests/
├── _path.py             sys.path shim (Three_TC + repo root)
├── test_geometry.py     7 pure-Python checks, no netket. Runs at L=2 and L=3.
└── test_hamiltonian.py  3 netket smoke checks. No ED — only one matvec.
```

`test_geometry.py` covers: counts, no-missing-lookups, stabilizer shapes,
per-qubit incidence (2 vertices / 4 plaquettes), **symplectic commutation
of every (A_v, B_p) pair** (the single most diagnostic check), `∏A_v = I`,
and **xy-layer `∏B_p = I`** (3D-specific PBC dependency that catches
plaquette-orientation bugs invisible in 2D).

`test_hamiltonian.py` covers, all on |all-up⟩ at L=2 PBC, h=0:
`⟨H⟩ = −N_p`, `H|ψ⟩` has exactly `1 + N_v` nonzero amplitudes with the
diagonal one at `−N_p` and the rest at `−1`, and `H = H†`. Uses
`H.to_sparse()` but never diagonalises.

These complement (not replace) `test_exact_diag.py` / `colab_exact_diag.py`
— they're the fast smoke-tests you'd run while editing geometry.

### Pedagogical points worth keeping for future explainers

- The `vertex_all` / `plaq_all` return is **transposed** from the
  intermediate `neighbors` stack: the outer loop in the return statement is
  over vertices, the inner over directions. Result is `(N_v, 6)`, not
  `(6, N_v)`. Easy to misread.
- The Hamiltonian module is **dimension-agnostic** — it iterates index
  lists, so 6-tuples just work. The only consumer assumption is "each
  entry is a list of qubit ids."
- `-1` in `vertex_all` only ever appears under OBC (missing-neighbor
  sentinel). PBC code paths never produce it; downstream `-1` filtering is
  inert there.

---

## Where the 2D code lives (as reference, not to modify)

The original Kufel et al. repo is the parent of this directory:

- `model/geometry.py`, `model/hamiltonian.py`, `model/networks.py`
- `simulation/optimizer.py` — TDVP inlined (instructive)
- `simulation/custom_sampler.py` — `WeightedRule + MultiRule`
- `scripts/run_example.sh` — bash wrapper for `main.py`
- `outputs/` — JSON + mpack from previous runs

User has been instructed to read these for the 2D reference but not edit them.

---

## Session addendum — validation harness + fermionic Hamiltonian (2026-06)

The headline comparison run from "What's NOT built yet" item 1 is now **built**
(though not yet executed end-to-end — it needs the Colab reference JSONs). See
`notes/pipeline.md` for the full pipeline and `notes/progress_log.md` Checkpoint 2.

- **`Three_TC/validation.py`** scores ansätze against the Colab L=2 exact JSON:
  `eps_E`, `Vscore`, `Δ⟨A_v/B_p/σ_x/σ_z⟩` (+MC error +pull), cost. Both models via
  `run_validation(..., fermionic=…)`. Fidelity is out (JSON has no state vector).
- **`create_hamiltonian_fermionic`** (`Three_TC/model/hamiltonian.py`) is the
  NetKet decorated-plaquette Hamiltonian — fermionic NQS training is now possible.
- **`colab_exact_diag.py`** gained `PARAMS["fermionic"]` and a `"model"` JSON tag.
- The same `ToricCNN`/`ToricCNN_full` ansätze validate **both** models (A_v is
  unchanged by the decoration).
- **Correction:** the pinned magnetization is **⟨σ_z⟩**, not ⟨σ_x⟩ (verified
  `log ψ(x)=log ψ(−x)` exactly). Discriminators: Δ⟨σ_z⟩, Δ⟨A_v⟩. (Fixed above.)
