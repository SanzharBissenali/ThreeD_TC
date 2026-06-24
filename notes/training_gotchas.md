# Training gotchas — what to check when the 3D NQS misbehaves

A running list of training failure modes and suspects, each with the lever to pull
if it resurfaces. Add to it as new ones turn up. For the SR/VMC loop internals see
`vmc_internals.md`; for the end-to-end pipeline see `pipeline.md`.

Started 2026-06-24.

---

## 1. SR linear solver: dense vs on-the-fly, and 2D↔3D parity  ← revisit if convergence stalls

**The knob.** `run_loop(qgt=...)` in `Three_TC/builders.py`. `auto` (default) picks
`QGTJacobianDense` iff `n_params ≤ 8192`, otherwise NetKet's matrix-free
`QGTOnTheFly` + conjugate gradient (CG). `--qgt dense` forces dense.

**Same method, different solve.** The QGT `S` is one object; dense vs on-the-fly only
change *how* `S⁻¹g` is computed (direct solve vs CG). Same gradient `g`, same
`diag_shift`, same update in exact arithmetic — so **the qgt choice does not change
training dynamics in principle.** `diag_shift` is the dynamics knob, not the
representation.

**In practice they can diverge:**
- on-the-fly CG stops at a tol / max-iters; under-converged ⇒ noisier, slower,
  approximate steps.
- on-the-fly **fails on GPU** for the larger sweep nets (the `inv_hidden 16 16`,
  13107–15411-param configs). This was the "skipped runs" bug. dense is exact,
  ~9× faster, and GPU-safe.
- **Conditioning rule:** dense wants `n_samples ≳ n_params`; below that `S` is
  rank-deficient and the solve leans entirely on regularization. All sweep configs
  sit under `n_samples = 16384` (param counts: `(8,)` → 795–2019; `(16,16)` →
  13107–15411).

**2D↔3D parity (the detail to revisit).** The 2D pipeline used dense *exclusively*,
but with a different regularized solve:

| | 2D `simulation/optimizer.py::run_tdvp` | 3D `builders.py::run_loop` (`--qgt dense`) |
|---|---|---|
| QGT | `QGTJacobianDense` (`:52`) | `QGTJacobianDense` (`:166`) |
| solver | `pinv_smooth` (SVD pseudo-inverse, `rtol=1e-30`) | NetKet `SR` default (Cholesky-type) |
| regularization | `diag_shift` + `diag_scale=0` + SVD `rtol` smoothing | `diag_shift`, `holomorphic=False` |
| driver | manual TDVP `θ ← θ + dt·dθ` | `nk.driver.VMC` + `Sgd` + `SR` |

Both are the same `diag_shift`-regularized natural gradient on the dense QGT. They
differ only in the regularized solve: **Cholesky needs an SPD `S` (the `diag_shift`
guarantees it) but handles near-degenerate `S` less gracefully than the SVD-based
`pinv_smooth`** — which is exactly why the 2D code chose `pinv_smooth`.

**Lever (not yet applied).** If 3D training stalls, oscillates, or trails 2D in a
near-singular regime, switch the 3D dense solve to `pinv_smooth` to match 2D — pass a
custom solver to `nk.optimizer.SR(..., solver=...)`, or replicate the manual
`S.solve(pinv_smooth, rtol=..., rtol_smooth=...)` from `run_tdvp`. Deliberately
*not* done now (2026-06-24); come back to it if the problem persists.

## 2. `diag_shift` — the actual dynamics knob

Too small ⇒ noisy/unstable solves near a singular `S` (and NaNs); too large ⇒
over-damped, slow descent. It's a swept axis. If a config diverges or NaNs, **raise
`diag_shift` first** (and/or lower `dt`).

## 3. No NaN guard in 3D `run_loop`

2D `run_tdvp` breaks on a NaN energy (`simulation/optimizer.py:76`); 3D `run_loop`
does **not** — a blown-up run keeps stepping on NaNs instead of stopping. If you see
NaN energies that never halt, that's why; add a guard, or fix the cause via #2.

## 4. GPU / float64

`train.py` forces `jax_enable_x64`; the Colab gate asserts `backend == 'gpu'`. A
silent float32 downcast (x64 off) or a CPU fallback changes convergence *and* timing —
verify both on any new machine before trusting a result that looks off.

## 5. Colab sequential-run fragility (worked around)

The notebook's automated loop could skip runs; replaced with one-cell-per-run. The
suspects were GPU VRAM preallocation across back-to-back subprocesses (mitigated with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` + `ALLOCATOR=platform`) and swallowed nonzero
exits (now checked). The actual skip cause was the missing `--qgt dense` (issue #1).
Revisit only if re-automating the sweep.
