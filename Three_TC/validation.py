"""
Three_TC/validation.py
─────────────────────────────────────────────────────────────────────────────
NQS-architecture validation harness for the L=2 PBC 3D toric code — both the
**bosonic** and the **fermionic** (decorated-plaquette) models.

Goodness is measured against an EXACT reference produced on Colab by
`Three_TC/tests/colab_exact_diag.py` (a JSON of expectation values; we never
re-diagonalise locally and the exact state vector is not available, so fidelity
is intentionally absent). A reference JSON carries a `"model"` field
("bosonic" or "fermionic"); references without it are treated as bosonic.

Per (model, architecture, hyperparameter config, h_z regime) we report
    eps_E   = (E_NQS - E0)/|E0|                  energy relative error (variational)
    Vscore  = N * Var(H) / <H>^2                 reference-free quality (-> 0 for an eigenstate)
    dA, dB  = |<A_v>_NQS - <A_v>_exact|, ...      absolute stabilizer deviations
    dMx,dMz = |<sigma_x>_NQS - <sigma_x>_exact|   absolute magnetization deviations
each NQS expectation carries its MC error_of_mean and a pull = deviation/err,
alongside the cost axis (n_params, runtime_s, n_iter).

Construction (geometry / Hamiltonian / ansatz / sampler / state) and the
optimization loop are shared with the training pipeline via `Three_TC.builders`,
so the model validation scores is exactly the model `train.py` trains.

The two architectures under test:
    fully-symmetric        ToricCNN        (A_v exactly enforced; pinned <sigma_z>=0, <A_v>=1)
    approximately-symmetric ToricCNN_full   (identity-init non-invariant block can break both)

See notes/pipeline.md for the end-to-end pipeline.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import netket as nk

# Re-export the shared builders (single source of truth in Three_TC.builders).
from Three_TC.builders import (  # noqa: F401
    build_geometry, build_hamiltonian, build_model, build_sampler,
    build_state, run_loop)


# =============================================================================
# Exact reference (loaded from a colab_exact_diag.py JSON — no local ED)
# =============================================================================

# Scalars we compare against, as emitted by colab_exact_diag.py's `result` dict.
_REF_KEYS = ("E0", "gap", "A_v_mean", "B_p_mean", "sx_mean", "sz_mean",
             "hx", "hz", "N")


def load_reference(json_path: str) -> Dict[str, float]:
    """Parse a colab_exact_diag.py output JSON into the exact comparison scalars."""
    with open(json_path) as f:
        d = json.load(f)
    ref = {k: d[k] for k in _REF_KEYS if k in d}
    ref["model"] = d.get("model", "bosonic")   # legacy bosonic JSONs lack the field
    ref["_path"] = json_path
    return ref


def find_reference(outputs_dir: str, hz: float, hx: float = 0.2,
                   model: str = "bosonic", tol: float = 1e-6) -> Dict[str, float]:
    """Locate the exact-diag JSON in `outputs_dir` matching (model, hx, hz).

    Matches on values *inside* the JSON, not the filename, so it is robust to
    float-formatting in `exact_diag_*.json` names. A JSON with no "model" field
    is treated as bosonic.
    """
    candidates = []
    for path in glob.glob(os.path.join(outputs_dir, "exact_diag_*.json")):
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if (abs(d.get("hx", 1e9) - hx) < tol and abs(d.get("hz", 1e9) - hz) < tol
                and d.get("model", "bosonic") == model):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No {model} exact_diag JSON in {outputs_dir} with hx={hx}, hz={hz}. "
            f"Run colab_exact_diag.py (fermionic={model=='fermionic'}) at that "
            f"point and drop the JSON here."
        )
    if len(candidates) > 1:
        raise ValueError(f"Multiple {model} references match hx={hx}, hz={hz}: {candidates}")
    return load_reference(candidates[0])


# =============================================================================
# Observable operators (mean over sites / stabilizers, one expect call each)
# =============================================================================

def _mean_operators(hi, geo, xz_stabs=None):
    """Build (A_v_mean, B_mean, Mx, Mz) operators, each already divided by count.

    A single summed operator per quantity gives the mean — and a proper MC error
    bar on that mean — in one `vs.expect` call. The plaquette operator B depends
    on the model:
        bosonic  (xz_stabs is None): B_p   = prod sigma^z over geo.plaq_all
        fermionic (xz_stabs given):  B~_p  = (prod sigma^z over z_edges)
                                             (prod sigma^x over x_edges)
    """
    N, N_v = geo.N, len(geo.vertex_all)

    A_op = 0
    for v in geo.vertex_all:
        term = 1
        for i in v:
            if i == -1:
                continue
            term = term * nk.operator.spin.sigmax(hi, int(i))
        A_op = A_op + term

    B_op = 0
    if xz_stabs is None:                                    # bosonic
        for p in geo.plaq_all:
            term = 1
            for i in p:
                if i == -1:
                    continue
                term = term * nk.operator.spin.sigmaz(hi, int(i))
            B_op = B_op + term
        n_B = len(geo.plaq_all)
    else:                                                   # fermionic
        for z_edges, x_edges, _ in xz_stabs:
            term = 1
            for i in z_edges:
                if i != -1:
                    term = term * nk.operator.spin.sigmaz(hi, int(i))
            for i in x_edges:
                if i != -1:
                    term = term * nk.operator.spin.sigmax(hi, int(i))
            B_op = B_op + term
        n_B = len(xz_stabs)

    Mx = sum(nk.operator.spin.sigmax(hi, i) for i in range(N))
    Mz = sum(nk.operator.spin.sigmaz(hi, i) for i in range(N))

    return A_op / N_v, B_op / n_B, Mx / N, Mz / N


def _m(s):
    """(mean, err) of a netket Stats object, real part."""
    return float(np.real(s.mean)), float(np.real(s.error_of_mean))


def nqs_observables(vs, Ham, geo, xz_stabs=None) -> Dict[str, float]:
    """Raw NQS expectation values + errors (NO exact reference needed).

    Keys mirror the colab_exact_diag.py reference schema (E0, A_v_mean, …) so the
    training pipeline can write them straight to its local JSON and W&B. The
    plaquette observable is B_p (bosonic) or the decorated B~_p (fermionic).
    """
    N = geo.N
    E = vs.expect(Ham)
    E_mean = float(np.real(E.mean))
    E_err = float(np.real(E.error_of_mean))
    E_var = float(np.real(E.variance))

    A_op, B_op, Mx_op, Mz_op = _mean_operators(vs.hilbert, geo, xz_stabs=xz_stabs)
    A_m, A_e = _m(vs.expect(A_op))
    B_m, B_e = _m(vs.expect(B_op))
    Mx_m, Mx_e = _m(vs.expect(Mx_op))
    Mz_m, Mz_e = _m(vs.expect(Mz_op))

    return {
        "E0": E_mean, "E_err": E_err, "E_var": E_var,
        "Vscore": N * E_var / E_mean**2 if E_mean != 0 else float("nan"),
        "A_v_mean": A_m, "A_v_err": A_e,
        "B_p_mean": B_m, "B_p_err": B_e,
        "sx_mean": Mx_m, "sx_err": Mx_e,
        "sz_mean": Mz_m, "sz_err": Mz_e,
    }


def _dev(nqs_mean, nqs_err, exact):
    """Absolute deviation + pull (deviation / MC error). pull is nan if err ~ 0."""
    d = abs(nqs_mean - exact)
    pull = d / nqs_err if nqs_err > 1e-12 else float("nan")
    return d, pull


def nqs_metrics(vs, Ham, geo, ref: Dict[str, float], xz_stabs=None) -> Dict[str, float]:
    """Goodness metrics for `vs` against exact `ref` (a load_reference dict).

    = `nqs_observables` (raw values) + deviations / pulls vs the reference.
    """
    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz_stabs)

    dA, pA = _dev(obs["A_v_mean"], obs["A_v_err"], ref["A_v_mean"])
    dB, pB = _dev(obs["B_p_mean"], obs["B_p_err"], ref["B_p_mean"])
    dMx, pMx = _dev(obs["sx_mean"], obs["sx_err"], ref["sx_mean"])
    dMz, pMz = _dev(obs["sz_mean"], obs["sz_err"], ref["sz_mean"])

    return {
        # energy
        "E_nqs": obs["E0"], "E_err": obs["E_err"], "E_exact": ref["E0"],
        "eps_E": (obs["E0"] - ref["E0"]) / abs(ref["E0"]),
        "Vscore": obs["Vscore"],
        # observables: NQS value, MC error, abs deviation, pull
        "A_nqs": obs["A_v_mean"], "A_err": obs["A_v_err"], "dA": dA, "pull_A": pA, "A_exact": ref["A_v_mean"],
        "B_nqs": obs["B_p_mean"], "B_err": obs["B_p_err"], "dB": dB, "pull_B": pB, "B_exact": ref["B_p_mean"],
        "Mx_nqs": obs["sx_mean"], "Mx_err": obs["sx_err"], "dMx": dMx, "pull_Mx": pMx, "Mx_exact": ref["sx_mean"],
        "Mz_nqs": obs["sz_mean"], "Mz_err": obs["sz_err"], "dMz": dMz, "pull_Mz": pMz, "Mz_exact": ref["sz_mean"],
    }


# =============================================================================
# Train one (config, regime) and collect a row
# =============================================================================

def train_one(config: Dict[str, Any], hz: float, ref: Dict[str, float],
              *, fermionic: bool = False, hx: float = 0.2, L: int = 2,
              n_iter: int = 100, dt: float = 0.02, diag_shift: float = 2e-3,
              n_samples: int = 4096, n_chains: int = 16, n_discard: int = 8,
              seed: int = 0) -> Dict[str, Any]:
    """Build (via shared builders), train (via shared run_loop), and score one
    architecture at one h_z regime against `ref`."""
    run_cfg = {
        "L": L, "bc": "PBC", "model": "fermionic" if fermionic else "bosonic",
        "hx": hx, "hy": 0.0, "hz": hz, "J": 1.0,
        "arch": config["arch"], "hidden": config.get("hidden", 8),
        "n_samples": n_samples, "n_chains": n_chains, "n_discard": n_discard,
        "seed": seed,
    }
    geo, hi, Ham, vs, xz_stabs = build_state(run_cfg)

    t0 = time.time()
    run_loop(vs, Ham, n_iter=n_iter, dt=dt, diag_shift=diag_shift)  # on_step=None
    runtime_s = time.time() - t0

    row = {
        "model": run_cfg["model"],
        "config": config.get("name", config["arch"]),
        "arch": config["arch"],
        "hidden": config.get("hidden", 8),
        "regime": config.get("_regime", ""),
        "hz": hz, "hx": hx,
        "n_params": int(vs.n_parameters),
        "n_iter": n_iter, "runtime_s": runtime_s, "seed": seed,
    }
    row.update(nqs_metrics(vs, Ham, geo, ref, xz_stabs=xz_stabs))
    return row


# =============================================================================
# Full sweep over configs x regimes
# =============================================================================

def run_validation(configs: List[Dict[str, Any]], regimes: List[Dict[str, Any]],
                   *, fermionic: bool = False, outputs_dir: str = "outputs",
                   hx: float = 0.2, out_path: Optional[str] = None,
                   **train_kwargs) -> List[Dict[str, Any]]:
    """Train every (config, regime) for one model and collect tidy rows.

    fermionic=False -> bosonic; True -> fermionic (decorated plaquettes). The
    matching reference JSON is selected by its "model" field.

    regimes: list of {"label": str, "hz": float, "ref_json": optional path}.
    configs: list of {"name": str, "arch": str, "hidden": int}.
    """
    marker = "fermionic" if fermionic else "bosonic"
    out_path = out_path or f"outputs/validation_L2_{marker}.json"

    # Resolve references up front so a missing JSON fails before any training.
    for rg in regimes:
        rg["_ref"] = (load_reference(rg["ref_json"]) if rg.get("ref_json")
                      else find_reference(outputs_dir, rg["hz"], hx=hx, model=marker))

    rows: List[Dict[str, Any]] = []
    for rg in regimes:
        for cfg in configs:
            cfg = {**cfg, "_regime": rg["label"]}
            print(f"[{marker} | {rg['label']:>13}  hz={rg['hz']:.3f}]  "
                  f"{cfg.get('name', cfg['arch'])} ...", flush=True)
            row = train_one(cfg, rg["hz"], rg["_ref"], fermionic=fermionic,
                            hx=hx, **train_kwargs)
            rows.append(row)
            print(f"    eps_E={row['eps_E']:+.2e}  Vscore={row['Vscore']:.2e}  "
                  f"dMz={row['dMz']:.3f} (pull {row['pull_Mz']:.1f})  "
                  f"dA={row['dA']:.3f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows -> {out_path}")
    return rows
