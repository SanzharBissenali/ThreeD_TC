"""
Minimal Weights & Biases logger for 3D toric code NQS experiments.

Usage:
    from Three_TC.utils.wandb_logger import init_run, log_step, finish_run

    run = init_run(
        project="approx-sym-3D-TC",
        entity="YOUR_WANDB_USERNAME",
        config={"Lx": 2, "Ly": 2, "Lz": 2, "bc": "PBC", "hx": 0.0, ...},
        name="cnn_inv_L2_h0",
        tags=["toy", "L=2", "h=0", "CNN_inv_only"],
    )

    for step in range(n_iter):
        driver.advance(1)
        E = vs.expect(Ham)
        log_step(run, step, E, vs)

    finish_run(run, vs, Ham, geo,
               extra={"runtime_s": elapsed,
                      "exact_E0": -32.0,
                      "vertex_flip_diff": 1e-7,
                      "translation_diff": 1e-7})

Install once:  pip install wandb
Auth once:     wandb login   (or set WANDB_API_KEY env var)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def init_run(
    project: str,
    entity: Optional[str],
    config: Dict[str, Any],
    name: Optional[str] = None,
    tags: Optional[list] = None,
):
    """Initialize a wandb run. Returns the run object."""
    import os

    # The wandb-core subprocess inherits these macOS allocator-debug vars and
    # spams stderr ("MallocStackLogging: can't turn off ...") which interleaves
    # with training stdout. Scrub them before wandb.init spawns the subprocess.
    for _v in ("MallocStackLogging", "MallocStackLoggingNoCompact",
               "MallocScribble", "MallocPreScribble"):
        os.environ.pop(_v, None)

    import wandb

    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        name=name,
        tags=tags,
        reinit=True,
    )


def log_step(run, step: int, E, vs) -> None:
    """Log per-step VMC scalars.

    Args:
        run: wandb run object
        step: integer iteration index
        E:   netket Stats object from vs.expect(H)
        vs:  variational state (for sampler/acceptance info)
    """
    acc = float(vs.sampler_state.n_accepted) / max(1, float(vs.sampler_state.n_steps))

    run.log(
        {
            "step": step,
            "energy":              float(np.real(E.mean)),
            "energy_error":        float(np.real(E.error_of_mean)),
            "energy_variance":     float(np.real(E.variance)),
            "tau_corr":            float(np.real(E.tau_corr)),
            "R_hat":               float(np.real(E.R_hat)),
            "mcmc_acceptance":     acc,
        },
        step=step,
    )


def finish_run(run, vs, Ham, geo, extra: Optional[Dict[str, Any]] = None,
               observables: Optional[Dict[str, Any]] = None) -> None:
    """End-of-run logging: stabilizer expectations + any extras you compute.

    If `observables` is given (e.g. from `validation.nqs_observables`), those are
    logged directly and the per-stabilizer recompute below is skipped — this is
    the cheap path used by the training pipeline. Otherwise <A_v>/<B_p> are
    computed here with ~1 vs.expect per stabilizer (keep the system small).
    """
    if observables is not None:
        summary = dict(observables)
        if extra is not None:
            summary.update(extra)
        for k, v in summary.items():
            run.summary[k] = v
        run.finish()
        return

    import netket as nk

    A_means = []
    for v in geo.vertex_all:
        op = 1
        for i in v:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmax(vs.hilbert, int(i))
        A_means.append(float(np.real(vs.expect(op).mean)))

    B_means = []
    for p in geo.plaq_all:
        op = 1
        for i in p:
            if i == -1:
                continue
            op = op * nk.operator.spin.sigmaz(vs.hilbert, int(i))
        B_means.append(float(np.real(vs.expect(op).mean)))

    summary = {
        "A_v_mean": float(np.mean(A_means)),
        "A_v_min":  float(np.min(A_means)),
        "A_v_max":  float(np.max(A_means)),
        "B_p_mean": float(np.mean(B_means)),
        "B_p_min":  float(np.min(B_means)),
        "B_p_max":  float(np.max(B_means)),
        "n_params": int(vs.n_parameters),
    }
    if extra is not None:
        summary.update(extra)

    for k, v in summary.items():
        run.summary[k] = v

    run.finish()
