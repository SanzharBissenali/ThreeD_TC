"""
Three_TC/train.py
─────────────────────────────────────────────────────────────────────────────
Standalone, config-driven training pipeline for the 3D toric code (bosonic +
fermionic) — the 3D analogue of the inherited 2D `main.py`.

Inputs: hyperparameters, system size L, Hamiltonian fields (hx, hy, hz), and
naming. Outputs: model weights (`.mpack`), a local run JSON (config + final
expectation values + training curve), and W&B curves/observables.

Usage (notebook / Python):
    from Three_TC.train import train
    res = train({"L": 2, "model": "fermionic", "arch": "ToricCNN_full",
                 "hx": 0.2, "hz": 0.2, "n_iter": 200, "wandb": False})

Usage (CLI / cluster):
    python -m Three_TC.train --L 2 --model fermionic --arch ToricCNN_full \
        --hx 0.2 --hz 0.2 --n_iter 200 --no_wandb

Construction and the optimization loop are shared with `validation.py` via
`Three_TC.builders`, so the trained model is exactly what validation scores.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

import numpy as np

from Three_TC.builders import build_state, run_loop, with_defaults
from Three_TC.validation import nqs_observables
from Three_TC.utils.wandb_logger import init_run, log_step, finish_run
from utils.io import save_model


TRAIN_DEFAULTS: Dict[str, Any] = {
    "n_iter": 100, "dt": 2e-2, "diag_shift": 2e-4, "lr_min": 2e-3,
    "out_dir": "outputs", "wandb": True,
    "wandb_project": "approx-sym-3D-TC",
    "wandb_entity": "models-california-institute-of-technology-caltech",
    "tags": None, "name": None,
}


def _run_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("name") or (
        f"{cfg['model']}_{cfg['arch']}_L{cfg['L']}_hx{cfg['hx']}_hz{cfg['hz']}")


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train one NQS run from a config dict; return a results dict.

    Side effects: writes `{out_dir}/{name}.mpack` (weights) and
    `{out_dir}/{name}.json` (config + observables + curve); logs to W&B if
    `config['wandb']`.
    """
    cfg = with_defaults({**TRAIN_DEFAULTS, **config})
    if cfg["hy"] != 0.0:
        raise NotImplementedError(
            "hy != 0 (sign problem) needs a complex ansatz; not supported yet.")
    name = _run_name(cfg)
    cfg["name"] = name
    os.makedirs(cfg["out_dir"], exist_ok=True)

    geo, hi, Ham, vs, xz_stabs = build_state(cfg)
    print(f"[train] {name}: N={geo.N}  n_params={vs.n_parameters}  model={cfg['model']}")

    run = None
    if cfg["wandb"]:
        run = init_run(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                       config=cfg, name=name,
                       tags=cfg["tags"] or [cfg["model"], cfg["arch"], f"L={cfg['L']}"])

    curve = {"step": [], "energy": [], "energy_err": [], "energy_spread": []}

    def on_step(step, E, vs):
        e   = float(np.real(E.mean))
        de  = float(np.real(E.error_of_mean))      # delta_E (MC error on the mean)
        var = float(np.real(E.variance))
        curve["step"].append(step)
        curve["energy"].append(e)
        curve["energy_err"].append(de)
        curve["energy_spread"].append(np.sqrt(var))
        print(f"  step {step:4d}/{cfg['n_iter']}:  E = {e:+.6f} ± {de:.6f}"
              f"   (spread, sqrt(var) = {np.sqrt(var):.4f})", flush=True)
        if run is not None:
            log_step(run, step, E, vs)

    t0 = time.time()
    run_loop(vs, Ham, n_iter=cfg["n_iter"], dt=cfg["dt"],
             diag_shift=cfg["diag_shift"], on_step=on_step, lr_min=cfg["lr_min"])
    runtime_s = time.time() - t0

    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz_stabs)
    print(f"[train] done in {runtime_s:.1f}s  E={obs['E0']:.4f}  Vscore={obs['Vscore']:.2e}  "
          f"<A_v>={obs['A_v_mean']:.3f}  <sz>={obs['sz_mean']:.3f}")

    # --- artifacts: model weights (.mpack) + local run JSON ---
    weights_base = os.path.join(cfg["out_dir"], name)
    save_model(vs, weights_base)                       # writes {weights_base}.mpack

    result = {
        "name": name, "config": cfg, "n_params": int(vs.n_parameters),
        "runtime_s": runtime_s, "observables": obs, "curve": curve,
        "weights": f"{weights_base}.mpack",
    }
    with open(f"{weights_base}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[train] saved {weights_base}.json and {weights_base}.mpack")

    if run is not None:
        finish_run(run, vs, Ham, geo,
                   extra={"runtime_s": runtime_s, "n_params": int(vs.n_parameters)},
                   observables=obs)
        try:                                           # optional: weights as W&B artifact
            import wandb
            art = wandb.Artifact(name.replace("/", "_"), type="model")
            art.add_file(f"{weights_base}.mpack")
            run.log_artifact(art)
        except Exception as e:                         # noqa: BLE001
            print(f"[train] W&B artifact upload skipped: {e}")

    return result


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> Dict[str, Any]:
    # default=SUPPRESS means an omitted flag is ABSENT from the parsed dict, so it
    # falls through to the code defaults (TRAIN_DEFAULTS + builders.DEFAULTS, applied
    # in train()). This keeps those dicts the single source of truth: editing
    # TRAIN_DEFAULTS actually changes the CLI behavior (and what W&B logs). Pass a
    # flag explicitly only when you want to override the code default for that run.
    D = argparse.SUPPRESS
    p = argparse.ArgumentParser(
        description="Train a 3D toric-code NQS (bosonic or fermionic). Omitted "
                    "options fall back to TRAIN_DEFAULTS / builders.DEFAULTS.")
    # System
    p.add_argument("--L", type=int, required=True, help="linear size (Lx=Ly=Lz)")
    p.add_argument("--bc", choices=["PBC", "OBC"], default=D)
    p.add_argument("--model", choices=["bosonic", "fermionic"], default=D)
    # Hamiltonian
    p.add_argument("--hx", type=float, default=D)
    p.add_argument("--hy", type=float, default=D)
    p.add_argument("--hz", type=float, default=D)
    p.add_argument("--J", type=float, default=D)
    # Architecture
    p.add_argument("--arch", choices=["ToricCNN", "ToricCNN_full"], default=D)
    p.add_argument("--hidden", type=int, default=D)
    # Training
    p.add_argument("--n_iter", type=int, default=D)
    p.add_argument("--dt", type=float, default=D, help="(initial) learning rate")
    p.add_argument("--lr_min", type=float, default=D,
                   help="if set, cosine-decay lr from --dt down to this over n_iter")
    p.add_argument("--diag_shift", type=float, default=D)
    p.add_argument("--seed", type=int, default=D)
    # Sampling
    p.add_argument("--n_samples", type=int, default=D)
    p.add_argument("--n_chains", type=int, default=D)
    p.add_argument("--n_discard", type=int, default=D)
    p.add_argument("--chunk_size", type=int, default=D)
    # Output / logging
    p.add_argument("--name", default=D, help="run name (default auto from params)")
    p.add_argument("--out_dir", default=D)
    p.add_argument("--wandb_project", default=D)
    p.add_argument("--wandb_entity", default=D)
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")

    cfg = vars(p.parse_args())
    # --no_wandb only forces wandb off; otherwise leave it to TRAIN_DEFAULTS.
    if cfg.pop("no_wandb", False):
        cfg["wandb"] = False
    return cfg


if __name__ == "__main__":
    train(_parse_args())
