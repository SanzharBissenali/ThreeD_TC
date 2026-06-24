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

import jax
jax.config.update("jax_enable_x64", True)  # float64 SR/QGT (esp. on GPU)

from Three_TC.builders import build_state, run_loop, with_defaults
from Three_TC.validation import nqs_observables
from Three_TC.utils.wandb_logger import init_run, log_step, finish_run
from utils.config import setup_environment
from utils.io import save_model


TRAIN_DEFAULTS: Dict[str, Any] = {
    "n_iter": 100, "dt": 2e-2, "diag_shift": 2e-4, "lr_min": 2e-3,
    "out_dir": "outputs", "wandb": True,
    "wandb_project": "approx-sym-3D-TC",
    "wandb_entity": "models-california-institute-of-technology-caltech",
    "tags": None, "name": None,
}

# Hardcoded reference points from threed_bosonic.json (L=2 PBC bosonic, hx=0.2,
# J=1): label -> (h_z, E_exact, gap). Selected with --hz_preset; sets both the
# field and the E_exact used for the delta figure of merit.
HZ_PRESETS: Dict[str, tuple] = {
    "hard": (0.1184210526315789, -32.2968435820, 0.062),   # small gap (hardest)
    "mid":  (0.3157894736842105, -33.9620095053, 0.943),   # validated point
    "easy": (0.5526315789473684, -38.5935624665, 3.452),   # large gap (easiest)
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

    # h_z preset -> set the field AND the E_exact used for the delta FOM.
    # --exact_E0 (or config["exact_E0"]) is the manual fallback at any h_z.
    if config.get("hz_preset"):
        hz, e0, _gap = HZ_PRESETS[config["hz_preset"]]
        cfg["hz"], cfg["exact_E0"] = hz, e0
    else:
        cfg["exact_E0"] = config.get("exact_E0")
    exact_E0 = cfg.get("exact_E0")

    # Device detection (reused util): picks GPU if present and returns the
    # default chain count (1024 GPU / 16 CPU). An explicit --n_chains still wins.
    _gpu, _node, n_chains_auto = setup_environment()
    is_gpu = n_chains_auto > 16          # setup_environment: 1024 GPU / 16 CPU
    if "n_chains" not in config:
        cfg["n_chains"] = n_chains_auto
    # Double the sample budget on GPU (cheap there) for lower-variance gradients;
    # an explicit --n_samples still wins.
    if "n_samples" not in config and is_gpu:
        cfg["n_samples"] = 2 * cfg["n_samples"]

    name = _run_name(cfg)
    cfg["name"] = name
    os.makedirs(cfg["out_dir"], exist_ok=True)

    geo, hi, Ham, vs, xz_stabs = build_state(cfg)
    # Resolved run metadata -> W&B config (and saved JSON): the param count and the
    # ACTUAL sampler sweep size. build_sampler defaults n_sweeps to geo.N*2 when it
    # is unset, so without this the raw config would log n_sweeps=None.
    cfg["n_params"] = int(vs.n_parameters)
    cfg["n_sweeps"] = int(vs.sampler.sweep_size)
    print(f"[train] {name}: N={geo.N}  n_params={cfg['n_params']}  model={cfg['model']}"
          f"  n_chains={cfg['n_chains']}  n_sweeps={cfg['n_sweeps']}"
          + (f"  E_exact={exact_E0}" if exact_E0 is not None else ""))

    run = None
    if cfg["wandb"]:
        run = init_run(project=cfg["wandb_project"], entity=cfg["wandb_entity"],
                       config=cfg, name=name, group=cfg.get("wandb_group"),
                       tags=cfg["tags"] or [cfg["model"], cfg["arch"], f"L={cfg['L']}"])

    curve = {"step": [], "energy": [], "energy_err": [], "energy_spread": [], "delta": []}

    def on_step(step, E, vs):
        e   = float(np.real(E.mean))
        de  = float(np.real(E.error_of_mean))      # delta_E (MC error on the mean)
        var = float(np.real(E.variance))
        delta = abs(e - exact_E0) / abs(exact_E0) if exact_E0 is not None else None
        curve["step"].append(step)
        curve["energy"].append(e)
        curve["energy_err"].append(de)
        curve["energy_spread"].append(np.sqrt(var))
        curve["delta"].append(delta)
        msg = (f"  step {step:4d}/{cfg['n_iter']}:  E = {e:+.6f} ± {de:.6f}"
               f"   (spread, sqrt(var) = {np.sqrt(var):.4f})")
        if delta is not None:
            msg += f"   delta = {delta:.3e}"
        print(msg, flush=True)
        if run is not None:
            log_step(run, step, E, vs, exact_E0=exact_E0)

    t0 = time.time()
    run_loop(vs, Ham, n_iter=cfg["n_iter"], dt=cfg["dt"],
             diag_shift=cfg["diag_shift"], on_step=on_step, lr_min=cfg["lr_min"],
             qgt=cfg.get("qgt", "auto"))
    runtime_s = time.time() - t0

    obs = nqs_observables(vs, Ham, geo, xz_stabs=xz_stabs)
    if exact_E0 is not None:                               # final FOM -> run.summary
        obs["E_exact"] = exact_E0
        obs["delta"] = abs(obs["E0"] - exact_E0) / abs(exact_E0)
    print(f"[train] done in {runtime_s:.1f}s  E={obs['E0']:.4f}  Vscore={obs['Vscore']:.2e}  "
          + (f"delta={obs['delta']:.3e}  " if exact_E0 is not None else "")
          + f"<A_v>={obs['A_v_mean']:.3f}  <sz>={obs['sz_mean']:.3f}")

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
        try:                                           # weights as W&B artifact —
            import wandb                                # must run BEFORE finish_run,
            art = wandb.Artifact(name.replace("/", "_"), type="model")  # which calls
            art.add_file(f"{weights_base}.mpack")       # run.finish() (closes the run)
            run.log_artifact(art)
        except Exception as e:                         # noqa: BLE001
            print(f"[train] W&B artifact upload skipped: {e}")
        finish_run(run, vs, Ham, geo,
                   extra={"runtime_s": runtime_s, "n_params": int(vs.n_parameters)},
                   observables=obs)

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
    p.add_argument("--hz_preset", choices=list(HZ_PRESETS), default=D,
                   help="set h_z AND E_exact from a hardcoded ED reference point "
                        "(hard/mid/easy); enables the delta figure of merit")
    p.add_argument("--exact_E0", type=float, default=D,
                   help="E_exact for the delta FOM at a custom h_z (alternative to "
                        "--hz_preset)")
    # Architecture
    p.add_argument("--arch", choices=["ToricCNN", "ToricCNN_full"], default=D)
    p.add_argument("--hidden", type=int, default=D)
    p.add_argument("--noninv_channels", type=int, default=D,
                   help="ToricCNN_full: edge channels C in each pre-Wilson block")
    p.add_argument("--n_noninv", type=int, default=D,
                   help="ToricCNN_full: number of non-invariant blocks before Wilson")
    p.add_argument("--inv_hidden", type=int, nargs="*", default=D,
                   help="ToricCNN_full: post-Wilson hidden widths, e.g. --inv_hidden 16 16")
    # Training
    p.add_argument("--n_iter", type=int, default=D)
    p.add_argument("--dt", type=float, default=D, help="(initial) learning rate")
    p.add_argument("--lr_min", type=float, default=D,
                   help="if set, cosine-decay lr from --dt down to this over n_iter")
    p.add_argument("--diag_shift", type=float, default=D)
    p.add_argument("--qgt", choices=["auto", "dense", "onthefly"], default=D,
                   help="SR geometric tensor: dense (QGTJacobianDense — fast, robust; "
                        "wants n_samples >= n_params), onthefly (matrix-free CG), or "
                        "auto (dense iff n_params <= 8192). Use 'dense' on GPU — the "
                        "onthefly/CG path is the one that fails there.")
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
    p.add_argument("--wandb_group", default=D,
                   help="wandb group tying a sweep's runs together for comparison "
                        "(e.g. the SLURM job name)")
    p.add_argument("--no_wandb", action="store_true", help="disable W&B logging")

    cfg = vars(p.parse_args())
    # --no_wandb only forces wandb off; otherwise leave it to TRAIN_DEFAULTS.
    if cfg.pop("no_wandb", False):
        cfg["wandb"] = False
    return cfg


if __name__ == "__main__":
    train(_parse_args())
