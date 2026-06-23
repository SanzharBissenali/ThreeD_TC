"""
Three_TC/builders.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for turning a `config` dict into a runnable VMC setup.

Both the training pipeline (`Three_TC/train.py`) and the validation harness
(`Three_TC/validation.py`) construct their geometry / Hamiltonian / ansatz /
sampler / variational state *here*, so the two can never drift apart: the model
you train is exactly the model validation scores.

The optimization loop (`run_loop`) also lives here, shared by both front-ends.

Config keys consumed (all optional except where noted; see DEFAULTS):
    System      : L (req), bc, model ∈ {"bosonic","fermionic"}
    Hamiltonian : hx, hy, hz, J
    Architecture: arch ∈ {"ToricCNN","ToricCNN_full"}, hidden
    Sampling    : n_samples, n_chains, n_discard, chunk_size, n_sweeps, seed
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import netket as nk

from simulation.custom_sampler import WeightedRule, MultiRule
from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.model.hamiltonian import (
    create_hamiltonian, create_hamiltonian_fermionic)
from Three_TC.model.fermionic_decoration import fermionic_plaquettes
from Three_TC.model.networks import ToricCNN, ToricCNN_full, KernelManager3D


DEFAULTS: Dict[str, Any] = {
    "bc": "PBC", "model": "bosonic",
    "hx": 0.0, "hy": 0.0, "hz": 0.0, "J": 1.0,
    "arch": "ToricCNN_full", "hidden": 8,
    "n_samples": 4096, "n_chains": 16, "n_discard": 8,
    "chunk_size": None, "n_sweeps": None, "seed": 0,
}


def with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `config` with DEFAULTS filled in and `dtype` derived."""
    cfg = {**DEFAULTS, **config}
    if "L" not in cfg:
        raise KeyError("config must specify system size 'L'")
    cfg.setdefault("dtype", "complex" if cfg["hy"] != 0.0 else "float64")
    return cfg


def _to_tuple(x):
    return tuple(_to_tuple(v) for v in x) if isinstance(x, list) else x


# =============================================================================
# Builders
# =============================================================================

def build_geometry(config: Dict[str, Any]):
    L = config["L"]
    return ThreeD_ToricCodeGeometry(Lx=L, Ly=L, Lz=L, bc=config.get("bc", "PBC"))


def build_hamiltonian(config: Dict[str, Any], geo, hi):
    """Returns (Ham, xz_stabs). xz_stabs is None for the bosonic model."""
    dtype = config.get("dtype", "complex" if config.get("hy", 0.0) != 0.0 else "float64")
    common = dict(hx=config.get("hx", 0.0), hy=config.get("hy", 0.0),
                  hz=config.get("hz", 0.0), J=config.get("J", 1.0), dtype=dtype)
    if config.get("model", "bosonic") == "fermionic":
        xz_stabs = fermionic_plaquettes(geo, J=config.get("J", 1.0))
        Ham = create_hamiltonian_fermionic(
            hi=hi, vertex_all=geo.vertex_all, xz_stabs=xz_stabs,
            bonds=geo.bonds, **common)
        return Ham, xz_stabs
    Ham = create_hamiltonian(
        hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all,
        bonds=geo.bonds, **common)
    return Ham, None


def build_model(config: Dict[str, Any], geo):
    """Instantiate the ansatz named by `config['arch']`.

    The same two ansätze serve both the bosonic and fermionic models: the Wilson
    4-product enforces A_v invariance, and A_v is unchanged by the decoration.
    """
    plaq_tuple = tuple(tuple(p) for p in geo.plaq_all)
    hidden = config.get("hidden", 8)
    arch = config.get("arch", "ToricCNN_full")
    km = KernelManager3D(geo,
                         radius_edge=config.get("radius_edge", 1.05),
                         radius_plaq=config.get("radius_plaq", 1.05))
    if arch == "ToricCNN":
        return ToricCNN(km=km, plaq_all=plaq_tuple, hidden=hidden)
    if arch == "ToricCNN_full":
        return ToricCNN_full(km=km, plaq_all=plaq_tuple, hidden=hidden)
    raise ValueError(f"unknown arch {arch!r} (expected ToricCNN or ToricCNN_full)")


def build_sampler(config: Dict[str, Any], hi, geo):
    """WeightedRule(LocalRule, vertex-cluster MultiRule) — the topological-phase fix."""
    vertex_clusters = np.array(geo.vertex_all)              # (N_v, 6)
    samp_ratio = geo.N / len(vertex_clusters)
    weighted = WeightedRule(
        (samp_ratio / (samp_ratio + 1), 1 - samp_ratio / (samp_ratio + 1)),
        [nk.sampler.rules.LocalRule(), MultiRule(vertex_clusters)],
    )
    n_sweeps = config.get("n_sweeps") or geo.N // 2
    return nk.sampler.MetropolisSampler(
        hi, rule=weighted, n_chains=config.get("n_chains", 16),
        n_sweeps=n_sweeps, dtype=jnp.int8)


def build_state(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any]:
    """Build everything: returns (geo, hi, Ham, vs, xz_stabs)."""
    cfg = with_defaults(config)
    geo = build_geometry(cfg)
    hi = nk.hilbert.Spin(s=1/2, N=geo.N)
    Ham, xz_stabs = build_hamiltonian(cfg, geo, hi)
    model = build_model(cfg, geo)
    sa = build_sampler(cfg, hi, geo)
    vs = nk.vqs.MCState(sa, model, n_samples=cfg["n_samples"],
                        n_discard_per_chain=cfg["n_discard"],
                        chunk_size=cfg["chunk_size"], seed=cfg["seed"])
    return geo, hi, Ham, vs, xz_stabs


# =============================================================================
# Shared optimization loop (one loop, two front-ends)
# =============================================================================

def run_loop(vs, Ham, n_iter: int, dt: float, diag_shift: float,
             on_step: Optional[Callable] = None, lr_min: Optional[float] = None):
    """VMC + Sgd + SR(diag_shift) for n_iter steps.

    Learning rate: constant `dt` by default, or — if `lr_min` is given — a cosine
    decay from `dt` down to `lr_min` across the `n_iter` steps
    (`optax.cosine_decay_schedule`, alpha = lr_min/dt).

    If `on_step` is given it is called as on_step(step, E, vs) each iteration
    (with a fresh `E = vs.expect(Ham)`); pass None to skip per-step expectation
    when only the final state is needed (cheaper).
    """
    if lr_min is not None and lr_min != dt:
        import optax
        lr = optax.cosine_decay_schedule(init_value=dt, decay_steps=n_iter,
                                         alpha=lr_min / dt)
    else:
        lr = dt
    opt = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=diag_shift)
    driver = nk.driver.VMC(Ham, opt, variational_state=vs, preconditioner=sr)
    for step in range(n_iter):
        driver.advance(1)
        if on_step is not None:
            on_step(step, vs.expect(Ham), vs)
    return vs
