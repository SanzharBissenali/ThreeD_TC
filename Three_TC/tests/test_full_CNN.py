"""
Let's test the full CNN with non-invariant, Wilson nonlinearity, and invariant blocks
unperturbed Hamiltonian, with L = 2, so that number of qubits 
is N = 24 (PBC -> 3 * 2 * 2 * 2)
"""

import _path  # noqa: F401   <-- ADD THIS FIRST
import json
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from tqdm import tqdm
from simulation.custom_sampler import WeightedRule, MultiRule

from Three_TC.model.geometry import ThreeD_ToricCodeGeometry
from Three_TC.model.hamiltonian import create_hamiltonian
from Three_TC.model.networks import ToricCNN, ToricCNN_full, KernelManager3D
import time
from Three_TC.utils.wandb_logger import init_run, log_step, finish_run


geo = ThreeD_ToricCodeGeometry(Lx=2, Ly=2, Lz=2, bc='PBC')
hi = nk.hilbert.Spin(s=1/2, N=geo.N)
Ham = create_hamiltonian(
    hi=hi, vertex_all=geo.vertex_all, plaq_all=geo.plaq_all, 
    bonds=geo.bonds, hx=0.2, hy=0.0, hz=0.2, dtype='float64'
)

# Firstly, let's define a model 

# Build geometry-exact convolution stencils + Wilson plaquette membership
km         = KernelManager3D(geo)
plaq_tuple = tuple(tuple(p) for p in geo.plaq_all)

model = ToricCNN_full(
    km=km,
    plaq_all=plaq_tuple,
    hidden=8,
)

# Let's check the identity-wiring to Machine Precision. UPDATE: IT WORKS. ERROR IS 1e-8

# import jax, jax.numpy as jnp
# key = jax.random.PRNGKey(0)
# sigma = 2 * jax.random.bernoulli(key, shape=(1, geo.N)).astype(jnp.int8) - 1

# # Build the symmetric-only model with the same RNG for the inv block
# from Three_TC.model.networks import ToricCNN
# model_sym = ToricCNN(plaq_all=plaq_tuple, L=geo.Lx, hidden=8)

# params_full = model.init(key, sigma)
# params_sym  = model_sym.init(key, sigma)

# out_full = model.apply(params_full, sigma)
# out_sym  = model_sym.apply(params_sym, sigma)
# print(out_full, out_sym, abs(out_full - out_sym))



# Secondly, we have to implement sampling

# MCMC sampler: single-spin flips, Metropolis
# sa = nk.sampler.MetropolisSampler(
#     hi,
#     rule=nk.sampler.rules.LocalRule(),
#     n_chains=16,
#     n_sweeps=geo.N // 2,      # standard heuristic
#     dtype=jnp.int8,
# )

vertex_clusters = np.array(geo.vertex_all)              # shape (N_v, 6)

samp_ratio = geo.N / len(vertex_clusters)
weighted = WeightedRule(
    (samp_ratio / (samp_ratio + 1), 1 - samp_ratio / (samp_ratio + 1)),
    [nk.sampler.rules.LocalRule(), MultiRule(vertex_clusters)],
)

sa = nk.sampler.MetropolisSampler(
    hi, rule=weighted,
    n_chains=16, n_sweeps=geo.N // 2, dtype=jnp.int8,
)

# Variational state: model + sampler + sample budget
vs = nk.vqs.MCState(sa, model, n_samples=4096, n_discard_per_chain=8)

print(f"N qubits: {geo.N}  |  n_params: {vs.n_parameters}")

# Thirdly, a training loop that lasts for 200 steps, where
# each step calls the sampling, evaluates the Energy, and gradients, 
# update the model weights.

# ---- before the training loop ----
config = {
    # System
    "Lx": geo.Lx, "Ly": geo.Ly, "Lz": geo.Lz, "bc": geo.bc, "N": geo.N,
    "N_plaq": len(geo.plaq_all), "N_vertices": len(geo.vertex_all),
    "hx": 0.2, "hy": 0.0, "hz": 0.2, "J": 1.0,
    "dim": 3, "dtype": "float64",
    "exact_E0": None,   # for L=2 PBC h=0; null otherwise

    # Architecture
    "model_type": "CNN_full",       # or "MLP" for the other run
    "channels_inv":   [3, 8, 1],         # whatever you used
    "channels_noninv": None,             # none yet
    "kernel_size": 3,
    "hidden_dim":  8,
    "activation":  "elu",

    # Training
    "n_iter": 100, "dt": 0.02, "diag_shift": 2*1e-3,
    "optimizer": "SGD", "preconditioner": "SR",
    "seed": 0,

    # Sampling
    "sampler_type": "6-flip",
    "n_samples":  4096,
    "n_chains":   16,
    "n_sweeps":   geo.N // 2,
    "n_discard":  8,
}

opt = nk.optimizer.Sgd(learning_rate=config['dt'])
sr  = nk.optimizer.SR(diag_shift=config['diag_shift'])

driver = nk.driver.VMC(Ham, opt, variational_state=vs, preconditioner=sr)

run = init_run(
    project="approx-sym-3D-TC",        
    entity="models-california-institute-of-technology-caltech",       
    config=config,
    name=f"{config['model_type']}_L{geo.Lx}_hx_0.2_hz_0.2",
    tags=["toy", f"L={geo.Lx}", "h=0", config["model_type"]],
)

t0 = time.time()
energies = []

for step in range(config["n_iter"]):
    driver.advance(1)
    E = vs.expect(Ham)
    energies.append(float(np.real(E.mean)))
    log_step(run, step, E, vs)
    print(f"step {step:3d}: E = {E.mean.real:+.4f}")


# Fourtly, return the list of energies, and the model weights. 
with open("test_full_CNN.json", "w") as f:
    json.dump({
        "energies": energies,
        "n_params": int(vs.n_parameters),
        "N": int(geo.N),
        "exact_E0": None,
    }, f)


# Build a translation map for the qubit lattice.
# T_x shifts every qubit by one unit in the +x direction; the new qubit index
# at position p comes from the old qubit at position p - (1,0,0) (mod L).
# N = geo.N
# shift = +1  # one lattice site in +x

# T = np.zeros(N, dtype=int)
# for i, c in enumerate(geo.arr_coord):
#     new_c = c.copy()
#     new_c[0] = (new_c[0] + shift) % geo.Lx
#     T[i] = geo._mapping3Dto1D(new_c)
# T = jnp.asarray(T)

# # Random spin config
# key = jax.random.PRNGKey(0)
# sigma = 2 * jax.random.bernoulli(key, shape=(N,)).astype(jnp.int8) - 1
# sigma_shifted = sigma[T]   # apply the translation

# # Evaluate log ψ
# log_psi      = vs.model.apply(vs.variables, sigma[None, :])[0]
# log_psi_T    = vs.model.apply(vs.variables, sigma_shifted[None, :])[0]
# print(f"log ψ(σ)     = {log_psi}")
# print(f"log ψ(T_x σ) = {log_psi_T}")
# print(f"difference   = {abs(log_psi - log_psi_T):.2e}  (expect ~1e-7)")


# ---- after the training loop ----
finish_run(run, vs, Ham, geo, extra={
    "runtime_s": time.time() - t0,
    "E_final":   float(np.mean(energies[-10:])),
    "E_final_std": float(np.std(energies[-10:])),
    "gap_to_target": float(np.mean(energies[-10:]) - config["exact_E0"]),
    # if you ran the symmetry checks:
    # "vertex_flip_diff": vertex_diff_value,
    # "translation_diff": translation_diff_value,
})