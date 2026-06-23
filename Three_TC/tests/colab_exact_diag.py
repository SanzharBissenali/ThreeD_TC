"""
colab_exact_diag.py
─────────────────────────────────────────────────────────────────────────────
Self-contained exact diagonalisation reference for the 3D toric code.

Pure scipy + numpy. Never builds the Hamiltonian as a stored sparse matrix —
defines H · psi as a function, so peak memory is just a few vectors of size
2^N (~128 MB at L=2 PBC), not a sparse matrix of 12 GB.

DROP-IN USAGE on Colab:
    1) Edit the PARAMS block.
    2) Run.

Output:
    exact_diag_L{L}_hx{hx}_hy{hy}_hz{hz}.json

Feasible up to N≈28 qubits on a modest machine.
3D toric code PBC has N = 3 L^3, so:
    L=2 PBC: N=24  ✓ (~16M states, ~30 seconds, ~1 GB RAM)
    L=3 PBC: N=81  ✗ (2^81 states, infeasible)
"""

# =============================================================================
# Force CPU before any JAX imports get triggered transitively
# =============================================================================
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =============================================================================
# PARAMS — edit these
# =============================================================================
PARAMS = {
    "Lx": 2, "Ly": 2, "Lz": 2,
    "hx": 0.2, "hy": 0.0, "hz": 0.2,
    "J":  1.0,
    "k":  2,            # number of lowest eigenvalues (>=2 for the gap)
    "fermionic": False, # True -> decorated-plaquette (fermionic) toric code
    "out": None,        # auto-named if None
}

# =============================================================================
# Imports
# =============================================================================
import json
import time
from collections import defaultdict

import numpy as np
import scipy.sparse.linalg as spla


# =============================================================================
# Geometry: PBC 3D toric code on the cubic lattice (qubits on edges)
# =============================================================================
class ThreeD_ToricCodeGeometry_PBC:
    def __init__(self, Lx, Ly, Lz):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.N = 3 * Lx * Ly * Lz
        self._build_coords()
        self.vertex_all = self._stabilizers_vertex()
        self.plaq_all   = self._stabilizers_plaquette()

    def _build_coords(self):
        Lx, Ly, Lz = self.Lx, self.Ly, self.Lz
        basis_atoms = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        coords = []
        for ix in range(Lx):
            for iy in range(Ly):
                for iz in range(Lz):
                    for a in basis_atoms:
                        coords.append((ix + a[0], iy + a[1], iz + a[2]))
        # match the original repo's lexsort(z, y, x) ordering
        coords.sort(key=lambda c: (c[2], c[1], c[0]))
        self.arr_coord = np.array(coords)
        self._coord_to_idx = {
            tuple((2 * np.asarray(c)).round().astype(int)): i
            for i, c in enumerate(self.arr_coord)
        }

    def _idx(self, coord):
        L = np.array([self.Lx, self.Ly, self.Lz])
        c = np.asarray(coord) % L
        return self._coord_to_idx[tuple((2 * c).round().astype(int))]

    def _stabilizers_vertex(self):
        out = []
        e = np.eye(3)
        for ix in range(self.Lx):
            for iy in range(self.Ly):
                for iz in range(self.Lz):
                    v = np.array([ix, iy, iz], dtype=float)
                    nbrs = []
                    for axis in range(3):
                        for sign in (+1, -1):
                            nbrs.append(self._idx(v + 0.5 * sign * e[axis]))
                    out.append(nbrs)
        return out

    def _stabilizers_plaquette(self):
        out = []
        e = np.eye(3)
        for c in range(3):
            a, b = [i for i in range(3) if i != c]
            center_off = 0.5 * e[a] + 0.5 * e[b]
            edge_offs  = [+0.5*e[a], -0.5*e[a], +0.5*e[b], -0.5*e[b]]
            for ix in range(self.Lx):
                for iy in range(self.Ly):
                    for iz in range(self.Lz):
                        center = np.array([ix, iy, iz], dtype=float) + center_off
                        edges  = [self._idx(center + o) for o in edge_offs]
                        out.append(edges)
        return out


# =============================================================================
# Fermionic decoration (self-contained port of fermionic_decoration.py)
# =============================================================================
def fermionic_plaquettes(geo, J=1.0):
    """Decorated plaquette stabilizers as (z_edges, x_edges, coef) triples.

    B~_p = (prod_{e in dp} sigma^z_e) * sigma^x_{e+} * sigma^x_{e-}, with the two
    sigma^x on the corner edges at ctr +/- 0.5*(e_a + e_b + e_c).  Identical
    convention to Three_TC/model/fermionic_decoration.py.
    """
    e = np.eye(3)
    out = []
    for c in range(3):                                   # plaquette normal axis
        a, b = [d for d in range(3) if d != c]           # in-plane axes
        for ix in range(geo.Lx):
            for iy in range(geo.Ly):
                for iz in range(geo.Lz):
                    ctr = np.array([ix, iy, iz], float) + 0.5 * e[a] + 0.5 * e[b]
                    z_edges = [geo._idx(ctr + s * 0.5 * e[ax])
                               for ax in (a, b) for s in (+1, -1)]
                    diag = 0.5 * (e[a] + e[b] + e[c])
                    x_edges = [geo._idx(ctr + diag), geo._idx(ctr - diag)]
                    out.append((z_edges, x_edges, -float(J)))
    return out


# =============================================================================
# Bit manipulation helpers
# =============================================================================
def z_string_eigvals(basis, mask, N):
    """For each basis state b in `basis`, return (-1)^popcount(b & mask) as float64.

    A Z-string ∏_{i in mask} σ_z_i is diagonal in the Z basis; its eigenvalue
    on basis state b is (-1)^(number of 1-bits in b at positions in mask).
    """
    parity = np.zeros_like(basis, dtype=np.int8)
    for i in range(N):
        if mask & (1 << i):
            parity ^= ((basis >> i) & 1).astype(np.int8)
    return 1.0 - 2.0 * parity.astype(np.float64)


# =============================================================================
# Hamiltonian as a LinearOperator (no stored matrix)
# =============================================================================
def make_hamiltonian_op(geo, hx=0.0, hy=0.0, hz=0.0, J=1.0, xz_stabs=None):
    """
    Returns (LinearOperator H, basis array) for the perturbed 3D toric code.

    Action on a state ψ is computed on the fly:
        H ψ = diag · ψ + Σ_{X-strings (mask, c)}  c · ψ[basis XOR mask]
                       + Σ_{Y-strings (mask, c)}  c · i^|mask| · (-1)^bits · ψ[basis XOR mask]
                       + Σ_{XZ-stabs (z,x,c)}     c · (-1)^bits(z) · ψ[basis XOR x]

    xz_stabs: optional list of (z_edges, x_edges, coef) triples (the decorated
        fermionic plaquettes). When given, they REPLACE the bosonic ∏Z plaquette
        term; the coef already carries its sign (-J). z- and x-supports are
        disjoint, so each is a genuine off-diagonal XZ stabilizer.

    Memory: O(2^N) for diag and basis arrays. No sparse matrix is stored.
    """
    N   = geo.N
    dim = 1 << N
    basis = np.arange(dim, dtype=np.int64)
    dtype = np.complex128 if hy != 0 else np.float64
    use_xz = xz_stabs is not None and len(xz_stabs) > 0

    # ------ Diagonal part: -J · ∏Z (plaquettes) - hz · σ_z (single sites) ------
    diag = np.zeros(dim, dtype=dtype)
    if not use_xz:                       # bosonic plaquettes only
        for p in geo.plaq_all:
            mask = 0
            for i in p: mask |= (1 << int(i))
            diag -= J * z_string_eigvals(basis, mask, N)
    if hz != 0:
        for i in range(N):
            diag -= hz * z_string_eigvals(basis, 1 << i, N)

    # ------ Off-diagonal: X-strings (vertex terms + σ_x field) ------
    x_strings = defaultdict(float)
    for v in geo.vertex_all:
        mask = 0
        for i in v: mask |= (1 << int(i))
        x_strings[mask] -= J
    if hx != 0:
        for i in range(N):
            x_strings[1 << i] -= hx
    x_strings = {m: c for m, c in x_strings.items() if c != 0}

    # ------ Off-diagonal: Y-strings (σ_y field only; YYYY terms not handled) ------
    y_strings = {}
    if hy != 0:
        for i in range(N):
            y_strings[1 << i] = -hy

    # Pre-compute Y-string signs (Z-parity at mask) and i^K phase
    y_sign_cache = {}
    for mask in y_strings:
        K = bin(mask).count("1")
        phase = (1j) ** K
        sign  = z_string_eigvals(basis, mask, N).astype(dtype)
        y_sign_cache[mask] = phase * sign  # complex array of shape (dim,)

    # ------ Off-diagonal: XZ stabilizers (decorated fermionic plaquettes) ------
    # Each is coef · (-1)^popcount(b & z_mask) · ψ[b XOR x_mask]. Cache the sign.
    xz_terms = []   # list of (x_mask, coef · sign_array)
    if use_xz:
        for z_edges, x_edges, coef in xz_stabs:
            zm = 0
            for i in z_edges: zm |= (1 << int(i))
            xm = 0
            for i in x_edges: xm |= (1 << int(i))
            sign = z_string_eigvals(basis, zm, N).astype(dtype)
            xz_terms.append((xm, coef * sign))

    def matvec(psi):
        psi = np.asarray(psi).astype(dtype, copy=False)
        out = diag * psi
        for mask, c in x_strings.items():
            out = out + c * psi[basis ^ mask]
        for mask, c in y_strings.items():
            out = out + c * y_sign_cache[mask] * psi[basis ^ mask]
        for xm, csign in xz_terms:
            out = out + csign * psi[basis ^ xm]
        return out

    H = spla.LinearOperator((dim, dim), matvec=matvec, dtype=dtype)
    return H, basis


# =============================================================================
# Observable expectations from the ground-state vector
# =============================================================================
def expect_x_string(psi, basis, mask):
    """⟨ψ| ∏_{i in mask} σ_x_i |ψ⟩  =  Σ_b conj(ψ[b]) · ψ[b XOR mask]."""
    return float(np.real(np.sum(np.conj(psi) * psi[basis ^ mask])))


def expect_z_string(psi, basis, mask, N):
    """⟨ψ| ∏_{i in mask} σ_z_i |ψ⟩  =  Σ_b |ψ[b]|^2 · (-1)^popcount(b & mask)."""
    return float(np.sum((np.abs(psi) ** 2) * z_string_eigvals(basis, mask, N)))


def expect_y_string(psi, basis, mask, N):
    """⟨ψ| ∏_{i in mask} σ_y_i |ψ⟩."""
    K = bin(mask).count("1")
    phase = (1j) ** K
    sign  = z_string_eigvals(basis, mask, N)
    val = phase * np.sum(np.conj(psi) * sign * psi[basis ^ mask])
    return float(np.real(val))


def expect_xz_string(psi, basis, z_mask, x_mask, N):
    """⟨ψ| Z(z_mask) X(x_mask) |ψ⟩ for a decorated plaquette (disjoint supports)."""
    sign = z_string_eigvals(basis, z_mask, N)
    return float(np.real(np.sum(np.conj(psi) * sign * psi[basis ^ x_mask])))


# =============================================================================
# Main
# =============================================================================
def run(params):
    geo = ThreeD_ToricCodeGeometry_PBC(params["Lx"], params["Ly"], params["Lz"])
    print(f"3D toric code PBC: L=({geo.Lx},{geo.Ly},{geo.Lz})")
    print(f"  N qubits      = {geo.N}")
    print(f"  N vertices    = {len(geo.vertex_all)}")
    print(f"  N plaquettes  = {len(geo.plaq_all)}")
    print(f"  Hilbert dim   = 2^{geo.N} = {1 << geo.N}")

    if geo.N > 28:
        raise SystemExit(f"\nN={geo.N} too large for in-memory exact diag.")

    fermionic = params.get("fermionic", False)
    xz_stabs = fermionic_plaquettes(geo, J=params["J"]) if fermionic else None
    if fermionic:
        print(f"  model         = fermionic (decorated plaquettes, |B~_p|={len(xz_stabs)})")

    print("\nBuilding Hamiltonian operator (matrix-free) ...")
    t0 = time.time()
    H, basis = make_hamiltonian_op(
        geo,
        hx=params["hx"], hy=params["hy"], hz=params["hz"], J=params["J"],
        xz_stabs=xz_stabs,
    )
    print(f"  setup took {time.time()-t0:.2f} s   dtype={H.dtype}   shape={H.shape}")

    print(f"\nRunning Lanczos (eigsh, k={params['k']}, which='SA') ...")
    t0 = time.time()
    # 'SA' = smallest algebraic; works for Hermitian operators
    evals, evecs = spla.eigsh(H, k=params["k"], which="SA")
    order = np.argsort(np.real(evals))
    evals = evals[order]
    evecs = evecs[:, order]
    psi0  = evecs[:, 0]
    psi0 /= np.linalg.norm(psi0)
    print(f"  Lanczos took {time.time()-t0:.2f} s")
    for j, e in enumerate(evals):
        print(f"  E_{j} = {np.real(e):.8f}")
    gap = float(np.real(evals[1] - evals[0])) if len(evals) >= 2 else None
    if gap is not None:
        print(f"  gap = E_1 - E_0 = {gap:.6f}")

    # ------ Observables ------
    print("\nComputing ground-state observables ...")
    N = geo.N
    sx = [expect_x_string(psi0, basis, 1 << i) for i in range(N)]
    sz = [expect_z_string(psi0, basis, 1 << i, N) for i in range(N)]
    sy = ([expect_y_string(psi0, basis, 1 << i, N) for i in range(N)]
          if params["hy"] != 0 else [0.0] * N)

    A_v = []
    for v in geo.vertex_all:
        mask = 0
        for i in v: mask |= (1 << int(i))
        A_v.append(expect_x_string(psi0, basis, mask))
    if fermionic:                 # decorated B~_p = Z(z_edges) X(x_edges)
        B_p = []
        for z_edges, x_edges, _ in xz_stabs:
            zm = 0
            for i in z_edges: zm |= (1 << int(i))
            xm = 0
            for i in x_edges: xm |= (1 << int(i))
            B_p.append(expect_xz_string(psi0, basis, zm, xm, N))
    else:                         # bosonic B_p = Z(plaquette)
        B_p = []
        for p in geo.plaq_all:
            mask = 0
            for i in p: mask |= (1 << int(i))
            B_p.append(expect_z_string(psi0, basis, mask, N))

    # ------ Save ------
    model = "fermionic" if fermionic else "bosonic"
    tag = "fermionic_" if fermionic else ""
    out_path = params["out"] or (
        f"exact_diag_{tag}L{params['Lx']}_hx{params['hx']}_hy{params['hy']}_hz{params['hz']}.json"
    )
    result = {
        "model": model,
        "Lx": params["Lx"], "Ly": params["Ly"], "Lz": params["Lz"], "bc": "PBC",
        "N": geo.N, "N_vertices": len(geo.vertex_all),
        "N_plaquettes": len(geo.plaq_all),
        "hx": params["hx"], "hy": params["hy"], "hz": params["hz"], "J": params["J"],
        "dtype": "complex" if params["hy"] != 0 else "float64",
        "E0": float(np.real(evals[0])),
        "E1": float(np.real(evals[1])) if len(evals) >= 2 else None,
        "gap": gap,
        "sx_per_qubit": sx, "sy_per_qubit": sy, "sz_per_qubit": sz,
        "sx_mean": float(np.mean(sx)), "sx_max_abs": float(np.max(np.abs(sx))),
        "sy_mean": float(np.mean(sy)),
        "sz_mean": float(np.mean(sz)),
        "A_v_per_vertex": A_v, "B_p_per_plaq": B_p,
        "A_v_mean": float(np.mean(A_v)), "A_v_min": float(np.min(A_v)),
        "B_p_mean": float(np.mean(B_p)), "B_p_min": float(np.min(B_p)),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

    print("\n--- Reference summary ---")
    print(f"  E_0          = {result['E0']:.6f}")
    print(f"  gap          = {result['gap']:.6f}" if gap is not None else "  gap          = n/a")
    print(f"  <σ_x>        mean={result['sx_mean']:+.4f}  max|.|={result['sx_max_abs']:.4f}")
    print(f"  <σ_y>        mean={result['sy_mean']:+.4f}")
    print(f"  <σ_z>        mean={result['sz_mean']:+.4f}")
    print(f"  <A_v>        mean={result['A_v_mean']:+.4f}  min={result['A_v_min']:+.4f}")
    print(f"  <B_p>        mean={result['B_p_mean']:+.4f}  min={result['B_p_min']:+.4f}")
    return result


if __name__ == "__main__":
    run(PARAMS)
