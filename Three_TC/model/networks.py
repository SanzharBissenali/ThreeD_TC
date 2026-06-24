"""
Three_TC/model/networks.py
─────────────────────────────────────────────────────────────────────────────
Neural-network ansätze for the 3D toric code, with a *geometry-exact* custom
convolution in the spirit of the 2D `model/networks.py::KernelManager`.

Why a custom kernel
-------------------
The 3D toric-code edges live at half-integer coordinates  v + ½ê_c  (c = edge
orientation), so the three orientations are three *interpenetrating* cubic
sublattices, NOT three co-located channels.  A vanilla `nn.Conv` that stuffs the
orientations into the channel axis pretends the x/y/z edge at a cube vertex sit
on top of each other; they don't (they are displaced by ½ a lattice vector in
three different directions).  That half-offset means the grid conv's notion of
"neighbour" is only approximately the physical neighbour.

`KernelManager3D` removes the approximation: for each *output* orientation it
builds one canonical stencil of (integer vertex-shift, input-orientation) pairs
whose member edges fall within a chosen Euclidean radius of the reference edge,
ordered with the "self" edge first.  Plaquettes (three face-normal sublattices)
are handled the same way.  Weights are shared by translation (one weight set per
orientation) → exact integer-translation equivariance, exactly like the 2D
hor/vert kernels, now generalised to a 3×3 orientation-pair structure.

Components
----------
    KernelManager3D      — precomputes gather/mask/scatter index arrays from a
                           ThreeD_ToricCodeGeometry (PBC).

    GeoConv3D            — masked-gather convolution on one sublattice family
                           ('edge' or 'plaq'); optional identity-at-self init.

    CNN_invariant_3D     — GeoConv3D on plaquette features (downstream of
                           Wilson). Random init, ELU. A_v-invariant because its
                           inputs are.
    CNN_noninvariant_3D  — GeoConv3D on raw edge features (upstream of Wilson).
                           Identity-at-self init + normalised sigmoid (±1 → ±1),
                           so at step 0 it is a pure pass-through.

    ToricCNN             — Wilson → CNN_invariant ×2 → mean      (symmetric-only)
    ToricCNN_full        — CNN_noninvariant → Wilson → CNN_invariant ×2 → mean
                           At step 0 the non-invariant block is identity, so the
                           full model reduces exactly to ToricCNN.

Because GeoConv3D consumes and produces features in flat qubit-index / plaquette
order, the full model no longer needs the old edges_3D gather + argsort scatter:
the conv handles the geometry internally and Wilson indexes its output directly.

Note on L=2: a half-lattice-exact kernel still cannot separate the +ê and −ê
neighbour when L=2 (they are the same site under PBC). That degeneracy is
intrinsic to L=2, not to the kernel; it is orthogonal to the half-offset
geometry this module fixes.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, List

import numpy as np
import jax.numpy as jnp
import flax.linen as nn


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper (kept for callers/tests; unused by the geometry-exact models)
# ─────────────────────────────────────────────────────────────────────────────

def compute_edges_3D(geo) -> np.ndarray:
    """(3, Lx, Ly, Lz) array of qubit indices, edges_3D[c, ix, iy, iz] = flat
    index of the edge at cube vertex (ix,iy,iz) pointing in direction c."""
    Lx, Ly, Lz = geo.Lx, geo.Ly, geo.Lz
    e = np.eye(3)
    L_box = np.array([Lx, Ly, Lz])
    edges_3D = np.zeros((3, Lx, Ly, Lz), dtype=int)
    for c in range(3):
        offset = 0.5 * e[c]
        for ix in range(Lx):
            for iy in range(Ly):
                for iz in range(Lz):
                    coord = np.array([ix, iy, iz], dtype=float) + offset
                    if geo.bc == "PBC":
                        coord = coord % L_box
                    edges_3D[c, ix, iy, iz] = geo._mapping3Dto1D(coord)
    return edges_3D


# ─────────────────────────────────────────────────────────────────────────────
# Stencil manager: turns geometry into per-orientation gather/mask/scatter arrays
# ─────────────────────────────────────────────────────────────────────────────

def _complement(c: int) -> List[int]:
    return [i for i in range(3) if i != c]


class KernelManager3D:
    """Geometry-exact convolution stencils for the 3D toric code (PBC).

    Public arrays (numpy):
        edge_gather (3, P, S_e) int   — for each output edge of orientation c
                                         (P = Lx·Ly·Lz of them), the qubit
                                         indices of its S_e stencil neighbours.
        edge_mask   (3, P, S_e) float — 1 where the neighbour exists (all 1 in
                                         PBC; the column is kept for OBC reuse).
        edge_out    (3, P)      int    — qubit index of each output edge
                                         (a permutation of 0..N-1).
        plaq_gather/plaq_mask/plaq_out — same, on the plaquette sublattices,
                                         in geometry.plaq_all order.
        self_index  = 0                — the stencil column that is the edge/
                                         plaquette itself (identity tap).

    The same stencil order is reused for every site of a given orientation, so a
    single weight per (orientation, stencil-column) gives exact translation
    equivariance.
    """

    def __init__(self, geo, radius_edge: float = 1.05, radius_plaq: float = 1.05):
        if geo.bc != "PBC":
            raise NotImplementedError(
                "KernelManager3D supports PBC only (CIRCULAR neighbourhoods).")
        self.geo = geo
        self.Lx, self.Ly, self.Lz = geo.Lx, geo.Ly, geo.Lz
        self.N = geo.N
        self.N_plaq = len(geo.plaq_all)
        self.radius_edge = float(radius_edge)
        self.radius_plaq = float(radius_plaq)

        self.edge_gather, self.edge_mask, self.edge_out, self.S_e = \
            self._build_edge_stencils(self.radius_edge)
        self.plaq_gather, self.plaq_mask, self.plaq_out, self.S_p = \
            self._build_plaq_stencils(self.radius_plaq)
        self.self_index = 0  # 'self' (dv=0, same orientation) sorts first

        # hashable/equatable by geometry so flax can hold it as a static field
        # without spurious recompiles
        self._key = (self.Lx, self.Ly, self.Lz, geo.bc,
                     round(self.radius_edge, 6), round(self.radius_plaq, 6))

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, KernelManager3D) and self._key == other._key

    # --- canonical stencil over (vertex-shift, input-orientation) ------------

    @staticmethod
    def _stencil(out_pos: np.ndarray, in_offsets, radius: float):
        """Ordered [(dv (3,) int, c_in)] within `radius` of out_pos.
        `in_offsets[c]` is the sub-cell offset of input-orientation c."""
        rmax = int(np.ceil(radius))
        rng = range(-rmax, rmax + 1)
        items = []
        for dv in itertools.product(rng, rng, rng):
            dv_f = np.array(dv, dtype=float)
            for c_in, off in enumerate(in_offsets):
                d = float(np.linalg.norm(dv_f + off - out_pos))
                if d <= radius + 1e-9:
                    items.append((round(d, 6), c_in, tuple(int(x) for x in dv)))
        # self (d=0, c_in==c_out, dv=0) sorts first
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        return [(np.array(dv, dtype=int), c_in) for (_, c_in, dv) in items]

    def _build_edge_stencils(self, radius: float):
        e = np.eye(3)
        in_offsets = [0.5 * e[c] for c in range(3)]
        L = np.array([self.Lx, self.Ly, self.Lz])
        gather, mask, out = [], [], []
        S_ref = None
        for c_out in range(3):
            stencil = self._stencil(0.5 * e[c_out], in_offsets, radius)
            if S_ref is None:
                S_ref = len(stencil)
            assert len(stencil) == S_ref, "stencil length must match by symmetry"
            g_c, m_c, o_c = [], [], []
            for ix, iy, iz in itertools.product(
                    range(self.Lx), range(self.Ly), range(self.Lz)):
                v = np.array([ix, iy, iz], dtype=float)
                o_c.append(self.geo._mapping3Dto1D((v + 0.5 * e[c_out]) % L))
                row, mrow = [], []
                for dv, c_in in stencil:
                    idx = self.geo._mapping3Dto1D((v + dv + 0.5 * e[c_in]) % L)
                    row.append(idx if idx != -1 else 0)
                    mrow.append(1.0 if idx != -1 else 0.0)
                g_c.append(row)
                m_c.append(mrow)
            gather.append(g_c)
            mask.append(m_c)
            out.append(o_c)
        return (np.asarray(gather, dtype=int), np.asarray(mask, dtype=float),
                np.asarray(out, dtype=int), S_ref)

    def _build_plaq_stencils(self, radius: float):
        e = np.eye(3)
        # face-normal c → plaquette-centre offset ½(ê_a + ê_b)
        plaq_off = []
        for c in range(3):
            a, b = _complement(c)
            plaq_off.append(0.5 * (e[a] + e[b]))
        nside = self.Lx * self.Ly * self.Lz

        def pidx(c, ix, iy, iz):  # geometry.plaq_all order: c outer, iz fastest
            return (c * nside
                    + ((ix % self.Lx) * self.Ly + (iy % self.Ly)) * self.Lz
                    + (iz % self.Lz))

        gather, mask, out = [], [], []
        S_ref = None
        for c_out in range(3):
            stencil = self._stencil(plaq_off[c_out], plaq_off, radius)
            if S_ref is None:
                S_ref = len(stencil)
            assert len(stencil) == S_ref, "stencil length must match by symmetry"
            g_c, m_c, o_c = [], [], []
            for ix, iy, iz in itertools.product(
                    range(self.Lx), range(self.Ly), range(self.Lz)):
                o_c.append(pidx(c_out, ix, iy, iz))
                row = [pidx(c_in, ix + dv[0], iy + dv[1], iz + dv[2])
                       for dv, c_in in stencil]
                g_c.append(row)
                m_c.append([1.0] * S_ref)
            gather.append(g_c)
            mask.append(m_c)
            out.append(o_c)
        return (np.asarray(gather, dtype=int), np.asarray(mask, dtype=float),
                np.asarray(out, dtype=int), S_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisers / activation
# ─────────────────────────────────────────────────────────────────────────────

def _geo_identity_init(self_index: int):
    """Kernel init that makes GeoConv3D a pass-through at step 0: identity in
    channel space placed on the 'self' stencil column, zeros elsewhere."""
    def init(key, shape, dtype=jnp.float64):
        # shape = (O, C_out, C_in, S)
        O, C_out, C_in, _ = shape
        w = jnp.zeros(shape, dtype=dtype)
        eye = jnp.broadcast_to(jnp.eye(C_out, C_in, dtype=dtype),
                               (O, C_out, C_in))
        return w.at[:, :, :, self_index].set(eye)
    return init


def _normalised_sigmoid(x):
    """sigmoid rescaled so ±1 → ±1 exactly (preserves the identity-init
    pass-through of the non-invariant block; a plain sigmoid would squash
    ±1 → ≈ ±0.73)."""
    return (nn.sigmoid(x) - 0.5) * (2 + 2 * jnp.e) / (jnp.e - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry-exact convolution
# ─────────────────────────────────────────────────────────────────────────────

class GeoConv3D(nn.Module):
    """Masked-gather convolution on one toric-code sublattice family.

    Input/output carry features in flat site order:
        input  : (..., C_in,  M)
        output : (..., C_out, M)
    with M = N for ``lattice='edge'`` (qubit order) or M = N_plaq for
    ``lattice='plaq'`` (geometry.plaq_all order).

    Weights have shape (3, C_out, C_in, S) — one stencil per output orientation,
    shared across all sites of that orientation (translation equivariance).
    """
    km: Any
    lattice: str                       # 'edge' or 'plaq'
    features_out: int
    activation: Callable = nn.elu
    identity_init: bool = False
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):
        if self.lattice == "edge":
            gather = jnp.asarray(self.km.edge_gather)   # (3, P, S)
            mask = jnp.asarray(self.km.edge_mask, dtype=self.dtype)
            out_idx = jnp.asarray(self.km.edge_out)     # (3, P)
            M = self.km.N
        elif self.lattice == "plaq":
            gather = jnp.asarray(self.km.plaq_gather)
            mask = jnp.asarray(self.km.plaq_mask, dtype=self.dtype)
            out_idx = jnp.asarray(self.km.plaq_out)
            M = self.km.N_plaq
        else:
            raise ValueError(
                f"lattice must be 'edge' or 'plaq', got {self.lattice!r}")

        O, P, S = gather.shape
        C_in = x.shape[-2]
        C_out = self.features_out

        if self.identity_init:
            kinit = _geo_identity_init(self.km.self_index)
        else:
            kinit = nn.initializers.normal(stddev=1.0 / np.sqrt(C_in * S))
        W = self.param("W", kinit, (O, C_out, C_in, S), self.dtype)
        b = self.param("b", nn.initializers.zeros, (O, C_out), self.dtype)

        lead = x.shape[:-2]
        x2 = x.reshape((-1, C_in, M)).astype(self.dtype)     # (B, C_in, M)
        xg = x2[:, :, gather] * mask                         # (B, C_in, O, P, S)
        y = jnp.einsum("ocis,biops->bocp", W, xg)            # (B, O, C_out, P)
        y = y + b[None, :, :, None]

        # scatter (B, O, C_out, P) → (B, C_out, M) via the output-index map
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape((y.shape[0], C_out, O * P))
        flat_idx = out_idx.reshape(-1)                       # permutation of 0..M-1
        out = jnp.zeros((y.shape[0], C_out, M), self.dtype).at[:, :, flat_idx].set(y)
        out = self.activation(out)
        return out.reshape((*lead, C_out, M))


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CNN_invariant_3D(nn.Module):
    """GeoConv3D on A_v-invariant plaquette features. Random init, ELU."""
    km: Any
    features_out: int
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., C_in, N_plaq)
        return GeoConv3D(self.km, "plaq", self.features_out,
                         activation=nn.elu, identity_init=False,
                         dtype=self.dtype)(x)


class CNN_noninvariant_3D(nn.Module):
    """GeoConv3D on raw edge features. Identity-at-self init + normalised
    sigmoid, so step 0 is a pure pass-through (recovers the symmetric net)."""
    km: Any
    features_out: int = 1
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., C_in, N)  qubit order
        return GeoConv3D(self.km, "edge", self.features_out,
                         activation=_normalised_sigmoid, identity_init=True,
                         dtype=self.dtype)(x)


# ─────────────────────────────────────────────────────────────────────────────
# Composed models
# ─────────────────────────────────────────────────────────────────────────────

class ToricCNN(nn.Module):
    """Symmetric-only architecture: Wilson → CNN_invariant ×2 → mean.
    Exactly A_v-invariant; sufficient for the A_v-preserving (h_x) sector."""
    km: Any
    plaq_all: tuple                # (N_plaq, 4) flat qubit indices, for Wilson
    hidden: int = 8
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., N) spins ±1
        plaq_idx = jnp.asarray(self.plaq_all)
        wilson = jnp.prod(x[..., plaq_idx], axis=-1).astype(self.dtype)  # (..., N_plaq)
        h = wilson[..., None, :]                                         # (..., 1, N_plaq)
        h = CNN_invariant_3D(self.km, self.hidden, self.dtype)(h)
        h = CNN_invariant_3D(self.km, 1, self.dtype)(h)
        return jnp.mean(h, axis=(-2, -1))                               # (...,) log ψ


class ToricCNN_full(nn.Module):
    """Full architecture: CNN_noninvariant ×n → Wilson (per channel) →
    CNN_invariant ×(depth) → mean.

    Capacity knobs (defaults reproduce the original 1-channel / [hidden,1] net):
        noninv_channels  C  — edge channels in the pre-Wilson block.
        n_noninv            — number of stacked pre-Wilson layers.
        inv_hidden  (tuple) — post-Wilson hidden widths; () → (hidden,).

    The pre-Wilson layers are eye-initialised on the 'self' stencil column, so at
    step 0 the deformation is the identity on channel 0 (raw spins → true flux)
    and the model is still an exact function of B_p, i.e. exactly A_v-invariant —
    a near-symmetric warm start. Training then learns the A_v-breaking (h_z)
    deformation. With C>1 the extra channels start at 0 and grow under training.
    """
    km: Any
    plaq_all: tuple
    hidden: int = 8
    noninv_channels: int = 4
    n_noninv: int = 2
    inv_hidden: tuple = (4, 4)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., N) spins ±1
        plaq_idx = jnp.asarray(self.plaq_all)
        C = self.noninv_channels

        h = x[..., None, :].astype(self.dtype)                          # (..., 1, N)
        for _ in range(self.n_noninv):
            h = CNN_noninvariant_3D(self.km, C, self.dtype)(h)         # (..., C, N)

        # Wilson 4-product per channel: (..., C, N) → (..., C, N_plaq)
        g = jnp.prod(h[..., plaq_idx], axis=-1)
        for w in (self.inv_hidden or (self.hidden,)):
            g = CNN_invariant_3D(self.km, w, self.dtype)(g)
        g = CNN_invariant_3D(self.km, 1, self.dtype)(g)
        return jnp.mean(g, axis=(-2, -1))                              # (...,) log ψ
