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
                           ThreeD_ToricCodeGeometry (PBC wraps; OBC masks the
                           neighbours that fall outside the open box).

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
    GeoCNN               — stacked GeoConv3D edge convs → mean, NO Wilson. Same
                           kernel as above but not A_v-invariant; the benchmark
                           that isolates what the Wilson invariance buys.

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
    """Geometry-exact convolution stencils for the 3D toric code (PBC or OBC).

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
        # PBC: neighbourhoods wrap (CIRCULAR). OBC: neighbours that fall outside
        # the open box are masked out (edge_mask/plaq_mask = 0) instead of wrapped,
        # so the same per-orientation stencil is reused with a per-site mask.
        if geo.bc not in ("PBC", "OBC"):
            raise NotImplementedError(f"unknown boundary condition {geo.bc!r}")
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
        pbc = self.geo.bc == "PBC"
        gather, mask, out = [], [], []
        S_ref, P_ref = None, None
        for c_out in range(3):
            stencil = self._stencil(0.5 * e[c_out], in_offsets, radius)
            if S_ref is None:
                S_ref = len(stencil)
            assert len(stencil) == S_ref, "stencil length must match by symmetry"
            g_c, m_c, o_c = [], [], []
            for ix, iy, iz in itertools.product(
                    range(self.Lx), range(self.Ly), range(self.Lz)):
                v = np.array([ix, iy, iz], dtype=float)
                ocoord = v + 0.5 * e[c_out]
                o_idx = self.geo._mapping3Dto1D(ocoord % L if pbc else ocoord)
                if o_idx == -1:
                    continue   # OBC: this output edge lives outside the open box
                o_c.append(o_idx)
                row, mrow = [], []
                for dv, c_in in stencil:
                    ncoord = v + dv + 0.5 * e[c_in]
                    idx = self.geo._mapping3Dto1D(ncoord % L if pbc else ncoord)
                    row.append(idx if idx != -1 else 0)
                    mrow.append(1.0 if idx != -1 else 0.0)
                g_c.append(row)
                m_c.append(mrow)
            if P_ref is None:
                P_ref = len(o_c)
            assert len(o_c) == P_ref, "per-orientation edge count must match by symmetry"
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

        # Coordinate-based gather against geometry.plaq_centers (single source of
        # truth, valid for both BC). A neighbour plaquette of orientation c_in at
        # integer vertex shift dv from the output plaquette sits at centre
        # out_corner + dv + plaq_off[c_in], where out_corner = out_centre -
        # plaq_off[c_out]. PBC wraps inside _plaq_center_to_idx; OBC returns -1
        # (→ mask 0) for neighbours off the open box.
        geo = self.geo
        centers, orient = geo.plaq_centers, geo.plaq_orient
        gather, mask, out = [], [], []
        S_ref, P_ref = None, None
        for c_out in range(3):
            stencil = self._stencil(plaq_off[c_out], plaq_off, radius)
            if S_ref is None:
                S_ref = len(stencil)
            assert len(stencil) == S_ref, "stencil length must match by symmetry"
            g_c, m_c, o_c = [], [], []
            for p in range(len(centers)):
                if orient[p] != c_out:
                    continue
                o_c.append(p)
                out_corner = np.asarray(centers[p], dtype=float) - plaq_off[c_out]
                row, mrow = [], []
                for dv, c_in in stencil:
                    ncenter = out_corner + np.asarray(dv) + plaq_off[c_in]
                    idx = geo._plaq_center_to_idx(ncenter)
                    row.append(idx if idx != -1 else 0)
                    mrow.append(1.0 if idx != -1 else 0.0)
                g_c.append(row)
                m_c.append(mrow)
            if P_ref is None:
                P_ref = len(o_c)
            assert len(o_c) == P_ref, "per-orientation plaquette count must match by symmetry"
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


class VanillaCNN(nn.Module):
    """Plain grid CNN baseline — the 'easy mode' diagnostic ansatz.

    Deliberately does NOT use KernelManager3D/GeoConv3D. The flat spin vector is
    folded into a (Lx, Ly, Lz, 3) tensor (the three edge orientations become a
    channel axis, co-located at each cube vertex) and run through standard
    `nn.Conv` with CIRCULAR padding. This is exactly the half-offset
    approximation the geometry-exact kernel removes — so comparing its MCMC
    acceptance against ToricCNN/ToricCNN_full isolates whether the custom gather/
    scatter kernel is what's collapsing acceptance.

    log ψ(x) is real (sum-pool over sites + final 1-channel conv); fine for the
    Perron–Frobenius-positive (h_y = 0) Hamiltonians here.

    Static fields:
        shape       (3, Lx, Ly, Lz) of the orientation→grid fold.
        edges_flat  flattened qubit indices in `shape` order (a permutation of
                    0..N-1); built once via `compute_edges_3D`.
    """
    shape: tuple                       # (3, Lx, Ly, Lz)
    edges_flat: tuple                  # len N = 3·Lx·Ly·Lz
    hidden: int = 8
    depth: int = 2
    kernel_size: int = 3
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., N) spins ±1
        O, Lx, Ly, Lz = self.shape
        idx = jnp.asarray(self.edges_flat).reshape(self.shape)   # (3, Lx, Ly, Lz)
        lead = x.shape[:-1]
        g = x[..., idx]                          # (..., 3, Lx, Ly, Lz)
        g = jnp.moveaxis(g, -4, -1)              # (..., Lx, Ly, Lz, 3) channels-last
        g = g.reshape((-1, Lx, Ly, Lz, O)).astype(self.dtype)    # (B, Lx, Ly, Lz, 3)
        ks = (self.kernel_size,) * 3
        for _ in range(self.depth):
            g = nn.Conv(features=self.hidden, kernel_size=ks, padding="CIRCULAR",
                        param_dtype=self.dtype)(g)
            g = nn.elu(g)
        g = nn.Conv(features=1, kernel_size=ks, padding="CIRCULAR",
                    param_dtype=self.dtype)(g)
        out = jnp.sum(g, axis=(1, 2, 3, 4))      # (B,) real log ψ
        return out.reshape(lead)


class VanillaWilsonCNN(nn.Module):
    """Wilson-sandwich architecture built from STANDARD grid convs (nn.Conv +
    CIRCULAR padding), NOT GeoConv3D.

    Same information flow as ToricCNN_full —
        edge features → per-channel Wilson 4-product (flux) → plaquette features
        → mean log ψ
    — but every conv folds the three sublattice orientations into a channel axis
    and runs a vanilla cubic grid conv (the half-offset approximation that
    KernelManager3D removes). Lets you A/B the geometry-exact kernel against a
    plain conv while *keeping* the Wilson nonlinearity. Defaults reproduce the
    requested noninv=[1], inv=[4,1].

    Static fields:
        shape       (3, Lx, Ly, Lz) of the edge orientation→grid fold.
        edges_flat  flattened qubit indices in `shape` order (perm of 0..N-1).
        plaq_all    (N_plaq, 4) flat qubit indices, for the Wilson product.
    Plaquette flat order is itself ravel(c, ix, iy, iz) over (3, Lx, Ly, Lz)
    (geometry.plaq_all order), so the plaquette fold needs no index map.
    """
    shape: tuple                       # (3, Lx, Ly, Lz)
    edges_flat: tuple                  # len N
    plaq_all: tuple                    # (N_plaq, 4)
    noninv_channels: int = 1
    n_noninv: int = 1
    inv_hidden: tuple = (4,)
    kernel_size: int = 3
    noninv_identity: bool = True       # identity-init noninv block (step-0 pass-through)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., N) spins ±1
        O, Lx, Ly, Lz = self.shape              # O = 3
        ks = (self.kernel_size,) * 3
        idx_flat = jnp.asarray(self.edges_flat)         # (N,) grid-cell → qubit id
        plaq_idx = jnp.asarray(self.plaq_all)           # (N_plaq, 4)
        lead = x.shape[:-1]
        x2 = x.reshape((-1, x.shape[-1])).astype(self.dtype)   # (B, N)
        B, N = x2.shape
        N_plaq = O * Lx * Ly * Lz

        def _grid_identity_init(key, shape, dtype=jnp.float64):
            # shape = (k, k, k, C_in_total, C_out_total); eye at spatial centre →
            # step-0 pass-through (channel i → channel i), zeros elsewhere.
            c = shape[0] // 2
            w = jnp.zeros(shape, dtype=dtype)
            return w.at[c, c, c].set(jnp.eye(shape[-2], shape[-1], dtype=dtype))

        def grid_conv(h, idx, M, C_out, act, identity=False):
            """(B, C_in, M) flat → fold to (Lx,Ly,Lz, C_in·3) → nn.Conv → (B, C_out, M).
            idx maps flat slot → grid-cell order (None ⇒ identity, for plaquettes)."""
            C_in = h.shape[1]
            hg = h if idx is None else h[:, :, idx]      # (B, C_in, M) grid order
            hg = hg.reshape((B, C_in, O, Lx, Ly, Lz))
            hg = jnp.transpose(hg, (0, 3, 4, 5, 1, 2)).reshape(
                (B, Lx, Ly, Lz, C_in * O))               # channels-last (C_in·O)
            kw = (dict(kernel_init=_grid_identity_init, bias_init=nn.initializers.zeros)
                  if identity else {})
            y = nn.Conv(features=C_out * O, kernel_size=ks, padding="CIRCULAR",
                        param_dtype=self.dtype, **kw)(hg)
            y = act(y).reshape((B, Lx, Ly, Lz, C_out, O))
            y = jnp.transpose(y, (0, 4, 5, 1, 2, 3)).reshape((B, C_out, M))  # grid order
            if idx is None:
                return y
            return jnp.zeros((B, C_out, M), self.dtype).at[:, :, idx].set(y)

        # noninvariant blocks on the EDGE grid (±1 → ±1 range via normalised sigmoid)
        h = x2[:, None, :]                               # (B, 1, N)
        for _ in range(self.n_noninv):
            h = grid_conv(h, idx_flat, N, self.noninv_channels, _normalised_sigmoid,
                          identity=self.noninv_identity)

        # per-channel Wilson 4-product: (B, C, N) → (B, C, N_plaq)
        g = jnp.prod(h[:, :, plaq_idx], axis=-1)

        # invariant blocks on the PLAQUETTE grid (flat order already = grid order)
        for w in self.inv_hidden:
            g = grid_conv(g, None, N_plaq, w, nn.elu)
        g = grid_conv(g, None, N_plaq, 1, nn.elu)        # final 1 channel
        return jnp.mean(g, axis=(1, 2)).reshape(lead)    # (...,) real log ψ


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


class GeoCNN(nn.Module):
    """Geometry-exact CNN baseline — KernelManager3D convs WITHOUT the Wilson
    4-product, so it is translation-equivariant but NOT A_v-invariant.

    Same kernel (GeoConv3D, same stencil/radius) and output reduction as
    ToricCNN_full, but the spins flow straight through stacked *edge* convs to a
    width-1 readout + mean — there is no flux nonlinearity enforcing vertex-
    operator invariance. This isolates the value of the *invariance* (not the
    kernel): A/B it against ToricCNN_full at matched depth / parameter count to
    see how much the Wilson symmetry, rather than the geometry-exact gather, is
    buying.

    `hidden` are the edge-conv channel widths; a final width-1 edge conv (+ mean
    over channels & sites) gives a real log ψ (h_y = 0 sector). Random init,
    ELU — like CNN_invariant_3D, but on raw edge features.
    """
    km: Any
    hidden: tuple = (4, 4, 4)
    dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, x):                      # x: (..., N) spins ±1
        h = x[..., None, :].astype(self.dtype)                         # (..., 1, N)
        for w in (self.hidden or (4,)):
            h = GeoConv3D(self.km, "edge", w, activation=nn.elu,
                          identity_init=False, dtype=self.dtype)(h)
        h = GeoConv3D(self.km, "edge", 1, activation=nn.elu,
                      identity_init=False, dtype=self.dtype)(h)
        return jnp.mean(h, axis=(-2, -1))                              # (...,) log ψ
