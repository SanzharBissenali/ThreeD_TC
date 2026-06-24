# NQS architecture considerations — 3D toric-code networks

Intuition notes on `Three_TC/model/networks.py`: what the conv actually does to an
input and why the tensor shapes look the way they do. For the training/SR loop see
`vmc_internals.md`; for failure modes see `training_gotchas.md`.

Started 2026-06-24.

---

## Why the conv acts on `(C, N)`, not `(L, L, L)`

A vanilla `nn.Conv3D` wants a `(C, L, L, L)` tensor and reads "who is my neighbour"
from **axis adjacency** — `(ix+1, iy, iz)` is next to `(ix, iy, iz)`. `GeoConv3D`
does **not** do this. It keeps features flat and supplies the neighbour relation as
an explicit lookup table.

**The grid did not disappear — it is the flat index.**

    N = 3 · Lx·Ly·Lz = 3 L³      (3 edge orientations × L³ cube vertices)

So `(1, N)` is really `(C_in=1, [orientation × ix × iy × iz])` in
`geo._mapping3Dto1D` order. For L=2: `L³=8`, `N=24`, input `(1, 24)`. Same for
plaquettes (`N_plaq = 3L³`). See `compute_edges_3D` (`networks.py:68`).

## Why not keep `(L, L, L)`?

The three edge orientations are **interpenetrating sublattices** offset by ½ a
lattice vector — they are *not* co-located at a vertex. Stuffing them into a
channel axis `(3, L, L, L)` and running a grid conv makes "neighbour" only
*approximate* (the half-offset error). This is the whole reason the custom kernel
exists (docstring `networks.py:8`).

## What `GeoConv3D` does instead (one sample, batch dropped)

The geometry is precomputed once in `KernelManager3D`:

- `edge_gather`  `(O=3, P=L³, S)` — for each output orientation & vertex, the S
  flat indices of its stencil neighbours (Euclidean radius ≤ `radius`, default 1.05).
- `edge_out`     `(3, L³)`        — where each `(orientation, vertex)` output lands
  in flat order (a permutation of `0..N-1`).

Forward pass (`networks.py:302`):

    x        : (C_in, N)                  flat sites
    xg       : (C_in, O=3, P=L³, S)       x[:, gather]  — GATHER neighbours
    W        : (O=3, C_out, C_in, S)      one weight set per output orientation
    y        : (O=3, C_out, P=L³)         einsum over C_in and the S taps
    out      : (C_out, N)                 SCATTER (orient,vertex) back to flat N

The `(O=3, P=L³)` pair *is* the `(L,L,L)`-like structure, carried as gather/scatter
axes instead of tensor dimensions. Weight sharing across the P=L³ sites (one set per
orientation) gives exact integer-translation equivariance — the property a grid
conv gets for free from sliding.

## Mental model

A standard conv = (1) gather a fixed neighbour stencil at every site, (2) contract
with shared weights, (3) write back; the `(L,L,L)` shape just makes step (1) an
index shift. `GeoConv3D` runs the **same three steps**, but the toric-code neighbour
relation is the irregular half-offset one, so it can't be an axis shift — the gather
table replaces it and the data stays flat.

**One-liner:** `N` is `3L³` flattened; `edge_gather` is the explicit "where are my
neighbours" map that a grid conv would otherwise get implicitly from the `(L,L,L)`
layout.

## The stencil size S (=15) — the conv footprint

`S` is the kernel footprint: how many taps each conv sums over per output site.
It's the 3D-toric analog of "a 3×3×3 grid conv has 27 taps", but defined by
**physical Euclidean distance** on the half-offset edge lattice, not axis steps.
Set entirely by `radius_edge` / `radius_plaq` (default 1.05, `networks.py:116`).

Reference edge = x-edge at `(0.5, 0, 0)`. Within radius 1.05:

| shell | dist | count | what they are |
|---|---|---|---|
| self        | 0.000 | 1 | the edge itself (identity tap, `self_index=0`) |
| nearest     | 0.707 | 8 | perpendicular edges sharing a vertex (4 y + 4 z) |
| same-orient | 1.000 | 6 | same-orientation x-edges one cell away, ±x±y±z |

→ **S = 1 + 8 + 6 = 15.** Plaquette stencil S_p = 15 the same way.

Key distances:
- `0.707 = √½` = gap between an x-edge midpoint and a perpendicular y/z midpoint
  (displacement `(0.5,−0.5,0)`). These 8 are the genuine physical nearest
  neighbours — **exactly the shell a grid conv gets wrong** (half-offset error).
- `1.000` = same-orientation edge one full lattice step away.

Radius as the knob:

    radius < 0.707      → S = 1   (self only, pointwise)
    0.707 ≤ r < 1.0     → S = 9   (self + 8 perpendicular)
    1.0  ≤ r < 1.06     → S = 15  ← default (adds 6 same-orient ring)
    larger              → pulls in diagonal shells, S grows fast

1.05 is chosen to sit just past 1.0 (minimal physically-complete neighbourhood)
but below √2 ≈ 1.414 so it doesn't blow up into diagonal taps.

S multiplies every weight tensor `(O=3, C_out, C_in, S)`, so it scales every
param count: e.g. a 4→4 layer is `3·4·4·15 = 720`. Bump the radius and every
layer grows with the new S.

## L=2 caveat

Even this half-offset-exact kernel cannot separate the `+ê` and `−ê` neighbour at
L=2 (same site under PBC). Intrinsic to L=2, not to the kernel; the stencil radius
buys nothing extra there (`networks.py:49`).
