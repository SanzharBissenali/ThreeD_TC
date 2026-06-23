"""
Module for creating the toric code Hamiltonian with various perturbations.
Fully copied from the 2D-version. 
"""

import netket as nk
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple, Optional, Union

def create_hamiltonian(
    hi: nk.hilbert.Spin,
    vertex_all: List[List[int]],
    plaq_all: List[List[int]],
    bonds: List[List[int]],
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    J: float = 1.0,
    Jy_v: float = 0.0,
    Jy_p: float = 0.0,
    Jbond: float = 0.0,
    dtype: Any = complex
) -> nk.operator.AbstractOperator:
    """
    Create the toric code Hamiltonian with perturbations.
    
    Args:
        hi: Hilbert space
        vertex_all: List of vertex operators
        plaq_all: List of plaquette operators
        bonds: List of nearest-neighbor bonds
        hx: X magnetic field strength
        hy: Y magnetic field strength
        hz: Z magnetic field strength
        J: Coupling strength
        Jy_v: Y vertex coupling
        Jy_p: Y plaquette coupling
        Jbond: Bond coupling
        dtype: Data type for the Hamiltonian
        
    Returns:
        The toric code Hamiltonian
    """
    H = 0
    N = hi.size
    
    # Add vertex terms
    for v in range(0, len(vertex_all)):
        # XXXXXX vertex terms
        op = 1
        for j in range(0, len(vertex_all[v])):
            if vertex_all[v][j] != -1:
                op *= nk.operator.spin.sigmax(hi, vertex_all[v][j], dtype=dtype)
        H += -J * op
        
        # YYYYYY vertex terms
        if Jy_v != 0:
            assert dtype == "complex", "YYYY vertex terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(vertex_all[v])):
                if vertex_all[v][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, vertex_all[v][j], dtype=dtype)
            H += -Jy_v * op
    
    # Add plaquette terms
    for p in range(0, len(plaq_all)):
        # ZZZZ plaquette terms
        op = 1
        for j in range(0, len(plaq_all[p])):
            if plaq_all[p][j] != -1:
                op *= nk.operator.spin.sigmaz(hi, plaq_all[p][j], dtype=dtype)
        H += -J * op
        
        # YYYY plaquette terms
        if Jy_p != 0:
            assert dtype == "complex", "YYYY plaquette terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(plaq_all[p])):
                if plaq_all[p][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, plaq_all[p][j], dtype=dtype)
            H += -Jy_p * op
    
    # Add magnetic field perturbations
    for j in range(0, N):
        if hz != 0:
            H += -nk.operator.spin.sigmaz(hi, j, dtype=dtype) * hz
        if hx != 0:
            H += -nk.operator.spin.sigmax(hi, j, dtype=dtype) * hx
        if hy != 0:
            assert dtype == "complex", "Y magnetic field requires complex Hamiltonian"
            H += -nk.operator.spin.sigmay(hi, j, dtype=dtype) * hy
    
    # Add 2-qubit perturbations (bonds)
    if Jbond != 0.0:
        for (x, y) in bonds:
            H += -nk.operator.spin.sigmax(hi, x) * nk.operator.spin.sigmax(hi, y) * Jbond
            H += -nk.operator.spin.sigmaz(hi, x) * nk.operator.spin.sigmaz(hi, y) * Jbond
            H += -nk.operator.spin.sigmay(hi, x) * nk.operator.spin.sigmay(hi, y) * Jbond
    
    # Convert to Pauli strings for more efficient implementation
    H = H.to_pauli_strings()

    return H


def create_hamiltonian_fermionic(
    hi: nk.hilbert.Spin,
    vertex_all: List[List[int]],
    xz_stabs: List[Tuple[List[int], List[int], float]],
    bonds: List[List[int]],
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    J: float = 1.0,
    Jy_v: float = 0.0,
    Jbond: float = 0.0,
    dtype: Any = complex,
) -> nk.operator.AbstractOperator:
    """Create the 3D *fermionic* toric-code Hamiltonian for NQS training.

    Identical to `create_hamiltonian` except the pure-sigma^z plaquette term B_p
    is replaced by the decorated stabilizer

        B~_p = (prod_{e in z_edges} sigma^z_e) * (prod_{e in x_edges} sigma^x_e),

    supplied as the (z_edges, x_edges, coef) triples returned by
    `Three_TC.model.fermionic_decoration.fermionic_plaquettes`.  z_edges (the 4
    boundary edges) and x_edges (the 2 corner edges) are disjoint, so each term
    is a genuine 6-body Pauli string with no sigma^y.

    The coef in each triple already carries its sign/magnitude (-J from
    `fermionic_plaquettes`), so it is used directly; the `J` argument here scales
    only the vertex stars, mirroring the bosonic convention where the vertex and
    plaquette couplings share a single J.  There is no `Jy_p` — a YYYY product on
    a decorated plaquette is not meaningful.  Vertex stars A_v, the h_x/h_y/h_z
    fields, and the optional bond terms are unchanged from the bosonic version.

    Args:
        hi: Hilbert space
        vertex_all: List of vertex stars (6-tuples of qubit ids in 3D PBC)
        xz_stabs: decorated plaquettes as (z_edges, x_edges, coef) triples
        bonds: List of nearest-neighbor bonds
        hx, hy, hz: magnetic field strengths
        J: vertex-star coupling strength
        Jy_v: Y vertex coupling
        Jbond: bond coupling
        dtype: Data type for the Hamiltonian

    Returns:
        The fermionic toric code Hamiltonian (as Pauli strings)
    """
    H = 0
    N = hi.size

    # Vertex stars: -J * XXXXXX  (unchanged from the bosonic model)
    for v in range(0, len(vertex_all)):
        op = 1
        for j in range(0, len(vertex_all[v])):
            if vertex_all[v][j] != -1:
                op *= nk.operator.spin.sigmax(hi, vertex_all[v][j], dtype=dtype)
        H += -J * op

        # YYYYYY vertex terms
        if Jy_v != 0:
            assert dtype == "complex", "YYYY vertex terms require complex Hamiltonian"
            op = 1
            for j in range(0, len(vertex_all[v])):
                if vertex_all[v][j] != -1:
                    op *= nk.operator.spin.sigmay(hi, vertex_all[v][j], dtype=dtype)
            H += -Jy_v * op

    # Decorated plaquettes: coef * (ZZZZ on z_edges) * (XX on x_edges)
    for z_edges, x_edges, coef in xz_stabs:
        op = 1
        for i in z_edges:
            if i != -1:
                op *= nk.operator.spin.sigmaz(hi, int(i), dtype=dtype)
        for i in x_edges:
            if i != -1:
                op *= nk.operator.spin.sigmax(hi, int(i), dtype=dtype)
        H += coef * op

    # Add magnetic field perturbations
    for j in range(0, N):
        if hz != 0:
            H += -nk.operator.spin.sigmaz(hi, j, dtype=dtype) * hz
        if hx != 0:
            H += -nk.operator.spin.sigmax(hi, j, dtype=dtype) * hx
        if hy != 0:
            assert dtype == "complex", "Y magnetic field requires complex Hamiltonian"
            H += -nk.operator.spin.sigmay(hi, j, dtype=dtype) * hy

    # Add 2-qubit perturbations (bonds)
    if Jbond != 0.0:
        for (x, y) in bonds:
            H += -nk.operator.spin.sigmax(hi, x) * nk.operator.spin.sigmax(hi, y) * Jbond
            H += -nk.operator.spin.sigmaz(hi, x) * nk.operator.spin.sigmaz(hi, y) * Jbond
            H += -nk.operator.spin.sigmay(hi, x) * nk.operator.spin.sigmay(hi, y) * Jbond

    # Convert to Pauli strings for more efficient implementation
    H = H.to_pauli_strings()

    return H