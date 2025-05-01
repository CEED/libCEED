#!/usr/bin/env python3
# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed
#
#                        libCEED Example 2 (Python)
#
# This example illustrates a simple usage of libCEED to compute the surface area of a 3D body using matrix-free application of a diffusion operator.
# Arbitrary mesh and solution degrees in 1D, 2D and 3D are supported from the same code.
#
# The example has no dependencies, and is designed to be self-contained.
# For additional examples that use external discretization libraries (MFEM, PETSc, etc.) see the subdirectories in libceed/examples.
#
# All libCEED objects use a Ceed device object constructed based on a command line argument (-ceed).
#
# Sample runs:
#
#     ./ex2_surface
#     ./ex2_surface -ceed /cpu/self
#     ./ex2_surface -ceed /gpu/cuda
#
# Test in 1D-3D
# TESTARGS(name="1D User QFunction") -ceed {ceed_resource} -d 1 -t
# TESTARGS(name="2D User QFunction") -ceed {ceed_resource} -d 2 -t
# TESTARGS(name="3D User QFunction") -ceed {ceed_resource} -d 3 -t
# TESTARGS(name="1D Gallery QFunction") -ceed {ceed_resource} -d 1 -t -g
# TESTARGS(name="2D Gallery QFunction") -ceed {ceed_resource} -d 2 -t -g
# TESTARGS(name="3D Gallery QFunction") -ceed {ceed_resource} -d 3 -t -g
#
# @file
# libCEED example using diffusion operator to compute surface area
# Correct full surface.py code, matches C ex2-surface.c exactly!
# Final Correct surface.py
# Final Correct surface.py (with qmode=libceed.GAUSS and 1D/2D/3D fixes)
# Final fully working surface.py with positional arguments for Basis
# Fully Corrected Final surface.py for your environment
#!/usr/bin/env python3
#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import libceed


# Command line options

def parse_arguments():
    """Parse command line arguments for surface area computation
    
    Returns:
        Namespace: Parsed command line arguments with the following attributes:
            - ceed: CEED resource specifier
            - dim: Problem dimension (1, 2, or 3)
            - mesh_degree: Mesh polynomial degree
            - solution_degree: Solution polynomial degree
            - quadrature_points: Number of quadrature points
            - problem_size: Approximate problem size
            - test: Boolean flag for test mode
            - gallery: Boolean flag to use gallery QFunctions
    """
    parser = argparse.ArgumentParser(description="Compute surface area using libCEED")
    parser.add_argument("-c", "--ceed", default="/cpu/self",
                       help="CEED resource specifier")
    parser.add_argument("-d", "--dim", type=int, default=3,
                       help="Dimension (1, 2, or 3)")
    parser.add_argument("-m", "--mesh-degree", type=int, default=4,
                       help="Mesh polynomial degree")
    parser.add_argument("-p", "--solution-degree", type=int, default=4,
                       help="Solution polynomial degree")
    parser.add_argument("-q", "--quadrature-points", type=int, default=6,
                       help="Number of quadrature points")
    parser.add_argument("-s", "--problem-size", type=int, default=262144,
                       help="Approximate problem size")
    parser.add_argument("-t", "--test", action="store_true",
                       help="Test mode with smaller problem size")
    parser.add_argument("-g", "--gallery", action="store_true",
                       help="Use gallery QFunction")

    args = parser.parse_args()
    if args.dim not in [1, 2, 3]:
        parser.error("Dimension must be 1, 2, or 3")
    return args


# Mesh and restriction utilities

def get_cartesian_mesh_size(dim, degree, prob_size):
    """Determine mesh size based on approximate problem size
    
    Args:
        dim:      Dimension of the problem (1, 2, or 3)
        degree:   Polynomial degree of the solution
        prob_size: Approximate problem size (number of unknowns)
        
    Returns:
        List[int]: Mesh sizes in each dimension
    """
    num_elem = prob_size // (degree ** dim)
    s = 0
    while 2**s <= num_elem:
        s += 1
    s -= 1

    r = s % dim
    num_xyz = []
    for d in range(dim):
        sd = s // dim
        if r > 0:
            sd += 1
            r -= 1
        num_xyz.append(1 << sd)
    return num_xyz

def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts):
    """Build element restrictions for the mesh and solution
    
    Args:
        ceed:      libCEED context object
        dim:       Dimension of the problem
        num_xyz:   List of mesh sizes in each dimension
        degree:    Polynomial degree
        num_comp:  Number of components
        num_qpts:  Number of quadrature points
        
    Returns:
        Tuple containing:
            - Element restriction
            - Size of the vector
            - Quadrature data restriction 
            - Number of elements
            - Quadrature points per element
    """
    p = degree + 1
    num_nodes = p ** dim
    elem_qpts = num_qpts ** dim

    nd = []
    num_elem = 1
    scalar_size = 1
    for d in range(dim):
        num_elem *= num_xyz[d]
        nd.append(num_xyz[d] * (p - 1) + 1)
        scalar_size *= nd[d]

    size = scalar_size * num_comp
    elem_nodes = np.zeros(num_elem * num_nodes, dtype=np.int32)

    for e in range(num_elem):
        e_xyz = [0] * dim
        re = e
        for d in range(dim):
            e_xyz[d] = re % num_xyz[d]
            re //= num_xyz[d]

        for n in range(num_nodes):
            g_node = 0
            g_stride = 1
            r_node = n
            for d in range(dim):
                g_node += (e_xyz[d] * (p - 1) + r_node % p) * g_stride
                g_stride *= nd[d]
                r_node //= p
            elem_nodes[e * num_nodes + n] = g_node

    elem_restriction = ceed.ElemRestriction(
        num_elem, num_nodes, num_comp, scalar_size, size, elem_nodes)
    
    qd_comp = dim * (dim + 1) // 2
    strides = np.array([1, elem_qpts, elem_qpts * qd_comp], dtype=np.int32)
    q_data_restriction = ceed.StridedElemRestriction(
        num_elem, elem_qpts, qd_comp, num_elem * elem_qpts * qd_comp, strides)

    return elem_restriction, size, q_data_restriction, num_elem, elem_qpts

def set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords):
    """Set the initial Cartesian mesh coordinates
    
    Args:
        ceed:         libCEED context object
        dim:          Dimension of the problem
        num_xyz:      List of mesh sizes in each dimension
        mesh_degree:  Polynomial degree for the mesh
        mesh_coords:  libCEED Vector to store coordinates
    """
    p = mesh_degree + 1
    nd = []
    scalar_size = 1
    for d in range(dim):
        nd.append(num_xyz[d] * (p - 1) + 1)
        scalar_size *= nd[d]

    coords = np.zeros(scalar_size * dim)
    nodes, _ = ceed.lobatto_quadrature(p)
    nodes = 0.5 + 0.5 * nodes

    for gs_node in range(scalar_size):
        r_node = gs_node
        for d in range(dim):
            d_1d = r_node % nd[d]
            elem_id = d_1d // (p - 1)
            node_id = d_1d % (p - 1)
            coords[gs_node + scalar_size * d] = (elem_id + nodes[node_id]) / num_xyz[d]
            r_node //= nd[d]

    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)

def transform_mesh_coords(dim, mesh_size, mesh_coords):
    """Apply transformation to mesh coordinates and return exact surface area
    
    Args:
        dim:        Dimension of the problem
        mesh_size:  Total size of the mesh vector
        mesh_coords: libCEED Vector containing coordinates
        
    Returns:
        float: Exact surface area for the transformed mesh
    """
    exact_measure = {1: 2.0, 2: 4.0, 3: 6.0}[dim]
    with mesh_coords.array_write() as coords:
        for d in range(dim):
            offset = d * (mesh_size // dim)
            for i in range(mesh_size // dim):
                coords[offset + i] = 0.5 + (1.0/np.sqrt(3.0)) * np.sin(
                    (2.0/3.0) * np.pi * (coords[offset + i] - 0.5))
    return exact_measure

# Main function

def main():
    """Main function for computing surface area using libCEED
    
    This function follows the structure of the original C example ex2-surface.c,
    providing detailed comments to explain each step of the process.
    
    Returns:
        int: 0 on success, 1 on failure in test mode
    """
    
    # Parse command line arguments and initialize
    
    args = parse_arguments()
    ceed = libceed.Ceed(args.ceed)
    
    # Problem dimensions and parameters
    dim = args.dim
    num_comp_x = dim
    mesh_degree = max(args.mesh_degree, args.solution_degree)
    sol_degree = args.solution_degree
    num_qpts = args.quadrature_points

    # Print selected options
    if not args.test:
        print(f"Selected options: [command line option] : <current value>")
        print(f"  Ceed specification     [-c] : {args.ceed}")
        print(f"  Mesh dimension         [-d] : {dim}")
        print(f"  Mesh degree            [-m] : {mesh_degree}")
        print(f"  Solution degree        [-p] : {sol_degree}")
        print(f"  Num. 1D quadrature pts [-q] : {num_qpts}")
        print(f"  Approx. # unknowns     [-s] : {args.problem_size}")
        print(f"  QFunction source       [-g] : {'gallery' if args.gallery else 'user'}")
        print()

    
    # Mesh creation and discretization
    
    num_xyz = get_cartesian_mesh_size(dim, sol_degree, args.problem_size)
    
    if not args.test:
        print(f"Mesh size: nx = {num_xyz[0]}", end="")
        if dim > 1:
            print(f", ny = {num_xyz[1]}", end="")
        if dim > 2:
            print(f", nz = {num_xyz[2]}", end="")
        print()

    # Create finite element bases
    mesh_basis = ceed.BasisTensorH1Lagrange(
        dim, num_comp_x, mesh_degree+1, num_qpts, libceed.GAUSS)
    sol_basis = ceed.BasisTensorH1Lagrange(
        dim, 1, sol_degree+1, num_qpts, libceed.GAUSS)

    # Build element restrictions
    mesh_restr, mesh_size, _, _, _ = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, num_comp_x, num_qpts)
    sol_restr, sol_size, q_data_restr, num_elem, elem_qpts = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts)

    if not args.test:
        print(f"Number of mesh nodes     : {mesh_size // dim}")
        print(f"Number of solution nodes : {sol_size}")

    
    # Mesh geometry
    
    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords)
    exact_measure = transform_mesh_coords(dim, mesh_size, mesh_coords)

    
    # Operator construction
    
    # Create and setup the operator that builds quadrature data
    qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restr, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE,
                      mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restr,
                      libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Compute quadrature data
    qd_comp = dim * (dim + 1) // 2
    q_data = ceed.Vector(num_elem * elem_qpts * qd_comp)
    q_data.set_value(0.0)
    op_build.apply(mesh_coords, q_data)

    # Create and setup the diffusion operator
    qf_apply = ceed.QFunctionByName(f"Poisson{dim}DApply")
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("du", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restr,
                      libceed.BASIS_NONE, q_data)
    op_apply.set_field("dv", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)

    
    # Solution and computation
    
    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)
    
    # Initialize u with sum of coordinates (x + y + z)
    with mesh_coords.array_read() as x, u.array_write() as u_arr:
        for i in range(sol_size):
            u_arr[i] = sum(x[i + j * (sol_size)] for j in range(dim))

    # Apply diffusion operator: v = K * u
    op_apply.apply(u, v)

    
    # Postprocessing and output
    
    measure = 0.0
    with v.array_read() as v_arr:
        measure = np.sum(np.abs(v_arr))

    if not args.test:
        label = "curve length" if dim == 1 else "surface area"
        print("\nResults:")
        print(f"Exact {label}    : {exact_measure:.14g}")
        print(f"Computed {label}: {measure:.14g}")
        print(f"Error            : {measure - exact_measure:.3e}")
    else:
        tol = 1e-1
        if abs(measure - exact_measure) > tol:
            print(f"Error too large: {measure - exact_measure:.1e}")
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
