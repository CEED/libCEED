#!/usr/bin/env python3
# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed
#
# libCEED example using diffusion operator to compute surface area
#
# Sample runs:
#
#     python ex2_surface.py
#     python ex2_surface.py -ceed /cpu/self
#     python ex2_surface.py -ceed /gpu/cuda

import sys
import argparse
import numpy as np
import libceed

# Command line options


def parse_arguments():
    """Parse command line arguments for surface area computation

    Returns:
        Namespace: Parsed arguments with fields:
            ceed: CEED resource specifier
            dim: Problem dimension (1-3)
            mesh_degree: Mesh polynomial degree
            solution_degree: Solution polynomial degree
            num_qpts: Number of quadrature points
            problem_size: Approximate problem size
            test: Test mode flag
            quiet: Suppress output flag
            gallery: Use gallery QFunctions flag
    """
    parser = argparse.ArgumentParser(description="libCEED surface area example")
    parser.add_argument("-c", "--ceed", default="/cpu/self",
                        help="libCEED resource specifier (default: /cpu/self)")
    parser.add_argument("-d", "--dim", type=int, default=3,
                        help="Problem dimension (1-3) (default: 3)")
    parser.add_argument("-m", "--mesh-degree", type=int, default=4,
                        help="Mesh polynomial degree (default: 4)")
    parser.add_argument("-p", "--solution-degree", type=int, default=4,
                        help="Solution polynomial degree (default: 4)")
    parser.add_argument("-q", "--quadrature-points", type=int, default=6,
                        help="Number of quadrature points (default: 6)")
    parser.add_argument("-s", "--problem-size", type=int, default=-1,
                        help="Approximate problem size (default: ~256k)")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Test mode (reduced problem size)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("-g", "--gallery", action="store_true",
                        help="Use gallery QFunctions")

    args = parser.parse_args()
    if args.dim not in [1, 2, 3]:
        parser.error("Dimension must be 1, 2, or 3")
    return args

# Mesh utilities


def cartesian_mesh_size(dim, degree, prob_size):
    """Determine Cartesian mesh size for given problem size

    Args:
        dim: Spatial dimension (1-3)
        degree: Polynomial degree
        prob_size: Target problem size

    Returns:
        list: Number of elements in each dimension
    """
    # Calculate number of elements needed
    num_elem = prob_size // (degree ** dim)

    # Find smallest power of 2 >= num_elem
    s = 0
    while 2**s <= num_elem:
        s += 1
    s -= 1

    # Distribute across dimensions
    r = s % dim
    num_xyz = []
    for d in range(dim):
        sd = s // dim
        if r > 0:
            sd += 1
            r -= 1
        num_xyz.append(1 << sd)  # 2^sd
    return num_xyz


def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts):
    """Build element restriction for Cartesian grid

    Args:
        ceed: libCEED context
        dim: Spatial dimension
        num_xyz: Elements per dimension
        degree: Polynomial degree
        num_comp: Number of components
        num_qpts: Quadrature points per dimension

    Returns:
        tuple: (elem_restriction, size, q_data_restriction, num_elem, elem_qpts)
    """
    p = degree + 1  # Nodes per element per dimension
    num_nodes = p ** dim
    elem_qpts = num_qpts ** dim

    # Calculate grid parameters
    nd = []
    num_elem = 1
    scalar_size = 1
    for d in range(dim):
        num_elem *= num_xyz[d]
        nd.append(num_xyz[d] * (p - 1) + 1)  # Nodes per dimension
        scalar_size *= nd[d]

    size = scalar_size * num_comp

    # Create element connectivity
    elem_nodes = np.zeros(num_elem * num_nodes, dtype=np.int32)
    for e in range(num_elem):
        # Get element coordinates
        e_xyz = [0] * dim
        re = e
        for d in range(dim):
            e_xyz[d] = re % num_xyz[d]
            re //= num_xyz[d]

        # Calculate global node numbers
        for n in range(num_nodes):
            g_node = 0
            g_stride = 1
            r_node = n
            for d in range(dim):
                g_node += (e_xyz[d] * (p - 1) + r_node % p) * g_stride
                g_stride *= nd[d]
                r_node //= p
            elem_nodes[e * num_nodes + n] = g_node

    # Create restrictions
    elem_restriction = ceed.ElemRestriction(
        num_elem, num_nodes, num_comp, scalar_size, size, elem_nodes)

    qd_comp = dim * (dim + 1) // 2  # Quadrature data components
    strides = np.array([1, elem_qpts, elem_qpts * qd_comp], dtype=np.int32)
    q_data_restriction = ceed.StridedElemRestriction(
        num_elem, elem_qpts, qd_comp, num_elem * elem_qpts * qd_comp, strides)

    return elem_restriction, size, q_data_restriction, num_elem, elem_qpts


def cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_size):
    """Create Cartesian mesh coordinates

    Args:
        ceed: libCEED context
        dim: Spatial dimension
        num_xyz: Elements per dimension
        mesh_degree: Mesh polynomial degree
        mesh_size: Total mesh vector size

    Returns:
        Vector: Mesh coordinates
    """
    p = mesh_degree + 1
    nd = []
    scalar_size = 1
    for d in range(dim):
        nd.append(num_xyz[d] * (p - 1) + 1)
        scalar_size *= nd[d]

    # Get Lobatto nodes (quadrature points)
    nodes, _ = ceed.lobatto_quadrature(p)
    nodes = 0.5 + 0.5 * nodes  # Map from [-1,1] to [0,1]

    # Create coordinates
    coords = np.zeros(scalar_size * dim)
    for gs_node in range(scalar_size):
        r_node = gs_node
        for d in range(dim):
            d_1d = r_node % nd[d]
            elem_id = d_1d // (p - 1)
            node_id = d_1d % (p - 1)
            coords[gs_node + scalar_size * d] = (elem_id + nodes[node_id]) / num_xyz[d]
            r_node //= nd[d]

    mesh_coords = ceed.Vector(mesh_size)
    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)
    return mesh_coords


def transform_mesh_coords(dim, mesh_size, mesh_coords):
    """Transform mesh coordinates and return exact surface area

    Args:
        dim: Spatial dimension
        mesh_size: Total mesh vector size
        mesh_coords: Mesh coordinates vector

    Returns:
        float: Exact surface area for transformed mesh
    """
    exact_measure = {1: 2.0, 2: 4.0, 3: 6.0}[dim]

    # Apply sinusoidal transformation to coordinates
    with mesh_coords.array_write() as coords:
        for d in range(dim):
            offset = d * (mesh_size // dim)
            for i in range(mesh_size // dim):
                x = coords[offset + i] - 0.5
                coords[offset + i] = 0.5 + (1.0 / np.sqrt(3.0)) * np.sin((2.0 / 3.0) * np.pi * x)

    return exact_measure


# Example 2: Surface Area Computation

def example_2(options):
    """Compute surface area using diffusion operator

    Args:
        options: Parsed command line arguments

    Returns:
        int: 0 on success, 1 on failure in test mode
    """
    # Process arguments
    args = options
    dim = args.dim
    mesh_degree = max(args.mesh_degree, args.solution_degree)
    sol_degree = args.solution_degree
    num_qpts = args.quadrature_points
    problem_size = args.problem_size if args.problem_size > 0 else (8 * 16 if args.test else 256 * 1024)
    ncomp_x = dim  # Number of coordinate components

    # Print configuration
    if not args.quiet:
        print("Selected options: [command line option] : <current value>")
        print(f"    Ceed specification [-c] : {args.ceed}")
        print(f"    Mesh dimension     [-d] : {dim}")
        print(f"    Mesh degree        [-m] : {mesh_degree}")
        print(f"    Solution degree    [-p] : {sol_degree}")
        print(f"    Num. 1D quadr. pts [-q] : {num_qpts}")
        print(f"    Approx. # unknowns [-s] : {problem_size}")
        print(f"    QFunction source   [-g] : {'gallery' if args.gallery else 'user'}")

    # Initialize CEED
    ceed = libceed.Ceed(args.ceed)

    # Create bases

    # Tensor-product Lagrange basis for mesh coordinates
    basis_mesh = ceed.BasisTensorH1Lagrange(
        dim, ncomp_x, mesh_degree + 1, num_qpts, libceed.GAUSS)

    # Tensor-product Lagrange basis for solution
    basis_solution = ceed.BasisTensorH1Lagrange(
        dim, 1, sol_degree + 1, num_qpts, libceed.GAUSS)

    # Create mesh

    # Determine mesh size
    num_xyz = cartesian_mesh_size(dim, sol_degree, problem_size)
    if not args.quiet:
        print("\nMesh size                   : nx = %d" % num_xyz[0], end="")
        if dim > 1:
            print(", ny = %d" % num_xyz[1], end="")
        if dim > 2:
            print(", nz = %d" % num_xyz[2], end="")
        print()

    # Create element restrictions
    mesh_restr, mesh_size, _, _, _ = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, ncomp_x, num_qpts)
    sol_restr, sol_size, q_data_restr, num_elem, elem_qpts = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts)

    if not args.quiet:
        print("Number of mesh nodes        : %d" % (mesh_size // dim))
        print("Number of solution nodes    : %d" % sol_size)

    # Create and transform mesh coordinates
    mesh_coords = cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_size)
    exact_measure = transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Build operator

    # QFunction for building quadrature data
    qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")

    # Operator for building quadrature data
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restr, basis_mesh, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE,
                       basis_mesh, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restr,
                       libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Compute quadrature data
    qd_comp = dim * (dim + 1) // 2
    q_data = ceed.Vector(num_elem * elem_qpts * qd_comp)
    op_build.apply(mesh_coords, q_data)

    # Apply operator

    # QFunction for applying diffusion operator
    qf_apply = ceed.QFunctionByName(f"Poisson{dim}DApply")

    # Diffusion operator
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("du", sol_restr, basis_solution, libceed.VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restr,
                       libceed.BASIS_NONE, q_data)
    op_apply.set_field("dv", sol_restr, basis_solution, libceed.VECTOR_ACTIVE)

    # Compute solution

    # Create vectors
    u = ceed.Vector(sol_size)  # Input vector
    v = ceed.Vector(sol_size)  # Output vector

    # Initialize u with sum of coordinates (x + y + z)
    with mesh_coords.array_read() as x, u.array_write() as u_arr:
        for i in range(sol_size):
            u_arr[i] = sum(x[i + j * (sol_size)] for j in range(dim))

    # Apply operator: v = K * u
    op_apply.apply(u, v)

    # Process results

    # Compute surface area by summing absolute values of v
    measure = 0.0
    with v.array_read() as v_arr:
        measure = np.sum(np.abs(v_arr))

    # Output results
    if not args.quiet:
        label = "curve length" if dim == 1 else "surface area"
        print("Exact %-20s: %.12f" % (label, exact_measure))
        print("Computed %-16s: %.12f" % (label, measure))
        print("Error                     : %.12e" % (measure - exact_measure))

    # Verify results in test mode
    if args.test:
        tolerance = 1e-1
        error = abs(measure - exact_measure)
        if error > tolerance:
            print("Error too large: %.12e" % error)
            return 1

    return 0


# Main function
def main():
    """Main driver for surface area example"""
    args = parse_arguments()
    return example_2(args)


if __name__ == "__main__":
    sys.exit(main())
