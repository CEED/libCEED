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
#     python ex2_surface.py -c /cpu/self
#     python ex2_surface.py -c /gpu/cuda

import sys
import os
import numpy as np
import libceed
import ex_common as common


def main():
    """Main driver for surface area example"""
    args = common.parse_arguments()
    return example_2(args)


def example_2(options):
    """Compute surface area using diffusion operator

    Args:
        args: Parsed command line arguments

    Returns:
        int: 0 on success, error code on failure
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
    mesh_basis = ceed.BasisTensorH1Lagrange(
        dim, ncomp_x, mesh_degree + 1, num_qpts, libceed.GAUSS)

    # Tensor-product Lagrange basis for solution
    solution_basis = ceed.BasisTensorH1Lagrange(
        dim, 1, sol_degree + 1, num_qpts, libceed.GAUSS)

    # Create mesh
    # Determine mesh size
    num_xyz = common.get_cartesian_mesh_size(dim, sol_degree, problem_size)
    if not args.quiet:
        print("\nMesh size                   : nx = %d" % num_xyz[0], end="")
        if dim > 1:
            print(", ny = %d" % num_xyz[1], end="")
        if dim > 2:
            print(", nz = %d" % num_xyz[2], end="")
        print()

    # Create element restrictions
    num_q_comp = dim * (dim + 1) // 2
    mesh_restriction, mesh_size, _, _, _ = common.build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, ncomp_x, num_q_comp, num_qpts, create_qdata=False)
    solution_restriction, sol_size, q_data_restriction, num_elem, elem_qpts = common.build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_q_comp, num_qpts, create_qdata=True)

    if not args.quiet:
        print("Number of mesh nodes        : %d" % (mesh_size // dim))
        print("Number of solution nodes    : %d" % sol_size)

    # Create and transform mesh coordinates
    mesh_coords = ceed.Vector(mesh_size)
    common.set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords)
    _, exact_surface_area = common.transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Create the QFunction that builds the diffusion operator (i.e. computes
    # its quadrature data) and set its context data
    qf_build = None
    if args.gallery:
        qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")
    else:
        build_ctx = ceed.QFunctionContext()
        ctx_data = np.array([dim, dim], dtype=np.int32)
        build_ctx.set_data(ctx_data)

        qfs_so = common.load_qfs_so()
        file_dir = os.path.dirname(os.path.abspath(__file__))

        qf_build = ceed.QFunction(1, qfs_so.build_diff,
                                  os.path.join(file_dir, "ex2-surface.h:build_diff"))
        qf_build.add_input("dx", dim * dim, libceed.EVAL_GRAD)
        qf_build.add_input("weights", 1, libceed.EVAL_WEIGHT)
        qf_build.add_output("qdata", num_q_comp, libceed.EVAL_NONE)
        qf_build.set_context(build_ctx)

    # Operator for building quadrature data
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restriction, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Compute quadrature data
    q_data = ceed.Vector(num_elem * elem_qpts * num_q_comp)
    op_build.apply(mesh_coords, q_data)

    # Create the QFunction that defines the action of the diffusion operator
    qf_diff = None
    if args.gallery:
        qf_diff = ceed.QFunctionByName(f"Poisson{dim}DApply")
    else:
        build_ctx = ceed.QFunctionContext()
        ctx_data = np.array([dim, dim], dtype=np.int32)
        build_ctx.set_data(ctx_data)

        qfs_so = common.load_qfs_so()
        file_dir = os.path.dirname(os.path.abspath(__file__))

        qf_diff = ceed.QFunction(1, qfs_so.apply_diff,
                                 os.path.join(file_dir, "ex2-surface.h:apply_diff"))
        qf_diff.add_input("du", dim, libceed.EVAL_GRAD)
        qf_diff.add_input("qdata", num_q_comp, libceed.EVAL_NONE)
        qf_diff.add_output("dv", dim, libceed.EVAL_GRAD)
        qf_diff.set_context(build_ctx)

    # Diffusion operator
    op_diff = ceed.Operator(qf_diff)
    op_diff.set_field("du", solution_restriction, solution_basis, libceed.VECTOR_ACTIVE)
    op_diff.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, q_data)
    op_diff.set_field("dv", solution_restriction, solution_basis, libceed.VECTOR_ACTIVE)

    # Create vectors
    u = ceed.Vector(sol_size)  # Input vector
    v = ceed.Vector(sol_size)  # Output vector

    # Initialize u with sum of coordinates (x + y + z)
    with mesh_coords.array_read() as x_array, u.array_write() as u_array:
        for i in range(sol_size):
            u_array[i] = sum(x_array[i + j * (sol_size)] for j in range(dim))

    # Apply operator: v = K * u
    op_diff.apply(u, v)

    # Compute surface area by summing absolute values of v
    surface_area = 0.0
    with v.array_read() as v_array:
        surface_area = np.sum(abs(v_array))

    if not args.test:
        print()
        print(f"Exact mesh surface area    : {exact_surface_area:.14g}")
        print(f"Computed mesh surface area : {surface_area:.14g}")
        print(f"Surface area error         : {surface_area - exact_surface_area:.14g}")
    else:
        # Test mode - check if error is within tolerance
        tol = 10000 * libceed.EPSILON if dim == 1 else 1e-1
        if abs(surface_area - exact_surface_area) > tol:
            print(f"Surface area error : {surface_area - exact_surface_area:.14g}")
            sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
