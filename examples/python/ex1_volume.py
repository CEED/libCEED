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
#     python ex1_volume.py
#     python ex1_volume -c /cpu/self
#     python ex1_volume -c /gpu/cuda

import sys
import os
import numpy as np
import libceed
import ex_common as common


def main():
    """Main function for volume example"""
    args = common.parse_arguments()
    return example_1(args)


def example_1(args):
    """Compute volume using mass operator

    Args:
        args: Parsed command line arguments

    Returns:
        int: 0 on success, error code on failure
    """
    # Process arguments
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
    num_q_comp = 1
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
    exact_volume, _ = common.transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Create the QFunction that builds the mass operator (i.e. computes its quadrature data) and set its context data
    qf_build = None
    if args.gallery:
        qf_build = ceed.QFunctionByName(f"Mass{dim}DBuild")
    else:
        build_ctx = ceed.QFunctionContext()
        ctx_data = np.array([dim, dim], dtype=np.int32)
        build_ctx.set_data(ctx_data)

        qfs_so = common.load_qfs_so()
        file_dir = os.path.dirname(os.path.abspath(__file__))

        qf_build = ceed.QFunction(1, qfs_so.build_mass,
                                  os.path.join(file_dir, "ex1-volume.h:build_mass"))
        qf_build.add_input("dx", dim * dim, libceed.EVAL_GRAD)
        qf_build.add_input("weights", 1, libceed.EVAL_WEIGHT)
        qf_build.add_output("qdata", num_q_comp, libceed.EVAL_NONE)
        qf_build.set_context(build_ctx)

    # Create the operator that builds the quadrature data for the mass operator
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restriction, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Compute the quadrature data for the mass operator
    q_data = ceed.Vector(num_elem * elem_qpts * num_q_comp)
    op_build.apply(mesh_coords, q_data)

    # Setup QFunction for applying the mass operator
    qf_mass = None
    if args.gallery:
        qf_mass = ceed.QFunctionByName("MassApply")
    else:
        build_ctx = ceed.QFunctionContext()
        ctx_data = np.array([dim, dim], dtype=np.int32)
        build_ctx.set_data(ctx_data)

        qfs_so = common.load_qfs_so()
        file_dir = os.path.dirname(os.path.abspath(__file__))

        qf_mass = ceed.QFunction(1, qfs_so.apply_mass,
                                 os.path.join(file_dir, "ex1-volume.h:apply_mass"))
        qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
        qf_mass.add_input("qdata", num_q_comp, libceed.EVAL_NONE)
        qf_mass.add_output("v", 1, libceed.EVAL_INTERP)
        qf_mass.set_context(build_ctx)

    # Create the mass operator
    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("u", solution_restriction, solution_basis, libceed.VECTOR_ACTIVE)
    op_mass.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, q_data)
    op_mass.set_field("v", solution_restriction, solution_basis, libceed.VECTOR_ACTIVE)

    # Create solution vectors
    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)
    u.set_value(1.0)  # Set all entries of u to 1.0

    # Apply mass operator: v = M * u
    op_mass.apply(u, v)

    # Compute volume by summing all entries in v
    volume = 0.0
    with v.array_read() as v_array:
        # Simply sum all values to compute the volume
        volume = np.sum(v_array)

    if not args.test:
        print()
        print(f"Exact mesh volume    : {exact_volume:.14g}")
        print(f"Computed mesh volume : {volume:.14g}")
        print(f"Volume error         : {volume - exact_volume:.14g}")
    else:
        # Test mode - check if error is within tolerance
        tol = 200 * libceed.EPSILON if dim == 1 else 1e-5
        if abs(volume - exact_volume) > tol:
            print(f"Volume error : {volume - exact_volume:.14g}")
            sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
