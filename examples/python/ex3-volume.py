#!/usr/bin/env python3
# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

"""
Example using libCEED with mass operator to compute volume.

This example illustrates a simple usage of libCEED to compute the volume of a 3D body
using matrix-free application of a mass operator. This example also demonstrates
libCEED's ability to handle multiple basis evaluation modes for the same input and
output vectors.

The example supports arbitrary mesh and solution degrees in 1D, 2D, and 3D from the
same code.
"""

import argparse
import math
import numpy as np
import libceed
from libceed import GAUSS, VECTOR_ACTIVE, VECTOR_NONE, ELEMRESTRICTION_NONE, BASIS_NONE
from libceed import EVAL_INTERP, EVAL_GRAD, EVAL_WEIGHT, EVAL_NONE
from libceed import MEM_HOST, COPY_VALUES, USE_POINTER


# ------------------------------------------------------------------------------
# QFunctions for mass operator
# ------------------------------------------------------------------------------
def build_mass_diff(dx, weights, qdata, ctx_data):
    """Build the quadrature data for the mass operator with diffusion.

    Args:
        dx: Gradient of basis functions at quadrature points (EVAL_GRAD)
        weights: Quadrature weights (EVAL_WEIGHT)
        qdata: Output array for quadrature data (EVAL_NONE)
        ctx_data: Context data with dimension information
    """
    dim = ctx_data[0]
    space_dim = ctx_data[1]

    # Get number of elements and quadrature points
    num_elem = dx.shape[0]
    num_qpts = dx.shape[1]

    # For each element and quadrature point
    for e in range(num_elem):
        for q in range(num_qpts):
            # Compute the Jacobian determinant
            # For simplicity, assuming affine elements
            # J = [dx/dxi, dx/deta, dx/dzeta]
            # J is represented as a dim x dim matrix
            J = np.zeros((dim, dim))

            # Fill the Jacobian matrix
            for d1 in range(dim):
                for d2 in range(dim):
                    J[d1, d2] = dx[e, q, d1, d2]

            # Compute determinant of Jacobian
            detJ = np.linalg.det(J)

            # Store quadrature weight * |J| for the mass operator
            qdata[e, q, 0] = weights[q] * abs(detJ)

            # For diffusion, would compute J^{-T} here
            # But we'll leave this part out since we only need mass for volume


def apply_mass_diff(u, du, qdata, v, dv, ctx_data):
    """Apply the mass operator with diffusion.

    Args:
        u: Input function at quadrature points (EVAL_INTERP)
        du: Gradient of input function at quadrature points (EVAL_GRAD)
        qdata: Quadrature data (EVAL_NONE)
        v: Output function at quadrature points (EVAL_INTERP)
        dv: Gradient of output function at quadrature points (EVAL_GRAD)
        ctx_data: Context data with dimension information
    """
    dim = ctx_data[0]

    # Get dimensions
    num_elem = u.shape[0]
    num_qpts = u.shape[1]

    # For each element and quadrature point
    for e in range(num_elem):
        for q in range(num_qpts):
            # Mass operator: v = w*|J|*u
            v[e, q, 0] = qdata[e, q, 0] * u[e, q, 0]

            # Diffusion operator: dv = 0 (not needed for volume computation)
            for d in range(dim):
                dv[e, q, d] = 0.0


# ------------------------------------------------------------------------------
# Get mesh size based on problem size
# ------------------------------------------------------------------------------
def get_cartesian_mesh_size(dim, degree, prob_size):
    """Determine mesh size based on approximate problem size.

    Args:
        dim: Dimension of the mesh
        degree: Polynomial degree for the solution
        prob_size: Approximate problem size

    Returns:
        List of element counts in each dimension
    """
    # Use approximate formula: prob_size ~ num_elem * degree^dim
    num_elem = prob_size // (degree**dim)
    s = 0
    while num_elem > 1:
        num_elem //= 2
        s += 1

    num_xyz = [1] * dim
    r = s % dim

    for d in range(dim):
        sd = s // dim
        if r > 0:
            sd += 1
            r -= 1
        num_xyz[d] = 1 << sd

    return num_xyz


# ------------------------------------------------------------------------------
# Build Cartesian restriction
# ------------------------------------------------------------------------------
def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts):
    """Build CeedElemRestriction objects for a Cartesian mesh.

    Args:
        ceed: Ceed object
        dim: Dimension of the mesh
        num_xyz: Number of elements in each dimension
        degree: Polynomial degree
        num_comp: Number of components
        num_qpts: Number of quadrature points in 1D

    Returns:
        Tuple containing:
        - restriction: Element restriction
        - q_data_restriction: Quadrature data restriction
        - size: Size of the vector
    """
    p = degree + 1
    num_nodes = p**dim  # Number of scalar nodes per element
    elem_qpts = num_qpts**dim  # Number of quadrature points per element

    nd = [0] * 3
    num_elem = 1
    scalar_size = 1

    for d in range(dim):
        num_elem *= num_xyz[d]
        nd[d] = num_xyz[d] * (p - 1) + 1
        scalar_size *= nd[d]

    # For mesh coordinates, we need to multiply by the dimension
    # This ensures the mesh coordinates vector size matches what the element restriction expects
    size = scalar_size * num_comp
    elem_nodes = np.zeros(num_elem * num_nodes, dtype=np.int32)

    for e in range(num_elem):
        e_xyz = [0] * 3
        re = e

        for d in range(dim):
            e_xyz[d] = re % num_xyz[d]
            re //= num_xyz[d]

        for l_nodes in range(num_nodes):
            g_nodes = 0
            g_nodes_stride = 1
            r_nodes = l_nodes

            for d in range(dim):
                g_nodes += (e_xyz[d] * (p - 1) + r_nodes % p) * g_nodes_stride
                g_nodes_stride *= nd[d]
                r_nodes //= p

            elem_nodes[e * num_nodes + l_nodes] = g_nodes

    # Create restriction
    # For mesh coordinates, we need to use a different stride
    if num_comp == dim:
        # This is for mesh coordinates - use stride equal to scalar_size
        # This matches the C implementation exactly
        restriction = ceed.ElemRestriction(
            num_elem, num_nodes, num_comp, scalar_size, size, elem_nodes
        )
    else:
        # This is for solution fields
        restriction = ceed.ElemRestriction(
            num_elem, num_nodes, num_comp, 1, scalar_size, elem_nodes
        )

    # Create strided restriction for quadrature data
    q_data_comp = 1  # Changed from 1 + dim * (dim + 1) // 2 to 1 for this example

    # Create q_data restriction using strided approach
    strides = np.array([1, elem_qpts, q_data_comp * elem_qpts], dtype=np.int32)
    q_data_restriction = ceed.StridedElemRestriction(
        num_elem, elem_qpts, q_data_comp, q_data_comp * elem_qpts * num_elem, strides
    )

    return restriction, q_data_restriction, size


# ------------------------------------------------------------------------------
# Set Cartesian mesh coordinates
# ------------------------------------------------------------------------------
def set_cartesian_mesh_coords(dim, num_xyz, mesh_degree, mesh_coords):
    """Set coordinates for a Cartesian mesh.

    Args:
        dim: Dimension of the mesh
        num_xyz: Number of elements in each dimension
        mesh_degree: Polynomial degree for the mesh
        mesh_coords: Vector to store mesh coordinates
    """
    p = mesh_degree + 1
    nd = [0] * 3
    scalar_size = 1

    for d in range(dim):
        nd[d] = num_xyz[d] * (p - 1) + 1
        scalar_size *= nd[d]

    # Get coordinates array
    coords = mesh_coords.get_array_write()

    # Use Lobatto quadrature points as nodes
    temp_ceed = libceed.Ceed("/cpu/self")
    nodes, _ = temp_ceed.lobatto_quadrature(p)
    # nodes are in [-1,1], shift to [0,1]
    nodes = 0.5 + 0.5 * nodes

    # Set coordinates
    for gs_nodes in range(scalar_size):
        r_nodes = gs_nodes

        for d in range(dim):
            d_1d = r_nodes % nd[d]
            # Calculate the coordinate for this dimension
            coord = ((d_1d // (p - 1)) + nodes[d_1d % (p - 1)]) / num_xyz[d]
            # Store the coordinate in the appropriate position
            # Use stride equal to scalar_size for mesh coordinates
            coords[gs_nodes + scalar_size * d] = coord
            r_nodes //= nd[d]

    mesh_coords.restore_array()


# ------------------------------------------------------------------------------
# Transform mesh coordinates for different dimensions
# ------------------------------------------------------------------------------
def transform_mesh_coords(dim, mesh_size, mesh_coords):
    """Apply a transformation to the mesh.

    Args:
        dim: Dimension of the mesh
        mesh_size: Size of the mesh vector
        mesh_coords: Vector with mesh coordinates

    Returns:
        Exact volume of the transformed domain
    """
    with mesh_coords.array_write() as coords:
        if dim == 1:
            # For 1D, transform [0,1] to [0,1] varying the mesh density
            num_nodes = mesh_size
            for i in range(num_nodes):
                coords[i] = 0.5 + 1.0 / math.sqrt(3.0) * math.sin(
                    (2.0 / 3.0) * math.pi * (coords[i] - 0.5)
                )
            exact_volume = 1.0
        else:
            # For dim > 1, map to quarter annulus
            num_nodes = mesh_size // dim

            for i in range(num_nodes):
                u = coords[i]  # First coordinate
                v = coords[i + num_nodes]  # Second coordinate

                # Map to polar coordinates
                u = 1.0 + u  # r in [1,2]
                v = math.pi / 2.0 * v  # phi in [0,pi/2]

                # Transform coordinates
                coords[i] = u * math.cos(v)  # x = r*cos(phi)
                coords[i + num_nodes] = u * math.sin(v)  # y = r*sin(phi)

            exact_volume = 3.0 / 4.0 * math.pi

    return exact_volume


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Example using libCEED with mass operator to compute volume"
    )
    parser.add_argument(
        "-c", "--ceed", type=str, default="/cpu/self", help="CEED resource specifier"
    )
    parser.add_argument(
        "-d", "--dim", type=int, default=3, help="Dimension of the mesh"
    )
    parser.add_argument(
        "-m",
        "--mesh-degree",
        type=int,
        default=4,
        help="Polynomial degree for the mesh",
    )
    parser.add_argument(
        "-p",
        "--sol-degree",
        type=int,
        default=4,
        help="Polynomial degree for the solution",
    )
    parser.add_argument(
        "-q",
        "--num-qpts",
        type=int,
        default=0,
        help="Number of quadrature points (default: sol-degree+2)",
    )
    parser.add_argument(
        "-s",
        "--prob-size",
        type=int,
        default=256 * 1024,
        help="Approximate problem size",
    )
    parser.add_argument(
        "-b", "--benchmark", type=int, default=0, help="Number of benchmark iterations"
    )
    parser.add_argument("-t", "--test", action="store_true", help="Enable test mode")
    parser.add_argument(
        "-g",
        "--gallery",
        action="store_true",
        help="Use gallery QFunction instead of user-defined QFunction",
    )

    args = parser.parse_args()

    # Validate dimension
    if args.dim not in [1, 2, 3]:
        parser.error("Dimension must be 1, 2, or 3")

    # Set default quadrature points
    if args.num_qpts == 0:
        args.num_qpts = args.sol_degree + 2

    # Set default problem size
    if args.prob_size == 0:
        args.prob_size = 8 * 16 if args.test else 256 * 1024

    return args


def run_example_3(args):
    """Run the volume computation example.

    Args:
        args: Command line arguments

    Returns:
        0 on success
    """
    # Set defaults
    dim = args.dim
    mesh_degree = args.mesh_degree
    sol_degree = args.sol_degree
    num_qpts = args.num_qpts
    prob_size = args.prob_size
    test = args.test

    # Print options
    if not test:
        print("Selected options: [command line option] : <current value>")
        print(f"  Ceed specification     [-c] : {args.ceed}")
        print(f"  Mesh dimension         [-d] : {dim}")
        print(f"  Mesh degree            [-m] : {mesh_degree}")
        print(f"  Solution degree        [-p] : {sol_degree}")
        print(f"  Num. 1D quadrature pts [-q] : {num_qpts}")
        print(f"  Approx. # unknowns     [-s] : {prob_size}")
        print(
            f"  QFunction source       [-g] : {'gallery' if args.gallery else 'user'}"
        )
        print("")

    # Initialize libCEED
    ceed = libceed.Ceed(args.ceed)

    # Determine mesh size based on problem size
    num_xyz = get_cartesian_mesh_size(dim, sol_degree, prob_size)
    if not test:
        print(f"Mesh size: nx = {num_xyz[0]}", end="")
        if dim > 1:
            print(f", ny = {num_xyz[1]}", end="")
        if dim > 2:
            print(f", nz = {num_xyz[2]}", end="")
        print("")

    # Build element restrictions for mesh coordinates
    mesh_restriction, _, mesh_size = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, dim, num_qpts
    )

    # Build element restrictions for solution (scalar field)
    sol_restriction, q_data_restriction, sol_size = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts
    )

    num_elem = 1
    for d in range(dim):
        num_elem *= num_xyz[d]

    elem_qpts = num_qpts**dim
    q_data = ceed.Vector(num_elem * elem_qpts)
    q_data.set_value(0.0)

    if not test:
        print(f"Number of mesh nodes     : {mesh_size // dim}")
        print(f"Number of solution nodes : {sol_size}")
        print(f"Mesh size (total)        : {mesh_size}")
        print(f"Solution size (total)    : {sol_size}")

    # Create vector with mesh coordinates
    # For mesh coordinates, we need to multiply by the dimension
    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(dim, num_xyz, mesh_degree, mesh_coords)

    # Apply transformation to the mesh
    exact_volume = transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Context for QFunction
    build_ctx = ceed.QFunctionContext()
    build_ctx_data = {"dim": dim, "space_dim": dim}
    build_ctx.set_data(np.array(list(build_ctx_data.values()), dtype=np.int32))

    # QFunction for mass operator - ALWAYS use gallery QFunction
    qf_build = ceed.QFunctionByName(f"Mass{dim}DBuild")
    qf_build.set_context(build_ctx)

    # Create the mesh basis
    mesh_basis = ceed.BasisTensorH1Lagrange(
        dim, dim, mesh_degree + 1, num_qpts, libceed.GAUSS
    )

    # Create the solution basis
    sol_basis = ceed.BasisTensorH1Lagrange(
        dim, 1, sol_degree + 1, num_qpts, libceed.GAUSS
    )

    # Create the operator for building the quadrature data
    op_build = ceed.Operator(qf_build)

    # Set up the fields for the operator
    op_build.set_field("dx", mesh_restriction, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field(
        "weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE
    )
    op_build.set_field(
        "qdata", q_data_restriction, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE
    )

    # Apply the operator to compute the quadrature data
    op_build.apply(mesh_coords, q_data)

    # Create the QFunction for applying mass operator
    if args.gallery:
        # Use gallery QFunction for mass application
        qf_apply = ceed.QFunctionByName("MassApply")
    else:
        # Use gallery QFunction for mass application
        qf_apply = ceed.QFunctionByName("MassApply")

    # Create the mass operator
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("u", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restriction, BASIS_NONE, q_data)
    op_apply.set_field("v", sol_restriction, sol_basis, VECTOR_ACTIVE)

    # Create solution vectors
    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)

    # Initialize 'u' with ones
    u.set_value(1.0)

    # Compute the mesh volume using the mass operator: volume = 1^T * M * 1
    op_apply.apply(u, v)

    # Benchmark runs
    if not test and args.benchmark > 0:
        print(f" Executing {args.benchmark} benchmarking runs...", end="")
        for _ in range(args.benchmark):
            op_apply.apply(u, v)

    # Compute and print the sum of entries of 'v' giving the mesh volume
    v_array = v.get_array_read()
    volume = np.sum(v_array)
    v.restore_array_read()

    if not test:
        print(" done.")
        print(f"Exact mesh volume    : {exact_volume:.14g}")
        print(f"Computed mesh volume : {volume:.14g}")
        print(f"Volume error         : {volume - exact_volume:.14g}")
    else:
        tol = 200 * np.finfo(float).eps if dim == 1 else 1e-5
        if abs(volume - exact_volume) > tol:
            print(f"Volume error : {volume - exact_volume:.1e}")

    # Cleanup is handled by Python's garbage collector

    return 0


def main():
    """Main function for volume computation example"""
    args = parse_arguments()
    return run_example_3(args)


if __name__ == "__main__":
    main()
