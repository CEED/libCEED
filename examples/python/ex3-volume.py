#!/usr/bin/env python3
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

"""
Example using libCEED to compute the volume of a 3D body using matrix-free application of a mass operator.
This example also uses a diffusion operator, which provides zero contribution to the computed volume but
demonstrates libCEED's ability to handle multiple basis evaluation modes for the same input and output vectors.
The example supports arbitrary mesh and solution degrees in 1D, 2D, and 3D from the same code.
"""

import argparse
import math
import numpy as np
import libceed
from libceed import GAUSS, VECTOR_ACTIVE, VECTOR_NONE, ELEMRESTRICTION_NONE, BASIS_NONE
from libceed import EVAL_INTERP, EVAL_GRAD, EVAL_WEIGHT, EVAL_NONE
from libceed import MEM_HOST, COPY_VALUES, USE_POINTER
import os
from sysconfig import get_config_var
import ctypes


def find_qfs_so(name, path):
    """Find the QFunctions shared library.
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    Returns:
        Filepath to shared library object
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def load_qfs_so():
    """Load the QFunctions shared library.
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    Returns:
        Loaded shared library object
    """
    file_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "build")
    qfs_so = find_qfs_so(
        "libceed_qfunctions" + get_config_var("EXT_SUFFIX"),
        file_dir)


<< << << < Updated upstream

# Load library
return ctypes.cdll.LoadLibrary(qfs_so)

== == == =

# Load library
return ctypes.cdll.LoadLibrary(qfs_so)
>>>>>> > Stashed changes


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
def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts, args=None):
    """Build CeedElemRestriction objects for a Cartesian mesh.
    Args:
        ceed: Ceed object
        dim: Dimension of the mesh
        num_xyz: Number of elements in each dimension
        degree: Polynomial degree
        num_comp: Number of components
        num_qpts: Number of quadrature points in 1D
        args: Command line arguments
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
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
        restriction = ceed.ElemRestriction(
            num_elem, num_nodes, num_comp, scalar_size, size, elem_nodes
        )
    else:
        # This is for solution fields
        restriction = ceed.ElemRestriction(
            num_elem, num_nodes, num_comp, 1, scalar_size, elem_nodes
        )

    # Create q_data restriction using strided approach
    # Number of components depends on dimension:
    # 1D: 2 components (1 for mass, 1 for diffusion)
    # 2D: 4 components (1 for mass, 3 for diffusion in Voigt notation)
    # 3D: 7 components (1 for mass, 6 for diffusion in Voigt notation)
    q_data_comp = 7 if dim == 3 else (4 if dim == 2 else 2)
    strides = np.array([1, elem_qpts, q_data_comp * elem_qpts], dtype=np.int32)
    q_data_restriction = ceed.StridedElemRestriction(
        num_elem, elem_qpts, q_data_comp,
        q_data_comp * elem_qpts * num_elem,  # Total size is number of elements times quadrature points
        strides
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
        "-c",
        "--ceed",
        default="/cpu/self",
        help="CEED resource specifier",
        dest="ceed_spec",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=3,
        help="Mesh dimension (default: 3)",
    )
    parser.add_argument(
        "-m",
        "--mesh-degree",
        type=int,
        default=4,
        help="Mesh polynomial degree (default: 4)",
        dest="mesh_degree",
    )
    parser.add_argument(
        "-p",
        "--solution-degree",
        type=int,
        default=4,
        help="Solution polynomial degree (default: 4)",
        dest="sol_degree",
    )
    parser.add_argument(
        "-q",
        "--quadrature-points",
        type=int,
        default=6,
        help="Number of quadrature points in 1D (default: 6)",
        dest="num_qpts",
    )
    parser.add_argument(
        "-s",
        "--problem-size",
        type=int,
        default=1000,
        help="Approximate problem size (default: 1000)",
        dest="problem_size",
    )
    return parser.parse_args()


def run_example_3(args):
    """Run Example 3.
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    Args:
        args: Parsed command line arguments
    """
    # Process command line arguments
    dim = args.dim
    mesh_degree = args.mesh_degree
    sol_degree = args.sol_degree
    num_qpts = args.num_qpts
    prob_size = args.problem_size
    ceed_spec = args.ceed_spec

    # Set up the mesh
    num_xyz = get_cartesian_mesh_size(dim, sol_degree, prob_size)
    mesh_degree = max(1, mesh_degree)  # Mesh degree must be at least 1

    # Print summary
    print(f"Selected options: mesh degree = {mesh_degree}, solution degree = {sol_degree}, "
          f"number of 1D quadrature points = {num_qpts}")
    print(f"Mesh size: {' x '.join(map(str, num_xyz))} elements")

    # Set up libCEED
    ceed = libceed.Ceed(ceed_spec)

    # Build CeedElemRestriction objects describing the mesh and solution fields
    mesh_restriction, _, mesh_size = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, dim, num_qpts)
    sol_restriction, q_data_restriction, sol_size = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts)

    # Create Vectors
    mesh_coords = ceed.Vector(mesh_size)
    q_data_comp = 7 if dim == 3 else (4 if dim == 2 else 2)
    q_data = ceed.Vector(q_data_comp * num_qpts**dim * np.prod(num_xyz))
    x = ceed.Vector(sol_size)
    ones = ceed.Vector(sol_size)
    vol = ceed.Vector(1)

    # Set up mesh coordinates
    set_cartesian_mesh_coords(dim, num_xyz, mesh_degree, mesh_coords)
    exact_volume = transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Create bases
    mesh_basis = ceed.BasisTensorH1Lagrange(dim, dim, mesh_degree + 1, num_qpts,
                                            GAUSS)
    sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree + 1, num_qpts,
                                           GAUSS)

    # Create QFunction context
    build_ctx = ceed.QFunctionContext()
    ctx_data = np.array([dim, dim], dtype=np.int32)
    build_ctx.set_data(ctx_data)

    # Load QFunctions
    qfs = load_qfs_so()
    file_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the QFunctions
    qf_setup = ceed.QFunction(1, qfs.build_mass_diff,
                              os.path.join(file_dir, "ex3-volume.h:build_mass_diff"))
    qf_setup.add_input("dx", dim * dim, EVAL_GRAD)
    qf_setup.add_input("weights", 1, EVAL_WEIGHT)
    qf_setup.add_output("qdata", q_data_comp, EVAL_NONE)
    qf_setup.set_context(build_ctx)

    qf_apply = ceed.QFunction(1, qfs.apply_mass_diff,
                              os.path.join(file_dir, "ex3-volume.h:apply_mass_diff"))
    qf_apply.add_input("u", 1, EVAL_INTERP)
    qf_apply.add_input("ug", dim, EVAL_GRAD)
    qf_apply.add_input("qdata", q_data_comp, EVAL_NONE)
    qf_apply.add_output("v", 1, EVAL_INTERP)
    qf_apply.add_output("vg", dim, EVAL_GRAD)
    qf_apply.set_context(build_ctx)

    # Create the operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("dx", mesh_restriction, mesh_basis, VECTOR_ACTIVE)
    op_setup.set_field("weights", ELEMRESTRICTION_NONE, mesh_basis, VECTOR_NONE)
    op_setup.set_field("qdata", q_data_restriction, BASIS_NONE, VECTOR_ACTIVE)

    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("u", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("ug", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restriction, BASIS_NONE, q_data)
    op_apply.set_field("v", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("vg", sol_restriction, sol_basis, VECTOR_ACTIVE)

    # Setup
    op_setup.apply(mesh_coords, q_data)

    # Apply mass operator
    ones.set_value(1.0)
    op_apply.apply(ones, x)

    # Compute the volume
    vol.set_value(0.0)
    with x.array_read() as x_array:
        volume = np.sum(x_array)
        vol.set_value(volume)

    # Print the computed and exact volumes
    with vol.array_read() as vol_array:
        computed_volume = vol_array[0]
        print(f"Exact volume    = {exact_volume:.14f}")
        print(f"Computed volume = {computed_volume:.14f}")
        print(f"Error          = {abs(computed_volume - exact_volume):.14e}")


def main():
    """Run the example."""
    args = parse_arguments()
    run_example_3(args)


if __name__ == "__main__":
    main()
