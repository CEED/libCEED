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
using matrix-free application of a mass operator. This example also uses a diffusion 
operator, which provides zero contribution to the computed volume but demonstrates 
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

# Import QFunctions
from ex3_volume_qfunctions import build_mass_diff, apply_mass_diff

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
    num_elem = prob_size // (degree ** dim)
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
    num_nodes = p ** dim  # Number of scalar nodes per element
    elem_qpts = num_qpts ** dim  # Number of quadrature points per element
    
    nd = [0] * 3
    num_elem = 1
    scalar_size = 1
    
    for d in range(dim):
        num_elem *= num_xyz[d]
        nd[d] = num_xyz[d] * (p - 1) + 1
        scalar_size *= nd[d]
    
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
    
    # Create restrictions
    # For the mesh coordinates, we need to handle multiple components
    # The element restriction should be set up for the same number of components as the basis
    # The size of the element restriction should be scalar_size, not size
    # The stride of the element restriction should be 1, not num_comp
    restriction = ceed.ElemRestriction(
        num_elem, num_nodes, num_comp, 1, scalar_size, 
        elem_nodes)
    
    q_data_comp = 1 + dim * (dim + 1) // 2
    
    # Create strides array for StridedElemRestriction
    strides = np.array([1, elem_qpts, q_data_comp * elem_qpts], dtype=np.int32)
    
    q_data_restriction = ceed.StridedElemRestriction(
        num_elem, elem_qpts, q_data_comp,
        q_data_comp * elem_qpts * num_elem,
        strides)
    
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
    # Create a temporary Ceed object to get the quadrature points
    temp_ceed = libceed.Ceed("/cpu/self")
    nodes, _ = temp_ceed.lobatto_quadrature(p)
    # nodes are in [-1,1], shift to [0,1]
    nodes = 0.5 + 0.5 * nodes
    
    # Set coordinates
    for gs_nodes in range(scalar_size):
        r_nodes = gs_nodes
        
        for d in range(dim):
            d_1d = r_nodes % nd[d]
            coords[gs_nodes * dim + d] = ((d_1d // (p - 1)) + nodes[d_1d % (p - 1)]) / num_xyz[d]
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
    coords = mesh_coords.get_array()
    
    if dim == 1:
        for i in range(mesh_size):
            # Map [0,1] to [0,1] varying the mesh density
            coords[i] = 0.5 + 1.0 / math.sqrt(3.0) * math.sin((2.0 / 3.0) * math.pi * (coords[i] - 0.5))
        exact_volume = 1.0
    else:
        # For dim > 1
        # Map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
        # coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
        num_nodes = mesh_size // dim
        
        for i in range(num_nodes):
            if dim >= 2:
                u = coords[i * dim]
                v = coords[i * dim + 1]
                
                u = 1.0 + u
                v = math.pi / 2.0 * v
                coords[i * dim] = u * math.cos(v)
                coords[i * dim + 1] = u * math.sin(v)
        
        exact_volume = 3.0 / 4.0 * math.pi
    
    mesh_coords.restore_array()
    return exact_volume


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    """Execute the example."""
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Example using libCEED with mass operator to compute volume')
    parser.add_argument('-c', '--ceed', type=str, default='/cpu/self',
                        help='CEED resource specifier')
    parser.add_argument('-d', '--dim', type=int, default=3,
                        help='Dimension of the mesh')
    parser.add_argument('-m', '--mesh-degree', type=int, default=4,
                        help='Polynomial degree for the mesh')
    parser.add_argument('-p', '--sol-degree', type=int, default=4,
                        help='Polynomial degree for the solution')
    parser.add_argument('-q', '--num-qpts', type=int, default=0,
                        help='Number of quadrature points (default: sol-degree+2)')
    parser.add_argument('-s', '--prob-size', type=int, default=256*1024,
                        help='Approximate problem size')
    parser.add_argument('-b', '--benchmark', type=int, default=0,
                        help='Number of benchmark iterations')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Enable test mode')
    
    # Add an alias for -q to --num-qpts
    parser.add_argument('--q', type=int, dest='num_qpts',
                        help='Alias for --num-qpts')
    
    args = parser.parse_args()
    
    # Set defaults
    dim = args.dim
    num_comp_x = dim
    mesh_degree = args.mesh_degree
    sol_degree = args.sol_degree
    num_qpts = args.num_qpts if args.num_qpts > 0 else sol_degree + 2
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
        print(f"  QFunction source            : Python")
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
    
    # Build element restrictions
    mesh_restriction, q_data_restriction, mesh_size = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, dim, num_qpts)

    sol_restriction, _, sol_size = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts)
    
    num_elem = 1
    for d in range(dim):
        num_elem *= num_xyz[d]
    
    elem_qpts = num_qpts ** dim
    q_data_comp = 1 + dim * (dim + 1) // 2
    q_data = ceed.Vector(num_elem * elem_qpts * q_data_comp)
    
    if not test:
        print(f"Number of mesh nodes     : {mesh_size // dim}")
        print(f"Number of solution nodes : {sol_size}")
        print(f"Mesh size (total)        : {mesh_size}")
        print(f"Solution size (total)    : {sol_size}")
    
    # Create vector with mesh coordinates
    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(dim, num_xyz, mesh_degree, mesh_coords)
    
    # Print vector sizes for debugging
    if not test:
        print(f"Mesh coordinates vector size: {mesh_coords.get_length()}")
        print(f"Element restriction expected size: {mesh_size}")
    
    # Apply transformation to the mesh
    exact_volume = transform_mesh_coords(dim, mesh_size, mesh_coords)
    
    # Context for QFunction
    build_ctx = ceed.QFunctionContext()
    build_ctx_data = {"dim": dim, "space_dim": dim}
    build_ctx.set_data(np.array(list(build_ctx_data.values()), dtype=np.int32))
    
    # Create the QFunction that builds the mass + diffusion operator
    qf_build = ceed.QFunction(1, "build_mass_diff", "ex3-volume-cl.py:build_mass_diff")
    qf_build.add_input("dx", dim * dim, EVAL_GRAD)
    qf_build.add_input("weights", 1, EVAL_WEIGHT)
    qf_build.add_output("qdata", q_data_comp, EVAL_NONE)
    qf_build.set_context(build_ctx)
    
    # Create the mesh basis
    # The mesh basis should have dim components to match the element restriction
    mesh_basis = ceed.BasisTensorH1Lagrange(dim, dim, mesh_degree + 1, 
                                             num_qpts, GAUSS)
    
    # Create the solution basis
    sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree + 1, 
                                            num_qpts, GAUSS)
    
    # Create a new operator for building the quadrature data
    op_build = ceed.Operator(qf_build)
    
    # Set up the fields for the operator
    op_build.set_field("dx", mesh_restriction, mesh_basis, VECTOR_ACTIVE)
    op_build.set_field("weights", ELEMRESTRICTION_NONE, mesh_basis, VECTOR_NONE)
    op_build.set_field("qdata", q_data_restriction, BASIS_NONE, VECTOR_ACTIVE)
    
    # Apply the operator to compute the quadrature data
    op_build.apply(mesh_coords, q_data)
    
    # Create the QFunction that defines the action of the mass + diffusion operator
    qf_apply = ceed.QFunction(1, "apply_mass_diff", "ex3-volume-cl.py:apply_mass_diff")
    qf_apply.add_input("u", 1, EVAL_INTERP)
    qf_apply.add_input("du", dim, EVAL_GRAD)
    qf_apply.add_input("qdata", q_data_comp, EVAL_NONE)
    qf_apply.add_output("v", 1, EVAL_INTERP)
    qf_apply.add_output("dv", dim, EVAL_GRAD)
    
    # Set the context
    qf_apply.set_context(build_ctx)
    
    # Create the mass + diffusion operator
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("u", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("du", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restriction, BASIS_NONE, q_data)
    op_apply.set_field("v", sol_restriction, sol_basis, VECTOR_ACTIVE)
    op_apply.set_field("dv", sol_restriction, sol_basis, VECTOR_ACTIVE)
    
    # Create auxiliary solution-size vectors
    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)
    
    # Initialize 'u' with ones
    u.set_value(1.0)
    
    # Compute the mesh volume using the mass + diffusion operator: volume = 1^T * M * 1
    op_apply.apply(u, v)
    
    # Benchmark runs
    if not test and args.benchmark > 0:
        print(f" Executing {args.benchmark} benchmarking runs...")
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


if __name__ == "__main__":
    main()