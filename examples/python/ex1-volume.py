#!/usr/bin/env python3
"""
Volume Computation Example using libCEED

This example computes the volume of a 1D, 2D, or 3D body using matrix-free
application of a mass operator.

Usage:
  python ex1-volume.py -d DIM [-m MDEG] [-p PDEG] [-q QPTS] [-c CEED] [-s SIZE] [-t] [-g]

Arguments:
  -d DIM   : Dimension (1, 2, or 3)
  -m MDEG  : Mesh polynomial degree (default: 4)
  -p PDEG  : Solution polynomial degree (default: 4)
  -q QPTS  : Number of quadrature points (default: p+2)
  -c CEED  : CEED resource specifier (default: /cpu/self)
  -s SIZE  : Approximate problem size (default: 256*1024)
  -t       : Test mode with smaller problem size (default: False)
  -g       : Use gallery QFunction instead of user-defined QFunction
"""

import sys
import argparse
import math
import numpy as np
import libceed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compute volume using libCEED")
    parser.add_argument("-c", "--ceed", default="/cpu/self",
                       help="CEED resource specifier")
    parser.add_argument("-d", "--dim", type=int, default=3,
                       help="Dimension (1, 2, or 3)")
    parser.add_argument("-m", "--mesh-degree", type=int, default=4,
                       help="Mesh polynomial degree")
    parser.add_argument("-p", "--solution-degree", type=int, default=4,
                       help="Solution polynomial degree")
    parser.add_argument("-q", "--quadrature-points", type=int, default=0,
                       help="Number of quadrature points (0 = default, p+2)")
    parser.add_argument("-s", "--problem-size", type=int, default=0,
                       help="Approximate problem size (0 = default, 256*1024)")
    parser.add_argument("-t", "--test", action="store_true",
                       help="Test mode with smaller problem size")
    parser.add_argument("-g", "--gallery", action="store_true",
                       help="Use gallery QFunction instead of user-defined QFunction")

    args = parser.parse_args()

    # Validate dimension
    if args.dim not in [1, 2, 3]:
        parser.error("Dimension must be 1, 2, or 3")

    # Set default quadrature points
    if args.quadrature_points == 0:
        args.quadrature_points = args.solution_degree + 2

    # Set default problem size
    if args.problem_size == 0:
        args.problem_size = 8*16 if args.test else 256*1024

    return args

def get_cartesian_mesh_size(dim, degree, prob_size):
    """Determine mesh size based on approximate problem size"""
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

def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts, create_qdata=True):
    """Build element restrictions for the mesh and solution"""
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

    # Set up element nodes (connectivity)
    elem_nodes = np.zeros(num_elem * num_nodes, dtype=np.int32)

    for e in range(num_elem):
        e_xyz = [0] * 3
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

    # Create element restriction
    elem_restriction = ceed.ElemRestriction(num_elem, num_nodes, num_comp, 1, size,
                                           elem_nodes, cmode=libceed.COPY_VALUES)

    # Create strided restriction for quadrature data
    q_data_restriction = None
    if create_qdata:
        # For quadrature data, we use standard strided setup with component stride = 1
        # and element stride = elem_qpts
        # First, let's try using basic restriction with identity indexing
        q_indices = np.arange(num_elem * elem_qpts, dtype=np.int32)
        q_data_restriction = ceed.ElemRestriction(
            num_elem, elem_qpts, 1, 1, num_elem * elem_qpts,
            q_indices, cmode=libceed.COPY_VALUES)

    return elem_restriction, size, q_data_restriction, num_elem, elem_qpts

def set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords):
    """Set the initial Cartesian mesh coordinates"""
    p = mesh_degree + 1
    nd = []
    scalar_size = 1

    for d in range(dim):
        nd.append(num_xyz[d] * (p - 1) + 1)
        scalar_size *= nd[d]

    # Create coordinate array
    coords = np.zeros(scalar_size * dim)

    # Get Lobatto nodes
    nodes, _ = ceed.lobatto_quadrature(p)

    # Shift from [-1,1] to [0,1]
    nodes = 0.5 + 0.5 * nodes

    # Set coordinates
    for gs_node in range(scalar_size):
        r_node = gs_node

        for d in range(dim):
            d_1d = r_node % nd[d]
            elem_id = d_1d // (p - 1)
            node_id = d_1d % (p - 1)
            coords[gs_node + scalar_size * d] = (elem_id + nodes[node_id]) / num_xyz[d]
            r_node //= nd[d]

    # Initialize mesh_coords vector and set array
    mesh_coords.set_value(0.0)
    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)

def transform_mesh_coords(dim, mesh_size, mesh_coords):
    """Apply transformation to mesh coordinates and return exact volume"""
    with mesh_coords.array_write() as coords:
        if dim == 1:
            # Map [0,1] to [0,1] varying the mesh density
            for i in range(mesh_size):
                coords[i] = 0.5 + 1.0 / math.sqrt(3.0) * math.sin((2.0 / 3.0) * math.pi * (coords[i] - 0.5))
            exact_volume = 1.0
        else:  # dim == 2 or dim == 3
            # Transform to quarter annulus
            num_nodes = mesh_size // dim

            for i in range(num_nodes):
                u = coords[i]  # First coordinate
                v = coords[i + num_nodes]  # Second coordinate

                # Transform to polar coordinates
                u = 1.0 + u  # r in [1,2]
                v = math.pi/2 * v  # phi in [0,pi/2]

                # Apply polar transformation
                coords[i] = u * math.cos(v)  # x = r*cos(phi)
                coords[i + num_nodes] = u * math.sin(v)  # y = r*sin(phi)
                # For 3D, z coordinate stays the same

            exact_volume = 3.0 / 4.0 * math.pi  # Volume of quarter annulus

    return exact_volume

def main():
    """Main function for volume computation example"""
    args = parse_arguments()

    # Initialize libCEED
    ceed = libceed.Ceed(args.ceed)

    # Dimensions
    dim = args.dim
    num_comp_x = dim
    mesh_degree = args.mesh_degree
    sol_degree = args.solution_degree
    num_qpts = args.quadrature_points

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

    # Determine mesh size
    num_xyz = get_cartesian_mesh_size(dim, sol_degree, args.problem_size)
    if not args.test:
        print(f"Mesh size: nx = {num_xyz[0]}", end="")
        if dim > 1: print(f", ny = {num_xyz[1]}", end="")
        if dim > 2: print(f", nz = {num_xyz[2]}", end="")
        print()

    # Create bases
    mesh_basis = ceed.BasisTensorH1Lagrange(dim, num_comp_x, mesh_degree+1, num_qpts, libceed.GAUSS)
    sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree+1, num_qpts, libceed.GAUSS)

    # Build element restrictions
    mesh_restriction, mesh_size, _, _, _ = build_cartesian_restriction(
        ceed, dim, num_xyz, mesh_degree, num_comp_x, num_qpts, create_qdata=False)
    sol_restriction, sol_size, q_data_restriction, num_elem, elem_qpts = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts, create_qdata=True)

    if not args.test:
        print(f"Number of mesh nodes     : {mesh_size//dim}")
        print(f"Number of solution nodes : {sol_size}")

    # Create mesh coordinates vector
    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords)

    # Apply transformation to mesh
    exact_volume = transform_mesh_coords(dim, mesh_size, mesh_coords)

    # Setup QFunction for building the mass operator
    # In this example, we always use the gallery QFunction
    # If we had user-defined QFunctions, we would use a conditional here
    qf_build = ceed.QFunctionByName(f"Mass{dim}DBuild")
    
    # Create the operator that builds the quadrature data for the mass operator
    op_build = ceed.Operator(qf_build)
    
    # For Mass{dim}DBuild, the "dx" field needs to use EVAL_GRAD for dimensions > 1
    # This is a critical difference between the 1D case and higher dimensions
    op_build.set_field("dx", mesh_restriction, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    # Compute the quadrature data for the mass operator
    q_data = ceed.Vector(num_elem * elem_qpts)
    q_data.set_value(0.0)
    op_build.apply(mesh_coords, q_data)
    
    # Take absolute value of quadrature data to ensure positive volume
    # This is needed because the determinant of the Jacobian might be negative
    # depending on the element orientation
    with q_data.array_write() as qdata:
        qdata[:] = np.abs(qdata[:])

    # Setup QFunction for applying the mass operator
    # In this example, we always use the gallery QFunction
    # If we had user-defined QFunctions, we would use a conditional here
    qf_mass = ceed.QFunctionByName("MassApply")
    
    # Create the mass operator
    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("u", sol_restriction, sol_basis, libceed.VECTOR_ACTIVE)
    op_mass.set_field("qdata", q_data_restriction, libceed.BASIS_NONE, q_data)
    op_mass.set_field("v", sol_restriction, sol_basis, libceed.VECTOR_ACTIVE)

    # Create solution vectors
    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)
    u.set_value(1.0)  # Set all entries of u to 1.0
    v.set_value(0.0)  # Set all entries of v to 0.0

    # Apply mass operator: v = M * u
    op_mass.apply(u, v)

    # Compute volume by summing all entries in v
    volume = 0.0
    with v.array_read() as v_array:
        # Simply sum all values to compute the volume
        volume = np.sum(v_array)
        
        # Optional debug info
        if not args.test and dim > 1:
            volume_abs = np.sum(np.abs(v_array))
            print(f"Raw volume: {volume:.14g}")
            print(f"Abs volume: {volume_abs:.14g}")

    if not args.test:
        print()
        print(f"Exact mesh volume    : {exact_volume:.14g}")
        print(f"Computed mesh volume : {volume:.14g}")
        print(f"Volume error         : {volume - exact_volume:.14g}")
        rel_error = abs((volume - exact_volume)/exact_volume)
        print(f"Relative error       : {rel_error:.14g}")
    else:
        # Test mode - check if error is within tolerance
        tol = 200 * libceed.EPSILON if dim == 1 else 1e-5
        if abs(volume - exact_volume) > tol:
            print(f"Volume error : {volume - exact_volume:.1e}")
            sys.exit(1)

    return 0

if __name__ == "__main__":
    sys.exit(main())