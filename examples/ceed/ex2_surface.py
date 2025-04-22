#!/usr/bin/env python3
"""
Surface Area Computation Example using libCEED
This example computes the surface area of a 1D, 2D, or 3D body using matrix-free
application of a diffusion operator.
Usage:
  python ex2-surface.py -d DIM [-m MDEG] [-p PDEG] [-q QPTS] [-c CEED] [-s SIZE] [-t] [-g]
"""

import sys
import argparse
import math
import numpy as np
import libceed


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute surface area using libCEED")
    parser.add_argument("-c", "--ceed", default="/cpu/self", help="CEED resource specifier")
    parser.add_argument("-d", "--dim", type=int, default=3, help="Dimension (1, 2, or 3)")
    parser.add_argument("-m", "--mesh-degree", type=int, default=4, help="Mesh polynomial degree")
    parser.add_argument("-p", "--solution-degree", type=int, default=4, help="Solution polynomial degree")
    parser.add_argument("-q", "--quadrature-points", type=int, default=0, help="Quadrature points (0 = p+2)")
    parser.add_argument("-s", "--problem-size", type=int, default=0, help="Approximate problem size (0 = 256*1024)")
    parser.add_argument("-t", "--test", action="store_true", help="Test mode")
    parser.add_argument("-g", "--gallery", action="store_true", help="Use gallery QFunction")
    return parser.parse_args()


def get_cartesian_mesh_size(dim, degree, prob_size):
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
    p = degree + 1
    num_nodes = p ** dim
    elem_qpts = num_qpts ** dim
    nd = [num_xyz[d] * (p - 1) + 1 for d in range(dim)]
    scalar_size = np.prod(nd)
    num_elem = np.prod(num_xyz)
    size = scalar_size * num_comp

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

    elem_restr = ceed.ElemRestriction(num_elem, num_nodes, num_comp, 1, size,
                                      elem_nodes, cmode=libceed.COPY_VALUES)

    q_data_restr = None
    if create_qdata:
        q_indices = np.arange(num_elem * elem_qpts, dtype=np.int32)
        q_data_restr = ceed.ElemRestriction(
            num_elem, elem_qpts, num_comp, 1, num_elem * elem_qpts * num_comp,
            q_indices, cmode=libceed.COPY_VALUES)

    return elem_restr, size, q_data_restr, num_elem, elem_qpts


def set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords):
    p = mesh_degree + 1
    nd = [num_xyz[d] * (p - 1) + 1 for d in range(dim)]
    scalar_size = np.prod(nd)
    coords = np.zeros(scalar_size * dim)
    nodes, _ = ceed.lobatto_quadrature(p)
    nodes = 0.5 + 0.5 * nodes
    for gs_node in range(scalar_size):
        r_node = gs_node
        for d in range(dim):
            d1d = r_node % nd[d]
            coords[gs_node + scalar_size * d] = ((d1d // (p - 1)) + nodes[d1d % (p - 1)]) / num_xyz[d]
            r_node //= nd[d]
    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)
    return scalar_size


def transform_mesh_coords(dim, mesh_size, mesh_coords):
    with mesh_coords.array_write() as coords:
        if dim == 1:
            for i in range(mesh_size):
                coords[i] = 0.5 + 1.0 / math.sqrt(3.0) * math.sin((2.0 / 3.0) * math.pi * (coords[i] - 0.5))
        else:
            num_nodes = mesh_size // dim
            for i in range(num_nodes):
                u = coords[i]
                v = coords[i + num_nodes]
                u = 1.0 + u
                v = math.pi / 2 * v
                coords[i] = u * math.cos(v)
                coords[i + num_nodes] = u * math.sin(v)
    return 2 if dim == 1 else 4 if dim == 2 else 6


def main():
    args = parse_arguments()
    ceed = libceed.Ceed(args.ceed)

    dim = args.dim
    num_comp_x = dim
    mesh_degree = max(args.mesh_degree, args.solution_degree)
    sol_degree = mesh_degree
    num_qpts = args.quadrature_points or sol_degree + 2
    prob_size = args.problem_size or (16 * 16 * dim * dim if args.test else 256 * 1024)

    if not args.test:
        print(f"Ceed specification     [-c] : {args.ceed}")
        print(f"Mesh dimension         [-d] : {dim}")
        print(f"Mesh degree            [-m] : {mesh_degree}")
        print(f"Solution degree        [-p] : {sol_degree}")
        print(f"Num. 1D quadrature pts [-q] : {num_qpts}")
        print(f"Approx. # unknowns     [-s] : {prob_size}")
        print(f"QFunction source       [-g] : {'gallery' if args.gallery else 'user'}")

    num_xyz = get_cartesian_mesh_size(dim, sol_degree, prob_size)
    if not args.test:
        print(f"Mesh size: nx = {num_xyz[0]}", end="")
        if dim > 1: print(f", ny = {num_xyz[1]}", end="")
        if dim > 2: print(f", nz = {num_xyz[2]}", end="")
        print()

    mesh_basis = ceed.BasisTensorH1Lagrange(dim, num_comp_x, mesh_degree + 1, num_qpts, libceed.GAUSS)
    sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree + 1, num_qpts, libceed.GAUSS)

    mesh_restr, mesh_size, _, _, _ = build_cartesian_restriction(
    ceed, dim, num_xyz, mesh_degree, num_comp_x, num_qpts, create_qdata=False)

    sol_restr_q, _, q_data_restr, num_elem, elem_qpts = build_cartesian_restriction(
    ceed, dim, num_xyz, sol_degree, dim*(dim + 1)//2, num_qpts)

    sol_restr, sol_size, _, _, _ = build_cartesian_restriction(
    ceed, dim, num_xyz, sol_degree, 1, num_qpts, create_qdata=False)
   

    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords)
    exact_surface_area = transform_mesh_coords(dim, mesh_size, mesh_coords)

    qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restr, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", q_data_restr, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)

    q_data = ceed.Vector(num_elem * elem_qpts * dim * (dim + 1) // 2)
    op_build.apply(mesh_coords, q_data)

    qf_apply = ceed.QFunctionByName(f"Poisson{dim}DApply")
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("du", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)
    op_apply.set_field("qdata", q_data_restr, libceed.BASIS_NONE, q_data)
    op_apply.set_field("dv", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)

    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)
    with mesh_coords.array_read() as x_array, u.array_write() as u_array:
        num_nodes = mesh_size // dim
        for i in range(num_nodes):
            u_array[i] = sum(x_array[i + j * num_nodes] for j in range(dim))


    op_apply.apply(u, v)

    with v.array_read() as v_array:
        surface_area = np.sum(np.abs(v_array))

    if not args.test:
        print(f"\nExact mesh surface area    : {exact_surface_area:.14g}")
        print(f"Computed mesh surface area : {surface_area:.14g}")
        print(f"Surface area error         : {surface_area - exact_surface_area:.14g}")
    else:
        tol = 10000 * libceed.EPSILON if dim == 1 else 1E-1
        if abs(surface_area - exact_surface_area) > tol:
            print(f"Surface area error         : {surface_area - exact_surface_area:.14g}")
            sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

