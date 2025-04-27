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
#     ./ex2-surface
#     ./ex2-surface -ceed /cpu/self
#     ./ex2-surface -ceed /gpu/cuda
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

import math
import numpy as np
import argparse
import libceed

# Determine mesh size based on dimension, degree, and problem size
def get_cartesian_mesh_size(dim, degree, prob_size):
    num_elem = prob_size // (degree ** dim)
    s = 0
    while num_elem > 1:
        num_elem //= 2
        s += 1
    r = s % dim
    num_xyz = []
    for d in range(dim):
        sd = s // dim
        if r > 0:
            sd += 1
            r -= 1
        num_xyz.append(1 << sd)
    return num_xyz

# Build Cartesian element restrictions
def build_cartesian_restriction(ceed, dim, num_xyz, degree, num_comp, num_qpts, for_qdata=False):
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

        for l in range(num_nodes):
            g = 0
            stride = 1
            rnode = l
            for d in range(dim):
                g += (e_xyz[d] * (p - 1) + rnode % p) * stride
                stride *= nd[d]
                rnode //= p
            elem_nodes[e * num_nodes + l] = g

    restriction = ceed.ElemRestriction(num_elem, num_nodes, num_comp, 1, size,
                                       elem_nodes, cmode=libceed.COPY_VALUES)

    if for_qdata:
        q_comp = dim * (dim + 1) // 2
        strides = np.array([1, elem_qpts, q_comp * elem_qpts], dtype=np.int32)
        qdata_restriction = ceed.StridedElemRestriction(
            num_elem, elem_qpts, q_comp, q_comp * elem_qpts * num_elem, strides)
        return restriction, qdata_restriction, size, num_elem, elem_qpts

    return restriction, size

# Set mesh coordinates
def set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords):
    p = mesh_degree + 1
    nd = [num_xyz[d] * (p - 1) + 1 for d in range(dim)]
    scalar_size = np.prod(nd)

    coords = np.zeros((scalar_size * dim), dtype=np.float64)
    nodes, _ = ceed.lobatto_quadrature(p)
    nodes = 0.5 + 0.5 * nodes

    for gs in range(scalar_size):
        r = gs
        for d in range(dim):
            d1d = r % nd[d]
            el = d1d // (p - 1)
            pt = d1d % (p - 1)
            coords[gs + scalar_size * d] = (el + nodes[pt]) / num_xyz[d]
            r //= nd[d]

    mesh_coords.set_array(coords, cmode=libceed.COPY_VALUES)

# Transform mesh coordinates
def transform_mesh_coords(dim, mesh_size, mesh_coords):
    with mesh_coords.array_write() as coords:
        for i in range(mesh_size):
            coords[i] = 0.5 + 1. / math.sqrt(3.) * math.sin((2. / 3.) * math.pi * (coords[i] - 0.5))
    exact_surface = 2.0 if dim == 1 else 4.0 if dim == 2 else 6.0
    return exact_surface

def main():
    # Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ceed", default="/cpu/self")
    parser.add_argument("-d", type=int, default=2)
    parser.add_argument("-m", type=int, default=4)
    parser.add_argument("-p", type=int, default=4)
    parser.add_argument("-q", type=int, default=6)
    parser.add_argument("-s", type=int, default=256 * 1024)
    args = parser.parse_args()

    ceed = libceed.Ceed(args.ceed)
    dim = args.d
    mesh_degree = max(args.m, args.p)
    sol_degree = mesh_degree
    num_qpts = args.q
    prob_size = args.s

    print(f"Ceed specification     [-c] : {args.ceed}")
    print(f"Mesh dimension         [-d] : {dim}")
    print(f"Mesh degree            [-m] : {mesh_degree}")
    print(f"Solution degree        [-p] : {sol_degree}")
    print(f"Num. 1D quadrature pts [-q] : {num_qpts}")
    print(f"Approx. # unknowns     [-s] : {prob_size}")

    num_xyz = get_cartesian_mesh_size(dim, sol_degree, prob_size)
    print(f"Mesh size: nx = {num_xyz[0]}" + (f", ny = {num_xyz[1]}" if dim > 1 else "") + (f", nz = {num_xyz[2]}" if dim > 2 else ""))

    mesh_restr, mesh_size = build_cartesian_restriction(ceed, dim, num_xyz, mesh_degree, dim, num_qpts)
    sol_restr, qdata_restr, sol_size, num_elem, elem_qpts = build_cartesian_restriction(
        ceed, dim, num_xyz, sol_degree, 1, num_qpts, for_qdata=True)

    mesh_coords = ceed.Vector(mesh_size)
    set_cartesian_mesh_coords(ceed, dim, num_xyz, mesh_degree, mesh_coords)

    exact_surface = transform_mesh_coords(dim, mesh_size, mesh_coords)

    qdata = ceed.Vector(num_elem * elem_qpts * (dim * (dim + 1) // 2))

    mesh_basis = ceed.BasisTensorH1Lagrange(dim, dim, mesh_degree + 1, num_qpts, libceed.GAUSS)
    sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree + 1, num_qpts, libceed.GAUSS)

    # Setup operator for building quadrature data
    qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")
    op_build = ceed.Operator(qf_build)
    op_build.set_field("dx", mesh_restr, mesh_basis, libceed.VECTOR_ACTIVE)
    op_build.set_field("weights", libceed.ELEMRESTRICTION_NONE, mesh_basis, libceed.VECTOR_NONE)
    op_build.set_field("qdata", qdata_restr, libceed.BASIS_NONE, libceed.VECTOR_ACTIVE)
    op_build.apply(mesh_coords, qdata)

    with qdata.array_write() as qdata_array:
        qdata_array[:] = np.abs(qdata_array[:])

    # Setup operator for applying quadrature data
    qf_apply = ceed.QFunctionByName(f"Poisson{dim}DApply")
    op_apply = ceed.Operator(qf_apply)
    op_apply.set_field("du", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)
    op_apply.set_field("qdata", qdata_restr, libceed.BASIS_NONE, qdata)
    op_apply.set_field("dv", sol_restr, sol_basis, libceed.VECTOR_ACTIVE)

    u = ceed.Vector(sol_size)
    v = ceed.Vector(sol_size)

    u.set_value(1.0)
    v.set_value(0.0)

    op_apply.apply(u, v)

    with v.array_read() as v_array:
        surface = np.sum(np.abs(v_array))

    print()
    print(f"Exact mesh surface area    : {exact_surface:.14g}")
    print(f"Computed mesh surface area : {surface:.14g}")
    print(f"Surface area error         : {surface - exact_surface:.14g}")

if __name__ == "__main__":
    main()
