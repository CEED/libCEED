#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:41:54 2025

@author: surindersinghchhabra
"""

import numpy as np
from math import fabs
import sys
import argparse
from libceed import Ceed, _ceed_cffi
from libceed.ceed_elemrestriction import ElemRestriction


# Import CEED constants from the compiled CFFI bindings
lib = _ceed_cffi.lib

QMODE_GAUSS        = lib.CEED_GAUSS
CEED_EVAL_GRAD     = lib.CEED_EVAL_GRAD
CEED_EVAL_WEIGHT   = lib.CEED_EVAL_WEIGHT
CEED_EVAL_NONE     = lib.CEED_EVAL_NONE
CEED_VECTOR_ACTIVE = lib.CEED_VECTOR_ACTIVE
CEED_VECTOR_NONE   = lib.CEED_VECTOR_NONE

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-ceed", type=str, default="/cpu/self")
parser.add_argument("-d", type=int, default=3)
parser.add_argument("-m", type=int, default=4)
parser.add_argument("-p", type=int, default=4)
parser.add_argument("-q", type=int, default=6)
parser.add_argument("-s", type=int, default=-1)
parser.add_argument("-g", action="store_true")
parser.add_argument("-t", action="store_true")
parser.add_argument("-b", type=int, default=0)
args = parser.parse_args()

ceed = Ceed(args.ceed)
dim = args.d
mesh_degree = sol_degree = max(args.m, args.p)
num_qpts = args.q
prob_size = args.s if args.s >= 0 else (16 * 16 * dim * dim if args.t else 256 * 1024)

if not args.t:
    print("Selected options: [command line option] : <current value>")
    print(f"  Ceed specification     [-c] : {args.ceed}")
    print(f"  Mesh dimension         [-d] : {dim}")
    print(f"  Mesh degree            [-m] : {mesh_degree}")
    print(f"  Solution degree        [-p] : {sol_degree}")
    print(f"  Num. 1D quadrature pts [-q] : {num_qpts}")
    print(f"  Approx. # unknowns     [-s] : {prob_size}")
    print(f"  QFunction source       [-g] : {'gallery' if args.g else 'header'}\n")

mesh_basis = ceed.BasisTensorH1Lagrange(dim, dim, mesh_degree + 1, num_qpts, QMODE_GAUSS)
sol_basis = ceed.BasisTensorH1Lagrange(dim, 1, sol_degree + 1, num_qpts, QMODE_GAUSS)

num_xyz = np.zeros(3, dtype=np.int32)
lib.GetCartesianMeshSize(dim, sol_degree, prob_size, num_xyz.ctypes.data_as(lib.CeedIntP))

if not args.t:
    print("Mesh size: nx =", num_xyz[0], end="")
    if dim > 1:
        print(", ny =", num_xyz[1], end="")
    if dim > 2:
        print(", nz =", num_xyz[2], end="")
    print()

size_ptr = np.zeros(1, dtype=np.int32)

mesh_restr = ElemRestriction.__new__(ElemRestriction)
lib.BuildCartesianRestriction(ceed._pointer[0], dim, num_xyz.ctypes.data, mesh_degree, dim,
                              size_ptr.ctypes.data_as(lib.CeedIntP), num_qpts, mesh_restr._pointer,
                              lib.CeedElemRestrictionP(0))
mesh_restr._ceed = ceed
mesh_size = size_ptr[0]

q_data_restr = ElemRestriction.__new__(ElemRestriction)
lib.BuildCartesianRestriction(ceed._pointer[0], dim, num_xyz.ctypes.data, sol_degree, dim * (dim + 1) // 2,
                              size_ptr.ctypes.data_as(lib.CeedIntP), num_qpts,
                              lib.CeedElemRestrictionP(0), q_data_restr._pointer)
q_data_restr._ceed = ceed

sol_restr = ElemRestriction.__new__(ElemRestriction)
lib.BuildCartesianRestriction(ceed._pointer[0], dim, num_xyz.ctypes.data, sol_degree, 1,
                              size_ptr.ctypes.data_as(lib.CeedIntP), num_qpts,
                              sol_restr._pointer, lib.CeedElemRestrictionP(0))
sol_restr._ceed = ceed
sol_size = size_ptr[0]

mesh_coords = ceed.Vector(mesh_size)
lib.SetCartesianMeshCoords(dim, num_xyz.ctypes.data, mesh_degree, mesh_coords._pointer[0])
exact_surface_area = lib.TransformMeshCoords(dim, mesh_size, mesh_coords._pointer[0])

ctx = ceed.QFunctionContext()
ctx_data = np.array([dim, dim], dtype=np.int32).view(np.float64)
ctx.set_data(ctx_data)

if args.g:
    qf_build = ceed.QFunctionByName(f"Poisson{dim}DBuild")
else:
    qf_build = ceed.QFunction(1, lib.build_diff, "build_diff")
    qf_build.add_input("dx", dim * dim, CEED_EVAL_GRAD)
    qf_build.add_input("weights", 1, CEED_EVAL_WEIGHT)
    qf_build.add_output("qdata", dim * (dim + 1) // 2, CEED_EVAL_NONE)
    qf_build.set_context(ctx)

op_build = ceed.Operator(qf_build)
op_build.set_field("dx", mesh_restr, mesh_basis, CEED_VECTOR_ACTIVE)
op_build.set_field("weights", None, mesh_basis, CEED_VECTOR_NONE)
op_build.set_field("qdata", q_data_restr, None, CEED_VECTOR_ACTIVE)

q_data = ceed.Vector(q_data_restr.get_l_layout().prod())
op_build.apply(mesh_coords, q_data)

if args.g:
    qf_apply = ceed.QFunctionByName(f"Poisson{dim}DApply")
else:
    qf_apply = ceed.QFunction(1, lib.apply_diff, "apply_diff")
    qf_apply.add_input("du", dim, CEED_EVAL_GRAD)
    qf_apply.add_input("qdata", dim * (dim + 1) // 2, CEED_EVAL_NONE)
    qf_apply.add_output("dv", dim, CEED_EVAL_GRAD)
    qf_apply.set_context(ctx)

op_apply = ceed.Operator(qf_apply)
op_apply.set_field("du", sol_restr, sol_basis, CEED_VECTOR_ACTIVE)
op_apply.set_field("qdata", q_data_restr, None, q_data)
op_apply.set_field("dv", sol_restr, sol_basis, CEED_VECTOR_ACTIVE)

u = ceed.Vector(sol_size)
v = ceed.Vector(sol_size)

x_array = mesh_coords.get_array_read()
u_array = u.get_array_write()
for i in range(sol_size):
    u_array[i] = sum(x_array[i + j * sol_size] for j in range(dim))
u.restore_array()
mesh_coords.restore_array_read()

op_apply.apply(u, v)

if args.b > 0:
    if not args.t:
        print(f" Executing {args.b} benchmarking runs...")
    for _ in range(args.b):
        op_apply.apply(u, v)

v_array = v.get_array_read()
surface_area = np.sum(np.abs(v_array))
v.restore_array_read()

if not args.t:
    print(" done.")
    print("Exact mesh surface area    : % .14g" % exact_surface_area)
    print("Computed mesh surface area : % .14g" % surface_area)
    print("Surface area error         : % .14g" % (surface_area - exact_surface_area))
else:
    tol = 10000. * np.finfo(np.float64).eps if dim == 1 else 1E-1
    if fabs(surface_area - exact_surface_area) > tol:
        print("Surface area error         : % .14g" % (surface_area - exact_surface_area))
