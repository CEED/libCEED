# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

# @file
# Test Ceed Basis functionality

import os
import math
import libceed
import numpy as np
import buildmats as bm
import check

TOL = libceed.EPSILON * 256

# -------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------


def eval(dim, x):
    result, center = 1, 0.1
    for d in range(dim):
        result *= math.tanh(x[d] - center)
        center += 0.1
    return result


def feval(x1, x2):
    return x1 * x1 + x2 * x2 + x1 * x2 + 1


def dfeval(x1, x2):
    return 2 * x1 + x2

# -------------------------------------------------------------------------------
# Test creation and distruction of a H1Lagrange basis
# -------------------------------------------------------------------------------


def test_300(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    b = ceed.BasisTensorH1Lagrange(1, 1, 4, 4, libceed.GAUSS_LOBATTO)
    print(b)
    del b

    b = ceed.BasisTensorH1Lagrange(1, 1, 4, 4, libceed.GAUSS)
    print(b)
    del b

    # Only run this test in double precision
    if libceed.lib.CEED_SCALAR_TYPE == libceed.SCALAR_FP64:
        stdout, stderr, ref_stdout = check.output(capsys)
        assert not stderr
        assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test GetNumNodes and GetNumQuadraturePoints for basis
# -------------------------------------------------------------------------------


def test_306(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    b = ceed.BasisTensorH1Lagrange(3, 1, 4, 5, libceed.GAUSS_LOBATTO)

    p = b.get_num_nodes()
    q = b.get_num_quadrature_points()

    assert p == 64
    assert q == 125

# -------------------------------------------------------------------------------
# Test interpolation in multiple dimensions
# -------------------------------------------------------------------------------


def test_313(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    for dim in range(1, 4):
        Q = 10
        Qdim = Q**dim
        Xdim = 2**dim
        x = np.empty(Xdim * dim, dtype=ceed.scalar_type())
        uq = np.empty(Qdim, dtype=ceed.scalar_type())

        for d in range(dim):
            for i in range(Xdim):
                x[d * Xdim + i] = 1 if (i %
                                        (2**(dim - d))) // (2**(dim - d - 1)) else -1

        X = ceed.Vector(Xdim * dim)
        X.set_array(x, cmode=libceed.USE_POINTER)
        Xq = ceed.Vector(Qdim * dim)
        Xq.set_value(0)
        U = ceed.Vector(Qdim)
        U.set_value(0)
        Uq = ceed.Vector(Qdim)

        bxl = ceed.BasisTensorH1Lagrange(dim, dim, 2, Q, libceed.GAUSS_LOBATTO)
        bul = ceed.BasisTensorH1Lagrange(dim, 1, Q, Q, libceed.GAUSS_LOBATTO)

        bxl.apply(1, libceed.EVAL_INTERP, X, Xq)

        with Xq.array_read() as xq:
            for i in range(Qdim):
                xx = np.empty(dim, dtype=ceed.scalar_type())
                for d in range(dim):
                    xx[d] = xq[d * Qdim + i]
                uq[i] = eval(dim, xx)

        Uq.set_array(uq, cmode=libceed.USE_POINTER)

        # This operation is the identity because the quadrature is collocated
        bul.T.apply(1, libceed.EVAL_INTERP, Uq, U)

        bxg = ceed.BasisTensorH1Lagrange(dim, dim, 2, Q, libceed.GAUSS)
        bug = ceed.BasisTensorH1Lagrange(dim, 1, Q, Q, libceed.GAUSS)

        bxg.apply(1, libceed.EVAL_INTERP, X, Xq)
        bug.apply(1, libceed.EVAL_INTERP, U, Uq)

        with Xq.array_read() as xq, Uq.array_read() as u:
            for i in range(Qdim):
                xx = np.empty(dim, dtype=ceed.scalar_type())
                for d in range(dim):
                    xx[d] = xq[d * Qdim + i]
                fx = eval(dim, xx)
                assert math.fabs(u[i] - fx) < 1E-4

# -------------------------------------------------------------------------------
# Test grad in multiple dimensions
# -------------------------------------------------------------------------------


def test_314(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    for dim in range(1, 4):
        P, Q = 8, 10
        Pdim = P**dim
        Qdim = Q**dim
        Xdim = 2**dim
        sum1 = sum2 = 0
        x = np.empty(Xdim * dim, dtype=ceed.scalar_type())
        u = np.empty(Pdim, dtype=ceed.scalar_type())

        for d in range(dim):
            for i in range(Xdim):
                x[d * Xdim + i] = 1 if (i %
                                        (2**(dim - d))) // (2**(dim - d - 1)) else -1

        X = ceed.Vector(Xdim * dim)
        X.set_array(x, cmode=libceed.USE_POINTER)
        Xq = ceed.Vector(Pdim * dim)
        Xq.set_value(0)
        U = ceed.Vector(Pdim)
        Uq = ceed.Vector(Qdim * dim)
        Uq.set_value(0)
        Ones = ceed.Vector(Qdim * dim)
        Ones.set_value(1)
        Gtposeones = ceed.Vector(Pdim)
        Gtposeones.set_value(0)

        # Get function values at quadrature points
        bxl = ceed.BasisTensorH1Lagrange(dim, dim, 2, P, libceed.GAUSS_LOBATTO)
        bxl.apply(1, libceed.EVAL_INTERP, X, Xq)

        with Xq.array_read() as xq:
            for i in range(Pdim):
                xx = np.empty(dim, dtype=ceed.scalar_type())
                for d in range(dim):
                    xx[d] = xq[d * Pdim + i]
                u[i] = eval(dim, xx)

        U.set_array(u, cmode=libceed.USE_POINTER)

        # Calculate G u at quadrature points, G' * 1 at dofs
        bug = ceed.BasisTensorH1Lagrange(dim, 1, P, Q, libceed.GAUSS)
        bug.apply(1, libceed.EVAL_GRAD, U, Uq)
        bug.T.apply(1, libceed.EVAL_GRAD, Ones, Gtposeones)

        # Check if 1' * G * u = u' * (G' * 1)
        with Gtposeones.array_read() as gtposeones, Uq.array_read() as uq:
            for i in range(Pdim):
                sum1 += gtposeones[i] * u[i]
            for i in range(dim * Qdim):
                sum2 += uq[i]

        assert math.fabs(sum1 - sum2) < 10. * TOL

# -------------------------------------------------------------------------------
# Test creation and destruction of a 2D Simplex non-tensor H1 basis
# -------------------------------------------------------------------------------


def test_320(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    P, Q, dim = 6, 4, 2

    in_array = np.empty(P, dtype=ceed.scalar_type())
    qref = np.empty(dim * Q, dtype=ceed.scalar_type())
    qweight = np.empty(Q, dtype=ceed.scalar_type())

    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    b = ceed.BasisH1(libceed.TRIANGLE, 1, P, Q, interp, grad, qref, qweight)

    print(b)
    del b

# -------------------------------------------------------------------------------
# Test integration with a 2D Simplex non-tensor H1 basis
# -------------------------------------------------------------------------------


def test_322(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    P, Q, dim = 6, 4, 2

    xr = np.array([0., 0.5, 1., 0., 0.5, 0., 0., 0.,
                   0., 0.5, 0.5, 1.], dtype=ceed.scalar_type())
    in_array = np.empty(P, dtype=ceed.scalar_type())
    qref = np.empty(dim * Q, dtype=ceed.scalar_type())
    qweight = np.empty(Q, dtype=ceed.scalar_type())

    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    b = ceed.BasisH1(libceed.TRIANGLE, 1, P, Q, interp, grad, qref, qweight)

    # Interpolate function to quadrature points
    for i in range(P):
        in_array[i] = feval(xr[0 * P + i], xr[1 * P + i])

    in_vec = ceed.Vector(P)
    in_vec.set_array(in_array, cmode=libceed.USE_POINTER)
    out_vec = ceed.Vector(Q)
    out_vec.set_value(0)
    weights_vec = ceed.Vector(Q)
    weights_vec.set_value(0)

    b.apply(1, libceed.EVAL_INTERP, in_vec, out_vec)
    b.apply(1, libceed.EVAL_WEIGHT, libceed.VECTOR_NONE, weights_vec)

    # Check values at quadrature points
    with out_vec.array_read() as out_array, weights_vec.array_read() as weights_array:
        sum = 0
        for i in range(Q):
            sum += out_array[i] * weights_array[i]
        assert math.fabs(sum - 17. / 24.) < 10. * TOL

# -------------------------------------------------------------------------------
# Test grad with a 2D Simplex non-tensor H1 basis
# -------------------------------------------------------------------------------


def test_323(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    P, Q, dim = 6, 4, 2

    xq = np.array([0.2, 0.6, 1. / 3., 0.2, 0.2, 0.2,
                   1. / 3., 0.6], dtype=ceed.scalar_type())
    xr = np.array([0., 0.5, 1., 0., 0.5, 0., 0., 0.,
                   0., 0.5, 0.5, 1.], dtype=ceed.scalar_type())
    in_array = np.empty(P, dtype=ceed.scalar_type())
    qref = np.empty(dim * Q, dtype=ceed.scalar_type())
    qweight = np.empty(Q, dtype=ceed.scalar_type())

    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    b = ceed.BasisH1(libceed.TRIANGLE, 1, P, Q, interp, grad, qref, qweight)

    # Interpolate function to quadrature points
    for i in range(P):
        in_array[i] = feval(xr[0 * P + i], xr[1 * P + i])

    in_vec = ceed.Vector(P)
    in_vec.set_array(in_array, cmode=libceed.USE_POINTER)
    out_vec = ceed.Vector(Q * dim)
    out_vec.set_value(0)

    b.apply(1, libceed.EVAL_GRAD, in_vec, out_vec)

    # Check values at quadrature points
    with out_vec.array_read() as out_array:
        for i in range(Q):
            value = dfeval(xq[0 * Q + i], xq[1 * Q + i])
            assert math.fabs(out_array[0 * Q + i] - value) < 10. * TOL

            value = dfeval(xq[1 * Q + i], xq[0 * Q + i])
            assert math.fabs(out_array[1 * Q + i] - value) < 10. * TOL

# -------------------------------------------------------------------------------
