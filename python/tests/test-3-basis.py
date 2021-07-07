# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

# @file
# Test Ceed Basis functionality

import os
import math
import libceed
import numpy as np
import buildmats as bm
import check

TOL = libceed.lib.CEED_EPSILON * 256

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

    stdout, stderr, ref_stdout = check.output(capsys)
# TODO: fix this for float or double
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test QR factorization
# -------------------------------------------------------------------------------


def test_301(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    qr = np.array([1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0],
                  dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    tau = np.empty(3, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

    qr, tau = ceed.qr_factorization(qr, tau, 4, 3)

    for i in range(len(qr)):
        if qr[i] <= TOL and qr[i] >= -TOL:
            qr[i] = 0
        print("%12.8f" % qr[i])

    for i in range(len(tau)):
        if tau[i] <= TOL and tau[i] >= -TOL:
            tau[i] = 0
        print("%12.8f" % tau[i])

    stdout, stderr, ref_stdout = check.output(capsys)
# TODO: fix this for float or double
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test Symmetric Schur Decomposition
# -------------------------------------------------------------------------------


def test_304(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    A = np.array([0.19996678, 0.0745459, -0.07448852, 0.0332866,
                  0.0745459, 1., 0.16666509, -0.07448852,
                  -0.07448852, 0.16666509, 1., 0.0745459,
                  0.0332866, -0.07448852, 0.0745459, 0.19996678],
                 dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

    lam = ceed.symmetric_schur_decomposition(A, 4)

    print("Q: ")
    for i in range(4):
        for j in range(4):
            if A[j + 4 * i] <= TOL and A[j + 4 * i] >= -TOL:
                A[j + 4 * i] = 0
            print("%12.8f" % A[j + 4 * i])

    print("lambda: ")
    for i in range(4):
        if lam[i] <= TOL and lam[i] >= -TOL:
            lam[i] = 0
        print("%12.8f" % lam[i])

    stdout, stderr, ref_stdout = check.output(capsys)
# TODO: fix this for float or double
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test Simultaneous Diagonalization
# -------------------------------------------------------------------------------


def test_305(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    M = np.array([0.19996678, 0.0745459, -0.07448852, 0.0332866,
                  0.0745459, 1., 0.16666509, -0.07448852,
                  -0.07448852, 0.16666509, 1., 0.0745459,
                  0.0332866, -0.07448852, 0.0745459, 0.19996678],
                 dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    K = np.array([3.03344425, -3.41501767, 0.49824435, -0.11667092,
                  -3.41501767, 5.83354662, -2.9167733, 0.49824435,
                  0.49824435, -2.9167733, 5.83354662, -3.41501767,
                  -0.11667092, 0.49824435, -3.41501767, 3.03344425],
                 dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

    x, lam = ceed.simultaneous_diagonalization(K, M, 4)

    print("x: ")
    for i in range(4):
        for j in range(4):
            if x[j + 4 * i] <= TOL and x[j + 4 * i] >= -TOL:
                x[j + 4 * i] = 0
            print("%12.8f" % x[j + 4 * i])

    print("lambda: ")
    for i in range(4):
        if lam[i] <= TOL and lam[i] >= -TOL:
            lam[i] = 0
        print("%12.8f" % lam[i])

    stdout, stderr, ref_stdout = check.output(capsys)
# TODO: fix this for float or double
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
        x = np.empty(Xdim * dim,
                     dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
        uq = np.empty(
            Qdim, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

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
                xx = np.empty(
                    dim, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
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
                xx = np.empty(
                    dim, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
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
        x = np.empty(Xdim * dim,
                     dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
        u = np.empty(
            Pdim, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

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
                xx = np.empty(
                    dim, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
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

    in_array = np.empty(
        P, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qref = np.empty(
        dim * Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

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
                   0., 0.5, 0.5, 1.], dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    in_array = np.empty(
        P, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qref = np.empty(
        dim * Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

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
                   1. / 3., 0.6], dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    xr = np.array([0., 0.5, 1., 0., 0.5, 0., 0., 0.,
                   0., 0.5, 0.5, 1.], dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    in_array = np.empty(
        P, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qref = np.empty(
        dim * Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        Q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])

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
