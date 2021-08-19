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
# Test Ceed Operator functionality

import os
import libceed
import numpy as np
import check
import buildmats as bm

TOL = libceed.EPSILON * 256

# -------------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------------


def load_qfs_so():
    from distutils.sysconfig import get_config_var
    import ctypes

    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs_so = os.path.join(
        file_dir,
        "libceed_qfunctions" + get_config_var("EXT_SUFFIX"))

    # Load library
    return ctypes.cdll.LoadLibrary(qfs_so)

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
# -------------------------------------------------------------------------------


def test_500(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(nx)
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u = ceed.Vector(nu)
    v = ceed.Vector(nu)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = i * (p - 1) + j
    ru = ceed.ElemRestriction(nelem, p, 1, 1, nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    u.set_value(0)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        for i in range(q):
            assert abs(v_array[i]) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
# -------------------------------------------------------------------------------


def test_501(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u = ceed.Vector(nu)
    v = ceed.Vector(nu)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = i * (p - 1) + j
    ru = ceed.ElemRestriction(nelem, p, 1, 1, nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    u.set_value(1.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(nu):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with multiple
#   components
# -------------------------------------------------------------------------------


def test_502(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u = ceed.Vector(2 * nu)
    v = ceed.Vector(2 * nu)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = 2 * (i * (p - 1) + j)
    ru = ceed.ElemRestriction(nelem, p, 2, 1, 2 * nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 2, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass_two,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass_two"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 2, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 2, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    with u.array() as u_array:
        for i in range(nu):
            u_array[2 * i] = 1.
            u_array[2 * i + 1] = 2.
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        total_1 = 0.0
        total_2 = 0.0
        for i in range(nu):
            total_1 = total_1 + v_array[2 * i]
            total_2 = total_2 + v_array[2 * i + 1]
        assert abs(total_1 - 1.0) < TOL
        assert abs(total_2 - 2.0) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with passive
#   inputs and outputs
# -------------------------------------------------------------------------------


def test_503(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u = ceed.Vector(nu)
    v = ceed.Vector(nu)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = i * (p - 1) + j
    ru = ceed.ElemRestriction(nelem, p, 1, 1, nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, u)
    op_mass.set_field("v", ru, bu, v)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    u.set_value(1)
    op_mass.apply(libceed.VECTOR_NONE, libceed.VECTOR_NONE)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(nu):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

# -------------------------------------------------------------------------------
# Test viewing of mass matrix operator
# -------------------------------------------------------------------------------


def test_504(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    qdata = ceed.Vector(nelem * q)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = i * (p - 1) + j
    ru = ceed.ElemRestriction(nelem, p, 1, 1, nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # View
    print(op_setup)
    print(op_mass)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test CeedOperatorApplyAdd
# -------------------------------------------------------------------------------


def test_505(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p = 5
    q = 8
    nx = nelem + 1
    nu = nelem * (p - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u = ceed.Vector(nu)
    v = ceed.Vector(nu)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem):
        for j in range(p):
            indu[p * i + j] = i * (p - 1) + j
    ru = ceed.ElemRestriction(nelem, p, 1, 1, nu, indu,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(1, 1, p, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix with v = 0
    u.set_value(1.)
    v.set_value(0.)
    op_mass.apply_add(u, v)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(nu):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

    # Apply mass matrix with v = 0
    v.set_value(1.)
    op_mass.apply_add(u, v)

    # Check
    with v.array_read() as v_array:
        total = -nu
        for i in range(nu):
            total = total + v_array[i]
        assert abs(total - 1.0) < 10. * TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
# -------------------------------------------------------------------------------


def test_510(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 12
    dim = 2
    p = 6
    q = 4
    nx, ny = 3, 2
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts = nelem * q

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs)
    for i in range(ndofs):
        x_array[i] = (1. / (nx * 2)) * (i % (nx * 2 + 1))
        x_array[i + ndofs] = (1. / (ny * 2)) * (i / (nx * 2 + 1))
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nqpts)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # Restrictions
    indx = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx[i * 2 * p + 0] = 2 + offset
        indx[i * 2 * p + 1] = 9 + offset
        indx[i * 2 * p + 2] = 16 + offset
        indx[i * 2 * p + 3] = 1 + offset
        indx[i * 2 * p + 4] = 8 + offset
        indx[i * 2 * p + 5] = 0 + offset

        indx[i * 2 * p + 6] = 14 + offset
        indx[i * 2 * p + 7] = 7 + offset
        indx[i * 2 * p + 8] = 0 + offset
        indx[i * 2 * p + 9] = 15 + offset
        indx[i * 2 * p + 10] = 8 + offset
        indx[i * 2 * p + 11] = 16 + offset

    rx = ceed.ElemRestriction(nelem, p, dim, ndofs, dim * ndofs, indx,
                              cmode=libceed.USE_POINTER)

    ru = ceed.ElemRestriction(nelem, p, 1, 1, ndofs, indx,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, nqpts, strides)

    # Bases
    qref = np.empty(
        dim * q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx = ceed.BasisH1(libceed.TRIANGLE, dim, p, q, interp, grad, qref, qweight)
    bu = ceed.BasisH1(libceed.TRIANGLE, 1, p, q, interp, grad, qref, qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass_2d,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    u.set_value(0.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        for i in range(ndofs):
            assert abs(v_array[i]) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator
# -------------------------------------------------------------------------------


def test_511(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 12
    dim = 2
    p = 6
    q = 4
    nx, ny = 3, 2
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts = nelem * q

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs,
                       dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(ndofs):
        x_array[i] = (1. / (nx * 2)) * (i % (nx * 2 + 1))
        x_array[i + ndofs] = (1. / (ny * 2)) * (i / (nx * 2 + 1))
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nqpts)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # Restrictions
    indx = np.zeros(nelem * p, dtype="int32")
    for i in range(nelem // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx[i * 2 * p + 0] = 2 + offset
        indx[i * 2 * p + 1] = 9 + offset
        indx[i * 2 * p + 2] = 16 + offset
        indx[i * 2 * p + 3] = 1 + offset
        indx[i * 2 * p + 4] = 8 + offset
        indx[i * 2 * p + 5] = 0 + offset

        indx[i * 2 * p + 6] = 14 + offset
        indx[i * 2 * p + 7] = 7 + offset
        indx[i * 2 * p + 8] = 0 + offset
        indx[i * 2 * p + 9] = 15 + offset
        indx[i * 2 * p + 10] = 8 + offset
        indx[i * 2 * p + 11] = 16 + offset

    rx = ceed.ElemRestriction(nelem, p, dim, ndofs, dim * ndofs, indx,
                              cmode=libceed.USE_POINTER)

    ru = ceed.ElemRestriction(nelem, p, 1, 1, ndofs, indx,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q, 1, nqpts, strides)

    # Bases
    qref = np.empty(
        dim * q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx = ceed.BasisH1(libceed.TRIANGLE, dim, p, q, interp, grad, qref, qweight)
    bu = ceed.BasisH1(libceed.TRIANGLE, 1, p, q, interp, grad, qref, qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass_2d,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass = ceed.Operator(qf_mass)
    op_mass.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass.set_field("u", ru, bu, libceed.VECTOR_ACTIVE)
    op_mass.set_field("v", ru, bu, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Apply mass matrix
    u.set_value(1.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(ndofs):
            total = total + v_array[i]
        assert abs(total - 1.0) < 10. * TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for composite mass matrix operator
# -------------------------------------------------------------------------------


def test_520(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem_tet, p_tet, q_tet = 6, 6, 4
    nelem_hex, p_hex, q_hex = 6, 3, 4
    nx, ny = 3, 3
    dim = 2
    nx_tet, ny_tet, nx_hex = 3, 1, 3
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts_tet, nqpts_hex = nelem_tet * q_tet, nelem_hex * q_hex * q_hex

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs)
    for i in range(ny * 2 + 1):
        for j in range(nx * 2 + 1):
            x_array[i + j * (ny * 2 + 1)] = i / (2 * ny)
            x_array[i + j * (ny * 2 + 1) + ndofs] = j / (2 * nx)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata_hex = ceed.Vector(nqpts_hex)
    qdata_tet = ceed.Vector(nqpts_tet)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # ------------------------- Tet Elements -------------------------

    # Restrictions
    indx_tet = np.zeros(nelem_tet * p_tet, dtype="int32")
    for i in range(nelem_tet // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx_tet[i * 2 * p_tet + 0] = 2 + offset
        indx_tet[i * 2 * p_tet + 1] = 9 + offset
        indx_tet[i * 2 * p_tet + 2] = 16 + offset
        indx_tet[i * 2 * p_tet + 3] = 1 + offset
        indx_tet[i * 2 * p_tet + 4] = 8 + offset
        indx_tet[i * 2 * p_tet + 5] = 0 + offset

        indx_tet[i * 2 * p_tet + 6] = 14 + offset
        indx_tet[i * 2 * p_tet + 7] = 7 + offset
        indx_tet[i * 2 * p_tet + 8] = 0 + offset
        indx_tet[i * 2 * p_tet + 9] = 15 + offset
        indx_tet[i * 2 * p_tet + 10] = 8 + offset
        indx_tet[i * 2 * p_tet + 11] = 16 + offset

    rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, dim, ndofs, dim * ndofs,
                                  indx_tet, cmode=libceed.USE_POINTER)

    ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, 1, 1, ndofs, indx_tet,
                                  cmode=libceed.USE_POINTER)
    strides = np.array([1, q_tet, q_tet], dtype="int32")
    rui_tet = ceed.StridedElemRestriction(nelem_tet, q_tet, 1, nqpts_tet,
                                          strides)

    # Bases
    qref = np.empty(dim * q_tet,
                    dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q_tet, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                          qweight)
    bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                          qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_tet.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_tet = ceed.Operator(qf_setup_tet)
    op_setup_tet.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_tet,
                           libceed.VECTOR_NONE)
    op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
    op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                           qdata_tet)

    op_mass_tet = ceed.Operator(qf_mass_tet)
    op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
    op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
    op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

    # ------------------------- Hex Elements -------------------------

    # Restrictions
    indx_hex = np.zeros(nelem_hex * p_hex * p_hex, dtype="int32")
    for i in range(nelem_hex):
        col = i % nx_hex
        row = i // nx_hex
        offset = (nx_tet * 2 + 1) * (ny_tet * 2) * (1 + row) + col * 2

        for j in range(p_hex):
            for k in range(p_hex):
                indx_hex[p_hex * (p_hex * i + k) + j] = offset + \
                    k * (nx_hex * 2 + 1) + j

    rx_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, dim, ndofs,
                                  dim * ndofs, indx_hex, cmode=libceed.USE_POINTER)

    ru_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, 1, 1, ndofs,
                                  indx_hex, cmode=libceed.USE_POINTER)
    strides = np.array([1, q_hex * q_hex, q_hex * q_hex], dtype="int32")
    rui_hex = ceed.StridedElemRestriction(nelem_hex, q_hex * q_hex, 1,
                                          nqpts_hex, strides)

    # Bases
    bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
    bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

    # QFunctions
    qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_hex.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_hex = ceed.Operator(qf_setup_tet)
    op_setup_hex.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_hex,
                           libceed.VECTOR_NONE)
    op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
    op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                           qdata_hex)

    op_mass_hex = ceed.Operator(qf_mass_hex)
    op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
    op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
    op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

    # ------------------------- Composite Operators -------------------------

    # Setup
    op_setup = ceed.CompositeOperator()
    op_setup.add_sub(op_setup_tet)
    op_setup.add_sub(op_setup_hex)
    op_setup.apply(x, libceed.VECTOR_NONE)

    # Apply mass matrix
    op_mass = ceed.CompositeOperator()
    op_mass.add_sub(op_mass_tet)
    op_mass.add_sub(op_mass_hex)

    u.set_value(0.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        for i in range(ndofs):
            assert abs(v_array[i]) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for composite mass matrix operator
# -------------------------------------------------------------------------------


def test_521(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem_tet, p_tet, q_tet = 6, 6, 4
    nelem_hex, p_hex, q_hex = 6, 3, 4
    nx, ny = 3, 3
    dim = 2
    nx_tet, ny_tet, nx_hex = 3, 1, 3
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts_tet, nqpts_hex = nelem_tet * q_tet, nelem_hex * q_hex * q_hex

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs,
                       dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(ny * 2 + 1):
        for j in range(nx * 2 + 1):
            x_array[i + j * (ny * 2 + 1)] = i / (2 * ny)
            x_array[i + j * (ny * 2 + 1) + ndofs] = j / (2 * nx)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata_hex = ceed.Vector(nqpts_hex)
    qdata_tet = ceed.Vector(nqpts_tet)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # ------------------------- Tet Elements -------------------------

    # Restrictions
    indx_tet = np.zeros(nelem_tet * p_tet, dtype="int32")
    for i in range(nelem_tet // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx_tet[i * 2 * p_tet + 0] = 2 + offset
        indx_tet[i * 2 * p_tet + 1] = 9 + offset
        indx_tet[i * 2 * p_tet + 2] = 16 + offset
        indx_tet[i * 2 * p_tet + 3] = 1 + offset
        indx_tet[i * 2 * p_tet + 4] = 8 + offset
        indx_tet[i * 2 * p_tet + 5] = 0 + offset

        indx_tet[i * 2 * p_tet + 6] = 14 + offset
        indx_tet[i * 2 * p_tet + 7] = 7 + offset
        indx_tet[i * 2 * p_tet + 8] = 0 + offset
        indx_tet[i * 2 * p_tet + 9] = 15 + offset
        indx_tet[i * 2 * p_tet + 10] = 8 + offset
        indx_tet[i * 2 * p_tet + 11] = 16 + offset

    rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, dim, ndofs, dim * ndofs,
                                  indx_tet, cmode=libceed.USE_POINTER)

    ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, 1, 1, ndofs, indx_tet,
                                  cmode=libceed.USE_POINTER)
    strides = np.array([1, q_tet, q_tet], dtype="int32")
    rui_tet = ceed.StridedElemRestriction(nelem_tet, q_tet, 1, nqpts_tet,
                                          strides)

    # Bases
    qref = np.empty(dim * q_tet,
                    dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q_tet, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                          qweight)
    bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                          qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_tet.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_tet = ceed.Operator(qf_setup_tet)
    op_setup_tet.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_tet,
                           libceed.VECTOR_NONE)
    op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
    op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                           qdata_tet)

    op_mass_tet = ceed.Operator(qf_mass_tet)
    op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
    op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
    op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

    # ------------------------- Hex Elements -------------------------

    # Restrictions
    indx_hex = np.zeros(nelem_hex * p_hex * p_hex, dtype="int32")
    for i in range(nelem_hex):
        col = i % nx_hex
        row = i // nx_hex
        offset = (nx_tet * 2 + 1) * (ny_tet * 2) * (1 + row) + col * 2

        for j in range(p_hex):
            for k in range(p_hex):
                indx_hex[p_hex * (p_hex * i + k) + j] = offset + \
                    k * (nx_hex * 2 + 1) + j

    rx_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, dim, ndofs,
                                  dim * ndofs, indx_hex, cmode=libceed.USE_POINTER)

    ru_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, 1, 1, ndofs,
                                  indx_hex, cmode=libceed.USE_POINTER)
    strides = np.array([1, q_hex * q_hex, q_hex * q_hex], dtype="int32")
    rui_hex = ceed.StridedElemRestriction(nelem_hex, q_hex * q_hex, 1,
                                          nqpts_hex, strides)

    # Bases
    bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
    bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

    # QFunctions
    qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_hex.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_hex = ceed.Operator(qf_setup_tet)
    op_setup_hex.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_hex,
                           libceed.VECTOR_NONE)
    op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
    op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                           qdata_hex)

    op_mass_hex = ceed.Operator(qf_mass_hex)
    op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
    op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
    op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

    # ------------------------- Composite Operators -------------------------

    # Setup
    op_setup = ceed.CompositeOperator()
    op_setup.add_sub(op_setup_tet)
    op_setup.add_sub(op_setup_hex)
    op_setup.apply(x, libceed.VECTOR_NONE)

    # Apply mass matrix
    op_mass = ceed.CompositeOperator()
    op_mass.add_sub(op_mass_tet)
    op_mass.add_sub(op_mass_hex)
    u.set_value(1.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(ndofs):
            total = total + v_array[i]
        assert abs(total - 1.0) < 10. * TOL

# -------------------------------------------------------------------------------
# Test viewing of composite mass matrix operator
# -------------------------------------------------------------------------------


def test_523(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    nelem_tet, p_tet, q_tet = 6, 6, 4
    nelem_hex, p_hex, q_hex = 6, 3, 4
    nx, ny = 3, 3
    dim = 2
    nx_tet, ny_tet, nx_hex = 3, 1, 3
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts_tet, nqpts_hex = nelem_tet * q_tet, nelem_hex * q_hex * q_hex

    # Vectors
    qdata_hex = ceed.Vector(nqpts_hex)
    qdata_tet = ceed.Vector(nqpts_tet)

    # ------------------------- Tet Elements -------------------------

    # Restrictions
    indx_tet = np.zeros(nelem_tet * p_tet, dtype="int32")
    for i in range(nelem_tet // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx_tet[i * 2 * p_tet + 0] = 2 + offset
        indx_tet[i * 2 * p_tet + 1] = 9 + offset
        indx_tet[i * 2 * p_tet + 2] = 16 + offset
        indx_tet[i * 2 * p_tet + 3] = 1 + offset
        indx_tet[i * 2 * p_tet + 4] = 8 + offset
        indx_tet[i * 2 * p_tet + 5] = 0 + offset

        indx_tet[i * 2 * p_tet + 6] = 14 + offset
        indx_tet[i * 2 * p_tet + 7] = 7 + offset
        indx_tet[i * 2 * p_tet + 8] = 0 + offset
        indx_tet[i * 2 * p_tet + 9] = 15 + offset
        indx_tet[i * 2 * p_tet + 10] = 8 + offset
        indx_tet[i * 2 * p_tet + 11] = 16 + offset

    rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, dim, ndofs, dim * ndofs,
                                  indx_tet, cmode=libceed.USE_POINTER)

    ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, 1, 1, ndofs, indx_tet,
                                  cmode=libceed.USE_POINTER)
    strides = np.array([1, q_tet, q_tet], dtype="int32")
    rui_tet = ceed.StridedElemRestriction(nelem_tet, q_tet, 1, nqpts_tet,
                                          strides)

    # Bases
    qref = np.empty(dim * q_tet,
                    dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q_tet, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                          qweight)
    bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                          qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_tet.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_tet = ceed.Operator(qf_setup_tet)
    op_setup_tet.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_tet,
                           libceed.VECTOR_NONE)
    op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
    op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                           qdata_tet)

    op_mass_tet = ceed.Operator(qf_mass_tet)
    op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
    op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
    op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

    # ------------------------- Hex Elements -------------------------

    # Restrictions
    indx_hex = np.zeros(nelem_hex * p_hex * p_hex, dtype="int32")
    for i in range(nelem_hex):
        col = i % nx_hex
        row = i // nx_hex
        offset = (nx_tet * 2 + 1) * (ny_tet * 2) * (1 + row) + col * 2

        for j in range(p_hex):
            for k in range(p_hex):
                indx_hex[p_hex * (p_hex * i + k) + j] = offset + \
                    k * (nx_hex * 2 + 1) + j

    rx_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, dim, ndofs,
                                  dim * ndofs, indx_hex,
                                  cmode=libceed.USE_POINTER)

    ru_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, 1, 1, ndofs,
                                  indx_hex, cmode=libceed.USE_POINTER)
    strides = np.array([1, q_hex * q_hex, q_hex * q_hex], dtype="int32")
    rui_hex = ceed.StridedElemRestriction(nelem_hex, q_hex * q_hex, 1,
                                          nqpts_hex, strides)

    # Bases
    bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
    bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

    # QFunctions
    qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_hex.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_hex = ceed.Operator(qf_setup_tet)
    op_setup_hex.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_hex,
                           libceed.VECTOR_NONE)
    op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
    op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                           qdata_hex)

    op_mass_hex = ceed.Operator(qf_mass_hex)
    op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
    op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
    op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

    # ------------------------- Composite Operators -------------------------

    # Setup
    op_setup = ceed.CompositeOperator()
    op_setup.add_sub(op_setup_tet)
    op_setup.add_sub(op_setup_hex)

    # Apply mass matrix
    op_mass = ceed.CompositeOperator()
    op_mass.add_sub(op_mass_tet)
    op_mass.add_sub(op_mass_hex)

    # View
    print(op_setup)
    print(op_mass)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# CeedOperatorApplyAdd for composite operator
# -------------------------------------------------------------------------------


def test_524(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem_tet, p_tet, q_tet = 6, 6, 4
    nelem_hex, p_hex, q_hex = 6, 3, 4
    nx, ny = 3, 3
    dim = 2
    nx_tet, ny_tet, nx_hex = 3, 1, 3
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts_tet, nqpts_hex = nelem_tet * q_tet, nelem_hex * q_hex * q_hex

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs,
                       dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(ny * 2 + 1):
        for j in range(nx * 2 + 1):
            x_array[i + j * (ny * 2 + 1)] = i / (2 * ny)
            x_array[i + j * (ny * 2 + 1) + ndofs] = j / (2 * nx)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata_hex = ceed.Vector(nqpts_hex)
    qdata_tet = ceed.Vector(nqpts_tet)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # ------------------------- Tet Elements -------------------------

    # Restrictions
    indx_tet = np.zeros(nelem_tet * p_tet, dtype="int32")
    for i in range(nelem_tet // 2):
        col = i % nx
        row = i // nx
        offset = col * 2 + row * (nx * 2 + 1) * 2

        indx_tet[i * 2 * p_tet + 0] = 2 + offset
        indx_tet[i * 2 * p_tet + 1] = 9 + offset
        indx_tet[i * 2 * p_tet + 2] = 16 + offset
        indx_tet[i * 2 * p_tet + 3] = 1 + offset
        indx_tet[i * 2 * p_tet + 4] = 8 + offset
        indx_tet[i * 2 * p_tet + 5] = 0 + offset

        indx_tet[i * 2 * p_tet + 6] = 14 + offset
        indx_tet[i * 2 * p_tet + 7] = 7 + offset
        indx_tet[i * 2 * p_tet + 8] = 0 + offset
        indx_tet[i * 2 * p_tet + 9] = 15 + offset
        indx_tet[i * 2 * p_tet + 10] = 8 + offset
        indx_tet[i * 2 * p_tet + 11] = 16 + offset

    rx_tet = ceed.ElemRestriction(nelem_tet, p_tet, dim, ndofs, dim * ndofs,
                                  indx_tet, cmode=libceed.USE_POINTER)

    ru_tet = ceed.ElemRestriction(nelem_tet, p_tet, 1, 1, ndofs, indx_tet,
                                  cmode=libceed.USE_POINTER)
    strides = np.array([1, q_tet, q_tet], dtype="int32")
    rui_tet = ceed.StridedElemRestriction(nelem_tet, q_tet, 1, nqpts_tet,
                                          strides)

    # Bases
    qref = np.empty(dim * q_tet,
                    dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    qweight = np.empty(
        q_tet, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    interp, grad = bm.buildmats(qref, qweight, libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])

    bx_tet = ceed.BasisH1(libceed.TRIANGLE, dim, p_tet, q_hex, interp, grad, qref,
                          qweight)
    bu_tet = ceed.BasisH1(libceed.TRIANGLE, 1, p_tet, q_hex, interp, grad, qref,
                          qweight)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup_tet = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_tet.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_tet.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_tet.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_tet = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_tet.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_tet.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_tet.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_tet = ceed.Operator(qf_setup_tet)
    op_setup_tet.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_tet,
                           libceed.VECTOR_NONE)
    op_setup_tet.set_field("dx", rx_tet, bx_tet, libceed.VECTOR_ACTIVE)
    op_setup_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED,
                           qdata_tet)

    op_mass_tet = ceed.Operator(qf_mass_tet)
    op_mass_tet.set_field("rho", rui_tet, libceed.BASIS_COLLOCATED, qdata_tet)
    op_mass_tet.set_field("u", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)
    op_mass_tet.set_field("v", ru_tet, bu_tet, libceed.VECTOR_ACTIVE)

    # ------------------------- Hex Elements -------------------------

    # Restrictions
    indx_hex = np.zeros(nelem_hex * p_hex * p_hex, dtype="int32")
    for i in range(nelem_hex):
        col = i % nx_hex
        row = i // nx_hex
        offset = (nx_tet * 2 + 1) * (ny_tet * 2) * (1 + row) + col * 2

        for j in range(p_hex):
            for k in range(p_hex):
                indx_hex[p_hex * (p_hex * i + k) + j] = offset + \
                    k * (nx_hex * 2 + 1) + j

    rx_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, dim, ndofs,
                                  dim * ndofs, indx_hex,
                                  cmode=libceed.USE_POINTER)

    ru_hex = ceed.ElemRestriction(nelem_hex, p_hex * p_hex, 1, 1, ndofs,
                                  indx_hex, cmode=libceed.USE_POINTER)
    strides = np.array([1, q_hex * q_hex, q_hex * q_hex], dtype="int32")
    rui_hex = ceed.StridedElemRestriction(nelem_hex, q_hex * q_hex, 1,
                                          nqpts_hex, strides)

    # Bases
    bx_hex = ceed.BasisTensorH1Lagrange(dim, dim, p_hex, q_hex, libceed.GAUSS)
    bu_hex = ceed.BasisTensorH1Lagrange(dim, 1, p_hex, q_hex, libceed.GAUSS)

    # QFunctions
    qf_setup_hex = ceed.QFunction(1, qfs.setup_mass_2d,
                                  os.path.join(file_dir, "test-qfunctions.h:setup_mass_2d"))
    qf_setup_hex.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup_hex.add_input("dx", dim * dim, libceed.EVAL_GRAD)
    qf_setup_hex.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass_hex = ceed.QFunction(1, qfs.apply_mass,
                                 os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass_hex.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass_hex.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass_hex.add_output("v", 1, libceed.EVAL_INTERP)

    # Operators
    op_setup_hex = ceed.Operator(qf_setup_tet)
    op_setup_hex.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx_hex,
                           libceed.VECTOR_NONE)
    op_setup_hex.set_field("dx", rx_hex, bx_hex, libceed.VECTOR_ACTIVE)
    op_setup_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED,
                           qdata_hex)

    op_mass_hex = ceed.Operator(qf_mass_hex)
    op_mass_hex.set_field("rho", rui_hex, libceed.BASIS_COLLOCATED, qdata_hex)
    op_mass_hex.set_field("u", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)
    op_mass_hex.set_field("v", ru_hex, bu_hex, libceed.VECTOR_ACTIVE)

    # ------------------------- Composite Operators -------------------------

    # Setup
    op_setup = ceed.CompositeOperator()
    op_setup.add_sub(op_setup_tet)
    op_setup.add_sub(op_setup_hex)
    op_setup.apply(x, libceed.VECTOR_NONE)

    # Apply mass matrix
    op_mass = ceed.CompositeOperator()
    op_mass.add_sub(op_mass_tet)
    op_mass.add_sub(op_mass_hex)
    u.set_value(1.)
    op_mass.apply(u, v)

    # Check
    with v.array_read() as v_array:
        total = 0.0
        for i in range(ndofs):
            total = total + v_array[i]
        assert abs(total - 1.0) < 10. * TOL

    # ApplyAdd mass matrix
    v.set_value(1.)
    op_mass.apply_add(u, v)

    # Check
    with v.array_read() as v_array:
        total = -ndofs
        for i in range(ndofs):
            total = total + v_array[i]
        assert abs(total - 1.0) < 10. * TOL

# -------------------------------------------------------------------------------
# Test assembly of mass matrix operator diagonal
# -------------------------------------------------------------------------------


def test_533(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 6
    p = 3
    q = 4
    dim = 2
    nx = 3
    ny = 2
    ndofs = (nx * 2 + 1) * (ny * 2 + 1)
    nqpts = nelem * q * q

    # Vectors
    x = ceed.Vector(dim * ndofs)
    x_array = np.zeros(dim * ndofs)
    for i in range(nx * 2 + 1):
        for j in range(ny * 2 + 1):
            x_array[i + j * (nx * 2 + 1) + 0 * ndofs] = i / (2 * nx)
            x_array[i + j * (nx * 2 + 1) + 1 * ndofs] = j / (2 * ny)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nqpts)
    u = ceed.Vector(ndofs)
    v = ceed.Vector(ndofs)

    # Restrictions
    indx = np.zeros(nelem * p * p, dtype="int32")
    for i in range(nelem):
        col = i % nx
        row = i // nx
        offset = col * (p - 1) + row * (nx * 2 + 1) * (p - 1)
        for j in range(p):
            for k in range(p):
                indx[p * (p * i + k) + j] = offset + k * (nx * 2 + 1) + j
    rx = ceed.ElemRestriction(nelem, p * p, dim, ndofs, dim * ndofs,
                              indx, cmode=libceed.USE_POINTER)

    ru = ceed.ElemRestriction(nelem, p * p, 1, 1, ndofs, indx,
                              cmode=libceed.USE_POINTER)
    strides = np.array([1, q * q, q * q], dtype="int32")
    rui = ceed.StridedElemRestriction(nelem, q * q, 1, nqpts, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(dim, dim, p, q, libceed.GAUSS)
    bu = ceed.BasisTensorH1Lagrange(dim, 1, p, q, libceed.GAUSS)

    # QFunctions
    qf_setup = ceed.QFunctionByName("Mass2DBuild")

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with
#   multigrid level, tensor basis and interpolation basis generation
# -------------------------------------------------------------------------------


def test_550(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p_coarse = 3
    p_fine = 5
    q = 8
    ncomp = 2
    nx = nelem + 1
    nu_coarse = nelem * (p_coarse - 1) + 1
    nu_fine = nelem * (p_fine - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u_coarse = ceed.Vector(ncomp * nu_coarse)
    v_coarse = ceed.Vector(ncomp * nu_coarse)
    u_fine = ceed.Vector(ncomp * nu_fine)
    v_fine = ceed.Vector(ncomp * nu_fine)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu_coarse = np.zeros(nelem * p_coarse, dtype="int32")
    for i in range(nelem):
        for j in range(p_coarse):
            indu_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j
    ru_coarse = ceed.ElemRestriction(nelem, p_coarse, ncomp, nu_coarse,
                                     ncomp * nu_coarse, indu_coarse,
                                     cmode=libceed.USE_POINTER)

    indu_fine = np.zeros(nelem * p_fine, dtype="int32")
    for i in range(nelem):
        for j in range(p_fine):
            indu_fine[p_fine * i + j] = i * (p_fine - 1) + j
    ru_fine = ceed.ElemRestriction(nelem, p_fine, ncomp, nu_fine,
                                   ncomp * nu_fine, indu_fine,
                                   cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")

    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu_coarse = ceed.BasisTensorH1Lagrange(1, ncomp, p_coarse, q, libceed.GAUSS)
    bu_fine = ceed.BasisTensorH1Lagrange(1, ncomp, p_fine, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass_two,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass_two"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", ncomp, libceed.EVAL_INTERP)
    qf_mass.add_output("v", ncomp, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass_fine = ceed.Operator(qf_mass)
    op_mass_fine.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass_fine.set_field("u", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)
    op_mass_fine.set_field("v", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Create multigrid level
    p_mult_fine = ceed.Vector(ncomp * nu_fine)
    p_mult_fine.set_value(1.0)
    [op_mass_coarse, op_prolong, op_restrict] = op_mass_fine.multigrid_create(p_mult_fine,
                                                                              ru_coarse,
                                                                              bu_coarse)

    # Apply coarse mass matrix
    u_coarse.set_value(1.0)
    op_mass_coarse.apply(u_coarse, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < 10. * TOL

    # Prolong coarse u
    op_prolong.apply(u_coarse, u_fine)

    # Apply mass matrix
    op_mass_fine.apply(u_fine, v_fine)

    # Check
    with v_fine.array_read() as v_array:
        total = 0.0
        for i in range(nu_fine * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < 10. * TOL

    # Restrict state to coarse grid
    op_restrict.apply(v_fine, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < 10. * TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with
#   multigrid level, tensor basis
# -------------------------------------------------------------------------------


def test_552(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p_coarse = 3
    p_fine = 5
    q = 8
    ncomp = 2
    nx = nelem + 1
    nu_coarse = nelem * (p_coarse - 1) + 1
    nu_fine = nelem * (p_fine - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u_coarse = ceed.Vector(ncomp * nu_coarse)
    v_coarse = ceed.Vector(ncomp * nu_coarse)
    u_fine = ceed.Vector(ncomp * nu_fine)
    v_fine = ceed.Vector(ncomp * nu_fine)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu_coarse = np.zeros(nelem * p_coarse, dtype="int32")
    for i in range(nelem):
        for j in range(p_coarse):
            indu_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j
    ru_coarse = ceed.ElemRestriction(nelem, p_coarse, ncomp, nu_coarse,
                                     ncomp * nu_coarse, indu_coarse,
                                     cmode=libceed.USE_POINTER)

    indu_fine = np.zeros(nelem * p_fine, dtype="int32")
    for i in range(nelem):
        for j in range(p_fine):
            indu_fine[p_fine * i + j] = i * (p_fine - 1) + j
    ru_fine = ceed.ElemRestriction(nelem, p_fine, ncomp, nu_fine,
                                   ncomp * nu_fine, indu_fine,
                                   cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")

    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu_coarse = ceed.BasisTensorH1Lagrange(1, ncomp, p_coarse, q, libceed.GAUSS)
    bu_fine = ceed.BasisTensorH1Lagrange(1, ncomp, p_fine, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("weights", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("rho", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass_two,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass_two"))
    qf_mass.add_input("rho", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", ncomp, libceed.EVAL_INTERP)
    qf_mass.add_output("v", ncomp, libceed.EVAL_INTERP)

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("rho", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass_fine = ceed.Operator(qf_mass)
    op_mass_fine.set_field("rho", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass_fine.set_field("u", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)
    op_mass_fine.set_field("v", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Create multigrid level
    p_mult_fine = ceed.Vector(ncomp * nu_fine)
    p_mult_fine.set_value(1.0)
    b_c_to_f = ceed.BasisTensorH1Lagrange(
        1, ncomp, p_coarse, p_fine, libceed.GAUSS_LOBATTO)
    interp_C_to_F = b_c_to_f.get_interp_1d()
    [op_mass_coarse, op_prolong, op_restrict] = op_mass_fine.multigrid_create_tensor_h1(p_mult_fine,
                                                                                        ru_coarse, bu_coarse, interp_C_to_F)

    # Apply coarse mass matrix
    u_coarse.set_value(1.0)
    op_mass_coarse.apply(u_coarse, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < TOL

    # Prolong coarse u
    op_prolong.apply(u_coarse, u_fine)

    # Apply mass matrix
    op_mass_fine.apply(u_fine, v_fine)

    # Check
    with v_fine.array_read() as v_array:
        total = 0.0
        for i in range(nu_fine * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < TOL

    # Restrict state to coarse grid
    op_restrict.apply(v_fine, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 2.0) < TOL

# -------------------------------------------------------------------------------
# Test creation, action, and destruction for mass matrix operator with
#   multigrid level, non-tensor basis
# -------------------------------------------------------------------------------


def test_553(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    nelem = 15
    p_coarse = 3
    p_fine = 5
    q = 8
    ncomp = 1
    nx = nelem + 1
    nu_coarse = nelem * (p_coarse - 1) + 1
    nu_fine = nelem * (p_fine - 1) + 1

    # Vectors
    x = ceed.Vector(nx)
    x_array = np.zeros(
        nx, dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    for i in range(nx):
        x_array[i] = i / (nx - 1.0)
    x.set_array(x_array, cmode=libceed.USE_POINTER)

    qdata = ceed.Vector(nelem * q)
    u_coarse = ceed.Vector(ncomp * nu_coarse)
    v_coarse = ceed.Vector(ncomp * nu_coarse)
    u_fine = ceed.Vector(ncomp * nu_fine)
    v_fine = ceed.Vector(ncomp * nu_fine)

    # Restrictions
    indx = np.zeros(nx * 2, dtype="int32")
    for i in range(nx):
        indx[2 * i + 0] = i
        indx[2 * i + 1] = i + 1
    rx = ceed.ElemRestriction(nelem, 2, 1, 1, nx, indx,
                              cmode=libceed.USE_POINTER)

    indu_coarse = np.zeros(nelem * p_coarse, dtype="int32")
    for i in range(nelem):
        for j in range(p_coarse):
            indu_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j
    ru_coarse = ceed.ElemRestriction(nelem, p_coarse, ncomp, nu_coarse,
                                     ncomp * nu_coarse, indu_coarse,
                                     cmode=libceed.USE_POINTER)

    indu_fine = np.zeros(nelem * p_fine, dtype="int32")
    for i in range(nelem):
        for j in range(p_fine):
            indu_fine[p_fine * i + j] = i * (p_fine - 1) + j
    ru_fine = ceed.ElemRestriction(nelem, p_fine, ncomp, nu_fine,
                                   ncomp * nu_fine, indu_fine,
                                   cmode=libceed.USE_POINTER)
    strides = np.array([1, q, q], dtype="int32")

    rui = ceed.StridedElemRestriction(nelem, q, 1, q * nelem, strides)

    # Bases
    bx = ceed.BasisTensorH1Lagrange(1, 1, 2, q, libceed.GAUSS)
    bu_coarse = ceed.BasisTensorH1Lagrange(1, ncomp, p_coarse, q, libceed.GAUSS)
    bu_fine = ceed.BasisTensorH1Lagrange(1, ncomp, p_fine, q, libceed.GAUSS)

    # QFunctions
    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunctionByName("Mass1DBuild")
    qf_mass = ceed.QFunctionByName("MassApply")

    # Operators
    op_setup = ceed.Operator(qf_setup)
    op_setup.set_field("weights", libceed.ELEMRESTRICTION_NONE, bx,
                       libceed.VECTOR_NONE)
    op_setup.set_field("dx", rx, bx, libceed.VECTOR_ACTIVE)
    op_setup.set_field("qdata", rui, libceed.BASIS_COLLOCATED,
                       libceed.VECTOR_ACTIVE)

    op_mass_fine = ceed.Operator(qf_mass)
    op_mass_fine.set_field("qdata", rui, libceed.BASIS_COLLOCATED, qdata)
    op_mass_fine.set_field("u", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)
    op_mass_fine.set_field("v", ru_fine, bu_fine, libceed.VECTOR_ACTIVE)

    # Setup
    op_setup.apply(x, qdata)

    # Create multigrid level
    p_mult_fine = ceed.Vector(ncomp * nu_fine)
    p_mult_fine.set_value(1.0)
    b_c_to_f = ceed.BasisTensorH1Lagrange(
        1, ncomp, p_coarse, p_fine, libceed.GAUSS_LOBATTO)
    interp_C_to_F = b_c_to_f.get_interp_1d()
    [op_mass_coarse, op_prolong, op_restrict] = op_mass_fine.multigrid_create_h1(p_mult_fine,
                                                                                 ru_coarse, bu_coarse, interp_C_to_F)

    # Apply coarse mass matrix
    u_coarse.set_value(1.0)
    op_mass_coarse.apply(u_coarse, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

    # Prolong coarse u
    op_prolong.apply(u_coarse, u_fine)

    # Apply mass matrix
    op_mass_fine.apply(u_fine, v_fine)

    # Check
    with v_fine.array_read() as v_array:
        total = 0.0
        for i in range(nu_fine * ncomp):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

    # Restrict state to coarse grid
    op_restrict.apply(v_fine, v_coarse)

    # Check
    with v_coarse.array_read() as v_array:
        total = 0.0
        for i in range(nu_coarse * ncomp):
            total = total + v_array[i]
        assert abs(total - 1.0) < TOL

# -------------------------------------------------------------------------------
