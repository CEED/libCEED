# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

# @file
# Test Ceed QFunction functionality

import os
import libceed
import numpy as np
import check

TOL = libceed.EPSILON * 256

# -------------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------------


def load_qfs_so():
    from sysconfig import get_config_var
    import ctypes

    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs_so = os.path.join(
        file_dir,
        "libceed_qfunctions" + get_config_var("EXT_SUFFIX"))

    # Load library
    return ctypes.cdll.LoadLibrary(qfs_so)

# -------------------------------------------------------------------------------
# Test creation, evaluation, and destruction for qfunction
# -------------------------------------------------------------------------------


def test_400(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    q = 8

    w_array = np.zeros(q, dtype=ceed.scalar_type())
    u_array = np.zeros(q, dtype=ceed.scalar_type())
    v_true = np.zeros(q, dtype=ceed.scalar_type())
    for i in range(q):
        x = 2. * i / (q - 1) - 1
        w_array[i] = 1 - x * x
        u_array[i] = 2 + 3 * x + 5 * x * x
        v_true[i] = w_array[i] * u_array[i]

    dx = ceed.Vector(q)
    dx.set_value(1)
    w = ceed.Vector(q)
    w.set_array(w_array, cmode=libceed.USE_POINTER)
    u = ceed.Vector(q)
    u.set_array(u_array, cmode=libceed.USE_POINTER)
    v = ceed.Vector(q)
    v.set_value(0)
    qdata = ceed.Vector(q)
    qdata.set_value(0)

    inputs = [dx, w]
    outputs = [qdata]
    qf_setup.apply(q, inputs, outputs)

    inputs = [qdata, u]
    outputs = [v]
    qf_mass.apply(q, inputs, outputs)

    with v.array_read() as v_array:
        for i in range(q):
            assert v_array[i] == v_true[i]

# -------------------------------------------------------------------------------
# Test creation, evaluation, and destruction for qfunction
# -------------------------------------------------------------------------------


def test_401(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    ctx_data = np.array([1., 2., 3., 4., 5.], dtype=ceed.scalar_type())
    ctx = ceed.QFunctionContext()
    ctx.set_data(ctx_data)
    qf_mass.set_context(ctx)

    q = 8

    w_array = np.zeros(q, dtype=ceed.scalar_type())
    u_array = np.zeros(q, dtype=ceed.scalar_type())
    v_true = np.zeros(q, dtype=ceed.scalar_type())
    for i in range(q):
        x = 2. * i / (q - 1) - 1
        w_array[i] = 1 - x * x
        u_array[i] = 2 + 3 * x + 5 * x * x
        v_true[i] = 5 * w_array[i] * u_array[i]

    dx = ceed.Vector(q)
    dx.set_value(1)
    w = ceed.Vector(q)
    w.set_array(w_array, cmode=libceed.USE_POINTER)
    u = ceed.Vector(q)
    u.set_array(u_array, cmode=libceed.USE_POINTER)
    v = ceed.Vector(q)
    v.set_value(0)
    qdata = ceed.Vector(q)
    qdata.set_value(0)

    inputs = [dx, w]
    outputs = [qdata]
    qf_setup.apply(q, inputs, outputs)

    inputs = [qdata, u]
    outputs = [v]
    qf_mass.apply(q, inputs, outputs)

    with v.array_read() as v_array:
        for i in range(q):
            assert abs(v_array[i] - v_true[i]) < TOL

# -------------------------------------------------------------------------------
# Test viewing of qfunction
# -------------------------------------------------------------------------------


def test_402(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    qfs = load_qfs_so()

    qf_setup = ceed.QFunction(1, qfs.setup_mass,
                              os.path.join(file_dir, "test-qfunctions.h:setup_mass"))
    qf_setup.add_input("w", 1, libceed.EVAL_WEIGHT)
    qf_setup.add_input("dx", 1, libceed.EVAL_GRAD)
    qf_setup.add_output("qdata", 1, libceed.EVAL_NONE)

    qf_mass = ceed.QFunction(1, qfs.apply_mass,
                             os.path.join(file_dir, "test-qfunctions.h:apply_mass"))
    qf_mass.add_input("qdata", 1, libceed.EVAL_NONE)
    qf_mass.add_input("u", 1, libceed.EVAL_INTERP)
    qf_mass.add_output("v", 1, libceed.EVAL_INTERP)

    print(qf_setup)
    print(qf_mass)

    if libceed.lib.CEED_SCALAR_TYPE == libceed.SCALAR_FP64:
        ctx_data = np.array([1., 2., 3., 4., 5.], dtype="float64")
    # Make ctx twice as long in fp32, so size will be the same as fp64 output
    else:
        ctx_data = np.array([1., 2., 3., 4., 5., 1., 2., 3., 4., 5.],
                            dtype="float32")
    ctx = ceed.QFunctionContext()
    ctx.set_data(ctx_data)
    print(ctx)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test creation, evaluation, and destruction for qfunction by name
# -------------------------------------------------------------------------------


def test_410(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    qf_setup = ceed.QFunctionByName("Mass1DBuild")
    qf_mass = ceed.QFunctionByName("MassApply")

    q = 8

    j_array = np.zeros(q, dtype=ceed.scalar_type())
    w_array = np.zeros(q, dtype=ceed.scalar_type())
    u_array = np.zeros(q, dtype=ceed.scalar_type())
    v_true = np.zeros(q, dtype=ceed.scalar_type())
    for i in range(q):
        x = 2. * i / (q - 1) - 1
        j_array[i] = 1
        w_array[i] = 1 - x * x
        u_array[i] = 2 + 3 * x + 5 * x * x
        v_true[i] = w_array[i] * u_array[i]

    j = ceed.Vector(q)
    j.set_array(j_array, cmode=libceed.USE_POINTER)
    w = ceed.Vector(q)
    w.set_array(w_array, cmode=libceed.USE_POINTER)
    u = ceed.Vector(q)
    u.set_array(u_array, cmode=libceed.USE_POINTER)
    v = ceed.Vector(q)
    v.set_value(0)
    qdata = ceed.Vector(q)
    qdata.set_value(0)

    inputs = [j, w]
    outputs = [qdata]
    qf_setup.apply(q, inputs, outputs)

    inputs = [w, u]
    outputs = [v]
    qf_mass.apply(q, inputs, outputs)

    with v.array_read() as v_array:
        for i in range(q):
            assert v_array[i] == v_true[i]

# -------------------------------------------------------------------------------
# Test creation, evaluation, and destruction of identity qfunction
# -------------------------------------------------------------------------------


def test_411(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    qf = ceed.IdentityQFunction(1, libceed.EVAL_INTERP, libceed.EVAL_INTERP)

    q = 8

    u_array = np.zeros(q, dtype=ceed.scalar_type())
    for i in range(q):
        u_array[i] = i * i

    u = ceed.Vector(q)
    u.set_array(u_array, cmode=libceed.USE_POINTER)
    v = ceed.Vector(q)
    v.set_value(0)

    inputs = [u]
    outputs = [v]
    qf.apply(q, inputs, outputs)

    with v.array_read() as v_array:
        for i in range(q):
            assert v_array[i] == i * i

# -------------------------------------------------------------------------------
# Test creation, evaluation, and destruction of identity qfunction with size>1
# -------------------------------------------------------------------------------


def test_412(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    size = 3
    qf = ceed.IdentityQFunction(size, libceed.EVAL_INTERP, libceed.EVAL_INTERP)

    q = 8

    u_array = np.zeros(q * size, dtype=ceed.scalar_type())
    for i in range(q * size):
        u_array[i] = i * i

    u = ceed.Vector(q * size)
    u.set_array(u_array, cmode=libceed.USE_POINTER)
    v = ceed.Vector(q * size)
    v.set_value(0)

    inputs = [u]
    outputs = [v]
    qf.apply(q, inputs, outputs)

    with v.array_read() as v_array:
        for i in range(q * size):
            assert v_array[i] == i * i

# -------------------------------------------------------------------------------
# Test viewing of qfunction by name
# -------------------------------------------------------------------------------


def test_413(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    qf_setup = ceed.QFunctionByName("Mass1DBuild")
    qf_mass = ceed.QFunctionByName("MassApply")

    print(qf_setup)
    print(qf_mass)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
