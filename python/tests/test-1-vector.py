# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

# @file
# Test Ceed Vector functionality

import os
import libceed
import numpy as np
import check

TOL = libceed.EPSILON * 256

# -------------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------------


def check_values(ceed, x, value):
    with x.array_read() as b:
        for i in range(len(b)):
            assert b[i] == value

# -------------------------------------------------------------------------------
# Test creation, setting, reading, restoring, and destroying of a vector
# -------------------------------------------------------------------------------


def test_100(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    with x.array_read() as b:
        for i in range(n):
            assert b[i] == 10 + i

# -------------------------------------------------------------------------------
# Test setValue
# -------------------------------------------------------------------------------


def test_101(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)
    n = 10
    x = ceed.Vector(n)
    value = 1
    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    with x.array() as b:
        for i in range(len(b)):
            assert b[i] == 10 + i

    x.set_value(3.0)
    check_values(ceed, x, 3.0)
    del x

    x = ceed.Vector(n)
    # Set value before setting or getting the array
    x.set_value(5.0)
    check_values(ceed, x, 5.0)

# -------------------------------------------------------------------------------
# Test getArrayRead state counter
# -------------------------------------------------------------------------------


def test_102(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)
    x.set_value(0)

    # Two read accesses should not generate an error
    a = x.get_array_read()
    b = x.get_array_read()

    x.restore_array_read()
    x.restore_array_read()

# -------------------------------------------------------------------------------
# Test setting one vector from array of another vector
# -------------------------------------------------------------------------------


def test_103(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10

    x = ceed.Vector(n)
    y = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    with x.array() as x_array:
        y.set_array(x_array, cmode=libceed.USE_POINTER)

    with y.array_read() as y_array:
        for i in range(n):
            assert y_array[i] == 10 + i

# -------------------------------------------------------------------------------
# Test getArray to modify array
# -------------------------------------------------------------------------------


def test_104(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10

    x = ceed.Vector(n)
    a = np.zeros(n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    with x.array() as b:
        b[3] = -3.14

    if libceed.lib.CEED_SCALAR_TYPE == libceed.SCALAR_FP32:
        assert a[3] == np.float32(-3.14)
    else:
        assert a[3] == -3.14

# -------------------------------------------------------------------------------
# Test creation, setting, reading, restoring, and destroying of a vector using
#   CEED_MEM_DEVICE
# -------------------------------------------------------------------------------


def test_105(ceed_resource):
    # Skip test for non-GPU backend
    if 'gpu' in ceed_resource:
        ceed = libceed.Ceed(ceed_resource)

        n = 10
        x = ceed.Vector(n)
        y = ceed.Vector(n)

        a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
        x.set_array(a, cmode=libceed.USE_POINTER)

        arr = x.get_array_read(memtype=libceed.MEM_DEVICE)
        y.set_array(arr, memtype=libceed.MEM_DEVICE)
        x.restore_array_read()

        with y.array_read() as b:
            for i in range(n):
                assert b[i] == 10 + i

# -------------------------------------------------------------------------------
# Test view
# -------------------------------------------------------------------------------


def test_107(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    print(x)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test norms
# -------------------------------------------------------------------------------


def test_108(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)

    a = np.arange(0, n, dtype=ceed.scalar_type())
    for i in range(n):
        if (i % 2 == 0):
            a[i] *= -1
    x.set_array(a, cmode=libceed.USE_POINTER)

    norm = x.norm(normtype=libceed.NORM_1)

    assert abs(norm - 45.) < TOL

    norm = x.norm()

    assert abs(norm - np.sqrt(285.)) < TOL

    norm = x.norm(normtype=libceed.NORM_MAX)

    assert abs(norm - 9.) < TOL

# -------------------------------------------------------------------------------
# Test vector copy
# -------------------------------------------------------------------------------


def test_109(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10

    x = ceed.Vector(n)
    y = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)

    with x.array() as x_array:
        y.copy(x_array)

    with y.array_read() as y_array:
        for i in range(n):
            assert y_array[i] == x_array[i]

# -------------------------------------------------------------------------------
# Test taking the reciprocal of a vector
# -------------------------------------------------------------------------------


def test_119(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.USE_POINTER)
    x.reciprocal()

    with x.array_read() as b:
        for i in range(n):
            assert abs(b[i] - 1. / (10 + i)) < TOL

# -------------------------------------------------------------------------------
# Test AXPY
# -------------------------------------------------------------------------------


def test_121(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)
    y = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.COPY_VALUES)
    y.set_array(a, cmode=libceed.COPY_VALUES)

    y.axpy(-0.5, x)
    with y.array() as b:
        assert np.allclose(.5 * a, b)

# -------------------------------------------------------------------------------
# Test pointwise multiplication
# -------------------------------------------------------------------------------


def test_122(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    w = ceed.Vector(n)
    x = ceed.Vector(n)
    y = ceed.Vector(n)

    a = np.arange(0, n, dtype=ceed.scalar_type())
    w.set_array(a, cmode=libceed.COPY_VALUES)
    x.set_array(a, cmode=libceed.COPY_VALUES)
    y.set_array(a, cmode=libceed.COPY_VALUES)

    w.pointwise_mult(x, y)
    with w.array() as b:
        for i in range(len(b)):
            assert abs(b[i] - i * i) < 1e-14

    w.pointwise_mult(w, y)
    with w.array() as b:
        for i in range(len(b)):
            assert abs(b[i] - i * i * i) < 1e-14

    w.pointwise_mult(x, w)
    with w.array() as b:
        for i in range(len(b)):
            assert abs(b[i] - i * i * i * i) < 1e-14

    y.pointwise_mult(y, y)
    with y.array() as b:
        for i in range(len(b)):
            assert abs(b[i] - i * i) < 1e-14

# -------------------------------------------------------------------------------
# Test Scale
# -------------------------------------------------------------------------------


def test_123(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.COPY_VALUES)

    x.scale(-0.5)
    with x.array() as b:
        assert np.allclose(-.5 * a, b)

# -------------------------------------------------------------------------------
# Test getArrayWrite to modify array
# -------------------------------------------------------------------------------


def test_124(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    n = 10

    x = ceed.Vector(n)

    with x.array_write() as a:
        for i in range(len(a)):
            a[i] = 3 * i

    with x.array_read() as a:
        for i in range(len(a)):
            assert a[i] == 3 * i

# -------------------------------------------------------------------------------
# Test AXPBY
# -------------------------------------------------------------------------------


def test_125(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    n = 10
    x = ceed.Vector(n)
    y = ceed.Vector(n)

    a = np.arange(10, 10 + n, dtype=ceed.scalar_type())
    x.set_array(a, cmode=libceed.COPY_VALUES)
    y.set_array(a, cmode=libceed.COPY_VALUES)

    y.axpby(-0.5, 1.0, x)
    with y.array() as b:
        assert np.allclose(.5 * a, b)

# -------------------------------------------------------------------------------
# Test modification of reshaped array
# -------------------------------------------------------------------------------


def test_199(ceed_resource):
    """Modification of reshaped array"""
    ceed = libceed.Ceed(ceed_resource)

    vec = ceed.Vector(12)
    vec.set_value(0.0)
    with vec.array(4, 3) as x:
        x[...] = np.eye(4, 3)

    with vec.array_read(3, 4) as x:
        assert np.all(x == np.eye(4, 3).reshape(3, 4))

# -------------------------------------------------------------------------------
