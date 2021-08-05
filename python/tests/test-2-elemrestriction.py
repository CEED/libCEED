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
# Test Ceed ElemRestriction functionality

import os
import libceed
import numpy as np
import check

# -------------------------------------------------------------------------------
# Test creation, use, and destruction of an element restriction
# -------------------------------------------------------------------------------


def test_200(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    x = ceed.Vector(num_elem + 1)
    a = np.arange(10, 10 + num_elem + 1,
                  dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    x.set_array(a, cmode=libceed.USE_POINTER)

    ind = np.zeros(2 * num_elem, dtype="int32")
    for i in range(num_elem):
        ind[2 * i + 0] = i
        ind[2 * i + 1] = i + 1
    r = ceed.ElemRestriction(num_elem, 2, 1, 1, num_elem + 1, ind,
                             cmode=libceed.USE_POINTER)

    y = ceed.Vector(2 * num_elem)
    y.set_value(0)

    r.apply(x, y)

    with y.array_read() as y_array:
        for i in range(2 * num_elem):
            assert 10 + (i + 1) // 2 == y_array[i]

# -------------------------------------------------------------------------------
# Test creation, use, and destruction of a strided element restriction
# -------------------------------------------------------------------------------


def test_201(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    x = ceed.Vector(2 * num_elem)
    a = np.arange(10, 10 + 2 * num_elem,
                  dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    x.set_array(a, cmode=libceed.USE_POINTER)

    strides = np.array([1, 2, 2], dtype="int32")
    r = ceed.StridedElemRestriction(num_elem, 2, 1, 2 * num_elem, strides)

    y = ceed.Vector(2 * num_elem)
    y.set_value(0)

    r.apply(x, y)

    with y.array_read() as y_array:
        for i in range(2 * num_elem):
            assert 10 + i == y_array[i]

# -------------------------------------------------------------------------------
# Test creation and destruction of a blocked element restriction
# -------------------------------------------------------------------------------


def test_202(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 8
    elem_size = 2
    num_blk = 2
    blk_size = 5

    x = ceed.Vector(num_elem + 1)
    a = np.arange(10, 10 + num_elem + 1,
                  dtype=libceed.scalar_types[libceed.lib.CEED_SCALAR_TYPE])
    x.set_array(a, cmode=libceed.COPY_VALUES)

    ind = np.zeros(2 * num_elem, dtype="int32")
    for i in range(num_elem):
        ind[2 * i + 0] = i
        ind[2 * i + 1] = i + 1
    r = ceed.BlockedElemRestriction(num_elem, elem_size, blk_size, 1, 1,
                                    num_elem + 1, ind, cmode=libceed.COPY_VALUES)

    y = ceed.Vector(num_blk * blk_size * elem_size)
    y.set_value(0)

    # NoTranspose
    r.apply(x, y)
    layout = r.get_layout()
    with y.array_read() as y_array:
        for i in range(elem_size):
            for j in range(1):
                for k in range(num_elem):
                    block = int(k / blk_size)
                    elem = k % blk_size
                    indx = (i * blk_size + elem) * \
                        layout[0] + j * blk_size * layout[1] + \
                        block * blk_size * layout[2]
                assert y_array[indx] == a[ind[k * elem_size + i]]

    x.set_value(0)
    r.T.apply(y, x)
    with x.array_read() as x_array:
        for i in range(num_elem + 1):
            assert x_array[i] == (10 + i) * (2.0 if i >
                                             0 and i < num_elem else 1.0)

# -------------------------------------------------------------------------------
# Test creation, use, and destruction of a blocked element restriction
# -------------------------------------------------------------------------------


def test_208(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 8
    elem_size = 2
    num_blk = 2
    blk_size = 5

    x = ceed.Vector(num_elem + 1)
    a = np.arange(10, 10 + num_elem + 1, dtype=libceed.scalar_types[
        libceed.lib.CEED_SCALAR_TYPE])
    x.set_array(a, cmode=libceed.COPY_VALUES)

    ind = np.zeros(2 * num_elem, dtype="int32")
    for i in range(num_elem):
        ind[2 * i + 0] = i
        ind[2 * i + 1] = i + 1
    r = ceed.BlockedElemRestriction(num_elem, elem_size, blk_size, 1, 1,
                                    num_elem + 1, ind, cmode=libceed.COPY_VALUES)

    y = ceed.Vector(blk_size * elem_size)
    y.set_value(0)

    # NoTranspose
    r.apply_block(1, x, y)
    layout = r.get_layout()
    with y.array_read() as y_array:
        for i in range(elem_size):
            for j in range(1):
                for k in range(blk_size, num_elem):
                    block = int(k / blk_size)
                    elem = k % blk_size
                    indx = (i * blk_size + elem) * layout[0] + j * blk_size * \
                        layout[1] + block * blk_size * \
                        layout[2] - blk_size * elem_size
                assert y_array[indx] == a[ind[k * elem_size + i]]

    x.set_value(0)
    r.T.apply_block(1, y, x)
    with x.array_read() as x_array:
        for i in range(blk_size, num_elem + 1):
            assert x_array[i] == (10 + i) * (2.0 if i >
                                             blk_size and i < num_elem else 1.0)

# -------------------------------------------------------------------------------
# Test getting the multiplicity of the indices in an element restriction
# -------------------------------------------------------------------------------


def test_209(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    ind = np.zeros(4 * num_elem, dtype="int32")
    for i in range(num_elem):
        ind[4 * i + 0] = i * 3 + 0
        ind[4 * i + 1] = i * 3 + 1
        ind[4 * i + 2] = i * 3 + 2
        ind[4 * i + 3] = i * 3 + 3
    r = ceed.ElemRestriction(num_elem, 4, 1, 1, 3 * num_elem + 1, ind,
                             cmode=libceed.USE_POINTER)

    mult = r.get_multiplicity()

    with mult.array_read() as mult_array:
        for i in range(3 * num_elem + 1):
            val = 1 + (1 if (i > 0 and i < 3 * num_elem and i % 3 == 0) else 0)
            assert val == mult_array[i]

# -------------------------------------------------------------------------------
# Test creation and view of an element restriction
# -------------------------------------------------------------------------------


def test_210(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    ind = np.zeros(2 * num_elem, dtype="int32")
    for i in range(num_elem):
        ind[2 * i + 0] = i + 0
        ind[2 * i + 1] = i + 1
    r = ceed.ElemRestriction(num_elem, 2, 1, 1, num_elem + 1, ind,
                             cmode=libceed.USE_POINTER)

    print(r)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test creation and view of a strided element restriction
# -------------------------------------------------------------------------------


def test_211(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    strides = np.array([1, 2, 2], dtype="int32")
    r = ceed.StridedElemRestriction(num_elem, 2, 1, num_elem + 1, strides)

    print(r)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
# Test creation and view of a blocked strided element restriction
# -------------------------------------------------------------------------------


def test_212(ceed_resource, capsys):
    ceed = libceed.Ceed(ceed_resource)

    num_elem = 3

    strides = np.array([1, 2, 2], dtype="int32")
    r = ceed.BlockedStridedElemRestriction(
        num_elem, 2, 2, 1, num_elem + 1, strides)

    print(r)

    stdout, stderr, ref_stdout = check.output(capsys)
    assert not stderr
    assert stdout == ref_stdout

# -------------------------------------------------------------------------------
