# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

# @file
# Test Ceed functionality

import libceed
import pytest

# -------------------------------------------------------------------------------
# Test creation and destruction of a Ceed object
# -------------------------------------------------------------------------------


def test_000(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

# -------------------------------------------------------------------------------
# Test return of Ceed backend prefered memory type
# -------------------------------------------------------------------------------


def test_001(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    memtype = ceed.get_preferred_memtype()

    assert memtype != "error"

# -------------------------------------------------------------------------------
# Test return of a CEED object full resource name
# -------------------------------------------------------------------------------


def test_002(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    resource = ceed.get_resource()

    assert resource == ceed_resource

# -------------------------------------------------------------------------------
# Test viewing of a CEED object
# -------------------------------------------------------------------------------


def test_003(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    print(ceed)

# -------------------------------------------------------------------------------
# Test CEED object error handling
# -------------------------------------------------------------------------------


def test_005(ceed_resource):
    ceed = libceed.Ceed(ceed_resource)

    vec = ceed.Vector(5)
    vec.set_value(0.0)
    array1 = vec.get_array()

    exception_raised = False
    try:
        array2 = vec.get_array()
    except BaseException:
        exception_raised = True

    assert exception_raised

    vec.restore_array()

# -------------------------------------------------------------------------------
