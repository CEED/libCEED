#!/usr/bin/env python3
# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import pytest
from argparse import Namespace

# The BP examples use PETSc, unlike ex1-ex3, so skip them where it is unavailable
pytest.importorskip("petsc4py")
import bpsraw  # noqa: E402

# -------------------------------------------------------------------------------


def test_401(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        problem='bp1',
        degree=1,
        q_extra=None,
        local=1000,
        test=True,
        benchmark=False,
        write_solution=False,
        ksp_max_it_clip=[15, 15],
    )
    assert bpsraw.example_bps(args) == 0

# -------------------------------------------------------------------------------


def test_402(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        problem='bp1',
        degree=2,
        q_extra=None,
        local=1000,
        test=True,
        benchmark=False,
        write_solution=False,
        ksp_max_it_clip=[15, 15],
    )
    assert bpsraw.example_bps(args) == 0

# -------------------------------------------------------------------------------


def test_403(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        problem='bp1',
        degree=3,
        q_extra=None,
        local=1000,
        test=True,
        benchmark=False,
        write_solution=False,
        ksp_max_it_clip=[15, 15],
    )
    assert bpsraw.example_bps(args) == 0

# -------------------------------------------------------------------------------
