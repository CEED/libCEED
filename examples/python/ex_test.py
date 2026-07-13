#!/usr/bin/env python3
# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import pytest
from argparse import Namespace
import ex1_volume
import ex2_surface
import ex3_volume
import bpsraw

# -------------------------------------------------------------------------------


def test_101(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=1,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex1_volume.example_1(args)

# -------------------------------------------------------------------------------


def test_101g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=1,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex1_volume.example_1(args)

# -------------------------------------------------------------------------------


def test_102(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=2,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex1_volume.example_1(args)

# -------------------------------------------------------------------------------


def test_102g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=2,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex1_volume.example_1(args)

# -------------------------------------------------------------------------------


def test_103(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=3,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex1_volume.example_1(args)

# -------------------------------------------------------------------------------


def test_103g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=3,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex1_volume.example_1(args)


# -------------------------------------------------------------------------------
def test_201(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=1,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_201g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=1,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_202(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=2,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_202g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=2,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_203(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=3,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_203g(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=3,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=True,
    )
    ex2_surface.example_2(args)

# -------------------------------------------------------------------------------


def test_301(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=1,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex3_volume.example_3(args)

# -------------------------------------------------------------------------------


def test_302(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=2,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex3_volume.example_3(args)

# -------------------------------------------------------------------------------


def test_303(ceed_resource):
    args = Namespace(
        ceed=ceed_resource,
        dim=3,
        mesh_degree=4,
        solution_degree=4,
        quadrature_points=6,
        problem_size=-1,
        test=True,
        quiet=True,
        gallery=False,
    )
    ex3_volume.example_3(args)

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
