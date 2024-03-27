# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import pytest

# -------------------------------------------------------------------------------
# Add --ceed command line argument
# -------------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--ceed", action="store", default='/cpu/self/ref/blocked')


@pytest.fixture(scope='session')
def ceed_resource(request):
    ceed_resource = request.config.option.ceed

    return ceed_resource

# -------------------------------------------------------------------------------
