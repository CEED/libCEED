# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import os
from setuptools import setup, Extension
import libceed
CEED_DIR = os.path.dirname(libceed.__file__)

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
qf_module = Extension("libceed_qfunctions",
                      include_dirs=[os.path.join(CEED_DIR, 'include')],
                      sources=["libceed-qfunctions.c"],
                      extra_compile_args=["-O3", "-std=c99",
                                          "-Wno-unused-variable",
                                          "-Wno-unused-function"])

setup(name="libceed_qfunctions",
      description="libceed qfunction pointers",
      ext_modules=[qf_module])

# ------------------------------------------------------------------------------
