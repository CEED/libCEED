# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
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

import os
from distutils.core import setup, Extension

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
qf_module = Extension("libceed_qfunctions",
                      include_dirs=[os.path.abspath("../../include")],
                      libraries=["ceed"],
                      library_dirs=[os.path.abspath("../../lib")],
                      runtime_library_dirs=[os.path.abspath("../../lib")],
                      sources=["libceed-qfunctions.c"],
                      extra_compile_args=["-O3", "-std=c99",
                                          "-Wno-unused-variable",
                                          "-Wno-unused-function"])

setup(name = "libceed_qfunctions",
      description = "libceed qfunction pointers",
      ext_modules = [qf_module])

# ------------------------------------------------------------------------------
