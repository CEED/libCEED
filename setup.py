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
# testbed platforms, in support of the nation"s exascale computing imperative.
# pylint: disable=no-name-in-module,import-error,unused-variable
import os
from setuptools import setup

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
def version():
  with open(os.path.abspath("ceed.pc.template")) as template:
    ceed_version = [line.split("Version:", 1)[1].strip() for line in template if
                    line.startswith("Version: ")]
  return ceed_version[0]

description = """
libCEED: the Code for Efficient Extensible Discretization API Library
=====================================================================

This low-level API library provides the efficient high-order discretization
methods developed by the ECP co-design Center for Efficient Exascale
Discretizations (CEED). While our focus is on high-order finite elements, the
approach is mostly algebraic and thus applicable to other discretizations in
factored form, as explained in the API documentation.

One of the challenges with high-order methods is that a global sparse matrix is
no longer a good representation of a high-order linear operator, both with
respect to the FLOPs needed for its evaluation, as well as the memory transfer
needed for a matvec.  Thus, high-order methods require a new "format" that still
represents a linear (or more generally non-linear) operator, but not through a
sparse matrix.

libCEED is to provides such a format, as well as supporting implementations and
data structures, that enable efficient operator evaluation on a variety of
computational device types (CPUs, GPUs, etc.). This new operator description is
algebraic and easy to incorporate in a wide variety of applications, without
significant refactoring of their own discretization infrastructure.
"""

classifiers = """
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: POSIX
Programming Language :: C
Programming Language :: C++
Programming Language :: CUDA
Programming Language :: Fortran
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

setup(name="libceed",
      version=version(),
      description="libceed python bindings",
      long_description="\n".join(description),
      classifiers= classifiers.split("\n")[1:-1],
      keywords=["libCEED"],
      platforms=["POSIX"],
      license="BSD 2",

      url="https://github.com/CEED/libCEED",

      author="libCEED Team",
      author_email="ceed-users@llnl.gov",

      requires=["numpy"],
      packages=["libceed"],
      package_dir={"libceed": "python"},

      setup_requires=["cffi"],
      cffi_modules=["python/build_ceed_cffi.py:ffibuilder"],
)

# ------------------------------------------------------------------------------
