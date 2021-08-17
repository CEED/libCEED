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
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------


def version():
    with open(os.path.abspath("ceed.pc.template")) as template:
        ceed_version = [line.split("Version:", 1)[1].strip() for line in template if
                        line.startswith("Version: ")]
    return ceed_version[0]


def requirements():
    with open('requirements.txt') as f:
        return f.readlines()


class libceed_build_ext(build_ext):
    def run(self):
        self.make_libceed_so()
        build_ext.run(self)

    def make_libceed_so(self):
        import subprocess
        subprocess.check_call(['make', '-j', '-B'])
        subprocess.check_call(
            ['make', 'install', 'prefix=' + os.path.join(self.build_lib, 'libceed')])


description = """
libCEED: Code for Efficient Extensible Discretization
=====================================================

libCEED is a lightweight library for expressing and manipulating operators that
arise in high-order element-based discretization of partial differential
equations.  libCEED's representations are much for efficient than assembled
sparse matrices, and can achieve very high performance on modern CPU and GPU
hardware.  This approach is applicable to a broad range of linear and nonlinear
problems, and includes facilities for preconditioning.  libCEED is meant to be
easy to incorporate into existing libraries and applications, and to build new
tools on top of.

libCEED has been developed as part of the DOE Exascale Computing Project
co-design Center for Efficient Exascale Discretizations (CEED).
"""

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: POSIX
Programming Language :: C
Programming Language :: C++
Programming Language :: Fortran
Programming Language :: Python
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries
"""

setup(name="libceed",
      version=version(),
      description="libCEED: Code for Efficient Extensible Discretization",
      long_description=description,
      long_description_content_type='text/x-rst',
      classifiers=classifiers.split("\n")[1:-1],
      keywords=["libCEED"],
      platforms=["POSIX"],
      license="BSD 2",
      license_file='LICENSE',
      url="https://libceed.readthedocs.io",
      download_url="https://github.com/CEED/libCEED/releases",
      project_urls={
          "Bug Tracker": "https://github.com/CEED/libCEED/issues",
          "Documentation": "https://libceed.readthedocs.io",
          "Source Code": "https://github.com/CEED/libCEED",
      },
      author="libCEED Team",
      author_email="ceed-users@llnl.gov",

      install_requires=requirements(),
      packages=["libceed"],
      package_dir={"libceed": "python"},
      include_package_data=True,

      setup_requires=["cffi"],
      cffi_modules=["python/build_ceed_cffi.py:ffibuilder"],
      cmdclass={'build_ext': libceed_build_ext},

      extras_require={
          'cuda': ['numba']
      },
      ext_modules=cythonize(Extension('ceed_dlpack', ['python/ceed_dlpack.pyx'],
                            include_dirs=[os.getcwd() + '/include/',
                                          os.getcwd() + '/include/ceed/']))
      )

# ------------------------------------------------------------------------------
