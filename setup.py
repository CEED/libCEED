# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
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

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------


def version():
    with open(os.path.abspath("ceed.pc.template")) as template:
        ceed_version = [line.split("Version:", 1)[1].strip() for line in template if
                        line.startswith("Version: ")]
    return ceed_version[0]


class libceed_build_ext(build_ext):
    def run(self):
        prefix = os.path.join(self.build_lib, 'libceed')
        self.make_libceed_so(prefix)
        build_ext.run(self)

    def make_libceed_so(self, prefix):
        import subprocess
        if hasattr(os, 'sched_getaffinity'):
            # number of available logical cores
            nproc = len(os.sched_getaffinity(0))
        else:
            nproc = os.cpu_count()
        subprocess.check_call([
            'make',
            '-j{}'.format(nproc),
            '--always-make',
            'install',
            'prefix=' + prefix,
            'FC=',  # Don't try to find Fortran (unused library build/install)
        ])


setup(
    version=version(),
    cffi_modules=["python/build_ceed_cffi.py:ffibuilder"],
    cmdclass={'build_ext': libceed_build_ext},
)


# ------------------------------------------------------------------------------
