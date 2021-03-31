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

import os
import re
from cffi import FFI
ffibuilder = FFI()

ceed_version_ge = re.compile(r'\s+\(!?CEED_VERSION.*')

# ------------------------------------------------------------------------------
# Provide C definitions to CFFI
# ------------------------------------------------------------------------------
with open(os.path.abspath("include/ceed/ceed.h")) as f:
    lines = [line.strip() for line in f if
             not line.startswith("#") and
             not line.startswith("  static") and
             "CeedErrorImpl" not in line and
             "const char *, ...);" not in line and
             not line.startswith("CEED_EXTERN const char *const") and
             not ceed_version_ge.match(line)]
    lines = [line.replace("CEED_EXTERN", "extern") for line in lines]
    header = '\n'.join(lines)
    header = header.split("static inline CeedInt CeedIntPow", 1)[0]
    header += '\nextern int CeedVectorGetState(CeedVector, uint64_t*);'
    # Note: cffi cannot handle vargs
    header = re.sub("va_list", "const char *", header)
ffibuilder.cdef(header)

ffibuilder.set_source("_ceed_cffi",
                      """
  #define va_list const char *
  #include <ceed.h>   // the C header of the library
  """,
                      include_dirs=[
                          os.path.abspath("include")],  # include path
                      libraries=["ceed"],   # library name, for the linker
                      library_dirs=['./lib'],  # library path, for the linker
                      # use libceed.so as installed
                      runtime_library_dirs=['$ORIGIN/libceed/lib']
                      )

# ------------------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

# ------------------------------------------------------------------------------
