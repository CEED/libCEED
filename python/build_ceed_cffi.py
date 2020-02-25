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
from cffi import FFI
ffibuilder = FFI()

# ------------------------------------------------------------------------------
# Provide C definitions to CFFI
# ------------------------------------------------------------------------------
with open(os.path.abspath("include/ceed.h")) as f:
    lines = [line.strip() for line in f if
               not line.startswith("#") and
               not line.startswith("  static") and
               "CeedErrorImpl" not in line and
               "const char *, ...);" not in line and
               "///" not in line and
               not line.startswith("CEED_EXTERN const char *const")]
    lines = [line.replace("CEED_EXTERN", "extern") for line in lines]
    header = ''.join(lines)
    header = header.split("static inline CeedInt CeedIntPow", 1)[0]
ffibuilder.cdef(header)

# ------------------------------------------------------------------------------
# Set source of libCEED header file
# ------------------------------------------------------------------------------
ceed_dir = os.getenv("CEED_DIR", None)
if ceed_dir:
  ceed_lib_dirs = [os.path.abspath("lib"), os.path.join(ceed_dir, "lib")]
else:
  ceed_lib_dirs = [os.path.abspath("lib")]

ffibuilder.set_source("_ceed_cffi",
  """
  #include <ceed.h>   // the C header of the library
  """,
  include_dirs = [os.path.abspath("include")], # include path
  libraries = ["ceed"],   # library name, for the linker
  library_dirs = [os.path.abspath("lib")], # library path, for the linker
  runtime_library_dirs = ceed_lib_dirs # library path, at runtime
)

# ------------------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

# ------------------------------------------------------------------------------
