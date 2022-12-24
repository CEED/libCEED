# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import os
import re
from cffi import FFI
ffibuilder = FFI()

ceed_version_ge = re.compile(r'\s+\(!?CEED_VERSION.*')


def get_ceed_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    prefix = os.path.dirname(here)
    # make install links the installed library in build/lbiceed.so Sadly, it
    # does not seem possible to obtain the bath that libceed_build_ext has/will
    # install to.
    libdir = os.path.join(prefix, "build")
    return prefix, libdir


ceed_dir, ceed_libdir = get_ceed_dirs()

# ------------------------------------------------------------------------------
# Provide C definitions to CFFI
# ------------------------------------------------------------------------------
lines = []
for header_path in ["include/ceed/types.h", "include/ceed/ceed.h"]:
    with open(os.path.abspath(header_path)) as f:
        lines += [line.strip() for line in f if
                  not (line.startswith("#") and not line.startswith("#include")) and
                  not line.startswith("  static") and
                  not line.startswith("  CEED_QFUNCTION_ATTR") and
                  "CeedErrorImpl" not in line and
                  "const char *, ...);" not in line and
                  not line.startswith("CEED_EXTERN const char *const") and
                  not ceed_version_ge.match(line)]
lines = [line.replace("CEED_EXTERN", "extern") for line in lines]

# Find scalar type inclusion line and insert definitions
for line in lines:
    if re.search("ceed-f32.h", line) is not None:
        insert_index = lines.index(line) + 1
        extra_lines = ['typedef float CeedScalar;']
        extra_lines.append('static const int CEED_SCALAR_TYPE;')
        extra_lines.append('static const double CEED_EPSILON;')
    elif re.search("ceed-f64.h", line) is not None:
        insert_index = lines.index(line) + 1
        extra_lines = ['typedef double CeedScalar;']
        extra_lines.append('static const int CEED_SCALAR_TYPE;')
        extra_lines.append('static const double CEED_EPSILON;')
lines[insert_index: insert_index] = extra_lines

# Remove all include statements now that scalar type has been dealt with
lines = [line for line in lines if not line.startswith("#include")]

# Build header from lines
header = '\n'.join(lines)
header = header.split("static inline CeedInt CeedIntPow", 1)[0]
header += '\nextern int CeedVectorGetState(CeedVector, uint64_t*);'
header += '\nextern int CeedElemRestrictionGetELayout(CeedElemRestriction, CeedInt (*layout)[3]);'

# Note: cffi cannot handle vargs
header = re.sub("va_list", "const char *", header)

ffibuilder.cdef(header)

ffibuilder.set_source("_ceed_cffi",
                      """
  #define va_list const char *
  #include <ceed/ceed.h>   // the C header of the library
  #include <ceed/backend.h> // declarations for the backend functions above
  """,
                      include_dirs=[
                          os.path.join(ceed_dir, "include")],  # include path
                      libraries=["ceed"],   # library name, for the linker
                      # library path, for the linker
                      library_dirs=[ceed_libdir],
                      # use libceed.so as installed
                      runtime_library_dirs=['$ORIGIN/libceed/lib']
                      )

# ------------------------------------------------------------------------------
# Builder
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    ffibuilder.compile(verbose=True)

# ------------------------------------------------------------------------------
