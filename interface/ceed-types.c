// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>

const char *const CeedErrorTypesShifted[] = {
  [CEED_ERROR_SUCCESS - CEED_ERROR_UNSUPPORTED] = "success",
  [CEED_ERROR_MINOR - CEED_ERROR_UNSUPPORTED] = "generic minor error",
  [CEED_ERROR_DIMENSION - CEED_ERROR_UNSUPPORTED] = "dimension mismatch",
  [CEED_ERROR_INCOMPLETE - CEED_ERROR_UNSUPPORTED] = "object setup incomplete",
  [CEED_ERROR_INCOMPATIBLE - CEED_ERROR_UNSUPPORTED] = "incompatible arguments",
  [CEED_ERROR_ACCESS - CEED_ERROR_UNSUPPORTED] = "access lock in use",
  [CEED_ERROR_MAJOR - CEED_ERROR_UNSUPPORTED] = "generic major error",
  [CEED_ERROR_BACKEND - CEED_ERROR_UNSUPPORTED] = "internal backend error",
  [CEED_ERROR_UNSUPPORTED - CEED_ERROR_UNSUPPORTED] = "operation unsupported by backend",
};
const char *const *CeedErrorTypes =  &CeedErrorTypesShifted[-
                                     CEED_ERROR_UNSUPPORTED];

const char *const CeedMemTypes[] = {
  [CEED_MEM_HOST] = "host",
  [CEED_MEM_DEVICE] = "device",
};

const char *const CeedCopyModes[] = {
  [CEED_COPY_VALUES] = "copy values",
  [CEED_USE_POINTER] = "use pointer",
  [CEED_OWN_POINTER] = "own pointer",
};

const char *const CeedTransposeModes[] = {
  [CEED_TRANSPOSE] = "transpose",
  [CEED_NOTRANSPOSE] = "no transpose",
};

const char *const CeedEvalModes[] = {
  [CEED_EVAL_NONE] = "none",
  [CEED_EVAL_INTERP] = "interpolation",
  [CEED_EVAL_GRAD] = "gradient",
  [CEED_EVAL_DIV] = "divergence",
  [CEED_EVAL_CURL] = "curl",
  [CEED_EVAL_WEIGHT] = "quadrature weights",
};

const char *const CeedQuadModes[] = {
  [CEED_GAUSS] = "Gauss",
  [CEED_GAUSS_LOBATTO] = "Gauss Lobatto",
};

const char *const CeedElemTopologies[] = {
  [CEED_LINE] = "line",
  [CEED_TRIANGLE] = "triangle",
  [CEED_QUAD] = "quadrilateral",
  [CEED_TET] = "tetrahedron",
  [CEED_PYRAMID] = "pyramid",
  [CEED_PRISM] = "prism",
  [CEED_HEX] = "hexahedron",
};
