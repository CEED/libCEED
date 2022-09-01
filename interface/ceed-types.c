// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

const char *const CeedErrorTypesShifted[] = {
    [CEED_ERROR_SUCCESS - CEED_ERROR_UNSUPPORTED]      = "success",
    [CEED_ERROR_MINOR - CEED_ERROR_UNSUPPORTED]        = "generic minor error",
    [CEED_ERROR_DIMENSION - CEED_ERROR_UNSUPPORTED]    = "dimension mismatch",
    [CEED_ERROR_INCOMPLETE - CEED_ERROR_UNSUPPORTED]   = "object setup incomplete",
    [CEED_ERROR_INCOMPATIBLE - CEED_ERROR_UNSUPPORTED] = "incompatible arguments",
    [CEED_ERROR_ACCESS - CEED_ERROR_UNSUPPORTED]       = "access lock in use",
    [CEED_ERROR_MAJOR - CEED_ERROR_UNSUPPORTED]        = "generic major error",
    [CEED_ERROR_BACKEND - CEED_ERROR_UNSUPPORTED]      = "internal backend error",
    [CEED_ERROR_UNSUPPORTED - CEED_ERROR_UNSUPPORTED]  = "operation unsupported by backend",
};
const char *const *CeedErrorTypes = &CeedErrorTypesShifted[-CEED_ERROR_UNSUPPORTED];

const char *const CeedMemTypes[] = {
    [CEED_MEM_HOST]   = "host",
    [CEED_MEM_DEVICE] = "device",
};

const char *const CeedCopyModes[] = {
    [CEED_COPY_VALUES] = "copy values",
    [CEED_USE_POINTER] = "use pointer",
    [CEED_OWN_POINTER] = "own pointer",
};

const char *const CeedTransposeModes[] = {
    [CEED_TRANSPOSE]   = "transpose",
    [CEED_NOTRANSPOSE] = "no transpose",
};

const char *const CeedEvalModes[] = {
    [CEED_EVAL_NONE] = "none", [CEED_EVAL_INTERP] = "interpolation",      [CEED_EVAL_GRAD] = "gradient", [CEED_EVAL_DIV] = "divergence",
    [CEED_EVAL_CURL] = "curl", [CEED_EVAL_WEIGHT] = "quadrature weights",
};

const char *const CeedQuadModes[] = {
    [CEED_GAUSS]         = "Gauss",
    [CEED_GAUSS_LOBATTO] = "Gauss Lobatto",
};

const char *const CeedElemTopologies[] = {
    [CEED_TOPOLOGY_LINE] = "line",       [CEED_TOPOLOGY_TRIANGLE] = "triangle", [CEED_TOPOLOGY_QUAD] = "quadrilateral",
    [CEED_TOPOLOGY_TET] = "tetrahedron", [CEED_TOPOLOGY_PYRAMID] = "pyramid",   [CEED_TOPOLOGY_PRISM] = "prism",
    [CEED_TOPOLOGY_HEX] = "hexahedron",
};

const char *const CeedContextFieldTypes[] = {
    [CEED_CONTEXT_FIELD_DOUBLE] = "double",
    [CEED_CONTEXT_FIELD_INT32]  = "int32",
};

const char *const CeedFESpaces[] = {
    [CEED_FE_SPACE_H1]   = "H^1 space",
    [CEED_FE_SPACE_HDIV] = "H(div) space",
};
