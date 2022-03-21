# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

from _ceed_cffi import ffi, lib
from abc import ABC

# ------------------------------------------------------------------------------
# Ceed Enums
# ------------------------------------------------------------------------------
# CeedMemType
MEM_HOST = lib.CEED_MEM_HOST
MEM_DEVICE = lib.CEED_MEM_DEVICE
mem_types = {MEM_HOST: "host",
             MEM_DEVICE: "device"}

# CeedScalarType
SCALAR_FP32 = lib.CEED_SCALAR_FP32
SCALAR_FP64 = lib.CEED_SCALAR_FP64
scalar_types = {SCALAR_FP32: "float32",
                SCALAR_FP64: "float64"}
# Machine eps corresponding to CeedScalar
EPSILON = lib.CEED_EPSILON

# CeedCopyMode
COPY_VALUES = lib.CEED_COPY_VALUES
USE_POINTER = lib.CEED_USE_POINTER
OWN_POINTER = lib.CEED_OWN_POINTER
copy_modes = {COPY_VALUES: "copy values",
              USE_POINTER: "use pointer",
              OWN_POINTER: "own pointer"}

# CeedNormType
NORM_1 = lib.CEED_NORM_1
NORM_2 = lib.CEED_NORM_2
NORM_MAX = lib.CEED_NORM_MAX
norm_types = {NORM_1: "L1 norm",
              NORM_2: "L2 norm",
              NORM_MAX: "max norm"}

# CeedTransposeMode
TRANSPOSE = lib.CEED_TRANSPOSE
NOTRANSPOSE = lib.CEED_NOTRANSPOSE
transpose_modes = {TRANSPOSE: "transpose",
                   NOTRANSPOSE: "no transpose"}

# CeedEvalMode
EVAL_NONE = lib.CEED_EVAL_NONE
EVAL_INTERP = lib.CEED_EVAL_INTERP
EVAL_GRAD = lib.CEED_EVAL_GRAD
EVAL_DIV = lib.CEED_EVAL_DIV
EVAL_CURL = lib.CEED_EVAL_CURL
EVAL_WEIGHT = lib.CEED_EVAL_WEIGHT
eval_modes = {EVAL_NONE: "none",
              EVAL_INTERP: "interpolation",
              EVAL_GRAD: "gradient",
              EVAL_DIV: "divergence",
              EVAL_CURL: "curl",
              EVAL_WEIGHT: "quadrature weights"}

# CeedQuadMode
GAUSS = lib.CEED_GAUSS
GAUSS_LOBATTO = lib.CEED_GAUSS_LOBATTO
quad_modes = {GAUSS: "Gauss",
              GAUSS_LOBATTO: "Gauss Lobatto"}

# CeedElemTopology
LINE = lib.CEED_TOPOLOGY_LINE
TRIANGLE = lib.CEED_TOPOLOGY_TRIANGLE
QUAD = lib.CEED_TOPOLOGY_QUAD
TET = lib.CEED_TOPOLOGY_TET
PYRAMID = lib.CEED_TOPOLOGY_PYRAMID
PRISM = lib.CEED_TOPOLOGY_PRISM
HEX = lib.CEED_TOPOLOGY_HEX
elem_topologies = {LINE: "line",
                   TRIANGLE: "triangle",
                   QUAD: "quadrilateral",
                   TET: "tetrahedron",
                   PYRAMID: "pyramid",
                   PRISM: "prism",
                   HEX: "hexahedron"}

# ------------------------------------------------------------------------------
# Ceed Constants
# ------------------------------------------------------------------------------

# Requests
REQUEST_IMMEDIATE = lib.CEED_REQUEST_IMMEDIATE
REQUEST_ORDERED = lib.CEED_REQUEST_ORDERED

# Object shell


class _CeedConstantObject(ABC):
    """Shell for holding constant Vector and Basis constants."""

    def __init__(self, constant):
        self._pointer = [constant]


# Vectors
VECTOR_ACTIVE = _CeedConstantObject(lib.CEED_VECTOR_ACTIVE)
VECTOR_NONE = _CeedConstantObject(lib.CEED_VECTOR_NONE)

# ElemRestriction
ELEMRESTRICTION_NONE = _CeedConstantObject(lib.CEED_ELEMRESTRICTION_NONE)

# Basis
BASIS_COLLOCATED = _CeedConstantObject(lib.CEED_BASIS_COLLOCATED)

# ------------------------------------------------------------------------------
