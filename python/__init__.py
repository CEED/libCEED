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

from .ceed import Ceed
from .ceed_vector import Vector
from .ceed_basis import Basis, BasisTensorH1, BasisTensorH1Lagrange, BasisH1
from .ceed_elemrestriction import ElemRestriction, IdentityElemRestriction, BlockedElemRestriction
from .ceed_qfunction import QFunction, QFunctionByName, IdentityQFunction
from .ceed_operator import Operator, CompositeOperator
from .ceed_constants import *

# ------------------------------------------------------------------------------
# All contents of module
# ------------------------------------------------------------------------------
__all__ = ["Ceed",
           "Vector",
           "Basis", "BasisTensorH1", "BasisTensorH1Lagrange", "BasisH1",
           "ElemRestriction", "IdentityElemRestriction", "BlockedElemRestriction",
           "QFunction", "QFunctionByName", "IdentityQFunction",
           "Operator", "CompositeOperator",
           "MEM_HOST", "MEM_DEVICE", "mem_types",
           "COPY_VALUES", "USE_POINTER", "OWN_POINTER", "copy_modes",
           "TRANSPOSE", "NOTRANSPOSE", "transpose_modes",
           "EVAL_NONE", "EVAL_INTERP", "EVAL_GRAD", "EVAL_DIV", "EVAL_CURL", "EVAL_WEIGHT", "eval_modes",
           "GAUSS", "GAUSS_LOBATTO", "quad_modes",
           "LINE", "TRIANGLE", "QUAD", "TET", "PYRAMID", "PRISM", "HEX", "elem_topologies",
           "REQUEST_IMMEDIATE", "REQUEST_ORDERED",
           "VECTOR_ACTIVE", "VECTOR_NONE"]

# ------------------------------------------------------------------------------
