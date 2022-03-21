// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef CEED_OCCA_KERNELS_ELEMRESTRICTION_HEADER
#define CEED_OCCA_KERNELS_ELEMRESTRICTION_HEADER

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - COMPONENT_COUNT            : CeedInt
// - ELEMENT_SIZE               : CeedInt
// - NODE_COUNT                 : CeedInt
// - TILE_SIZE                  : int
// - USES_INDICES               : bool
// - STRIDE_TYPE                : ceed::occa::StrideType
// - NODE_STRIDE                : Optional[CeedInt]
// - COMPONENT_STRIDE           : Optional[CeedInt]
// - ELEMENT_STRIDE             : Optional[CeedInt]
// - UNSTRIDED_COMPONENT_STRIDE : Optional[CeedInt]

extern const char *occa_elem_restriction_source;

#endif
