/// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
/// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
/// reserved. See files LICENSE and NOTICE for details.
///
/// This file is part of CEED, a collection of benchmarks, miniapps, software
/// libraries and APIs for efficient high-order finite element and spectral
/// element discretizations for exascale applications. For more information and
/// source code availability see http://github.com/ceed.
///
/// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
/// a collaborative effort of two U.S. Department of Energy organizations (Office
/// of Science and the National Nuclear Security Administration) responsible for
/// the planning and preparation of a capable exascale ecosystem, including
/// software, applications, hardware, advanced system engineering and early
/// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Public header for definitions related to using FP32 floating point (single
/// precision) for CeedScalar.  Include this header in ceed/ceed.h to use 
/// float instead of double. 
#ifndef _ceed_f32_h
#define _ceed_f32_h

/// Set base scalar type to FP32.  (See CeedScalarType enum in ceed/ceed.h
/// for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP32
typedef float CeedScalar;

/// Data alignment of 32 bits
#define CEED_ALIGN 32

/// Machine epsilon
#define CEED_EPSILON 6e-08

#endif
