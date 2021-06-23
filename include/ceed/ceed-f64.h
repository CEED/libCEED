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
/// Public header for definitions related to using FP64 floating point (double
/// precision) for CeedScalar. This is the default header included in ceed/ceed.h. 
#ifndef _ceed_f64_h
#define _ceed_f64_h

#include <ceed/ceed.h>

/// Set base scalar type to FP64.  (See CeedScalarType enum in ceed/ceed.h
/// for all options.)
#define CEED_SCALAR_TYPE CEED_SCALAR_FP64
typedef double CeedScalar;

/// Data alignment of 64 bits
#define CEED_ALIGN 64

/// Machine epsilon
#define CEED_EPSILON 1e-16

#endif
