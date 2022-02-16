// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

#ifndef _ceed_xsmm_h
#define _ceed_xsmm_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/hash.h>
#include <libxsmm.h>

#if !defined(LIBXSMM_VERSION_GE)
#define LIBXSMM_VERSION_GE(major, minor, update, patch)                           \
  (LIBXSMM_VERSION_MAJOR > major ||                                               \
   (LIBXSMM_VERSION_MAJOR == major &&                                             \
    (LIBXSMM_VERSION_MINOR > minor ||                                             \
     (LIBXSMM_VERSION_MINOR == minor &&                                           \
      (LIBXSMM_VERSION_UPDATE > update ||                                         \
       (LIBXSMM_VERSION_UPDATE == update && LIBXSMM_VERSION_PATCH >= patch ))))))
#endif

#if LIBXSMM_VERSION_GE(1, 17, 0, 0)
# define LIBXSMM_MMFUNCTION_KERNEL(a, b, c) kernel(a, b, c)
#else
# define LIBXSMM_MMFUNCTION_KERNEL(a, b, c) kernel(a, b, c, NULL, NULL, NULL)
#endif

// Instantiate khash structs and methods
CeedHashIJKLMInit(f32, libxsmm_smmfunction)
CeedHashIJKLMInit(f64, libxsmm_dmmfunction)

typedef struct {
  bool is_tensor;
  CeedInt P, Q, dim;
  khash_t(f32) *lookup_f32;
  khash_t(f64) *lookup_f64;
} CeedTensorContract_Xsmm;

CEED_INTERN int CeedTensorContractCreate_f32_Xsmm(CeedBasis basis,
    CeedTensorContract contract);

CEED_INTERN int CeedTensorContractCreate_f64_Xsmm(CeedBasis basis,
    CeedTensorContract contract);

#endif // _ceed_xsmm_h
