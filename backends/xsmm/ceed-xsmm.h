// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_xsmm_h
#define _ceed_xsmm_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/hash.h>
#include <libxsmm.h>

#if !defined(LIBXSMM_VERSION_GE)
#define LIBXSMM_VERSION_GE(major, minor, update, patch)                  \
  (LIBXSMM_VERSION_MAJOR > major ||                                      \
   (LIBXSMM_VERSION_MAJOR == major &&                                    \
    (LIBXSMM_VERSION_MINOR > minor || (LIBXSMM_VERSION_MINOR == minor && \
                                       (LIBXSMM_VERSION_UPDATE > update || (LIBXSMM_VERSION_UPDATE == update && LIBXSMM_VERSION_PATCH >= patch))))))
#endif

#if LIBXSMM_VERSION_GE(1, 17, 0, 0)
#define LIBXSMM_MMFUNCTION_KERNEL(a, b, c) kernel(a, b, c)
#else
#define LIBXSMM_MMFUNCTION_KERNEL(a, b, c) kernel(a, b, c, NULL, NULL, NULL)
#endif

// Instantiate khash structs and methods
CeedHashIJKLMInit(f32, libxsmm_smmfunction) CeedHashIJKLMInit(f64, libxsmm_dmmfunction)

    typedef struct {
  bool    is_tensor;
  CeedInt P, Q, dim;
  khash_t(f32) * lookup_f32;
  khash_t(f64) * lookup_f64;
} CeedTensorContract_Xsmm;

CEED_INTERN int CeedTensorContractCreate_f32_Xsmm(CeedBasis basis, CeedTensorContract contract);

CEED_INTERN int CeedTensorContractCreate_f64_Xsmm(CeedBasis basis, CeedTensorContract contract);

#endif  // _ceed_xsmm_h
