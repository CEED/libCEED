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

// Instantiate khash structs and methods
CeedHashIJKLMInit(m32, libxsmm_dmmfunction)

typedef struct {
  bool is_tensor;
  CeedInt P, Q, dim;
  khash_t(m32) *lookup;
} CeedTensorContract_Xsmm;

CEED_INTERN int CeedTensorContractCreate_Xsmm(CeedBasis basis,
    CeedTensorContract contract);

#endif // _ceed_xsmm_h
