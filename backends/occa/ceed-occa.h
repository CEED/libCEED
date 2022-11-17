// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <assert.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <string.h>
#include <sys/stat.h>

// *****************************************************************************
#define OCCA_PATH_MAX 4096

// *****************************************************************************
// used to get Dl_info struct declaration (vs _GNU_SOURCE?)
#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <dlfcn.h>

// *****************************************************************************
#include "occa.h"

// *****************************************************************************
#define NO_OFFSET 0
#define TILE_SIZE 32
#define NO_PROPS occaDefault

// *****************************************************************************
// * CeedVector Occa struct
// *****************************************************************************
typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  occaMemory  d_array;
} CeedVector_Occa;

// *****************************************************************************
// * CeedElemRestriction Occa struct
// *****************************************************************************
#define CEED_OCCA_NUM_RESTRICTION_KERNELS 8
typedef struct {
  bool       strided;
  occaMemory d_indices;
  occaMemory d_toffsets;
  occaMemory d_tindices;
  occaKernel kRestrict[CEED_OCCA_NUM_RESTRICTION_KERNELS];
} CeedElemRestriction_Occa;

// *****************************************************************************
// * CeedBasis Occa struct
// *****************************************************************************
typedef struct {
  bool                ready;
  CeedElemRestriction er;
  occaMemory          qref1d;
  occaMemory          qweight1d;
  occaMemory          interp1d;
  occaMemory          grad1d;
  occaMemory          tmp0, tmp1;
  occaKernel          kZero, kInterp, kGrad, kWeight;
} CeedBasis_Occa;

// *****************************************************************************
// * CeedOperator Occa struct
// *****************************************************************************
typedef struct {
  CeedVector  *Evecs;  /// E-vectors needed to apply operator (in followed by out)
  CeedScalar **Edata;
  CeedVector  *evecsin;   /// Input E-vectors needed to apply operator
  CeedVector  *evecsout;  /// Output E-vectors needed to apply operator
  CeedVector  *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector  *qvecsout;  /// Output Q-vectors needed to apply operator
  CeedInt      numein;
  CeedInt      numeout;
} CeedOperator_Occa;

// *****************************************************************************
// * CeedQFunction Occa struct
// *****************************************************************************
#define N_MAX_IDX 16
typedef struct {
  bool         ready;
  CeedInt      idx, odx;
  CeedInt      iOf7[N_MAX_IDX];
  CeedInt      oOf7[N_MAX_IDX];
  int          nc, dim, nelem, elemsize, e;
  occaMemory   o_indata, o_outdata;
  occaMemory   d_ctx, d_idx, d_odx;
  char        *oklPath;
  const char  *qFunctionName;
  occaKernel   kQFunctionApply;
  CeedOperator op;
} CeedQFunction_Occa;

// *****************************************************************************
// * CeedQFunctionContext Occa struct
// *****************************************************************************
typedef struct {
  CeedScalar *h_data;
  CeedScalar *h_data_allocated;
} CeedQFunctionContext_Occa;

// *****************************************************************************
// * Ceed Occa struct
// *****************************************************************************
typedef struct {
  occaDevice device;
  bool       ocl;
  char      *libceed_dir;
  char      *occa_cache_dir;
} Ceed_Occa;

// *****************************************************************************
CEED_INTERN int CeedOklPath_Occa(const Ceed, const char *, const char *, char **);

// *****************************************************************************
CEED_INTERN int CeedOklDladdr_Occa(Ceed);

// *****************************************************************************
CEED_INTERN int CeedBasisCreateTensorH1_Occa(CeedInt dim, CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                             const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisCreateH1_Occa(CeedElemTopology topo, CeedInt dim, CeedInt ndof, CeedInt nqpts, const CeedScalar *interp1d,
                                       const CeedScalar *grad1d, const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisApplyElems_Occa(CeedBasis basis, CeedInt Q, CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

// *****************************************************************************
CEED_INTERN int CeedOperatorCreate_Occa(CeedOperator op);

// *****************************************************************************
CEED_INTERN int CeedQFunctionCreate_Occa(CeedQFunction qf);

// *****************************************************************************
CEED_INTERN int CeedQFunctionContextCreate_Occa(CeedQFunctionContext ctx);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreate_Occa(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *indices,
                                               const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreateBlocked_Occa(const CeedMemType mtype, const CeedCopyMode cmode, const CeedInt *indices,
                                                      const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedVectorCreate_Occa(CeedInt n, CeedVector vec);
