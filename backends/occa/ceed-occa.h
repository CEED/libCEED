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
#include <ceed-backend.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
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
  occaMemory d_array;
} CeedVector_Occa;

// *****************************************************************************
// * CeedElemRestriction Occa struct
// *****************************************************************************
#define CEED_OCCA_NUM_RESTRICTION_KERNELS 7
typedef struct {
  bool identity;
  occaMemory d_indices;
  occaMemory d_toffsets;
  occaMemory d_tindices;
  occaKernel kRestrict[CEED_OCCA_NUM_RESTRICTION_KERNELS];
} CeedElemRestriction_Occa;

// *****************************************************************************
// * CeedBasis Occa struct
// *****************************************************************************
typedef struct {
  bool ready;
  CeedElemRestriction er;
  occaMemory qref1d;
  occaMemory qweight1d;
  occaMemory interp1d;
  occaMemory grad1d;
  occaMemory tmp0,tmp1;
  occaKernel kZero,kInterp,kGrad,kWeight;
} CeedBasis_Occa;

// *****************************************************************************
// * CeedOperator Occa struct
// *****************************************************************************
typedef struct {
  CeedVector *Evecs; /// E-vectors needed to apply operator (in followed by out)
  CeedScalar **Edata;
  CeedVector *evecsin;   /// Input E-vectors needed to apply operator
  CeedVector *evecsout;   /// Output E-vectors needed to apply operator
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_Occa;

// *****************************************************************************
// * CeedQFunction Occa struct
// *****************************************************************************
#define N_MAX_IDX 16
typedef struct {
  bool ready;
  CeedInt idx,odx;
  CeedInt iOf7[N_MAX_IDX];
  CeedInt oOf7[N_MAX_IDX];
  int nc, dim, nelem, elemsize, e;
  occaMemory o_indata, o_outdata;
  occaMemory d_ctx, d_idx, d_odx;
  char *oklPath;
  const char *qFunctionName;
  occaKernel kQFunctionApply;
  CeedOperator op;
} CeedQFunction_Occa;

// *****************************************************************************
// * Ceed Occa struct
// *****************************************************************************
typedef struct {
  occaDevice device;
  bool debug;
  bool ocl;
  char *libceed_dir;
  char *occa_cache_dir;
} Ceed_Occa;

// *****************************************************************************
CEED_INTERN int CeedOklPath_Occa(const Ceed, const char *, const char *,
                                 char **);

// *****************************************************************************
CEED_INTERN int CeedOklDladdr_Occa(Ceed);

// *****************************************************************************
// CEED_DEBUG_COLOR default value, forward CeedDebug* declarations & dbg macros
// *****************************************************************************
#ifndef CEED_DEBUG_COLOR
#define CEED_DEBUG_COLOR 0
#endif
void CeedDebugImpl(const Ceed,const char *,...);
void CeedDebugImpl256(const Ceed,const unsigned char,const char *,...);
#define CeedDebug(ceed,format, ...) CeedDebugImpl(ceed,format, ## __VA_ARGS__)
#define CeedDebug256(ceed,color, ...) CeedDebugImpl256(ceed,color, ## __VA_ARGS__)
#define dbg(...) CeedDebug256(ceed,(unsigned char)CEED_DEBUG_COLOR, ## __VA_ARGS__)

// *****************************************************************************
CEED_INTERN int CeedBasisCreateTensorH1_Occa(CeedInt dim,
    CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
    const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisCreateH1_Occa(CeedElemTopology topo,
                                       CeedInt dim, CeedInt ndof, CeedInt nqpts,
                                       const CeedScalar *interp1d,
                                       const CeedScalar *grad1d,
                                       const CeedScalar *qref1d,
                                       const CeedScalar *qweight1d,
                                       CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisApplyElems_Occa(CeedBasis basis, CeedInt Q,
    CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

// *****************************************************************************
CEED_INTERN int CeedOperatorCreate_Occa(CeedOperator op);

// *****************************************************************************
CEED_INTERN int CeedQFunctionCreate_Occa(CeedQFunction qf);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreate_Occa(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreateBlocked_Occa(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedVectorCreate_Occa(CeedInt n, CeedVector vec);
