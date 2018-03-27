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
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <ceed-impl.h>

// *****************************************************************************
#include "occa.h"

// *****************************************************************************
#define NO_OFFSET 0
#define NO_PROPS occaDefault
#define TILE_SIZE 32

// *****************************************************************************
// * CeedVector Occa struct
// *****************************************************************************
typedef struct {
  CeedScalar *h_array;
  CeedScalar *used_pointer;
  occaMemory d_array;
} CeedVector_Occa;

// *****************************************************************************
// * CeedElemRestriction Occa struct
// *****************************************************************************
typedef struct {
  occaMemory d_indices;
  occaMemory d_toffsets;
  occaMemory d_tindices;
  occaKernel kRestrict[9];
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
  CeedVector etmp;
  CeedVector BEu,BEv;
  CeedVector qdata;
} CeedOperator_Occa;

// *****************************************************************************
// * CeedQFunction Occa struct
// *****************************************************************************
typedef struct {
  bool op, ready;
  int nc, dim, nelem, elemsize, e;
  occaMemory d_q,d_u,b_u,d_v,b_v;
  char *oklPath;
  const char *qFunctionName;
  occaKernel kQFunctionApply;
} CeedQFunction_Occa;

// *****************************************************************************
// * Ceed Occa struct
// *****************************************************************************
typedef struct {
  occaDevice device;
  bool debug;
} Ceed_Occa;

// *****************************************************************************
// CEED_DEBUG_COLOR default value, forward CeedDebug* declarations & dbg macros
// *****************************************************************************
#ifndef CEED_DEBUG_COLOR
#define CEED_DEBUG_COLOR 0
#endif
void CeedDebugImpl(const Ceed,const char*,...);
void CeedDebugImpl256(const Ceed,const unsigned char,const char*,...);
#define CeedDebug(ceed,format, ...) CeedDebugImpl(ceed,format, ## __VA_ARGS__)
#define CeedDebug256(ceed,color, ...) CeedDebugImpl256(ceed,color, ## __VA_ARGS__)
#define dbg(...) CeedDebug256(ceed,(unsigned char)CEED_DEBUG_COLOR, ## __VA_ARGS__)

// *****************************************************************************
int CeedBasisCreateTensorH1_Occa(Ceed ceed, CeedInt dim, CeedInt P1d,
                                 CeedInt Q1d, const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis);

// *****************************************************************************
int CeedBasisApplyElems_Occa(CeedBasis basis, CeedInt Q,
                             CeedTransposeMode tmode, CeedEvalMode emode,
                             const CeedVector u, CeedVector v);

// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op);

// *****************************************************************************
int CeedQFunctionCreate_Occa(CeedQFunction qf);

// *****************************************************************************
int CeedElemRestrictionCreate_Occa(const CeedElemRestriction res,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   const CeedInt *indices);

// *****************************************************************************
int CeedVectorCreate_Occa(Ceed ceed, CeedInt n, CeedVector vec);
