// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

// magma functions specific to ceed

#include <string.h>
#include <ceed-backend.h>
#include "magma.h"

typedef struct {
  CeedScalar *dqref1d;
  CeedScalar *dinterp1d;
  CeedScalar *dgrad1d;
  CeedScalar *dqweight1d;
} CeedBasis_Magma;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setupdone;
} CeedQFunction_Magma;

#define USE_MAGMA_BATCH
#define USE_MAGMA_BATCH2
#define USE_MAGMA_BATCH3
#define USE_MAGMA_BATCH4

#ifdef __cplusplus
CEED_INTERN {
#endif
  void magmablas_dbasis_apply_batched_eval_interp(magma_int_t P, magma_int_t Q,
      magma_int_t dim, magma_int_t ncomp, const double *dT,
      CeedTransposeMode tmode, const double *dU, magma_int_t ustride,
      double *dV, magma_int_t vstride, magma_int_t batchCount);

  void magmablas_dbasis_apply_batched_eval_grad(magma_int_t P, magma_int_t Q,
      magma_int_t dim, magma_int_t ncomp, magma_int_t nqpt,
      const double* dinterp1d, const double *dgrad1d, CeedTransposeMode tmode,
      const double *dU, magma_int_t ustride, double *dV, magma_int_t vstride,
      magma_int_t batchCount, magma_int_t dim_ctr);

  void magmablas_dbasis_apply_batched_eval_weight(magma_int_t Q,
      magma_int_t dim, const double *dqweight1d, double *dV,
      magma_int_t vstride, magma_int_t batchCount);

  magma_int_t
  magma_isdevptr(const void *A);

  CEED_INTERN int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d,
      CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
      const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

  CEED_INTERN int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim,
                                          CeedInt ndof, CeedInt nqpts,
                                          const CeedScalar *interp,
                                          const CeedScalar *grad,
                                          const CeedScalar *qref,
                                          const CeedScalar *qweight,
                                          CeedBasis basis);
  #ifdef __cplusplus
}
  #endif

#define CeedDebug(...)
//#define CeedDebug(format, ...) fprintf(stderr, format, ## __VA_ARGS__)

// comment the line below to use the default magma_is_devptr function
#define magma_is_devptr magma_isdevptr

// batch stride, override using -DMAGMA_BATCH_STRIDE=<desired-value>
#ifndef MAGMA_BATCH_STRIDE
#define MAGMA_BATCH_STRIDE (1000)
#endif
