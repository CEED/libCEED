// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <string.h>
#include "ceed-magma.h"
#ifdef CEED_MAGMA_USE_HIP
#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#else
#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#endif

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisApply_Magma(CeedBasis basis, CeedInt nelem,
                         CeedTransposeMode tmode, CeedEvalMode emode,
                         CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedInt dim, ncomp, ndof;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  const CeedScalar *u;
  CeedScalar *v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &u); CeedChkBackend(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArrayWrite(V, CEED_MEM_DEVICE, &v); CeedChkBackend(ierr);

  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);

  CeedInt P1d, Q1d;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);

  CeedDebug(ceed, "\033[01m[CeedBasisApply_Magma] vsize=%" CeedInt_FMT
            ", comp = %" CeedInt_FMT, ncomp*CeedIntPow(P1d, dim), ncomp);

  if (tmode == CEED_TRANSPOSE) {
    CeedSize length;
    ierr = CeedVectorGetLength(V, &length); CeedChkBackend(ierr);
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magmablas_slaset(MagmaFull, length, 1, 0., 0., (float *) v, length,
                       data->queue);
    } else {
      magmablas_dlaset(MagmaFull, length, 1, 0., 0., (double *) v, length,
                       data->queue);
    }
    ceed_magma_queue_sync( data->queue );
  }

  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = P1d, Q = Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d; Q = P1d;
    }

    // Define element sizes for dofs/quad
    CeedInt elquadsize = CeedIntPow(Q1d, dim);
    CeedInt eldofssize = CeedIntPow(P1d, dim);

    // E-vector ordering -------------- Q-vector ordering
    //  component                        component
    //    elem                             elem
    //       node                            node

    // ---  Define strides for NOTRANSPOSE mode: ---
    // Input (u) is E-vector, output (v) is Q-vector

    // Element strides
    CeedInt u_elstride = eldofssize;
    CeedInt v_elstride = elquadsize;
    // Component strides
    CeedInt u_compstride = nelem * eldofssize;
    CeedInt v_compstride = nelem * elquadsize;

    // ---  Swap strides for TRANSPOSE mode: ---
    if (tmode == CEED_TRANSPOSE) {
      // Input (u) is Q-vector, output (v) is E-vector
      // Element strides
      v_elstride = eldofssize;
      u_elstride = elquadsize;
      // Component strides
      v_compstride = nelem * eldofssize;
      u_compstride = nelem * elquadsize;
    }

    CeedInt nthreads = 1;
    CeedInt ntcol = 1;
    CeedInt shmem = 0;
    CeedInt maxPQ = CeedIntMax(P, Q);

    switch (dim) {
    case 1:
      nthreads = maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_1D);
      shmem += sizeof(CeedScalar) * ntcol * ( ncomp * (1*P + 1*Q) );
      shmem += sizeof(CeedScalar) * (P*Q);
      break;
    case 2:
      nthreads = maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_2D);
      shmem += P*Q    *sizeof(CeedScalar);  // for sT
      shmem += ntcol * ( P*maxPQ*sizeof(
                           CeedScalar) );  // for reforming rU we need PxP, and for the intermediate output we need PxQ
      break;
    case 3:
      nthreads = maxPQ*maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_3D);
      shmem += sizeof(CeedScalar)* (P*Q);  // for sT
      shmem += sizeof(CeedScalar)* ntcol * (CeedIntMax(P*P*maxPQ,
                                            P*Q*Q));  // rU needs P^2xP, the intermediate output needs max(P^2xQ,PQ^2)
    }
    CeedInt grid = (nelem + ntcol-1) / ntcol;
    void *args[] = {&impl->dinterp1d,
                    &u, &u_elstride, &u_compstride,
                    &v, &v_elstride, &v_compstride,
                    &nelem
                   };

    if (tmode == CEED_TRANSPOSE) {
      ierr = CeedRunKernelDimSharedMagma(ceed, impl->magma_interp_tr, grid,
                                         nthreads, ntcol, 1, shmem,
                                         args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimSharedMagma(ceed, impl->magma_interp, grid,
                                         nthreads, ntcol, 1, shmem,
                                         args); CeedChkBackend(ierr);
    }
  }
  break;
  case CEED_EVAL_GRAD: {
    CeedInt P = P1d, Q = Q1d;
    // In CEED_NOTRANSPOSE mode:
    // u is (P^dim x nc), column-major layout (nc = ncomp)
    // v is (Q^dim x nc x dim), column-major layout (nc = ncomp)
    // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d, Q = P1d;
    }

    // Define element sizes for dofs/quad
    CeedInt elquadsize = CeedIntPow(Q1d, dim);
    CeedInt eldofssize = CeedIntPow(P1d, dim);

    // E-vector ordering -------------- Q-vector ordering
    //                                  dim
    //  component                        component
    //    elem                              elem
    //       node                            node

    // ---  Define strides for NOTRANSPOSE mode: ---
    // Input (u) is E-vector, output (v) is Q-vector

    // Element strides
    CeedInt u_elstride = eldofssize;
    CeedInt v_elstride = elquadsize;
    // Component strides
    CeedInt u_compstride = nelem * eldofssize;
    CeedInt v_compstride = nelem * elquadsize;
    // Dimension strides
    CeedInt u_dimstride = 0;
    CeedInt v_dimstride = nelem * elquadsize * ncomp;

    // ---  Swap strides for TRANSPOSE mode: ---
    if (tmode == CEED_TRANSPOSE) {
      // Input (u) is Q-vector, output (v) is E-vector
      // Element strides
      v_elstride = eldofssize;
      u_elstride = elquadsize;
      // Component strides
      v_compstride = nelem * eldofssize;
      u_compstride = nelem * elquadsize;
      // Dimension strides
      v_dimstride = 0;
      u_dimstride = nelem * elquadsize * ncomp;

    }

    CeedInt nthreads = 1;
    CeedInt ntcol = 1;
    CeedInt shmem = 0;
    CeedInt maxPQ = CeedIntMax(P, Q);

    switch (dim) {
    case 1:
      nthreads = maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_1D);
      shmem += sizeof(CeedScalar) * ntcol * (ncomp * (1*P + 1*Q));
      shmem += sizeof(CeedScalar) * (P*Q);
      break;
    case 2:
      nthreads = maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_2D);
      shmem += sizeof(CeedScalar) * 2*P*Q;  // for sTinterp and sTgrad
      shmem += sizeof(CeedScalar) * ntcol *
               (P*maxPQ);  // for reforming rU we need PxP, and for the intermediate output we need PxQ
      break;
    case 3:
      nthreads = maxPQ * maxPQ;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_3D);
      shmem += sizeof(CeedScalar) * 2*P*Q;  // for sTinterp and sTgrad
      shmem += sizeof(CeedScalar) * ntcol * CeedIntMax(P*P*P,
               (P*P*Q) +
               (P*Q*Q));  // rU needs P^2xP, the intermediate outputs need (P^2.Q + P.Q^2)
    }
    CeedInt grid = (nelem + ntcol-1) / ntcol;
    void *args[] = {&impl->dinterp1d, &impl->dgrad1d,
                    &u, &u_elstride, &u_compstride, &u_dimstride,
                    &v, &v_elstride, &v_compstride, &v_dimstride,
                    &nelem
                   };

    if (tmode == CEED_TRANSPOSE) {
      ierr = CeedRunKernelDimSharedMagma(ceed, impl->magma_grad_tr, grid,
                                         nthreads, ntcol, 1, shmem,
                                         args); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimSharedMagma(ceed, impl->magma_grad, grid,
                                         nthreads, ntcol, 1, shmem,
                                         args); CeedChkBackend(ierr);
    }
  }
  break;
  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // LCOV_EXCL_STOP
    CeedInt Q = Q1d;
    CeedInt eldofssize = CeedIntPow(Q, dim);
    CeedInt nthreads = 1;
    CeedInt ntcol = 1;
    CeedInt shmem = 0;

    switch (dim) {
    case 1:
      nthreads = Q;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_1D);
      shmem += sizeof(CeedScalar) * Q;  // for dqweight1d
      shmem += sizeof(CeedScalar) * ntcol * Q; // for output
      break;
    case 2:
      nthreads = Q;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_2D);
      shmem += sizeof(CeedScalar) * Q;  // for dqweight1d
      break;
    case 3:
      nthreads = Q * Q;
      ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_3D);
      shmem += sizeof(CeedScalar) * Q;  // for dqweight1d
    }
    CeedInt grid = (nelem + ntcol-1) / ntcol;
    void *args[] = {&impl->dqweight1d, &v, &eldofssize, &nelem};

    ierr = CeedRunKernelDimSharedMagma(ceed, impl->magma_weight, grid,
                                       nthreads, ntcol, 1, shmem,
                                       args); CeedChkBackend(ierr);
  }
  break;
  // LCOV_EXCL_START
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // must sync to ensure completeness
  ceed_magma_queue_sync( data->queue );

  if (emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisApplyNonTensor_f64_Magma(CeedBasis basis, CeedInt nelem,
                                      CeedTransposeMode tmode, CeedEvalMode emode,
                                      CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  CeedInt dim, ncomp, ndof, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChkBackend(ierr);
  const CeedScalar *du;
  CeedScalar *dv;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &du); CeedChkBackend(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArrayWrite(V, CEED_MEM_DEVICE, &dv); CeedChkBackend(ierr);

  CeedBasisNonTensor_Magma *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);

  CeedDebug(ceed, "\033[01m[CeedBasisApplyNonTensor_Magma] vsize=%" CeedInt_FMT
            ", comp = %" CeedInt_FMT, ncomp*ndof, ncomp);

  if (tmode == CEED_TRANSPOSE) {
    CeedSize length;
    ierr = CeedVectorGetLength(V, &length);
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magmablas_slaset(MagmaFull, length, 1, 0., 0., (float *) dv, length,
                       data->queue);
    } else {
      magmablas_dlaset(MagmaFull, length, 1, 0., 0., (double *) dv, length,
                       data->queue);
    }
    ceed_magma_queue_sync( data->queue );
  }

  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE)
      magma_dgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                            P, nelem*ncomp, Q,
                            1.0, (double *)impl->dinterp, P,
                            (double *)du, Q,
                            0.0, (double *)dv, P, data->queue);
    else
      magma_dgemm_nontensor(MagmaTrans, MagmaNoTrans,
                            Q, nelem*ncomp, P,
                            1.0, (double *)impl->dinterp, P,
                            (double *)du, P,
                            0.0, (double *)dv, Q, data->queue);
  }
  break;

  case CEED_EVAL_GRAD: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE) {
      CeedScalar beta = 0.0;
      for(int d=0; d<dim; d++) {
        if (d>0)
          beta = 1.0;
        magma_dgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                              P, nelem*ncomp, Q,
                              1.0, (double *)(impl->dgrad + d*P*Q), P,
                              (double *)(du + d*nelem*ncomp*Q), Q,
                              beta, (double *)dv, P, data->queue);
      }
    } else {
      for(int d=0; d< dim; d++)
        magma_dgemm_nontensor(MagmaTrans, MagmaNoTrans,
                              Q, nelem*ncomp, P,
                              1.0, (double *)(impl->dgrad + d*P*Q), P,
                              (double *)du, P,
                              0.0, (double *)(dv + d*nelem*ncomp*Q), Q, data->queue);
    }
  }
  break;

  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // LCOV_EXCL_STOP

    int elemsPerBlock = 1;//basis->Q1d < 7 ? optElems[basis->Q1d] : 1;
    int grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)?
                                       1 : 0 );
    magma_weight_nontensor(grid, nqpt, nelem, nqpt, impl->dqweight, dv,
                           data->queue);
    CeedChkBackend(ierr);
  }
  break;

  // LCOV_EXCL_START
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // must sync to ensure completeness
  ceed_magma_queue_sync( data->queue );

  if (emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &du); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &dv); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

int CeedBasisApplyNonTensor_f32_Magma(CeedBasis basis, CeedInt nelem,
                                      CeedTransposeMode tmode, CeedEvalMode emode,
                                      CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  CeedInt dim, ncomp, ndof, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChkBackend(ierr);
  const CeedScalar *du;
  CeedScalar *dv;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &du); CeedChkBackend(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArrayWrite(V, CEED_MEM_DEVICE, &dv); CeedChkBackend(ierr);

  CeedBasisNonTensor_Magma *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);

  CeedDebug(ceed, "\033[01m[CeedBasisApplyNonTensor_Magma] vsize=%" CeedInt_FMT
            ", comp = %" CeedInt_FMT, ncomp*ndof, ncomp);

  if (tmode == CEED_TRANSPOSE) {
    CeedSize length;
    ierr = CeedVectorGetLength(V, &length);
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magmablas_slaset(MagmaFull, length, 1, 0., 0., (float *) dv, length,
                       data->queue);
    } else {
      magmablas_dlaset(MagmaFull, length, 1, 0., 0., (double *) dv, length,
                       data->queue);
    }
    ceed_magma_queue_sync( data->queue );
  }

  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE)
      magma_sgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                            P, nelem*ncomp, Q,
                            1.0, (float *)impl->dinterp, P,
                            (float *)du, Q,
                            0.0, (float *)dv, P, data->queue);
    else
      magma_sgemm_nontensor(MagmaTrans, MagmaNoTrans,
                            Q, nelem*ncomp, P,
                            1.0, (float *)impl->dinterp, P,
                            (float *)du, P,
                            0.0, (float *)dv, Q, data->queue);
  }
  break;

  case CEED_EVAL_GRAD: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE) {
      CeedScalar beta = 0.0;
      for(int d=0; d<dim; d++) {
        if (d>0)
          beta = 1.0;
        magma_sgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                              P, nelem*ncomp, Q,
                              1.0, (float *)(impl->dgrad + d*P*Q), P,
                              (float *)(du + d*nelem*ncomp*Q), Q,
                              beta, (float *)dv, P, data->queue);
      }
    } else {
      for(int d=0; d< dim; d++)
        magma_sgemm_nontensor(MagmaTrans, MagmaNoTrans,
                              Q, nelem*ncomp, P,
                              1.0, (float *)(impl->dgrad + d*P*Q), P,
                              (float *)du, P,
                              0.0, (float *)(dv + d*nelem*ncomp*Q), Q, data->queue);
    }
  }
  break;

  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // LCOV_EXCL_STOP

    int elemsPerBlock = 1;//basis->Q1d < 7 ? optElems[basis->Q1d] : 1;
    int grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)?
                                       1 : 0 );
    magma_weight_nontensor(grid, nqpt, nelem, nqpt, impl->dqweight, dv,
                           data->queue);
    CeedChkBackend(ierr);
  }
  break;

  // LCOV_EXCL_START
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // must sync to ensure completeness
  ceed_magma_queue_sync( data->queue );

  if (emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &du); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &dv); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisDestroy_Magma(CeedBasis basis) {
  int ierr;
  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);

  ierr = magma_free(impl->dqref1d); CeedChkBackend(ierr);
  ierr = magma_free(impl->dinterp1d); CeedChkBackend(ierr);
  ierr = magma_free(impl->dgrad1d); CeedChkBackend(ierr);
  ierr = magma_free(impl->dqweight1d); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  #ifdef CEED_MAGMA_USE_HIP
  ierr = hipModuleUnload(impl->module); CeedChk_Hip(ceed, ierr);
  #else
  ierr = cuModuleUnload(impl->module); CeedChk_Cu(ceed, ierr);
  #endif

  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisDestroyNonTensor_Magma(CeedBasis basis) {
  int ierr;
  CeedBasisNonTensor_Magma *impl;
  ierr = CeedBasisGetData(basis, &impl); CeedChkBackend(ierr);

  ierr = magma_free(impl->dqref); CeedChkBackend(ierr);
  ierr = magma_free(impl->dinterp); CeedChkBackend(ierr);
  ierr = magma_free(impl->dgrad); CeedChkBackend(ierr);
  ierr = magma_free(impl->dqweight); CeedChkBackend(ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                  const CeedScalar *interp1d,
                                  const CeedScalar *grad1d,
                                  const CeedScalar *qref1d,
                                  const CeedScalar *qweight1d, CeedBasis basis) {
  int ierr;
  CeedBasis_Magma *impl;
  ierr = CeedCalloc(1,&impl); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  // Check for supported parameters
  CeedInt ncomp = 0;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  // Compile kernels
  char *magma_common_path;
  char *interp_path, *grad_path, *weight_path;
  char *basis_kernel_source;
  ierr = CeedGetJitAbsolutePath(ceed,
                                "ceed/jit-source/magma/magma_common_device.h",
                                &magma_common_path); CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading Basis Kernel Source -----\n");
  ierr = CeedLoadSourceToBuffer(ceed, magma_common_path,
                                &basis_kernel_source);
  CeedChkBackend(ierr);
  char *interp_name_base = "ceed/jit-source/magma/interp";
  CeedInt interp_name_len = strlen(interp_name_base) + 6;
  char interp_name[interp_name_len];
  snprintf(interp_name, interp_name_len, "%s-%" CeedInt_FMT "d.h",
           interp_name_base, dim);
  ierr = CeedGetJitAbsolutePath(ceed, interp_name, &interp_path);
  CeedChkBackend(ierr);
  ierr = CeedLoadSourceToInitializedBuffer(ceed, interp_path,
         &basis_kernel_source);
  CeedChkBackend(ierr);
  char *grad_name_base = "ceed/jit-source/magma/grad";
  CeedInt grad_name_len = strlen(grad_name_base) + 6;
  char grad_name[grad_name_len];
  snprintf(grad_name, grad_name_len, "%s-%" CeedInt_FMT "d.h", grad_name_base,
           dim);
  ierr = CeedGetJitAbsolutePath(ceed, grad_name, &grad_path);
  CeedChkBackend(ierr);
  ierr = CeedLoadSourceToInitializedBuffer(ceed, grad_path,
         &basis_kernel_source);
  CeedChkBackend(ierr);
  char *weight_name_base = "ceed/jit-source/magma/weight";
  CeedInt weight_name_len = strlen(weight_name_base) + 6;
  char weight_name[weight_name_len];
  snprintf(weight_name, weight_name_len, "%s-%" CeedInt_FMT "d.h",
           weight_name_base, dim);
  ierr = CeedGetJitAbsolutePath(ceed, weight_name, &weight_path);
  CeedChkBackend(ierr);
  ierr = CeedLoadSourceToInitializedBuffer(ceed, weight_path,
         &basis_kernel_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2,
               "----- Loading Basis Kernel Source Complete! -----\n");
  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip
  // data
  Ceed delegate;
  ierr = CeedGetDelegate(ceed, &delegate); CeedChkBackend(ierr);
  ierr = CeedCompileMagma(delegate, basis_kernel_source, &impl->module, 5,
                          "DIM", dim,
                          "NCOMP", ncomp,
                          "P", P1d,
                          "Q", Q1d,
                          "MAXPQ", CeedIntMax(P1d, Q1d));
  CeedChkBackend(ierr);

  // Kernel setup
  switch (dim) {
  case 1:
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpn_1d_kernel",
                              &impl->magma_interp);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpt_1d_kernel",
                              &impl->magma_interp_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradn_1d_kernel",
                              &impl->magma_grad);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradt_1d_kernel",
                              &impl->magma_grad_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_weight_1d_kernel",
                              &impl->magma_weight);
    CeedChkBackend(ierr);
    break;
  case 2:
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpn_2d_kernel",
                              &impl->magma_interp);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpt_2d_kernel",
                              &impl->magma_interp_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradn_2d_kernel",
                              &impl->magma_grad);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradt_2d_kernel",
                              &impl->magma_grad_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_weight_2d_kernel",
                              &impl->magma_weight);
    CeedChkBackend(ierr);
    break;
  case 3:
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpn_3d_kernel",
                              &impl->magma_interp);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_interpt_3d_kernel",
                              &impl->magma_interp_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradn_3d_kernel",
                              &impl->magma_grad);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_gradt_3d_kernel",
                              &impl->magma_grad_tr);
    CeedChkBackend(ierr);
    ierr = CeedGetKernelMagma(ceed, impl->module, "magma_weight_3d_kernel",
                              &impl->magma_weight);
    CeedChkBackend(ierr);
  }

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Magma); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Magma); CeedChkBackend(ierr);

  // Copy qref1d to the GPU
  ierr = magma_malloc((void **)&impl->dqref1d, Q1d*sizeof(qref1d[0]));
  CeedChkBackend(ierr);
  magma_setvector(Q1d, sizeof(qref1d[0]), qref1d, 1, impl->dqref1d, 1,
                  data->queue);

  // Copy interp1d to the GPU
  ierr = magma_malloc((void **)&impl->dinterp1d, Q1d*P1d*sizeof(interp1d[0]));
  CeedChkBackend(ierr);
  magma_setvector(Q1d*P1d, sizeof(interp1d[0]), interp1d, 1, impl->dinterp1d, 1,
                  data->queue);

  // Copy grad1d to the GPU
  ierr = magma_malloc((void **)&impl->dgrad1d, Q1d*P1d*sizeof(grad1d[0]));
  CeedChkBackend(ierr);
  magma_setvector(Q1d*P1d, sizeof(grad1d[0]), grad1d, 1, impl->dgrad1d, 1,
                  data->queue);

  // Copy qweight1d to the GPU
  ierr = magma_malloc((void **)&impl->dqweight1d, Q1d*sizeof(qweight1d[0]));
  CeedChkBackend(ierr);
  magma_setvector(Q1d, sizeof(qweight1d[0]), qweight1d, 1, impl->dqweight1d, 1,
                  data->queue);

  ierr = CeedBasisSetData(basis, impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim, CeedInt ndof,
                            CeedInt nqpts, const CeedScalar *interp,
                            const CeedScalar *grad, const CeedScalar *qref,
                            const CeedScalar *qweight, CeedBasis basis) {
  int ierr;
  CeedBasisNonTensor_Magma *impl;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);

  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP64) {
    ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                  CeedBasisApplyNonTensor_f64_Magma);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                  CeedBasisApplyNonTensor_f32_Magma);
    CeedChkBackend(ierr);
  }
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Magma); CeedChkBackend(ierr);

  ierr = CeedCalloc(1,&impl); CeedChkBackend(ierr);
  ierr = CeedBasisSetData(basis, impl); CeedChkBackend(ierr);

  // Copy qref to the GPU
  ierr = magma_malloc((void **)&impl->dqref, nqpts*sizeof(qref[0]));
  CeedChkBackend(ierr);
  magma_setvector(nqpts, sizeof(qref[0]), qref, 1, impl->dqref, 1, data->queue);

  // Copy interp to the GPU
  ierr = magma_malloc((void **)&impl->dinterp, nqpts*ndof*sizeof(interp[0]));
  CeedChkBackend(ierr);
  magma_setvector(nqpts*ndof, sizeof(interp[0]), interp, 1, impl->dinterp, 1,
                  data->queue);

  // Copy grad to the GPU
  ierr = magma_malloc((void **)&impl->dgrad, nqpts*ndof*dim*sizeof(grad[0]));
  CeedChkBackend(ierr);
  magma_setvector(nqpts*ndof*dim, sizeof(grad[0]), grad, 1, impl->dgrad, 1,
                  data->queue);

  // Copy qweight to the GPU
  ierr = magma_malloc((void **)&impl->dqweight, nqpts*sizeof(qweight[0]));
  CeedChkBackend(ierr);
  magma_setvector(nqpts, sizeof(qweight[0]), qweight, 1, impl->dqweight, 1,
                  data->queue);

  return CEED_ERROR_SUCCESS;
}
