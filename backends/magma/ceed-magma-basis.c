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

#include "ceed-magma.h"

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisApply_Magma(CeedBasis basis, CeedInt nelem,
                         CeedTransposeMode tmode, CeedEvalMode emode,
                         CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedInt dim, ncomp, ndof;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChk(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);

  const CeedScalar *u;
  CeedScalar *v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_DEVICE, &v); CeedChk(ierr);

  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);

  CeedInt P1d, Q1d;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);

  CeedDebug("\033[01m[CeedBasisApply_Magma] vsize=%d, comp = %d",
            ncomp*CeedIntPow(P1d, dim), ncomp);

  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(V, &length);
    magmablas_dlaset(MagmaFull, length, 1, 0., 0., v, length, data->queue);
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

    ierr = magma_interp(P, Q, dim, ncomp, 
            impl->dinterp1d, tmode, 
            u, u_elstride, u_compstride, 
            v, v_elstride, v_compstride, 
            nelem, data->basis_kernel_mode, data->maxthreads, 
            data->queue);
    if(ierr != 0) CeedError(ceed, 1, "MAGMA: launch failure detected for magma_interp");
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

    ierr = magma_grad( P, Q, dim, ncomp,  
            impl->dinterp1d, impl->dgrad1d, tmode, 
            u, u_elstride, u_compstride, u_dimstride, 
            v, v_elstride, v_compstride, v_dimstride, 
            nelem, data->basis_kernel_mode, data->maxthreads, 
            data->queue);
    if(ierr != 0) CeedError(ceed, 1, "MAGMA: launch failure detected for magma_grad");
  }
  break;
  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      // LCOV_EXCL_START
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // LCOV_EXCL_STOP
    CeedInt Q = Q1d;
    int eldofssize = CeedIntPow(Q, dim);
    ierr = magma_weight(Q, dim, impl->dqweight1d, v, eldofssize, nelem, data->basis_kernel_mode, data->maxthreads, data->queue);
    if(ierr != 0) CeedError(ceed, 1, "MAGMA: launch failure detected for magma_weight");
  }
  break;
  // LCOV_EXCL_START
  case CEED_EVAL_DIV:
    return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(ceed, 1,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // must sync to ensure completeness
  ceed_magma_queue_sync( data->queue );

  if (emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisApplyNonTensor_Magma(CeedBasis basis, CeedInt nelem,
                                  CeedTransposeMode tmode, CeedEvalMode emode,
                                  CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);

  CeedInt dim, ncomp, ndof, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  const CeedScalar *du;
  CeedScalar *dv;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &du); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
    // LCOV_EXCL_STOP
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_DEVICE, &dv); CeedChk(ierr);

  CeedBasisNonTensor_Magma *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);

  CeedDebug("\033[01m[CeedBasisApplyNonTensor_Magma] vsize=%d, comp = %d",
            ncomp*ndof, ncomp);

  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(V, &length);
    magmablas_dlaset(MagmaFull, length, 1, 0., 0., dv, length, data->queue);
    ceed_magma_queue_sync( data->queue );
  }
  switch (emode) {
  case CEED_EVAL_INTERP: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE)
      magma_dgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                            P, nelem*ncomp, Q,
                            1.0, impl->dinterp, P,
                            du, Q,
                            0.0, dv, P, data->queue);
    else
      magma_dgemm_nontensor(MagmaTrans, MagmaNoTrans,
                            Q, nelem*ncomp, P,
                            1.0, impl->dinterp, P,
                            du, P,
                            0.0, dv, Q, data->queue);
  }
  break;

  case CEED_EVAL_GRAD: {
    CeedInt P = ndof, Q = nqpt;
    if (tmode == CEED_TRANSPOSE) {
      double beta = 0.0;
      for(int d=0; d<dim; d++) {
        if (d>0)
          beta = 1.0;
        magma_dgemm_nontensor(MagmaNoTrans, MagmaNoTrans,
                              P, nelem*ncomp, Q,
                              1.0, impl->dgrad + d*P*Q, P,
                              du + d*nelem*ncomp*Q, Q,
                              beta, dv, P, data->queue);
      }
    } else {
      for(int d=0; d< dim; d++)
        magma_dgemm_nontensor(MagmaTrans, MagmaNoTrans,
                              Q, nelem*ncomp, P,
                              1.0, impl->dgrad + d*P*Q, P,
                              du, P,
                              0.0, dv + d*nelem*ncomp*Q, Q, data->queue);
    }
  }
  break;

  case CEED_EVAL_WEIGHT: {
    if (tmode == CEED_TRANSPOSE)
      // LCOV_EXCL_START
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // LCOV_EXCL_STOP

    int elemsPerBlock = 1;//basis->Q1d < 7 ? optElems[basis->Q1d] : 1;
    int grid = nelem/elemsPerBlock + ( (nelem/elemsPerBlock*elemsPerBlock<nelem)?
                                       1 : 0 );
    magma_weight_nontensor(grid, nqpt, nelem, nqpt, impl->dqweight, dv, data->queue);
    CeedChk(ierr);
  }
  break;

  // LCOV_EXCL_START
  case CEED_EVAL_DIV:
    return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
  case CEED_EVAL_CURL:
    return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
  case CEED_EVAL_NONE:
    return CeedError(ceed, 1,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // must sync to ensure completeness
  ceed_magma_queue_sync( data->queue );

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &du); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &dv); CeedChk(ierr);
  return 0;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisDestroy_Magma(CeedBasis basis) {
  int ierr;
  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);

  ierr = magma_free(impl->dqref1d); CeedChk(ierr);
  ierr = magma_free(impl->dinterp1d); CeedChk(ierr);
  ierr = magma_free(impl->dgrad1d); CeedChk(ierr);
  ierr = magma_free(impl->dqweight1d); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

#ifdef __cplusplus
CEED_INTERN "C"
#endif
int CeedBasisDestroyNonTensor_Magma(CeedBasis basis) {
  int ierr;
  CeedBasisNonTensor_Magma *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);

  ierr = magma_free(impl->dqref); CeedChk(ierr);
  ierr = magma_free(impl->dinterp); CeedChk(ierr);
  ierr = magma_free(impl->dgrad); CeedChk(ierr);
  ierr = magma_free(impl->dqweight); CeedChk(ierr);

  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
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
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Magma); CeedChk(ierr);

  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

  // Copy qref1d to the GPU
  ierr = magma_malloc((void **)&impl->dqref1d, Q1d*sizeof(qref1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qref1d[0]), qref1d, 1, impl->dqref1d, 1,
                  data->queue);

  // Copy interp1d to the GPU
  ierr = magma_malloc((void **)&impl->dinterp1d, Q1d*P1d*sizeof(interp1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(interp1d[0]), interp1d, 1, impl->dinterp1d, 1,
                  data->queue);

  // Copy grad1d to the GPU
  ierr = magma_malloc((void **)&impl->dgrad1d, Q1d*P1d*sizeof(grad1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(grad1d[0]), grad1d, 1, impl->dgrad1d, 1,
                  data->queue);

  // Copy qweight1d to the GPU
  ierr = magma_malloc((void **)&impl->dqweight1d, Q1d*sizeof(qweight1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qweight1d[0]), qweight1d, 1, impl->dqweight1d, 1,
                  data->queue);

  return 0;
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
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  Ceed_Magma *data;
  ierr = CeedGetData(ceed, (void *)&data); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyNonTensor_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Magma); CeedChk(ierr);

  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

  // Copy qref to the GPU
  ierr = magma_malloc((void **)&impl->dqref, nqpts*sizeof(qref[0]));
  CeedChk(ierr);
  magma_setvector(nqpts, sizeof(qref[0]), qref, 1, impl->dqref, 1, data->queue);

  // Copy interp to the GPU
  ierr = magma_malloc((void **)&impl->dinterp, nqpts*ndof*sizeof(interp[0]));
  CeedChk(ierr);
  magma_setvector(nqpts*ndof, sizeof(interp[0]), interp, 1, impl->dinterp, 1,
                  data->queue);

  // Copy grad to the GPU
  ierr = magma_malloc((void **)&impl->dgrad, nqpts*ndof*dim*sizeof(grad[0]));
  CeedChk(ierr);
  magma_setvector(nqpts*ndof*dim, sizeof(grad[0]), grad, 1, impl->dgrad, 1,
                  data->queue);

  // Copy qweight to the GPU
  ierr = magma_malloc((void **)&impl->dqweight, nqpts*sizeof(qweight[0]));
  CeedChk(ierr);
  magma_setvector(nqpts, sizeof(qweight[0]), qweight, 1, impl->dqweight, 1,
                  data->queue);

  return 0;
}
