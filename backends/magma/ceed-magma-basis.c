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
extern "C"
#endif
int CeedBasisApply_Magma(CeedBasis basis, CeedInt nelem, 
			 CeedTransposeMode tmode, CeedEvalMode emode,
			 CeedVector U, CeedVector V) 
{
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedInt dim, ncomp, ndof, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChk(ierr);
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_DEVICE, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_DEVICE, &v); CeedChk(ierr);

  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, (void*)&impl); CeedChk(ierr);

  CeedInt P1d, Q1d;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);

  CeedDebug("\033[01m[CeedBasisApply_Magma] vsize=%d, comp = %d",
            ncomp*CeedIntPow(P1d, dim), ncomp);

  if (tmode == CEED_TRANSPOSE) {
     CeedInt length;
     ierr = CeedVectorGetLength(V, &length);
     magmablas_dlaset(MagmaFull, length, 1, 0., 0., v, length );
  }
  if (emode & CEED_EVAL_INTERP) {
    CeedInt P = P1d, Q = Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d; Q = P1d;
    }

    int elquadsize = ncomp*CeedIntPow(Q, dim);
    int eldofssize = ncomp*CeedIntPow(P, dim);
    magmablas_dbasis_apply_batched_eval_interp(P, Q, dim, ncomp,
					       impl->dinterp1d, tmode, 
					       u, eldofssize, 
					       v, elquadsize, 
					       nelem);  
  }
  if (emode & CEED_EVAL_GRAD) {
    CeedInt P = P1d, Q = Q1d;
    // In CEED_NOTRANSPOSE mode:
    // u is (P^dim x nc), column-major layout (nc = ncomp)
    // v is (Q^dim x nc x dim), column-major layout (nc = ncomp)
    // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d, Q = P1d;
    }

    int elquadsize = ncomp*CeedIntPow(Q, dim);
    int eldofssize = ncomp*CeedIntPow(P, dim);
    magmablas_dbasis_apply_batched_eval_grad(P, Q, dim, ncomp, nqpt,                                    
					     impl->dinterp1d, impl->dgrad1d, tmode, 
					     u, eldofssize, 
					     v, elquadsize, 
					     nelem);       
   }
  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q = Q1d;
    int eldofssize = ncomp*CeedIntPow(Q, dim);
    magmablas_dbasis_apply_batched_eval_weight(Q, dim, impl->dqweight1d, 
					       v, eldofssize, 
					       nelem);
  }

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int CeedBasisDestroy_Magma(CeedBasis basis) 
{
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
extern "C"
#endif
int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis) 
{
  int ierr;
  CeedBasis_Magma *impl;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Magma); CeedChk(ierr);

  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

  // Copy qref1d to the GPU
  ierr = magma_malloc((void**)&impl->dqref1d, Q1d*sizeof(qref1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qref1d[0]), qref1d, 1, impl->dqref1d, 1);
  
  // Copy interp1d to the GPU
  ierr = magma_malloc((void**)&impl->dinterp1d, Q1d*P1d*sizeof(interp1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(interp1d[0]), interp1d, 1, impl->dinterp1d, 1);

  // Copy grad1d to the GPU
  ierr = magma_malloc((void**)&impl->dgrad1d, Q1d*P1d*sizeof(grad1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(grad1d[0]), grad1d, 1, impl->dgrad1d, 1);

  // Copy qweight1d to the GPU
  ierr = magma_malloc((void**)&impl->dqweight1d, Q1d*sizeof(qweight1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qweight1d[0]), qweight1d, 1, impl->dqweight1d, 1);

  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim,
                                   CeedInt ndof, CeedInt nqpts,
                                   const CeedScalar *interp,
                                   const CeedScalar *grad,
                                   const CeedScalar *qref,
                                   const CeedScalar *qweight,
                                   CeedBasis basis) 
{
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  return CeedError(ceed, 1, "Backend does not implement non-tensor bases");
}
