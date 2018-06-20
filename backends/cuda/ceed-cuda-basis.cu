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

#include <ceed-impl.h>
#include <string.h>
#include "ceed-cuda.cuh"

__device__ int powInt(int base, int power) {
  int result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

__device__ void tensorContract(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                               const CeedScalar *t, CeedTransposeMode tmode,
                               const CeedInt Add,
                               const CeedScalar *u, CeedScalar *v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add) {
    for (CeedInt q=0; q<A*J*C; q++) {
      v[q] = (CeedScalar) 0.0;
    }
  }

  const CeedScalar *uP = u;
  for (CeedInt a=0; a<A; a++) {
    for (CeedInt b=0; b<B; b++) {
      CeedScalar *vP = v + a * J * C;
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=0; c<C; c++) {
          *vP += tq * uP[c];
          vP++;
        }
      }
      uP += C;
    }
  }
}

__global__ void interp(const CeedInt nelem, const CeedInt dim, const CeedInt P, const CeedInt Q, const CeedInt bufLen, const CeedInt preStart,
    const CeedInt nqpt, const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *u, CeedScalar *v) {
  const int elem = blockIdx.x*blockDim.x + threadIdx.x;
  if (elem >= nelem) return;

  u += elem * nqpt;

  const CeedInt add = (tmode == CEED_TRANSPOSE);
  CeedInt pre = preStart;
  CeedInt post = 1;

  CeedScalar* tmp1 = new CeedScalar[bufLen];
  CeedScalar* tmp2 = new CeedScalar[bufLen];

  for (CeedInt d = 0; d < dim; d++) {
    CeedScalar *in, *out;
    if (d % 2) {
      in = tmp1;
      out = tmp2;
    } else {
      in = tmp2;
      out = tmp1;
    }
    tensorContract(pre, P, post, Q, interp1d, tmode, add && (d == dim - 1),
        d == 0 ? u : in, d == dim - 1 ? v : out);
    pre /= P;
    post *= Q;
  }

  delete [] tmp1;
  delete [] tmp2;
}

__global__ void grad(const CeedInt nelem, const CeedInt dim, const CeedInt P, const CeedInt Q, const CeedInt bufLen, CeedInt pre,
    const CeedInt nqpt, const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *grad1d, const CeedScalar *u, CeedScalar *v) {
  const int elem = blockIdx.x*blockDim.x + threadIdx.x;
  const int dim1 = blockIdx.y*blockDim.y + threadIdx.y;

  if (elem >= nelem || dim1 >= dim) return;
  
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  CeedInt post = 1;

  CeedScalar* tmp1 = new CeedScalar[bufLen];
  CeedScalar* tmp2 = new CeedScalar[bufLen];
  
  for (CeedInt dim2 = 0; dim2 < dim; dim2++) {
    CeedScalar *in, *out;
    if (dim2 % 2) {
      in = tmp1;
      out = tmp2;
    } else {
      in = tmp2;
      out = tmp1;
    }

    tensorContract(pre, P, post, Q, dim1 == dim2 ? grad1d : interp1d, tmode, add && (dim2 == dim - 1),
        dim2 == 0 ? u : in, dim2 == dim - 1 ? v : out);
    pre /= P;
    post *= Q;
  }

  delete [] tmp1;
  delete [] tmp2;
}

__global__ void weight(const CeedInt dim, const CeedInt Q, const CeedScalar *qweight1d, CeedScalar *v) {
  for (CeedInt d=0; d<dim; d++) {
    CeedInt pre = powInt(Q, dim-d-1), post = powInt(Q, d);
    for (CeedInt i=0; i<pre; i++) {
      for (CeedInt j=0; j<Q; j++) {
        for (CeedInt k=0; k<post; k++) {
          v[(i*Q + j)*post + k] = qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
        }
      }
    }
  }
}

static int CeedBasisApply_Cuda(CeedBasis basis, CeedTransposeMode tmode,
                              CeedEvalMode emode,
                              const CeedScalar *u, CeedScalar *v) {
  return 1;
}

int CeedBasisApplyElems_Cuda(CeedBasis basis, CeedTransposeMode tmode,
                              CeedEvalMode emode,
                              const CeedVector u, CeedVector v) {
  int ierr;
  const Ceed_Cuda* ceed = (Ceed_Cuda*)basis->ceed->data;
  const CeedBasis_Cuda *data = (CeedBasis_Cuda*)basis->data;
  const CeedInt dim = basis->dim;
  const CeedInt nelem = data->er->nelem;
  const CeedInt ndof = basis->ndof;
  const CeedInt nqpt = ndof*CeedPowInt(basis->Q1d, dim);

  const CeedScalar *du = ((CeedVector_Cuda*) u->data)->d_array;
  CeedScalar *dv = ((CeedVector_Cuda*) v->data)->d_array;

  CeedInt P, Q;
  if (tmode == CEED_TRANSPOSE) {
    P = basis->Q1d;
    Q = basis->P1d;
    const CeedInt vsize = ndof*CeedPowInt(basis->P1d, dim);

    ierr = cudaMemset(dv, 0, vsize); CeedChk(ierr);
  } else {
    P = basis->P1d;
    Q = basis->Q1d;
  }
  const CeedInt pre = ndof * CeedPowInt(P, dim - 1);
  const CeedInt bufLen = ndof * Q * CeedPowInt(max(P, Q), dim - 1);

  if (emode & CEED_EVAL_INTERP) {
    run1d(ceed, interp, nelem, dim, P, Q, bufLen, pre, nqpt, tmode, data->d_interp1d, du, dv);
    CeedChk(cudaGetLastError());
  }
  du += nelem * CeedPowInt(basis->Q1d, 2) * ndof * dim;
  if (emode & CEED_EVAL_GRAD) {
    run2d(ceed, grad, nelem, dim, P, Q, bufLen, pre, nqpt, tmode, data->d_interp1d, data->d_grad1d, du, dv);
    CeedChk(cudaGetLastError());
  }
  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    weight<<<1,1>>>(dim, basis->Q1d, data->d_qweight1d, dv);
    CeedChk(cudaGetLastError());
  }
  return 0;
}

static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  return 0;
}

int CeedBasisCreateTensorH1_Cuda(Ceed ceed, CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  int ierr;
  CeedBasis_Cuda *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  basis->data = data;

  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc(&data->d_qweight1d, qBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d, qBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc(&data->d_interp1d, iBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

  ierr = cudaMalloc(&data->d_grad1d, iBytes); CeedChk(ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

  basis->Apply = CeedBasisApply_Cuda;
  basis->Destroy = CeedBasisDestroy_Cuda;
  return 0;
}
