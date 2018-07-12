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

__device__ void tensorContract(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
    const CeedScalar *t, CeedTransposeMode tmode,
    const CeedInt Add,
    const CeedScalar *u, CeedScalar *v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add) {
    const CeedInt zeroLen = A * J * C;
    for (CeedInt q=0; q< zeroLen; q++) {
      v[q] = (CeedScalar) 0.0;
    }
  }

  for (CeedInt a=0; a<A; a++) {
    for (CeedInt b=0; b<B; b++) {
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
        }
      }
    }
  }
}

__global__ void interp(const CeedInt nelem, const CeedInt dim, const CeedInt ndof, const CeedInt elemsize, const CeedInt P, const CeedInt Q, const CeedInt nqpt, const CeedInt bufLen,
    const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *u, CeedScalar *v, CeedScalar *tmp1, CeedScalar *tmp2) {
  const int elem = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ CeedScalar s_interp1d[];

  for (int i = threadIdx.x; i < P * Q; i+=blockDim.x) {
    s_interp1d[i] = interp1d[i];
  }

  if (elem >= nelem) return;

  CeedInt add;
  if (tmode == CEED_TRANSPOSE) {
    add = true;
    u += nqpt * ndof * elem;
    v += elem * ndof * elemsize;
  } else {
    add = false;
    u += elem * ndof * elemsize;
    v += nqpt * ndof * elem;
  }

  const CeedInt tmpOffset = elem * bufLen;
  tmp1 += tmpOffset;
  tmp2 += tmpOffset;

  CeedInt pre = ndof * CeedPowInt(P, dim - 1);
  CeedInt post = 1;

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
}

__global__ void grad(const CeedInt nelem, const CeedInt dim, const CeedInt ndof, const CeedInt elemsize, const CeedInt P, const CeedInt Q, const CeedInt nqpt, const CeedInt bufLen,
    const CeedTransposeMode tmode, const CeedScalar *interp1d, const CeedScalar *grad1d, const CeedScalar *u, CeedScalar *v, CeedScalar *tmp1, CeedScalar *tmp2) {
  const int elem = blockIdx.x*blockDim.x + threadIdx.x;

  extern __shared__ CeedScalar s_interp1d[];
  CeedScalar *s_grad1d = s_interp1d + P * Q;

  for (int i = threadIdx.x; i < P * Q; i += blockDim.x) {
    s_interp1d[i] = interp1d[i];
    s_grad1d[i] = grad1d[i];
  }

  if (elem >= nelem) return;

  CeedInt add;
  if (tmode == CEED_TRANSPOSE) {
    add = true;
    u += nqpt * ndof * elem * dim;
    v += elem * ndof * elemsize;
  } else {
    add = false;
    u += elem * ndof * elemsize;
    v += nqpt * ndof * elem * dim;
  }

  const CeedInt tmpOffset = elem * bufLen;
  tmp1 += tmpOffset;
  tmp2 += tmpOffset;

  for (CeedInt dim1 = 0; dim1 < dim; dim1++) {
    CeedInt pre = ndof * CeedPowInt(P, dim - 1);
    CeedInt post = 1;

    for (CeedInt dim2 = 0; dim2 < dim; dim2++) {
      CeedScalar *in, *out;
      if (dim2 % 2) {
        in = tmp1;
        out = tmp2;
      } else {
        in = tmp2;
        out = tmp1;
      }

      tensorContract(pre, P, post, Q, dim1 == dim2 ? s_grad1d : s_interp1d, tmode, add && (dim2 == dim - 1),
          dim2 == 0 ? u : in, dim2 == dim - 1 ? v : out);
      pre /= P;
      post *= Q;
    }

    if (tmode == CEED_TRANSPOSE) {
      u += nqpt * ndof;
    } else {
      v += nqpt * ndof;
    }
  }
}

__global__ void weight(const CeedInt dim, const CeedInt Q, const CeedScalar *qweight1d, CeedScalar *v) {
  for (CeedInt d=0; d<dim; d++) {
    CeedInt pre = CeedPowInt(Q, dim-d-1), post = CeedPowInt(Q, d);
    for (CeedInt i=0; i<pre; i++) {
      for (CeedInt j=0; j<Q; j++) {
        for (CeedInt k=0; k<post; k++) {
          v[(i*Q + j)*post + k] = qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
        }
      }
    }
  }
}

int CeedBasisApplyElems_Cuda(CeedBasis basis, const CeedInt nelem, CeedTransposeMode tmode,
    CeedEvalMode emode, const CeedVector u, CeedVector v) {
  int ierr;
  const Ceed_Cuda* ceed = (Ceed_Cuda*)basis->ceed->data;
  CeedBasis_Cuda *data = (CeedBasis_Cuda*)basis->data;
  const CeedInt dim = basis->dim;
  const CeedInt elemsize = CeedPowInt(basis->P1d, dim);
  const CeedInt ndof = basis->ndof;
  const CeedInt nqpt = CeedPowInt(basis->Q1d, dim);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const CeedInt P = transpose?basis->Q1d:basis->P1d;
  const CeedInt Q = transpose?basis->P1d:basis->Q1d;
  const CeedInt bufLen = ndof * Q * CeedPowInt(std::max(P, Q), dim-1);

  if (!data->ready) {
    data->ready = true;
    const CeedInt qBytes = basis->Q1d * sizeof(CeedScalar);
    ierr = cudaMalloc(&data->d_qweight1d, qBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_qweight1d, basis->qweight1d, qBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

    const CeedInt iBytes = qBytes * basis->P1d;
    ierr = cudaMalloc(&data->d_interp1d, iBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_interp1d, basis->interp1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

    ierr = cudaMalloc(&data->d_grad1d, iBytes); CeedChk(ierr);
    ierr = cudaMemcpy(data->d_grad1d, basis->grad1d, iBytes, cudaMemcpyHostToDevice); CeedChk(ierr);

    const CeedInt tBytes = bufLen * nelem * sizeof(CeedScalar);
    ierr = cudaMalloc(&data->d_tmp1, tBytes); CeedChk(ierr);
    ierr = cudaMalloc(&data->d_tmp2, tBytes); CeedChk(ierr);
  }

  const CeedScalar *d_u;
  CeedScalar *d_v;
  CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u);
  CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v);

  if (tmode == CEED_TRANSPOSE) {
    ierr = cudaMemset(d_v, 0, v->length * sizeof(CeedScalar)); CeedChk(ierr);
  }
  if (emode & CEED_EVAL_INTERP) {
    ierr = run1d(ceed, interp, P * Q * sizeof(CeedScalar),
        nelem, dim, ndof, elemsize, P, Q, nqpt, bufLen, tmode,
        data->d_interp1d, d_u, d_v, data->d_tmp1, data->d_tmp2); CeedChk(ierr);

    if (transpose) {
      d_u += nelem * nqpt * ndof;
    } else {
      d_v += nelem * nqpt * ndof;
    }
  }

  if (emode & CEED_EVAL_GRAD) {
    ierr = run1d(ceed, grad, 2 * P * Q * sizeof(CeedScalar),
        nelem, dim, ndof, elemsize, P, Q, nqpt, bufLen, tmode,
        data->d_interp1d, data->d_grad1d, d_u, d_v, data->d_tmp1, data->d_tmp2); CeedChk(ierr);

    if (transpose) {
      d_u += nelem * nqpt * ndof * dim;
    } else {
      d_v += nelem * nqpt * ndof * dim;
    }
  }

  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(basis->ceed, 1,
          "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    weight<<<1,1>>>(dim, basis->Q1d, data->d_qweight1d, d_v);
    ierr = cudaGetLastError(); CeedChk(ierr);
  }

  return 0;
}

static int CeedBasisApply_Cuda(CeedBasis basis, CeedTransposeMode tmode,
    CeedEvalMode emode,
    const CeedScalar *u, CeedScalar *v) {
  int ierr;

  CeedVector tmp_u, tmp_v;
  const CeedInt ndof = basis->ndof;
  const CeedInt dim = basis->dim;
  const CeedInt nqpt = CeedPowInt(basis->Q1d, dim);
  const CeedInt nppt = CeedPowInt(basis->P1d, dim);
  const CeedInt esize = (emode & CEED_EVAL_INTERP ? nqpt * ndof : 0) + (emode & CEED_EVAL_GRAD ? nqpt * ndof * dim : 0) + (emode & CEED_EVAL_WEIGHT ? nqpt : 0);
  const CeedInt usize = tmode == CEED_TRANSPOSE ? esize : nppt*ndof;
  const CeedInt vsize = tmode == CEED_TRANSPOSE ? nppt*ndof : esize;
  ierr = CeedVectorCreate(basis->ceed, usize, &tmp_u); CeedChk(ierr);
  ierr = CeedVectorCreate(basis->ceed, vsize, &tmp_v); CeedChk(ierr);

  if (!(emode & CEED_EVAL_WEIGHT)) {
    ierr = CeedVectorSetArray(tmp_u, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)u); CeedChk(ierr);
  }
  ierr = CeedVectorSetArray(tmp_v, CEED_MEM_HOST, CEED_USE_POINTER, v); CeedChk(ierr);

  CeedBasisApplyElems_Cuda(basis, 1, tmode, emode, tmp_u, tmp_v);

  ierr = CeedVectorGetArray(tmp_v, CEED_MEM_HOST, &v); CeedChk(ierr);

  ierr = CeedVectorDestroy(&tmp_u); CeedChk(ierr);
  ierr = CeedVectorDestroy(&tmp_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  int ierr;

  CeedBasis_Cuda *data = (CeedBasis_Cuda *) basis->data;

  if (data->ready) {
    ierr = cudaFree(data->d_qweight1d); CeedChk(ierr);
    ierr = cudaFree(data->d_interp1d); CeedChk(ierr);
    ierr = cudaFree(data->d_grad1d); CeedChk(ierr);
    ierr = cudaFree(data->d_tmp1); CeedChk(ierr);
    ierr = cudaFree(data->d_tmp2); CeedChk(ierr);
  }

  ierr = CeedFree(&data); CeedChk(ierr);

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
  data->ready = false;

  basis->Apply = CeedBasisApply_Cuda;
  basis->Destroy = CeedBasisDestroy_Cuda;
  return 0;
}
