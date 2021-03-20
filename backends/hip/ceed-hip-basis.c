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

#include <ceed.h>
#include <ceed-backend.h>
#include <hip/hip_runtime.h>
#include "ceed-hip.h"
#include "ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Tensor Basis Kernels
//------------------------------------------------------------------------------
// *INDENT-OFF*
static const char *basiskernels = QUOTE(

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *__restrict__ interp1d,
                                  const CeedScalar *__restrict__ u,
                                  CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[BASIS_Q1D * BASIS_P1D + 2 * BASIS_BUF_LEN];
  CeedScalar *s_interp1d = s_mem;
  CeedScalar *s_buf1 = s_mem + BASIS_Q1D * BASIS_P1D;
  CeedScalar *s_buf2 = s_buf1 + BASIS_BUF_LEN;
  for (CeedInt k = i; k < BASIS_Q1D * BASIS_P1D; k += blockDim.x) {
    s_interp1d[k] = interp1d[k];
  }

  const CeedInt P = transpose ? BASIS_Q1D : BASIS_P1D;
  const CeedInt Q = transpose ? BASIS_P1D : BASIS_Q1D;
  const CeedInt stride0 = transpose ? 1 : BASIS_P1D;
  const CeedInt stride1 = transpose ? BASIS_P1D : 1;
  const CeedInt u_stride = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;
  const CeedInt v_stride = transpose ? BASIS_ELEMSIZE : BASIS_NQPT;
  const CeedInt u_comp_stride = nelem * (transpose ? BASIS_NQPT : BASIS_ELEMSIZE);
  const CeedInt v_comp_stride = nelem * (transpose ? BASIS_ELEMSIZE : BASIS_NQPT);
  const CeedInt u_size = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NCOMP; ++comp) {
      const CeedScalar *cur_u = u + elem * u_stride + comp * u_comp_stride;
      CeedScalar *cur_v = v + elem * v_stride + comp * v_comp_stride;
      for (CeedInt k = i; k < u_size; k += blockDim.x) {
        s_buf1[k] = cur_u[k];
      }
      CeedInt pre = u_size;
      CeedInt post = 1;
      for (CeedInt d = 0; d < BASIS_DIM; d++) {
        __syncthreads();
        // Update buffers used
        pre /= P;
        const CeedScalar *in = d % 2 ? s_buf2 : s_buf1;
        CeedScalar *out = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buf1 : s_buf2);

        // Contract along middle index
        const CeedInt writeLen = pre * post * Q;
        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c = k % post;
          const CeedInt j = (k / post) % Q;
          const CeedInt a = k / (post * Q);

          CeedScalar vk = 0;
          for (CeedInt b = 0; b < P; b++)
            vk += s_interp1d[j*stride0 + b*stride1] * in[(a*P + b)*post + c];

          out[k] = vk;
        }

        post *= Q;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *__restrict__ interp1d,
                                const CeedScalar *__restrict__ grad1d,
                                const CeedScalar *__restrict__ u,
                                CeedScalar *__restrict__ v) {
  const CeedInt i = threadIdx.x;

  __shared__ CeedScalar s_mem[2 * (BASIS_Q1D * BASIS_P1D + BASIS_BUF_LEN)];
  CeedScalar *s_interp1d = s_mem;
  CeedScalar *s_grad1d = s_interp1d + BASIS_Q1D * BASIS_P1D;
  CeedScalar *s_buf1 = s_grad1d + BASIS_Q1D * BASIS_P1D;
  CeedScalar *s_buf2 = s_buf1 + BASIS_BUF_LEN;
  for (CeedInt k = i; k < BASIS_Q1D * BASIS_P1D; k += blockDim.x) {
    s_interp1d[k] = interp1d[k];
    s_grad1d[k] = grad1d[k];
  }

  const CeedInt P = transpose ? BASIS_Q1D : BASIS_P1D;
  const CeedInt Q = transpose ? BASIS_P1D : BASIS_Q1D;
  const CeedInt stride0 = transpose ? 1 : BASIS_P1D;
  const CeedInt stride1 = transpose ? BASIS_P1D : 1;
  const CeedInt u_stride = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;
  const CeedInt v_stride = transpose ? BASIS_ELEMSIZE : BASIS_NQPT;
  const CeedInt u_comp_stride = nelem * (transpose ? BASIS_NQPT : BASIS_ELEMSIZE);
  const CeedInt v_comp_stride = nelem * (transpose ? BASIS_ELEMSIZE : BASIS_NQPT);
  const CeedInt u_dim_stride = transpose ? nelem * BASIS_NQPT * BASIS_NCOMP : 0;
  const CeedInt v_dim_stride = transpose ? 0 : nelem * BASIS_NQPT * BASIS_NCOMP;

  // Apply basis element by element
  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NCOMP; ++comp) {

      // dim*dim contractions for grad
      for (CeedInt dim1 = 0; dim1 < BASIS_DIM; dim1++) {
        CeedInt pre = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;
        CeedInt post = 1;
        const CeedScalar *cur_u = u + elem * u_stride + dim1 * u_dim_stride +
                                  comp * u_comp_stride;
        CeedScalar *cur_v = v + elem * v_stride + dim1 * v_dim_stride + comp *
                            v_comp_stride;
        for (CeedInt dim2 = 0; dim2 < BASIS_DIM; dim2++) {
          __syncthreads();
          // Update buffers used
          pre /= P;
          const CeedScalar *op = dim1 == dim2 ? s_grad1d : s_interp1d;
          const CeedScalar *in = dim2 == 0 ? cur_u : (dim2 % 2 ? s_buf2 : s_buf1);
          CeedScalar *out = dim2 == BASIS_DIM - 1 ? cur_v : (dim2 % 2 ? s_buf1 : s_buf2);

          // Contract along middle index
          const CeedInt writeLen = pre * post * Q;
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c = k % post;
            const CeedInt j = (k / post) % Q;
            const CeedInt a = k / (post * Q);
            CeedScalar vk = 0;
            for (CeedInt b = 0; b < P; b++)
              vk += op[j * stride0 + b * stride1] * in[(a * P + b) * post + c];

            if (transpose && dim2 == BASIS_DIM - 1)
              out[k] += vk;
            else
              out[k] = vk;
          }

          post *= Q;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// 1D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight1d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i)
    w1d[i] = qweight1d[i];

  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x)
    for (int i = 0; i < BASIS_Q1D; ++i) {
      const int ind = e*BASIS_Q1D + i; // sequential
      w[ind] = w1d[i];
    }
}

//------------------------------------------------------------------------------
// 2D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight2d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i)
    w1d[i] = qweight1d[i];

  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x)
    for (int i = 0; i < BASIS_Q1D; ++i)
      for (int j = 0; j < BASIS_Q1D; ++j) {
        const int ind = e*BASIS_Q1D*BASIS_Q1D + i + j*BASIS_Q1D; // sequential
        w[ind] = w1d[i]*w1d[j];
      }
}

//------------------------------------------------------------------------------
// 3D quadrature weights
//------------------------------------------------------------------------------
__device__ void weight3d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i)
    w1d[i] = qweight1d[i];

  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x)
    for (int i = 0; i < BASIS_Q1D; ++i)
      for (int j = 0; j < BASIS_Q1D; ++j)
        for (int k = 0; k < BASIS_Q1D; ++k) {
          const int ind = e*BASIS_Q1D*BASIS_Q1D*BASIS_Q1D + i + j*BASIS_Q1D +
                          k*BASIS_Q1D*BASIS_Q1D; // sequential
          w[ind] = w1d[i]*w1d[j]*w1d[k];
        }
}

//------------------------------------------------------------------------------
// Quadrature weights
//------------------------------------------------------------------------------
extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight1d,
                                  CeedScalar *__restrict__ v) {
  if (BASIS_DIM==1)
    weight1d(nelem, qweight1d, v);
  else if (BASIS_DIM==2)
    weight2d(nelem, qweight1d, v);
  else if (BASIS_DIM==3)
    weight3d(nelem, qweight1d, v);
}

);

//------------------------------------------------------------------------------
// Non-Tensor Basis Kernels
//------------------------------------------------------------------------------
static const char *kernelsNonTensorRef = QUOTE(

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *d_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;

  const double *U;
  double V;
  //TODO load B in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    for (int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) { // run with Q threads
        U = d_U + elem*P + comp*nelem*P;
        V = 0.0;
        for (int i = 0; i < P; ++i)
          V += d_B[i + tid*P]*U[i];

        d_V[elem*Q + comp*nelem*Q + tid] = V;
      } else { // run with P threads
        U = d_U + elem*Q + comp*nelem*Q;
        V = 0.0;
        for (int i = 0; i < Q; ++i)
          V += d_B[tid + i*P]*U[i];

        d_V[elem*P + comp*nelem*P + tid] = V;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *d_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;

  const double *U;
  //TODO load G in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    for (int comp=0; comp<BASIS_NCOMP; comp++) {
      if (!transpose) { // run with Q threads
        double V[BASIS_DIM];
        U = d_U + elem*P + comp*nelem*P;
        for (int dim = 0; dim < BASIS_DIM; dim++)
          V[dim] = 0.0;

        for (int i = 0; i < P; ++i) {
          const double val = U[i];
          for(int dim = 0; dim < BASIS_DIM; dim++)
            V[dim] += d_G[i + tid*P + dim*P*Q]*val;
        }
        for (int dim = 0; dim < BASIS_DIM; dim++) {
          d_V[elem*Q + comp*nelem*Q + dim*BASIS_NCOMP*nelem*Q + tid] = V[dim];
        }
      } else { // run with P threads
        double V = 0.0;
        for (int dim = 0; dim < BASIS_DIM; dim++) {
          U = d_U + elem*Q + comp*nelem*Q +dim*BASIS_NCOMP*nelem*Q;
          for (int i = 0; i < Q; ++i)
            V += d_G[tid + i*P + dim*P*Q]*U[i];
        }
        d_V[elem*P + comp*nelem*P + tid] = V;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Weight
//------------------------------------------------------------------------------
extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight,
                                  CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;
  //TODO load qweight in shared memory if blockDim.z > 1?
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    d_V[elem*Q + tid] = qweight[tid];
  }
}

);
// *INDENT-ON*

//------------------------------------------------------------------------------
// Basis apply - tensor
//------------------------------------------------------------------------------
int CeedBasisApply_Hip(CeedBasis basis, const CeedInt nelem,
                       CeedTransposeMode tmode,
                       CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasis_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const int maxblocksize = 64;

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Hip(ceed,ierr);
  }

  // Basis action
  switch (emode) {
  case CEED_EVAL_INTERP: {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose,
                          &data->d_interp1d, &d_u, &d_v
                         };
    CeedInt Q1d, dim;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChkBackend(ierr);
    ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
    CeedInt blocksize = CeedIntPow(Q1d, dim);
    blocksize = blocksize > maxblocksize ? maxblocksize : blocksize;

    ierr = CeedRunKernelHip(ceed, data->interp, nelem, blocksize, interpargs);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_GRAD: {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d,
                        &data->d_grad1d, &d_u, &d_v
                       };
    CeedInt blocksize = maxblocksize;

    ierr = CeedRunKernelHip(ceed, data->grad, nelem, blocksize, gradargs);
    CeedChkBackend(ierr);
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    const int blocksize = 64;
    int gridsize = nelem/blocksize;
    if (blocksize * gridsize < nelem)
      gridsize += 1;

    ierr = CeedRunKernelHip(ceed, data->weight, gridsize, blocksize,
                            weightargs); CeedChkBackend(ierr);
  } break;
  // LCOV_EXCL_START
  // Evaluate the divergence to/from the quadrature points
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  // Evaluate the curl to/from the quadrature points
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  // Take no action, BasisApply should not have been called
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Basis apply - non-tensor
//------------------------------------------------------------------------------
int CeedBasisApplyNonTensor_Hip(CeedBasis basis, const CeedInt nelem,
                                CeedTransposeMode tmode, CeedEvalMode emode,
                                CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedBasisNonTensor_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);
  CeedInt nnodes, nqpt;
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChkBackend(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  int elemsPerBlock = 1;
  int grid = nelem/elemsPerBlock+((nelem/elemsPerBlock*elemsPerBlock<nelem)?1:0);

  // Read vectors
  const CeedScalar *d_u;
  CeedScalar *d_v;
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr);

  // Clear v for transpose operation
  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChkBackend(ierr);
    ierr = hipMemset(d_v, 0, length * sizeof(CeedScalar));
    CeedChk_Hip(ceed, ierr);
  }

  // Apply basis operation
  switch (emode) {
  case CEED_EVAL_INTERP: {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose,
                          &data->d_interp, &d_u, &d_v
                         };
    if (!transpose) {
      ierr = CeedRunKernelDimHip(ceed, data->interp, grid, nqpt, 1,
                                 elemsPerBlock, interpargs); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimHip(ceed, data->interp, grid, nnodes, 1,
                                 elemsPerBlock, interpargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_GRAD: {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_grad,
                        &d_u, &d_v
                       };
    if (!transpose) {
      ierr = CeedRunKernelDimHip(ceed, data->grad, grid, nqpt, 1,
                                 elemsPerBlock, gradargs); CeedChkBackend(ierr);
    } else {
      ierr = CeedRunKernelDimHip(ceed, data->grad, grid, nnodes, 1,
                                 elemsPerBlock, gradargs); CeedChkBackend(ierr);
    }
  } break;
  case CEED_EVAL_WEIGHT: {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight, &d_v};
    ierr = CeedRunKernelDimHip(ceed, data->weight, grid, nqpt, 1,
                               elemsPerBlock, weightargs); CeedChkBackend(ierr);
  } break;
  // LCOV_EXCL_START
  // Evaluate the divergence to/from the quadrature points
  case CEED_EVAL_DIV:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_DIV not supported");
  // Evaluate the curl to/from the quadrature points
  case CEED_EVAL_CURL:
    return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_CURL not supported");
  // Take no action, BasisApply should not have been called
  case CEED_EVAL_NONE:
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "CEED_EVAL_NONE does not make sense in this context");
    // LCOV_EXCL_STOP
  }

  // Restore vectors
  if (emode != CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChkBackend(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroy_Hip(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasis_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_qweight1d); CeedChk_Hip(ceed,ierr);
  ierr = hipFree(data->d_interp1d); CeedChk_Hip(ceed,ierr);
  ierr = hipFree(data->d_grad1d); CeedChk_Hip(ceed,ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy non-tensor basis
//------------------------------------------------------------------------------
static int CeedBasisDestroyNonTensor_Hip(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);

  CeedBasisNonTensor_Hip *data;
  ierr = CeedBasisGetData(basis, &data); CeedChkBackend(ierr);

  CeedChk_Hip(ceed, hipModuleUnload(data->module));

  ierr = hipFree(data->d_qweight); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_interp); CeedChk_Hip(ceed, ierr);
  ierr = hipFree(data->d_grad); CeedChk_Hip(ceed, ierr);

  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create tensor
//------------------------------------------------------------------------------
int CeedBasisCreateTensorH1_Hip(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasis_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy data to GPU
  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Hip(ceed,ierr);
  ierr = hipMemcpy(data->d_qweight1d, qweight1d, qBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed,ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = hipMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Hip(ceed,ierr);
  ierr = hipMemcpy(data->d_interp1d, interp1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed,ierr);

  ierr = hipMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Hip(ceed,ierr);
  ierr = hipMemcpy(data->d_grad1d, grad1d, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed,ierr);

  // Complie basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedCompileHip(ceed, basiskernels, &data->module, 7,
                        "BASIS_Q1D", Q1d,
                        "BASIS_P1D", P1d,
                        "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                            Q1d : P1d, dim),
                        "BASIS_DIM", dim,
                        "BASIS_NCOMP", ncomp,
                        "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                        "BASIS_NQPT", CeedIntPow(Q1d, dim)
                       ); CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "interp", &data->interp);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "grad", &data->grad);
  CeedChkBackend(ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "weight", &data->weight);
  CeedChkBackend(ierr);
  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create non-tensor
//------------------------------------------------------------------------------
int CeedBasisCreateH1_Hip(CeedElemTopology topo, CeedInt dim, CeedInt nnodes,
                          CeedInt nqpts, const CeedScalar *interp,
                          const CeedScalar *grad, const CeedScalar *qref,
                          const CeedScalar *qweight, CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
  CeedBasisNonTensor_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);

  // Copy basis data to GPU
  const CeedInt qBytes = nqpts * sizeof(CeedScalar);
  ierr = hipMalloc((void **)&data->d_qweight, qBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_qweight, qweight, qBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt iBytes = qBytes * nnodes;
  ierr = hipMalloc((void **)&data->d_interp, iBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_interp, interp, iBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  const CeedInt gBytes = qBytes * nnodes * dim;
  ierr = hipMalloc((void **)&data->d_grad, gBytes); CeedChk_Hip(ceed, ierr);
  ierr = hipMemcpy(data->d_grad, grad, gBytes,
                   hipMemcpyHostToDevice); CeedChk_Hip(ceed, ierr);

  // Compile basis kernels
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChkBackend(ierr);
  ierr = CeedCompileHip(ceed, kernelsNonTensorRef, &data->module, 4,
                        "Q", nqpts,
                        "P", nnodes,
                        "BASIS_DIM", dim,
                        "BASIS_NCOMP", ncomp
                       ); CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "interp", &data->interp);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "grad", &data->grad);
  CeedChk_Hip(ceed, ierr);
  ierr = CeedGetKernelHip(ceed, data->module, "weight", &data->weight);
  CeedChk_Hip(ceed, ierr);

  ierr = CeedBasisSetData(basis, data); CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApplyNonTensor_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
