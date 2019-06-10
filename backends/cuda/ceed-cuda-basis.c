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
#include <ceed.h>
#include "ceed-cuda.h"

static const char *basiskernels = QUOTE(
                                    extern "C" __global__ void interpInterleaved(const CeedInt nelem,
                                        const int transpose,
                                        const CeedScalar *__restrict__ interp1d, const CeedScalar *__restrict__ u,
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
  const CeedInt u_stride = BASIS_NCOMP * (transpose ? BASIS_NQPT :
                                          BASIS_ELEMSIZE);
  const CeedInt v_stride = BASIS_NCOMP * (transpose ? BASIS_ELEMSIZE :
                                          BASIS_NQPT);

  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    const CeedScalar *cur_u = u + elem * u_stride;
    CeedScalar *cur_v = v + elem * v_stride;
    for (CeedInt k = i; k < u_stride; k += blockDim.x) {
      s_buf1[k] = cur_u[k];
    }

    CeedInt pre = u_stride;
    CeedInt post = 1;
    for (CeedInt d = 0; d < BASIS_DIM; d++) {
      __syncthreads();

      pre /= P;
      const CeedScalar *in = d % 2 ? s_buf2 : s_buf1;
      CeedScalar *out = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buf1 : s_buf2);

      const CeedInt writeLen = pre * post * Q;
      for (CeedInt k = i; k < writeLen; k += blockDim.x) {
        const CeedInt c = k % post;
        const CeedInt j = (k / post) % Q;
        const CeedInt a = k / (post * Q);
        CeedScalar vk = 0;
        for (CeedInt b = 0; b < P; b++) {
          vk += s_interp1d[j * stride0 + b * stride1] * in[(a * P + b) * post + c];
        }

        out[k] = vk;
      }

      post *= Q;
    }
  }
}

extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *__restrict__ interp1d, const CeedScalar *__restrict__ u,
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
  const CeedInt u_stride = transpose ? BASIS_NQPT : BASIS_NCOMP * BASIS_ELEMSIZE;
  const CeedInt v_stride = transpose ? BASIS_NCOMP * BASIS_ELEMSIZE : BASIS_NQPT;
  const CeedInt u_comp_stride = transpose ? nelem * BASIS_NQPT : BASIS_ELEMSIZE;
  const CeedInt v_comp_stride = transpose ? BASIS_ELEMSIZE : nelem * BASIS_NQPT;
  const CeedInt u_size = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;

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
        pre /= P;
        const CeedScalar *in = d % 2 ? s_buf2 : s_buf1;
        CeedScalar *out = d == BASIS_DIM - 1 ? cur_v : (d % 2 ? s_buf1 : s_buf2);

        const CeedInt writeLen = pre * post * Q;
        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c = k % post;
          const CeedInt j = (k / post) % Q;
          const CeedInt a = k / (post * Q);
          CeedScalar vk = 0;
          for (CeedInt b = 0; b < P; b++) {
            vk += s_interp1d[j * stride0 + b * stride1] * in[(a * P + b) * post + c];
          }

          out[k] = vk;
        }

        post *= Q;
      }
    }
  }
}

extern "C" __global__ void gradInterleaved(const CeedInt nelem,
    const int transpose,
    const CeedScalar *__restrict__ interp1d,
    const CeedScalar *__restrict__ grad1d, const CeedScalar *__restrict__ u,
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
  const CeedInt u_stride = BASIS_NCOMP * (transpose ? BASIS_NQPT *BASIS_DIM :
                                          BASIS_ELEMSIZE);
  const CeedInt v_stride = BASIS_NCOMP * (transpose ? BASIS_ELEMSIZE : BASIS_NQPT
                                          * BASIS_DIM);

  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    const CeedScalar *cur_u = u + elem * u_stride;
    CeedScalar *cur_v = v + elem * v_stride;

    for (CeedInt dim1 = 0; dim1 < BASIS_DIM; dim1++) {
      CeedInt pre = BASIS_NCOMP * (transpose ? BASIS_NQPT : BASIS_ELEMSIZE);
      CeedInt post = 1;
      for (CeedInt dim2 = 0; dim2 < BASIS_DIM; dim2++) {
        __syncthreads();

        pre /= P;
        const CeedScalar *op = dim1 == dim2 ? s_grad1d : s_interp1d;
        const CeedScalar *in = dim2 == 0 ? cur_u : (dim2 % 2 ? s_buf2 : s_buf1);
        CeedScalar *out = dim2 == BASIS_DIM - 1 ? cur_v : (dim2 % 2 ? s_buf1 : s_buf2);

        const CeedInt writeLen = pre * post * Q;
        for (CeedInt k = i; k < writeLen; k += blockDim.x) {
          const CeedInt c = k % post;
          const CeedInt j = (k / post) % Q;
          const CeedInt a = k / (post * Q);
          CeedScalar vk = 0;
          for (CeedInt b = 0; b < P; b++) {
            vk += op[j * stride0 + b * stride1] * in[(a * P + b) * post + c];
          }

          if (transpose && dim2 == BASIS_DIM - 1)
            out[k] += vk;
          else
            out[k] = vk;
        }

        post *= Q;
      }
      if (transpose) {
        cur_u += BASIS_NQPT * BASIS_NCOMP;
      } else {
        cur_v += BASIS_NQPT * BASIS_NCOMP;
      }
    }
  }
}

extern "C" __global__ void grad(const CeedInt nelem, const int transpose,
                                const CeedScalar *__restrict__ interp1d,
                                const CeedScalar *__restrict__ grad1d, const CeedScalar *__restrict__ u,
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
  const CeedInt u_stride = transpose ? BASIS_NQPT : BASIS_NCOMP * BASIS_ELEMSIZE;
  const CeedInt v_stride = transpose ? BASIS_NCOMP * BASIS_ELEMSIZE : BASIS_NQPT;
  const CeedInt u_comp_stride = transpose ? nelem * BASIS_NQPT : BASIS_ELEMSIZE;
  const CeedInt v_comp_stride = transpose ? BASIS_ELEMSIZE : nelem * BASIS_NQPT;
  const CeedInt u_dim_stride = transpose ? nelem * BASIS_NQPT * BASIS_NCOMP : 0;
  const CeedInt v_dim_stride = transpose ? 0 : nelem * BASIS_NQPT * BASIS_NCOMP;

  for (CeedInt elem = blockIdx.x; elem < nelem; elem += gridDim.x) {
    for (CeedInt comp = 0; comp < BASIS_NCOMP; ++comp) {

      for (CeedInt dim1 = 0; dim1 < BASIS_DIM; dim1++) {
        CeedInt pre = transpose ? BASIS_NQPT : BASIS_ELEMSIZE;
        CeedInt post = 1;
        const CeedScalar *cur_u = u + elem * u_stride + dim1 * u_dim_stride + comp *
                                  u_comp_stride;
        CeedScalar *cur_v = v + elem * v_stride + dim1 * v_dim_stride + comp *
                            v_comp_stride;
        for (CeedInt dim2 = 0; dim2 < BASIS_DIM; dim2++) {
          __syncthreads();

          pre /= P;
          const CeedScalar *op = dim1 == dim2 ? s_grad1d : s_interp1d;
          const CeedScalar *in = dim2 == 0 ? cur_u : (dim2 % 2 ? s_buf2 : s_buf1);
          CeedScalar *out = dim2 == BASIS_DIM - 1 ? cur_v : (dim2 % 2 ? s_buf1 : s_buf2);

          const CeedInt writeLen = pre * post * Q;
          for (CeedInt k = i; k < writeLen; k += blockDim.x) {
            const CeedInt c = k % post;
            const CeedInt j = (k / post) % Q;
            const CeedInt a = k / (post * Q);
            CeedScalar vk = 0;
            for (CeedInt b = 0; b < P; b++) {
              vk += op[j * stride0 + b * stride1] * in[(a * P + b) * post + c];
            }

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

// extern "C" __global__ void weight(const CeedScalar * __restrict__ qweight1d, CeedScalar * __restrict__ v) {
//   CeedInt pre = BASIS_NQPT;
//   CeedInt post = 1;
//   for (CeedInt d=0; d<BASIS_DIM; d++) {
//     pre /= BASIS_Q1D;
//     for (CeedInt i=0; i<pre; i++) {
//       for (CeedInt j=0; j<BASIS_Q1D; j++) {
//         for (CeedInt k=0; k<post; k++) {
//           v[(i*BASIS_Q1D + j)*post + k] = qweight1d[j] * (d == 0 ? 1 : v[(i*BASIS_Q1D + j)*post + k]);
//         }
//       }
//     }
//     post *= BASIS_Q1D;
//   }
// }
// extern "C" __global__ void weight(const CeedInt nelem,
//                                   const CeedScalar * __restrict__ qweight1d, CeedScalar * __restrict__ v) {
//   CeedScalar r_w = qweight1d[threadIdx.x%BASIS_Q1D];
//   for (int q = blockIdx.x * blockDim.x + threadIdx.x;
//        q < nelem*BASIS_DIM*BASIS_Q1D;
//        q += blockDim.x * gridDim.x) {
//     if (threadIdx.x < BASIS_DIM*BASIS_Q1D) v[q] = r_w;
//   }
// }

__device__ void weight1d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < BASIS_Q1D; ++i) {
      const int ind = e*BASIS_Q1D + i;//sequential
      w[ind] = w1d[i];
    }
  }
}

__device__ void weight2d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < BASIS_Q1D; ++i) {
      for (int j = 0; j < BASIS_Q1D; ++j) {
        const int ind = e*BASIS_Q1D*BASIS_Q1D + i + j*BASIS_Q1D;//sequential
        w[ind] = w1d[i]*w1d[j];
      }
    }
  }
}

__device__ void weight3d(const CeedInt nelem, const CeedScalar *qweight1d,
                         CeedScalar *w) {
  CeedScalar w1d[BASIS_Q1D];
  for (int i = 0; i < BASIS_Q1D; ++i) {
    w1d[i] = qweight1d[i];
  }
  for (int e = blockIdx.x * blockDim.x + threadIdx.x;
       e < nelem;
       e += blockDim.x * gridDim.x) {
    for (int i = 0; i < BASIS_Q1D; ++i) {
      for (int j = 0; j < BASIS_Q1D; ++j) {
        for (int k = 0; k < BASIS_Q1D; ++k) {
          const int ind = e*BASIS_Q1D*BASIS_Q1D*BASIS_Q1D + i + j*BASIS_Q1D +
                          k*BASIS_Q1D*BASIS_Q1D;//sequential
          w[ind] = w1d[i]*w1d[j]*w1d[k];
        }
      }
    }
  }
}

extern "C" __global__ void weight(const CeedInt nelem,
                                  const CeedScalar *__restrict__ qweight1d, CeedScalar *__restrict__ v) {
  if (BASIS_DIM==1) {
    weight1d(nelem, qweight1d, v);
  } else if (BASIS_DIM==2) {
    weight2d(nelem, qweight1d, v);
  } else if (BASIS_DIM==3) {
    weight3d(nelem, qweight1d, v);
  }
}

                                  );

int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt nelem,
                        CeedTransposeMode tmode,
                        CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_Cuda *ceed_Cuda;
  CeedGetData(ceed, (void *) &ceed_Cuda); CeedChk(ierr);
  CeedBasis_Cuda *data;
  CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const int maxblocksize = 32;//ceed_Cuda->optblocksize;

  const CeedScalar *d_u;
  CeedScalar *d_v;
  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChk(ierr);

  if (tmode == CEED_TRANSPOSE) {
    CeedInt length;
    ierr = CeedVectorGetLength(v, &length); CeedChk(ierr);
    ierr = cudaMemset(d_v, 0, length * sizeof(CeedScalar)); CeedChk_Cu(ceed,ierr);
  }
  if (emode == CEED_EVAL_INTERP) {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d, &d_u, &d_v};
    CeedInt Q1d, dim;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
    CeedInt blocksize = CeedIntPow(Q1d, dim);
    blocksize = blocksize > maxblocksize ? maxblocksize : blocksize;
    ierr = run_kernel(ceed, data->interp, nelem, blocksize, interpargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_GRAD) {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d, &data->d_grad1d, &d_u, &d_v};
    CeedInt blocksize = maxblocksize;
    ierr = run_kernel(ceed, data->grad, nelem, blocksize, gradargs); CeedChk(ierr);
  } else if (emode == CEED_EVAL_WEIGHT) {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    const int blocksize = 32;
    int gridsize = nelem/blocksize;
    if (blocksize * gridsize < nelem)
      gridsize += 1;
    ierr = run_kernel(ceed, data->weight, gridsize, blocksize, weightargs);
    CeedChk(ierr);
  }

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, &d_u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(v, &d_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_Cuda(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  CeedBasis_Cuda *data;
  ierr = CeedBasisGetData(basis, (void *) &data); CeedChk(ierr);

  CeedChk_Cu(ceed, cuModuleUnload(data->module));

  ierr = cudaFree(data->d_qweight1d); CeedChk_Cu(ceed,ierr);
  ierr = cudaFree(data->d_interp1d); CeedChk_Cu(ceed,ierr);
  ierr = cudaFree(data->d_grad1d); CeedChk_Cu(ceed,ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);

  const CeedInt qBytes = Q1d * sizeof(CeedScalar);
  ierr = cudaMalloc((void **)&data->d_qweight1d, qBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_qweight1d, qweight1d, qBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  const CeedInt iBytes = qBytes * P1d;
  ierr = cudaMalloc((void **)&data->d_interp1d, iBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_interp1d, interp1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  ierr = cudaMalloc((void **)&data->d_grad1d, iBytes); CeedChk_Cu(ceed,ierr);
  ierr = cudaMemcpy(data->d_grad1d, grad1d, iBytes,
                    cudaMemcpyHostToDevice); CeedChk_Cu(ceed,ierr);

  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = compile(ceed, basiskernels, &data->module, 7,
                 "BASIS_Q1D", Q1d,
                 "BASIS_P1D", P1d,
                 "BASIS_BUF_LEN", ncomp * CeedIntPow(Q1d > P1d ?
                     Q1d : P1d, dim),
                 "BASIS_DIM", dim,
                 "BASIS_NCOMP", ncomp,
                 "BASIS_ELEMSIZE", CeedIntPow(P1d, dim),
                 "BASIS_NQPT", CeedIntPow(Q1d, dim)
                ); CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, "interp", &data->interp);
  CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, "grad", &data->grad);
  CeedChk(ierr);
  ierr = get_kernel(ceed, data->module, "weight", &data->weight);
  CeedChk(ierr);

  ierr = CeedBasisSetData(basis, (void *)&data);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Cuda);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Cuda);
  CeedChk(ierr);
  return 0;
}

int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim,
                           CeedInt ndof, CeedInt nqpts,
                           const CeedScalar *interp,
                           const CeedScalar *grad,
                           const CeedScalar *qref,
                           const CeedScalar *qweight,
                           CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement generic H1 basis");
}
