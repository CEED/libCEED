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

#ifndef hip_tensor_basis_kernels
#define hip_tensor_basis_kernels

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
// *INDENT-ON*

#endif // hip_tensor_basis_kernels
