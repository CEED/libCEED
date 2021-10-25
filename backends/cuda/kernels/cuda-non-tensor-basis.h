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

#ifndef _cuda_tensor_non_basis_kernels
#define _cuda_tensor_non_basis_kernels

//------------------------------------------------------------------------------
// Non-Tensor Basis Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void interp(const CeedInt nelem, const int transpose,
                                  const CeedScalar *d_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;

  const CeedScalar *U;
  CeedScalar V;
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

  const CeedScalar *U;
  //TODO load G in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < nelem;
       elem += gridDim.x*blockDim.z) {
    for (int comp=0; comp<BASIS_NCOMP; comp++) {
      if (!transpose) { // run with Q threads
        CeedScalar V[BASIS_DIM];
        U = d_U + elem*P + comp*nelem*P;
        for (int dim = 0; dim < BASIS_DIM; dim++)
          V[dim] = 0.0;

        for (int i = 0; i < P; ++i) {
          const CeedScalar val = U[i];
          for(int dim = 0; dim < BASIS_DIM; dim++)
            V[dim] += d_G[i + tid*P + dim*P*Q]*val;
        }
        for (int dim = 0; dim < BASIS_DIM; dim++) {
          d_V[elem*Q + comp*nelem*Q + dim*BASIS_NCOMP*nelem*Q + tid] = V[dim];
        }
      } else { // run with P threads
        CeedScalar V = 0.0;
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

#endif // _cuda_tensor_non_basis_kernels
