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

#include <ceed/ceed.h>

//------------------------------------------------------------------------------
// Non-Tensor Basis Kernels
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Interp
//------------------------------------------------------------------------------
extern "C" __global__ void Interp(const CeedInt num_elem, const int transpose,
                                  const CeedScalar *d_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;

  const CeedScalar *U;
  CeedScalar V;
  //TODO load B in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for (int comp = 0; comp < BASIS_NCOMP; comp++) {
      if (!transpose) { // run with Q threads
        U = d_U + elem*P + comp*num_elem*P;
        V = 0.0;
        for (int i = 0; i < P; ++i)
          V += d_B[i + tid*P]*U[i];

        d_V[elem*Q + comp*num_elem*Q + tid] = V;
      } else { // run with P threads
        U = d_U + elem*Q + comp*num_elem*Q;
        V = 0.0;
        for (int i = 0; i < Q; ++i)
          V += d_B[tid + i*P]*U[i];

        d_V[elem*P + comp*num_elem*P + tid] = V;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const int transpose,
                                const CeedScalar *d_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;

  const CeedScalar *U;
  //TODO load G in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for (int comp=0; comp<BASIS_NCOMP; comp++) {
      if (!transpose) { // run with Q threads
        CeedScalar V[BASIS_DIM];
        U = d_U + elem*P + comp*num_elem*P;
        for (int dim = 0; dim < BASIS_DIM; dim++)
          V[dim] = 0.0;

        for (int i = 0; i < P; ++i) {
          const CeedScalar val = U[i];
          for(int dim = 0; dim < BASIS_DIM; dim++)
            V[dim] += d_G[i + tid*P + dim*P*Q]*val;
        }
        for (int dim = 0; dim < BASIS_DIM; dim++) {
          d_V[elem*Q + comp*num_elem*Q + dim*BASIS_NCOMP*num_elem*Q + tid] = V[dim];
        }
      } else { // run with P threads
        CeedScalar V = 0.0;
        for (int dim = 0; dim < BASIS_DIM; dim++) {
          U = d_U + elem*Q + comp*num_elem*Q +dim*BASIS_NCOMP*num_elem*Q;
          for (int i = 0; i < Q; ++i)
            V += d_G[tid + i*P + dim*P*Q]*U[i];
        }
        d_V[elem*P + comp*num_elem*P + tid] = V;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Weight
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem,
                                  const CeedScalar *__restrict__ q_weight,
                                  CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;
  //TODO load q_weight in shared memory if blockDim.z > 1?
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    d_V[elem*Q + tid] = q_weight[tid];
  }
}

//------------------------------------------------------------------------------
