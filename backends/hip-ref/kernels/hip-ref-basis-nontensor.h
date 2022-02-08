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
extern "C" __global__ void Interp(const CeedInt num_elem, const CeedInt transpose,
                                  const CeedScalar *d_B,
                                  const CeedScalar *__restrict__ d_U,
                                  CeedScalar *__restrict__ d_V) {
  const CeedInt t_id = threadIdx.x;

  const CeedScalar *U;
  CeedScalar V;
  //TODO load B in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      if (transpose) { // run with P threads
        U = d_U + elem*BASIS_Q + comp*num_elem*BASIS_Q;
        V = 0.0;
        for (CeedInt i = 0; i < BASIS_Q; i++)
          V += d_B[t_id + i*BASIS_P]*U[i];

        d_V[elem*BASIS_P + comp*num_elem*BASIS_P + t_id] = V;
      } else { // run with Q threads
        U = d_U + elem*BASIS_P + comp*num_elem*BASIS_P;
        V = 0.0;
        for (CeedInt i = 0; i < BASIS_P; i++)
          V += d_B[i + t_id*BASIS_P]*U[i];

        d_V[elem*BASIS_Q + comp*num_elem*BASIS_Q + t_id] = V;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Grad
//------------------------------------------------------------------------------
extern "C" __global__ void Grad(const CeedInt num_elem, const CeedInt transpose,
                                const CeedScalar *d_G,
                                const CeedScalar *__restrict__ d_U,
                                CeedScalar *__restrict__ d_V) {
  const CeedInt t_id = threadIdx.x;

  const CeedScalar *U;
  //TODO load G in shared memory if blockDim.z > 1?

  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    for (CeedInt comp = 0; comp < BASIS_NUM_COMP; comp++) {
      if (transpose) { // run with P threads
        CeedScalar V = 0.0;
        for (CeedInt dim = 0; dim < BASIS_DIM; dim++) {
          U = d_U + elem*BASIS_Q + comp*num_elem*BASIS_Q +
              dim*BASIS_NUM_COMP*num_elem*BASIS_Q;
          for (CeedInt i = 0; i < BASIS_Q; i++)
            V += d_G[t_id + i*BASIS_P + dim*BASIS_P*BASIS_Q]*U[i];
        }
        d_V[elem*BASIS_P + comp*num_elem*BASIS_P + t_id] = V;
      } else { // run with Q threads
        CeedScalar V[BASIS_DIM];
        U = d_U + elem*BASIS_P + comp*num_elem*BASIS_P;
        for (CeedInt dim = 0; dim < BASIS_DIM; dim++)
          V[dim] = 0.0;

        for (CeedInt i = 0; i < BASIS_P; i++) {
          const CeedScalar val = U[i];
          for(CeedInt dim = 0; dim < BASIS_DIM; dim++)
            V[dim] += d_G[i + t_id*BASIS_P + dim*BASIS_P*BASIS_Q]*val;
        }
        for (CeedInt dim = 0; dim < BASIS_DIM; dim++) {
          d_V[elem*BASIS_Q + comp*num_elem*BASIS_Q +
              dim*BASIS_NUM_COMP*num_elem*BASIS_Q + t_id] = V[dim];
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Weight
//------------------------------------------------------------------------------
extern "C" __global__ void Weight(const CeedInt num_elem,
                                  const CeedScalar *__restrict__ qweight,
                                  CeedScalar *__restrict__ d_V) {
  const CeedInt t_id = threadIdx.x;
  //TODO load qweight in shared memory if blockDim.z > 1?
  for (CeedInt elem = blockIdx.x*blockDim.z + threadIdx.z; elem < num_elem;
       elem += gridDim.x*blockDim.z) {
    d_V[elem*BASIS_Q + t_id] = qweight[t_id];
  }
}

//------------------------------------------------------------------------------
