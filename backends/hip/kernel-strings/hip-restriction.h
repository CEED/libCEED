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

#ifndef hip_restriction_kernels
#define hip_restriction_kernels

//------------------------------------------------------------------------------
// ElemRestriction Kernels
//------------------------------------------------------------------------------
// *INDENT-OFF*
static const char *restrictionkernels = QUOTE(

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrStrided(const CeedInt nelem,
                                       const CeedScalar *__restrict__ u,
                                       CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
        elem*RESTRICTION_ELEMSIZE] =
          u[locNode*STRIDE_NODES + comp*STRIDE_COMP + elem*STRIDE_ELEM];
  }
}

//------------------------------------------------------------------------------
// L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void noTrOffset(const CeedInt nelem,
                                      const CeedInt *__restrict__ indices,
                                      const CeedScalar *__restrict__ u,
                                      CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt ind = indices[node];
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
        elem*RESTRICTION_ELEMSIZE] =
          u[ind + comp*RESTRICTION_COMPSTRIDE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void trStrided(const CeedInt nelem,
    const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x;
      node < nelem*RESTRICTION_ELEMSIZE;
      node += blockDim.x * gridDim.x) {
    const CeedInt locNode = node % RESTRICTION_ELEMSIZE;
    const CeedInt elem = node / RESTRICTION_ELEMSIZE;

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[locNode*STRIDE_NODES + comp*STRIDE_COMP + elem*STRIDE_ELEM] +=
          u[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
            elem*RESTRICTION_ELEMSIZE];
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
extern "C" __global__ void trOffset(const CeedInt *__restrict__ lvec_indices,
                                    const CeedInt *__restrict__ tindices,
                                    const CeedInt *__restrict__ toffsets,
                                    const CeedScalar *__restrict__ u,
                                    CeedScalar *__restrict__ v) {
  CeedScalar value[RESTRICTION_NCOMP];

  for (CeedInt i = blockIdx.x * blockDim.x + threadIdx.x;
       i < RESTRICTION_NNODES;
       i += blockDim.x * gridDim.x) {
    const CeedInt ind = lvec_indices[i];
    const CeedInt rng1 = toffsets[i];
    const CeedInt rngN = toffsets[i+1];

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      value[comp] = 0.0;

    for (CeedInt j = rng1; j < rngN; ++j) {
      const CeedInt tind = tindices[j];
      CeedInt locNode = tind % RESTRICTION_ELEMSIZE;
      CeedInt elem = tind / RESTRICTION_ELEMSIZE;

      for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
        value[comp] += u[locNode + comp*RESTRICTION_ELEMSIZE*RESTRICTION_NELEM +
                         elem*RESTRICTION_ELEMSIZE];
    }

    for (CeedInt comp = 0; comp < RESTRICTION_NCOMP; ++comp)
      v[ind + comp*RESTRICTION_COMPSTRIDE] += value[comp];
  }
}

);
// *INDENT-ON*

#endif // hip_restriction_kernels
