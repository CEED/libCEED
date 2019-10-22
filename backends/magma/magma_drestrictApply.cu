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
#include <magma.h>

//////////////////////////////////////////////////////////////////////////////////////////

static __global__ void 
magma_readDofs_kernel(const int NCOMP, const int nnodes, const int nelem,
                      int *indices, 
                      const double *du, double *dv)
{
  const int  pid = threadIdx.x;
  const int elem = blockIdx.x;
 
  for (CeedInt i = pid; i < nnodes; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * nnodes] : i + elem * nnodes;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // dv[i+comp*nnodes+elem*NCOMP*nnodes] = du[ind + nnodes * comp];
            dv[i+elem*nnodes+comp*nnodes*nelem] = du[ind + nnodes * comp];
        }
  }
}

static __global__ void
magma_readDofsTranspose_kernel(const int NCOMP, const int nnodes, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    CeedInt   cb = pid%NCOMP;
    CeedInt   tb = blockDim.x;
    __shared__ CeedScalar dofs[tb][NCOMP];
    __shared__ const CeedInt  ind[nnodes];
    for (CeedInt i = pid; i < nnodes; i += tb) {
        ind[i] = indices ? indices[i + elem * nnodes] : i + elem * nnodes;

        __syncthreads();

        for (CeedInt j = i/NCOMP; j<min(tb, nnodes); j+=NCOMP)
            dofs[j][cb] = du[cb + ind[j] * NCOMP];

        __syncthreads();

        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // dv[i+comp*nnodes+elem*NCOMP*nnodes] = dofs[i][comp];
            dv[i+elem*nnodes+comp*nnodes*nelem] = dofs[i][comp];
        }
    }
}

static __global__ void 
magma_writeDofs_kernel(const int NCOMP, const int nnodes, const int nelem,
                      int *indices, 
                      const double *du, double *dv)
{
// fill kernal here

}

static __global__ void
magma_writeDofsTranspose_kernel(const int NCOMP, const int nnodes, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{

 //fill kernel here

}


//////////////////////////////////////////////////////////////////////////////////////////

// ReadDofs 
extern "C" void
magma_readDofs(const magma_int_t NCOMP, 
               const magma_int_t nnodes, 
               const magma_int_t nelem, magma_int_t *indices, 
	       const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofs_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem, 
                                                            indices, du, dv);
}

// NonTensor weight function
extern "C" void
magma_readDofsTranspose(const magma_int_t NCOMP,
                        const magma_int_t nnodes,
                        const magma_int_t nelem, magma_int_t *indices,
                        const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofsTranspose_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem,
                                                                     indices, du, dv);
}

// WriteDofs 
extern "C" void
magma_writeDofs(const magma_int_t NCOMP, 
                const magma_int_t nnodes, 
                const magma_int_t nelem, magma_int_t *indices, 
	        const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofs_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem, 
                                                            indices, du, dv);
}

// NonTensor weight function
extern "C" void
magma_writeDofsTranspose(const magma_int_t NCOMP,
                         const magma_int_t nnodes,
                         const magma_int_t nelem, magma_int_t *indices,
                         const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofsTranspose_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem,
                                                                     indices, du, dv);
}
