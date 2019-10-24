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
#include "atomics.cuh"

//////////////////////////////////////////////////////////////////////////////////////////

// dv(i, c, e) = du( ind(i, e), c)  
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

// dv(i, c, e) = du( c, ind(i,e))
template<int TBLOCK, int MAXCOMP>
static __global__ void
magma_readDofsTranspose_kernel(const int NCOMP, const int nnodes, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    const CeedInt   cb = pid%NCOMP;
    __shared__ CeedScalar    dofs[TBLOCK][MAXCOMP];
    __shared__ CeedInt        ind[TBLOCK];
    for (CeedInt i = pid, k = 0; i < nnodes; i += TBLOCK, k += TBLOCK) {
        ind[pid] = indices ? indices[i + elem * nnodes] : i + elem * nnodes;

        __syncthreads();

        int ncols = (k + TBLOCK < nnodes) ? TBLOCK: nnodes - k;
        for(int j=0, row = pid/NCOMP; j<ncols; j+=TBLOCK/NCOMP)
            if (row < TBLOCK/NCOMP)
                dofs[j+row][cb] = du[cb + ind[j+row] * NCOMP];

        __syncthreads();

        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // dv[i+comp*nnodes+elem*NCOMP*nnodes] = dofs[i][comp];
            dv[i+elem*nnodes+comp*nnodes*nelem] = dofs[i][comp];
        }
    }
}

// dv( ind(i, e), c) = du(i, c, e) 
static __global__ void 
magma_writeDofs_kernel(const int NCOMP, const int nnodes, const int nelem,
                      int *indices, 
                      const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < nnodes; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * nnodes] : i + elem * nnodes;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // magmablas_datomic_add(&dv[ind + nnodes * comp], 
            //                       du[i+comp*nnodes+elem*NCOMP*nnodes]);
            magmablas_datomic_add(&dv[ind + nnodes * comp],
                                  du[i+elem*nnodes+comp*nnodes*nelem]);
        }
    }
}

// dv( c, ind(i,e)) = du(i, c, e)
template<int TBLOCK, int MAXCOMP>
static __global__ void
magma_writeDofsTranspose_kernel(const int NCOMP, const int nnodes, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    CeedInt   cb = pid%NCOMP;
    __shared__ CeedScalar dofs[TBLOCK][MAXCOMP];
    __shared__ CeedInt     ind[TBLOCK];
    for (CeedInt i = pid, k=0; i < nnodes; i += TBLOCK, k+= TBLOCK) {
        ind[pid] = indices ? indices[i + elem * nnodes] : i + elem * nnodes;

        __syncthreads();
        
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            //dofs[i][comp] = du[i+comp*nnodes+elem*NCOMP*nnodes];
            dofs[i][comp] = du[i+elem*nnodes+comp*nnodes*nelem];
        }

        __syncthreads();

        int ncols = (k + TBLOCK < nnodes) ? TBLOCK: nnodes - k;
        for(int j=0, row = pid/NCOMP; j<ncols; j+=TBLOCK/NCOMP)
            if (row < TBLOCK/NCOMP)
                magmablas_datomic_add(&dv[cb + ind[j+row] * NCOMP], dofs[j+row][cb]);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////

// ReadDofs to device memory in tensor dv of size nnodes x NCOMP x nelem
// dv(i, c, e) = du( ind(i, e), c)    
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

// ReadDofsTranspose to device memory in tensor dv of size nnodes x NCOMP x nelem
// dv(i, c, e) = du( c, ind(i,e)) 
extern "C" void
magma_readDofsTranspose(const magma_int_t NCOMP,
                        const magma_int_t nnodes,
                        const magma_int_t nelem, magma_int_t *indices,
                        const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    assert(NCOMP<=4);
    magma_readDofsTranspose_kernel<256,4><<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem,
                                                               indices, du, dv);
}

// WriteDofs 
// dv( ind(i, e), c) = du(i, c, e)
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

// WriteDofsTranspose
// dv( c, ind(i,e)) = du(i, c, e)
extern "C" void
magma_writeDofsTranspose(const magma_int_t NCOMP,
                         const magma_int_t nnodes,
                         const magma_int_t nelem, magma_int_t *indices,
                         const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    assert(NCOMP<=4);
    magma_writeDofsTranspose_kernel<256,4><<<grid, threads, 0, NULL>>>(NCOMP, nnodes, nelem,
                                                                       indices, du, dv);
}
