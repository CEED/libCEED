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
// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from L-vector (du) to E-vector (dv):
//
// dv(i, e, c) = du( ind(i, e), c)  
//         or
// dv(i, e, c) = du(i, e, c)
static __global__ void 
magma_readDofs_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                      int *indices, 
                      const double *du, double *dv)
{
  const int  pid = threadIdx.x;
  const int elem = blockIdx.x;
 
  for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            dv[i+elem*esize+comp*esize*nelem] = du[ind + nnodes * comp];
        }
  }
}

// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from L-vector (du) to E-vector (dv), with L-vector in transpose format:
//
// dv(i, e, c) = du(c, ind(i, e))  
//         or
// dv(i, e, c) = du(c, i, e)
template<int TBLOCK, int MAXCOMP>
static __global__ void
magma_readDofsTranspose_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            dv[i+elem*esize+comp*esize*nelem] = du[comp + ind * NCOMP];
        }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from L-vector (du) to E-vector (dv), with strides provided 
//  to describe the L-vector layout
//
// dv(i, e, c) = du( i * strides[0] + c * strides[1] + e * strides[2] )  
static __global__ void 
magma_readDofsStrided_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                      const int *strides, 
                      const double *du, double *dv)
{
  const int  pid = threadIdx.x;
  const int elem = blockIdx.x;
 
  for (CeedInt i = pid; i < esize; i += blockDim.x) {
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            dv[i+elem*esize+comp*esize*nelem] = du[i * strides[0] + 
                                                   comp * strides[1] + 
                                                   elem * strides[2]];
        }
  }
}

// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from E-vector (du) to L-vector (dv):
//
// dv(ind(i, e), c) = du(i, e, c)
//         or
// dv(i, e, c) = du(i, e, c)
static __global__ void 
magma_writeDofs_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                      int *indices, 
                      const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // magmablas_datomic_add(&dv[ind + esize * comp], 
            //                       du[i+comp*esize+elem*NCOMP*esize]);
            magmablas_datomic_add(dv + (ind + nnodes * comp),
                                  du[i+elem*esize+comp*esize*nelem]);
        }
    }
}

// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from E-vector (du) to L-vector (dv), with L-vector in transpose format:
//
// dv(c, ind(i, e)) = du(i, e, c)
//         or
// dv(c, i, e) = du(i, e, c)
template<int TBLOCK, int MAXCOMP>
static __global__ void
magma_writeDofsTranspose_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                               int *indices,
                               const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        const CeedInt ind = indices ? indices[i + elem * esize] : i + elem * esize;
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            magmablas_datomic_add(dv + (comp + ind * NCOMP),
                                  du[i+elem*esize+comp*esize*nelem]);
        }
    }
}

// Fastest index listed first
// i : related to nodes
// e : elements
// c: component
// Go from E-vector (du) to L-vector (dv), with strides provided 
//  to describe the L-vector layout
//
// dv( i * strides[0] + c * strides[1] + e * strides[2] ) = du(i, e, c) 
static __global__ void 
magma_writeDofsStrided_kernel(const int NCOMP, const int nnodes, const int esize, const int nelem,
                      const int *strides, 
                      const double *du, double *dv)
{
    const int  pid = threadIdx.x;
    const int elem = blockIdx.x;

    for (CeedInt i = pid; i < esize; i += blockDim.x) {
        for (CeedInt comp = 0; comp < NCOMP; ++comp) {
            // magmablas_datomic_add(&dv[ind + esize * comp], 
            //                       du[i+comp*esize+elem*NCOMP*esize]);
            magmablas_datomic_add(dv + (i * strides[0] + comp * strides[1] + 
                                        elem * strides[2]),
                                  du[i+elem*esize+comp*esize*nelem]);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

// ReadDofs to device memory
// du is L-vector, size nnodes * NCOMP
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofs(const magma_int_t NCOMP, 
               const magma_int_t nnodes,
               const magma_int_t esize, 
               const magma_int_t nelem, magma_int_t *indices, 
               const double *du, double *dv, 
               magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofs_kernel<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>
    (NCOMP, nnodes, esize, nelem, indices, du, dv);
}

// ReadDofsTranspose to device memory
// du is L-vector (in tranpose format), size nnodes * NCOMP
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsTranspose(const magma_int_t NCOMP,
                        const magma_int_t nnodes,
                        const magma_int_t esize, 
                        const magma_int_t nelem, magma_int_t *indices,
                        const double *du, double *dv, 
                        magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    assert(NCOMP<=4);
    magma_readDofsTranspose_kernel<256,4><<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>
    (NCOMP, nnodes, esize, nelem, indices, du, dv);
}

// ReadDofs to device memory, strided description for L-vector
// du is L-vector, size nnodes * NCOMP
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsStrided(const magma_int_t NCOMP, 
                      const magma_int_t nnodes,
                      const magma_int_t esize, 
                      const magma_int_t nelem, const int *strides, 
	              const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_readDofsStrided_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, esize, nelem, 
                                                             strides, du, dv);
}

// WriteDofs from device memory
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size nnodes * NCOMP 
extern "C" void
magma_writeDofs(const magma_int_t NCOMP, 
                const magma_int_t nnodes, 
                const magma_int_t esize, 
                const magma_int_t nelem, magma_int_t *indices, 
                const double *du, double *dv, 
                magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofs_kernel<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>
    (NCOMP, nnodes, esize, nelem, indices, du, dv);
}

// WriteDofsTranspose from device memory
// du is E-vector (in transpose format), size nelem * esize * NCOMP
// dv is L-vector, size nnodes * NCOMP 
extern "C" void
magma_writeDofsTranspose(const magma_int_t NCOMP,
                         const magma_int_t nnodes,
                         const magma_int_t esize, 
                         const magma_int_t nelem, magma_int_t *indices,
                         const double *du, double *dv, 
                         magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    assert(NCOMP<=4);
    magma_writeDofsTranspose_kernel<256,4><<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>
    (NCOMP, nnodes, esize, nelem, indices, du, dv);
}

// WriteDofs from device memory, strided description for L-vector
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size nnodes * NCOMP 
extern "C" void
magma_writeDofsStrided(const magma_int_t NCOMP, 
                       const magma_int_t nnodes,
                       const magma_int_t esize, 
                       const magma_int_t nelem, const int *strides, 
	               const double *du, double *dv)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    magma_writeDofsStrided_kernel<<<grid, threads, 0, NULL>>>(NCOMP, nnodes, esize, nelem, 
                                                              strides, du, dv);
}
