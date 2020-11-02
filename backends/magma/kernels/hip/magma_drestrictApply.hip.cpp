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
#include "hip/hip_runtime.h"
#include <magma_v2.h>
#include "../common/elem_restriction.h"

//////////////////////////////////////////////////////////////////////////////////////////
// ReadDofs to device memory
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                     const magma_int_t esize, const magma_int_t nelem,
                     magma_int_t *offsets, const double *du, double *dv,
                     magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    hipLaunchKernelGGL(magma_readDofsOffset_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), NCOMP, compstride,
      esize, nelem, offsets, du, dv);
}

// ReadDofs to device memory, strided description for L-vector
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                      const magma_int_t nelem, const int *strides,
                      const double *du, double *dv,
                      magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    hipLaunchKernelGGL(magma_readDofsStrided_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), NCOMP, esize, nelem, 
      strides, du, dv);
}

// WriteDofs from device memory
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize 
extern "C" void
magma_writeDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                      const magma_int_t esize, const magma_int_t nelem,
                      magma_int_t *offsets, const double *du, double *dv,
                      magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    hipLaunchKernelGGL(magma_writeDofsOffset_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), NCOMP, compstride,
      esize, nelem, offsets, du, dv);
}

// WriteDofs from device memory, strided description for L-vector
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize
extern "C" void
magma_writeDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                       const magma_int_t nelem, const int *strides,
                       const double *du, double *dv,
                       magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = 256;

    hipLaunchKernelGGL(magma_writeDofsStrided_kernel, dim3(grid), dim3(threads), 0, magma_queue_get_hip_stream(queue), NCOMP, esize, nelem, 
      strides, du, dv);
}
