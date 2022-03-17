// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <magma.h>
#include "../common/elem_restriction.h"

//////////////////////////////////////////////////////////////////////////////////////////
// ReadDofs to device memory
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                     const magma_int_t esize, const magma_int_t nelem,
                     magma_int_t *offsets, const CeedScalar *du, CeedScalar *dv,
                     magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = MAGMA_ERSTR_THREADS;

    magma_readDofsOffset_kernel<<<grid, threads, 0,
      magma_queue_get_cuda_stream(queue)>>>(NCOMP, compstride,
      esize, nelem, offsets, du, dv);
}

// ReadDofs to device memory, strided description for L-vector
// du is L-vector, size lsize
// dv is E-vector, size nelem * esize * NCOMP
extern "C" void
magma_readDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                      const magma_int_t nelem, const int *strides,
                      const CeedScalar *du, CeedScalar *dv,
                      magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = MAGMA_ERSTR_THREADS;

    magma_readDofsStrided_kernel<<<grid, threads, 0,
      magma_queue_get_cuda_stream(queue)>>>(NCOMP, esize, nelem, 
      strides, du, dv);
}

// WriteDofs from device memory
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize 
extern "C" void
magma_writeDofsOffset(const magma_int_t NCOMP, const magma_int_t compstride,
                      const magma_int_t esize, const magma_int_t nelem,
                      magma_int_t *offsets, const CeedScalar *du, CeedScalar *dv,
                      magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = MAGMA_ERSTR_THREADS;
   
    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magma_writeDofsOffset_kernel_s<<<grid, threads, 0,
        magma_queue_get_cuda_stream(queue)>>>(NCOMP, compstride,
        esize, nelem, offsets, (float*)du, (float*)dv);
    }
    else {
      magma_writeDofsOffset_kernel_d<<<grid, threads, 0,
        magma_queue_get_cuda_stream(queue)>>>(NCOMP, compstride,
        esize, nelem, offsets, (double*)du, (double*)dv);
    }
}

// WriteDofs from device memory, strided description for L-vector
// du is E-vector, size nelem * esize * NCOMP
// dv is L-vector, size lsize
extern "C" void
magma_writeDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                       const magma_int_t nelem, const int *strides,
                       const CeedScalar *du, CeedScalar *dv,
                       magma_queue_t queue)
{
    magma_int_t grid    = nelem;
    magma_int_t threads = MAGMA_ERSTR_THREADS;

    if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
      magma_writeDofsStrided_kernel_s<<<grid, threads, 0,
        magma_queue_get_cuda_stream(queue)>>>(NCOMP, esize, nelem, 
        strides, (float*)du, (float*)dv);
    }
    else {
      magma_writeDofsStrided_kernel_d<<<grid, threads, 0,
        magma_queue_get_cuda_stream(queue)>>>(NCOMP, esize, nelem, 
        strides, (double*)du, (double*)dv);
    }
}
