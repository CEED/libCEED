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

#ifndef ATOMICS_CUH
#define ATOMICS_CUH

#include "magma_internal.h"
/******************************************************************************/
// Atomic adds 
/******************************************************************************/
__device__ static __inline__ float 
magmablas_satomic_add(float* address, float val)
{
    return atomicAdd(address, val);
}

/******************************************************************************/
__device__ static __inline__ double 
magmablas_datomic_add(double* address, double val)
{
#if __CUDA_ARCH__ < 600    // atomic add for double precision is natively supported on sm_60
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#else
    return atomicAdd(address, val);
#endif
}

/******************************************************************************/
__device__ static __inline__ magmaFloatComplex 
magmablas_catomic_add(magmaFloatComplex* address, magmaFloatComplex val)
{
    float re = magmablas_satomic_add( (float*) (&(*address).x) ,val.x);
    float im = magmablas_satomic_add( (float*) (&(*address).y) ,val.y);
    return make_cuFloatComplex(re, im);
}

/******************************************************************************/
__device__ static __inline__ magmaDoubleComplex 
magmablas_zatomic_add(magmaDoubleComplex* address, magmaDoubleComplex val)
{
    double re = magmablas_datomic_add( (double*) (&(*address).x) ,val.x);
    double im = magmablas_datomic_add( (double*) (&(*address).y) ,val.y);
    return make_cuDoubleComplex(re, im);
}

/******************************************************************************/
#endif // ATOMICS_CUH
