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

#ifndef ATOMICS_HIP_H
#define ATOMICS_HIP_H


// TODO: Do we need this file anymore?
// Will need changes to work for HIP complex
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
   return atomicAdd(address, val);
}

/******************************************************************************/
__device__ static __inline__ magmaFloatComplex 
magmablas_catomic_add(magmaFloatComplex* address, magmaFloatComplex val)
{
//    float re = magmablas_satomic_add( (float*) (&(*address).x) ,val.x);
//    float im = magmablas_satomic_add( (float*) (&(*address).y) ,val.y);
//    return make_hipFloatComplex(re, im);
}

/******************************************************************************/
__device__ static __inline__ magmaDoubleComplex 
magmablas_zatomic_add(magmaDoubleComplex* address, magmaDoubleComplex val)
{
//    double re = magmablas_datomic_add( (double*) (&(*address).x) ,val.x);
//    double im = magmablas_datomic_add( (double*) (&(*address).y) ,val.y);
//    return make_hipDoubleComplex(re, im);
}

/******************************************************************************/
#endif // ATOMICS_HIP_H
