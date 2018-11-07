// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

// magma functions specific to ceed

#include <ceed-impl.h>
#include "magma.h"

#define USE_MAGMA_BATCH
#define USE_MAGMA_BATCH2
#define USE_MAGMA_BATCH3

void magma_dtensor_contract(Ceed ceed,
                            CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                            const CeedScalar *t, CeedTransposeMode tmode,
                            const CeedInt Add,
                            const CeedScalar *u, CeedScalar *v);

int t30_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int t30_mass( void *ctx, CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int t20_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int t20_mass(void *ctx,  CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int ex1_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int ex1_mass(void *ctx,  CeedInt Q, const CeedScalar *const *in,
             CeedScalar *const *out);
int t400_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);
int t400_mass(void *ctx,  CeedInt Q, const CeedScalar *const *in,
             CeedScalar *const *out);
int t500_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
               CeedScalar *const *out);
int t500_mass(void *ctx,  CeedInt Q, const CeedScalar *const *in,
              CeedScalar *const *out);

#define CeedDebug(...)
//#define CeedDebug(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
