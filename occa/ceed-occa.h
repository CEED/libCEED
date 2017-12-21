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

// *****************************************************************************
#include <string.h>
#include <assert.h>
#include <stdbool.h>

// *****************************************************************************
#include <ceed-dbg.h>
#include <ceed-impl.h>

// *****************************************************************************
#include "occa.h"

// *****************************************************************************
#define NO_OFFSET 0
#define NO_PROPS occaDefault
#define TILE_SIZE 32
extern occaDevice device;


// *****************************************************************************
// * CeedVectorOcca struct
// *****************************************************************************
typedef struct {
  CeedScalar* host;
  occaMemory* device;
} CeedVectorOcca;


// *****************************************************************************
// * CeedElemRestrictionOcca struct
// *****************************************************************************
typedef struct {
  const CeedInt* host;
  occaMemory* device;
  occaKernel kRestrict;
} CeedElemRestrictionOcca;


// **[ basis ] *****************************************************************
int CeedBasisCreateTensorH1Occa(Ceed ceed, CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar* interp1d,
                                const CeedScalar* grad1d,
                                const CeedScalar* qref1d,
                                const CeedScalar* qweight1d,
                                CeedBasis basis);

// **[ operator ]***************************************************************
int CeedOperatorCreateOcca(CeedOperator op);

// **[ qfunction ]**************************************************************
int CeedQFunctionCreateOcca(CeedQFunction qf);

// **[ restriction ]************************************************************
int CeedElemRestrictionCreateOcca(const CeedElemRestriction res,
                                  const CeedMemType mtype,
                                  const CeedCopyMode cmode,
                                  const CeedInt* indices);
int CeedTensorContractOcca(Ceed ceed,
                           CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                           const CeedScalar* t, CeedTransposeMode tmode,
                           const CeedScalar* u, CeedScalar* v);

// **[ vector ] ****************************************************************
int CeedVectorCreateOcca(Ceed ceed, CeedInt n, CeedVector vec);

