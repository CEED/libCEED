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

/// @file
/// Implementation of the Synthetic Turbulence Generation (STG) algorithm
/// presented in Shur et al. 2014

#include <ceed.h>
#include <petsc.h>

#ifndef stg_shur14_struct
#define stg_shur14_struct

/*
 * Access data arrays via:
 *  CeedScalar (*sigma)[ctx->nmodes] = (CeedScalar (*)[ctx->nmodes])&ctx->data[ctx->offsets.sigma];
 */
typedef struct STGShur14Context_ *STGShur14Context;
struct STGShur14Context_ {
  CeedInt nmodes;   // !< Number of wavemodes
  CeedInt nprofs;   // !< Number of profile points in STGInflow.dat
  CeedScalar alpha; // !< Geometric growth rate of kappa

  struct {
    CeedInt sigma, d, phi; // !< Random number set, [nmodes,3], [nmodes,3], [nmodes]
    CeedInt kappa;   // !< Wavemode frequencies in increasing order, [nmodes]
    CeedInt prof_dw; // !< Distance to wall for Inflow Profie, [nprof]
    CeedInt ubar;    // !< Mean velocity, [nprof, 3]
    CeedInt cij;     // !< Cholesky decomposition [nprof, 6]
    CeedInt eps;     // !< Turbulent Disspation [nprof, 6]
    CeedInt lt;      // !< Tubulent Length Scale [nprof, 6]
  } offsets;       // !< Holds offsets for each array in data
  CeedScalar data[]; //!< Holds concatenated scalar array data
};
#endif


PetscErrorCode SetupSTGContext(STGShur14Context stg_ctx);
