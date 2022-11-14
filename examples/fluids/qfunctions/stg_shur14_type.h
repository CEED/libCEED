// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef stg_shur14_type_h
#define stg_shur14_type_h

#include <ceed.h>

#include "newtonian_types.h"

/* Access data arrays via:
 *  CeedScalar (*sigma)[ctx->nmodes] = (CeedScalar (*)[ctx->nmodes])&ctx->data[ctx->offsets.sigma];
 *  CeedScalar *eps = &ctx->data[ctx->offsets.eps]; */
typedef struct STGShur14Context_ *STGShur14Context;
struct STGShur14Context_ {
  CeedInt                          nmodes;              // !< Number of wavemodes
  CeedInt                          nprofs;              // !< Number of profile points in STGInflow.dat
  CeedInt                          nynodes;             // !< Number of mesh nodes in the y direction
  CeedScalar                       alpha;               // !< Geometric growth rate of kappa
  CeedScalar                       u0;                  // !< Convective velocity
  CeedScalar                       time;                // !< Solution time
  CeedScalar                       P0;                  // !< Inlet pressure
  CeedScalar                       theta0;              // !< Inlet temperature
  bool                             is_implicit;         // !< Whether using implicit time integration
  bool                             mean_only;           // !< Only apply the mean profile
  CeedScalar                       dx;                  // !< dx used for h calculation
  CeedScalar                       dz;                  // !< dz used for h calculation
  bool                             prescribe_T;         // !< Prescribe temperature weakly
  bool                             use_fluctuating_IC;  // !< Only apply the mean profile
  struct NewtonianIdealGasContext_ newtonian_ctx;

  struct {
    size_t sigma, d, phi;  // !< Random number set, [nmodes,3], [nmodes,3], [nmodes]
    size_t kappa;          // !< Wavemode frequencies in increasing order, [nmodes]
    size_t wall_dist;      // !< Distance to wall for Inflow Profie, [nprof]
    size_t ubar;           // !< Mean velocity, [nprof, 3]
    size_t cij;            // !< Cholesky decomposition [nprof, 6]
    size_t eps;            // !< Turbulent Disspation [nprof, 6]
    size_t lt;             // !< Tubulent Length Scale [nprof, 6]
    size_t ynodes;         // !< Locations of nodes in y direction [nynodes]
  } offsets;               // !< Holds offsets for each array in data
  size_t     total_bytes;  // !< Total size of struct plus array
  CeedScalar data[1];      // !< Holds concatenated scalar array data
};

#endif
