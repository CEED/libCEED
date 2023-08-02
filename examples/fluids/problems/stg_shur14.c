// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Implementation of the Synthetic Turbulence Generation (STG) algorithm
/// presented in Shur et al. 2014

#include "stg_shur14.h"

#include <ceed.h>
#include <math.h>
#include <petscdm.h>
#include <stdlib.h>

#include "../navierstokes.h"
#include "../qfunctions/stg_shur14.h"

STGShur14Context global_stg_ctx;

/*
 * @brief Perform Cholesky decomposition on array of symmetric 3x3 matrices
 *
 * This assumes the input matrices are in order [11,22,33,12,13,23].
 * This format is also used for the output.
 *
 * @param[in]  comm   MPI_Comm
 * @param[in]  nprofs Number of matrices in Rij
 * @param[in]  Rij    Array of the symmetric matrices [6,nprofs]
 * @param[out] Cij    Array of the Cholesky Decomposition matrices, [6,nprofs]
 */
PetscErrorCode CalcCholeskyDecomp(MPI_Comm comm, PetscInt nprofs, const CeedScalar Rij[6][nprofs], CeedScalar Cij[6][nprofs]) {
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < nprofs; i++) {
    Cij[0][i] = sqrt(Rij[0][i]);
    Cij[3][i] = Rij[3][i] / Cij[0][i];
    Cij[1][i] = sqrt(Rij[1][i] - pow(Cij[3][i], 2));
    Cij[4][i] = Rij[4][i] / Cij[0][i];
    Cij[5][i] = (Rij[5][i] - Cij[3][i] * Cij[4][i]) / Cij[1][i];
    Cij[2][i] = sqrt(Rij[2][i] - pow(Cij[4][i], 2) - pow(Cij[5][i], 2));

    PetscCheck(!isnan(Cij[0][i]) && !isnan(Cij[1][i]) && !isnan(Cij[2][i]), comm, PETSC_ERR_FP,
               "Cholesky decomposition failed at profile point %" PetscInt_FMT ". Either STGInflow has non-SPD matrix or contains nan.", i + 1);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Read the STGInflow file and load the contents into stg_ctx
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 * Assumes there are 14 columns in the file.
 *
 * Function calculates the Cholesky decomposition from the Reynolds stress profile found in the file.
 *
 * @param[in]     comm    MPI_Comm for the program
 * @param[in]     path    Path to the STGInflow.dat file
 * @param[in,out] stg_ctx STGShur14Context where the data will be loaded into
 */
static PetscErrorCode ReadSTGInflow(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], STGShur14Context stg_ctx) {
  PetscInt       dims[2];
  int            ndims;
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **array;

  PetscFunctionBeginUser;

  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));

  CeedScalar  rij[6][stg_ctx->nprofs];
  CeedScalar *wall_dist              = &stg_ctx->data[stg_ctx->offsets.wall_dist];
  CeedScalar *eps                    = &stg_ctx->data[stg_ctx->offsets.eps];
  CeedScalar *lt                     = &stg_ctx->data[stg_ctx->offsets.lt];
  CeedScalar(*ubar)[stg_ctx->nprofs] = (CeedScalar(*)[stg_ctx->nprofs]) & stg_ctx->data[stg_ctx->offsets.ubar];

  for (PetscInt i = 0; i < stg_ctx->nprofs; i++) {
    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not have correct number of columns (%d instead of %" PetscInt_FMT ")", i, path, ndims, dims[1]);

    wall_dist[i] = (CeedScalar)atof(array[0]);
    ubar[0][i]   = (CeedScalar)atof(array[1]);
    ubar[1][i]   = (CeedScalar)atof(array[2]);
    ubar[2][i]   = (CeedScalar)atof(array[3]);
    rij[0][i]    = (CeedScalar)atof(array[4]);
    rij[1][i]    = (CeedScalar)atof(array[5]);
    rij[2][i]    = (CeedScalar)atof(array[6]);
    rij[3][i]    = (CeedScalar)atof(array[7]);
    rij[4][i]    = (CeedScalar)atof(array[8]);
    rij[5][i]    = (CeedScalar)atof(array[9]);
    lt[i]        = (CeedScalar)atof(array[12]);
    eps[i]       = (CeedScalar)atof(array[13]);

    PetscCheck(wall_dist[i] >= 0, comm, PETSC_ERR_FILE_UNEXPECTED, "Distance to wall in %s cannot be negative", path);
    PetscCheck(lt[i] >= 0, comm, PETSC_ERR_FILE_UNEXPECTED, "Turbulent length scale in %s cannot be negative", path);
    PetscCheck(eps[i] >= 0, comm, PETSC_ERR_FILE_UNEXPECTED, "Turbulent dissipation in %s cannot be negative", path);
  }
  CeedScalar(*cij)[stg_ctx->nprofs] = (CeedScalar(*)[stg_ctx->nprofs]) & stg_ctx->data[stg_ctx->offsets.cij];
  PetscCall(CalcCholeskyDecomp(comm, stg_ctx->nprofs, rij, cij));
  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Read the STGRand file and load the contents into stg_ctx
 *
 * Assumes that the first line of the file has the number of rows and columns as the only two entries, separated by a single space.
 * Assumes there are 7 columns in the file.
 *
 * @param[in]     comm    MPI_Comm for the program
 * @param[in]     path    Path to the STGRand.dat file
 * @param[in,out] stg_ctx STGShur14Context where the data will be loaded into
 */
static PetscErrorCode ReadSTGRand(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], STGShur14Context stg_ctx) {
  PetscInt       dims[2];
  int            ndims;
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **array;

  PetscFunctionBeginUser;
  PetscCall(PHASTADatFileOpen(comm, path, char_array_len, dims, &fp));

  CeedScalar *phi                     = &stg_ctx->data[stg_ctx->offsets.phi];
  CeedScalar(*d)[stg_ctx->nmodes]     = (CeedScalar(*)[stg_ctx->nmodes]) & stg_ctx->data[stg_ctx->offsets.d];
  CeedScalar(*sigma)[stg_ctx->nmodes] = (CeedScalar(*)[stg_ctx->nmodes]) & stg_ctx->data[stg_ctx->offsets.sigma];

  for (PetscInt i = 0; i < stg_ctx->nmodes; i++) {
    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
    PetscCheck(ndims == dims[1], comm, PETSC_ERR_FILE_UNEXPECTED,
               "Line %" PetscInt_FMT " of %s does not have correct number of columns (%d instead of %" PetscInt_FMT ")", i, path, ndims, dims[1]);

    d[0][i]     = (CeedScalar)atof(array[0]);
    d[1][i]     = (CeedScalar)atof(array[1]);
    d[2][i]     = (CeedScalar)atof(array[2]);
    phi[i]      = (CeedScalar)atof(array[3]);
    sigma[0][i] = (CeedScalar)atof(array[4]);
    sigma[1][i] = (CeedScalar)atof(array[5]);
    sigma[2][i] = (CeedScalar)atof(array[6]);
  }
  PetscCall(PetscFClose(comm, fp));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * @brief Read STG data from input paths and put in STGShur14Context
 *
 * Reads data from input paths and puts them into a STGShur14Context object.
 * Data stored initially in `*stg_ctx` will be copied over to the new STGShur14Context instance.
 *
 * @param[in]     comm            MPI_Comm for the program
 * @param[in]     dm              DM for the program
 * @param[in]     stg_inflow_path Path to STGInflow.dat file
 * @param[in]     stg_rand_path   Path to STGRand.dat file
 * @param[in,out] stg_ctx         Pointer to STGShur14Context where the data will be loaded into
 */
PetscErrorCode GetSTGContextData(const MPI_Comm comm, const DM dm, char stg_inflow_path[PETSC_MAX_PATH_LEN], char stg_rand_path[PETSC_MAX_PATH_LEN],
                                 STGShur14Context *stg_ctx) {
  PetscInt nmodes, nprofs;
  PetscFunctionBeginUser;

  // Get options
  PetscCall(PHASTADatFileGetNRows(comm, stg_rand_path, &nmodes));
  PetscCall(PHASTADatFileGetNRows(comm, stg_inflow_path, &nprofs));
  PetscCheck(nmodes < STG_NMODES_MAX, comm, PETSC_ERR_SUP,
             "Number of wavemodes in %s (%" PetscInt_FMT ") exceeds STG_NMODES_MAX (%d). Change size of STG_NMODES_MAX and recompile", stg_rand_path,
             nmodes, STG_NMODES_MAX);

  {
    STGShur14Context temp_ctx;
    PetscCall(PetscCalloc1(1, &temp_ctx));
    *temp_ctx                   = **stg_ctx;
    temp_ctx->nmodes            = nmodes;
    temp_ctx->nprofs            = nprofs;
    temp_ctx->offsets.sigma     = 0;
    temp_ctx->offsets.d         = nmodes * 3;
    temp_ctx->offsets.phi       = temp_ctx->offsets.d + nmodes * 3;
    temp_ctx->offsets.kappa     = temp_ctx->offsets.phi + nmodes;
    temp_ctx->offsets.wall_dist = temp_ctx->offsets.kappa + nmodes;
    temp_ctx->offsets.ubar      = temp_ctx->offsets.wall_dist + nprofs;
    temp_ctx->offsets.cij       = temp_ctx->offsets.ubar + nprofs * 3;
    temp_ctx->offsets.eps       = temp_ctx->offsets.cij + nprofs * 6;
    temp_ctx->offsets.lt        = temp_ctx->offsets.eps + nprofs;
    PetscInt total_num_scalars  = temp_ctx->offsets.lt + nprofs;
    ;
    temp_ctx->total_bytes = sizeof(*temp_ctx) + total_num_scalars * sizeof(temp_ctx->data[0]);
    PetscCall(PetscFree(*stg_ctx));
    PetscCall(PetscMalloc(temp_ctx->total_bytes, stg_ctx));
    **stg_ctx = *temp_ctx;
    PetscCall(PetscFree(temp_ctx));
  }

  PetscCall(ReadSTGInflow(comm, stg_inflow_path, *stg_ctx));
  PetscCall(ReadSTGRand(comm, stg_rand_path, *stg_ctx));

  {  // -- Calculate kappa
    CeedScalar *kappa     = &(*stg_ctx)->data[(*stg_ctx)->offsets.kappa];
    CeedScalar *wall_dist = &(*stg_ctx)->data[(*stg_ctx)->offsets.wall_dist];
    CeedScalar *lt        = &(*stg_ctx)->data[(*stg_ctx)->offsets.lt];
    CeedScalar  le, le_max = 0;

    CeedPragmaSIMD for (PetscInt i = 0; i < (*stg_ctx)->nprofs; i++) {
      le = PetscMin(2 * wall_dist[i], 3 * lt[i]);
      if (le_max < le) le_max = le;
    }
    CeedScalar kmin = M_PI / le_max;

    CeedPragmaSIMD for (PetscInt i = 0; i < (*stg_ctx)->nmodes; i++) { kappa[i] = kmin * pow((*stg_ctx)->alpha, i); }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupSTG(const MPI_Comm comm, const DM dm, ProblemData *problem, User user, const bool prescribe_T, const CeedScalar theta0,
                        const CeedScalar P0) {
  char                     stg_inflow_path[PETSC_MAX_PATH_LEN] = "./STGInflow.dat";
  char                     stg_rand_path[PETSC_MAX_PATH_LEN]   = "./STGRand.dat";
  PetscBool                mean_only = PETSC_FALSE, use_stgstrong = PETSC_FALSE, use_fluctuating_IC = PETSC_FALSE, given_stg_dx = PETSC_FALSE;
  CeedScalar               u0 = 0.0, alpha = 1.01, stg_dx = 1.0e-3;
  CeedQFunctionContext     stg_context;
  NewtonianIdealGasContext newtonian_ig_ctx;
  PetscFunctionBeginUser;

  // Get options
  PetscOptionsBegin(comm, NULL, "STG Boundary Condition Options", NULL);
  PetscCall(PetscOptionsString("-stg_inflow_path", "Path to STGInflow.dat", NULL, stg_inflow_path, stg_inflow_path, sizeof(stg_inflow_path), NULL));
  PetscCall(PetscOptionsString("-stg_rand_path", "Path to STGInflow.dat", NULL, stg_rand_path, stg_rand_path, sizeof(stg_rand_path), NULL));
  PetscCall(PetscOptionsReal("-stg_alpha", "Growth rate of the wavemodes", NULL, alpha, &alpha, NULL));
  PetscCall(PetscOptionsReal("-stg_u0", "Advective velocity for the fluctuations", NULL, u0, &u0, NULL));
  PetscCall(PetscOptionsBool("-stg_mean_only", "Only apply mean profile", NULL, mean_only, &mean_only, NULL));
  PetscCall(PetscOptionsBool("-stg_strong", "Enforce STG inflow strongly", NULL, use_stgstrong, &use_stgstrong, NULL));
  PetscCall(PetscOptionsBool("-stg_fluctuating_IC", "\"Extrude\" the fluctuations through the domain as an initial condition", NULL,
                             use_fluctuating_IC, &use_fluctuating_IC, NULL));
  PetscCall(PetscOptionsReal("-stg_dx", "Element size in streamwise direction at inflow", NULL, stg_dx, &stg_dx, &given_stg_dx));
  PetscOptionsEnd();

  PetscCall(PetscCalloc1(1, &global_stg_ctx));
  global_stg_ctx->alpha              = alpha;
  global_stg_ctx->u0                 = u0;
  global_stg_ctx->is_implicit        = user->phys->implicit;
  global_stg_ctx->prescribe_T        = prescribe_T;
  global_stg_ctx->mean_only          = mean_only;
  global_stg_ctx->use_fluctuating_IC = use_fluctuating_IC;
  global_stg_ctx->theta0             = theta0;
  global_stg_ctx->P0                 = P0;

  {
    // Calculate dx assuming constant spacing
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    PetscInt nmax = 3, faces[3];
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
    global_stg_ctx->dx = given_stg_dx ? stg_dx : domain_size[0] / faces[0];
  }

  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx);
  global_stg_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx);

  PetscCall(GetSTGContextData(comm, dm, stg_inflow_path, stg_rand_path, &global_stg_ctx));

  CeedQFunctionContextCreate(user->ceed, &stg_context);
  CeedQFunctionContextSetData(stg_context, CEED_MEM_HOST, CEED_USE_POINTER, global_stg_ctx->total_bytes, global_stg_ctx);
  CeedQFunctionContextSetDataDestroy(stg_context, CEED_MEM_HOST, FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(stg_context, "solution time", offsetof(struct STGShur14Context_, time), 1, "Physical time of the solution");

  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction         = ICsSTG;
  problem->ics.qfunction_loc     = ICsSTG_loc;
  problem->ics.qfunction_context = stg_context;

  if (use_stgstrong) {
    // Use default boundary integral QF (BoundaryIntegral) in newtonian.h
    problem->use_strong_bc_ceed = PETSC_TRUE;
    problem->bc_from_ics        = PETSC_FALSE;
  } else {
    problem->apply_inflow.qfunction              = STGShur14_Inflow;
    problem->apply_inflow.qfunction_loc          = STGShur14_Inflow_loc;
    problem->apply_inflow_jacobian.qfunction     = STGShur14_Inflow_Jacobian;
    problem->apply_inflow_jacobian.qfunction_loc = STGShur14_Inflow_Jacobian_loc;
    CeedQFunctionContextReferenceCopy(stg_context, &problem->apply_inflow.qfunction_context);
    CeedQFunctionContextReferenceCopy(stg_context, &problem->apply_inflow_jacobian.qfunction_context);
    problem->bc_from_ics = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// @brief Set STG strongly enforce components using DMAddBoundary
PetscErrorCode SetupStrongSTG(DM dm, SimpleBC bc, ProblemData *problem, Physics phys) {
  DMLabel label;
  PetscFunctionBeginUser;

  PetscInt comps[5], num_comps = 4;
  switch (phys->state_var) {
    case STATEVAR_CONSERVATIVE:
      // {0,1,2,3} for rho, rho*u, rho*v, rho*w
      for (int i = 0; i < 4; i++) comps[i] = i;
      break;

    case STATEVAR_PRIMITIVE:
      // {1,2,3,4} for u, v, w, T
      for (int i = 0; i < 4; i++) comps[i] = i + 1;
      break;
  }

  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  if (bc->num_inflow > 0) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "STG", label, bc->num_inflow, bc->inflows, 0, num_comps, comps, NULL, NULL, NULL, NULL));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongSTG_QF(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt num_comp_q, CeedInt stg_data_size,
                                 CeedInt q_data_size_sur, CeedQFunction *qf_strongbc) {
  PetscFunctionBeginUser;
  CeedQFunctionCreateInterior(ceed, 1, STGShur14_Inflow_StrongQF, STGShur14_Inflow_StrongQF_loc, qf_strongbc);
  CeedQFunctionAddInput(*qf_strongbc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE);
  CeedQFunctionAddInput(*qf_strongbc, "x", num_comp_x, CEED_EVAL_NONE);
  CeedQFunctionAddInput(*qf_strongbc, "scale", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(*qf_strongbc, "stg data", stg_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(*qf_strongbc, "q", num_comp_q, CEED_EVAL_NONE);

  CeedQFunctionSetContext(*qf_strongbc, problem->ics.qfunction_context);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongSTG_PreProcessing(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt stg_data_size, CeedInt q_data_size_sur,
                                            CeedQFunction *qf_strongbc) {
  PetscFunctionBeginUser;
  CeedQFunctionCreateInterior(ceed, 1, Preprocess_STGShur14, Preprocess_STGShur14_loc, qf_strongbc);
  CeedQFunctionAddInput(*qf_strongbc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE);
  CeedQFunctionAddInput(*qf_strongbc, "x", num_comp_x, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(*qf_strongbc, "stg data", stg_data_size, CEED_EVAL_NONE);

  CeedQFunctionSetContext(*qf_strongbc, problem->ics.qfunction_context);
  PetscFunctionReturn(PETSC_SUCCESS);
}
