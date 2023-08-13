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
                                 STGShur14Context *stg_ctx, const CeedScalar ynodes[]) {
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
    temp_ctx->offsets.ynodes    = temp_ctx->offsets.lt + nprofs;
    PetscInt total_num_scalars  = temp_ctx->offsets.ynodes + temp_ctx->nynodes;
    temp_ctx->total_bytes       = sizeof(*temp_ctx) + total_num_scalars * sizeof(temp_ctx->data[0]);
    PetscCall(PetscFree(*stg_ctx));
    PetscCall(PetscMalloc(temp_ctx->total_bytes, stg_ctx));
    **stg_ctx = *temp_ctx;
    PetscCall(PetscFree(temp_ctx));
  }

  PetscCall(ReadSTGInflow(comm, stg_inflow_path, *stg_ctx));
  PetscCall(ReadSTGRand(comm, stg_rand_path, *stg_ctx));

  if ((*stg_ctx)->nynodes > 0) {
    CeedScalar *ynodes_ctx = &(*stg_ctx)->data[(*stg_ctx)->offsets.ynodes];
    for (PetscInt i = 0; i < (*stg_ctx)->nynodes; i++) ynodes_ctx[i] = ynodes[i];
  }

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
                        const CeedScalar P0, const CeedScalar ynodes[], const CeedInt nynodes) {
  Ceed                     ceed                                = user->ceed;
  char                     stg_inflow_path[PETSC_MAX_PATH_LEN] = "./STGInflow.dat";
  char                     stg_rand_path[PETSC_MAX_PATH_LEN]   = "./STGRand.dat";
  PetscBool                mean_only = PETSC_FALSE, use_stgstrong = PETSC_FALSE, use_fluctuating_IC = PETSC_FALSE;
  CeedScalar               u0 = 0.0, alpha = 1.01;
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
  global_stg_ctx->nynodes            = nynodes;

  {
    // Calculate dx assuming constant spacing
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    PetscInt nmax = 3, faces[3];
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
    global_stg_ctx->dx = domain_size[0] / faces[0];
    global_stg_ctx->dz = domain_size[2] / faces[2];
  }

  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx));
  global_stg_ctx->newtonian_ctx = *newtonian_ig_ctx;
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx));

  PetscCall(GetSTGContextData(comm, dm, stg_inflow_path, stg_rand_path, &global_stg_ctx, ynodes));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &stg_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(stg_context, CEED_MEM_HOST, CEED_USE_POINTER, global_stg_ctx->total_bytes, global_stg_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(stg_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(stg_context, "solution time", offsetof(struct STGShur14Context_, time), 1,
                                                         "Physical time of the solution"));

  PetscCallCeed(ceed, CeedQFunctionContextDestroy(&problem->ics.qfunction_context));
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
    PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(stg_context, &problem->apply_inflow.qfunction_context));
    PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(stg_context, &problem->apply_inflow_jacobian.qfunction_context));
    problem->bc_from_ics = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscScalar FindDy(const PetscScalar ynodes[], const PetscInt nynodes, const PetscScalar y) {
  const PetscScalar half_mindy = 0.5 * (ynodes[1] - ynodes[0]);
  // ^^assuming min(dy) is first element off the wall
  PetscInt idx = -1;  // Index

  for (PetscInt i = 0; i < nynodes; i++) {
    if (y < ynodes[i] + half_mindy) {
      idx = i;
      break;
    }
  }
  if (idx == 0) return ynodes[1] - ynodes[0];
  else if (idx == nynodes - 1) return ynodes[nynodes - 2] - ynodes[nynodes - 1];
  else return 0.5 * (ynodes[idx + 1] - ynodes[idx - 1]);
}

// Function passed to DMAddBoundary
// NOTE: Not used in favor of QFunction-based method
PetscErrorCode StrongSTGbcFunc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[], void *ctx) {
  PetscFunctionBeginUser;

  const STGShur14Context stg_ctx = (STGShur14Context)ctx;
  PetscScalar            qn[stg_ctx->nmodes], u[3], ubar[3], cij[6], eps, lt;
  const bool             mean_only = stg_ctx->mean_only;
  const PetscScalar      dx        = stg_ctx->dx;
  const PetscScalar      dz        = stg_ctx->dz;
  const PetscScalar      mu        = stg_ctx->newtonian_ctx.mu;
  const PetscScalar      theta0    = stg_ctx->theta0;
  const PetscScalar      P0        = stg_ctx->P0;
  const PetscScalar      cv        = stg_ctx->newtonian_ctx.cv;
  const PetscScalar      cp        = stg_ctx->newtonian_ctx.cp;
  const PetscScalar      Rd        = cp - cv;

  const CeedScalar rho = P0 / (Rd * theta0);
  InterpolateProfile(x[1], ubar, cij, &eps, &lt, stg_ctx);
  if (!mean_only) {
    const PetscInt     nynodes = stg_ctx->nynodes;
    const PetscScalar *ynodes  = &stg_ctx->data[stg_ctx->offsets.ynodes];
    const PetscScalar  h[3]    = {dx, FindDy(ynodes, nynodes, x[1]), dz};
    CalcSpectrum(x[1], eps, lt, h, mu / rho, qn, stg_ctx);
    STGShur14_Calc(x, time, ubar, cij, qn, u, stg_ctx);
  } else {
    for (CeedInt j = 0; j < 3; j++) u[j] = ubar[j];
  }

  bcval[0] = rho;
  bcval[1] = rho * u[0];
  bcval[2] = rho * u[1];
  bcval[3] = rho * u[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

    case STATEVAR_ENTROPY:
      // {1,2,3,4} for S_momentum, S_energy
      for (int i = 0; i < 4; i++) comps[i] = i + 1;
      break;
  }

  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  // Set wall BCs
  if (bc->num_inflow > 0) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "STG", label, bc->num_inflow, bc->inflows, 0, num_comps, comps, (void (*)(void))StrongSTGbcFunc,
                            NULL, global_stg_ctx, NULL));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongSTG_QF(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt num_comp_q, CeedInt stg_data_size,
                                 CeedInt q_data_size_sur, CeedQFunction *qf_strongbc) {
  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, STGShur14_Inflow_StrongQF, STGShur14_Inflow_StrongQF_loc, qf_strongbc));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "x", num_comp_x, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "scale", 1, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "stg data", stg_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_strongbc, "q", num_comp_q, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_strongbc, problem->ics.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupStrongSTG_PreProcessing(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt stg_data_size, CeedInt q_data_size_sur,
                                            CeedQFunction *qf_strongbc) {
  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, Preprocess_STGShur14, Preprocess_STGShur14_loc, qf_strongbc));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "surface qdata", q_data_size_sur, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddInput(*qf_strongbc, "x", num_comp_x, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(*qf_strongbc, "stg data", stg_data_size, CEED_EVAL_NONE));

  PetscCallCeed(ceed, CeedQFunctionSetContext(*qf_strongbc, problem->ics.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}
