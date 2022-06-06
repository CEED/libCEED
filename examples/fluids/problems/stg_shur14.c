// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Implementation of the Synthetic Turbulence Generation (STG) algorithm
/// presented in Shur et al. 2014

#include <stdlib.h>
#include <math.h>
#include <petsc.h>
#include "../navierstokes.h"
#include "stg_shur14.h"
#include "../qfunctions/stg_shur14.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

STGShur14Context global_stg_ctx;

/*
 * @brief Perform Cholesky decomposition on array of symmetric 3x3 matrices
 *
 * This assumes the input matrices are in order [11,22,33,12,13,23]. This
 * format is also used for the output.
 *
 * @param[in]  comm   MPI_Comm
 * @param[in]  nprofs Number of matrices in Rij
 * @param[in]  Rij    Array of the symmetric matrices [6,nprofs]
 * @param[out] Cij    Array of the Cholesky Decomposition matrices, [6,nprofs]
 */
PetscErrorCode CalcCholeskyDecomp(MPI_Comm comm, PetscInt nprofs,
                                  const CeedScalar Rij[6][nprofs], CeedScalar Cij[6][nprofs]) {
  PetscFunctionBeginUser;
  for (PetscInt i=0; i<nprofs; i++) {
    Cij[0][i] = sqrt(Rij[0][i]);
    Cij[3][i] = Rij[3][i] / Cij[0][i];
    Cij[1][i] = sqrt(Rij[1][i] - pow(Cij[3][i], 2) );
    Cij[4][i] = Rij[4][i] / Cij[0][i];
    Cij[5][i] = (Rij[5][i] - Cij[3][i]*Cij[4][i]) / Cij[1][i];
    Cij[2][i] = sqrt(Rij[2][i] - pow(Cij[4][i], 2) - pow(Cij[5][i], 2));

    if (isnan(Cij[0][i]) || isnan(Cij[1][i]) || isnan(Cij[2][i]))
      SETERRQ(comm, -1, "Cholesky decomposition failed at profile point %d. "
              "Either STGInflow has non-SPD matrix or contains nan.", i+1);
  }
  PetscFunctionReturn(0);
}


/*
 * @brief Open a PHASTA *.dat file, grabbing dimensions and file pointer
 *
 * This function opens the file specified by `path` using `PetscFOpen` and
 * passes the file pointer in `fp`. It is not closed in this function, thus
 * `fp` must be closed sometime after this function has been called (using
 * `PetscFClose` for example).
 *
 * Assumes that the first line of the file has the number of rows and columns
 * as the only two entries, separated by a single space
 *
 * @param[in] comm MPI_Comm for the program
 * @param[in] path Path to the file
 * @param[in] char_array_len Length of the character array that should contain each line
 * @param[out] dims Dimensions of the file, taken from the first line of the file
 * @param[out] fp File pointer to the opened file
 */
static PetscErrorCode OpenPHASTADatFile(const MPI_Comm comm,
                                        const char path[PETSC_MAX_PATH_LEN], const PetscInt char_array_len,
                                        PetscInt dims[2], FILE **fp) {
  PetscErrorCode ierr;
  PetscInt ndims;
  char line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  ierr = PetscFOpen(comm, path, "r", fp); CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(comm, *fp, char_array_len, line); CHKERRQ(ierr);
  ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
  if (ndims != 2) SETERRQ(comm, -1,
                            "Found %d dimensions instead of 2 on the first line of %s",
                            ndims, path);

  for (PetscInt i=0; i<ndims; i++)  dims[i] = atoi(array[i]);
  ierr = PetscStrToArrayDestroy(ndims, array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * @brief Get the number of rows for the PHASTA file at path
 *
 * Assumes that the first line of the file has the number of rows and columns
 * as the only two entries, separated by a single space
 *
 * @param[in] comm MPI_Comm for the program
 * @param[in] path Path to the file
 * @param[out] nrows Number of rows
 */
static PetscErrorCode GetNRows(const MPI_Comm comm,
                               const char path[PETSC_MAX_PATH_LEN], PetscInt *nrows) {
  PetscErrorCode ierr;
  const PetscInt char_array_len = 512;
  PetscInt dims[2];
  FILE *fp;

  PetscFunctionBeginUser;
  ierr = OpenPHASTADatFile(comm, path, char_array_len, dims, &fp); CHKERRQ(ierr);
  *nrows = dims[0];
  ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * @brief Read the STGInflow file and load the contents into stg_ctx
 *
 * Assumes that the first line of the file has the number of rows and columns
 * as the only two entries, separated by a single space.
 * Assumes there are 14 columns in the file
 *
 * Function calculates the Cholesky decomposition from the Reynolds stress
 * profile found in the file
 *
 * @param[in] comm MPI_Comm for the program
 * @param[in] path Path to the STGInflow.dat file
 * @param[inout] stg_ctx STGShur14Context where the data will be loaded into
 */
static PetscErrorCode ReadSTGInflow(const MPI_Comm comm,
                                    const char path[PETSC_MAX_PATH_LEN], STGShur14Context stg_ctx) {
  PetscErrorCode ierr;
  PetscInt ndims, dims[2];
  FILE *fp;
  const PetscInt char_array_len=512;
  char line[char_array_len];
  char **array;

  PetscFunctionBeginUser;

  ierr = OpenPHASTADatFile(comm, path, char_array_len, dims, &fp); CHKERRQ(ierr);

  CeedScalar rij[6][stg_ctx->nprofs];
  CeedScalar *prof_dw = &stg_ctx->data[stg_ctx->offsets.prof_dw];
  CeedScalar *eps = &stg_ctx->data[stg_ctx->offsets.eps];
  CeedScalar *lt = &stg_ctx->data[stg_ctx->offsets.lt];
  CeedScalar (*ubar)[stg_ctx->nprofs] = (CeedScalar (*)[stg_ctx->nprofs])
                                        &stg_ctx->data[stg_ctx->offsets.ubar];

  for (PetscInt i=0; i<stg_ctx->nprofs; i++) {
    ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
    ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
    if (ndims < dims[1]) SETERRQ(comm, -1,
                                   "Line %d of %s does not contain enough columns (%d instead of %d)", i,
                                   path, ndims, dims[1]);

    prof_dw[i] = (CeedScalar) atof(array[0]);
    ubar[0][i] = (CeedScalar) atof(array[1]);
    ubar[1][i] = (CeedScalar) atof(array[2]);
    ubar[2][i] = (CeedScalar) atof(array[3]);
    rij[0][i]  = (CeedScalar) atof(array[4]);
    rij[1][i]  = (CeedScalar) atof(array[5]);
    rij[2][i]  = (CeedScalar) atof(array[6]);
    rij[3][i]  = (CeedScalar) atof(array[7]);
    rij[4][i]  = (CeedScalar) atof(array[8]);
    rij[5][i]  = (CeedScalar) atof(array[9]);
    lt[i]      = (CeedScalar) atof(array[12]);
    eps[i]     = (CeedScalar) atof(array[13]);

    if (prof_dw[i] < 0) SETERRQ(comm, -1,
                                  "Distance to wall in %s cannot be negative", path);
    if (lt[i] < 0) SETERRQ(comm, -1,
                             "Turbulent length scale in %s cannot be negative", path);
    if (eps[i] < 0) SETERRQ(comm, -1,
                              "Turbulent dissipation in %s cannot be negative", path);

  }
  CeedScalar (*cij)[stg_ctx->nprofs]  = (CeedScalar (*)[stg_ctx->nprofs])
                                        &stg_ctx->data[stg_ctx->offsets.cij];
  ierr = CalcCholeskyDecomp(comm, stg_ctx->nprofs, rij, cij); CHKERRQ(ierr);
  ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * @brief Read the STGRand file and load the contents into stg_ctx
 *
 * Assumes that the first line of the file has the number of rows and columns
 * as the only two entries, separated by a single space.
 * Assumes there are 7 columns in the file
 *
 * @param[in]    comm    MPI_Comm for the program
 * @param[in]    path    Path to the STGRand.dat file
 * @param[inout] stg_ctx STGShur14Context where the data will be loaded into
 */
static PetscErrorCode ReadSTGRand(const MPI_Comm comm,
                                  const char path[PETSC_MAX_PATH_LEN],
                                  STGShur14Context stg_ctx) {
  PetscErrorCode ierr;
  PetscInt ndims, dims[2];
  FILE *fp;
  const PetscInt char_array_len = 512;
  char line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  ierr = OpenPHASTADatFile(comm, path, char_array_len, dims, &fp); CHKERRQ(ierr);

  CeedScalar *phi = &stg_ctx->data[stg_ctx->offsets.phi];
  CeedScalar (*d)[stg_ctx->nmodes]     = (CeedScalar (*)[stg_ctx->nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.d];
  CeedScalar (*sigma)[stg_ctx->nmodes] = (CeedScalar (*)[stg_ctx->nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.sigma];

  for (PetscInt i=0; i<stg_ctx->nmodes; i++) {
    ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
    ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
    if (ndims < dims[1]) SETERRQ(comm, -1,
                                   "Line %d of %s does not contain enough columns (%d instead of %d)", i,
                                   path, ndims, dims[1]);

    d[0][i]     = (CeedScalar) atof(array[0]);
    d[1][i]     = (CeedScalar) atof(array[1]);
    d[2][i]     = (CeedScalar) atof(array[2]);
    phi[i]      = (CeedScalar) atof(array[3]);
    sigma[0][i] = (CeedScalar) atof(array[4]);
    sigma[1][i] = (CeedScalar) atof(array[5]);
    sigma[2][i] = (CeedScalar) atof(array[6]);
  }
  ierr = PetscFClose(comm, fp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * @brief Read STG data from input paths and put in STGShur14Context
 *
 * Reads data from input paths and puts them into a STGShur14Context object.
 * Data stored initially in `*pstg_ctx` will be copied over to the new
 * STGShur14Context instance.
 *
 * @param[in]    comm            MPI_Comm for the program
 * @param[in]    dm              DM for the program
 * @param[in]    stg_inflow_path Path to STGInflow.dat file
 * @param[in]    stg_rand_path   Path to STGRand.dat file
 * @param[inout] pstg_ctx        Pointer to STGShur14Context where the data will be loaded into
 */
PetscErrorCode GetSTGContextData(const MPI_Comm comm, const DM dm,
                                 char stg_inflow_path[PETSC_MAX_PATH_LEN],
                                 char stg_rand_path[PETSC_MAX_PATH_LEN],
                                 STGShur14Context *pstg_ctx,
                                 const CeedScalar ynodes[]) {
  PetscErrorCode ierr;
  PetscInt nmodes, nprofs;
  STGShur14Context stg_ctx;
  PetscFunctionBeginUser;

  // Get options
  ierr = GetNRows(comm, stg_rand_path, &nmodes); CHKERRQ(ierr);
  ierr = GetNRows(comm, stg_inflow_path, &nprofs); CHKERRQ(ierr);
  if (nmodes > STG_NMODES_MAX)
    SETERRQ(comm, 1, "Number of wavemodes in %s (%d) exceeds STG_NMODES_MAX (%d). "
            "Change size of STG_NMODES_MAX and recompile", stg_rand_path, nmodes,
            STG_NMODES_MAX);

  {
    STGShur14Context s;
    ierr = PetscCalloc1(1, &s); CHKERRQ(ierr);
    *s = **pstg_ctx;
    s->nmodes = nmodes;
    s->nprofs = nprofs;
    s->offsets.sigma   = 0;
    s->offsets.d       = nmodes*3;
    s->offsets.phi     = s->offsets.d       + nmodes*3;
    s->offsets.kappa   = s->offsets.phi     + nmodes;
    s->offsets.prof_dw = s->offsets.kappa   + nmodes;
    s->offsets.ubar    = s->offsets.prof_dw + nprofs;
    s->offsets.cij     = s->offsets.ubar    + nprofs*3;
    s->offsets.eps     = s->offsets.cij     + nprofs*6;
    s->offsets.lt      = s->offsets.eps     + nprofs;
    s->offsets.ynodes  = s->offsets.lt      + nprofs;
    PetscInt total_num_scalars = s->offsets.ynodes + s->nynodes;
    s->total_bytes = sizeof(*stg_ctx) + total_num_scalars*sizeof(stg_ctx->data[0]);
    ierr = PetscMalloc(s->total_bytes, &stg_ctx); CHKERRQ(ierr);
    *stg_ctx = *s;
    ierr = PetscFree(s); CHKERRQ(ierr);
  }

  ierr = ReadSTGInflow(comm, stg_inflow_path, stg_ctx); CHKERRQ(ierr);
  ierr = ReadSTGRand(comm, stg_rand_path, stg_ctx); CHKERRQ(ierr);

  if (stg_ctx->nynodes > 0) {
    CeedScalar *ynodes_ctx = &stg_ctx->data[stg_ctx->offsets.ynodes];
    for (PetscInt i=0; i<stg_ctx->nynodes; i++) ynodes_ctx[i] = ynodes[i];
  }

  // -- Calculate kappa
  {
    CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
    CeedScalar *prof_dw = &stg_ctx->data[stg_ctx->offsets.prof_dw];
    CeedScalar *lt = &stg_ctx->data[stg_ctx->offsets.lt];
    CeedScalar le, le_max=0;

    CeedPragmaSIMD
    for (PetscInt i=0; i<stg_ctx->nprofs; i++) {
      le = PetscMin(2*prof_dw[i], 3*lt[i]);
      if (le_max < le) le_max = le;
    }
    CeedScalar kmin = M_PI/le_max;

    CeedPragmaSIMD
    for (PetscInt i=0; i<stg_ctx->nmodes; i++) {
      kappa[i] = kmin*pow(stg_ctx->alpha, i);
    }
  } //end calculate kappa

  ierr = PetscFree(*pstg_ctx); CHKERRQ(ierr);
  *pstg_ctx = stg_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SetupSTG(const MPI_Comm comm, const DM dm, ProblemData *problem,
                        User user, const bool prescribe_T,
                        const CeedScalar theta0, const CeedScalar P0,
                        const CeedScalar ynodes[], const CeedInt nynodes) {
  PetscErrorCode ierr;
  char stg_inflow_path[PETSC_MAX_PATH_LEN] = "./STGInflow.dat";
  char stg_rand_path[PETSC_MAX_PATH_LEN]   = "./STGRand.dat";
  PetscBool  mean_only     = PETSC_FALSE,
             use_stgstrong = PETSC_FALSE;
  CeedScalar u0            = 0.0,
             alpha         = 1.01;
  CeedQFunctionContext stg_context;
  NewtonianIdealGasContext newtonian_ig_ctx;
  PetscFunctionBeginUser;

  // Get options
  PetscOptionsBegin(comm, NULL, "STG Boundary Condition Options", NULL);
  ierr = PetscOptionsString("-stg_inflow_path", "Path to STGInflow.dat", NULL,
                            stg_inflow_path, stg_inflow_path,
                            sizeof(stg_inflow_path), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-stg_rand_path", "Path to STGInflow.dat", NULL,
                            stg_rand_path,stg_rand_path,
                            sizeof(stg_rand_path), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-stg_alpha", "Growth rate of the wavemodes", NULL,
                          alpha, &alpha, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-stg_u0", "Advective velocity for the fluctuations",
                          NULL, u0, &u0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-stg_mean_only", "Only apply mean profile",
                          NULL, mean_only, &mean_only, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-stg_strong", "Enforce STG inflow strongly",
                          NULL, use_stgstrong, &use_stgstrong, NULL); CHKERRQ(ierr);
  PetscOptionsEnd();

  ierr = PetscCalloc1(1, &global_stg_ctx); CHKERRQ(ierr);
  global_stg_ctx->alpha         = alpha;
  global_stg_ctx->u0            = u0;
  global_stg_ctx->is_implicit   = user->phys->implicit;
  global_stg_ctx->prescribe_T   = prescribe_T;
  global_stg_ctx->mean_only     = mean_only;
  global_stg_ctx->theta0        = theta0;
  global_stg_ctx->P0            = P0;
  global_stg_ctx->nynodes       = nynodes;

  {
    // Calculate dx assuming constant spacing
    PetscReal domain_min[3], domain_max[3], domain_size[3];
    ierr = DMGetBoundingBox(dm, domain_min, domain_max); CHKERRQ(ierr);
    for (PetscInt i=0; i<3; i++) domain_size[i] = domain_max[i] - domain_min[i];

    PetscInt nmax = 3, faces[3];
    ierr = PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax,
                                   NULL); CHKERRQ(ierr);
    global_stg_ctx->dx = domain_size[0]/faces[0];
    global_stg_ctx->dz = domain_size[2]/faces[2];
  }

  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context,
                              CEED_MEM_HOST, &newtonian_ig_ctx);
  global_stg_ctx->newtonian_ctx = *newtonian_ig_ctx;
  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context,
                                  &newtonian_ig_ctx);

  ierr = GetSTGContextData(comm, dm, stg_inflow_path, stg_rand_path,
                           &global_stg_ctx, ynodes); CHKERRQ(ierr);

  CeedQFunctionContextDestroy(&problem->apply_inflow.qfunction_context);
  CeedQFunctionContextCreate(user->ceed, &stg_context);
  CeedQFunctionContextSetData(stg_context, CEED_MEM_HOST,
                              CEED_USE_POINTER, global_stg_ctx->total_bytes, global_stg_ctx);
  CeedQFunctionContextSetDataDestroy(stg_context, CEED_MEM_HOST,
                                     FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(stg_context, "solution time",
                                     offsetof(struct STGShur14Context_, time), 1,
                                     "Phyiscal time of the solution");

  if (use_stgstrong) {
    // Use default boundary integral QF (BoundaryIntegral) in newtonian.h
    problem->bc_from_ics                = PETSC_FALSE;
  } else {
    problem->apply_inflow.qfunction         = STGShur14_Inflow;
    problem->apply_inflow.qfunction_loc     = STGShur14_Inflow_loc;
    problem->apply_inflow.qfunction_context = stg_context;
    problem->bc_from_ics                    = PETSC_TRUE;
  }
  // global_stg_ctx = global_stg_ctx;

  PetscFunctionReturn(0);
}

static inline PetscScalar FindDy(const PetscScalar ynodes[],
                                 const PetscInt nynodes, const PetscScalar y) {

  const PetscScalar half_mindy = 0.5 * (ynodes[1] - ynodes[0]);
  // ^^assuming min(dy) is first element off the wall
  PetscInt idx = -1; // Index

  for (PetscInt i=0; i<nynodes; i++) {
    if (y < ynodes[i] + half_mindy) {
      idx = i; break;
    }
  }
  if      (idx == 0)          return ynodes[1] - ynodes[0];
  else if (idx == nynodes-1)  return ynodes[nynodes-2] - ynodes[nynodes-1];
  else                        return 0.5 * (ynodes[idx+1] - ynodes[idx-1]);
}

// Function passed to DMAddBoundary
PetscErrorCode StrongSTGbcFunc(PetscInt dim, PetscReal time,
                               const PetscReal x[], PetscInt Nc, PetscScalar bcval[], void *ctx) {
  PetscFunctionBeginUser;

  const STGShur14Context stg_ctx = (STGShur14Context) ctx;
  PetscScalar qn[stg_ctx->nmodes], u[3], ubar[3], cij[6], eps, lt;
  const bool mean_only      = stg_ctx->mean_only;
  const PetscScalar dx      = stg_ctx->dx;
  const PetscScalar dz      = stg_ctx->dz;
  const PetscScalar mu      = stg_ctx->newtonian_ctx.mu;
  const PetscScalar theta0  = stg_ctx->theta0;
  const PetscScalar P0      = stg_ctx->P0;
  const PetscScalar cv      = stg_ctx->newtonian_ctx.cv;
  const PetscScalar cp      = stg_ctx->newtonian_ctx.cp;
  const PetscScalar Rd      = cp - cv;

  const CeedScalar rho = P0 / (Rd * theta0);
  InterpolateProfile(x[1], ubar, cij, &eps, &lt, stg_ctx);
  if (!mean_only) {
    const PetscInt    nynodes = stg_ctx->nynodes;
    const PetscScalar *ynodes = &stg_ctx->data[stg_ctx->offsets.ynodes];
    const PetscScalar h[3]    = {dx, FindDy(ynodes, nynodes, x[1]), dz};
    CalcSpectrum(x[1], eps, lt, h, mu/rho, qn, stg_ctx);
    STGShur14_Calc(x, time, ubar, cij, qn, u, stg_ctx);
  } else {
    for (CeedInt j=0; j<3; j++) u[j] = ubar[j];
  }

  bcval[0] = rho;
  bcval[1] = rho * u[0];
  bcval[2] = rho * u[1];
  bcval[3] = rho * u[2];
  PetscFunctionReturn(0);
}

PetscErrorCode SetupStrongSTG(DM dm, SimpleBC bc, ProblemData *problem) {
  PetscErrorCode ierr;
  DMLabel label;
  const PetscInt comps[] = {0, 1, 2, 3};
  const PetscInt num_comps = 4;
  PetscFunctionBeginUser;

  ierr = DMGetLabel(dm, "Face Sets", &label); CHKERRQ(ierr);
  // Set wall BCs
  if (bc->num_inflow > 0) {
    ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "STG", label,
                         bc->num_inflow, bc->inflows, 0, num_comps,
                         comps, (void(*)(void))StrongSTGbcFunc,
                         NULL, global_stg_ctx, NULL);  CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
