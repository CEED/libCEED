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
#include "../qfunctions/stg_shur14.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

/*
 * @brief Perform Cholesky decomposition on array of symmetric 3x3 matrices
 *
 * This assumes the input matrices are in order [11,22,33,12,13,23]. This
 * format is also used for the output.
 *
 * @param[in] nprofs Number of matrices in Rij
 * @param[in] Rij Array of the symmetric matrices [6,nprofs]
 * @param[out] Cij Array of the Cholesky Decomposition matrices, [6,nprofs]
 */
static void CalcCholeskyDecomp(int nprofs,
                               const CeedScalar Rij[6][nprofs], CeedScalar Cij[6][nprofs]) {

  CeedPragmaSIMD
  for(int i=0; i<nprofs; i++) {
    Cij[0][i] = sqrt(Rij[0][i]);
    Cij[3][i] = Rij[3][i] / Cij[0][i];
    Cij[1][i] = sqrt(Rij[1][i] - pow(Cij[3][i], 2) );
    Cij[4][i] = Rij[4][i] / Cij[0][i];
    Cij[5][i] = (Rij[5][i] - Cij[3][i]*Cij[4][i]) / Cij[1][i];
    Cij[2][i] = sqrt(Rij[2][i] - pow(Cij[4][i], 2) - pow(Cij[5][i], 2));
  }
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
                                        const char path[PETSC_MAX_PATH_LEN], const int char_array_len, int dims[2],
                                        FILE **fp) {
  PetscErrorCode ierr;
  int ndims;
  char line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  ierr = PetscFOpen(comm, path, "r", fp); CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(comm, *fp, char_array_len, line); CHKERRQ(ierr);
  ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
  if(ndims != 2) SETERRQ(comm, -1,
                           "Found %d dimensions instead of 2 on the first line of %s",
                           ndims, path);

  for (int i=0; i<ndims; i++)  dims[i] = atoi(array[i]);
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
                               const char path[PETSC_MAX_PATH_LEN], int *nrows) {
  PetscErrorCode ierr;
  const int char_array_len = 512;
  int dims[2];
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
  int ndims, dims[2];
  FILE *fp;
  const int char_array_len=512;
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

  for (int i=0; i<stg_ctx->nprofs; i++) {
    ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
    ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
    if(ndims < dims[1]) SETERRQ(comm, -1,
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
    eps[i]     = (CeedScalar) atof(array[12]);
    lt[i]      = (CeedScalar) atof(array[13]);

    if(prof_dw[i] < 0) SETERRQ(comm, -1,
                                 "Distance to wall in %s cannot be negative", path);
    if(lt[i] < 0) SETERRQ(comm, -1,
                            "Turbulent length scale in %s cannot be negative", path);
    if(eps[i] < 0) SETERRQ(comm, -1,
                             "Turbulent dissipation in %s cannot be negative", path);

    CeedScalar (*cij)[stg_ctx->nprofs]  = (CeedScalar (*)[stg_ctx->nprofs])
                                          &stg_ctx->data[stg_ctx->offsets.cij];
    CalcCholeskyDecomp(stg_ctx->nprofs, rij, cij);
  }
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
 * @param[in] comm MPI_Comm for the program
 * @param[in] path Path to the STGRand.dat file
 * @param[inout] stg_ctx STGShur14Context where the data will be loaded into
 */
static PetscErrorCode ReadSTGRand(const MPI_Comm comm,
                                  const char path[PETSC_MAX_PATH_LEN],
                                  STGShur14Context stg_ctx) {

  PetscErrorCode ierr;
  int ndims, dims[2];
  FILE *fp;
  const int char_array_len = 512;
  char line[char_array_len];
  char **array;

  PetscFunctionBeginUser;
  ierr = OpenPHASTADatFile(comm, path, char_array_len, dims, &fp); CHKERRQ(ierr);

  CeedScalar *phi = &stg_ctx->data[stg_ctx->offsets.phi];
  CeedScalar (*d)[stg_ctx->nmodes]     = (CeedScalar (*)[stg_ctx->nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.d];
  CeedScalar (*sigma)[stg_ctx->nmodes] = (CeedScalar (*)[stg_ctx->nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.sigma];

  for (int i=0; i<stg_ctx->nmodes; i++) {
    ierr = PetscSynchronizedFGets(comm, fp, char_array_len, line); CHKERRQ(ierr);
    ierr = PetscStrToArray(line, ' ', &ndims, &array); CHKERRQ(ierr);
    if(ndims < dims[1]) SETERRQ(comm, -1,
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


PetscErrorCode CreateSTGContext(MPI_Comm comm, STGShur14Context stg_ctx) {
  PetscErrorCode ierr;
  char stg_inflow_path[PETSC_MAX_PATH_LEN] = "./STGInflow.dat";
  char stg_rand_path[PETSC_MAX_PATH_LEN] = "./STGRand.dat";
  CeedScalar u0=0.0, alpha=1.01;
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
  PetscOptionsEnd();

  int nmodes, nprofs;
  ierr = GetNRows(comm, stg_rand_path, &nmodes); CHKERRQ(ierr);
  ierr = GetNRows(comm, stg_inflow_path, &nprofs); CHKERRQ(ierr);

  {
    STGShur14Context s;
    ierr = PetscCalloc1(1, &s); CHKERRQ(ierr);
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
    int total_num_scalars = s->offsets.lt + nprofs;
    size_t bytes = sizeof(*stg_ctx) + total_num_scalars*sizeof(stg_ctx->data[0]);
    ierr = PetscMalloc(bytes, &stg_ctx); CHKERRQ(ierr);
    *stg_ctx = *s;
    ierr = PetscFree(s); CHKERRQ(ierr);
  }
  stg_ctx->alpha = alpha;
  stg_ctx->u0 = u0;

  ierr = ReadSTGInflow(comm, stg_inflow_path, stg_ctx); CHKERRQ(ierr);
  ierr = ReadSTGRand(comm, stg_rand_path, stg_ctx); CHKERRQ(ierr);

  // -- Calculate kappa
  {
    CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
    CeedScalar *prof_dw = &stg_ctx->data[stg_ctx->offsets.prof_dw];
    CeedScalar *lt = &stg_ctx->data[stg_ctx->offsets.lt];
    CeedScalar le, le_max=0;

    CeedPragmaSIMD
    for(int i=0; i<stg_ctx->nprofs; i++) {
      le = PetscMax(2*prof_dw[i], 3*lt[i]);
      if(le_max < le) le_max = le;
    }
    CeedScalar kmin = 0.5*M_PI/le_max;

    CeedPragmaSIMD
    for(int i=0; i<stg_ctx->nmodes; i++) {
      kappa[i] = kmin*pow(stg_ctx->alpha, i);
    }
  } //end calculate kappa
  PetscFunctionReturn(0);
}

PetscErrorCode SetupContext_STGShur14(Ceed ceed, CeedData ceed_data,
                                      AppCtx app_ctx, SetupContext setup_ctx, Physics phys) {
  PetscFunctionBeginUser;

  CeedQFunctionContextCreate(ceed, &ceed_data->stg_shur14_context);
  CeedQFunctionContextSetData(ceed_data->stg_shur14_context, CEED_MEM_HOST,
                              CEED_USE_POINTER,
                              sizeof(*phys->stg_shur14_ctx), phys->stg_shur14_ctx);
  CeedQFunctionSetContext(ceed_data->qf_apply_inflow,
                          ceed_data->stg_shur14_context);
  PetscFunctionReturn(0);
}
