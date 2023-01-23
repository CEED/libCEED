// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up Blasius Boundary Layer

#include "../qfunctions/blasius.h"

#include "../navierstokes.h"
#include "stg_shur14.h"

PetscErrorCode CompressibleBlasiusResidual(SNES snes, Vec X, Vec R, void *ctx) {
  const BlasiusContext blasius = (BlasiusContext)ctx;
  const PetscScalar   *Tf, *Th;  // Chebyshev coefficients
  PetscScalar         *r, f[4], h[4];
  PetscInt             N  = blasius->n_cheb;
  PetscScalar          Ma = Mach(&blasius->newtonian_ctx, blasius->T_inf, blasius->U_inf), Pr = Prandtl(&blasius->newtonian_ctx),
              gamma = HeatCapacityRatio(&blasius->newtonian_ctx);
  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &Tf));
  Th = Tf + N;
  PetscCall(VecGetArray(R, &r));

  // Left boundary conditions f = f' = 0
  ChebyshevEval(N, Tf, -1., blasius->eta_max, f);
  r[0] = f[0];
  r[1] = f[1];

  // f - right end boundary condition
  ChebyshevEval(N, Tf, 1., blasius->eta_max, f);
  r[2] = f[1] - 1.;

  for (int i = 0; i < N - 3; i++) {
    ChebyshevEval(N, Tf, blasius->X[i], blasius->eta_max, f);
    ChebyshevEval(N - 1, Th, blasius->X[i], blasius->eta_max, h);
    // mu and rho generally depend on h.
    // We naively assume constant mu.
    // For an ideal gas at constant pressure, density is inversely proportional to enthalpy.
    // The *_tilde values are *relative* to their freestream values, and we proved first derivatives here.
    const PetscScalar mu_tilde[2]     = {1, 0};
    const PetscScalar rho_tilde[2]    = {1 / h[0], -h[1] / PetscSqr(h[0])};
    const PetscScalar mu_rho_tilde[2] = {
        mu_tilde[0] * rho_tilde[0],
        mu_tilde[1] * rho_tilde[0] + mu_tilde[0] * rho_tilde[1],
    };
    r[3 + i]     = 2 * (mu_rho_tilde[0] * f[3] + mu_rho_tilde[1] * f[2]) + f[2] * f[0];
    r[N + 2 + i] = (mu_rho_tilde[0] * h[2] + mu_rho_tilde[1] * h[1]) + Pr * f[0] * h[1] + Pr * (gamma - 1) * mu_rho_tilde[0] * PetscSqr(Ma * f[2]);
  }

  // h - left end boundary condition
  ChebyshevEval(N - 1, Th, -1., blasius->eta_max, h);
  r[N] = h[0] - blasius->T_wall / blasius->T_inf;

  // h - right end boundary condition
  ChebyshevEval(N - 1, Th, 1., blasius->eta_max, h);
  r[N + 1] = h[0] - 1.;

  // Restore vectors
  PetscCall(VecRestoreArrayRead(X, &Tf));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeChebyshevCoefficients(BlasiusContext blasius) {
  SNES                snes;
  Vec                 sol, res;
  PetscReal          *w;
  PetscInt            N = blasius->n_cheb;
  SNESConvergedReason reason;
  const PetscScalar  *cheb_coefs;
  PetscFunctionBegin;

  // Allocate memory
  PetscCall(PetscMalloc2(N - 3, &blasius->X, N - 3, &w));
  PetscCall(PetscDTGaussQuadrature(N - 3, -1., 1., blasius->X, w));

  // Snes solve
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(VecCreate(PETSC_COMM_SELF, &sol));
  PetscCall(VecSetSizes(sol, PETSC_DECIDE, 2 * N - 1));
  PetscCall(VecSetFromOptions(sol));
  // Constant relative enthalpy 1 as initial guess
  PetscCall(VecSetValue(sol, N, 1., INSERT_VALUES));
  PetscCall(VecDuplicate(sol, &res));
  PetscCall(SNESSetFunction(snes, res, CompressibleBlasiusResidual, blasius));
  PetscCall(SNESSetOptionsPrefix(snes, "chebyshev_"));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "The Chebyshev solve failed.\n");

  // Assign Chebyshev coefficients
  PetscCall(VecGetArrayRead(sol, &cheb_coefs));
  for (int i = 0; i < N; i++) blasius->Tf_cheb[i] = cheb_coefs[i];
  for (int i = 0; i < N - 1; i++) blasius->Th_cheb[i] = cheb_coefs[i + N];

  // Destroy objects
  PetscCall(PetscFree2(blasius->X, w));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&res));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeBoundaryCoefficientsResidual(SNES snes, Vec x, Vec f, void *ctx) {
  const BlasiusContext blasius_ctx = (BlasiusContext)ctx;
  const PetscScalar   *xx;
  PetscScalar         *ff;
  PetscFunctionBegin;

  PetscScalar delta1    = blasius_ctx->delta1;
  PetscScalar delta2    = blasius_ctx->delta2;
  PetscScalar delta_max = blasius_ctx->delta_max;
  PetscScalar r1        = blasius_ctx->r1;
  PetscScalar sigma2    = blasius_ctx->sigma2;
  PetscScalar sigma1    = sigma2 * r1 * delta1 * log(r1) / ((delta2 - delta1) * (r1 - 1) + r1 * delta1 * log(r1));
  PetscScalar k1        = log(r1) / sigma1;
  PetscScalar b1        = delta1 * k1 / (r1 - 1);

  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));

  ff[0] = xx[0] * PetscExpScalar(xx[1] * sigma2) - b1 * r1;
  ff[1] = xx[0] * PetscExpScalar(xx[1]) / xx[1] - b1 * r1 / xx[1] - (delta_max - delta2);

  // Restore vectors
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));

  PetscFunctionReturn(0);
}

PetscErrorCode ComputeBoundaryCoefficients(BlasiusContext blasius_ctx) {
  SNES                snes;
  Vec                 x, r;
  PetscScalar        *xx;
  SNESConvergedReason reason;
  PetscFunctionBegin;

  // Create solution and residual vectors
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 2));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x, 1.));
  PetscCall(VecDuplicate(x, &r));

  // Create nonlinear solver context
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "boundary_mesh_"));
  PetscCall(SNESSetFunction(snes, r, ComputeBoundaryCoefficientsResidual, blasius_ctx));
  PetscCall(SNESSetFromOptions(snes));

  // Solve
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  if (reason < 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "The snes solve for computing boundary mesh coefficients failed.\n");

  // Assign b3 and k3
  PetscCall(VecGetArray(x, &xx));
  blasius_ctx->b3 = xx[0];
  blasius_ctx->k3 = xx[1];
  PetscCall(VecRestoreArray(x, &xx));

  // Cleanup
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode GetYNodeLocs(const MPI_Comm comm, const char path[PETSC_MAX_PATH_LEN], PetscReal **pynodes, PetscInt *nynodes) {
  PetscInt       ndims, dims[2];
  FILE          *fp;
  const PetscInt char_array_len = 512;
  char           line[char_array_len];
  char         **array;
  PetscReal     *node_locs;
  PetscFunctionBeginUser;

  PetscCall(PetscFOpen(comm, path, "r", &fp));
  PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
  PetscCall(PetscStrToArray(line, ' ', &ndims, &array));

  for (PetscInt i = 0; i < ndims; i++) dims[i] = atoi(array[i]);
  if (ndims < 2) dims[1] = 1;  // Assume 1 column of data is not otherwise specified
  *nynodes = dims[0];
  PetscCall(PetscMalloc1(*nynodes, &node_locs));

  for (PetscInt i = 0; i < dims[0]; i++) {
    PetscCall(PetscSynchronizedFGets(comm, fp, char_array_len, line));
    PetscCall(PetscStrToArray(line, ' ', &ndims, &array));
    if (ndims < dims[1])
      SETERRQ(comm, -1, "Line %" PetscInt_FMT " of %s does not contain enough columns (%" PetscInt_FMT " instead of %" PetscInt_FMT ")", i, path,
              ndims, dims[1]);

    node_locs[i] = (PetscReal)atof(array[0]);
  }
  PetscCall(PetscFClose(comm, fp));
  *pynodes = node_locs;
  PetscFunctionReturn(0);
}

/* \brief Modify the domain and mesh for blasius
 *
 * Modifies mesh such that `N` elements are within `refine_height` with a geometric growth ratio of `growth`. Excess elements are then distributed
 * linearly in logspace to the top surface.
 *
 * The top surface is also angled downwards, so that it may be used as an outflow.
 * It's angle is controlled by `top_angle` (in units of degrees).
 *
 * If `node_locs` is not NULL, then the nodes will be placed at `node_locs` locations.
 * If it is NULL, then the modified coordinate values will be set in the array, along with `num_node_locs`.
 */
static PetscErrorCode ModifyMesh(MPI_Comm comm, DM dm, BlasiusContext blasius_ctx, PetscInt dim, PetscReal top_angle, PetscReal *node_locs[],
                                 PetscInt *num_node_locs) {
  PetscInt     narr, ncoords;
  PetscReal    domain_min[3], domain_max[3], domain_size[3];
  PetscScalar *arr_coords;
  Vec          vec_coords;
  PetscFunctionBeginUser;

  PetscReal angle_coeff = tan(top_angle * (M_PI / 180));

  // Compute b3 and k3
  ComputeBoundaryCoefficients(blasius_ctx);
  printf("\nb3 = %.12f \nk3 = %.12f \n", blasius_ctx->b3, blasius_ctx->k3);  // TODO: drop

  // Get domain boundary information
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < 3; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // Get coords array from DM
  PetscCall(DMGetCoordinatesLocal(dm, &vec_coords));
  PetscCall(VecGetLocalSize(vec_coords, &narr));
  PetscCall(VecGetArray(vec_coords, &arr_coords));

  PetscScalar(*coords)[dim] = (PetscScalar(*)[dim])arr_coords;
  ncoords                   = narr / dim;

  // h1 = exp(double x)

  // Get mesh information
  PetscInt nmax = 3, faces[3];
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-dm_plex_box_faces", faces, &nmax, NULL));
  // Get element size of the box mesh, for indexing each node
  const PetscReal dybox = domain_size[1] / faces[1];

  if (!*node_locs) {
    // Gather all parameters necessary to define the generating function
    // Use Newton solver (SNES) to get b3 and k3 values for the generating function
    // send the constructed generating function into:
    // DMPlexRemapGeometry(DM dm, PetscReal time, void (*func)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[],
    // const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
    // PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]))
    // *node_locs = temp_node_locs;
  } else {
    // Error checking
    if (*num_node_locs < faces[1] + 1) {
      SETERRQ(comm, -1,
              "The y_node_locs_path has too few locations; "
              "There are %d + 1 nodes, but only %d locations given",
              faces[1] + 1, *num_node_locs);
    }
    if (*num_node_locs > faces[1] + 1) {
      PetscCall(PetscPrintf(comm,
                            "WARNING: y_node_locs_path has more locations (%d) "
                            "than the mesh has nodes (%d). This maybe unintended.\n",
                            *num_node_locs, faces[1] + 1));
    }
    PetscScalar max_y = (*node_locs)[faces[1]];

    for (PetscInt i = 0; i < ncoords; i++) {
      // Determine which y-node we're at
      PetscInt y_box_index = round(coords[i][1] / dybox);
      coords[i][1]         = (1 - (coords[i][0] - domain_min[0]) * angle_coeff / max_y) * (*node_locs)[y_box_index];
    }
  }

  PetscCall(VecRestoreArray(vec_coords, &arr_coords));
  PetscCall(DMSetCoordinatesLocal(dm, vec_coords));

  PetscFunctionReturn(0);
}

PetscErrorCode NS_BLASIUS(ProblemData *problem, DM dm, void *ctx, SimpleBC bc) {
  User                     user    = *(User *)ctx;
  MPI_Comm                 comm    = PETSC_COMM_WORLD;
  PetscBool                use_stg = PETSC_FALSE;
  BlasiusContext           blasius_ctx;
  NewtonianIdealGasContext newtonian_ig_ctx;
  CeedQFunctionContext     blasius_context;

  PetscFunctionBeginUser;
  PetscCall(NS_NEWTONIAN_IG(problem, dm, ctx, bc));
  PetscCall(PetscCalloc1(1, &blasius_ctx));

  // ------------------------------------------------------
  //               SET UP Blasius
  // ------------------------------------------------------
  problem->ics.qfunction     = ICsBlasius;
  problem->ics.qfunction_loc = ICsBlasius_loc;

  CeedScalar  U_inf                                = 40;      // m/s
  CeedScalar  T_inf                                = 288.;    // K
  CeedScalar  T_wall                               = 288.;    // K
  CeedScalar  delta0                               = 4.2e-3;  // m
  CeedScalar  P0                                   = 1.01e5;  // Pa
  PetscScalar delta1                               = 50;
  PetscScalar delta2                               = 500;
  PetscScalar delta_max                            = 5000;
  PetscScalar r1                                   = 50;
  PetscScalar sigma2                               = 0.8;
  CeedInt     N                                    = 20;           // Number of Chebyshev terms
  PetscBool   weakT                                = PETSC_FALSE;  // weak density or temperature
  PetscReal   mesh_refine_height                   = 5.9e-4;       // m
  PetscReal   mesh_growth                          = 1.08;         // [-]
  PetscInt    mesh_Ndelta                          = 45;           // [-]
  PetscReal   mesh_top_angle                       = 5;            // degrees
  char        mesh_ynodes_path[PETSC_MAX_PATH_LEN] = "";

  PetscOptionsBegin(comm, NULL, "Options for BLASIUS problem", NULL);
  PetscCall(PetscOptionsBool("-weakT", "Change from rho weak to T weak at inflow", NULL, weakT, &weakT, NULL));
  PetscCall(PetscOptionsScalar("-velocity_infinity", "Velocity at boundary layer edge", NULL, U_inf, &U_inf, NULL));
  PetscCall(PetscOptionsScalar("-temperature_infinity", "Temperature at boundary layer edge", NULL, T_inf, &T_inf, NULL));
  PetscCall(PetscOptionsScalar("-temperature_wall", "Temperature at wall", NULL, T_wall, &T_wall, NULL));
  PetscCall(PetscOptionsScalar("-delta0", "Boundary layer height at inflow", NULL, delta0, &delta0, NULL));
  PetscCall(PetscOptionsScalar("-P0", "Pressure at outflow", NULL, P0, &P0, NULL));
  PetscCall(PetscOptionsScalar("-delta1", "TODO", NULL, delta1, &delta1, NULL));
  PetscCall(PetscOptionsScalar("-delta2", "TODO", NULL, delta2, &delta2, NULL));
  PetscCall(PetscOptionsScalar("-delta_max", "TODO", NULL, delta_max, &delta_max, NULL));
  PetscCall(PetscOptionsScalar("-r1", "TODO", NULL, r1, &r1, NULL));
  PetscCall(PetscOptionsScalar("-sigma2", "TODO", NULL, sigma2, &sigma2, NULL));
  PetscCall(PetscOptionsInt("-n_chebyshev", "Number of Chebyshev terms", NULL, N, &N, NULL));
  PetscCheck(3 <= N && N <= BLASIUS_MAX_N_CHEBYSHEV, comm, PETSC_ERR_ARG_OUTOFRANGE, "-n_chebyshev %" PetscInt_FMT " must be in range [3, %d]", N,
             BLASIUS_MAX_N_CHEBYSHEV);
  PetscCall(PetscOptionsBoundedInt("-platemesh_Ndelta", "Velocity at boundary layer edge", NULL, mesh_Ndelta, &mesh_Ndelta, NULL, 1));
  PetscCall(PetscOptionsScalar("-platemesh_refine_height", "Height of boundary layer mesh refinement", NULL, mesh_refine_height, &mesh_refine_height,
                               NULL));
  PetscCall(PetscOptionsScalar("-platemesh_growth", "Geometric growth rate of boundary layer mesh", NULL, mesh_growth, &mesh_growth, NULL));
  PetscCall(
      PetscOptionsScalar("-platemesh_top_angle", "Geometric top_angle rate of boundary layer mesh", NULL, mesh_top_angle, &mesh_top_angle, NULL));
  PetscCall(PetscOptionsString("-platemesh_y_node_locs_path",
                               "Path to file with y node locations. "
                               "If empty, will use the algorithmic mesh warping.",
                               NULL, mesh_ynodes_path, mesh_ynodes_path, sizeof(mesh_ynodes_path), NULL));
  PetscCall(PetscOptionsBool("-stg_use", "Use STG inflow boundary condition", NULL, use_stg, &use_stg, NULL));
  PetscOptionsEnd();

  PetscScalar meter  = user->units->meter;
  PetscScalar second = user->units->second;
  PetscScalar Kelvin = user->units->Kelvin;
  PetscScalar Pascal = user->units->Pascal;

  T_inf *= Kelvin;
  T_wall *= Kelvin;
  P0 *= Pascal;
  U_inf *= meter / second;
  delta0 *= meter;
  delta1 *= meter;
  delta2 *= meter;
  delta_max *= meter;

  blasius_ctx->weakT     = weakT;
  blasius_ctx->U_inf     = U_inf;
  blasius_ctx->T_inf     = T_inf;
  blasius_ctx->T_wall    = T_wall;
  blasius_ctx->delta0    = delta0;
  blasius_ctx->P0        = P0;
  blasius_ctx->delta1    = delta1;
  blasius_ctx->delta2    = delta2;
  blasius_ctx->delta_max = delta_max;
  blasius_ctx->r1        = r1;
  blasius_ctx->sigma2    = sigma2;
  blasius_ctx->n_cheb    = N;
  blasius_ctx->implicit  = user->phys->implicit;

  PetscReal *mesh_ynodes  = NULL;
  PetscInt   mesh_nynodes = 0;
  if (strcmp(mesh_ynodes_path, "")) {
    PetscCall(GetYNodeLocs(comm, mesh_ynodes_path, &mesh_ynodes, &mesh_nynodes));
  }
  PetscCall(ModifyMesh(comm, dm, blasius_ctx, problem->dim, mesh_top_angle, &mesh_ynodes, &mesh_nynodes));

  // Some properties depend on parameters from NewtonianIdealGas
  CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &newtonian_ig_ctx);

  newtonian_ig_ctx->P0       = P0;
  blasius_ctx->newtonian_ctx = *newtonian_ig_ctx;

  {
    PetscReal domain_min[3], domain_max[3];
    PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
    blasius_ctx->x_inflow = domain_min[0];
    blasius_ctx->eta_max  = 5 * domain_max[1] / blasius_ctx->delta0;
  }
  if (!use_stg) PetscCall(ComputeChebyshevCoefficients(blasius_ctx));

  CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &newtonian_ig_ctx);

  CeedQFunctionContextCreate(user->ceed, &blasius_context);
  CeedQFunctionContextSetData(blasius_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*blasius_ctx), blasius_ctx);
  CeedQFunctionContextSetDataDestroy(blasius_context, CEED_MEM_HOST, FreeContextPetsc);

  CeedQFunctionContextDestroy(&problem->ics.qfunction_context);
  problem->ics.qfunction_context = blasius_context;
  if (use_stg) {
    PetscCall(SetupSTG(comm, dm, problem, user, weakT, T_inf, P0, mesh_ynodes, mesh_nynodes));
  } else {
    problem->apply_inflow.qfunction              = Blasius_Inflow;
    problem->apply_inflow.qfunction_loc          = Blasius_Inflow_loc;
    problem->apply_inflow_jacobian.qfunction     = Blasius_Inflow_Jacobian;
    problem->apply_inflow_jacobian.qfunction_loc = Blasius_Inflow_Jacobian_loc;
    CeedQFunctionContextReferenceCopy(blasius_context, &problem->apply_inflow.qfunction_context);
    CeedQFunctionContextReferenceCopy(blasius_context, &problem->apply_inflow_jacobian.qfunction_context);
  }
  PetscCall(PetscFree(mesh_ynodes));
  PetscFunctionReturn(0);
}
