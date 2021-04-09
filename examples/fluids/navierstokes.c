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

//                        libCEED + PETSc Example: Navier-Stokes
//
// This example demonstrates a simple usage of libCEED with PETSc to solve a
// Navier-Stokes problem.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>] navierstokes
//
// Sample runs:
//
//     ./navierstokes -ceed /cpu/self -problem density_current -degree 1
//     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
//
//TESTARGS(name="dc_explicit") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-explicit.bin
//TESTARGS(name="dc_implicit_stab_none") -ceed {ceed_resource} -test -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -center 62.5,62.5,187.5 -rc 100. -thetaC -35. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-dc-implicit-stab-none.bin
//TESTARGS(name="adv_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection -strong_form 1 -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-explicit-strong.bin
//TESTARGS(name="adv_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -problem_advection_wind translation -problem_advection_wind_translation .53,-1.33,-2.65 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv-translation-implicit-stab-su.bin
//TESTARGS(name="adv2d_rotation_explicit_strong") -ceed {ceed_resource} -test -problem advection2d -strong_form 1 -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ts_dt 1e-3 -compare_final_state_atol 1E-11 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-explicit-strong.bin
//TESTARGS(name="adv2d_rotation_implicit_stab_supg") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab supg -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-rotation-implicit-stab-supg.bin
//TESTARGS(name="adv2d_translation_implicit_stab_su") -ceed {ceed_resource} -test -problem advection2d -CtauS .3 -stab su -degree 3 -dm_plex_box_faces 1,1,2 -units_kilogram 1e-9 -lx 125 -ly 125 -lz 250 -rc 100. -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -problem_advection_wind translation -problem_advection_wind_translation .53,-1.33,0 -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-adv2d-translation-implicit-stab-su.bin
//TESTARGS(name="euler_implicit") -ceed {ceed_resource} -test -problem euler_vortex -degree 3 -dm_plex_box_faces 1,1,2 -units_meter 1e-4 -lx 125 -ly 125 -lz 1 -problem_euler_mean_velocity 1.4,-2.,0 -vortex_strength 2 -ksp_atol 1e-4 -ksp_rtol 1e-3 -ksp_type bcgs -snes_atol 1e-3 -snes_lag_jacobian 100 -snes_lag_jacobian_persists -snes_mf_operator -ts_dt 1e-3 -implicit -ts_type alpha -compare_final_state_atol 5E-4 -compare_final_state_filename examples/fluids/tests-output/fluids-navierstokes-euler-implicit.bin

/// @file
/// Navier-Stokes example using PETSc

const char help[] = "Solve Navier-Stokes using PETSc and libCEED\n";

#include "navierstokes.h"

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  DM dm, dmcoord, dmviz;
  Mat interpviz;
  TS ts;
  TSAdapt adapt;
  User user;
  Units units;
  SetupContext ctxSetupData;
  Physics ctxPhysData;
  char ceedresource[4096] = "/cpu/self", problemName[] = "density_current";
  PetscFunctionList problems = NULL;
  PetscInt localNelemVol, lnodes, gnodes, steps;
  const PetscInt ncompq = 5;
  PetscMPIInt rank;
  PetscScalar ftime;
  Vec Q, Qloc, Xloc;
  Ceed ceed;
  CeedInt numP, numQ;
  CeedVector xcorners, qdata, q0ceed;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction restrictx, restrictq, restrictqdi;
  CeedQFunction qf_setupVol, qf_ics, qf_rhsVol, qf_ifunctionVol;
  CeedQFunctionContext ctxSetup, ctxNS, ctxAdvection, ctxEuler;
  CeedOperator op_setupVol, op_ics;
  CeedScalar Rd;
  CeedMemType memtyperequested;
  problemData *problem = NULL;
  PetscInt    viz_refine = 0, Xloc_size;
  SimpleBC bc;
  double start, cpu_time_used;
  // Test variables
  PetscBool test;
  PetscScalar testtol = 0.;
  char filepath[PETSC_MAX_PATH_LEN];
  // Check PETSc CUDA support
  PetscBool petschavecuda, setmemtyperequest = PETSC_FALSE;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*

  PetscInt outputfreq    = 10;       // -
  PetscInt contsteps     = 0;        // -
  PetscInt degree        = 1;        // -
  PetscInt qextra        = 2;        // -
  PetscInt qextraSur     = 2;        // -

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &problem); CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &bc); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &ctxSetupData); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &ctxPhysData); CHKERRQ(ierr);

  // Register problems to be available on the command line
  ierr = PetscFunctionListAdd(&problems, "density_current", NS_DENSITY_CURRENT);
  CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problems, "euler_vortex", NS_EULER_VORTEX);
  CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problems, "advection", NS_ADVECTION);
  CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problems, "advection2d", NS_ADVECTION2D);
  CHKERRQ(ierr);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test", "Run in test mode",
                          NULL, test=PETSC_FALSE, &test, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-compare_final_state_atol",
                            "Test absolute tolerance",
                            NULL, testtol, &testtol, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-compare_final_state_filename", "Test filename",
                            NULL, filepath, filepath,
                            sizeof(filepath), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsFList("-problem", "Problem to solve", NULL, problems,
                           problemName, problemName, sizeof(problemName),
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-viz_refine",
                         "Regular refinement levels for visualization",
                         NULL, viz_refine, &viz_refine, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-output_freq",
                         "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, contsteps, &contsteps, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  PetscBool userQextraSur;
  ierr = PetscOptionsInt("-qextra_boundary",
                         "Number of extra quadrature points on in/outflow faces",
                         NULL, qextraSur, &qextraSur, &userQextraSur);
  CHKERRQ(ierr);

  ierr = PetscStrncpy(user->outputdir, ".", 2); CHKERRQ(ierr);
  ierr = PetscOptionsString("-output_dir", "Output directory",
                            NULL, user->outputdir, user->outputdir,
                            sizeof(user->outputdir), NULL); CHKERRQ(ierr);
  memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;
  ierr = PetscOptionsEnum("-memtype",
                          "CEED MemType requested", NULL,
                          memTypes, (PetscEnum)memtyperequested,
                          (PetscEnum *)&memtyperequested, &setmemtyperequest);
  CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  {
    // Choose the problem from the list of registered problems
    PetscErrorCode (*p)(problemData *, void *, void *, void *);
    ierr = PetscFunctionListFind(problems, problemName, &p); CHKERRQ(ierr);
    if (!p) SETERRQ1(PETSC_COMM_SELF, 1, "Problem '%s' not found", problemName);
    ierr = (*p)(problem, &ctxSetupData, &units, &ctxPhysData); CHKERRQ(ierr);
  }

  const CeedInt dim = problem->dim, ncompx = problem->dim,
                qdatasizeVol = problem->qdatasizeVol;

  // Create the mesh
  {
    const PetscReal scale[3] = {ctxSetupData->lx, ctxSetupData->ly, ctxSetupData->lz};
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, scale,
                               NULL, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  }

  // Distribute the mesh over processes
  {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dm); CHKERRQ(ierr);
      dm  = dmDist;
    }
  }
  ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);

  // Setup DM
  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = SetUpDM(dm, problem, degree, bc, ctxPhysData, ctxSetupData);
  CHKERRQ(ierr);

  // Refine DM for high-order viz
  dmviz = NULL;
  interpviz = NULL;
  if (viz_refine) {
    DM dmhierarchy[viz_refine+1];

    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
    dmhierarchy[0] = dm;
    for (PetscInt i = 0, d = degree; i < viz_refine; i++) {
      Mat interp_next;

      ierr = DMRefine(dmhierarchy[i], MPI_COMM_NULL, &dmhierarchy[i+1]);
      CHKERRQ(ierr);
      ierr = DMClearDS(dmhierarchy[i+1]); CHKERRQ(ierr);
      ierr = DMClearFields(dmhierarchy[i+1]); CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dmhierarchy[i+1], dmhierarchy[i]); CHKERRQ(ierr);
      d = (d + 1) / 2;
      if (i + 1 == viz_refine) d = 1;
      ierr = SetUpDM(dmhierarchy[i+1], problem, d, bc, ctxPhysData,
                     ctxSetupData); CHKERRQ(ierr);
      ierr = DMCreateInterpolation(dmhierarchy[i], dmhierarchy[i+1],
                                   &interp_next, NULL); CHKERRQ(ierr);
      if (!i) interpviz = interp_next;
      else {
        Mat C;
        ierr = MatMatMult(interp_next, interpviz, MAT_INITIAL_MATRIX,
                          PETSC_DECIDE, &C); CHKERRQ(ierr);
        ierr = MatDestroy(&interp_next); CHKERRQ(ierr);
        ierr = MatDestroy(&interpviz); CHKERRQ(ierr);
        interpviz = C;
      }
    }
    for (PetscInt i=1; i<viz_refine; i++) {
      ierr = DMDestroy(&dmhierarchy[i]); CHKERRQ(ierr);
    }
    dmviz = dmhierarchy[viz_refine];
  }
  ierr = DMCreateGlobalVector(dm, &Q); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Qloc); CHKERRQ(ierr);
  ierr = VecGetSize(Qloc, &lnodes); CHKERRQ(ierr);
  lnodes /= ncompq;

  // Initialize CEED
  CeedInit(ceedresource, &ceed);
  // Set memtype
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);
  // Check memtype compatibility
  if (!setmemtyperequest)
    memtyperequested = memtypebackend;
  else if (!petschavecuda && memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // Set number of 1D nodes and quadrature points
  numP = degree + 1;
  numQ = numP + qextra;

  // Print summary
  if (!test) {
    CeedInt gdofs, odofs;
    int comm_size;
    char box_faces_str[PETSC_MAX_PATH_LEN] = "NONE";
    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    gnodes = gdofs/ncompq;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetString(NULL, NULL, "-dm_plex_box_faces", box_faces_str,
                                 sizeof(box_faces_str), NULL); CHKERRQ(ierr);
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    ierr = PetscPrintf(comm,
                       "\n-- Navier-Stokes solver - libCEED + PETSc --\n"
                       "  rank(s)                              : %d\n"
                       "  Problem:\n"
                       "    Problem Name                       : %s\n"
                       "    Stabilization                      : %s\n"
                       "  PETSc:\n"
                       "    Box Faces                          : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (P)       : %d\n"
                       "    Number of 1D Quadrature Points (Q) : %d\n"
                       "    Global DoFs                        : %D\n"
                       "    Owned DoFs                         : %D\n"
                       "    DoFs per node                      : %D\n"
                       "    Global nodes                       : %D\n"
                       "    Owned nodes                        : %D\n",
                       comm_size, problemName, StabilizationTypes[ctxPhysData->stab],
                       box_faces_str, usedresource, CeedMemTypes[memtypebackend],
                       (setmemtyperequest) ? CeedMemTypes[memtyperequested] : "none",
                       numP, numQ, gdofs, odofs, ncompq, gnodes, lnodes);
    CHKERRQ(ierr);
  }

  // Set up global mass vector
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // Set up libCEED
  // CEED Bases
  CeedInit(ceedresource, &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompq, numP, numQ, CEED_GAUSS,
                                  &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numQ, CEED_GAUSS,
                                  &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, 2, numP,
                                  CEED_GAUSS_LOBATTO, &basisxc);
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // CEED Restrictions
  ierr = GetRestrictionForDomain(ceed, dm, 0, 0, 0, numP, numQ,
                                 qdatasizeVol, &restrictq, &restrictx,
                                 &restrictqdi); CHKERRQ(ierr);

  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Xloc, &Xloc_size); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, Xloc_size, &xcorners); CHKERRQ(ierr);

  // Create the CEED vectors that will be needed in setup
  CeedInt NqptsVol;
  CeedBasisGetNumQuadraturePoints(basisq, &NqptsVol);
  CeedElemRestrictionGetNumElements(restrictq, &localNelemVol);
  CeedVectorCreate(ceed, qdatasizeVol*localNelemVol*NqptsVol, &qdata);
  CeedElemRestrictionCreateVector(restrictq, &q0ceed, NULL);

  // Create the Q-function that builds the quadrature data for the NS operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupVol, problem->setupVol_loc,
                              &qf_setupVol);
  CeedQFunctionAddInput(qf_setupVol, "dx", ncompx*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupVol, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);

  // Create the Q-function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", ncompq, CEED_EVAL_NONE);

  qf_rhsVol = NULL;
  if (problem->applyVol_rhs) { // Create the Q-function that defines the action of the RHS operator
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_rhs,
                                problem->applyVol_rhs_loc, &qf_rhsVol);
    CeedQFunctionAddInput(qf_rhsVol, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhsVol, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_rhsVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_rhsVol, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhsVol, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhsVol, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  qf_ifunctionVol = NULL;
  if (problem->applyVol_ifunction) { // Create the Q-function that defines the action of the IFunction
    CeedQFunctionCreateInterior(ceed, 1, problem->applyVol_ifunction,
                                problem->applyVol_ifunction_loc, &qf_ifunctionVol);
    CeedQFunctionAddInput(qf_ifunctionVol, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunctionVol, "dq", ncompq*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ifunctionVol, "qdot", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_ifunctionVol, "qdata", qdatasizeVol, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_ifunctionVol, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunctionVol, "v", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ifunctionVol, "dv", ncompq*dim, CEED_EVAL_GRAD);
  }

  // Create the operator that builds the quadrature data for the NS operator
  CeedOperatorCreate(ceed, qf_setupVol, NULL, NULL, &op_setupVol);
  CeedOperatorSetField(op_setupVol, "dx", restrictx, basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupVol, "weight", CEED_ELEMRESTRICTION_NONE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupVol, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", restrictx, basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", restrictq,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(restrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(restrictq, &user->gceed, NULL);

  if (qf_rhsVol) { // Create the RHS physics operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_rhsVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx, basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq, basisq, CEED_VECTOR_ACTIVE);
    user->op_rhs_vol = op;
  }

  if (qf_ifunctionVol) { // Create the IFunction operator
    CeedOperator op;
    CeedOperatorCreate(ceed, qf_ifunctionVol, NULL, NULL, &op);
    CeedOperatorSetField(op, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dq", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "qdot", restrictq, basisq, user->qdotceed);
    CeedOperatorSetField(op, "qdata", restrictqdi,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(op, "x", restrictx, basisx, xcorners);
    CeedOperatorSetField(op, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op, "dv", restrictq, basisq, CEED_VECTOR_ACTIVE);
    user->op_ifunction_vol = op;
  }

  // Set up CEED for the boundaries
  CeedInt height = 1;
  CeedInt dimSur = dim - height;
  CeedInt numP_Sur = degree + 1;
  CeedInt numQ_Sur = numP_Sur + qextraSur;
  const CeedInt qdatasizeSur = problem->qdatasizeSur;
  CeedBasis basisxSur, basisxcSur, basisqSur;
  CeedInt NqptsSur;
  CeedQFunction qf_setupSur, qf_applySur;

  // CEED bases for the boundaries
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompq, numP_Sur, numQ_Sur,
                                  CEED_GAUSS,
                                  &basisqSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numQ_Sur, CEED_GAUSS,
                                  &basisxSur);
  CeedBasisCreateTensorH1Lagrange(ceed, dimSur, ncompx, 2, numP_Sur,
                                  CEED_GAUSS_LOBATTO, &basisxcSur);
  CeedBasisGetNumQuadraturePoints(basisqSur, &NqptsSur);

  // Create the Q-function that builds the quadrature data for the Surface operator
  CeedQFunctionCreateInterior(ceed, 1, problem->setupSur, problem->setupSur_loc,
                              &qf_setupSur);
  CeedQFunctionAddInput(qf_setupSur, "dx", ncompx*dimSur, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupSur, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupSur, "qdataSur", qdatasizeSur, CEED_EVAL_NONE);

  // Creat Q-Function for Boundaries
  qf_applySur = NULL;
  if (problem->applySur) {
    CeedQFunctionCreateInterior(ceed, 1, problem->applySur,
                                problem->applySur_loc, &qf_applySur);
    CeedQFunctionAddInput(qf_applySur, "q", ncompq, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_applySur, "qdataSur", qdatasizeSur, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_applySur, "x", ncompx, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_applySur, "v", ncompq, CEED_EVAL_INTERP);
  }

  // Create CEED Operator for the whole domain
  if (!ctxPhysData->implicit)
    ierr = CreateOperatorForDomain(ceed, dm, bc, ctxPhysData,
                                   user->op_rhs_vol,
                                   qf_applySur, qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, basisxSur, basisqSur,
                                   &user->op_rhs); CHKERRQ(ierr);
  if (ctxPhysData->implicit)
    ierr = CreateOperatorForDomain(ceed, dm, bc, ctxPhysData,
                                   user->op_ifunction_vol,
                                   qf_applySur, qf_setupSur,
                                   height, numP_Sur, numQ_Sur, qdatasizeSur,
                                   NqptsSur, basisxSur, basisqSur,
                                   &user->op_ifunction); CHKERRQ(ierr);
  // Set up contex for QFunctions
  CeedQFunctionContextCreate(ceed, &ctxSetup);
  CeedQFunctionContextSetData(ctxSetup, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof *ctxSetupData, ctxSetupData);
  if (qf_ics && strcmp(problemName, "euler_vortex") != 0)
    CeedQFunctionSetContext(qf_ics, ctxSetup);

  CeedQFunctionContextCreate(ceed, &ctxNS);
  CeedQFunctionContextSetData(ctxNS, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->ctxNSData, ctxPhysData->ctxNSData);

  CeedQFunctionContextCreate(ceed, &ctxEuler);
  CeedQFunctionContextSetData(ctxEuler, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->ctxEulerData, ctxPhysData->ctxEulerData);

  CeedQFunctionContextCreate(ceed, &ctxAdvection);
  CeedQFunctionContextSetData(ctxAdvection, CEED_MEM_HOST, CEED_USE_POINTER,
                              sizeof ctxPhysData->ctxAdvectionData, ctxPhysData->ctxAdvectionData);

  if (strcmp(problemName, "density_current") == 0) {
    if (qf_rhsVol) CeedQFunctionSetContext(qf_rhsVol, ctxNS);
    if (qf_ifunctionVol) CeedQFunctionSetContext(qf_ifunctionVol, ctxNS);
  } else if (strcmp(problemName, "euler_vortex") == 0) {
    if (qf_ics) CeedQFunctionSetContext(qf_ics, ctxEuler);
    if (qf_rhsVol) CeedQFunctionSetContext(qf_rhsVol, ctxEuler);
    if (qf_ifunctionVol) CeedQFunctionSetContext(qf_ifunctionVol, ctxEuler);
    if (qf_applySur) CeedQFunctionSetContext(qf_applySur, ctxEuler);
  } else {
    if (qf_rhsVol) CeedQFunctionSetContext(qf_rhsVol, ctxAdvection);
    if (qf_ifunctionVol) CeedQFunctionSetContext(qf_ifunctionVol, ctxAdvection);
    if (qf_applySur) CeedQFunctionSetContext(qf_applySur, ctxAdvection);
  }

  // Set up user structure
  user->comm = comm;
  user->outputfreq = outputfreq;
  user->contsteps = contsteps;
  user->units = units;
  user->dm = dm;
  user->dmviz = dmviz;
  user->interpviz = interpviz;
  user->ceed = ceed;
  user->phys = ctxPhysData;

  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);

  ierr = VectorPlacePetscVec(q0ceed, Qloc); CHKERRQ(ierr);

  // Apply Setup Ceed Operators
  ierr = VectorPlacePetscVec(xcorners, Xloc); CHKERRQ(ierr);
  CeedOperatorApply(op_setupVol, xcorners, qdata, CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, restrictq, basisq, restrictqdi, qdata,
                                 user->M); CHKERRQ(ierr);

  ierr = ICs_FixMultiplicity(op_ics, xcorners, q0ceed, dm, Qloc, Q, restrictq,
                             ctxSetup, 0.0); CHKERRQ(ierr);
  if (1) { // Record boundary values from initial condition and override DMPlexInsertBoundaryValues()
    // We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow.  If we
    // disable this, we should still get the same results due to the problem->bc function, but with potentially much
    // slower execution.
    Vec Qbc;
    ierr = DMGetNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = VecCopy(Qloc, Qbc); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
    ierr = VecAXPY(Qbc, -1., Qloc); CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(dm, "Qbc", &Qbc); CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)dm,
                                      "DMPlexInsertBoundaryValues_C", DMPlexInsertBoundaryValues_NS);
    CHKERRQ(ierr);
  }

  MPI_Comm_rank(comm, &rank);
  if (!rank) {ierr = PetscMkdir(user->outputdir); CHKERRQ(ierr);}
  // Gather initial Q values
  // In case of continuation of simulation, set up initial values from binary file
  if (contsteps) { // continue from existent solution
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    // Read input
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-solution.bin",
                         user->outputdir);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Q, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &Qloc); CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm); CHKERRQ(ierr);
  if (ctxPhysData->implicit) {
    ierr = TSSetType(ts, TSBDF); CHKERRQ(ierr);
    if (user->op_ifunction) {
      ierr = TSSetIFunction(ts, NULL, IFunction_NS, &user); CHKERRQ(ierr);
    } else {                    // Implicit integrators can fall back to using an RHSFunction
      ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
    }
  } else {
    if (!user->op_rhs) SETERRQ(comm, PETSC_ERR_ARG_NULL,
                                 "Problem does not provide RHSFunction");
    ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
    ierr = TSRKSetType(ts, TSRK5F); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts, NULL, RHS_NS, &user); CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-2 * units->second); CHKERRQ(ierr);
  if (test) {ierr = TSSetMaxSteps(ts, 10); CHKERRQ(ierr);}
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps) { // print initial condition
    if (!test) {
      ierr = TSMonitor_NS(ts, 0, 0., Q, user); CHKERRQ(ierr);
    }
  } else { // continue from time of last output
    PetscReal time;
    PetscInt count;
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-time.bin",
                         user->outputdir); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(ts, time * user->units->second); CHKERRQ(ierr);
  }
  if (!test) {
    ierr = TSMonitorSet(ts, TSMonitor_NS, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject)ts); CHKERRQ(ierr);
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(ts, &ftime); CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution (sec): %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }

  // Get error
  if (problem->non_zero_time && !test) {
    Vec Qexact, Qexactloc;
    PetscReal rel_error, norm_error, norm_exact;
    ierr = DMCreateGlobalVector(dm, &Qexact); CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecGetSize(Qexactloc, &lnodes); CHKERRQ(ierr);

    ierr = ICs_FixMultiplicity(op_ics, xcorners, q0ceed, dm, Qexactloc, Qexact,
                               restrictq, ctxSetup, ftime); CHKERRQ(ierr);
    ierr = VecNorm(Qexact, NORM_1, &norm_exact); CHKERRQ(ierr);
    ierr = VecAXPY(Q, -1.0, Qexact);  CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_1, &norm_error); CHKERRQ(ierr);
    rel_error = norm_error / norm_exact;
    CeedVectorDestroy(&q0ceed);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Relative Error: %g\n",
                       (double)rel_error); CHKERRQ(ierr);
    // Clean up vectors
    ierr = DMRestoreLocalVector(dm, &Qexactloc); CHKERRQ(ierr);
    ierr = VecDestroy(&Qexact); CHKERRQ(ierr);
  }

  // Output Statistics
  ierr = TSGetStepNumber(ts, &steps); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps, (double)ftime); CHKERRQ(ierr);
  }
  // Output numerical values from command line
  ierr = VecViewFromOptions(Q, NULL, "-vec_view"); CHKERRQ(ierr);

  // Compare reference solution values with current test run for CI
  if (test) {
    PetscViewer viewer;
    // Read reference file
    Vec Qref;
    PetscReal error, Qrefnorm;
    ierr = VecDuplicate(Q, &Qref); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Qref, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    // Compute error with respect to reference solution
    ierr = VecAXPY(Q, -1.0, Qref);  CHKERRQ(ierr);
    ierr = VecNorm(Qref, NORM_MAX, &Qrefnorm); CHKERRQ(ierr);
    ierr = VecScale(Q, 1./Qrefnorm); CHKERRQ(ierr);
    ierr = VecNorm(Q, NORM_MAX, &error); CHKERRQ(ierr);
    ierr = VecDestroy(&Qref); CHKERRQ(ierr);
    // Check error
    if (error > testtol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
                         "Test failed with error norm %g\n",
                         (double)error); CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  // Clean up libCEED
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->qdotceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&xcorners);
  CeedBasisDestroy(&basisq);
  CeedBasisDestroy(&basisx);
  CeedBasisDestroy(&basisxc);
  CeedElemRestrictionDestroy(&restrictq);
  CeedElemRestrictionDestroy(&restrictx);
  CeedElemRestrictionDestroy(&restrictqdi);
  CeedQFunctionDestroy(&qf_setupVol);
  CeedQFunctionDestroy(&qf_ics);
  CeedQFunctionDestroy(&qf_rhsVol);
  CeedQFunctionDestroy(&qf_ifunctionVol);
  CeedQFunctionContextDestroy(&ctxSetup);
  CeedQFunctionContextDestroy(&ctxNS);
  CeedQFunctionContextDestroy(&ctxAdvection);
  CeedQFunctionContextDestroy(&ctxEuler);
  CeedOperatorDestroy(&op_setupVol);
  CeedOperatorDestroy(&op_ics);
  CeedOperatorDestroy(&user->op_rhs_vol);
  CeedOperatorDestroy(&user->op_ifunction_vol);
  CeedDestroy(&ceed);
  CeedBasisDestroy(&basisqSur);
  CeedBasisDestroy(&basisxSur);
  CeedBasisDestroy(&basisxcSur);
  CeedQFunctionDestroy(&qf_setupSur);
  CeedQFunctionDestroy(&qf_applySur);
  CeedOperatorDestroy(&user->op_rhs);
  CeedOperatorDestroy(&user->op_ifunction);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = MatDestroy(&interpviz); CHKERRQ(ierr);
  ierr = DMDestroy(&dmviz); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = PetscFree(problem); CHKERRQ(ierr);
  ierr = PetscFree(bc); CHKERRQ(ierr);
  ierr = PetscFree(ctxSetupData); CHKERRQ(ierr);
  ierr = PetscFree(ctxPhysData); CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&problems); CHKERRQ(ierr);
  return PetscFinalize();
}
