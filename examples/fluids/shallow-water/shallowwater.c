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

//              libCEED + PETSc Example: Shallow-water equations
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// shallow-water equations on a cubed-sphere (i.e., a tensor-product discrete
// sphere, obtained by projecting a cube inscribed in a sphere onto the surface
// of the sphere).
//
// The code uses higher level communication protocols in PETSc's DMPlex.
//
// Build with:
//
//     make [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     shallow-water
//     shallow-water -ceed /cpu/self -dm_refine 1 snes_fd_color -ts_fd_color -degree 1
//     shallow-water -ceed /gpu/occa -dm_refine 1 snes_fd_color -ts_fd_color -degree 1
//
//TESTARGS -ceed {ceed_resource} -test -dm_refine 1 snes_fd_color -ts_fd_color -degree 1

/// @file
/// Shallow-water equations example using PETSc

const char help[] = "Solve the shallow-water equations using PETSc and libCEED\n";

#include "sw_headers.h"

int main(int argc, char **argv) {
  // PETSc context
  PetscInt ierr;
  MPI_Comm comm;
  DM dm, dmviz;
  TS ts;
  TSAdapt adapt;
  Mat J, interpviz;
  User user;
  Units units;
  problemType problemChoice;
  problemData *problem = NULL;
  StabilizationType stab;
  PetscBool implicit;
  PetscInt degree, qextra, outputfreq, steps, contsteps;
  PetscMPIInt rank;
  PetscScalar ftime;
  Vec Q, Qloc, Xloc;
  const CeedInt ncompx = 3;
  PetscInt viz_refine = 0;
  PetscBool read_mesh, simplex, test;
  PetscInt topodim = 2, ncompq = 3, lnodes;
  // libCEED context
  char ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self",
                                          filename[PETSC_MAX_PATH_LEN];
  Ceed ceed;
  CeedData ceeddata;
  CeedMemType memtyperequested;
  CeedScalar CtauS       = 0.;         // dimensionless
  CeedScalar strong_form = 0.;         // [0,1]
  PetscScalar meter      = 1e-2;       // 1 meter in scaled length units
  PetscScalar second     = 1e-2;       // 1 second in scaled time units
  PetscScalar Omega      = 7.29212e-5; // Earth rotation rate (1/s)
  PetscScalar R_e        = 6.37122e6;  // Earth radius (m)
  PetscScalar g          = 9.81;       // gravitational acceleration (m/s^2)
  PetscScalar H0         = 0;          // constant mean height (m)
  PetscScalar gamma      = 0;          // angle between axis of rotation and polar axis
  PetscScalar mpersquareds;
  // Check PETSc CUDA support
  PetscBool petschavecuda, setmemtyperequest = PETSC_FALSE;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*
  // Performance context
  double start, cpu_time_used;

  // Initialize PETSc
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // Allocate PETSc context
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  ierr = PetscMalloc1(1, &units); CHKERRQ(ierr);

  // Parse command line options
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Shallow-water equations in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test", "Run in test mode",
                          NULL, test=PETSC_FALSE, &test, NULL); CHKERRQ(ierr);
  problemChoice = SWE_ADVECTION;
  ierr = PetscOptionsEnum("-problem", "Problem to solve", NULL,
                          problemTypes, (PetscEnum)problemChoice,
                          (PetscEnum *)&problemChoice, NULL); CHKERRQ(ierr);
  problem = &problemOptions[problemChoice];
  ierr = PetscOptionsEnum("-stab", "Stabilization method", NULL,
                          StabilizationTypes, (PetscEnum)(stab = STAB_NONE),
                          (PetscEnum *)&stab, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation",
                          NULL, implicit=PETSC_FALSE, &implicit, NULL);
  CHKERRQ(ierr);
  if (!implicit && stab != STAB_NONE) {
    ierr = PetscPrintf(comm, "Warning! Use -stab only with -implicit\n");
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsScalar("-CtauS",
                            "Scale coefficient for tau (nondimensional)",
                            NULL, CtauS, &CtauS, NULL); CHKERRQ(ierr);
  if (stab == STAB_NONE && CtauS != 0) {
    ierr = PetscPrintf(comm,
                       "Warning! Use -CtauS only with -stab su or -stab supg\n");
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsScalar("-strong_form",
                            "Strong (1) or weak/integrated by parts (0) advection residual",
                            NULL, strong_form, &strong_form, NULL);
  CHKERRQ(ierr);
  if (problemChoice == SWE_GEOSTROPHIC && (CtauS != 0 || strong_form != 0)) {
    ierr = PetscPrintf(comm,
                       "Warning! Problem geostrophic does not support -CtauS or -strong_form\n");
    CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-viz_refine",
                         "Regular refinement levels for visualization",
                         NULL, viz_refine, &viz_refine, NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-units_meter", "1 meter in scaled length units",
                            NULL, meter, &meter, NULL); CHKERRQ(ierr);
  meter = fabs(meter);
  ierr = PetscOptionsScalar("-units_second","1 second in scaled time units",
                            NULL, second, &second, NULL); CHKERRQ(ierr);
  second = fabs(second);
  outputfreq = 10;
  ierr = PetscOptionsInt("-output_freq", "Frequency of output, in number of steps",
                         NULL, outputfreq, &outputfreq, NULL); CHKERRQ(ierr);
  contsteps = 0;
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, contsteps, &contsteps, NULL); CHKERRQ(ierr);
  degree = 3;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 2;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-g", "Gravitational acceleration",
                            NULL, g, &g, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-H0", "Mean height",
                            NULL, H0, &H0, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsScalar("-H0", "Angle between axis of rotation and polar axis",
                            NULL, gamma, &gamma, NULL); CHKERRQ(ierr);
  PetscStrncpy(user->outputfolder, ".", 2);
  ierr = PetscOptionsString("-of", "Output folder",
                            NULL, user->outputfolder, user->outputfolder,
                            sizeof(user->outputfolder), NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  simplex = PETSC_FALSE;
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells",
                          NULL, simplex, &simplex, NULL); CHKERRQ(ierr);
  memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;
  ierr = PetscOptionsEnum("-memtype",
                          "CEED MemType requested", NULL,
                          memTypes, (PetscEnum)memtyperequested,
                          (PetscEnum *)&memtyperequested, &setmemtyperequest);
  CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Define derived units
  mpersquareds = meter / PetscSqr(second);

  // Scale variables to desired units
  Omega /= second;
  g *= mpersquareds;

  // Set up the libCEED context
  PhysicsContext_s phys_ctx =  {
    .u0 = 0.,
    .v0 = 0.,
    .h0 = .1,
    .Omega = Omega,
    .R = R_e/R_e, // normalize to unit sphere (R=1)
    .g = g,
    .H0 = H0,
    .gamma = gamma,
    .time = 0.
  };

  ProblemContext_s probl_ctx =  {
    .g = g,
    .H0 = H0,
    .CtauS = CtauS,
    .strong_form = strong_form,
    .stabilization = stab
  };

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface, not a box.
    PetscBool simplex = PETSC_FALSE;
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topodim, simplex,
                                  phys_ctx.R, &dm);
    CHKERRQ(ierr);
    // Set the object name
    ierr = PetscObjectSetName((PetscObject)dm, "Sphere"); CHKERRQ(ierr);
    // Define cube panels (charts)
    DMLabel label;
    PetscInt c, cStart, cEnd, npanel, permidx[6] = {5, 1, 4, 0, 3, 2};
    ierr = DMCreateLabel(dm, "panel");
    ierr = DMGetLabel(dm, "panel", &label);
    // Assign different panel (chart) values to the six faces of the cube
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
    for (c = cStart, npanel = 0; c < cEnd; c++) {
      ierr = DMLabelSetValue(label, c, permidx[npanel++]); CHKERRQ(ierr);
    }
    // Distribute mesh over processes
    {
      DM dmDist = NULL;
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
      if (dmDist) {
        ierr = DMDestroy(&dm); CHKERRQ(ierr);
        dm  = dmDist;
      }
    }
    // Refine DMPlex with uniform refinement using runtime option -dm_refine
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    ierr = ProjectToUnitSphere(dm); CHKERRQ(ierr);
    // View DMPlex via runtime option
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  }

  // Create DM
  ierr = DMLocalizeCoordinates(dm); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = SetupDM(dm, degree, ncompq, topodim);
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
      ierr = SetupDM(dmhierarchy[i+1], degree, ncompq, topodim); CHKERRQ(ierr);
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

  // Print grid information
  PetscInt numP = degree + 1, numQ = numP + qextra;
  PetscInt gdofs, odofs;
  {
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    int comm_size;
    ierr = MPI_Comm_size(comm, &comm_size); CHKERRQ(ierr);

    ierr = VecGetSize(Q, &gdofs); CHKERRQ(ierr);
    ierr = VecGetLocalSize(Q, &odofs); CHKERRQ(ierr);
    VecType vectype;
    ierr = VecGetType(Q, &vectype); CHKERRQ(ierr);

    PetscInt cStart, cEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);

    if (!test) {
      ierr = PetscPrintf(comm,
                         "\n-- CEED Shallow-water equations solver on the cubed-sphere -- libCEED + PETSc --\n"
                         "  rank(s)                              : %d\n"
                         "  Problem:\n"
                         "    Problem Name                       : %s\n"
                         "    Stabilization                      : %s\n"
                         "  PETSc:\n"
                         "    PETSc Vec Type                     : %s\n"
                         "  libCEED:\n"
                         "    libCEED Backend                    : %s\n"
                         "    libCEED Backend MemType            : %s\n"
                         "    libCEED User Requested MemType     : %s\n"
                         "  FEM space:\n"
                         "    Number of 1D Basis Nodes (P)       : %d\n"
                         "    Number of 1D Quadrature Points (Q) : %d\n"
                         "    Local Elements                     : %D\n"
                         "    Global nodes                       : %D\n"
                         "    Owned nodes                        : %D\n"
                         "    DoFs per node                      : %D\n"
                         "    Global DoFs                        : %D\n"
                         "    Owned DoFs                         : %D\n",
                         comm_size, problemTypes[problemChoice],
                         StabilizationTypes[stab], ceedresource, usedresource,
                         CeedMemTypes[memtypebackend],
                         (setmemtyperequest) ?
                         CeedMemTypes[memtyperequested] : "none", numP, numQ,
                         cEnd - cStart, gdofs/ncompq, odofs/ncompq, ncompq,
                         gdofs, odofs); CHKERRQ(ierr);
    }
  }

  // Set up global mass vector
  ierr = VecDuplicate(Q, &user->M); CHKERRQ(ierr);

  // Setup global lat-long vector for different panels (charts) of the cube
  Mat T;
  ierr = FindPanelEdgeNodes(dm, &phys_ctx, ncompq, &T); CHKERRQ(ierr);

  // Setup libCEED's objects
  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceed(dm, ceed, degree, qextra, ncompx, ncompq, user, ceeddata,
                      problem, &phys_ctx, &probl_ctx); CHKERRQ(ierr);

  // Set up PETSc context
  // Set up units structure
  units->meter = meter;
  units->second = second;
  units->mpersquareds = mpersquareds;

  // Set up user structure
  user->comm = comm;
  user->outputfreq = outputfreq;
  user->contsteps = contsteps;
  user->units = units;
  user->dm = dm;
  user->dmviz = dmviz;
  user->interpviz = interpviz;
  user->ceed = ceed;

  // Calculate qdata and ICs
  // Set up state global and local vectors
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(user->q0ceed, Qloc); CHKERRQ(ierr);

  // Apply Setup Ceed Operators
  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(ceeddata->xcorners, Xloc); CHKERRQ(ierr);
  CeedOperatorApply(ceeddata->op_setup, ceeddata->xcorners, ceeddata->qdata,
                    CEED_REQUEST_IMMEDIATE);
  ierr = ComputeLumpedMassMatrix(ceed, dm, ceeddata->Erestrictq,
                                 ceeddata->basisq, ceeddata->Erestrictqdi,
                                 ceeddata->qdata, user->M); CHKERRQ(ierr);

  // Apply IC operator and fix multiplicity of initial state vector
  ierr = ICs_FixMultiplicity(ceeddata->op_ics, ceeddata->xcorners, user->q0ceed,
                             dm, Qloc, Q, ceeddata->Erestrictq,
                             &phys_ctx, 0.0); CHKERRQ(ierr);

  MPI_Comm_rank(comm, &rank);
  if (!rank) {
    ierr = PetscMkdir(user->outputfolder); CHKERRQ(ierr);
  }
  // Gather initial Q values
  // In case of continuation of simulation, set up initial values from binary file
  if (contsteps) { // continue from existent solution
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    // Read input
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/swe-solution.bin",
                         user->outputfolder);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = VecLoad(Q, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &Qloc); CHKERRQ(ierr);

  // Set up the MatShell for the associated Jacobian operator
  ierr = MatCreateShell(comm, ncompq*odofs, ncompq*odofs,
                        PETSC_DETERMINE, PETSC_DETERMINE, user, &J);
  CHKERRQ(ierr);
  // Set the MatShell operation needed for the Jacobian
  ierr = MatShellSetOperation(J, MATOP_MULT,
                              (void (*)(void))ApplyJacobian_SW); CHKERRQ(ierr);

  // Create and setup TS
  ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm); CHKERRQ(ierr);
  ierr = TSSetType(ts, TSARKIMEX); CHKERRQ(ierr);
  // Tell the TS which functions to use for Explicit part (RHS), Implicit part and Jacobian
  ierr = TSSetRHSFunction(ts, NULL, FormRHSFunction_SW, &user); CHKERRQ(ierr);
  ierr = TSSetIFunction(ts, NULL, FormIFunction_SW, &user); CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts, J, J, FormJacobian_SW, &user); CHKERRQ(ierr);

  // Other TS options
  ierr = TSSetMaxTime(ts, 500. * units->second); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 1.e-2 * units->second); CHKERRQ(ierr);
  if (test) {ierr = TSSetMaxSteps(ts, 1); CHKERRQ(ierr);}
  ierr = TSGetAdapt(ts, &adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,
                              1.e-12 * units->second,
                              1.e2 * units->second); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
  if (!contsteps) { // print initial condition
    if (!test) {
      ierr = TSMonitor_SW(ts, 0, 0., Q, user); CHKERRQ(ierr);
    }
  } else { // continue from time of last output
    PetscReal time;
    PetscInt count;
    PetscViewer viewer;
    char filepath[PETSC_MAX_PATH_LEN];
    ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/ns-time.bin",
                         user->outputfolder); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(comm, filepath, FILE_MODE_READ, &viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryRead(viewer, &time, 1, &count, PETSC_REAL);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = TSSetTime(ts, time * user->units->second); CHKERRQ(ierr);
  }
  if (!test) {
    ierr = TSMonitorSet(ts, TSMonitor_SW, user, NULL); CHKERRQ(ierr);
  }

  // Solve
  start = MPI_Wtime();
  ierr = PetscBarrier((PetscObject)ts); CHKERRQ(ierr);
  ierr = TSSolve(ts, Q); CHKERRQ(ierr);
  cpu_time_used = MPI_Wtime() - start;
  ierr = TSGetSolveTime(ts,&ftime); CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE, &cpu_time_used, 1, MPI_DOUBLE, MPI_MIN,
                       comm); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time taken for solution: %g\n",
                       (double)cpu_time_used); CHKERRQ(ierr);
  }

  // Output Statistics
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  if (!test) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Time integrator took %D time steps to reach final time %g\n",
                       steps,(double)ftime); CHKERRQ(ierr);
  }

  // Clean up libCEED
  CeedVectorDestroy(&ceeddata->qdata);
  CeedVectorDestroy(&user->qceed);
  CeedVectorDestroy(&user->fceed);
  CeedVectorDestroy(&user->gceed);
  CeedVectorDestroy(&user->jceed);
  CeedVectorDestroy(&ceeddata->xcorners);
  CeedBasisDestroy(&ceeddata->basisq);
  CeedBasisDestroy(&ceeddata->basisx);
  CeedElemRestrictionDestroy(&ceeddata->Erestrictq);
  CeedElemRestrictionDestroy(&ceeddata->Erestrictx);
  CeedElemRestrictionDestroy(&ceeddata->Erestrictqdi);
  CeedQFunctionDestroy(&ceeddata->qf_setup);
  CeedQFunctionDestroy(&ceeddata->qf_ics);
  CeedQFunctionDestroy(&ceeddata->qf_explicit);
  CeedQFunctionDestroy(&ceeddata->qf_implicit);
  CeedQFunctionDestroy(&ceeddata->qf_jacobian);
  CeedOperatorDestroy(&ceeddata->op_setup);
  CeedOperatorDestroy(&ceeddata->op_ics);
  CeedOperatorDestroy(&ceeddata->op_explicit);
  CeedOperatorDestroy(&ceeddata->op_implicit);
  CeedOperatorDestroy(&ceeddata->op_jacobian);
  CeedDestroy(&ceed);

  // Clean up PETSc
  ierr = VecDestroy(&Q); CHKERRQ(ierr);
  ierr = VecDestroy(&user->M); CHKERRQ(ierr);
  ierr = MatDestroy(&interpviz); CHKERRQ(ierr);
  ierr = DMDestroy(&dmviz); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PetscFree(units); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
