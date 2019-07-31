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

//                        libCEED + PETSc Example: CEED BPs 3-6 with Multigrid
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make multigrid [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     multigrid -problem bp3
//     multigrid -problem bp4 -ceed /cpu/self
//     multigrid -problem bp5 -ceed /cpu/occa
//     multigrid -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3

/// @file
/// CEED BPs 1-6 multigrid example using PETSc
const char help[] = "Solve CEED BPs using p-multigrid with PETSc and DMPlex\n";

#define multigrid
#include "setup.h"

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, qextra, *lsize, *xlsize, *gsize, dim = 3,
           melem[3] = {3, 3, 3}, ncompu = 1, numlevels = degree, *leveldegrees;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution;
  DM  *dm, dmOrig;
  KSP ksp;
  PC pc;
  Mat *matO, *matI, *matR;
  Vec *X, *Xloc, *mult, rhs, rhsloc, diagloc;
  UserO *userO;
  UserIR *userI, *userR;
  Ceed ceed;
  CeedData *ceeddata;
  CeedVector rhsceed, diagceed, target;
  CeedQFunction qf_error, qf_restrict, qf_prolong;
  CeedOperator op_error;
  bpType bpChoice;
  coarsenType coarsen;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpChoice = CEED_BP3;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bpTypes, (PetscEnum)bpChoice, (PetscEnum *)&bpChoice,
                          NULL); CHKERRQ(ierr);
  ncompu = bpOptions[bpChoice].ncompu;
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution",
                          "Write solution for visualization",
                          NULL, write_solution, &write_solution, NULL);
  CHKERRQ(ierr);
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  if (degree < 1) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                             "-degree %D must be at least 1", degree);
  qextra = bpOptions[bpChoice].qextra;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  coarsen = COARSEN_UNIFORM;
  ierr = PetscOptionsEnum("-coarsen",
                          "Coarsening strategy to use", NULL,
                          coarsenTypes, (PetscEnum)coarsen,
                          (PetscEnum *)&coarsen, NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  if (!read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells","Number of cells per dimension", NULL,
                                melem, &tmp, NULL); CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dmOrig);
    CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, melem, NULL,
                               NULL, NULL, PETSC_TRUE,&dmOrig); CHKERRQ(ierr);
  }

  {
    DM dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dmOrig, &part); CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
    ierr = DMPlexDistribute(dmOrig, 0, NULL, &dmDist); CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dmOrig); CHKERRQ(ierr);
      dmOrig = dmDist;
    }
  }

  // Allocate arrays for PETSc objects for each level
  switch (coarsen) {
  case COARSEN_UNIFORM:
    numlevels = degree;
    break;
  case COARSEN_LOGARITHMIC:
    numlevels = ceil(log(degree)/log(2)) + 1;
    break;
  }
  ierr = PetscMalloc1(numlevels, &leveldegrees); CHKERRQ(ierr);
  switch (coarsen) {
  case COARSEN_UNIFORM:
    for (int i=0; i<numlevels; i++) leveldegrees[i] = i + 1;
    break;
  case COARSEN_LOGARITHMIC:
    for (int i=0; i<numlevels-1; i++) leveldegrees[i] = pow(2,i);
    leveldegrees[numlevels-1] = degree;
    break;
  }
  ierr = PetscMalloc1(numlevels, &dm); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &X); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &Xloc); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &mult); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &userO); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &userI); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &userR); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &matO); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &matI); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &matR); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &lsize); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &xlsize); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &gsize); CHKERRQ(ierr);

  // Setup DM and Operator Mat Shells for each level
  for (CeedInt i=0; i<numlevels; i++) {
    // Create DM
    ierr = DMClone(dmOrig, &dm[i]); CHKERRQ(ierr);
    ierr = SetupDMByDegree(dm[i], leveldegrees[i], ncompu, bpChoice);
    CHKERRQ(ierr);

    // Create vectors
    ierr = DMCreateGlobalVector(dm[i], &X[i]); CHKERRQ(ierr);
    ierr = VecGetLocalSize(X[i], &lsize[i]); CHKERRQ(ierr);
    ierr = VecGetSize(X[i], &gsize[i]); CHKERRQ(ierr);
    ierr = DMCreateLocalVector(dm[i], &Xloc[i]); CHKERRQ(ierr);
    ierr = VecGetSize(Xloc[i], &xlsize[i]); CHKERRQ(ierr);

    // Operator
    ierr = PetscMalloc1(1, &userO[i]); CHKERRQ(ierr);
    ierr = MatCreateShell(comm, lsize[i], lsize[i], gsize[i], gsize[i],
                          userO[i], &matO[i]); CHKERRQ(ierr);
    ierr = MatShellSetOperation(matO[i], MATOP_MULT,
                                (void(*)(void))MatMult_Ceed);
    ierr = MatShellSetOperation(matO[i], MATOP_GET_DIAGONAL,
                                (void(*)(void))MatGetDiag);
    CHKERRQ(ierr);

    // Level transfers
    if (i > 0) {
      // Interp
      ierr = PetscMalloc1(1, &userI[i]); CHKERRQ(ierr);
      ierr = MatCreateShell(comm, lsize[i], lsize[i-1], gsize[i], gsize[i-1],
                            userI[i], &matI[i]); CHKERRQ(ierr);
      ierr = MatShellSetOperation(matI[i], MATOP_MULT,
                                  (void(*)(void))MatMult_Interp);
      CHKERRQ(ierr);

      // Restrict
      ierr = PetscMalloc1(1, &userR[i]); CHKERRQ(ierr);
      ierr = MatCreateShell(comm, lsize[i-1], lsize[i], gsize[i-1], gsize[i],
                            userR[i], &matR[i]); CHKERRQ(ierr);
      ierr = MatShellSetOperation(matR[i], MATOP_MULT,
                                  (void(*)(void))MatMult_Restrict);
      CHKERRQ(ierr);
    }
  }
  ierr = VecDuplicate(X[numlevels-1], &rhs); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceedresource, &ceed);

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + qextra;
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc + PCMG --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global Nodes                       : %D\n"
                       "    Owned Nodes                        : %D\n"
                       "  Multigrid:\n"
                       "    Number of Levels                   : %d\n",
                       bpChoice+1, usedresource, P, Q,
                       gsize[numlevels-1]/ncompu, lsize[numlevels-1]/ncompu,
                       numlevels); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(Xloc[numlevels-1], &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xlsize[numlevels-1], &rhsceed);
  CeedVectorSetArray(rhsceed, CEED_MEM_HOST, CEED_USE_POINTER, r);

  // Set up libCEED operators on each level
  ierr = PetscMalloc1(numlevels, &ceeddata); CHKERRQ(ierr);
  for (int i=0; i<numlevels; i++) {
    // Print level information
    if (!test_mode && (i == 0 || i == numlevels-1)) {
      ierr = PetscPrintf(comm,"    Level %D (%s):\n"
                         "      Number of 1D Basis Nodes (p)     : %d\n"
                         "      Global Nodes                     : %D\n"
                         "      Owned Nodes                      : %D\n",
                         i, (i? "fine" : "coarse"), leveldegrees[i] + 1,
                         gsize[i]/ncompu, lsize[i]/ncompu); CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(1, &ceeddata[i]); CHKERRQ(ierr);
    ierr = SetupLibceedByDegree(dm[i], ceed, leveldegrees[i], dim, qextra,
                                ncompu, gsize[i], xlsize[i], bpChoice,
                                ceeddata[i], i==(numlevels-1), rhsceed,
                                &target); CHKERRQ(ierr);
  }

  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm[numlevels-1], rhsloc, ADD_VALUES, rhs);
  CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm[numlevels-1], rhsloc, ADD_VALUES, rhs);
  CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  // Create the restriction/interpolation Q-function
  CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                              &qf_restrict);
  CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                              &qf_prolong);

  // Set up libCEED level transfer operators
  ierr = CeedLevelTransferSetup(ceed, numlevels, ncompu, bpChoice, ceeddata,
                                leveldegrees, qf_restrict, qf_prolong);
  CHKERRQ(ierr);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].error,
                              bpOptions[bpChoice].errorfname, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", ncompu, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_error);
  CeedOperatorSetField(op_error, "u", ceeddata[numlevels-1]->Erestrictu,
                       CEED_TRANSPOSE, ceeddata[numlevels-1]->basisu,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceeddata[numlevels-1]->Erestrictui,
                       CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", ceeddata[numlevels-1]->Erestrictui,
                       CEED_NOTRANSPOSE, CEED_BASIS_COLLOCATED,
                       CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  for (int i=0; i<numlevels; i++) {
    PetscScalar *x;

    // CEED vector
    ierr = VecGetArray(Xloc[i], &x); CHKERRQ(ierr);
    CeedVectorSetArray(ceeddata[i]->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceeddata[i]->Erestrictu,
                                       CEED_NOTRANSPOSE,
                                       ceeddata[i]->xceed);

    // Restore vector
    ierr = VecRestoreArray(Xloc[i], &x); CHKERRQ(ierr);

    // Creat mult vector
    ierr = VecDuplicate(Xloc[i], &mult[i]); CHKERRQ(ierr);

    // Local-to-global
    ierr = VecZeroEntries(X[i]); CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm[i], Xloc[i], ADD_VALUES, X[i]);
    CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm[i], Xloc[i], ADD_VALUES, X[i]);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(Xloc[i]); CHKERRQ(ierr);

    // Global-to-local
    ierr = DMGlobalToLocalBegin(dm[i], X[i], INSERT_VALUES, mult[i]);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm[i], X[i], INSERT_VALUES, mult[i]);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X[i]); CHKERRQ(ierr);

    // Multiplicity scaling
    ierr = VecReciprocal(mult[i]);
  }

  // Set up Mat
  for (int i=0; i<numlevels; i++) {
    // User Operator
    userO[i]->comm = comm;
    userO[i]->dm = dm[i];
    userO[i]->Xloc = Xloc[i];
    ierr = VecDuplicate(Xloc[i], &userO[i]->Yloc); CHKERRQ(ierr);
    userO[i]->xceed = ceeddata[i]->xceed;
    userO[i]->yceed = ceeddata[i]->yceed;
    userO[i]->op = ceeddata[i]->op_apply;
    userO[i]->ceed = ceed;

    // Set up diagonal
    const CeedScalar *ceedarray;
    PetscScalar *petscarray;
    CeedInt length;

    ierr = VecDuplicate(X[i], &userO[i]->diag); CHKERRQ(ierr);
    ierr = VecDuplicate(Xloc[i], &diagloc); CHKERRQ(ierr);

    // -- Local diagonal
    CeedOperatorAssembleLinearDiagonal(userO[i]->op, &diagceed,
                                       CEED_REQUEST_IMMEDIATE);

    // -- Copy values
    CeedVectorGetArrayRead(diagceed, CEED_MEM_HOST, &ceedarray);
    ierr = VecGetArray(diagloc, &petscarray); CHKERRQ(ierr);
    CeedVectorGetLength(diagceed, &length);
    for (CeedInt i=0; i<length; i++)
      petscarray[i] = ceedarray[i];
    CeedVectorRestoreArrayRead(diagceed, &ceedarray);
    ierr = VecRestoreArray(diagloc, &petscarray); CHKERRQ(ierr);

    // -- Global diagonal
    ierr = VecZeroEntries(userO[i]->diag); CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(userO[i]->dm, diagloc, ADD_VALUES,
                                userO[i]->diag); CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(userO[i]->dm, diagloc, ADD_VALUES,
                              userO[i]->diag); CHKERRQ(ierr);

    // -- Cleanup
    ierr = VecDestroy(&diagloc); CHKERRQ(ierr);
    CeedVectorDestroy(&diagceed);

    if (i > 0) {
      // Interp Operator
      userI[i]->comm = comm;
      userI[i]->dmc = dm[i-1];
      userI[i]->dmf = dm[i];
      userI[i]->Xloc = Xloc[i-1];
      userI[i]->Yloc = userO[i]->Yloc;
      userI[i]->mult = mult[i];
      userI[i]->ceedvecc = userO[i-1]->xceed;
      userI[i]->ceedvecf = userO[i]->yceed;
      userI[i]->op = ceeddata[i]->op_interp;
      userI[i]->ceed = ceed;

      // Restrict Operator
      userR[i]->comm = comm;
      userR[i]->dmc = dm[i-1];
      userR[i]->dmf = dm[i];
      userR[i]->Xloc = Xloc[i];
      userR[i]->Yloc = userO[i-1]->Yloc;
      userR[i]->mult = mult[i];
      userR[i]->ceedvecf = userO[i]->xceed;
      userR[i]->ceedvecc = userO[i-1]->yceed;
      userR[i]->op = ceeddata[i]->op_restrict;
      userR[i]->ceed = ceed;
    }
  }

  // Set up KSP
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, matO[numlevels-1], matO[numlevels-1]);
  CHKERRQ(ierr);

  // Set up PCMG
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  PCMGCycleType pcgmcycletype = PC_MG_CYCLE_V;
  {
    ierr = PCSetType(pc, PCMG); CHKERRQ(ierr);

    // PCMG levels
    ierr = PCMGSetLevels(pc, numlevels, NULL); CHKERRQ(ierr);
    for (int i=0; i<numlevels; i++) {
      // Smoother
      KSP smoother;
      PC smoother_pc;
      ierr = PCMGGetSmoother(pc, i, &smoother); CHKERRQ(ierr);
      ierr = KSPSetType(smoother, KSPCHEBYSHEV); CHKERRQ(ierr);
      ierr = KSPChebyshevEstEigSet(smoother, 0, 0.1, 0, 1.1); CHKERRQ(ierr);
      ierr = KSPChebyshevEstEigSetUseNoisy(smoother, PETSC_TRUE); CHKERRQ(ierr);
      ierr = KSPSetOperators(smoother, matO[i], matO[i]); CHKERRQ(ierr);
      ierr = KSPGetPC(smoother, &smoother_pc); CHKERRQ(ierr);
      ierr = PCSetType(smoother_pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(smoother_pc, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);

      // Work vector
      if (i < numlevels-1) {
        ierr = PCMGSetX(pc, i, X[i]); CHKERRQ(ierr);
      }

      // Level transfers
      if (i > 0) {
        // Interpolation
        ierr = PCMGSetInterpolation(pc, i, matI[i]); CHKERRQ(ierr);

        // Restriction
        ierr = PCMGSetRestriction(pc, i, matR[i]); CHKERRQ(ierr);
      }

      // Coarse solve
      KSP coarse;
      PC coarse_pc;
      ierr = PCMGGetCoarseSolve(pc, &coarse); CHKERRQ(ierr);
      ierr = KSPSetType(coarse, KSPCG); CHKERRQ(ierr);
      ierr = KSPSetOperators(coarse, matO[0], matO[0]); CHKERRQ(ierr);
      ierr = KSPSetTolerances(coarse, 1e-10, 1e-10, PETSC_DEFAULT,
                              PETSC_DEFAULT); CHKERRQ(ierr);
      ierr = KSPGetPC(coarse, &coarse_pc); CHKERRQ(ierr);
      ierr = PCSetType(coarse_pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(coarse_pc, PC_JACOBI_DIAGONAL); CHKERRQ(ierr);
    }

    // PCMG options
    ierr = PCMGSetType(pc, PC_MG_MULTIPLICATIVE); CHKERRQ(ierr);
    ierr = PCMGSetNumberSmooth(pc, 3); CHKERRQ(ierr);
    ierr = PCMGSetCycleType(pc, pcgmcycletype); CHKERRQ(ierr);
  }

  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(X[numlevels-1]); CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X[numlevels-1]); CHKERRQ(ierr);
    my_rt = MPI_Wtime() - my_rt_start;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm);
    CHKERRQ(ierr);
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5);
      CHKERRQ(ierr);
    } else {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20);
      CHKERRQ(ierr);
    }
  }

  // Timed solve
  ierr = VecZeroEntries(X[numlevels-1]); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X[numlevels-1]); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;

  // Output results
  {
    KSPType ksptype;
    PCMGType pcmgtype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    ierr = PCMGGetType(pc, &pcmgtype); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %D\n"
                         "    Final rnorm                        : %e\n",
                         ksptype, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(comm,
                         "  PCMG:\n"
                         "    PCMG Type                          : %s\n"
                         "    PCMG Cycle Type                    : %s\n",
                         PCMGTypes[pcmgtype],
                         PCMGCycleTypes[pcgmcycletype]); CHKERRQ(ierr);
    }
    if (!test_mode) {
      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal maxerror;
      ierr = ComputeErrorMax(userO[numlevels-1], op_error, X[numlevels-1], target,
                             &maxerror); CHKERRQ(ierr);
      PetscReal tol = 5e-2;
      if (!test_mode || maxerror > tol) {
        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        CHKERRQ(ierr);
        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(comm,
                           "    Pointwise Error (max)              : %e\n"
                           "    CG Solve Time                      : %g (%g) sec\n",
                           (double)maxerror, rt_max, rt_min); CHKERRQ(ierr);
      }
    }
    if (benchmark_mode && (!test_mode)) {
      ierr = PetscPrintf(comm,
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         1e-6*gsize[numlevels-1]*its/rt_max,
                         1e-6*gsize[numlevels-1]*its/rt_min);
      CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk"); CHKERRQ(ierr);
    ierr = VecView(X[numlevels-1], vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln); CHKERRQ(ierr);
  }

  // Cleanup
  for (int i=0; i<numlevels; i++) {
    ierr = VecDestroy(&X[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&Xloc[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&mult[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&userO[i]->Yloc); CHKERRQ(ierr);
    ierr = VecDestroy(&userO[i]->diag); CHKERRQ(ierr);
    ierr = MatDestroy(&matO[i]); CHKERRQ(ierr);
    ierr = PetscFree(userO[i]); CHKERRQ(ierr);
    if (i > 0) {
      ierr = MatDestroy(&matI[i]); CHKERRQ(ierr);
      ierr = PetscFree(userI[i]); CHKERRQ(ierr);
      ierr = MatDestroy(&matR[i]); CHKERRQ(ierr);
      ierr = PetscFree(userR[i]); CHKERRQ(ierr);
    }
    ierr = CeedDataDestroy(i, ceeddata[i]); CHKERRQ(ierr);
    ierr = DMDestroy(&dm[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(leveldegrees); CHKERRQ(ierr);
  ierr = PetscFree(dm); CHKERRQ(ierr);
  ierr = PetscFree(X); CHKERRQ(ierr);
  ierr = PetscFree(Xloc); CHKERRQ(ierr);
  ierr = PetscFree(mult); CHKERRQ(ierr);
  ierr = PetscFree(matO); CHKERRQ(ierr);
  ierr = PetscFree(matI); CHKERRQ(ierr);
  ierr = PetscFree(matR); CHKERRQ(ierr);
  ierr = PetscFree(ceeddata); CHKERRQ(ierr);
  ierr = PetscFree(userO); CHKERRQ(ierr);
  ierr = PetscFree(userI); CHKERRQ(ierr);
  ierr = PetscFree(userR); CHKERRQ(ierr);
  ierr = PetscFree(lsize); CHKERRQ(ierr);
  ierr = PetscFree(xlsize); CHKERRQ(ierr);
  ierr = PetscFree(gsize); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = DMDestroy(&dmOrig); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedQFunctionDestroy(&qf_restrict);
  CeedQFunctionDestroy(&qf_prolong);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
