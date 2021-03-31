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
//     multigrid -problem bp4
//     multigrid -problem bp5 -ceed /cpu/self
//     multigrid -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3

/// @file
/// CEED BPs 1-6 multigrid example using PETSc
const char help[] = "Solve CEED BPs using p-multigrid with PETSc and DMPlex\n";

#define multigrid
#include "setup.h"

// Transition from a value of "a" for x=0, to a value of "b" for x=1.  Optionally
// smooth -- see the commented versions at the end.
static double step(const double a, const double b, double x) {
  if (x <= 0) return a;
  if (x >= 1) return b;
  return a + (b-a) * (x);
}

// 1D transformation at the right boundary
static double right(const double eps, const double x) {
  return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
static double left(const double eps, const double x) {
  return 1-right(eps,1-x);
}

// Apply 3D Kershaw mesh transformation
// The eps parameters are in (0, 1]
// Uniform mesh is recovered for eps=1
static PetscErrorCode kershaw(DM dmorig, PetscScalar eps) {
  PetscErrorCode ierr;
  Vec coord;
  PetscInt ncoord;
  PetscScalar *c;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dmorig, &coord); CHKERRQ(ierr);
  ierr = VecGetLocalSize(coord, &ncoord); CHKERRQ(ierr);
  ierr = VecGetArray(coord, &c); CHKERRQ(ierr);

  for (PetscInt i = 0; i < ncoord; i += 3) {
    PetscScalar x = c[i], y = c[i+1], z = c[i+2];
    PetscInt layer = x*6;
    PetscScalar lambda = (x-layer/6.0)*6;
    c[i] = x;

    switch (layer) {
    case 0:
      c[i+1] = left(eps, y);
      c[i+2] = left(eps, z);
      break;
    case 1:
    case 4:
      c[i+1] = step(left(eps, y), right(eps, y), lambda);
      c[i+2] = step(left(eps, z), right(eps, z), lambda);
      break;
    case 2:
      c[i+1] = step(right(eps, y), left(eps, y), lambda/2);
      c[i+2] = step(right(eps, z), left(eps, z), lambda/2);
      break;
    case 3:
      c[i+1] = step(right(eps, y), left(eps, y), (1+lambda)/2);
      c[i+2] = step(right(eps, z), left(eps, z), (1+lambda)/2);
      break;
    default:
      c[i+1] = right(eps, y);
      c[i+2] = right(eps, z);
    }
  }
  ierr = VecRestoreArray(coord, &c); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, qextra, *lsize, *xlsize, *gsize, dim = 3, fineLevel,
           melem[3] = {3, 3, 3}, ncompu = 1, numlevels = degree, *leveldegrees;
  PetscScalar *r;
  PetscScalar eps = 1.0;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution;
  PetscLogStage solvestage;
  DM  *dm, dmorig;
  SNES snesdummy;
  KSP ksp;
  PC pc;
  Mat *matO, *matPR, matcoarse;
  Vec *X, *Xloc, *mult, rhs, rhsloc;
  PetscMemType memtype;
  UserO *userO;
  UserProlongRestr *userPR;
  Ceed ceed;
  CeedData *ceeddata;
  CeedVector rhsceed, target;
  CeedQFunction qferror, qfrestrict, qfprolong;
  CeedOperator operror;
  bpType bpchoice;
  coarsenType coarsen;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpchoice = CEED_BP3;
  ierr = PetscOptionsEnum("-problem",
                          "CEED benchmark problem to solve", NULL,
                          bpTypes, (PetscEnum)bpchoice, (PetscEnum *)&bpchoice,
                          NULL); CHKERRQ(ierr);
  ncompu = bpOptions[bpchoice].ncompu;
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
  ierr = PetscOptionsScalar("-eps",
                            "Epsilon parameter for Kershaw mesh transformation",
                            NULL, eps, &eps, NULL);
  if (eps > 1 || eps <= 0) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                                      "-eps %D must be (0,1]", eps);
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  if (degree < 1) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                             "-degree %D must be at least 1", degree);
  qextra = bpOptions[bpchoice].qextra;
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

  // Set up libCEED
  CeedInit(ceedresource, &ceed);
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dmorig);
    CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, melem, NULL,
                               NULL, NULL, PETSC_TRUE,&dmorig); CHKERRQ(ierr);
  }

  {
    DM dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dmorig, &part); CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
    ierr = DMPlexDistribute(dmorig, 0, NULL, &dmDist); CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dmorig); CHKERRQ(ierr);
      dmorig = dmDist;
    }
  }

  // apply Kershaw mesh transformation
  ierr = kershaw(dmorig, eps); CHKERRQ(ierr);

  VecType vectype;
  switch (memtypebackend) {
  case CEED_MEM_HOST: vectype = VECSTANDARD; break;
  case CEED_MEM_DEVICE: {
    const char *resolved;
    CeedGetResource(ceed, &resolved);
    if (strstr(resolved, "/gpu/cuda")) vectype = VECCUDA;
    else if (strstr(resolved, "/gpu/hip/occa"))
      vectype = VECSTANDARD; // https://github.com/CEED/libCEED/issues/678
    else if (strstr(resolved, "/gpu/hip")) vectype = VECHIP;
    else vectype = VECSTANDARD;
  }
  }
  ierr = DMSetVecType(dmorig, vectype); CHKERRQ(ierr);
  ierr = DMSetFromOptions(dmorig); CHKERRQ(ierr);

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
  fineLevel = numlevels - 1;

  switch (coarsen) {
  case COARSEN_UNIFORM:
    for (int i=0; i<numlevels; i++) leveldegrees[i] = i + 1;
    break;
  case COARSEN_LOGARITHMIC:
    for (int i=0; i<numlevels - 1; i++) leveldegrees[i] = pow(2,i);
    leveldegrees[fineLevel] = degree;
    break;
  }
  ierr = PetscMalloc1(numlevels, &dm); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &X); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &Xloc); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &mult); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &userO); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &userPR); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &matO); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &matPR); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &lsize); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &xlsize); CHKERRQ(ierr);
  ierr = PetscMalloc1(numlevels, &gsize); CHKERRQ(ierr);

  // Setup DM and Operator Mat Shells for each level
  for (CeedInt i=0; i<numlevels; i++) {
    // Create DM
    ierr = DMClone(dmorig, &dm[i]); CHKERRQ(ierr);
    ierr = DMGetVecType(dmorig, &vectype); CHKERRQ(ierr);
    ierr = DMSetVecType(dm[i], vectype); CHKERRQ(ierr);
    ierr = SetupDMByDegree(dm[i], leveldegrees[i], ncompu, bpchoice);
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
                                (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
    ierr = MatShellSetOperation(matO[i], MATOP_GET_DIAGONAL,
                                (void(*)(void))MatGetDiag); CHKERRQ(ierr);
    ierr = MatShellSetVecType(matO[i], vectype); CHKERRQ(ierr);

    // Level transfers
    if (i > 0) {
      // Interp
      ierr = PetscMalloc1(1, &userPR[i]); CHKERRQ(ierr);
      ierr = MatCreateShell(comm, lsize[i], lsize[i-1], gsize[i], gsize[i-1],
                            userPR[i], &matPR[i]); CHKERRQ(ierr);
      ierr = MatShellSetOperation(matPR[i], MATOP_MULT,
                                  (void(*)(void))MatMult_Prolong);
      CHKERRQ(ierr);
      ierr = MatShellSetOperation(matPR[i], MATOP_MULT_TRANSPOSE,
                                  (void(*)(void))MatMult_Restrict);
      CHKERRQ(ierr);
      ierr = MatShellSetVecType(matPR[i], vectype); CHKERRQ(ierr);
    }
  }
  ierr = VecDuplicate(X[fineLevel], &rhs); CHKERRQ(ierr);

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + qextra;

    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    ierr = VecGetType(X[0], &vectype); CHKERRQ(ierr);

    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc + PCMG --\n"
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global Nodes                       : %D\n"
                       "    Owned Nodes                        : %D\n"
                       "    DoF per node                       : %D\n"
                       "  Multigrid:\n"
                       "    Number of Levels                   : %d\n",
                       bpchoice+1, vectype, usedresource,
                       CeedMemTypes[memtypebackend],
                       P, Q, gsize[fineLevel]/ncompu, lsize[fineLevel]/ncompu,
                       ncompu, numlevels); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(Xloc[fineLevel], &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhsloc, &r, &memtype); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xlsize[fineLevel], &rhsceed);
  CeedVectorSetArray(rhsceed, MemTypeP2C(memtype), CEED_USE_POINTER, r);

  // Set up libCEED operators on each level
  ierr = PetscMalloc1(numlevels, &ceeddata); CHKERRQ(ierr);
  for (int i=0; i<numlevels; i++) {
    // Print level information
    if (!test_mode && (i == 0 || i == fineLevel)) {
      ierr = PetscPrintf(comm,"    Level %D (%s):\n"
                         "      Number of 1D Basis Nodes (p)     : %d\n"
                         "      Global Nodes                     : %D\n"
                         "      Owned Nodes                      : %D\n",
                         i, (i? "fine" : "coarse"), leveldegrees[i] + 1,
                         gsize[i]/ncompu, lsize[i]/ncompu); CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(1, &ceeddata[i]); CHKERRQ(ierr);
    ierr = SetupLibceedByDegree(dm[i], ceed, leveldegrees[i], dim, qextra,
                                ncompu, gsize[i], xlsize[i], bpchoice,
                                ceeddata[i], i==(fineLevel), rhsceed, &target);
    CHKERRQ(ierr);
  }

  // Gather RHS
  CeedVectorTakeArray(rhsceed, MemTypeP2C(memtype), NULL);
  ierr = VecRestoreArrayAndMemType(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm[fineLevel], rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  // Create the restriction/interpolation QFunction
  CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_NONE, CEED_EVAL_INTERP,
                              &qfrestrict);
  CeedQFunctionCreateIdentity(ceed, ncompu, CEED_EVAL_INTERP, CEED_EVAL_NONE,
                              &qfprolong);

  // Set up libCEED level transfer operators
  ierr = CeedLevelTransferSetup(ceed, numlevels, ncompu, bpchoice, ceeddata,
                                leveldegrees, qfrestrict, qfprolong);
  CHKERRQ(ierr);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpchoice].error,
                              bpOptions[bpchoice].errorfname, &qferror);
  CeedQFunctionAddInput(qferror, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qferror, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qferror, "error", ncompu, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qferror, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &operror);
  CeedOperatorSetField(operror, "u", ceeddata[fineLevel]->Erestrictu,
                       ceeddata[fineLevel]->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(operror, "true_soln", ceeddata[fineLevel]->Erestrictui,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(operror, "error", ceeddata[fineLevel]->Erestrictui,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  for (int i=0; i<numlevels; i++) {
    PetscScalar *x;

    // CEED vector
    ierr = VecZeroEntries(Xloc[i]); CHKERRQ(ierr);
    ierr = VecGetArray(Xloc[i], &x); CHKERRQ(ierr);
    CeedVectorSetArray(ceeddata[i]->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceeddata[i]->Erestrictu,
                                       ceeddata[i]->xceed);
    CeedVectorSyncArray(ceeddata[i]->xceed, CEED_MEM_HOST);

    // Restore vector
    ierr = VecRestoreArray(Xloc[i], &x); CHKERRQ(ierr);

    // Creat mult vector
    ierr = VecDuplicate(Xloc[i], &mult[i]); CHKERRQ(ierr);

    // Local-to-global
    ierr = VecZeroEntries(X[i]); CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dm[i], Xloc[i], ADD_VALUES, X[i]);
    CHKERRQ(ierr);
    ierr = VecZeroEntries(Xloc[i]); CHKERRQ(ierr);

    // Global-to-local
    ierr = DMGlobalToLocal(dm[i], X[i], INSERT_VALUES, mult[i]);
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
    userO[i]->op = ceeddata[i]->opapply;
    userO[i]->ceed = ceed;

    if (i > 0) {
      // Prolongation/Restriction Operator
      userPR[i]->comm = comm;
      userPR[i]->dmf = dm[i];
      userPR[i]->dmc = dm[i-1];
      userPR[i]->locvecc = Xloc[i-1];
      userPR[i]->locvecf = userO[i]->Yloc;
      userPR[i]->multvec = mult[i];
      userPR[i]->ceedvecc = userO[i-1]->xceed;
      userPR[i]->ceedvecf = userO[i]->yceed;
      userPR[i]->opprolong = ceeddata[i]->opprolong;
      userPR[i]->oprestrict = ceeddata[i]->oprestrict;
      userPR[i]->ceed = ceed;
    }
  }

  // Setup dummy SNES for AMG coarse solve
  ierr = SNESCreate(comm, &snesdummy); CHKERRQ(ierr);
  ierr = SNESSetDM(snesdummy, dm[0]); CHKERRQ(ierr);
  ierr = SNESSetSolution(snesdummy, X[0]); CHKERRQ(ierr);

  // -- Jacobian matrix
  ierr = DMSetMatType(dm[0], MATAIJ); CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm[0], &matcoarse); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snesdummy, matcoarse, matcoarse, NULL,
                         NULL); CHKERRQ(ierr);

  // -- Residual evaluation function
  ierr = SNESSetFunction(snesdummy, X[0], FormResidual_Ceed,
                         userO[0]); CHKERRQ(ierr);

  // -- Form Jacobian
  ierr = SNESComputeJacobianDefaultColor(snesdummy, X[0], matO[0],
                                         matcoarse, NULL); CHKERRQ(ierr);

  // Set up KSP
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, matO[fineLevel], matO[fineLevel]);
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
      if (i < numlevels - 1) {
        ierr = PCMGSetX(pc, i, X[i]); CHKERRQ(ierr);
      }

      // Level transfers
      if (i > 0) {
        // Interpolation
        ierr = PCMGSetInterpolation(pc, i, matPR[i]); CHKERRQ(ierr);
      }

      // Coarse solve
      KSP coarse;
      PC coarse_pc;
      ierr = PCMGGetCoarseSolve(pc, &coarse); CHKERRQ(ierr);
      ierr = KSPSetType(coarse, KSPPREONLY); CHKERRQ(ierr);
      ierr = KSPSetOperators(coarse, matcoarse, matcoarse); CHKERRQ(ierr);

      ierr = KSPGetPC(coarse, &coarse_pc); CHKERRQ(ierr);
      ierr = PCSetType(coarse_pc, PCGAMG); CHKERRQ(ierr);

      ierr = KSPSetOptionsPrefix(coarse, "coarse_"); CHKERRQ(ierr);
      ierr = PCSetOptionsPrefix(coarse_pc, "coarse_"); CHKERRQ(ierr);
      ierr = KSPSetFromOptions(coarse); CHKERRQ(ierr);
      ierr = PCSetFromOptions(coarse_pc); CHKERRQ(ierr);
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
    ierr = VecZeroEntries(X[fineLevel]); CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X[fineLevel]); CHKERRQ(ierr);
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
  ierr = VecZeroEntries(X[fineLevel]); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStageRegister("Solve Stage", &solvestage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(solvestage); CHKERRQ(ierr);

  // -- Solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X[fineLevel]); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;


  // -- Performance logging
  ierr = PetscLogStagePop();

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
      ierr = ComputeErrorMax(userO[fineLevel], operror, X[fineLevel], target,
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
                         1e-6*gsize[fineLevel]*its/rt_max,
                         1e-6*gsize[fineLevel]*its/rt_min);
      CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtu"); CHKERRQ(ierr);
    ierr = VecView(X[fineLevel], vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln); CHKERRQ(ierr);
  }

  // Cleanup
  for (int i=0; i<numlevels; i++) {
    ierr = VecDestroy(&X[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&Xloc[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&mult[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&userO[i]->Yloc); CHKERRQ(ierr);
    ierr = MatDestroy(&matO[i]); CHKERRQ(ierr);
    ierr = PetscFree(userO[i]); CHKERRQ(ierr);
    if (i > 0) {
      ierr = MatDestroy(&matPR[i]); CHKERRQ(ierr);
      ierr = PetscFree(userPR[i]); CHKERRQ(ierr);
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
  ierr = PetscFree(matPR); CHKERRQ(ierr);
  ierr = PetscFree(ceeddata); CHKERRQ(ierr);
  ierr = PetscFree(userO); CHKERRQ(ierr);
  ierr = PetscFree(userPR); CHKERRQ(ierr);
  ierr = PetscFree(lsize); CHKERRQ(ierr);
  ierr = PetscFree(xlsize); CHKERRQ(ierr);
  ierr = PetscFree(gsize); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = MatDestroy(&matcoarse); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = SNESDestroy(&snesdummy); CHKERRQ(ierr);
  ierr = DMDestroy(&dmorig); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qferror);
  CeedQFunctionDestroy(&qfrestrict);
  CeedQFunctionDestroy(&qfprolong);
  CeedOperatorDestroy(&operror);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
