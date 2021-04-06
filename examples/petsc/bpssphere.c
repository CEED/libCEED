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

//                        libCEED + PETSc Example: CEED BPs
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps,
// on a closed surface, such as the one of a discrete sphere.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bpssphere [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bpssphere -problem bp1 -degree 3
//     bpssphere -problem bp2 -degree 3
//     bpssphere -problem bp3 -degree 3
//     bpssphere -problem bp4 -degree 3
//     bpssphere -problem bp5 -degree 3 -ceed /cpu/self
//     bpssphere -problem bp6 -degree 3 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3 -dm_refine 2

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bps.c for a "raw" implementation using a structured grid.
/// and bpsdmplex.c for an implementation using an unstructured grid.
static const char help[] = "Solve CEED BPs on a sphere using DMPlex in PETSc\n";

#include <ceed.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>
#include "bpssphere.h"

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self",
                                          filename[PETSC_MAX_PATH_LEN];
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, qextra, lsize, gsize, topodim = 2, ncompx = 3,
           ncompu = 1, xlsize;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution, simplex;
  PetscLogStage solvestage;
  Vec X, Xloc, rhs, rhsloc;
  Mat matO;
  KSP ksp;
  DM  dm;
  UserO userO;
  Ceed ceed;
  CeedData ceeddata;
  CeedQFunction qfError;
  CeedOperator opError;
  CeedVector rhsceed, target;
  bpType bpChoice;
  VecType vectype;
  PetscMemType memtype;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpChoice = CEED_BP1;
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
  qextra = bpOptions[bpChoice].qextra;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  simplex = PETSC_FALSE;
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells",
                          NULL, simplex, &simplex, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface, not a box
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topodim, simplex, 1., &dm);
    CHKERRQ(ierr);
    // Set the object name
    ierr = PetscObjectSetName((PetscObject)dm, "Sphere"); CHKERRQ(ierr);
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
  ierr = SetupDMByDegree(dm, degree, ncompu, topodim, false, (BCFunction)NULL);
  CHKERRQ(ierr);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &lsize); CHKERRQ(ierr);
  ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetSize(Xloc, &xlsize); CHKERRQ(ierr);
  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);

  // Operator
  ierr = PetscMalloc1(1, &userO); CHKERRQ(ierr);
  ierr = MatCreateShell(comm, lsize, lsize, gsize, gsize,
                        userO, &matO); CHKERRQ(ierr);
  ierr = MatShellSetOperation(matO, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceedresource, &ceed);
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  ierr = DMGetVecType(dm, &vectype); CHKERRQ(ierr);
  if (!vectype) { // Not yet set by user -dm_vec_type
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
    ierr = DMSetVecType(dm, vectype); CHKERRQ(ierr);
  }

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + qextra;
    const char *usedresource;
    CeedGetResource(ceed, &usedresource);
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d on the Sphere -- libCEED + PETSc --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n",
                       bpChoice+1, ceedresource, CeedMemTypes[memtypebackend], P, Q,
                       gsize/ncompu); CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhsloc, &r, &memtype); CHKERRQ(ierr);
  CeedVectorCreate(ceed, xlsize, &rhsceed);
  CeedVectorSetArray(rhsceed, MemTypeP2C(memtype), CEED_USE_POINTER, r);

  // Setup libCEED's objects
  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, topodim, qextra,
                              ncompx, ncompu, gsize, xlsize, bpOptions[bpChoice],
                              ceeddata, true, rhsceed, &target); CHKERRQ(ierr);

  // Gather RHS
  CeedVectorTakeArray(rhsceed, MemTypeP2C(memtype), NULL);
  ierr = VecRestoreArrayAndMemType(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpChoice].error,
                              bpOptions[bpChoice].errorfname, &qfError);
  CeedQFunctionAddInput(qfError, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qfError, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qfError, "error", ncompu, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qfError, NULL, NULL, &opError);
  CeedOperatorSetField(opError, "u", ceeddata->Erestrictu,
                       ceeddata->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(opError, "true_soln", ceeddata->Erestrictui,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(opError, "error", ceeddata->Erestrictui,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  userO->comm = comm;
  userO->dm = dm;
  userO->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &userO->Yloc); CHKERRQ(ierr);
  userO->Xceed = ceeddata->Xceed;
  userO->Yceed = ceeddata->Yceed;
  userO->op = ceeddata->opApply;
  userO->ceed = ceed;

  // Setup solver
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (bpChoice == CEED_BP1 || bpChoice == CEED_BP2) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
      MatNullSpace nullspace;

      ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, 0, &nullspace);
      CHKERRQ(ierr);
      ierr = MatSetNullSpace(matO, nullspace); CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nullspace); CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, matO, matO); CHKERRQ(ierr);

  // First run, if benchmarking
  if (benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
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
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStageRegister("Solve Stage", &solvestage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(solvestage); CHKERRQ(ierr);

  // -- Solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  ierr = PetscLogStagePop();

  // Output results
  {
    KSPType ksptype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %D\n"
                         "    Final rnorm                        : %e\n",
                         ksptype, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
    }
    if (!test_mode) {
      ierr = PetscPrintf(comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal maxerror;
      ierr = ComputeErrorMax(userO, opError, X, target, &maxerror);
      CHKERRQ(ierr);
      PetscReal tol = 5e-4;
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
                         1e-6*gsize*its/rt_max, 1e-6*gsize*its/rt_min); CHKERRQ(ierr);
    }
  }

  // Output solution
  if (write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtu"); CHKERRQ(ierr);
    ierr = VecView(X, vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vtkviewersoln); CHKERRQ(ierr);
  }

  // Cleanup
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&userO->Yloc); CHKERRQ(ierr);
  ierr = MatDestroy(&matO); CHKERRQ(ierr);
  ierr = PetscFree(userO); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceeddata); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qfError);
  CeedOperatorDestroy(&opError);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
