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
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bps [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./bps -problem bp1 -degree 3
//     ./bps -problem bp2 -ceed /cpu/self -degree 3
//     ./bps -problem bp3 -ceed /gpu/occa -degree 3
//     ./bps -problem bp4 -ceed /cpu/occa -degree 3
//     ./bps -problem bp5 -ceed /omp/occa -degree 3
//     ./bps -problem bp6 -ceed /ocl/occa -degree 3
//
//TESTARGS -ceed {ceed_resource} -test -problem bp5 -degree 3

/// @file
/// CEED BPs example using PETSc with DMPlex
/// See bpsraw.c for a "raw" implementation using a structured grid.
const char help[] = "Solve CEED BPs using PETSc with DMPlex\n";

#include <stdbool.h>
#include <string.h>
#include <petscksp.h>
#include <petscdmplex.h>
#include <ceed.h>
#include "setup.h"

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

// Utility function, compute three factors of an integer
static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d=0,sizeleft=size; d<3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(sizeleft, 1./(3 - d)));
    while (try * (sizeleft / try) != sizeleft) try++;
    m[reverse ? 2-d : d] = try;
    sizeleft /= try;
  }
}

static int Max3(const PetscInt a[3]) {
  return PetscMax(a[0], PetscMax(a[1], a[2]));
}

static int Min3(const PetscInt a[3]) {
  return PetscMin(a[0], PetscMin(a[1], a[2]));
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt degree = 3, qextra, lsize, gsize, dim = 3, melem[3] = {3, 3, 3},
           ncompu = 1, xlsize, localnodes;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, read_mesh, write_solution,
            userlnodes = PETSC_FALSE;
  PetscLogStage solvestage;
  Vec X, Xloc, rhs, rhsloc;
  Mat matO;
  KSP ksp;
  DM  dm;
  UserO userO;
  Ceed ceed;
  CeedData ceeddata;
  CeedQFunction qferror;
  CeedOperator operror;
  CeedVector rhsceed, target;
  CeedMemType memtyperequested;
  bpType bpchoice;

  // Check PETSc CUDA support
  PetscBool petschavecuda, setmemtyperequest = PETSC_FALSE;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL); CHKERRQ(ierr);
  bpchoice = CEED_BP1;
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
  degree = test_mode ? 3 : 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = bpOptions[bpchoice].qextra;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  if (!read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL,
                                melem, &tmp, NULL); CHKERRQ(ierr);
  }
  memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;
  ierr = PetscOptionsEnum("-memtype",
                          "CEED MemType requested", NULL,
                          memTypes, (PetscEnum)memtyperequested,
                          (PetscEnum *)&memtyperequested, &setmemtyperequest);
  CHKERRQ(ierr);
  localnodes = 1000;
  ierr = PetscOptionsInt("-local_nodes",
                         "Target number of locally owned nodes per process",
                         NULL, localnodes, &localnodes, &userlnodes);
  CHKERRQ(ierr);

  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    if (userlnodes) {
      // Find a nicely composite number of elements no less than lnodes
      PetscMPIInt size;
      ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
      for (PetscInt gelem = PetscMax(1, size * localnodes / PetscPowInt(degree, dim));
           ;
           gelem++) {
        Split3(gelem, melem, true);
        if (Max3(melem) / Min3(melem) <= 2) break;
      }
    }
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, melem, NULL,
                               NULL, NULL, PETSC_TRUE, &dm); CHKERRQ(ierr);
  }

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

  // Set up libCEED
  CeedInit(ceedresource, &ceed);
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  // Check memtype compatibility
  if (!setmemtyperequest)
    memtyperequested = memtypebackend;
  else if (!petschavecuda && memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // Create DM
  ierr = SetupDMByDegree(dm, degree, ncompu, bpchoice);
  CHKERRQ(ierr);

  // Create vectors
  if (memtyperequested == CEED_MEM_DEVICE) {
    ierr = DMSetVecType(dm, VECCUDA); CHKERRQ(ierr);
  }
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
  ierr = MatShellSetOperation(matO, MATOP_GET_DIAGONAL,
                              (void(*)(void))MatGetDiag); CHKERRQ(ierr);
  if (memtyperequested == CEED_MEM_DEVICE) {
    ierr = MatShellSetVecType(matO, VECCUDA); CHKERRQ(ierr);
  }

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + qextra;

    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    VecType vectype;
    ierr = VecGetType(X, &vectype); CHKERRQ(ierr);

    PetscInt cStart, cEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc --\n"
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n"
                       "    Local Elements                     : %D\n"
                       "    Owned nodes                        : %D\n"
                       "    DoF per node                       : %D\n",
                       bpchoice+1, vectype, usedresource,
                       CeedMemTypes[memtypebackend],
                       (setmemtyperequest) ?
                       CeedMemTypes[memtyperequested] : "none",
                       P, Q, gsize/ncompu, cEnd - cStart, lsize/ncompu,
                       ncompu);
    CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  if (memtyperequested == CEED_MEM_HOST) {
    ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArray(rhsloc, &r); CHKERRQ(ierr);
  }
  CeedVectorCreate(ceed, xlsize, &rhsceed);
  CeedVectorSetArray(rhsceed, memtyperequested, CEED_USE_POINTER, r);

  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, dim, qextra,
                              ncompu, gsize, xlsize, bpchoice, ceeddata,
                              true, rhsceed, &target); CHKERRQ(ierr);

  // Gather RHS
  CeedVectorSyncArray(rhsceed, memtyperequested);
  if (memtyperequested == CEED_MEM_HOST) {
    ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  } else {
    ierr = VecCUDARestoreArray(rhsloc, &r); CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[bpchoice].error,
                              bpOptions[bpchoice].errorfname, &qferror);
  CeedQFunctionAddInput(qferror, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qferror, "true_soln", ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qferror, "error", ncompu, CEED_EVAL_NONE);

  // Create the error operator
  CeedOperatorCreate(ceed, qferror, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &operror);
  CeedOperatorSetField(operror, "u", ceeddata->Erestrictu,
                       ceeddata->basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(operror, "true_soln", ceeddata->Erestrictui,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(operror, "error", ceeddata->Erestrictui,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Set up Mat
  userO->comm = comm;
  userO->dm = dm;
  userO->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &userO->Yloc); CHKERRQ(ierr);
  userO->xceed = ceeddata->xceed;
  userO->yceed = ceeddata->yceed;
  userO->op = ceeddata->opapply;
  userO->ceed = ceed;
  userO->memtype = memtyperequested;
  if (memtyperequested == CEED_MEM_HOST) {
    userO->VecGetArray = VecGetArray;
    userO->VecGetArrayRead = VecGetArrayRead;
    userO->VecRestoreArray = VecRestoreArray;
    userO->VecRestoreArrayRead = VecRestoreArrayRead;
  } else {
    userO->VecGetArray = VecCUDAGetArray;
    userO->VecGetArrayRead = VecCUDAGetArrayRead;
    userO->VecRestoreArray = VecCUDARestoreArray;
    userO->VecRestoreArrayRead = VecCUDARestoreArrayRead;
  }

  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (bpchoice == CEED_BP1 || bpchoice == CEED_BP2) {
      ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
      ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    }
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
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
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

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
      ierr = ComputeErrorMax(userO, operror, X, target, &maxerror);
      CHKERRQ(ierr);
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
                         1e-6*gsize*its/rt_max,
                         1e-6*gsize*its/rt_min); CHKERRQ(ierr);
    }
  }

  if (write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(comm, &vtkviewersoln); CHKERRQ(ierr);
    ierr = PetscViewerSetType(vtkviewersoln, PETSCVIEWERVTK); CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk"); CHKERRQ(ierr);
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
  CeedQFunctionDestroy(&qferror);
  CeedOperatorDestroy(&operror);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
