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

// -----------------------------------------------------------------------------
// Parameter structure for running problems
// -----------------------------------------------------------------------------
typedef struct RunParams_ *RunParams;
struct RunParams_ {
  MPI_Comm comm;
  PetscBool test_mode, benchmark_mode, read_mesh, userlnodes, setmemtyperequest,
            petschavecuda, write_solution;
  char *filename, *ceedresource, *hostname;
  PetscInt localnodes, degree, qextra, dim, ncompu, *melem;
  PetscInt ksp_max_it_clip[2];
  PetscMPIInt rankspernode;
  bpType bpchoice;
  CeedMemType memtyperequested;
  PetscLogStage solvestage;
};

// -----------------------------------------------------------------------------
// Main body of program, called in a loop for performance benchmarking purposes
// -----------------------------------------------------------------------------

static PetscErrorCode Run(RunParams rp) {
  PetscInt ierr;
  double my_rt_start, my_rt, rt_min, rt_max;
  PetscInt xlsize, lsize, gsize;
  PetscScalar *r;
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

  PetscFunctionBeginUser;
  // Setup DM
  if (rp->read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, rp->filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    if (rp->userlnodes) {
      // Find a nicely composite number of elements no less than global nodes
      PetscMPIInt size;
      ierr = MPI_Comm_size(rp->comm, &size); CHKERRQ(ierr);
      for (PetscInt gelem =
             PetscMax(1, size * rp->localnodes / PetscPowInt(rp->degree, rp->dim));
           ;
           gelem++) {
        Split3(gelem, rp->melem, true);
        if (Max3(rp->melem) / Min3(rp->melem) <= 2) break;
      }
    }
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, rp->dim, PETSC_FALSE, rp->melem,
                               NULL, NULL, NULL, PETSC_TRUE, &dm); CHKERRQ(ierr);
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
  CeedInit(rp->ceedresource, &ceed);
  CeedMemType memtypebackend;
  CeedGetPreferredMemType(ceed, &memtypebackend);

  // Check memtype compatibility
  if (!rp->setmemtyperequest)
    rp->memtyperequested = memtypebackend;
  else if (!rp->petschavecuda && rp->memtyperequested == CEED_MEM_DEVICE)
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_SUP_SYS,
             "PETSc was not built with CUDA. "
             "Requested MemType CEED_MEM_DEVICE is not supported.", NULL);

  // Create DM
  ierr = SetupDMByDegree(dm, rp->degree, rp->ncompu, rp->bpchoice);
  CHKERRQ(ierr);

  // Create vectors
  if (rp->memtyperequested == CEED_MEM_DEVICE) {
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
  ierr = MatCreateShell(rp->comm, lsize, lsize, gsize, gsize,
                        userO, &matO); CHKERRQ(ierr);
  ierr = MatShellSetOperation(matO, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
  ierr = MatShellSetOperation(matO, MATOP_GET_DIAGONAL,
                              (void(*)(void))MatGetDiag); CHKERRQ(ierr);
  if (rp->memtyperequested == CEED_MEM_DEVICE) {
    ierr = MatShellSetVecType(matO, VECCUDA); CHKERRQ(ierr);
  }

  // Print summary
  if (!rp->test_mode) {
    PetscInt P = rp->degree + 1, Q = P + rp->qextra;

    const char *usedresource;
    CeedGetResource(ceed, &usedresource);

    VecType vectype;
    ierr = VecGetType(X, &vectype); CHKERRQ(ierr);

    PetscInt cStart, cEnd;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
    PetscMPIInt comm_size;
    ierr = MPI_Comm_size(rp->comm, &comm_size); CHKERRQ(ierr);
    ierr = PetscPrintf(rp->comm,
                       "\n-- CEED Benchmark Problem %d -- libCEED + PETSc --\n"
                       "  MPI:\n"
                       "    Hostname                           : %s\n"
                       "    Total ranks                        : %d\n"
                       "    Ranks per compute node             : %d\n"
                       "  PETSc:\n"
                       "    PETSc Vec Type                     : %s\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "    libCEED User Requested MemType     : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (P)       : %d\n"
                       "    Number of 1D Quadrature Points (Q) : %d\n"
                       "    Global nodes                       : %D\n"
                       "    Local Elements                     : %D\n"
                       "    Owned nodes                        : %D\n"
                       "    DoF per node                       : %D\n",
                       rp->bpchoice+1,
                       rp->hostname,
                       comm_size, rp->rankspernode,
                       vectype, usedresource,
                       CeedMemTypes[memtypebackend],
                       (rp->setmemtyperequest) ?
                       CeedMemTypes[rp->memtyperequested] : "none",
                       P, Q, gsize/rp->ncompu, cEnd - cStart, lsize/rp->ncompu,
                       rp->ncompu);
    CHKERRQ(ierr);
  }

  // Create RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  if (rp->memtyperequested == CEED_MEM_HOST) {
    ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArray(rhsloc, &r); CHKERRQ(ierr);
  }
  CeedVectorCreate(ceed, xlsize, &rhsceed);
  CeedVectorSetArray(rhsceed, rp->memtyperequested, CEED_USE_POINTER, r);

  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, rp->degree, rp->dim, rp->qextra,
                              rp->ncompu, gsize, xlsize, rp->bpchoice, ceeddata,
                              true, rhsceed, &target); CHKERRQ(ierr);

  // Gather RHS
  CeedVectorSyncArray(rhsceed, rp->memtyperequested);
  if (rp->memtyperequested == CEED_MEM_HOST) {
    ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  } else {
    ierr = VecCUDARestoreArray(rhsloc, &r); CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, rhsloc, ADD_VALUES, rhs); CHKERRQ(ierr);
  CeedVectorDestroy(&rhsceed);

  // Create the error Q-function
  CeedQFunctionCreateInterior(ceed, 1, bpOptions[rp->bpchoice].error,
                              bpOptions[rp->bpchoice].errorfname, &qferror);
  CeedQFunctionAddInput(qferror, "u", rp->ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qferror, "true_soln", rp->ncompu, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qferror, "error", rp->ncompu, CEED_EVAL_NONE);

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
  userO->comm = rp->comm;
  userO->dm = dm;
  userO->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &userO->Yloc); CHKERRQ(ierr);
  userO->xceed = ceeddata->xceed;
  userO->yceed = ceeddata->yceed;
  userO->op = ceeddata->opapply;
  userO->ceed = ceed;
  userO->memtype = rp->memtyperequested;
  if (rp->memtyperequested == CEED_MEM_HOST) {
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

  ierr = KSPCreate(rp->comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    if (rp->bpchoice == CEED_BP1 || rp->bpchoice == CEED_BP2) {
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
  if (rp->benchmark_mode) {
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1);
    CHKERRQ(ierr);
    my_rt_start = MPI_Wtime();
    ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
    my_rt = MPI_Wtime() - my_rt_start;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, rp->comm);
    CHKERRQ(ierr);
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                              rp->ksp_max_it_clip[0]);
      CHKERRQ(ierr);
    } else {
      ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                              rp->ksp_max_it_clip[1]);
      CHKERRQ(ierr);
    }
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // Timed solve
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)ksp); CHKERRQ(ierr);

  // -- Performance logging
  ierr = PetscLogStagePush(rp->solvestage); CHKERRQ(ierr);

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
    if (!rp->test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(rp->comm,
                         "  KSP:\n"
                         "    KSP Type                           : %s\n"
                         "    KSP Convergence                    : %s\n"
                         "    Total KSP Iterations               : %D\n"
                         "    Final rnorm                        : %e\n",
                         ksptype, KSPConvergedReasons[reason], its,
                         (double)rnorm); CHKERRQ(ierr);
    }
    if (!rp->test_mode) {
      ierr = PetscPrintf(rp->comm,"  Performance:\n"); CHKERRQ(ierr);
    }
    {
      PetscReal maxerror;
      ierr = ComputeErrorMax(userO, operror, X, target, &maxerror);
      CHKERRQ(ierr);
      PetscReal tol = 5e-2;
      if (!rp->test_mode || maxerror > tol) {
        ierr = MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, rp->comm);
        CHKERRQ(ierr);
        ierr = MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, rp->comm);
        CHKERRQ(ierr);
        ierr = PetscPrintf(rp->comm,
                           "    Pointwise Error (max)              : %e\n"
                           "    CG Solve Time                      : %g (%g) sec\n",
                           (double)maxerror, rt_max, rt_min); CHKERRQ(ierr);
      }
    }
    if (rp->benchmark_mode && (!rp->test_mode)) {
      ierr = PetscPrintf(rp->comm,
                         "    DoFs/Sec in CG                     : %g (%g) million\n",
                         1e-6*gsize*its/rt_max,
                         1e-6*gsize*its/rt_min); CHKERRQ(ierr);
    }
  }

  if (rp->write_solution) {
    PetscViewer vtkviewersoln;

    ierr = PetscViewerCreate(rp->comm, &vtkviewersoln); CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr, commsize;
  RunParams rp;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN];
  char *ceedresources[30];
  PetscInt num_ceedresources = 30;
  char hostname[PETSC_MAX_PATH_LEN];

  PetscInt dim = 3, melem[3] = {3, 3, 3};
  PetscInt num_degrees = 30, degree[30] = {}, num_localnodes = 2, localnodes[2] = {};
  PetscMPIInt rankspernode;
  PetscBool degree_set;
  bpType bpchoices[10];
  PetscInt num_bpchoices = 10;

  // Check PETSc CUDA support
  PetscBool petschavecuda;
  // *INDENT-OFF*
  #ifdef PETSC_HAVE_CUDA
  petschavecuda = PETSC_TRUE;
  #else
  petschavecuda = PETSC_FALSE;
  #endif
  // *INDENT-ON*

  // Initialize PETSc
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &commsize);
  if (ierr != MPI_SUCCESS) return ierr;
  #if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  {
    MPI_Comm splitcomm;
    ierr = MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &splitcomm);
    CHKERRQ(ierr);
    ierr = MPI_Comm_size(splitcomm, &rankspernode); CHKERRQ(ierr);
    ierr = MPI_Comm_free(&splitcomm); CHKERRQ(ierr);
  }
  #else
  rankspernode = -1; // Unknown
  #endif

  // Setup all parameters needed in Run()
  ierr = PetscMalloc1(1, &rp); CHKERRQ(ierr);
  rp->comm = comm;

  // Read command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  CHKERRQ(ierr);
  {
    PetscBool set;
    ierr = PetscOptionsEnumArray("-problem", "CEED benchmark problem to solve",
                                 NULL,
                                 bpTypes, (PetscEnum *)bpchoices, &num_bpchoices, &set);
    CHKERRQ(ierr);
    if (!set) {
      bpchoices[0] = CEED_BP1;
      num_bpchoices = 1;
    }
  }
  rp->test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, rp->test_mode, &rp->test_mode, NULL); CHKERRQ(ierr);
  rp->benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, rp->benchmark_mode, &rp->benchmark_mode, NULL);
  CHKERRQ(ierr);
  rp->write_solution = PETSC_FALSE;
  ierr = PetscOptionsBool("-write_solution", "Write solution for visualization",
                          NULL, rp->write_solution, &rp->write_solution, NULL);
  CHKERRQ(ierr);
  degree[0] = rp->test_mode ? 3 : 2;
  ierr = PetscOptionsIntArray("-degree",
                              "Polynomial degree of tensor product basis", NULL,
                              degree, &num_degrees, &degree_set); CHKERRQ(ierr);
  if (!degree_set)
    num_degrees = 1;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points", NULL,
                         rp->qextra, &rp->qextra, NULL); CHKERRQ(ierr);
  {
    PetscBool set;
    ierr = PetscOptionsStringArray("-ceed",
                                   "CEED resource specifier (comma-separated list)", NULL,
                                   ceedresources, &num_ceedresources, &set); CHKERRQ(ierr);
    if (!set) {
      ierr = PetscStrallocpy( "/cpu/self", &ceedresources[0]); CHKERRQ(ierr);
      num_ceedresources = 1;
    }
  }
  ierr = PetscGetHostName(hostname, sizeof hostname); CHKERRQ(ierr);
  ierr = PetscOptionsString("-hostname", "Hostname for output", NULL, hostname,
                            hostname, sizeof(hostname), NULL); CHKERRQ(ierr);
  rp->read_mesh = PETSC_FALSE;
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL, filename,
                            filename, sizeof(filename), &rp->read_mesh);
  CHKERRQ(ierr);
  rp->filename = filename;
  if (!rp->read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL,
                                melem, &tmp, NULL); CHKERRQ(ierr);
  }
  rp->memtyperequested = petschavecuda ? CEED_MEM_DEVICE : CEED_MEM_HOST;
  ierr = PetscOptionsEnum("-memtype", "CEED MemType requested", NULL, memTypes,
                          (PetscEnum)rp->memtyperequested,
                          (PetscEnum *)&rp->memtyperequested, &rp->setmemtyperequest);
  CHKERRQ(ierr);

  localnodes[0] = 1000;
  ierr = PetscOptionsIntArray("-local_nodes",
                              "Target number of locally owned nodes per "
                              "process (single value or min,max)",
                              NULL, localnodes, &num_localnodes, &rp->userlnodes);
  CHKERRQ(ierr);
  if (num_localnodes < 2)
    localnodes[1] = 2 * localnodes[0];
  {
    PetscInt two = 2;
    rp->ksp_max_it_clip[0] = 5;
    rp->ksp_max_it_clip[1] = 20;
    ierr = PetscOptionsIntArray("-ksp_max_it_clip",
                                "Min and max number of iterations to use during benchmarking",
                                NULL, rp->ksp_max_it_clip, &two, NULL); CHKERRQ(ierr);
  }

  if (rp->benchmark_mode && !degree_set) {
    PetscInt maxdegree = 8;
    ierr = PetscOptionsInt("-max_degree", "Max degree to run in benchmark mode",
                           NULL, maxdegree, &maxdegree, NULL);
    CHKERRQ(ierr);
    for (PetscInt i = 0; i < maxdegree; i++)
      degree[i] = i + 1;
    num_degrees = maxdegree;
  }
  {
    PetscBool flg;
    PetscInt p = rankspernode;
    ierr = PetscOptionsInt("-p", "Number of MPI ranks per node", NULL,
                           p, &p, &flg);
    CHKERRQ(ierr);
    if (flg) rankspernode = p;
  }

  ierr = PetscOptionsEnd();
  CHKERRQ(ierr);

  // Register PETSc logging stage
  ierr = PetscLogStageRegister("Solve Stage", &rp->solvestage);
  CHKERRQ(ierr);

  rp->petschavecuda = petschavecuda;
  rp->hostname = hostname;
  rp->dim = dim;
  rp->melem = melem;
  rp->rankspernode = rankspernode;

  for (PetscInt b = 0; b < num_bpchoices; b++) {
    rp->bpchoice = bpchoices[b];
    rp->ncompu = bpOptions[rp->bpchoice].ncompu;
    rp->qextra = bpOptions[rp->bpchoice].qextra;
    for (PetscInt d = 0; d < num_degrees; d++) {
      PetscInt deg = degree[d];
      for (PetscInt n = localnodes[0]; n < localnodes[1]; n *= 2) {
        rp->degree = deg;
        rp->localnodes = n;
        for (PetscInt c = 0; c < num_ceedresources; c++) {
          rp->ceedresource = ceedresources[c];
          ierr = Run(rp);
          CHKERRQ(ierr);
        }
      }
    }
  }
  // Clear memory
  ierr = PetscFree(rp); CHKERRQ(ierr);
  for (PetscInt i=0; i<num_ceedresources; i++) {
    ierr = PetscFree(ceedresources[i]); CHKERRQ(ierr);
  }
  return PetscFinalize();
}
