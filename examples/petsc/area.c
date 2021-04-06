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

//                        libCEED + PETSc Example: Surface Area
//
// This example demonstrates a simple usage of libCEED with PETSc to calculate
// the surface area of a simple closed surface, such as the one of a cube or a
// tensor-product discrete sphere via the mass operator.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make area [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//   Sequential:
//
//     ./area -problem cube -degree 3 -dm_refine 2
//     ./area -problem sphere -degree 3 -dm_refine 2
//
//   In parallel:
//
//     mpiexec -n 4 ./area -problem cube -degree 3 -dm_refine 2
//     mpiexec -n 4 ./area -problem sphere -degree 3 -dm_refine 2
//
//   The above example runs use 2 levels of refinement for the mesh.
//   Use -dm_refine k, for k levels of uniform refinement.
//
//TESTARGS -ceed {ceed_resource} -test -degree 3 -dm_refine 1

/// @file
/// libCEED example using the mass operator to compute a cube or a cubed-sphere surface area using PETSc with DMPlex
static const char help[] =
  "Compute surface area of a cube or a cubed-sphere using DMPlex in PETSc\n";

#include <ceed.h>
#include <petscdmplex.h>
#include <string.h>
#include "area.h"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscInt lsize, gsize, xlsize,
           qextra  = 1, // default number of extra quadrature points
           ncompx  = 3, // number of components of 3D physical coordinates
           ncompu  = 1, // dimension of field to which apply mass operator
           topodim = 2, // topological dimension of manifold
           degree  = 3; // default degree for finite element bases
  PetscBool read_mesh = PETSC_FALSE,
            test_mode = PETSC_FALSE,
            simplex = PETSC_FALSE;
  Vec U, Uloc, V, Vloc;
  DM  dm;
  UserO user;
  Ceed ceed;
  CeedData ceeddata;
  problemType problemChoice;
  VecType vectype;
  PetscMemType memtype;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read command line options
  ierr = PetscOptionsBegin(comm, NULL, "CEED surface area problem with PETSc",
                           NULL);
  CHKERRQ(ierr);
  problemChoice = SPHERE;
  ierr = PetscOptionsEnum("-problem",
                          "Problem to solve", NULL,
                          problemTypes, (PetscEnum)problemChoice,
                          (PetscEnum *)&problemChoice,
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices, or tensor product cells",
                          NULL, simplex, &simplex, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
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
    ierr = PetscObjectSetName((PetscObject)dm, problemTypes[problemChoice]);
    CHKERRQ(ierr);
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
    if (problemChoice == SPHERE) {
      ierr = ProjectToUnitSphere(dm); CHKERRQ(ierr);
    }
    // View DMPlex via runtime option
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  }

  // Create DM
  ierr = SetupDMByDegree(dm, degree, ncompu, topodim, false, (BCFunction)NULL);
  CHKERRQ(ierr);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &U); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &lsize); CHKERRQ(ierr);
  ierr = VecGetSize(U, &gsize); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetSize(Uloc, &xlsize); CHKERRQ(ierr);
  ierr = VecDuplicate(U, &V); CHKERRQ(ierr);
  ierr = VecDuplicate(Uloc, &Vloc); CHKERRQ(ierr);

  // Setup user structure
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);

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
                       "\n-- libCEED + PETSc Surface Area of a Manifold --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "    libCEED Backend MemType            : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n"
                       "    DoF per node                       : %D\n"
                       "    Global DoFs                        : %D\n",
                       usedresource, CeedMemTypes[memtypebackend], P, Q,
                       gsize/ncompu, ncompu, gsize); CHKERRQ(ierr);
  }

  // Setup libCEED's objects and apply setup operator
  ierr = PetscMalloc1(1, &ceeddata); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, topodim, qextra, ncompx, ncompu,
                              gsize, xlsize, problemOptions[problemChoice], ceeddata,
                              false, (CeedVector)NULL, (CeedVector *)NULL);
  CHKERRQ(ierr);

  // Setup output vector
  PetscScalar *v;
  ierr = VecZeroEntries(Vloc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(Vloc, &v, &memtype); CHKERRQ(ierr);
  CeedVectorSetArray(ceeddata->Yceed, MemTypeP2C(memtype), CEED_USE_POINTER, v);

  // Compute the mesh volume using the mass operator: area = 1^T \cdot M \cdot 1
  if (!test_mode) {
    ierr = PetscPrintf(comm,
                       "Computing the mesh area using the formula: area = 1^T M 1\n");
    CHKERRQ(ierr);
  }

  // Initialize u with ones
  CeedVectorSetValue(ceeddata->Xceed, 1.0);

  // Apply the mass operator: 'u' -> 'v'
  CeedOperatorApply(ceeddata->opApply, ceeddata->Xceed, ceeddata->Yceed,
                    CEED_REQUEST_IMMEDIATE);

  // Gather output vector
  CeedVectorTakeArray(ceeddata->Yceed, CEED_MEM_HOST, NULL);
  ierr = VecRestoreArrayAndMemType(Vloc, &v); CHKERRQ(ierr);
  ierr = VecZeroEntries(V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);

  // Compute and print the sum of the entries of 'v' giving the mesh surface area
  PetscScalar area;
  ierr = VecSum(V, &area); CHKERRQ(ierr);

  // Compute the exact surface area and print the result
  CeedScalar exact_surfarea = 4 * M_PI;
  if (problemChoice == CUBE) {
    PetscScalar l = 1.0/PetscSqrtReal(3.0); // half edge of the cube
    exact_surfarea = 6 * (2*l) * (2*l);
  }

  PetscReal error = fabs(area - exact_surfarea);
  PetscReal tol = 5e-6;
  if (!test_mode || error > tol) {
    ierr = PetscPrintf(comm, "Exact mesh surface area    : % .14g\n",
                       exact_surfarea);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Computed mesh surface area : % .14g\n", area);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Area error                 : % .14g\n", error);
    CHKERRQ(ierr);
  }

  // Cleanup
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&Uloc); CHKERRQ(ierr);
  ierr = VecDestroy(&V); CHKERRQ(ierr);
  ierr = VecDestroy(&Vloc); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceeddata); CHKERRQ(ierr);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
