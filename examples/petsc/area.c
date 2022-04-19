// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

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

#include <stdbool.h>
#include <string.h>
#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>

#include "area.h"
#include "include/areaproblemdata.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/matops.h"
#include "include/structs.h"
#include "include/libceedsetup.h"

#if PETSC_VERSION_LT(3,12,0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscInt l_size, g_size, xl_size,
           q_extra     = 1, // default number of extra quadrature points
           num_comp_x  = 3, // number of components of 3D physical coordinates
           num_comp_u  = 1, // dimension of field to which apply mass operator
           topo_dim    = 2, // topological dimension of manifold
           degree      = 3; // default degree for finite element bases
  PetscBool read_mesh = PETSC_FALSE,
            test_mode = PETSC_FALSE,
            simplex = PETSC_FALSE;
  Vec U, U_loc, V, V_loc;
  DM  dm;
  UserO user;
  Ceed ceed;
  CeedData ceed_data;
  ProblemType problem_choice;
  VecType vec_type;
  PetscMemType mem_type;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED surface area problem with PETSc", NULL);
  problem_choice = SPHERE;
  ierr = PetscOptionsEnum("-problem",
                          "Problem to solve", NULL,
                          problem_types, (PetscEnum)problem_choice,
                          (PetscEnum *)&problem_choice,
                          NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-q_extra", "Number of extra quadrature points",
                         NULL, q_extra, &q_extra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceed_resource, ceed_resource,
                            sizeof(ceed_resource), NULL); CHKERRQ(ierr);
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
  PetscOptionsEnd();

  // Setup DM
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE,
                                &dm);
    CHKERRQ(ierr);
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface, not a box
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topo_dim, simplex, 1., &dm);
    CHKERRQ(ierr);
    if (problem_choice == CUBE) {
      ierr = DMPlexCreateCoordinateSpace(dm, 1, NULL); CHKERRQ(ierr);
    }
    // Set the object name
    ierr = PetscObjectSetName((PetscObject)dm, problem_types[problem_choice]);
    CHKERRQ(ierr);
    // Refine DMPlex with uniform refinement using runtime option -dm_refine
    ierr = DMPlexSetRefinementUniform(dm, PETSC_TRUE); CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
    // View DMPlex via runtime option
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  }

  // Create DM
  ierr = SetupDMByDegree(dm, degree, q_extra, num_comp_u, topo_dim, false,
                         (BCFunction)NULL);
  CHKERRQ(ierr);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &U); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U, &l_size); CHKERRQ(ierr);
  ierr = VecGetSize(U, &g_size); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetSize(U_loc, &xl_size); CHKERRQ(ierr);
  ierr = VecDuplicate(U, &V); CHKERRQ(ierr);
  ierr = VecDuplicate(U_loc, &V_loc); CHKERRQ(ierr);

  // Setup user structure
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);
  if (!vec_type) { // Not yet set by user -dm_vec_type
    switch (mem_type_backend) {
    case CEED_MEM_HOST: vec_type = VECSTANDARD; break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip/occa"))
        vec_type = VECSTANDARD; // https://github.com/CEED/libCEED/issues/678
      else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
      else vec_type = VECSTANDARD;
    }
    }
    ierr = DMSetVecType(dm, vec_type); CHKERRQ(ierr);
  }

  // Print summary
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    ierr = PetscPrintf(comm,
                       "\n-- libCEED + PETSc Surface Area of a Manifold --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                         : %s\n"
                       "    libCEED Backend MemType                 : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)            : %" CeedInt_FMT "\n"
                       "    Number of 1D Quadrature Points (q)      : %" CeedInt_FMT "\n"
                       "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                       "    Global nodes                            : %" PetscInt_FMT "\n"
                       "    DoF per node                            : %" PetscInt_FMT "\n"
                       "    Global DoFs                             : %" PetscInt_FMT "\n",
                       used_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra,
                       g_size/num_comp_u, num_comp_u, g_size); CHKERRQ(ierr);
  }

  // Setup libCEED's objects and apply setup operator
  ierr = PetscMalloc1(1, &ceed_data); CHKERRQ(ierr);
  ierr = SetupLibceedByDegree(dm, ceed, degree, topo_dim, q_extra, num_comp_x,
                              num_comp_u, g_size, xl_size,
                              problem_options[problem_choice], ceed_data,
                              false, (CeedVector)NULL, (CeedVector *)NULL);
  CHKERRQ(ierr);

  // Setup output vector
  PetscScalar *v;
  ierr = VecZeroEntries(V_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(V_loc, &v, &mem_type); CHKERRQ(ierr);
  CeedVectorSetArray(ceed_data->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER,
                     v);

  // Compute the mesh volume using the mass operator: area = 1^T \cdot M \cdot 1
  if (!test_mode) {
    ierr = PetscPrintf(comm,
                       "Computing the mesh area using the formula:  area = 1^T M 1\n");
    CHKERRQ(ierr);
  }

  // Initialize u with ones
  CeedVectorSetValue(ceed_data->x_ceed, 1.0);

  // Apply the mass operator: 'u' -> 'v'
  CeedOperatorApply(ceed_data->op_apply, ceed_data->x_ceed, ceed_data->y_ceed,
                    CEED_REQUEST_IMMEDIATE);

  // Gather output vector
  CeedVectorTakeArray(ceed_data->y_ceed, CEED_MEM_HOST, NULL);
  ierr = VecRestoreArrayAndMemType(V_loc, &v); CHKERRQ(ierr);
  ierr = VecZeroEntries(V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, V_loc, ADD_VALUES, V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, V_loc, ADD_VALUES, V); CHKERRQ(ierr);

  // Compute and print the sum of the entries of 'v' giving the mesh surface area
  PetscScalar area;
  ierr = VecSum(V, &area); CHKERRQ(ierr);

  // Compute the exact surface area and print the result
  CeedScalar exact_surface_area = 4 * M_PI;
  if (problem_choice == CUBE) {
    exact_surface_area = 6 * 2 * 2; // surface of [-1, 1]^3
  }

  PetscReal error = fabs(area - exact_surface_area);
  PetscReal tol = 5e-6;
  if (!test_mode || error > tol) {
    ierr = PetscPrintf(comm,
                       "Exact mesh surface area                     : % .14g\n",
                       exact_surface_area);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "Computed mesh surface area                  : % .14g\n", area);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,
                       "Area error                                  : % .14g\n", error);
    CHKERRQ(ierr);
  }

  // Cleanup
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&U); CHKERRQ(ierr);
  ierr = VecDestroy(&U_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&V); CHKERRQ(ierr);
  ierr = VecDestroy(&V_loc); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = CeedDataDestroy(0, ceed_data); CHKERRQ(ierr);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
