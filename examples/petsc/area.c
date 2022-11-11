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
static const char help[] = "Compute surface area of a cube or a cubed-sphere using DMPlex in PETSc\n";

#include "area.h"

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <stdbool.h>
#include <string.h>

#include "include/areaproblemdata.h"
#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/petscversion.h"
#include "include/structs.h"

#if PETSC_VERSION_LT(3, 12, 0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
  MPI_Comm comm;
  char     filename[PETSC_MAX_PATH_LEN], ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscInt l_size, g_size, xl_size,
      q_extra                    = 1,  // default number of extra quadrature points
      num_comp_x                 = 3,  // number of components of 3D physical coordinates
      num_comp_u                 = 1,  // dimension of field to which apply mass operator
      topo_dim                   = 2,  // topological dimension of manifold
      degree                     = 3;  // default degree for finite element bases
  PetscBool            read_mesh = PETSC_FALSE, test_mode = PETSC_FALSE, simplex = PETSC_FALSE;
  Vec                  U, U_loc, V, V_loc;
  DM                   dm;
  OperatorApplyContext op_apply_ctx;
  Ceed                 ceed;
  CeedData             ceed_data;
  ProblemType          problem_choice;
  VecType              vec_type;
  PetscMemType         mem_type;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Read command line options
  PetscOptionsBegin(comm, NULL, "CEED surface area problem with PETSc", NULL);
  problem_choice = SPHERE;
  PetscCall(PetscOptionsEnum("-problem", "Problem to solve", NULL, problem_types, (PetscEnum)problem_choice, (PetscEnum *)&problem_choice, NULL));
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &read_mesh));
  PetscCall(PetscOptionsBool("-simplex", "Use simplices, or tensor product cells", NULL, simplex, &simplex, NULL));
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  PetscOptionsEnd();

  // Setup DM
  if (read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE, &dm));
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface, not a box
    PetscCall(DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topo_dim, simplex, 1., &dm));
    if (problem_choice == CUBE) {
      PetscCall(DMPlexCreateCoordinateSpace(dm, 1, NULL));
    }
    // Set the object name
    PetscCall(PetscObjectSetName((PetscObject)dm, problem_types[problem_choice]));
    // Refine DMPlex with uniform refinement using runtime option -dm_refine
    PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
    PetscCall(DMSetFromOptions(dm));
    // View DMPlex via runtime option
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  }

  // Create DM
  PetscCall(SetupDMByDegree(dm, degree, q_extra, num_comp_u, topo_dim, false));

  // Create vectors
  PetscCall(DMCreateGlobalVector(dm, &U));
  PetscCall(VecGetLocalSize(U, &l_size));
  PetscCall(VecGetSize(U, &g_size));
  PetscCall(DMCreateLocalVector(dm, &U_loc));
  PetscCall(VecGetSize(U_loc, &xl_size));
  PetscCall(VecDuplicate(U, &V));
  PetscCall(VecDuplicate(U_loc, &V_loc));

  // Setup op_apply_ctx structure
  PetscCall(PetscMalloc1(1, &op_apply_ctx));

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  PetscCall(DMGetVecType(dm, &vec_type));
  if (!vec_type) {  // Not yet set by op_apply_ctx -dm_vec_type
    switch (mem_type_backend) {
      case CEED_MEM_HOST:
        vec_type = VECSTANDARD;
        break;
      case CEED_MEM_DEVICE: {
        const char *resolved;
        CeedGetResource(ceed, &resolved);
        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
        else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
        else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
        else vec_type = VECSTANDARD;
      }
    }
    PetscCall(DMSetVecType(dm, vec_type));
  }

  // Print summary
  if (!test_mode) {
    PetscInt    P = degree + 1, Q = P + q_extra;
    const char *used_resource;
    CeedGetResource(ceed, &used_resource);
    PetscCall(PetscPrintf(comm,
                          "\n-- libCEED + PETSc Surface Area of a Manifold --\n"
                          "  libCEED:\n"
                          "    libCEED Backend                         : %s\n"
                          "    libCEED Backend MemType                 : %s\n"
                          "  Mesh:\n"
                          "    Solution Order (P)                      : %" CeedInt_FMT "\n"
                          "    Quadrature Order (Q)                    : %" CeedInt_FMT "\n"
                          "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                          "    Global nodes                            : %" PetscInt_FMT "\n"
                          "    DoF per node                            : %" PetscInt_FMT "\n"
                          "    Global DoFs                             : %" PetscInt_FMT "\n",
                          used_resource, CeedMemTypes[mem_type_backend], P, Q, q_extra, g_size / num_comp_u, num_comp_u, g_size));
  }

  // Setup libCEED's objects and apply setup operator
  PetscCall(PetscMalloc1(1, &ceed_data));
  PetscCall(SetupLibceedByDegree(dm, ceed, degree, topo_dim, q_extra, num_comp_x, num_comp_u, g_size, xl_size, problem_options[problem_choice],
                                 ceed_data, false, (CeedVector)NULL, (CeedVector *)NULL));

  // Setup output vector
  PetscScalar *v;
  PetscCall(VecZeroEntries(V_loc));
  PetscCall(VecGetArrayAndMemType(V_loc, &v, &mem_type));
  CeedVectorSetArray(ceed_data->y_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, v);

  // Compute the mesh volume using the mass operator: area = 1^T \cdot M \cdot 1
  if (!test_mode) {
    PetscCall(PetscPrintf(comm, "Computing the mesh area using the formula: area = 1^T M 1\n"));
  }

  // Initialize u with ones
  CeedVectorSetValue(ceed_data->x_ceed, 1.0);

  // Apply the mass operator: 'u' -> 'v'
  CeedOperatorApply(ceed_data->op_apply, ceed_data->x_ceed, ceed_data->y_ceed, CEED_REQUEST_IMMEDIATE);

  // Gather output vector
  CeedVectorTakeArray(ceed_data->y_ceed, CEED_MEM_HOST, NULL);
  PetscCall(VecRestoreArrayAndMemType(V_loc, &v));
  PetscCall(VecZeroEntries(V));
  PetscCall(DMLocalToGlobalBegin(dm, V_loc, ADD_VALUES, V));
  PetscCall(DMLocalToGlobalEnd(dm, V_loc, ADD_VALUES, V));

  // Compute and print the sum of the entries of 'v' giving the mesh surface area
  PetscScalar area;
  PetscCall(VecSum(V, &area));

  // Compute the exact surface area and print the result
  CeedScalar exact_surface_area = 4 * M_PI;
  if (problem_choice == CUBE) {
    exact_surface_area = 6 * 2 * 2;  // surface of [-1, 1]^3
  }

  PetscReal error = fabs(area - exact_surface_area);
  PetscReal tol   = 5e-6;
  if (!test_mode || error > tol) {
    PetscCall(PetscPrintf(comm, "Exact mesh surface area                     : % .14g\n", exact_surface_area));
    PetscCall(PetscPrintf(comm, "Computed mesh surface area                  : % .14g\n", area));
    PetscCall(PetscPrintf(comm, "Area error                                  : % .14g\n", error));
  }

  // Cleanup
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&U_loc));
  PetscCall(VecDestroy(&V));
  PetscCall(VecDestroy(&V_loc));
  PetscCall(PetscFree(op_apply_ctx));
  PetscCall(CeedDataDestroy(0, ceed_data));
  CeedDestroy(&ceed);
  return PetscFinalize();
}
