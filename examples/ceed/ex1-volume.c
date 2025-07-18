// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                             libCEED Example 1
//
// This example illustrates a simple usage of libCEED to compute the volume of a 3D body using matrix-free application of a mass operator.
// Arbitrary mesh and solution degrees in 1D, 2D and 3D are supported from the same code.
//
// The example has no dependencies, and is designed to be self-contained.
// For additional examples that use external discretization libraries (MFEM, PETSc, etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command line argument (-ceed).
//
// Build with:
//
//     make ex1-volume [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ./ex1-volume
//     ./ex1-volume -ceed /cpu/self
//     ./ex1-volume -ceed /gpu/cuda
//
// Test in 1D-3D
//TESTARGS(name="1D User QFunction") -ceed {ceed_resource} -d 1 -t
//TESTARGS(name="2D User QFunction") -ceed {ceed_resource} -d 2 -t
//TESTARGS(name="3D User QFunction") -ceed {ceed_resource} -d 3 -t
//TESTARGS(name="1D Gallery QFunction") -ceed {ceed_resource} -d 1 -t -g
//TESTARGS(name="2D Gallery QFunction") -ceed {ceed_resource} -d 2 -t -g
//TESTARGS(name="3D Gallery QFunction") -ceed {ceed_resource} -d 3 -t -g

/// @file
/// libCEED example using mass operator to compute volume

#include "ex1-volume.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Auxiliary functions
int        GetCartesianMeshSize(CeedInt dim, CeedInt degree, CeedInt prob_size, CeedInt num_xyz[dim]);
int        BuildCartesianRestriction(Ceed ceed, CeedInt dim, CeedInt num_xyz[dim], CeedInt degree, CeedInt num_comp, CeedInt *size, CeedInt num_qpts,
                                     CeedElemRestriction *restriction, CeedElemRestriction *q_data_restriction);
int        SetCartesianMeshCoords(CeedInt dim, CeedInt num_xyz[dim], CeedInt mesh_degree, CeedVector mesh_coords);
CeedScalar TransformMeshCoords(CeedInt dim, CeedInt mesh_size, CeedVector mesh_coords);

// Main example
int main(int argc, const char *argv[]) {
  const char *ceed_spec   = "/cpu/self";
  CeedInt     dim         = 3;               // dimension of the mesh
  CeedInt     num_comp_x  = 3;               // number of x components
  CeedInt     mesh_degree = 4;               // polynomial degree for the mesh
  CeedInt     sol_degree  = 4;               // polynomial degree for the solution
  CeedInt     num_qpts    = sol_degree + 2;  // number of 1D quadrature points
  CeedInt     prob_size   = -1;              // approximate problem size
  CeedInt     help = 0, test = 0, gallery = 0, benchmark = 0;

  // Process command line arguments.
  for (int ia = 1; ia < argc; ia++) {
    // LCOV_EXCL_START
    int next_arg = ((ia + 1) < argc), parse_error = 0;
    if (!strcmp(argv[ia], "-h")) {
      help = 1;
    } else if (!strcmp(argv[ia], "-c") || !strcmp(argv[ia], "-ceed")) {
      parse_error = next_arg ? ceed_spec = argv[++ia], 0 : 1;
    } else if (!strcmp(argv[ia], "-d")) {
      parse_error = next_arg ? dim = atoi(argv[++ia]), 0 : 1;
      num_comp_x                   = dim;
    } else if (!strcmp(argv[ia], "-m")) {
      parse_error = next_arg ? mesh_degree = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia], "-p")) {
      parse_error = next_arg ? sol_degree = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia], "-q")) {
      parse_error = next_arg ? num_qpts = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia], "-s")) {
      parse_error = next_arg ? prob_size = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia], "-b")) {
      parse_error = next_arg ? benchmark = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia], "-t")) {
      test = 1;
    } else if (!strcmp(argv[ia], "-g")) {
      gallery = 1;
    }
    if (parse_error) {
      printf("Error parsing command line options.\n");
      return 1;
    }
    // LCOV_EXCL_STOP
  }
  if (prob_size < 0) prob_size = test ? 8 * 16 : 256 * 1024;

  // Print the values of all options:
  if (!test || help) {
    // LCOV_EXCL_START
    printf("Selected options: [command line option] : <current value>\n");
    printf("  Ceed specification     [-c] : %s\n", ceed_spec);
    printf("  Mesh dimension         [-d] : %" CeedInt_FMT "\n", dim);
    printf("  Mesh degree            [-m] : %" CeedInt_FMT "\n", mesh_degree);
    printf("  Solution degree        [-p] : %" CeedInt_FMT "\n", sol_degree);
    printf("  Num. 1D quadrature pts [-q] : %" CeedInt_FMT "\n", num_qpts);
    printf("  Approx. # unknowns     [-s] : %" CeedInt_FMT "\n", prob_size);
    printf("  QFunction source       [-g] : %s\n", gallery ? "gallery" : "header");
    if (help) {
      printf("Test/quiet mode is %s\n", (test ? "ON" : "OFF (use -t to enable)"));
      return 0;
    }
    printf("\n");
    // LCOV_EXCL_STOP
  }

  // Select appropriate backend and logical device based on the (-ceed) command line argument.
  Ceed ceed;

  CeedInit(ceed_spec, &ceed);

  // Construct the mesh and solution bases.
  CeedBasis mesh_basis, sol_basis;

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, mesh_degree + 1, num_qpts, CEED_GAUSS, &mesh_basis);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, sol_degree + 1, num_qpts, CEED_GAUSS, &sol_basis);

  // Determine the mesh size based on the given approximate problem size.
  CeedInt num_xyz[dim];

  GetCartesianMeshSize(dim, sol_degree, prob_size, num_xyz);
  if (!test) {
    // LCOV_EXCL_START
    printf("Mesh size: nx = %" CeedInt_FMT, num_xyz[0]);
    if (dim > 1) printf(", ny = %" CeedInt_FMT, num_xyz[1]);
    if (dim > 2) printf(", nz = %" CeedInt_FMT, num_xyz[2]);
    printf("\n");
    // LCOV_EXCL_STOP
  }

  // Build CeedElemRestriction objects describing the mesh and solution discrete representations.
  CeedInt             mesh_size, sol_size;
  CeedElemRestriction mesh_restriction, sol_restriction, q_data_restriction;

  BuildCartesianRestriction(ceed, dim, num_xyz, mesh_degree, num_comp_x, &mesh_size, num_qpts, &mesh_restriction, NULL);
  BuildCartesianRestriction(ceed, dim, num_xyz, sol_degree, 1, &sol_size, num_qpts, &sol_restriction, &q_data_restriction);
  if (!test) {
    // LCOV_EXCL_START
    printf("Number of mesh nodes     : %" CeedInt_FMT "\n", mesh_size / dim);
    printf("Number of solution nodes : %" CeedInt_FMT "\n", sol_size);
    // LCOV_EXCL_STOP
  }

  // Create a CeedVector with the mesh coordinates.
  CeedVector mesh_coords;

  CeedVectorCreate(ceed, mesh_size, &mesh_coords);
  SetCartesianMeshCoords(dim, num_xyz, mesh_degree, mesh_coords);

  // Apply a transformation to the mesh.
  CeedScalar exact_volume = TransformMeshCoords(dim, mesh_size, mesh_coords);

  // Context data to be passed to the 'build_mass' QFunction.
  CeedQFunctionContext build_ctx;
  struct BuildContext  build_ctx_data;

  build_ctx_data.dim = build_ctx_data.space_dim = dim;
  CeedQFunctionContextCreate(ceed, &build_ctx);
  CeedQFunctionContextSetData(build_ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(build_ctx_data), &build_ctx_data);

  // Create the QFunction that builds the mass operator (i.e. computes its quadrature data) and set its context data.
  CeedQFunction qf_build;

  if (gallery) {
    // This creates the QFunction via the gallery.
    char name[13] = "";
    snprintf(name, sizeof name, "Mass%" CeedInt_FMT "DBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name, &qf_build);
  } else {
    // This creates the QFunction directly.
    CeedQFunctionCreateInterior(ceed, 1, build_mass, build_mass_loc, &qf_build);
    CeedQFunctionAddInput(qf_build, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_build, "weights", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_build, "qdata", 1, CEED_EVAL_NONE);
    CeedQFunctionSetContext(qf_build, build_ctx);
  }

  // Create the operator that builds the quadrature data for the mass operator.
  CeedOperator op_build;

  CeedOperatorCreate(ceed, qf_build, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_build);
  CeedOperatorSetField(op_build, "dx", mesh_restriction, mesh_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_build, "weights", CEED_ELEMRESTRICTION_NONE, mesh_basis, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_build, "qdata", q_data_restriction, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  // Compute the quadrature data for the mass operator.
  CeedVector q_data;
  CeedInt    elem_qpts = CeedIntPow(num_qpts, dim);
  CeedInt    num_elem  = 1;

  for (CeedInt d = 0; d < dim; d++) num_elem *= num_xyz[d];
  CeedVectorCreate(ceed, num_elem * elem_qpts, &q_data);
  CeedOperatorApply(op_build, mesh_coords, q_data, CEED_REQUEST_IMMEDIATE);

  // Create the QFunction that defines the action of the mass operator.
  CeedQFunction qf_apply;

  if (gallery) {
    // This creates the QFunction via the gallery.
    CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_apply);
  } else {
    // This creates the QFunction directly.
    CeedQFunctionCreateInterior(ceed, 1, apply_mass, apply_mass_loc, &qf_apply);
    CeedQFunctionAddInput(qf_apply, "u", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_apply, "qdata", 1, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_apply, "v", 1, CEED_EVAL_INTERP);
  }

  // Create the mass operator.
  CeedOperator op_apply;

  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);
  CeedOperatorSetField(op_apply, "u", sol_restriction, sol_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", q_data_restriction, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_apply, "v", sol_restriction, sol_basis, CEED_VECTOR_ACTIVE);

  // Create auxiliary solution-size vectors.
  CeedVector u, v;

  CeedVectorCreate(ceed, sol_size, &u);
  CeedVectorCreate(ceed, sol_size, &v);

  // Initialize 'u' with ones.
  CeedVectorSetValue(u, 1.0);

  // Compute the mesh volume using the mass operator: volume = 1^T \cdot M \cdot 1
  CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);

  // Benchmark runs
  if (!test && benchmark) {
    // LCOV_EXCL_START
    printf(" Executing %d benchmarking runs...\n", benchmark);
    // LCOV_EXCL_STOP
  }
  for (CeedInt i = 0; i < benchmark; i++) {
    // LCOV_EXCL_START
    CeedOperatorApply(op_apply, u, v, CEED_REQUEST_IMMEDIATE);
    // LCOV_EXCL_STOP
  }

  // Compute and print the sum of the entries of 'v' giving the mesh volume.
  CeedScalar volume = 0.;

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < sol_size; i++) volume += v_array[i];
    CeedVectorRestoreArrayRead(v, &v_array);
  }
  if (!test) {
    // LCOV_EXCL_START
    printf(" done.\n");
    printf("Exact mesh volume    : % .14g\n", exact_volume);
    printf("Computed mesh volume : % .14g\n", volume);
    printf("Volume error         : % .14g\n", volume - exact_volume);
    // LCOV_EXCL_STOP
  } else {
    CeedScalar tol = (dim == 1 ? 200. * CEED_EPSILON : dim == 2 ? 1E-5 : 1E-5);

    if (fabs(volume - exact_volume) > tol) printf("Volume error : % .1e\n", volume - exact_volume);
  }

  // Free dynamically allocated memory.
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&mesh_coords);
  CeedOperatorDestroy(&op_apply);
  CeedQFunctionDestroy(&qf_apply);
  CeedQFunctionContextDestroy(&build_ctx);
  CeedOperatorDestroy(&op_build);
  CeedQFunctionDestroy(&qf_build);
  CeedElemRestrictionDestroy(&sol_restriction);
  CeedElemRestrictionDestroy(&mesh_restriction);
  CeedElemRestrictionDestroy(&q_data_restriction);
  CeedBasisDestroy(&sol_basis);
  CeedBasisDestroy(&mesh_basis);
  CeedDestroy(&ceed);
  return 0;
}

int GetCartesianMeshSize(CeedInt dim, CeedInt degree, CeedInt prob_size, CeedInt num_xyz[dim]) {
  // Use the approximate formula:
  //    prob_size ~ num_elem * degree^dim
  CeedInt num_elem = prob_size / CeedIntPow(degree, dim);
  CeedInt s        = 0;  // find s: num_elem/2 < 2^s <= num_elem

  while (num_elem > 1) {
    num_elem /= 2;
    s++;
  }
  CeedInt r = s % dim;

  for (CeedInt d = 0; d < dim; d++) {
    CeedInt sd = s / dim;

    if (r > 0) {
      sd++;
      r--;
    }
    num_xyz[d] = 1 << sd;
  }
  return 0;
}

int BuildCartesianRestriction(Ceed ceed, CeedInt dim, CeedInt num_xyz[dim], CeedInt degree, CeedInt num_comp, CeedInt *size, CeedInt num_qpts,
                              CeedElemRestriction *restriction, CeedElemRestriction *q_data_restriction) {
  CeedInt p         = degree + 1;
  CeedInt num_nodes = CeedIntPow(p, dim);         // number of scalar nodes per element
  CeedInt elem_qpts = CeedIntPow(num_qpts, dim);  // number of qpts per element
  CeedInt nd[3], num_elem = 1, scalar_size = 1;

  for (CeedInt d = 0; d < dim; d++) {
    num_elem *= num_xyz[d];
    nd[d] = num_xyz[d] * (p - 1) + 1;
    scalar_size *= nd[d];
  }
  *size = scalar_size * num_comp;
  // elem:         0             1                 n-1
  //           |---*-...-*---|---*-...-*---|- ... -|--...--|
  // num_nodes:   0   1    p-1  p  p+1       2*p             n*p
  CeedInt *elem_nodes = malloc(sizeof(CeedInt) * num_elem * num_nodes);

  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt e_xyz[3] = {1, 1, 1}, re = e;

    for (CeedInt d = 0; d < dim; d++) {
      e_xyz[d] = re % num_xyz[d];
      re /= num_xyz[d];
    }
    CeedInt *local_elem_nodes = elem_nodes + e * num_nodes;

    for (CeedInt l_nodes = 0; l_nodes < num_nodes; l_nodes++) {
      CeedInt g_nodes = 0, g_nodes_stride = 1, r_nodes = l_nodes;

      for (CeedInt d = 0; d < dim; d++) {
        g_nodes += (e_xyz[d] * (p - 1) + r_nodes % p) * g_nodes_stride;
        g_nodes_stride *= nd[d];
        r_nodes /= p;
      }
      local_elem_nodes[l_nodes] = g_nodes;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, num_nodes, num_comp, scalar_size, num_comp * scalar_size, CEED_MEM_HOST, CEED_COPY_VALUES, elem_nodes,
                            restriction);
  if (q_data_restriction) {
    CeedElemRestrictionCreateStrided(ceed, num_elem, elem_qpts, num_comp, num_comp * elem_qpts * num_elem, CEED_STRIDES_BACKEND, q_data_restriction);
  }
  free(elem_nodes);
  return 0;
}

int SetCartesianMeshCoords(CeedInt dim, CeedInt num_xyz[dim], CeedInt mesh_degree, CeedVector mesh_coords) {
  CeedInt p = mesh_degree + 1;
  CeedInt nd[3], scalar_size = 1;

  for (CeedInt d = 0; d < dim; d++) {
    nd[d] = num_xyz[d] * (p - 1) + 1;
    scalar_size *= nd[d];
  }
  CeedScalar *coords;

  CeedVectorGetArrayWrite(mesh_coords, CEED_MEM_HOST, &coords);
  CeedScalar *nodes = malloc(sizeof(CeedScalar) * p);

  // The H1 basis uses Lobatto quadrature points as nodes.
  CeedLobattoQuadrature(p, nodes, NULL);  // nodes are in [-1,1]
  for (CeedInt i = 0; i < p; i++) nodes[i] = 0.5 + 0.5 * nodes[i];
  for (CeedInt gs_nodes = 0; gs_nodes < scalar_size; gs_nodes++) {
    CeedInt r_nodes = gs_nodes;

    for (CeedInt d = 0; d < dim; d++) {
      CeedInt d_1d = r_nodes % nd[d];

      coords[gs_nodes + scalar_size * d] = ((d_1d / (p - 1)) + nodes[d_1d % (p - 1)]) / num_xyz[d];
      r_nodes /= nd[d];
    }
  }
  free(nodes);
  CeedVectorRestoreArray(mesh_coords, &coords);
  return 0;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#endif

CeedScalar TransformMeshCoords(CeedInt dim, CeedInt mesh_size, CeedVector mesh_coords) {
  CeedScalar  exact_volume;
  CeedScalar *coords;

  CeedVectorGetArray(mesh_coords, CEED_MEM_HOST, &coords);
  if (dim == 1) {
    for (CeedInt i = 0; i < mesh_size; i++) {
      // map [0,1] to [0,1] varying the mesh density
      coords[i] = 0.5 + 1. / sqrt(3.) * sin((2. / 3.) * M_PI * (coords[i] - 0.5));
    }
    exact_volume = 1.;
  } else {
    CeedInt num_nodes = mesh_size / dim;

    for (CeedInt i = 0; i < num_nodes; i++) {
      // map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
      // coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
      CeedScalar u = coords[i], v = coords[i + num_nodes];

      u                     = 1. + u;
      v                     = M_PI_2 * v;
      coords[i]             = u * cos(v);
      coords[i + num_nodes] = u * sin(v);
    }
    exact_volume = 3. / 4. * M_PI;
  }
  CeedVectorRestoreArray(mesh_coords, &coords);
  return exact_volume;
}
