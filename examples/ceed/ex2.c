// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

//                             libCEED Example 2
//
// This example illustrates a simple usage of libCEED to compute the surface
// area of a 3D body using matrix-free application of a diff operator.
// Arbitrary mesh and solution orders in 1D, 2D and 3D are supported from the
// same code.
//
// The example has no dependencies, and is designed to be self-contained. For
// additional examples that use external discretization libraries (MFEM, PETSc,
// etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command
// line argument (-ceed).
//
// Build with:
//
//     make ex2 [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     ex2
//     ex2 -ceed /cpu/self
//     ex2 -ceed /gpu/occa
//     ex2 -ceed /cpu/occa
//     ex2 -ceed /omp/occa
//     ex2 -ceed /ocl/occa
//     ex2 -m ../../../mfem/data/fichera.mesh
//     ex2 -m ../../../mfem/data/star.vtk -o 3
//     ex2 -m ../../../mfem/data/inline-segment.mesh -o 8
//
// Next line is grep'd from tap.sh to set its arguments
// Test in 1D-3D
//TESTARGS -ceed {ceed_resource} -d 2 -t
//TESTARGS -ceed {ceed_resource} -d 1 -t -g
//TESTARGS -ceed {ceed_resource} -d 2 -t -g
//TESTARGS -ceed {ceed_resource} -d 3 -t -g

/// @file
/// libCEED example using diffusion operator to compute surface area

#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ex2.h"

// Auxiliary functions.
int GetCartesianMeshSize(int dim, int order, int prob_size, int nxyz[3]);
int BuildCartesianRestriction(Ceed ceed, int dim, int nxyz[3], int order,
                              int ncomp, CeedInt *size, CeedInt num_qpts,
                              CeedElemRestriction *restr,
                              CeedElemRestriction *restr_i);
int SetCartesianMeshCoords(int dim, int nxyz[3], int mesh_order,
                           CeedVector mesh_coords);
CeedScalar TransformMeshCoords(int dim, int mesh_size, CeedVector mesh_coords);


int main(int argc, const char *argv[]) {
  const char *ceed_spec = "/cpu/self";
  int dim        = 3;           // dimension of the mesh
  int ncompx     = 3;           // number of x components
  int mesh_order = 4;           // polynomial degree for the mesh
  int sol_order  = 4;           // polynomial degree for the solution
  int num_qpts   = sol_order+2; // number of 1D quadrature points
  int prob_size  = -1;          // approximate problem size
  int help = 0, test = 0, gallery = 0;

  // Process command line arguments.
  for (int ia = 1; ia < argc; ia++) {
    int next_arg = ((ia+1) < argc), parse_error = 0;
    if (!strcmp(argv[ia],"-h")) {
      help = 1;
    } else if (!strcmp(argv[ia],"-c") || !strcmp(argv[ia],"-ceed")) {
      parse_error = next_arg ? ceed_spec = argv[++ia], 0 : 1;
    } else if (!strcmp(argv[ia],"-d")) {
      parse_error = next_arg ? dim = atoi(argv[++ia]), 0 : 1;
      ncompx = dim;
    } else if (!strcmp(argv[ia],"-m")) {
      parse_error = next_arg ? mesh_order = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia],"-o")) {
      parse_error = next_arg ? sol_order = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia],"-q")) {
      parse_error = next_arg ? num_qpts = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia],"-s")) {
      parse_error = next_arg ? prob_size = atoi(argv[++ia]), 0 : 1;
    } else if (!strcmp(argv[ia],"-t")) {
      test = 1;
    } else if (!strcmp(argv[ia],"-g")) {
      gallery = 1;
    }
    if (parse_error) {
      printf("Error parsing command line options.\n");
      return 1;
    }
  }
  if (prob_size < 0) prob_size = test ? 16*16*dim*dim : 256*1024;

  // Set mesh_order = sol_order.
  mesh_order = fmax(mesh_order, sol_order);
  sol_order = mesh_order;

  // Print the values of all options:
  if (!test || help) {
    printf("Selected options: [command line option] : <current value>\n");
    printf("  Ceed specification [-c] : %s\n", ceed_spec);
    printf("  Mesh dimension     [-d] : %d\n", dim);
    printf("  Mesh order         [-m] : %d\n", mesh_order);
    printf("  Solution order     [-o] : %d\n", sol_order);
    printf("  Num. 1D quadr. pts [-q] : %d\n", num_qpts);
    printf("  Approx. # unknowns [-s] : %d\n", prob_size);
    printf("  QFunction source   [-g] : %s\n", gallery?"gallery":"header");
    if (help) {
      printf("Test/quiet mode is %s\n", (test?"ON":"OFF (use -t to enable)"));
      return 0;
    }
    printf("\n");
  }

  // Select appropriate backend and logical device based on the <ceed-spec>
  // command line argument.
  Ceed ceed;
  CeedInit(ceed_spec, &ceed);

  // Construct the mesh and solution bases.
  CeedBasis mesh_basis, sol_basis;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, ncompx, mesh_order+1, num_qpts,
                                  CEED_GAUSS, &mesh_basis);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, sol_order+1, num_qpts,
                                  CEED_GAUSS, &sol_basis);

  // Determine the mesh size based on the given approximate problem size.
  int nxyz[3];
  GetCartesianMeshSize(dim, sol_order, prob_size, nxyz);

  if (!test) {
    printf("Mesh size: nx = %d", nxyz[0]);
    if (dim > 1) { printf(", ny = %d", nxyz[1]); }
    if (dim > 2) { printf(", nz = %d", nxyz[2]); }
    printf("\n");
  }

  // Build CeedElemRestriction objects describing the mesh and solution discrete
  // representations.
  CeedInt mesh_size, sol_size;
  CeedElemRestriction mesh_restr, sol_restr, mesh_restr_i, sol_restr_i,
                      qdata_restr_i;
  BuildCartesianRestriction(ceed, dim, nxyz, mesh_order, ncompx, &mesh_size,
                            num_qpts, &mesh_restr, &mesh_restr_i);
  BuildCartesianRestriction(ceed, dim, nxyz, sol_order, dim*(dim+1)/2,
                            &sol_size, num_qpts, NULL, &qdata_restr_i);
  BuildCartesianRestriction(ceed, dim, nxyz, sol_order, 1, &sol_size,
                            num_qpts, &sol_restr, &sol_restr_i);
  if (!test) {
    printf("Number of mesh nodes     : %d\n", mesh_size/dim);
    printf("Number of solution nodes : %d\n", sol_size);
  }

  // Create a CeedVector with the mesh coordinates.
  CeedVector mesh_coords;
  CeedVectorCreate(ceed, mesh_size, &mesh_coords);
  SetCartesianMeshCoords(dim, nxyz, mesh_order, mesh_coords);

  // Apply a transformation to the mesh.
  CeedScalar exact_sa = TransformMeshCoords(dim, mesh_size, mesh_coords);

  // Context data to be passed to the 'f_build_diff' Q-function.
  struct BuildContext build_ctx;
  build_ctx.dim = build_ctx.space_dim = dim;

  // Create the Q-function that builds the diffusion operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunction build_qfunc;
  switch (gallery) {
  case 0:
    // This creates the QFunction directly.
    CeedQFunctionCreateInterior(ceed, 1, f_build_diff,
                                f_build_diff_loc, &build_qfunc);
    CeedQFunctionAddInput(build_qfunc, "dx", ncompx*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(build_qfunc, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionSetContext(build_qfunc, &build_ctx, sizeof(build_ctx));
    break;
  case 1: {
    // This creates the QFunction via the gallery.
    char name[16] = "";
    snprintf(name, sizeof name, "Poisson%dDBuild", dim);
    CeedQFunctionCreateInteriorByName(ceed, name, &build_qfunc);
    break;
  }
  }

  // Create the operator that builds the quadrature data for the diffusion
  // operator.
  CeedOperator build_oper;
  CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
  CeedOperatorSetField(build_oper, "dx", mesh_restr, CEED_NOTRANSPOSE,
                       mesh_basis,CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(build_oper, "weights", mesh_restr_i, CEED_NOTRANSPOSE,
                       mesh_basis, CEED_VECTOR_NONE);
  CeedOperatorSetField(build_oper, "qdata", qdata_restr_i, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Compute the quadrature data for the diffusion operator.
  CeedVector qdata;
  CeedInt elem_qpts = CeedIntPow(num_qpts, dim);
  CeedInt num_elem = 1;
  for (int d = 0; d < dim; d++)
    num_elem *= nxyz[d];
  CeedVectorCreate(ceed, num_elem*elem_qpts*dim*(dim+1)/2, &qdata);
  if (!test) {
    printf("Computing the quadrature data for the diffusion operator ...");
    fflush(stdout);
  }
  CeedOperatorApply(build_oper, mesh_coords, qdata,
                    CEED_REQUEST_IMMEDIATE);
  if (!test) {
    printf(" done.\n");
  }

  // Create the Q-function that defines the action of the diffusion operator.
  CeedQFunction apply_qfunc;
  switch (gallery) {
  case 0:
    // This creates the QFunction directly.
    CeedQFunctionCreateInterior(ceed, 1, f_apply_diff,
                                f_apply_diff_loc, &apply_qfunc);
    CeedQFunctionAddInput(apply_qfunc, "du", dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(apply_qfunc, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(apply_qfunc, "dv", dim, CEED_EVAL_GRAD);
    CeedQFunctionSetContext(apply_qfunc, &build_ctx, sizeof(build_ctx));
    break;
  case 1: {
    // This creates the QFunction via the gallery.
    char name[16] = "";
    snprintf(name, sizeof name, "Poisson%dDApply", dim);
    CeedQFunctionCreateInteriorByName(ceed, name, &apply_qfunc);
    break;
  }
  }

  // Create the diffusion operator.
  CeedOperator oper;
  CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
  CeedOperatorSetField(oper, "du", sol_restr, CEED_NOTRANSPOSE,
                       sol_basis, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(oper, "qdata", qdata_restr_i, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(oper, "dv", sol_restr, CEED_NOTRANSPOSE,
                       sol_basis, CEED_VECTOR_ACTIVE);

  // Compute the mesh surface area using the diff operator:
  //                                             sa = 1^T \cdot abs( K \cdot x).
  if (!test) {
    printf("Computing the mesh surface area using the formula: sa = 1^T.|K.x| ...");
    fflush(stdout);
  }

  // Create auxiliary solution-size vectors.
  CeedVector u, v;
  CeedVectorCreate(ceed, sol_size, &u);
  CeedVectorCreate(ceed, sol_size, &v);

  // Initialize 'u' with sum of coordinates, x+y+z.
  CeedScalar *u_host;
  const CeedScalar *x_host;
  CeedVectorGetArray(u, CEED_MEM_HOST, &u_host);
  CeedVectorGetArrayRead(mesh_coords, CEED_MEM_HOST, &x_host);
  for (CeedInt i = 0; i < sol_size; i++) {
    u_host[i] = 0;
    for (CeedInt d = 0; d < dim; d++)
      u_host[i] += x_host[i+d*sol_size];
  }
  CeedVectorRestoreArray(u, &u_host);
  CeedVectorRestoreArrayRead(mesh_coords, &x_host);

  // Apply the diffusion operator: 'u' -> 'v'.
  CeedOperatorApply(oper, u, v, CEED_REQUEST_IMMEDIATE);

  // Compute and print the sum of the entries of 'v' giving the mesh surface area.
  const CeedScalar *v_host;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_host);
  CeedScalar sa = 0.;
  for (CeedInt i = 0; i < sol_size; i++) {
    sa += fabs(v_host[i]);
  }
  CeedVectorRestoreArrayRead(v, &v_host);
  if (!test) {
    printf(" done.\n");
    printf("Exact mesh surface area    : % .14g\n", exact_sa);
    printf("Computed mesh surface area : % .14g\n", sa);
    printf("Surface area error         : % .14g\n", sa-exact_sa);
  } else {
    CeedScalar tol = (dim==1? 1E-12 : dim==2? 1E-1 : 1E-1);
    if (fabs(sa-exact_sa)>tol)
      printf("Surface area error         : % .14g\n", sa-exact_sa);
  }

  // Free dynamically allocated memory.
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&qdata);
  CeedVectorDestroy(&mesh_coords);
  CeedOperatorDestroy(&oper);
  CeedQFunctionDestroy(&apply_qfunc);
  CeedOperatorDestroy(&build_oper);
  CeedQFunctionDestroy(&build_qfunc);
  CeedElemRestrictionDestroy(&sol_restr);
  CeedElemRestrictionDestroy(&mesh_restr);
  CeedElemRestrictionDestroy(&sol_restr_i);
  CeedElemRestrictionDestroy(&mesh_restr_i);
  CeedBasisDestroy(&sol_basis);
  CeedBasisDestroy(&mesh_basis);
  CeedDestroy(&ceed);
  return 0;
}


int GetCartesianMeshSize(int dim, int order, int prob_size, int nxyz[3]) {
  // Use the approximate formula:
  //    prob_size ~ num_elem * order^dim
  CeedInt num_elem = prob_size / CeedIntPow(order, dim);
  CeedInt s = 0;  // find s: num_elem/2 < 2^s <= num_elem
  while (num_elem > 1) {
    num_elem /= 2;
    s++;
  }
  CeedInt r = s%dim;
  for (int d = 0; d < dim; d++) {
    int sd = s/dim;
    if (r > 0) { sd++; r--; }
    nxyz[d] = 1 << sd;
  }
  return 0;
}

int BuildCartesianRestriction(Ceed ceed, int dim, int nxyz[3], int order,
                              int ncomp, CeedInt *size, CeedInt num_qpts,
                              CeedElemRestriction *restr,
                              CeedElemRestriction *restr_i) {
  CeedInt p = order, pp1 = p+1;
  CeedInt nnodes = CeedIntPow(pp1, dim); // number of scal. nodes per element
  CeedInt elem_qpts = CeedIntPow(num_qpts, dim); // number of qpts per element
  CeedInt nd[3], num_elem = 1, scalar_size = 1;
  for (int d = 0; d < dim; d++) {
    num_elem *= nxyz[d];
    nd[d] = nxyz[d]*p + 1;
    scalar_size *= nd[d];
  }
  *size = scalar_size*ncomp;
  // elem:         0             1                 n-1
  //        |---*-...-*---|---*-...-*---|- ... -|--...--|
  // nnodes:   0   1    p-1  p  p+1       2*p             n*p
  CeedInt *el_nodes = malloc(sizeof(CeedInt)*num_elem*nnodes);
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt exyz[3], re = e;
    for (int d = 0; d < dim; d++) { exyz[d] = re%nxyz[d]; re /= nxyz[d]; }
    CeedInt *loc_el_nodes = el_nodes + e*nnodes;
    for (int lnodes = 0; lnodes < nnodes; lnodes++) {
      CeedInt gnodes = 0, gnodes_stride = 1, rnodes = lnodes;
      for (int d = 0; d < dim; d++) {
        gnodes += (exyz[d]*p + rnodes%pp1) * gnodes_stride;
        gnodes_stride *= nd[d];
        rnodes /= pp1;
      }
      loc_el_nodes[lnodes] = gnodes;
    }
  }
  if (restr)
    CeedElemRestrictionCreate(ceed, num_elem, nnodes, scalar_size,
                              ncomp, CEED_MEM_HOST,
                              CEED_COPY_VALUES, el_nodes, restr);
  if (restr_i)
    CeedElemRestrictionCreateIdentity(ceed, num_elem, elem_qpts,
                                      elem_qpts*num_elem,
                                      ncomp, restr_i);
  free(el_nodes);
  return 0;
}

int SetCartesianMeshCoords(int dim, int nxyz[3], int mesh_order,
                           CeedVector mesh_coords) {
  CeedInt p = mesh_order;
  CeedInt nd[3], num_elem = 1, scalar_size = 1;
  for (int d = 0; d < dim; d++) {
    num_elem *= nxyz[d];
    nd[d] = nxyz[d]*p + 1;
    scalar_size *= nd[d];
  }
  CeedScalar *coords;
  CeedVectorGetArray(mesh_coords, CEED_MEM_HOST, &coords);
  CeedScalar *nodes = malloc(sizeof(CeedScalar)*(p+1));
  // The H1 basis uses Lobatto quadrature points as nodes.
  CeedLobattoQuadrature(p+1, nodes, NULL); // nodes are in [-1,1]
  for (CeedInt i = 0; i <= p; i++) { nodes[i] = 0.5+0.5*nodes[i]; }
  for (CeedInt gsnodes = 0; gsnodes < scalar_size; gsnodes++) {
    CeedInt rnodes = gsnodes;
    for (int d = 0; d < dim; d++) {
      CeedInt d1d = rnodes%nd[d];
      coords[gsnodes+scalar_size*d] = ((d1d/p)+nodes[d1d%p]) / nxyz[d];
      rnodes /= nd[d];
    }
  }
  free(nodes);
  CeedVectorRestoreArray(mesh_coords, &coords);
  return 0;
}

#ifndef M_PI
#define M_PI    3.14159265358979323846
#define M_PI_2  1.57079632679489661923
#endif

CeedScalar TransformMeshCoords(int dim, int mesh_size, CeedVector mesh_coords) {
  CeedScalar exact_sa = (dim==1? 2 : dim==2? 4 : 6);
  CeedScalar *coords;

  CeedVectorGetArray(mesh_coords, CEED_MEM_HOST, &coords);
  for (CeedInt i = 0; i < mesh_size; i++) {
    // map [0,1] to [0,1] varying the mesh density
    coords[i] = 0.5+1./sqrt(3.)*sin((2./3.)*M_PI*(coords[i]-0.5));
  }
  CeedVectorRestoreArray(mesh_coords, &coords);

  return exact_sa;
}
