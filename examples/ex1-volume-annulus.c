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

#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };


/// libCEED Q-function for building quadrature data for a mass operator
static int f_build_mass(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  // u[1] is Jacobians, size (Q x nc x dim) with column-major layout
  // u[4] is quadrature weights, size (Q)
  struct BuildContext *bc = (struct BuildContext*)ctx;
  CeedScalar *qd = (CeedScalar*)qdata;
  const CeedScalar *J = u[1], *qw = u[4];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      qd[i] = J[i] * qw[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // 0 2
      // 1 3
      qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
               J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
               J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * qw[i];
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}


/// libCEED Q-function for applying a mass operator
static int f_apply_mass(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  const CeedScalar *w = (const CeedScalar*)qdata;
  for (CeedInt i=0; i<Q; i++) v[0][i] = w[i] * u[0][i];
  return 0;
}


// Auxiliary functions.
int GetCartesianMeshSize(int dim, int order, int prob_size, int nxyz[3]);
int BuildCartesianRestriction(Ceed ceed, int dim, int nxyz[3], int order,
                              int ncomp, CeedInt *size,
                              CeedElemRestriction *restr);
int SetCartesianMeshCoords(int dim, int nxyz[3], int mesh_order,
                           CeedVector mesh_coords);
CeedScalar TransformMeshCoords(int dim, int mesh_size, CeedVector mesh_coords);

//TESTARGS -c {ceed_resource} -t
int main(int argc, const char *argv[]) {
  const char *ceed_spec = "/cpu/self";
  int dim        = 3;           // dimension of the mesh
  int mesh_order = 4;           // polynomial degree for the mesh
  int sol_order  = 4;           // polynomial degree for the solution
  int num_qpts   = sol_order+2; // number of 1D quadrature points
  int prob_size  = -1;          // approximate problem size
  int help = 0, test = 0;

  // Process command line arguments.
  for (int ia = 1; ia < argc; ia++) {
    int next_arg = ((ia+1) < argc), parse_error = 0;
    if (!strcmp(argv[ia],"-h")) {
      help = 1;
    } else if (!strcmp(argv[ia],"-c")) {
      parse_error = next_arg ? ceed_spec = argv[++ia], 0 : 1;
    } else if (!strcmp(argv[ia],"-d")) {
      parse_error = next_arg ? dim = atoi(argv[++ia]), 0 : 1;
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
    }
    if (parse_error) {
      printf("Error parsing command line options.\n");
      return 1;
    }
  }
  if (prob_size < 0) prob_size = test ? 8*16 : 256*1024;

  // Print the values of all options:
  if (!test || help) {
    printf("Selected options: [command line option] : <current value>\n");
    printf("  Ceed specification [-c] : %s\n", ceed_spec);
    printf("  Mesh dimension     [-d] : %d\n", dim);
    printf("  Mesh order         [-m] : %d\n", mesh_order);
    printf("  Solution order     [-o] : %d\n", sol_order);
    printf("  Num. 1D quadr. pts [-q] : %d\n", num_qpts);
    printf("  Approx. # unknowns [-s] : %d\n", prob_size);
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
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, mesh_order+1, num_qpts,
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
  CeedElemRestriction mesh_restr, sol_restr;
  BuildCartesianRestriction(ceed, dim, nxyz, mesh_order, dim, &mesh_size,
                            &mesh_restr);
  BuildCartesianRestriction(ceed, dim, nxyz, sol_order, 1, &sol_size,
                            &sol_restr);
  if (!test) {
    printf("Number of mesh nodes    : %d\n", mesh_size/dim);
    printf("Number of solution dofs : %d\n", sol_size);
  }

  // Create a CeedVector with the mesh coordinates.
  CeedVector mesh_coords;
  CeedVectorCreate(ceed, mesh_size, &mesh_coords);
  SetCartesianMeshCoords(dim, nxyz, mesh_order, mesh_coords);

  // Apply a transformation to the mesh.
  CeedScalar exact_vol = TransformMeshCoords(dim, mesh_size, mesh_coords);

  // Context data to be passed to the 'f_build_mass' Q-function.
  struct BuildContext build_ctx;
  build_ctx.dim = build_ctx.space_dim = dim;

  // Create the Q-function that builds the mass operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunction build_qfunc;
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              (CeedEvalMode)(CEED_EVAL_GRAD|CEED_EVAL_WEIGHT),
                              CEED_EVAL_NONE, f_build_mass,
                              __FILE__":f_build_mass", &build_qfunc);
  CeedQFunctionSetContext(build_qfunc, &build_ctx, sizeof(build_ctx));

  // Create the operator that builds the quadrature data for the mass operator.
  CeedOperator build_oper;
  CeedOperatorCreate(ceed, mesh_restr, mesh_basis, build_qfunc, NULL, NULL,
                     &build_oper);

  // Compute the quadrature data for the mass operator.
  CeedVector qdata;
  CeedOperatorGetQData(build_oper, &qdata);
  if (!test) {
    printf("Computing the quadrature data for the mass operator ...");
    fflush(stdout);
  }
  CeedOperatorApply(build_oper, qdata, mesh_coords, NULL,
                    CEED_REQUEST_IMMEDIATE);
  if (!test) {
    printf(" done.\n");
  }

  // Create the Q-function that defines the action of the mass operator.
  CeedQFunction apply_qfunc;
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP, f_apply_mass,
                              __FILE__":f_apply_mass", &apply_qfunc);

  // Create the mass operator.
  CeedOperator oper;
  CeedOperatorCreate(ceed, sol_restr, sol_basis, apply_qfunc, NULL, NULL,
                     &oper);

  // Compute the mesh volume using the mass operator: vol = 1^T.M.1.
  if (!test) {
    printf("Computing the mesh volume using the formula: vol = 1^T.M.1 ...");
    fflush(stdout);
  }

  // Create auxiliary solution-size vectors.
  CeedVector u, v;
  CeedVectorCreate(ceed, sol_size, &u);
  CeedVectorCreate(ceed, sol_size, &v);

  // Initialize 'u' with ones.
  CeedScalar *u_host;
  CeedVectorGetArray(u, CEED_MEM_HOST, &u_host);
  for (CeedInt i = 0; i < sol_size; i++) {
    u_host[i] = 1.;
  }
  CeedVectorRestoreArray(u, &u_host);

  // Apply the mass operator: 'u' -> 'v'.
  CeedOperatorApply(oper, qdata, u, v, CEED_REQUEST_IMMEDIATE);

  // Compute and print the sum of the entries of 'v' giving the mesh volume.
  const CeedScalar *v_host;
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_host);
  CeedScalar vol = 0.;
  for (CeedInt i = 0; i < sol_size; i++) {
    vol += v_host[i];
  }
  CeedVectorRestoreArrayRead(v, &v_host);
  if (!test) {
    printf(" done.\n");
    printf("Exact mesh volume    : % .14g\n", exact_vol);
    printf("Computed mesh volume : % .14g\n", vol);
    printf("Volume error         : % .14g\n", vol-exact_vol);
  } else {
    printf("Volume error : % .1e\n", vol-exact_vol);
  }

  // Free dynamically allocated memory.
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&u);
  CeedOperatorDestroy(&oper);
  CeedQFunctionDestroy(&apply_qfunc);
  // qdata is owned by build_oper
  CeedOperatorDestroy(&build_oper);
  CeedQFunctionDestroy(&build_qfunc);
  CeedVectorDestroy(&mesh_coords);
  CeedElemRestrictionDestroy(&sol_restr);
  CeedElemRestrictionDestroy(&mesh_restr);
  CeedBasisDestroy(&sol_basis);
  CeedBasisDestroy(&mesh_basis);
  CeedDestroy(&ceed);
  return 0;
}


int GetCartesianMeshSize(int dim, int order, int prob_size, int nxyz[3]) {
  // Use the approximate formula:
  //    prob_size ~ num_elem * order^dim
  CeedInt num_elem = prob_size / CeedPowInt(order, dim);
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
                              int ncomp, CeedInt *size,
                              CeedElemRestriction *restr) {
  CeedInt p = order, pp1 = p+1;
  CeedInt ndof = CeedPowInt(pp1, dim); // number of scal. dofs per element
  CeedInt nd[3], num_elem = 1, scalar_size = 1;
  for (int d = 0; d < dim; d++) {
    num_elem *= nxyz[d];
    nd[d] = nxyz[d]*p + 1;
    scalar_size *= nd[d];
  }
  *size = scalar_size*ncomp;
  // elem:         0             1                 n-1
  //        |---*-...-*---|---*-...-*---|- ... -|--...--|
  // dof:   0   1    p-1  p  p+1       2*p             n*p
  CeedInt *el_dof = malloc(sizeof(CeedInt)*num_elem*ndof);
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt exyz[3], re = e;
    for (int d = 0; d < dim; d++) { exyz[d] = re%nxyz[d]; re /= nxyz[d]; }
    CeedInt *loc_el_dof = el_dof + e*ndof;
    for (int ldof = 0; ldof < ndof; ldof++) {
      CeedInt gdof = 0, gdof_stride = 1, rdof = ldof;
      for (int d = 0; d < dim; d++) {
        gdof += (exyz[d]*p + rdof%pp1) * gdof_stride;
        gdof_stride *= nd[d];
        rdof /= pp1;
      }
      loc_el_dof[ldof] = gdof;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, ndof, scalar_size, CEED_MEM_HOST,
                            CEED_COPY_VALUES, el_dof, restr);
  free(el_dof);
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
  for (CeedInt gsdof = 0; gsdof < scalar_size; gsdof++) {
    CeedInt rdof = gsdof;
    for (int d = 0; d < dim; d++) {
      CeedInt d1d = rdof%nd[d];
      coords[gsdof+scalar_size*d] = ((d1d/p)+nodes[d1d%p]) / nxyz[d];
      rdof /= nd[d];
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
  CeedScalar exact_volume;
  CeedScalar *coords;
  CeedVectorGetArray(mesh_coords, CEED_MEM_HOST, &coords);
  if (dim == 1) {
    for (CeedInt i = 0; i < mesh_size; i++) {
      // map [0,1] to [0,1] varying the mesh density
      coords[i] = 0.5+1./sqrt(3.)*sin((2./3.)*M_PI*(coords[i]-0.5));
    }
    exact_volume = 1.;
  } else {
    CeedInt num_nodes = mesh_size/dim;
    for (CeedInt i = 0; i < num_nodes; i++) {
      // map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
      // coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
      CeedScalar u = coords[i], v = coords[i+num_nodes];
      u = 1.+u;
      v = M_PI_2*v;
      coords[i] = u*cos(v);
      coords[i+num_nodes] = u*sin(v);
    }
    exact_volume = 3./4.*M_PI;
  }
  CeedVectorRestoreArray(mesh_coords, &coords);
  return exact_volume;
}
