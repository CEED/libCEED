/// @file
/// QFunction definitions for mass operator example using MFEM

/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a mass operator
static int f_build_mass(void *ctx, CeedInt Q, CeedQFunctionArguments args) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  BuildContext *bc = (BuildContext *)ctx;
  const CeedScalar *J = args.in[0], *qw = args.in[1];
  CeedScalar *rho = args.out[0];

  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      rho[i] = J[i] * qw[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // 0 2
      // 1 3
      rho[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      rho[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
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
static int f_apply_mass(void *ctx, CeedInt Q, CeedQFunctionArguments args) {
  const CeedScalar *u = args.in[0], *w = args.in[1];
  CeedScalar *v = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    v[i] = w[i] * u[i];
  }
  return 0;
}
