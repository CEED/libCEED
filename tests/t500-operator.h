/// @file
/// QFunction definitions for t500-operator.c

//! [QFunction User Code]
static int setup(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *weight = args.in[0], *dxdX = args.in[1];
  CeedScalar *rho = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * dxdX[i];
  }
  return 0;
}

static int mass(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *rho = args.in[0], *u = args.in[1];
  CeedScalar *v = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}
//! [QFunction User Code]
