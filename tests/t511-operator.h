/// @file
/// QFunction declarations for t511-operator.c

static int setup(void *ctx, CeedInt Q, CeedQFunctionArguments args) {
  const CeedScalar *weight = args.in[0], *J = args.in[1];
  CeedScalar *rho = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }
  return 0;
}

static int mass(void *ctx, CeedInt Q, CeedQFunctionArguments args) {
  const CeedScalar *rho = args.in[0], *u = args.in[1];
  CeedScalar *v = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}
