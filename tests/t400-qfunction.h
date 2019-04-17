/// @file
/// QFunction definitions for t400-qfunction.c

static int setup(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *w = args.in[0];
  CeedScalar *qdata = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    qdata[i] = w[i];
  }
  return 0;
}

static int mass(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *qdata = args.in[0], *u = args.in[1];
  CeedScalar *v = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    v[i] = qdata[i] * u[i];
  }
  return 0;
}
