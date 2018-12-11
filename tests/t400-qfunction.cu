// *****************************************************************************
extern "C" __global__ void setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  const CeedScalar *w = in[0];
  CeedScalar *qdata = out[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < Q;
    i += blockDim.x * gridDim.x)
  {
    qdata[i] = w[i];
  }
}

// *****************************************************************************
extern "C" __global__ void mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  const CeedScalar *qdata = in[0], *u = in[1];
  CeedScalar *v = out[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < Q;
    i += blockDim.x * gridDim.x)
  {
    v[i] = qdata[i] * u[i];
  }
}

