extern "C" __global__ void setup(void *ctx, int Q, const double *const *in,
                      double *const *out) {
  // const double *w = in[0];
  // double *qdata = out[0];
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x;
  //   i < Q;
  //   i += blockDim.x * gridDim.x)
  // {
  //   qdata[i] = w[i];
  // }
}

extern "C" __global__ void mass(void *ctx, int Q, const double *const *in,
                     double *const *out) {
  // const double *qdata = in[0], *u = in[1];
  // double *v = out[0];
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x;
  //   i < Q;
  //   i += blockDim.x * gridDim.x)
  // {
  //   v[i] = qdata[i] * u[i];
  // }
}

