#include <ceed-impl.h>
#include "magma.h" 
#include "magma_check_cudaerror.h"         

// t20 QFunctions ===============================================================

__global__
void t20_setup_kernel(const CeedScalar *in, CeedScalar *out) 
{
   out[threadIdx.x] = in[threadIdx.x];
}

extern "C" int t20_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
 
  const CeedScalar *w = in[0];
  CeedScalar *qdata = out[0];
  if (magma_is_devptr(w)){
     t20_setup_kernel<<<1,Q,0>>>(w, qdata);
     magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
        qdata[i] = w[i];
    }

  return 0;
}

// ====================================

__global__ void 
t20_mass_kernel(const CeedScalar *qdata, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = qdata[threadIdx.x] * u[threadIdx.x];
}

extern "C" int t20_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  
  const CeedScalar *qdata = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(qdata)){
    t20_mass_kernel<<<1,Q,0>>>(qdata, u, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = qdata[i] * u[i];
    }

  return 0;
}

// t30 QFunctions ===============================================================

__global__ void
t30_setup_kernel(const CeedScalar *weight, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = weight[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t30_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar *rho = out[0];

  if (magma_is_devptr(weight)){
    t30_setup_kernel<<<1,Q,0>>>(weight, dxdX, rho);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
      rho[i] = weight[i]*dxdX[i];
    }

  return 0;
}

// ====================================

__global__ void
t30_mass_kernel(const CeedScalar *rho, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = rho[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t30_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(rho)){
    t30_mass_kernel<<<1,Q,0>>>(rho, u, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = rho[i] * u[i];
    }

  return 0;
}
