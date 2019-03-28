#include <ceed-impl.h>
#include "ceed-magma.h" 
#include "magma.h" 
#include "magma_check_cudaerror.h"         

// t20 QFunctions ===============================================================

static __global__
void t20_setup_kernel(const CeedScalar *in, CeedScalar *out) 
{
   out[threadIdx.x] = in[threadIdx.x];
}

extern "C" int t20_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
 
  const CeedScalar *w = in[0];
  CeedScalar *qdata = out[0];
  if (magma_is_devptr(w)==1){
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

static __global__ void 
t20_mass_kernel(const CeedScalar *qdata, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = qdata[threadIdx.x] * u[threadIdx.x];
}

extern "C" int t20_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  
  const CeedScalar *qdata = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(qdata)==1){
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

static __global__ void
t30_setup_kernel(const CeedScalar *weight, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = weight[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t30_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar *rho = out[0];

  if (magma_is_devptr(weight)==1){
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

static __global__ void
t30_mass_kernel(const CeedScalar *rho, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = rho[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t30_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(rho)==1){
    t30_mass_kernel<<<1,Q,0>>>(rho, u, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = rho[i] * u[i];
    }

  return 0;
}

// ex1 QFunctions ===============================================================

/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };

static __global__ void
ex1_setup_kernel1(const CeedScalar *weight, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = weight[threadIdx.x]*u[threadIdx.x];
}

static __global__ void
ex1_setup_kernel2(const CeedScalar *J, const CeedScalar *qw, CeedScalar *qd)
{
   int i = threadIdx.x, Q = blockDim.x;
   qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
}

static __global__ void
ex1_setup_kernel3(const CeedScalar *J, const CeedScalar *qw, CeedScalar *qd)
{
   int i = threadIdx.x, Q = blockDim.x;
   qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                  J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                  J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * qw[i];
}

extern "C" int ex1_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
   // in[0] is Jacobians, size (Q x nc x dim) with column-major layout
   // in[1] is quadrature weights, size (Q)
   struct BuildContext *bc = (struct BuildContext*)ctx;
   const CeedScalar *J = in[0], *qw = in[1];
   CeedScalar *qd = out[0];
   switch (bc->dim + 10*bc->space_dim) {
   case 11:
    if (magma_is_devptr(qd)==1){
       ex1_setup_kernel1<<<1,Q,0>>>(J, qw, qd);
       magma_check_cudaerror();
    } else
       for (CeedInt i=0; i<Q; i++) {
          qd[i] = J[i] * qw[i];
       }
    break;
   case 22:
    if (magma_is_devptr(qd)==1){
       ex1_setup_kernel2<<<1,Q,0>>>(J, qw, qd);
       magma_check_cudaerror();
    } else
       for (CeedInt i=0; i<Q; i++) {
         // 0 2
         // 1 3
         qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
       }
    break;
  case 33:
   if (magma_is_devptr(qd)==1){
       ex1_setup_kernel3<<<1,Q,0>>>(J, qw, qd);
       magma_check_cudaerror();
    } else 
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

// ====================================

static __global__ void
ex1_mass_kernel(const CeedScalar *rho, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = rho[threadIdx.x]*u[threadIdx.x];
}

extern "C" int ex1_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  const CeedScalar *u = in[0], *w = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(v)==1){
    ex1_mass_kernel<<<1,Q,0>>>(u, w, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = w[i] * u[i];
    }

  return 0;
}

// t400 QFunctions ==============================================================

static __global__
void t400_setup_kernel(const CeedScalar *in, CeedScalar *out)
{
   out[threadIdx.x] = in[threadIdx.x];
}

extern "C" int t400_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {

  const CeedScalar *w = in[0];
  CeedScalar *qdata = out[0];
  if (magma_is_devptr(w)==1){
     t400_setup_kernel<<<1,Q,0>>>(w, qdata);
     magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
        qdata[i] = w[i];
    }

  return 0;
}

// ====================================

static __global__ void
t400_mass_kernel(const CeedScalar *qdata, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = qdata[threadIdx.x] * u[threadIdx.x];
}

extern "C" int t400_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {

  const CeedScalar *qdata = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(qdata)==1){
    t400_mass_kernel<<<1,Q,0>>>(qdata, u, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = qdata[i] * u[i];
    }

  return 0;
}

// t500 QFunctions ==============================================================

static __global__ void
t500_setup_kernel(const CeedScalar *weight, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = weight[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t500_setup(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar *rho = out[0];

  if (magma_is_devptr(weight)==1){
    t500_setup_kernel<<<1,Q,0>>>(weight, dxdX, rho);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
      rho[i] = weight[i]*dxdX[i];
    }

  return 0;
}

// ====================================

static __global__ void
t500_mass_kernel(const CeedScalar *rho, const CeedScalar *u, CeedScalar *v)
{
   v[threadIdx.x] = rho[threadIdx.x]*u[threadIdx.x];
}

extern "C" int t500_mass(void *ctx, CeedInt Q, const CeedScalar *const *in,
                        CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];
  if (magma_is_devptr(rho)==1){
    t500_mass_kernel<<<1,Q,0>>>(rho, u, v);
    magma_check_cudaerror();
  }
  else
    for (CeedInt i=0; i<Q; i++) {
       v[i] = rho[i] * u[i];
    }

  return 0;
}
