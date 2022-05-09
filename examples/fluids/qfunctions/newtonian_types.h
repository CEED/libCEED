#ifndef newtonian_types_h
#define newtonian_types_h

#include <ceed/ceed.h>
#include "stabilization_types.h"

typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
struct NewtonianIdealGasContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g[3];
  CeedScalar c_tau;
  CeedScalar Ctau_t;
  CeedScalar Ctau_v;
  CeedScalar Ctau_C;
  CeedScalar Ctau_M;
  CeedScalar Ctau_E;
  CeedScalar dt;
  StabilizationType stabilization;
};

#endif // newtonian_types_h
