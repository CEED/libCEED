#ifndef newtonian_types_h
#define newtonian_types_h

#include <ceed/ceed.h>
#include "stabilization_types.h"

typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  CeedScalar theta0;
  CeedScalar thetaC;
  CeedScalar P0;
  CeedScalar N;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g[3];
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar center[3];
  CeedScalar dc_axis[3];
  CeedScalar time;
  int wind_type;              // See WindType: 0=ROTATION, 1=TRANSLATION
  int bubble_type;            // See BubbleType: 0=SPHERE, 1=CYLINDER
  int bubble_continuity_type; // See BubbleContinuityType: 0=SMOOTH, 1=BACK_SHARP 2=THICK
};

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
