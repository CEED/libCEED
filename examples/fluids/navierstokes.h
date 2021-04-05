#ifndef navierstokes_h
#define navierstokes_h

#include <ceed.h>
#include <petscdm.h>

#include "qfunctions/common.h"
#include "qfunctions/setup-boundary.h"
#include "qfunctions/advection.h"
#include "qfunctions/advection2d.h"
#include "qfunctions/euler-vortex.h"
#include "qfunctions/densitycurrent.h"

// Wind Options for Advection
typedef enum {
  ADVECTION_WIND_ROTATION = 0,
  ADVECTION_WIND_TRANSLATION = 1,
} WindType;
static const char *const WindTypes[] = {
  "rotation",
  "translation",
  "WindType", "ADVECTION_WIND_", NULL
};

// Euler test cases
typedef enum {
  EULER_TEST_NONE = 0,
  EULER_TEST_1 = 1,
  EULER_TEST_2 = 2,
  EULER_TEST_3 = 3,
  EULER_TEST_4 = 4,
} EulerTestType;
static const char *const EulerTestTypes[] = {
  "none",
  "t1",
  "t2",
  "t3",
  "t4",
  "EulerTestType", "EULER_TEST_", NULL
};

typedef enum {
  STAB_NONE = 0,
  STAB_SU = 1,   // Streamline Upwind
  STAB_SUPG = 2, // Streamline Upwind Petrov-Galerkin
} StabilizationType;
static const char *const StabilizationTypes[] = {
  "none",
  "SU",
  "SUPG",
  "StabilizationType", "STAB_", NULL
};

typedef struct User_ *User;
typedef struct Units_ *Units;
typedef struct Physics_ *Physics;
typedef struct SimpleBC_ *SimpleBC;
struct SimpleBC_ {
  PetscInt nwall, nslip[3];
  PetscInt walls[6], slips[3][6];
  PetscBool userbc;
};

// Problem specific data
typedef struct {
  CeedInt dim, qdatasizeVol, qdatasizeSur;
  CeedQFunctionUser setupVol, setupSur, ics, applyVol_rhs, applyVol_ifunction,
                    applySur;
  const char *setupVol_loc, *setupSur_loc, *ics_loc, *applyVol_rhs_loc,
        *applyVol_ifunction_loc, *applySur_loc;
  bool non_zero_time;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                       PetscScalar[], void *);
  PetscErrorCode (*bc_fnc)(DM, SimpleBC, Physics, void *);
} problemData;

// PETSc user data

struct User_ {
  MPI_Comm comm;
  PetscInt outputfreq;
  DM dm;
  DM dmviz;
  Mat interpviz;
  Ceed ceed;
  Units units;
  CeedVector qceed, qdotceed, gceed;
  CeedOperator op_rhs_vol, op_rhs, op_ifunction_vol, op_ifunction;
  Vec M;
  char outputdir[PETSC_MAX_PATH_LEN];
  PetscInt contsteps;
  Physics phys;
};

struct Units_ {
  // fundamental units
  PetscScalar meter;
  PetscScalar kilogram;
  PetscScalar second;
  PetscScalar Kelvin;
  // derived units
  PetscScalar Pascal;
  PetscScalar JperkgK;
  PetscScalar mpersquareds;
  PetscScalar WpermK;
  PetscScalar kgpercubicm;
  PetscScalar kgpersquaredms;
  PetscScalar Joulepercubicm;
  PetscScalar Joule;
};

// Setup Context for QFunctions
struct Physics_ {
  NSContext ctxNSData;
  EulerContext ctxEulerData;
  AdvectionContext ctxAdvectionData;
  WindType wind_type;
  EulerTestType eulertest;
  StabilizationType stab;
  PetscBool implicit;
  PetscBool hasCurrentTime;
  PetscBool hasNeumann;
};

// Setup function for each problem
extern PetscErrorCode NS_DENSITY_CURRENT(problemData *problem,
    void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_EULER_VORTEX(problemData *problem,
                                      void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_ADVECTION(problemData *problem,
                                   void *ctxSetupData, void *ctx, void *ctxPhys);
extern PetscErrorCode NS_ADVECTION2D(problemData *problem,
                                     void *ctxSetupData, void *ctx, void *ctxPhys);

// Boundary Condition Functions
extern PetscErrorCode BC_DENSITY_CURRENT(DM dm, SimpleBC bc, Physics phys,
    void *ctxSetupData);
extern PetscErrorCode BC_EULER_VORTEX(DM dm, SimpleBC bc, Physics phys,
                                      void *ctxSetupData);
extern PetscErrorCode BC_ADVECTION(DM dm, SimpleBC bc, Physics phys,
                                   void *ctxSetupData);
extern PetscErrorCode BC_ADVECTION2D(DM dm, SimpleBC bc, Physics phys,
                                     void *ctxSetupData);

#endif
