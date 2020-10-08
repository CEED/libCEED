#ifndef navierstokes_h
#define navierstokes_h

#include <ceed.h>
#include <petscdm.h>

#include "common.h"
#include "setup-boundary.h"
#include "advection.h"
#include "advection2d.h"
#include "densitycurrent.h"

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

// Test Options
typedef enum {
  TEST_NONE = 0,               // Non test mode
  TEST_EXPLICIT = 1,           // Explicit test
  TEST_IMPLICIT_STAB_NONE = 2, // Implicit test no stab
  TEST_IMPLICIT_STAB_SUPG = 3, // Implicit test supg stab
} testType;
static const char *const testTypes[] = {
  "none",
  "explicit",
  "implicit_stab_none",
  "implicit_stab_supg",
  "testType", "TEST_", NULL
};

// Tests specific data
typedef struct {
  PetscScalar testtol;
  const char *filepath;
} testData;

// Problem specific data
typedef struct {
  CeedInt dim, qdatasizeVol, qdatasizeSur;
  CeedQFunctionUser setupVol, setupSur, ics, applyVol_rhs, applyVol_ifunction,
                    applySur;
  PetscErrorCode (*bc)(PetscInt, PetscReal, const PetscReal[], PetscInt,
                       PetscScalar[], void *);
  const char *setupVol_loc, *setupSur_loc, *ics_loc, *applyVol_rhs_loc,
        *applyVol_ifunction_loc, *applySur_loc;
  bool non_zero_time;
} problemData;

// PETSc user data
typedef struct User_ *User;
typedef struct Units_ *Units;

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
  char outputfolder[PETSC_MAX_PATH_LEN];
  PetscInt contsteps;
  DCContext ctxDCData;
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

typedef struct SimpleBC_ *SimpleBC;
struct SimpleBC_ {
  PetscInt nwall, nslip[3];
  PetscInt walls[6], slips[3][6];
  PetscBool userbc;
};

extern PetscErrorCode NS_DENSITY_CURRENT(problemData *problem);

#endif
