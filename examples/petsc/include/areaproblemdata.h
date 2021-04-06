#ifndef areaproblemdata_h
#define areaproblemdata_h

#include <ceed.h>
#include <petsc.h>
#include "../include/structs.h"
#include "../qfunctions/area/areacube.h"
#include "../qfunctions/area/areasphere.h"

// -----------------------------------------------------------------------------
// Problem Option Data
// -----------------------------------------------------------------------------

// Problem options
typedef enum {
  CUBE = 0, SPHERE = 1
} problemType;

static bpData problemOptions[6] = {
  [CUBE] = {
    .ncompx = 3,
    .ncompu = 1,
    .topodim = 2,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeoCube,
    .apply = Mass,
    .setupgeofname = SetupMassGeoCube_loc,
    .applyfname = Mass_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS,
    .enforcebc = PETSC_FALSE,
  },
  [SPHERE] = {
    .ncompx = 3,
    .ncompu = 1,
    .topodim = 2,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeoSphere,
    .apply = Mass,
    .setupgeofname = SetupMassGeoSphere_loc,
    .applyfname = Mass_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS,
    .enforcebc = PETSC_FALSE,
  }
};

#endif // areaproblemdata_h
