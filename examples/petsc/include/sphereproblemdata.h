#ifndef sphereproblemdata_h
#define sphereproblemdata_h

#include <ceed.h>
#include <petsc.h>
#include "../include/structs.h"
#include "../qfunctions/bps/bp1sphere.h"
#include "../qfunctions/bps/bp2sphere.h"
#include "../qfunctions/bps/bp3sphere.h"
#include "../qfunctions/bps/bp4sphere.h"
#include "../qfunctions/bps/common.h"

// -----------------------------------------------------------------------------
// BP Option Data
// -----------------------------------------------------------------------------

// BP options
typedef enum {
  CEED_BP1 = 0, CEED_BP2 = 1, CEED_BP3 = 2,
  CEED_BP4 = 3, CEED_BP5 = 4, CEED_BP6 = 5
} bpType;

static bpData bpOptions[6] = {
  [CEED_BP1] = {
    .ncompu = 1,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeo,
    .setuprhs = SetupMassRhs,
    .apply = Mass,
    .error = Error,
    .setupgeofname = SetupMassGeo_loc,
    .setuprhsfname = SetupMassRhs_loc,
    .applyfname = Mass_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS
  },
  [CEED_BP2] = {
    .ncompu = 3,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 1,
    .qextra = 1,
    .setupgeo = SetupMassGeo,
    .setuprhs = SetupMassRhs3,
    .apply = Mass3,
    .error = Error3,
    .setupgeofname = SetupMassGeo_loc,
    .setuprhsfname = SetupMassRhs3_loc,
    .applyfname = Mass3_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_INTERP,
    .outmode = CEED_EVAL_INTERP,
    .qmode = CEED_GAUSS
  },
  [CEED_BP3] = {
    .ncompu = 1,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 4,
    .qextra = 1,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs_loc,
    .applyfname = Diff_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS
  },
  [CEED_BP4] = {
    .ncompu = 3,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 4,
    .qextra = 1,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs3_loc,
    .applyfname = Diff_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS
  },
  [CEED_BP5] = {
    .ncompu = 1,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 4,
    .qextra = 0,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs_loc,
    .applyfname = Diff_loc,
    .errorfname = Error_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS_LOBATTO
  },
  [CEED_BP6] = {
    .ncompu = 3,
    .ncompx = 3,
    .topodim = 3,
    .qdatasize = 4,
    .qextra = 0,
    .setupgeo = SetupDiffGeo,
    .setuprhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setupgeofname = SetupDiffGeo_loc,
    .setuprhsfname = SetupDiffRhs3_loc,
    .applyfname = Diff_loc,
    .errorfname = Error3_loc,
    .inmode = CEED_EVAL_GRAD,
    .outmode = CEED_EVAL_GRAD,
    .qmode = CEED_GAUSS_LOBATTO
  }
};

#endif // sphereproblemdata_h
