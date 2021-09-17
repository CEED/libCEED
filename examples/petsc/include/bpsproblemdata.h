#ifndef bpsproblemdata_h
#define bpsproblemdata_h

#include <ceed.h>
#include <petsc.h>
#include "../include/bcfunctions.h"
#include "../include/structs.h"
#include "../qfunctions/bps/bp1.h"
#include "../qfunctions/bps/bp2.h"
#include "../qfunctions/bps/bp3.h"
#include "../qfunctions/bps/bp4.h"
#include "../qfunctions/bps/common.h"

// -----------------------------------------------------------------------------
// BP Option Data
// -----------------------------------------------------------------------------

// BP options
typedef enum {
  CEED_BP1 = 0, CEED_BP2 = 1, CEED_BP3 = 2,
  CEED_BP4 = 3, CEED_BP5 = 4, CEED_BP6 = 5
} BPType;

BPData bp_options[6] = {
  [CEED_BP1] = {
    .num_comp_u = 1,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 1,
    .q_extra = 1,
    .setup_geo = SetupMassGeo,
    .setup_rhs = SetupMassRhs,
    .apply = Mass,
    .error = Error,
    .setup_geo_loc = SetupMassGeo_loc,
    .setup_rhs_loc = SetupMassRhs_loc,
    .apply_loc = Mass_loc,
    .error_loc = Error_loc,
    .in_mode = CEED_EVAL_INTERP,
    .out_mode = CEED_EVAL_INTERP,
    .q_mode = CEED_GAUSS,
    .enforce_bc = PETSC_FALSE,
    .bc_func = BCsMass
  },
  [CEED_BP2] = {
    .num_comp_u = 3,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 1,
    .q_extra = 1,
    .setup_geo = SetupMassGeo,
    .setup_rhs = SetupMassRhs3,
    .apply = Mass3,
    .error = Error3,
    .setup_geo_loc = SetupMassGeo_loc,
    .setup_rhs_loc = SetupMassRhs3_loc,
    .apply_loc = Mass3_loc,
    .error_loc = Error3_loc,
    .in_mode = CEED_EVAL_INTERP,
    .out_mode = CEED_EVAL_INTERP,
    .q_mode = CEED_GAUSS,
    .enforce_bc = PETSC_FALSE,
    .bc_func = BCsMass
  },
  [CEED_BP3] = {
    .num_comp_u = 1,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 7,
    .q_extra = 1,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs_loc,
    .apply_loc = Diff_loc,
    .error_loc = Error_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS,
    .enforce_bc = PETSC_TRUE,
    .bc_func = BCsDiff
  },
  [CEED_BP4] = {
    .num_comp_u = 3,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 7,
    .q_extra = 1,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs3_loc,
    .apply_loc = Diff3_loc,
    .error_loc = Error3_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS,
    .enforce_bc = PETSC_TRUE,
    .bc_func = BCsDiff
  },
  [CEED_BP5] = {
    .num_comp_u = 1,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 7,
    .q_extra = 0,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs,
    .apply = Diff,
    .error = Error,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs_loc,
    .apply_loc = Diff_loc,
    .error_loc = Error_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS_LOBATTO,
    .enforce_bc = PETSC_TRUE,
    .bc_func = BCsDiff
  },
  [CEED_BP6] = {
    .num_comp_u = 3,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 7,
    .q_extra = 0,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs3_loc,
    .apply_loc = Diff3_loc,
    .error_loc = Error3_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS_LOBATTO,
    .enforce_bc = PETSC_TRUE,
    .bc_func = BCsDiff
  }
};

#endif // bpsproblemdata_h
