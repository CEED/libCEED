#ifndef libceed_petsc_examples_sphere_problem_data_h
#define libceed_petsc_examples_sphere_problem_data_h

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

static BPData bp_options[6] = {
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
    .q_mode = CEED_GAUSS
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
    .q_mode = CEED_GAUSS
  },
  [CEED_BP3] = {
    .num_comp_u = 1,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 4,
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
    .q_mode = CEED_GAUSS
  },
  [CEED_BP4] = {
    .num_comp_u = 3,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 4,
    .q_extra = 1,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs3_loc,
    .apply_loc = Diff_loc,
    .error_loc = Error3_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS
  },
  [CEED_BP5] = {
    .num_comp_u = 1,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 4,
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
    .q_mode = CEED_GAUSS_LOBATTO
  },
  [CEED_BP6] = {
    .num_comp_u = 3,
    .num_comp_x = 3,
    .topo_dim = 3,
    .q_data_size = 4,
    .q_extra = 0,
    .setup_geo = SetupDiffGeo,
    .setup_rhs = SetupDiffRhs3,
    .apply = Diff3,
    .error = Error3,
    .setup_geo_loc = SetupDiffGeo_loc,
    .setup_rhs_loc = SetupDiffRhs3_loc,
    .apply_loc = Diff_loc,
    .error_loc = Error3_loc,
    .in_mode = CEED_EVAL_GRAD,
    .out_mode = CEED_EVAL_GRAD,
    .q_mode = CEED_GAUSS_LOBATTO
  }
};

#endif // libceed_petsc_examples_sphere_problem_data_h
