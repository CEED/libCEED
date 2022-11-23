#ifndef libceed_petsc_examples_area_problem_data_h
#define libceed_petsc_examples_area_problem_data_h

#include <ceed.h>
#include <petsc.h>

#include "../include/structs.h"
#include "../qfunctions/area/areacube.h"
#include "../qfunctions/area/areasphere.h"

// -----------------------------------------------------------------------------
// Problem Option Data
// -----------------------------------------------------------------------------

// Problem options
typedef enum { CUBE = 0, SPHERE = 1 } ProblemType;

static BPData problem_options[6] = {
    [CUBE] =
        {
                .num_comp_x    = 3,
                .num_comp_u    = 1,
                .topo_dim      = 2,
                .q_data_size   = 1,
                .q_extra       = 1,
                .setup_geo     = SetupMassGeoCube,
                .apply         = Mass,
                .setup_geo_loc = SetupMassGeoCube_loc,
                .apply_loc     = Mass_loc,
                .in_mode       = CEED_EVAL_INTERP,
                .out_mode      = CEED_EVAL_INTERP,
                .q_mode        = CEED_GAUSS,
                .enforce_bc    = PETSC_FALSE,
                },
    [SPHERE] = {
                .num_comp_x    = 3,
                .num_comp_u    = 1,
                .topo_dim      = 2,
                .q_data_size   = 1,
                .q_extra       = 1,
                .setup_geo     = SetupMassGeoSphere,
                .apply         = Mass,
                .setup_geo_loc = SetupMassGeoSphere_loc,
                .apply_loc     = Mass_loc,
                .in_mode       = CEED_EVAL_INTERP,
                .out_mode      = CEED_EVAL_INTERP,
                .q_mode        = CEED_GAUSS,
                .enforce_bc    = PETSC_FALSE,
                }
};

#endif  // libceed_petsc_examples_area_problem_data_h
