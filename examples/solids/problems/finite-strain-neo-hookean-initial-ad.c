#include <ceed.h>
#include "../include/structs.h"
#include "../include/setup-libceed.h"
#include "../problems/problems.h"
#include "../problems/neo-hookean.h"
#include "../qfunctions/common.h"
#include "../qfunctions/finite-strain-neo-hookean.h"
#include "../qfunctions/finite-strain-neo-hookean-initial-ad.h"

static const char *const field_names[] = {"gradu", "Swork", "tape"};
static CeedInt field_sizes[] = {9, 6, 1};

ProblemData finite_strain_neo_Hookean_initial_ad = {
  .setup_geo = SetupGeo,
  .setup_geo_loc = SetupGeo_loc,
  .q_data_size = 10,
  .quadrature_mode = CEED_GAUSS,
  .residual = ElasFSInitialNHF_AD,
  .residual_loc = ElasFSInitialNHF_AD_loc,
  .number_fields_stored = sizeof(field_sizes) / sizeof(*field_sizes),
  .field_names = field_names,
  .field_sizes = field_sizes,
  .jacobian = ElasFSInitialNHdF_AD,
  .jacobian_loc = ElasFSInitialNHdF_AD_loc,
  .energy = ElasFSNHEnergy,
  .energy_loc = ElasFSNHEnergy_loc,
  .diagnostic = ElasFSNHDiagnostic,
  .diagnostic_loc = ElasFSNHDiagnostic_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasFSInitialNH_AD(DM dm, DM dm_energy,
    DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
    PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector force_ceed, CeedVector neumann_ceed,
    CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx,
                               phys_ctx, finite_strain_neo_Hookean_initial_ad,
                               fine_level, num_comp_u, U_g_size, U_loc_size,
                               force_ceed, neumann_ceed, data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasFSInitialNH_AD(DM dm, Ceed ceed,
    AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedLevel(dm, ceed, app_ctx,
                           finite_strain_neo_Hookean_initial_ad,
                           level, num_comp_u, U_g_size, U_loc_size, fine_mult, data);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode ProblemRegister_ElasFSInitialNH_AD(ProblemFunctions
    problem_functions) {
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&problem_functions->setupPhysics, "FSInitial-NH-AD",
                              PhysicsContext_NH); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupSmootherPhysics,
                              "FSInitial-NH-AD", PhysicsSmootherContext_NH); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupLibceedFineLevel,
                              "FSInitial-NH-AD", SetupLibceedFineLevel_ElasFSInitialNH_AD); CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&problem_functions->setupLibceedLevel,
                              "FSInitial-NH-AD", SetupLibceedLevel_ElasFSInitialNH_AD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};
