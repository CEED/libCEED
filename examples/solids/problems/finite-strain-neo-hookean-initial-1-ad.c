#include <ceed.h>
#include "../include/structs.h"
#include "../include/setup-libceed.h"
#include "../problems/problems.h"
#include "../problems/neo-hookean.h"
#include "../qfunctions/common.h"
#include "../qfunctions/finite-strain-neo-hookean-initial-1-ad.h"

static const char *const field_names[] = {"gradu"};
static CeedInt field_sizes[] = {9};

ProblemData finite_strain_neo_Hookean_initial_1_ad = {
  .setup_geo = SetupGeo,
  .setup_geo_loc = SetupGeo_loc,
  .geo_data_size = 10,
  .quadrature_mode = CEED_GAUSS,
  .residual = ElasFSInitialNH1F_AD,
  .residual_loc = ElasFSInitialNH1F_AD_loc,
  .number_fields_stored = 1,
  .field_names = field_names,
  .field_sizes = field_sizes,
  .jacobian = ElasFSInitialNH1dF_AD,
  .jacobian_loc = ElasFSInitialNH1dF_AD_loc,
  .energy = ElasFSInitialNH1Energy_AD,
  .energy_loc = ElasFSInitialNH1Energy_AD_loc,
  .diagnostic = ElasFSInitialNH1Diagnostic_AD,
  .diagnostic_loc = ElasFSInitialNH1Diagnostic_AD_loc,
};

PetscErrorCode SetupLibceedFineLevel_ElasFSInitialNH1_AD(DM dm, DM dm_energy,
    DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
    PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector force_ceed, CeedVector neumann_ceed,
    CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedFineLevel(dm, dm_energy, dm_diagnostic, ceed, app_ctx,
                               phys_ctx, finite_strain_neo_Hookean_initial_1_ad,
                               fine_level, num_comp_u, U_g_size, U_loc_size,
                               force_ceed, neumann_ceed, data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

PetscErrorCode SetupLibceedLevel_ElasFSInitialNH1_AD(DM dm, Ceed ceed,
    AppCtx app_ctx, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
    PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SetupLibceedLevel(dm, ceed, app_ctx,
                           finite_strain_neo_Hookean_initial_1_ad,
                           level, num_comp_u, U_g_size, U_loc_size, fine_mult, data);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
};
