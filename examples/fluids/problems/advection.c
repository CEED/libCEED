// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Utility functions for setting up ADVECTION

#include "../qfunctions/advection.h"

#include <ceed.h>
#include <petscdm.h>

#include "../navierstokes.h"
#include "../qfunctions/setupgeo.h"
#include "../qfunctions/setupgeo2d.h"

// @brief Create CeedOperator for stabilized mass KSP for explicit timestepping
//
// Only used for SUPG stabilization
PetscErrorCode CreateKSPMassOperator_AdvectionStabilized(User user, CeedOperator *op_mass) {
  Ceed                 ceed = user->ceed;
  CeedInt              num_comp_q, q_data_size;
  CeedQFunction        qf_mass;
  CeedElemRestriction  elem_restr_q, elem_restr_qd_i;
  CeedBasis            basis_q;
  CeedVector           q_data;
  CeedQFunctionContext qf_ctx = NULL;
  PetscInt             dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(user->dm, &dim));
  {  // Get restriction and basis from the RHS function
    CeedOperator     *sub_ops;
    CeedOperatorField field;
    PetscInt          sub_op_index = 0;  // will be 0 for the volume op

    PetscCallCeed(ceed, CeedCompositeOperatorGetSubList(user->op_rhs_ctx->op, &sub_ops));
    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "q", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_q));
    PetscCallCeed(ceed, CeedOperatorFieldGetBasis(field, &basis_q));

    PetscCallCeed(ceed, CeedOperatorGetFieldByName(sub_ops[sub_op_index], "qdata", &field));
    PetscCallCeed(ceed, CeedOperatorFieldGetElemRestriction(field, &elem_restr_qd_i));
    PetscCallCeed(ceed, CeedOperatorFieldGetVector(field, &q_data));

    PetscCallCeed(ceed, CeedOperatorGetContext(sub_ops[sub_op_index], &qf_ctx));
  }

  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_q, &num_comp_q));
  PetscCallCeed(ceed, CeedElemRestrictionGetNumComponents(elem_restr_qd_i, &q_data_size));

  switch (dim) {
    case 2:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Advection2D, MassFunction_Advection2D_loc, &qf_mass));
      break;
    case 3:
      PetscCallCeed(ceed, CeedQFunctionCreateInterior(ceed, 1, MassFunction_Advection, MassFunction_Advection_loc, &qf_mass));
      break;
  }

  PetscCallCeed(ceed, CeedQFunctionSetContext(qf_mass, qf_ctx));
  PetscCallCeed(ceed, CeedQFunctionSetUserFlopsEstimate(qf_mass, 0));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q_dot", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "q", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddInput(qf_mass, "qdata", q_data_size, CEED_EVAL_NONE));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "v", 5, CEED_EVAL_INTERP));
  PetscCallCeed(ceed, CeedQFunctionAddOutput(qf_mass, "Grad_v", 5 * dim, CEED_EVAL_GRAD));

  PetscCallCeed(ceed, CeedOperatorCreate(ceed, qf_mass, NULL, NULL, op_mass));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q_dot", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "q", elem_restr_q, basis_q, user->q_ceed));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "qdata", elem_restr_qd_i, CEED_BASIS_NONE, q_data));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));
  PetscCallCeed(ceed, CeedOperatorSetField(*op_mass, "Grad_v", elem_restr_q, basis_q, CEED_VECTOR_ACTIVE));

  PetscCallCeed(ceed, CeedQFunctionDestroy(&qf_mass));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NS_ADVECTION(ProblemData *problem, DM dm, void *ctx, SimpleBC bc) {
  WindType             wind_type;
  AdvectionICType      advectionic_type;
  BubbleContinuityType bubble_continuity_type;
  StabilizationType    stab;
  StabilizationTauType stab_tau;
  SetupContextAdv      setup_context;
  User                 user = *(User *)ctx;
  MPI_Comm             comm = user->comm;
  Ceed                 ceed = user->ceed;
  PetscBool            implicit;
  AdvectionContext     advection_ctx;
  CeedQFunctionContext advection_context;
  PetscInt             dim;

  PetscFunctionBeginUser;
  PetscCall(PetscCalloc1(1, &setup_context));
  PetscCall(PetscCalloc1(1, &advection_ctx));
  PetscCall(DMGetDimension(dm, &dim));

  // ------------------------------------------------------
  //               SET UP ADVECTION
  // ------------------------------------------------------
  switch (dim) {
    case 2:
      problem->dim                               = 2;
      problem->q_data_size_vol                   = 5;
      problem->q_data_size_sur                   = 3;
      problem->setup_vol.qfunction               = Setup2d;
      problem->setup_vol.qfunction_loc           = Setup2d_loc;
      problem->setup_sur.qfunction               = SetupBoundary2d;
      problem->setup_sur.qfunction_loc           = SetupBoundary2d_loc;
      problem->ics.qfunction                     = ICsAdvection2d;
      problem->ics.qfunction_loc                 = ICsAdvection2d_loc;
      problem->ics_l2rhs.qfunction               = ICsAdvection2d_L2Rhs;
      problem->ics_l2rhs.qfunction_loc           = ICsAdvection2d_L2Rhs_loc;
      problem->apply_vol_rhs.qfunction           = RHS_Advection2d;
      problem->apply_vol_rhs.qfunction_loc       = RHS_Advection2d_loc;
      problem->apply_vol_ifunction.qfunction     = IFunction_Advection2d;
      problem->apply_vol_ifunction.qfunction_loc = IFunction_Advection2d_loc;
      problem->apply_inflow.qfunction            = Advection2d_InOutFlow;
      problem->apply_inflow.qfunction_loc        = Advection2d_InOutFlow_loc;
      problem->non_zero_time                     = PETSC_TRUE;
      problem->print_info                        = PRINT_ADVECTION;
      break;
    case 3:
      problem->dim                               = 3;
      problem->q_data_size_vol                   = 10;
      problem->q_data_size_sur                   = 10;
      problem->setup_vol.qfunction               = Setup;
      problem->setup_vol.qfunction_loc           = Setup_loc;
      problem->setup_sur.qfunction               = SetupBoundary;
      problem->setup_sur.qfunction_loc           = SetupBoundary_loc;
      problem->ics.qfunction                     = ICsAdvection;
      problem->ics.qfunction_loc                 = ICsAdvection_loc;
      problem->ics_l2rhs.qfunction               = ICsAdvection_L2Rhs;
      problem->ics_l2rhs.qfunction_loc           = ICsAdvection_L2Rhs_loc;
      problem->apply_vol_rhs.qfunction           = RHS_Advection;
      problem->apply_vol_rhs.qfunction_loc       = RHS_Advection_loc;
      problem->apply_vol_ifunction.qfunction     = IFunction_Advection;
      problem->apply_vol_ifunction.qfunction_loc = IFunction_Advection_loc;
      problem->apply_inflow.qfunction            = Advection_InOutFlow;
      problem->apply_inflow.qfunction_loc        = Advection_InOutFlow_loc;
      problem->non_zero_time                     = PETSC_FALSE;
      problem->print_info                        = PRINT_ADVECTION;
      break;
  }

  // ------------------------------------------------------
  //             Create the libCEED context
  // ------------------------------------------------------
  CeedScalar rc              = 1000.;  // m (Radius of bubble)
  CeedScalar CtauS           = 0.;     // dimensionless
  PetscBool  strong_form     = PETSC_FALSE;
  CeedScalar E_wind          = 1.e6;  // J
  CeedScalar Ctau_a          = PetscPowScalarInt(user->app_ctx->degree, 2);
  CeedScalar Ctau_t          = 0.;
  PetscReal  wind[3]         = {1., 0, 0};  // m/s
  CeedScalar diffusion_coeff = 0.;
  PetscReal  domain_min[3], domain_max[3], domain_size[3];
  PetscCall(DMGetBoundingBox(dm, domain_min, domain_max));
  for (PetscInt i = 0; i < problem->dim; i++) domain_size[i] = domain_max[i] - domain_min[i];

  // ------------------------------------------------------
  //             Create the PETSc context
  // ------------------------------------------------------
  PetscScalar meter    = 1e-2;  // 1 meter in scaled length units
  PetscScalar kilogram = 1e-6;  // 1 kilogram in scaled mass units
  PetscScalar second   = 1e-2;  // 1 second in scaled time units
  PetscScalar Joule;

  // ------------------------------------------------------
  //              Command line Options
  // ------------------------------------------------------
  PetscOptionsBegin(comm, NULL, "Options for ADVECTION problem", NULL);
  // -- Physics
  PetscCall(PetscOptionsScalar("-rc", "Characteristic radius of thermal bubble", NULL, rc, &rc, NULL));
  PetscBool translation;
  PetscCall(PetscOptionsEnum("-wind_type", "Wind type in Advection", NULL, WindTypes, (PetscEnum)(wind_type = WIND_ROTATION), (PetscEnum *)&wind_type,
                             &translation));
  PetscInt  n = problem->dim;
  PetscBool user_wind;
  PetscCall(PetscOptionsRealArray("-wind_translation", "Constant wind vector", NULL, wind, &n, &user_wind));
  PetscCall(PetscOptionsScalar("-diffusion_coeff", "Diffusion coefficient", NULL, diffusion_coeff, &diffusion_coeff, NULL));
  PetscCall(PetscOptionsScalar("-CtauS", "Scale coefficient for tau (nondimensional)", NULL, CtauS, &CtauS, NULL));
  PetscCall(PetscOptionsBool("-strong_form", "Strong (true) or weak/integrated by parts (false) advection residual", NULL, strong_form, &strong_form,
                             NULL));
  PetscCall(PetscOptionsScalar("-E_wind", "Total energy of inflow wind", NULL, E_wind, &E_wind, NULL));
  PetscCall(PetscOptionsEnum("-advection_ic_type", "Initial condition for Advection problem", NULL, AdvectionICTypes,
                             (PetscEnum)(advectionic_type = ADVECTIONIC_BUBBLE_SPHERE), (PetscEnum *)&advectionic_type, NULL));
  bubble_continuity_type = problem->dim == 3 ? BUBBLE_CONTINUITY_SMOOTH : BUBBLE_CONTINUITY_COSINE;
  PetscCall(PetscOptionsEnum("-bubble_continuity", "Smooth, back_sharp, or thick", NULL, BubbleContinuityTypes, (PetscEnum)bubble_continuity_type,
                             (PetscEnum *)&bubble_continuity_type, NULL));
  PetscCall(PetscOptionsEnum("-stab", "Stabilization method", NULL, StabilizationTypes, (PetscEnum)(stab = STAB_NONE), (PetscEnum *)&stab, NULL));
  PetscCall(PetscOptionsEnum("-stab_tau", "Stabilization constant, tau", NULL, StabilizationTauTypes, (PetscEnum)(stab_tau = STAB_TAU_CTAU),
                             (PetscEnum *)&stab_tau, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_t", "Stabilization time constant", NULL, Ctau_t, &Ctau_t, NULL));
  PetscCall(PetscOptionsScalar("-Ctau_a", "Coefficient for the stabilization ", NULL, Ctau_a, &Ctau_a, NULL));
  PetscCall(PetscOptionsBool("-implicit", "Use implicit (IFunction) formulation", NULL, implicit = PETSC_FALSE, &implicit, NULL));

  
  {
    PetscBool use_l2_project;
    PetscCall(PetscOptionsBool("-use_l2_project", "Use L^2 projection for initial condition", NULL, use_l2_project = PETSC_FALSE, &use_l2_project, NULL));
    if (!use_l2_project) {
      problem->ics_l2rhs.qfunction     = NULL;
      problem->ics_l2rhs.qfunction_loc = NULL;
    }
  }

  // -- Units
  PetscCall(PetscOptionsScalar("-units_meter", "1 meter in scaled length units", NULL, meter, &meter, NULL));
  meter = fabs(meter);
  PetscCall(PetscOptionsScalar("-units_kilogram", "1 kilogram in scaled mass units", NULL, kilogram, &kilogram, NULL));
  kilogram = fabs(kilogram);
  PetscCall(PetscOptionsScalar("-units_second", "1 second in scaled time units", NULL, second, &second, NULL));
  second = fabs(second);

  // -- Warnings
  if (wind_type == WIND_ROTATION && user_wind) {
    PetscCall(PetscPrintf(comm, "Warning! Use -wind_translation only with -wind_type translation\n"));
  }
  if (wind_type == WIND_TRANSLATION && advectionic_type == ADVECTIONIC_BUBBLE_CYLINDER && wind[2] != 0.) {
    wind[2] = 0;
    PetscCall(
        PetscPrintf(comm, "Warning! Background wind in the z direction should be zero (-wind_translation x,x,0) with -advection_ic_type cylinder\n"));
  }
  if (stab == STAB_NONE && CtauS != 0) {
    PetscCall(PetscPrintf(comm, "Warning! Use -CtauS only with -stab su or -stab supg\n"));
  }
  PetscOptionsEnd();

  if (stab == STAB_SUPG) problem->create_mass_operator = CreateKSPMassOperator_AdvectionStabilized;

  // ------------------------------------------------------
  //           Set up the PETSc context
  // ------------------------------------------------------
  // -- Define derived units
  Joule = kilogram * PetscSqr(meter) / PetscSqr(second);

  user->units->meter    = meter;
  user->units->kilogram = kilogram;
  user->units->second   = second;
  user->units->Joule    = Joule;

  // ------------------------------------------------------
  //           Set up the libCEED context
  // ------------------------------------------------------
  // -- Scale variables to desired units
  E_wind *= Joule;
  rc = fabs(rc) * meter;
  for (PetscInt i = 0; i < problem->dim; i++) {
    wind[i] *= (meter / second);
    domain_size[i] *= meter;
  }
  problem->dm_scale = meter;

  // -- Setup Context
  setup_context->rc                     = rc;
  setup_context->lx                     = domain_size[0];
  setup_context->ly                     = domain_size[1];
  setup_context->lz                     = problem->dim == 3 ? domain_size[2] : 0.;
  setup_context->wind[0]                = wind[0];
  setup_context->wind[1]                = wind[1];
  setup_context->wind[2]                = problem->dim == 3 ? wind[2] : 0.;
  setup_context->wind_type              = wind_type;
  setup_context->initial_condition_type = advectionic_type;
  setup_context->bubble_continuity_type = bubble_continuity_type;
  setup_context->time                   = 0;

  // -- QFunction Context
  user->phys->implicit             = implicit;
  advection_ctx->CtauS             = CtauS;
  advection_ctx->E_wind            = E_wind;
  advection_ctx->implicit          = implicit;
  advection_ctx->strong_form       = strong_form;
  advection_ctx->stabilization     = stab;
  advection_ctx->stabilization_tau = stab_tau;
  advection_ctx->Ctau_a            = Ctau_a;
  advection_ctx->Ctau_t            = Ctau_t;
  advection_ctx->diffusion_coeff   = diffusion_coeff;

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &problem->ics.qfunction_context));
  PetscCallCeed(ceed,
                CeedQFunctionContextSetData(problem->ics.qfunction_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*setup_context), setup_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(problem->ics.qfunction_context, CEED_MEM_HOST, FreeContextPetsc));
  if (problem->ics_l2rhs.qfunction) PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(problem->ics.qfunction_context, &problem->ics_l2rhs.qfunction_context));

  PetscCallCeed(ceed, CeedQFunctionContextCreate(user->ceed, &advection_context));
  PetscCallCeed(ceed, CeedQFunctionContextSetData(advection_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*advection_ctx), advection_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextSetDataDestroy(advection_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCeed(ceed, CeedQFunctionContextRegisterDouble(advection_context, "timestep size", offsetof(struct AdvectionContext_, dt), 1,
                                                         "Size of timestep, delta t"));
  problem->apply_vol_rhs.qfunction_context = advection_context;
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(advection_context, &problem->apply_vol_ifunction.qfunction_context));
  PetscCallCeed(ceed, CeedQFunctionContextReferenceCopy(advection_context, &problem->apply_inflow.qfunction_context));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PRINT_ADVECTION(User user, ProblemData *problem, AppCtx app_ctx) {
  MPI_Comm         comm = user->comm;
  Ceed             ceed = user->ceed;
  SetupContextAdv  setup_ctx;
  AdvectionContext advection_ctx;

  PetscFunctionBeginUser;
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->ics.qfunction_context, CEED_MEM_HOST, &setup_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextGetData(problem->apply_vol_rhs.qfunction_context, CEED_MEM_HOST, &advection_ctx));
  PetscCall(PetscPrintf(comm,
                        "  Problem:\n"
                        "    Problem Name                       : %s\n"
                        "    Stabilization                      : %s\n"
                        "    Initial Condition Type             : %s\n"
                        "    Bubble Continuity                  : %s\n"
                        "    Wind Type                          : %s\n",
                        app_ctx->problem_name, StabilizationTypes[advection_ctx->stabilization], AdvectionICTypes[setup_ctx->initial_condition_type],
                        BubbleContinuityTypes[setup_ctx->bubble_continuity_type], WindTypes[setup_ctx->wind_type]));

  if (setup_ctx->wind_type == WIND_TRANSLATION) {
    switch (problem->dim) {
      case 2:
        PetscCall(PetscPrintf(comm, "    Background Wind                    : %f,%f\n", setup_ctx->wind[0], setup_ctx->wind[1]));
        break;
      case 3:
        PetscCall(
            PetscPrintf(comm, "    Background Wind                    : %f,%f,%f\n", setup_ctx->wind[0], setup_ctx->wind[1], setup_ctx->wind[2]));
        break;
    }
  }
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->ics.qfunction_context, &setup_ctx));
  PetscCallCeed(ceed, CeedQFunctionContextRestoreData(problem->apply_vol_rhs.qfunction_context, &advection_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
