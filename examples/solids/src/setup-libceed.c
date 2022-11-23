// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED setup for solid mechanics example using PETSc

#include "../include/setup-libceed.h"

#include "../include/structs.h"
#include "../include/utils.h"
#include "../qfunctions/constant-force.h"      // Constant forcing function
#include "../qfunctions/manufactured-force.h"  // Manufactured solution forcing
#include "../qfunctions/traction-boundary.h"   // Traction boundaries

#if PETSC_VERSION_LT(3, 14, 0)
#define DMPlexGetClosureIndices(a, b, c, d, e, f, g, h, i) DMPlexGetClosureIndices(a, b, c, d, f, g, i)
#define DMPlexRestoreClosureIndices(a, b, c, d, e, f, g, h, i) DMPlexRestoreClosureIndices(a, b, c, d, f, g, i)
#endif

// -----------------------------------------------------------------------------
// Problem options
// -----------------------------------------------------------------------------
// Forcing function data
forcingData forcing_options[3] = {
    [FORCE_NONE]  = {.setup_forcing = NULL,               .setup_forcing_loc = NULL                  },
    [FORCE_CONST] = {.setup_forcing = SetupConstantForce, .setup_forcing_loc = SetupConstantForce_loc},
    [FORCE_MMS]   = {.setup_forcing = SetupMMSForce,      .setup_forcing_loc = SetupMMSForce_loc     }
};

// -----------------------------------------------------------------------------
// libCEED Functions
// -----------------------------------------------------------------------------
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedInt level, CeedData data) {
  PetscFunctionBegin;

  // Vectors
  CeedVectorDestroy(&data->x_ceed);
  CeedVectorDestroy(&data->y_ceed);
  CeedVectorDestroy(&data->geo_data);
  for (CeedInt i = 0; i < SOLIDS_MAX_NUMBER_FIELDS; i++) CeedVectorDestroy(&data->stored_fields[i]);
  CeedVectorDestroy(&data->geo_data_diagnostic);
  CeedVectorDestroy(&data->true_soln);
  // Restrictions
  CeedElemRestrictionDestroy(&data->elem_restr_x);
  CeedElemRestrictionDestroy(&data->elem_restr_u);
  CeedElemRestrictionDestroy(&data->elem_restr_geo_data_i);
  for (CeedInt i = 0; i < SOLIDS_MAX_NUMBER_FIELDS; i++) CeedElemRestrictionDestroy(&data->elem_restr_stored_fields_i[i]);
  CeedElemRestrictionDestroy(&data->elem_restr_energy);
  CeedElemRestrictionDestroy(&data->elem_restr_diagnostic);
  CeedElemRestrictionDestroy(&data->elem_restr_geo_data_diagnostic_i);
  // Bases
  CeedBasisDestroy(&data->basis_x);
  CeedBasisDestroy(&data->basis_u);
  CeedBasisDestroy(&data->basis_energy);
  CeedBasisDestroy(&data->basis_diagnostic);
  // QFunctions
  CeedQFunctionDestroy(&data->qf_residual);
  CeedQFunctionDestroy(&data->qf_jacobian);
  CeedQFunctionDestroy(&data->qf_energy);
  CeedQFunctionDestroy(&data->qf_diagnostic);
  // Operators
  CeedOperatorDestroy(&data->op_residual);
  CeedOperatorDestroy(&data->op_jacobian);
  CeedOperatorDestroy(&data->op_energy);
  CeedOperatorDestroy(&data->op_diagnostic);
  // Restriction and Prolongation data
  CeedBasisDestroy(&data->basis_c_to_f);
  CeedOperatorDestroy(&data->op_prolong);
  CeedOperatorDestroy(&data->op_restrict);

  PetscCall(PetscFree(data));

  PetscFunctionReturn(0);
};

// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp, 1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES, elem_restr_offsets, elem_restr);
  PetscCall(PetscFree(elem_restr_offsets));

  PetscFunctionReturn(0);
};

// Utility function to get Ceed Restriction for each domain
PetscErrorCode GetRestrictionForDomain(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, PetscInt value, CeedInt Q, CeedInt q_data_size,
                                       CeedElemRestriction *elem_restr_q, CeedElemRestriction *elem_restr_x, CeedElemRestriction *elem_restr_qd_i) {
  DM      dm_coord;
  CeedInt dim, num_local_elem;
  CeedInt Q_dim;

  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));
  dim -= height;
  Q_dim = CeedIntPow(Q, dim);
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  if (elem_restr_q) {
    PetscCall(CreateRestrictionFromPlex(ceed, dm, height, domain_label, value, elem_restr_q));
  }
  if (elem_restr_x) {
    PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label, value, elem_restr_x));
  }
  if (elem_restr_qd_i) {
    CeedElemRestrictionGetNumElements(*elem_restr_q, &num_local_elem);
    CeedElemRestrictionCreateStrided(ceed, num_local_elem, Q_dim, q_data_size, q_data_size * num_local_elem * Q_dim, CEED_STRIDES_BACKEND,
                                     elem_restr_qd_i);
  }

  PetscFunctionReturn(0);
};

// Set up libCEED on the fine grid for a given degree
PetscErrorCode SetupLibceedFineLevel(DM dm, DM dm_energy, DM dm_diagnostic, Ceed ceed, AppCtx app_ctx, CeedQFunctionContext phys_ctx,
                                     ProblemData problem_data, PetscInt fine_level, PetscInt num_comp_u, PetscInt U_g_size, PetscInt U_loc_size,
                                     CeedVector force_ceed, CeedVector neumann_ceed, CeedData *data) {
  CeedInt            P = app_ctx->level_degrees[fine_level] + 1;
  CeedInt            Q = app_ctx->level_degrees[fine_level] + 1 + app_ctx->q_extra;
  CeedInt            dim, num_comp_x, num_comp_e = 1, num_comp_d = 5;
  CeedInt            num_qpts;
  CeedInt            q_data_size    = problem_data.q_data_size;
  forcingType        forcing_choice = app_ctx->forcing_choice;
  DM                 dm_coord;
  Vec                coords;
  PetscInt           c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedVector         x_coord;
  CeedQFunction      qf_setup_geo, qf_residual, qf_jacobian, qf_energy, qf_diagnostic;
  CeedOperator       op_setup_geo, op_residual, op_jacobian, op_energy, op_diagnostic;

  PetscFunctionBeginUser;

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  PetscCall(DMGetDimension(dm, &dim));
  num_comp_x = dim;
  // -- Coordinate basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, problem_data.quadrature_mode, &data[fine_level]->basis_x);
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, Q, problem_data.quadrature_mode, &data[fine_level]->basis_u);
  // -- Energy basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_e, P, Q, problem_data.quadrature_mode, &data[fine_level]->basis_energy);
  // -- Diagnostic output basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, P, CEED_GAUSS_LOBATTO, &data[fine_level]->basis_diagnostic);

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));

  // -- Coordinate restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, 0, 0, 0, &(data[fine_level]->elem_restr_x)));
  // -- Solution restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &data[fine_level]->elem_restr_u));
  // -- Energy restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm_energy, 0, 0, 0, &data[fine_level]->elem_restr_energy));
  // -- Diagnostic data restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm_diagnostic, 0, 0, 0, &data[fine_level]->elem_restr_diagnostic));

  // -- Stored data at quadrature points
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  CeedBasisGetNumQuadraturePoints(data[fine_level]->basis_u, &num_qpts);
  // ---- Geometric data restriction, residual and Jacobian operators
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, num_elem * num_qpts * q_data_size, CEED_STRIDES_BACKEND,
                                   &data[fine_level]->elem_restr_geo_data_i);
  // ---- Stored field restrictions
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, problem_data.field_sizes[i], num_elem * num_qpts * problem_data.field_sizes[i],
                                     CEED_STRIDES_BACKEND, &data[fine_level]->elem_restr_stored_fields_i[i]);
  }
  // ---- Geometric data restriction, diagnostic operator
  CeedElemRestrictionCreateStrided(ceed, num_elem, P * P * P, q_data_size, num_elem * P * P * P * q_data_size, CEED_STRIDES_BACKEND,
                                   &data[fine_level]->elem_restr_geo_data_diagnostic_i);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coordArray));

  CeedElemRestrictionCreateVector(data[fine_level]->elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coordArray);
  PetscCall(VecRestoreArrayRead(coords, &coordArray));

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables
  CeedVectorCreate(ceed, U_loc_size, &data[fine_level]->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &data[fine_level]->y_ceed);
  // -- Geometric data vector
  CeedVectorCreate(ceed, num_elem * num_qpts * q_data_size, &data[fine_level]->geo_data);
  // -- Stored field vectors
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedVectorCreate(ceed, num_elem * num_qpts * problem_data.field_sizes[i], &data[fine_level]->stored_fields[i]);
  }
  // -- Collocated geometric data vector
  CeedVectorCreate(ceed, num_elem * P * P * P * q_data_size, &data[fine_level]->geo_data_diagnostic);

  // ---------------------------------------------------------------------------
  // Geometric factor computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the quadrature data
  //   geo_data returns dXdx_i,j and w * det.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.setup_geo, problem_data.setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);
  // -- Operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "dx", data[fine_level]->elem_restr_x, data[fine_level]->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, data[fine_level]->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "qdata", data[fine_level]->elem_restr_geo_data_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Compute the quadrature data
  CeedOperatorApply(op_setup_geo, x_coord, data[fine_level]->geo_data, CEED_REQUEST_IMMEDIATE);
  // -- Cleanup
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);

  // ---------------------------------------------------------------------------
  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the
  //   non-linear PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.residual, problem_data.residual_loc, &qf_residual);
  CeedQFunctionAddInput(qf_residual, "du", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_residual, "dv", num_comp_u * dim, CEED_EVAL_GRAD);
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedQFunctionAddOutput(qf_residual, problem_data.field_names[i], problem_data.field_sizes[i], CEED_EVAL_NONE);
  }
  CeedQFunctionSetContext(qf_residual, phys_ctx);
  // -- Operator
  CeedOperatorCreate(ceed, qf_residual, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_residual);
  CeedOperatorSetField(op_residual, "du", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "qdata", data[fine_level]->elem_restr_geo_data_i, CEED_BASIS_COLLOCATED, data[fine_level]->geo_data);
  CeedOperatorSetField(op_residual, "dv", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedOperatorSetField(op_residual, problem_data.field_names[i], data[fine_level]->elem_restr_stored_fields_i[i], CEED_BASIS_COLLOCATED,
                         data[fine_level]->stored_fields[i]);
  }
  // -- Save libCEED data
  data[fine_level]->qf_residual = qf_residual;
  data[fine_level]->op_residual = op_residual;

  // ---------------------------------------------------------------------------
  // Jacobian evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the action of the
  //   Jacobian for each linear solve.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.jacobian, problem_data.jacobian_loc, &qf_jacobian);
  CeedQFunctionAddInput(qf_jacobian, "delta du", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_jacobian, "qdata", q_data_size, CEED_EVAL_NONE);
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedQFunctionAddInput(qf_jacobian, problem_data.field_names[i], problem_data.field_sizes[i], CEED_EVAL_NONE);
  }
  CeedQFunctionAddOutput(qf_jacobian, "delta dv", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionSetContext(qf_jacobian, phys_ctx);
  // -- Operator
  CeedOperatorCreate(ceed, qf_jacobian, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_jacobian);
  CeedOperatorSetField(op_jacobian, "delta du", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_jacobian, "qdata", data[fine_level]->elem_restr_geo_data_i, CEED_BASIS_COLLOCATED, data[fine_level]->geo_data);
  CeedOperatorSetField(op_jacobian, "delta dv", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  for (CeedInt i = 0; i < problem_data.number_fields_stored; i++) {
    CeedOperatorSetField(op_jacobian, problem_data.field_names[i], data[fine_level]->elem_restr_stored_fields_i[i], CEED_BASIS_COLLOCATED,
                         data[fine_level]->stored_fields[i]);
  }
  // -- Save libCEED data
  data[fine_level]->qf_jacobian = qf_jacobian;
  data[fine_level]->op_jacobian = op_jacobian;

  // ---------------------------------------------------------------------------
  // Traction boundary conditions, if needed
  // ---------------------------------------------------------------------------
  if (app_ctx->bc_traction_count > 0) {
    // -- Setup
    DMLabel domain_label;
    PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));
    PetscCall(DMGetDimension(dm, &dim));

    // -- Basis
    CeedInt   height = 1;
    CeedBasis basis_x_face, basis_u_face;
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, num_comp_x, 2, Q, problem_data.quadrature_mode, &basis_x_face);
    CeedBasisCreateTensorH1Lagrange(ceed, dim - height, num_comp_u, P, Q, problem_data.quadrature_mode, &basis_u_face);
    // -- QFunction
    CeedQFunction        qf_traction;
    CeedQFunctionContext traction_ctx;
    CeedQFunctionCreateInterior(ceed, 1, SetupTractionBCs, SetupTractionBCs_loc, &qf_traction);
    CeedQFunctionContextCreate(ceed, &traction_ctx);
    CeedQFunctionSetContext(qf_traction, traction_ctx);
    CeedQFunctionAddInput(qf_traction, "dx", num_comp_x * (num_comp_x - height), CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_traction, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_traction, "v", num_comp_u, CEED_EVAL_INTERP);

    // -- Compute contribution on each boundary face
    for (CeedInt i = 0; i < app_ctx->bc_traction_count; i++) {
      CeedElemRestriction elem_restr_x_face, elem_restr_u_face;
      CeedOperator        op_traction;
      CeedQFunctionContextSetData(traction_ctx, CEED_MEM_HOST, CEED_USE_POINTER, 3 * sizeof(CeedScalar), app_ctx->bc_traction_vector[i]);
      // Setup restriction
      PetscCall(
          GetRestrictionForDomain(ceed, dm, 1, domain_label, app_ctx->bc_traction_faces[i], Q, 0, &elem_restr_u_face, &elem_restr_x_face, NULL));
      // ---- Create boundary Operator
      CeedOperatorCreate(ceed, qf_traction, NULL, NULL, &op_traction);
      CeedOperatorSetField(op_traction, "dx", elem_restr_x_face, basis_x_face, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op_traction, "weight", CEED_ELEMRESTRICTION_NONE, basis_x_face, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_traction, "v", elem_restr_u_face, basis_u_face, CEED_VECTOR_ACTIVE);
      // ---- Compute traction on face
      CeedOperatorApplyAdd(op_traction, x_coord, neumann_ceed, CEED_REQUEST_IMMEDIATE);
      // ---- Cleanup
      CeedElemRestrictionDestroy(&elem_restr_x_face);
      CeedElemRestrictionDestroy(&elem_restr_u_face);
      CeedOperatorDestroy(&op_traction);
    }
    // -- Cleanup
    CeedBasisDestroy(&basis_x_face);
    CeedBasisDestroy(&basis_u_face);
    CeedQFunctionDestroy(&qf_traction);
    CeedQFunctionContextDestroy(&traction_ctx);
  }

  // ---------------------------------------------------------------------------
  // Forcing term, if needed
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the forcing term (RHS)
  //   for the non-linear PDE.
  // ---------------------------------------------------------------------------
  if (forcing_choice != FORCE_NONE) {
    CeedQFunction qf_setup_force;
    CeedOperator  op_setup_force;

    // -- QFunction
    CeedQFunctionCreateInterior(ceed, 1, forcing_options[forcing_choice].setup_forcing, forcing_options[forcing_choice].setup_forcing_loc,
                                &qf_setup_force);
    CeedQFunctionAddInput(qf_setup_force, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_force, "qdata", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_force, "force", num_comp_u, CEED_EVAL_INTERP);
    if (forcing_choice == FORCE_MMS) {
      CeedQFunctionSetContext(qf_setup_force, phys_ctx);
    } else {
      CeedQFunctionContext ctxForcing;
      CeedQFunctionContextCreate(ceed, &ctxForcing);
      CeedQFunctionContextSetData(ctxForcing, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*app_ctx->forcing_vector), app_ctx->forcing_vector);
      CeedQFunctionSetContext(qf_setup_force, ctxForcing);
      CeedQFunctionContextDestroy(&ctxForcing);
    }
    // -- Operator
    CeedOperatorCreate(ceed, qf_setup_force, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_force);
    CeedOperatorSetField(op_setup_force, "x", data[fine_level]->elem_restr_x, data[fine_level]->basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_force, "qdata", data[fine_level]->elem_restr_geo_data_i, CEED_BASIS_COLLOCATED, data[fine_level]->geo_data);
    CeedOperatorSetField(op_setup_force, "force", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
    // -- Compute forcing term
    CeedOperatorApply(op_setup_force, x_coord, force_ceed, CEED_REQUEST_IMMEDIATE);
    // -- Cleanup
    CeedQFunctionDestroy(&qf_setup_force);
    CeedOperatorDestroy(&op_setup_force);
  }

  // ---------------------------------------------------------------------------
  // True solution, for MMS
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the true solution at
  //   the mesh nodes for validation with the manufactured solution.
  // ---------------------------------------------------------------------------
  if (problem_data.true_soln) {
    CeedScalar       *true_array;
    const CeedScalar *mult_array;
    CeedVector        mult_vec;
    CeedBasis         basis_x_true;
    CeedQFunction     qf_true;
    CeedOperator      op_true;

    // -- Solution vector
    CeedVectorCreate(ceed, U_loc_size, &(data[fine_level]->true_soln));
    // -- Basis
    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, P, CEED_GAUSS_LOBATTO, &basis_x_true);
    // QFunction
    CeedQFunctionCreateInterior(ceed, 1, problem_data.true_soln, problem_data.true_soln_loc, &qf_true);
    CeedQFunctionAddInput(qf_true, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_true, "true solution", num_comp_u, CEED_EVAL_NONE);
    // Operator
    CeedOperatorCreate(ceed, qf_true, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_true);
    CeedOperatorSetField(op_true, "x", data[fine_level]->elem_restr_x, basis_x_true, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_true, "true solution", data[fine_level]->elem_restr_u, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    // -- Compute true solution
    CeedOperatorApply(op_true, x_coord, data[fine_level]->true_soln, CEED_REQUEST_IMMEDIATE);
    // -- Multiplicity calculation
    CeedElemRestrictionCreateVector(data[fine_level]->elem_restr_u, &mult_vec, NULL);
    CeedVectorSetValue(mult_vec, 0.);
    CeedElemRestrictionGetMultiplicity(data[fine_level]->elem_restr_u, mult_vec);
    // -- Multiplicity correction
    CeedVectorGetArray(data[fine_level]->true_soln, CEED_MEM_HOST, &true_array);
    CeedVectorGetArrayRead(mult_vec, CEED_MEM_HOST, &mult_array);
    for (CeedInt i = 0; i < U_loc_size; i++) true_array[i] /= mult_array[i];
    CeedVectorRestoreArray(data[fine_level]->true_soln, &true_array);
    CeedVectorRestoreArrayRead(mult_vec, &mult_array);
    // -- Cleanup
    CeedVectorDestroy(&mult_vec);
    CeedBasisDestroy(&basis_x_true);
    CeedQFunctionDestroy(&qf_true);
    CeedOperatorDestroy(&op_true);
  }

  // ---------------------------------------------------------------------------
  // Local energy computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the strain energy
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.energy, problem_data.energy_loc, &qf_energy);
  CeedQFunctionAddInput(qf_energy, "du", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_energy, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_energy, "energy", num_comp_e, CEED_EVAL_INTERP);
  CeedQFunctionSetContext(qf_energy, phys_ctx);
  // -- Operator
  CeedOperatorCreate(ceed, qf_energy, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_energy);
  CeedOperatorSetField(op_energy, "du", data[fine_level]->elem_restr_u, data[fine_level]->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_energy, "qdata", data[fine_level]->elem_restr_geo_data_i, CEED_BASIS_COLLOCATED, data[fine_level]->geo_data);
  CeedOperatorSetField(op_energy, "energy", data[fine_level]->elem_restr_energy, data[fine_level]->basis_energy, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fine_level]->qf_energy = qf_energy;
  data[fine_level]->op_energy = op_energy;

  // ---------------------------------------------------------------------------
  // Diagnostic value computation
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes nodal diagnostic quantities
  // ---------------------------------------------------------------------------
  // Geometric factors
  // -- Coordinate basis
  CeedBasis basis_x;
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, CEED_GAUSS_LOBATTO, &basis_x);
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.setup_geo, problem_data.setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);
  // -- Operator
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "dx", data[fine_level]->elem_restr_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "qdata", data[fine_level]->elem_restr_geo_data_diagnostic_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Compute the quadrature data
  CeedOperatorApply(op_setup_geo, x_coord, data[fine_level]->geo_data_diagnostic, CEED_REQUEST_IMMEDIATE);
  // -- Cleanup
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);

  // Diagnostic quantities
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data.diagnostic, problem_data.diagnostic_loc, &qf_diagnostic);
  CeedQFunctionAddInput(qf_diagnostic, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_diagnostic, "du", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_diagnostic, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_diagnostic, "diagnostic values", num_comp_u + num_comp_d, CEED_EVAL_NONE);
  CeedQFunctionSetContext(qf_diagnostic, phys_ctx);
  // -- Operator
  CeedOperatorCreate(ceed, qf_diagnostic, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_diagnostic);
  CeedOperatorSetField(op_diagnostic, "u", data[fine_level]->elem_restr_u, data[fine_level]->basis_diagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diagnostic, "du", data[fine_level]->elem_restr_u, data[fine_level]->basis_diagnostic, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_diagnostic, "qdata", data[fine_level]->elem_restr_geo_data_diagnostic_i, CEED_BASIS_COLLOCATED,
                       data[fine_level]->geo_data_diagnostic);
  CeedOperatorSetField(op_diagnostic, "diagnostic values", data[fine_level]->elem_restr_diagnostic, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data
  data[fine_level]->qf_diagnostic = qf_diagnostic;
  data[fine_level]->op_diagnostic = op_diagnostic;

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
};

// Set up libCEED multigrid level for a given degree
//   Prolongation and Restriction are between level and level+1
PetscErrorCode SetupLibceedLevel(DM dm, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, PetscInt level, PetscInt num_comp_u, PetscInt U_g_size,
                                 PetscInt U_loc_size, CeedVector fine_mult, CeedData *data) {
  CeedInt      fine_level = app_ctx->num_levels - 1;
  CeedInt      P          = app_ctx->level_degrees[level] + 1;
  CeedInt      Q          = app_ctx->level_degrees[fine_level] + 1 + app_ctx->q_extra;
  CeedInt      dim;
  CeedOperator op_jacobian, op_prolong, op_restrict;

  PetscFunctionBeginUser;

  PetscCall(DMGetDimension(dm, &dim));

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  // -- Solution restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &data[level]->elem_restr_u));

  // ---------------------------------------------------------------------------
  // libCEED bases
  // ---------------------------------------------------------------------------
  // -- Solution basis
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_u, P, Q, problem_data.quadrature_mode, &data[level]->basis_u);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  CeedVectorCreate(ceed, U_loc_size, &data[level]->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &data[level]->y_ceed);

  // ---------------------------------------------------------------------------
  // Coarse Grid, Prolongation, and Restriction Operators
  // ---------------------------------------------------------------------------
  // Create the Operators that compute the prolongation and
  //   restriction between the p-multigrid levels and the coarse grid eval.
  // ---------------------------------------------------------------------------
  CeedOperatorMultigridLevelCreate(data[level + 1]->op_jacobian, fine_mult, data[level]->elem_restr_u, data[level]->basis_u, &op_jacobian,
                                   &op_prolong, &op_restrict);

  // -- Save libCEED data
  data[level]->op_jacobian     = op_jacobian;
  data[level + 1]->op_prolong  = op_prolong;
  data[level + 1]->op_restrict = op_restrict;

  PetscFunctionReturn(0);
};
