#include "../include/setup-libceed.h"

#include <stdio.h>

#include "../basis/Hdiv-hex.h"
#include "../basis/Hdiv-quad.h"
#include "../basis/L2-P0.h"
#include "../include/petsc-macros.h"
#include "../include/setup-boundary.h"
#include "ceed/ceed.h"

// -----------------------------------------------------------------------------
// Destroy libCEED objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedData ceed_data, ProblemData problem_data) {
  PetscFunctionBegin;

  // Vectors
  CeedVectorDestroy(&ceed_data->x_ceed);
  CeedVectorDestroy(&ceed_data->y_ceed);
  CeedVectorDestroy(&ceed_data->x_t_ceed);
  CeedVectorDestroy(&ceed_data->x_coord);
  // Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_U_i);  // U = [p,u]
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_p);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_p_i);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u0);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_p0);
  // Bases
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_u);
  CeedBasisDestroy(&ceed_data->basis_p);
  CeedBasisDestroy(&ceed_data->basis_u_face);
  if (problem_data->has_ts) {
    CeedVectorDestroy(&ceed_data->u0_ceed);
    CeedVectorDestroy(&ceed_data->v0_ceed);
    CeedVectorDestroy(&ceed_data->p0_ceed);
    CeedVectorDestroy(&ceed_data->q0_ceed);
    CeedVectorDestroy(&ceed_data->rhs_u0_ceed);
    CeedVectorDestroy(&ceed_data->rhs_p0_ceed);
    CeedQFunctionDestroy(&ceed_data->qf_rhs_u0);
    CeedOperatorDestroy(&ceed_data->op_rhs_u0);
    CeedQFunctionDestroy(&ceed_data->qf_ics_u);
    CeedOperatorDestroy(&ceed_data->op_ics_u);
    CeedQFunctionDestroy(&ceed_data->qf_rhs_p0);
    CeedOperatorDestroy(&ceed_data->op_rhs_p0);
    CeedQFunctionDestroy(&ceed_data->qf_ics_p);
    CeedOperatorDestroy(&ceed_data->op_ics_p);
  }
  // QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_residual);
  CeedQFunctionDestroy(&ceed_data->qf_error);
  // Operators
  CeedOperatorDestroy(&ceed_data->op_residual);
  CeedOperatorDestroy(&ceed_data->op_error);
  if (!problem_data->has_ts) {
    // QFunctions
    CeedQFunctionDestroy(&ceed_data->qf_jacobian);
    // Operators
    CeedOperatorDestroy(&ceed_data->op_jacobian);
  }

  // data for post-processing
  if (problem_data->view_solution) {
    CeedVectorDestroy(&ceed_data->up_ceed);
    CeedVectorDestroy(&ceed_data->vp_ceed);
    CeedVectorDestroy(&ceed_data->u_ceed);
    CeedElemRestrictionDestroy(&ceed_data->elem_restr_u_H1);
    CeedQFunctionDestroy(&ceed_data->qf_rhs_H1);
    CeedOperatorDestroy(&ceed_data->op_rhs_H1);
    CeedQFunctionDestroy(&ceed_data->qf_post_mass);
    CeedOperatorDestroy(&ceed_data->op_post_mass);
  }
  PetscCall(PetscFree(ceed_data));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED on the fine grid for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, DM dm_u0, DM dm_p0, DM dm_H1, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data) {
  CeedInt P = app_ctx->degree + 1;
  // Number of quadratures in 1D, q_extra is set in cl-options.c
  CeedInt            Q = P + 1 + app_ctx->q_extra;
  CeedInt            dim, num_comp_x, num_comp_u, num_comp_p;
  DM                 dm_coord;
  Vec                coords;
  PetscInt           c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedQFunction      qf_true, qf_residual, qf_jacobian, qf_error;
  CeedOperator       op_true, op_residual, op_jacobian, op_error;

  PetscFunctionBeginUser;
  // ---------------------------------------------------------------------------
  // libCEED bases:Hdiv basis_u and Lagrange basis_x
  // ---------------------------------------------------------------------------
  dim        = problem_data->dim;
  num_comp_x = dim;
  num_comp_u = 1;  // one vector dof
  num_comp_p = 1;  // one scalar dof
  // Number of quadratures per element
  CeedInt num_qpts = PetscPowInt(Q, dim);
  // Pressure and velocity dof per element
  CeedInt    P_p = 1, P_u = dim * PetscPowInt(P, dim);
  CeedScalar q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar div[P_u * num_qpts], interp_u[dim * P_u * num_qpts], interp_p[P_p * num_qpts], *grad = NULL;
  if (dim == 2) {
    HdivBasisQuad(Q, q_ref, q_weights, interp_u, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp_u, P_u, num_qpts, interp_u, div, q_ref, q_weights, &ceed_data->basis_u);
    L2BasisP0(dim, Q, q_ref, q_weights, interp_p, problem_data->quadrature_mode);
    CeedBasisCreateH1(ceed, CEED_TOPOLOGY_QUAD, num_comp_p, 1, num_qpts, interp_p, grad, q_ref, q_weights, &ceed_data->basis_p);
    HdivBasisQuad(Q, q_ref, q_weights, interp_u, div, CEED_GAUSS_LOBATTO);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp_u, P_u, num_qpts, interp_u, div, q_ref, q_weights, &ceed_data->basis_u_face);
  } else {
    HdivBasisHex(Q, q_ref, q_weights, interp_u, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_HEX, num_comp_u, P_u, num_qpts, interp_u, div, q_ref, q_weights, &ceed_data->basis_u);
    L2BasisP0(dim, Q, q_ref, q_weights, interp_p, problem_data->quadrature_mode);
    CeedBasisCreateH1(ceed, CEED_TOPOLOGY_HEX, num_comp_p, 1, num_qpts, interp_p, grad, q_ref, q_weights, &ceed_data->basis_p);
    HdivBasisHex(Q, q_ref, q_weights, interp_u, div, CEED_GAUSS_LOBATTO);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_HEX, num_comp_u, P_u, num_qpts, interp_u, div, q_ref, q_weights, &ceed_data->basis_u_face);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q, problem_data->quadrature_mode, &ceed_data->basis_x);

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL));
  CeedInt  height       = 0;  // 0 means no boundary conditions
  DMLabel  domain_label = 0;
  PetscInt value        = 0;
  // -- Coordinate restriction
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label, value, &ceed_data->elem_restr_x));
  // -- Solution restriction
  PetscCall(CreateRestrictionFromPlexOriented(ceed, dm, dm_u0, dm_p0, P, &ceed_data->elem_restr_u, &ceed_data->elem_restr_p,
                                              &ceed_data->elem_restr_u0, &ceed_data->elem_restr_p0));
  // -- Geometric ceed_data restriction
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem            = c_end - c_start;
  ceed_data->num_elem = num_elem;
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, (dim + 1), (dim + 1) * num_elem * num_qpts, CEED_STRIDES_BACKEND,
                                   &ceed_data->elem_restr_U_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, 1, 1 * num_elem * num_qpts, CEED_STRIDES_BACKEND, &ceed_data->elem_restr_p_i);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coordArray));
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord, NULL);
  CeedVectorSetArray(ceed_data->x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coordArray);
  PetscCall(VecRestoreArrayRead(coords, &coordArray));

  // ---------------------------------------------------------------------------
  // Setup true solution for [p,u]
  // ---------------------------------------------------------------------------
  CeedVector true_vec, true_force;
  CeedVectorCreate(ceed, num_elem * num_qpts * (dim + 1), &true_vec);
  CeedVectorCreate(ceed, num_elem * num_qpts * 1, &true_force);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->true_solution, problem_data->true_solution_loc, &qf_true);
  CeedQFunctionSetContext(qf_true, problem_data->true_qfunction_ctx);
  CeedQFunctionContextDestroy(&problem_data->true_qfunction_ctx);
  CeedQFunctionAddInput(qf_true, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_true, "true force", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_true, "true solution", dim + 1, CEED_EVAL_NONE);
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_true, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_true);
  if (problem_data->has_ts) {
    double final_time = app_ctx->t_final;
    CeedOperatorGetContextFieldLabel(op_true, "final_time", &app_ctx->ctx_residual_ut->final_time_label);
    CeedOperatorSetContextDouble(op_true, app_ctx->ctx_residual_ut->final_time_label, &final_time);
  }
  CeedOperatorSetField(op_true, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op_true, "true force", ceed_data->elem_restr_p_i, CEED_BASIS_COLLOCATED, true_force);
  CeedOperatorSetField(op_true, "true solution", ceed_data->elem_restr_U_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // Setup true solution
  CeedOperatorApply(op_true, ceed_data->x_coord, true_vec, CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Setup initial conditions
  // ---------------------------------------------------------------------------
  if (problem_data->has_ts) {
    // ---------------------------------------------------------------------------
    // Setup RHS for u field
    // ---------------------------------------------------------------------------
    CeedQFunction qf_rhs_u0;
    CeedOperator  op_rhs_u0;
    // Create the q-function that sets up the RHS
    CeedQFunctionCreateInterior(ceed, 1, problem_data->rhs_u0, problem_data->rhs_u0_loc, &qf_rhs_u0);
    CeedQFunctionSetContext(qf_rhs_u0, problem_data->rhs_u0_qfunction_ctx);
    CeedQFunctionContextDestroy(&problem_data->rhs_u0_qfunction_ctx);
    CeedQFunctionAddInput(qf_rhs_u0, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_rhs_u0, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhs_u0, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddOutput(qf_rhs_u0, "rhs_u0", dim, CEED_EVAL_INTERP);
    // Create the operator that builds the RHS
    CeedOperatorCreate(ceed, qf_rhs_u0, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_rhs_u0);
    CeedOperatorSetField(op_rhs_u0, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_rhs_u0, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs_u0, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs_u0, "rhs_u0", ceed_data->elem_restr_u0, ceed_data->basis_u, CEED_VECTOR_ACTIVE);

    // -- Save libCEED data to apply operator in setup-ts.c
    ceed_data->qf_rhs_u0 = qf_rhs_u0;
    ceed_data->op_rhs_u0 = op_rhs_u0;
    // ---------------------------------------------------------------------------
    // Setup qfunction for initial conditions u0
    // ---------------------------------------------------------------------------
    CeedQFunction qf_ics_u;
    CeedOperator  op_ics_u;
    CeedQFunctionCreateInterior(ceed, 1, problem_data->ics_u, problem_data->ics_u_loc, &qf_ics_u);
    CeedQFunctionAddInput(qf_ics_u, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_ics_u, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ics_u, "u", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ics_u, "v", dim, CEED_EVAL_INTERP);
    // Create the operator that builds the initial conditions
    CeedOperatorCreate(ceed, qf_ics_u, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_ics_u);
    CeedOperatorSetField(op_ics_u, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_ics_u, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_ics_u, "u", ceed_data->elem_restr_u0, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_ics_u, "v", ceed_data->elem_restr_u0, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in setup-ts.c
    ceed_data->qf_ics_u = qf_ics_u;
    ceed_data->op_ics_u = op_ics_u;
    // -- Operator action variables: we use them in setup-ts.c
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_u0, &ceed_data->u0_ceed, NULL);
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_u0, &ceed_data->v0_ceed, NULL);
    // ---------------------------------------------------------------------------
    // Setup RHS for p field
    // ---------------------------------------------------------------------------
    CeedQFunction qf_rhs_p0;
    CeedOperator  op_rhs_p0;
    // Create the q-function that sets up the RHS
    CeedQFunctionCreateInterior(ceed, 1, problem_data->rhs_p0, problem_data->rhs_p0_loc, &qf_rhs_p0);
    CeedQFunctionAddInput(qf_rhs_p0, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_rhs_p0, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_rhs_p0, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddOutput(qf_rhs_p0, "rhs_p0", 1, CEED_EVAL_INTERP);
    // Create the operator that builds the RHS
    CeedOperatorCreate(ceed, qf_rhs_p0, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_rhs_p0);
    CeedOperatorSetField(op_rhs_p0, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_rhs_p0, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs_p0, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs_p0, "rhs_p0", ceed_data->elem_restr_p0, ceed_data->basis_p, CEED_VECTOR_ACTIVE);

    // -- Save libCEED data to apply operator in setup-ts.c
    ceed_data->qf_rhs_p0 = qf_rhs_p0;
    ceed_data->op_rhs_p0 = op_rhs_p0;
    // ---------------------------------------------------------------------------
    // Setup qfunction for initial conditions p0
    // ---------------------------------------------------------------------------
    CeedQFunction qf_ics_p;
    CeedOperator  op_ics_p;
    CeedQFunctionCreateInterior(ceed, 1, problem_data->ics_p, problem_data->ics_p_loc, &qf_ics_p);
    CeedQFunctionAddInput(qf_ics_p, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_ics_p, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_ics_p, "p", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_ics_p, "q", 1, CEED_EVAL_INTERP);
    // Create the operator that builds the initial conditions
    CeedOperatorCreate(ceed, qf_ics_p, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_ics_p);
    CeedOperatorSetField(op_ics_p, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_ics_p, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_ics_p, "p", ceed_data->elem_restr_p0, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_ics_p, "q", ceed_data->elem_restr_p0, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in setup-ts.c
    ceed_data->qf_ics_p = qf_ics_p;
    ceed_data->op_ics_p = op_ics_p;
    // -- Operator action variables: we use them in setup-ts.c
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_p0, &ceed_data->p0_ceed, NULL);
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_p0, &ceed_data->q0_ceed, NULL);
  }

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables: we use them in setup-solvers.c/setup-ts.c
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->x_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->y_ceed, NULL);
  // -- Operator action variables: we use them in setup-ts.c
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->x_t_ceed, NULL);
  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual, problem_data->residual_loc, &qf_residual);
  CeedQFunctionSetContext(qf_residual, problem_data->residual_qfunction_ctx);
  CeedQFunctionContextDestroy(&problem_data->residual_qfunction_ctx);
  CeedQFunctionAddInput(qf_residual, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_residual, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_residual, "div_u", 1, CEED_EVAL_DIV);
  CeedQFunctionAddInput(qf_residual, "p", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_residual, "true force", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_residual, "x", num_comp_x, CEED_EVAL_INTERP);
  if (problem_data->has_ts) {
    CeedQFunctionAddInput(qf_residual, "p_t", 1, CEED_EVAL_INTERP);
  }
  CeedQFunctionAddOutput(qf_residual, "v", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_residual, "div_v", 1, CEED_EVAL_DIV);
  CeedQFunctionAddOutput(qf_residual, "q", 1, CEED_EVAL_INTERP);

  // -- Operator
  CeedOperatorCreate(ceed, qf_residual, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_residual);
  if (problem_data->has_ts) {
    // double t = app_ctx->ctx_residual_ut->t;
    CeedOperatorGetContextFieldLabel(op_residual, "time", &app_ctx->ctx_residual_ut->solution_time_label);
    // CeedOperatorContextGetFieldLabel(op_residual, "time_step",
    //                                  &app_ctx->ctx_residual_ut->timestep_label);
    // CeedOperatorContextSetDouble(op_residual,
    //                              app_ctx->ctx_residual_ut->solution_time_label, &t);
  }
  CeedOperatorSetField(op_residual, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_residual, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op_residual, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "div_u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "true force", ceed_data->elem_restr_p_i, CEED_BASIS_COLLOCATED, true_force);
  CeedOperatorSetField(op_residual, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  if (problem_data->has_ts) {
    CeedOperatorSetField(op_residual, "p_t", ceed_data->elem_restr_p, ceed_data->basis_p, ceed_data->x_t_ceed);
  }
  CeedOperatorSetField(op_residual, "v", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "div_v", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "q", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_residual = qf_residual;
  ceed_data->op_residual = op_residual;
  if (!problem_data->has_ts) {
    // ---------------------------------------------------------------------------
    // Add Pressure boundary condition. See setup-boundary.c
    // ---------------------------------------------------------------------------
    // DMAddBoundariesPressure(ceed, ceed_data, app_ctx, problem_data, dm);

    // Local jacobian evaluator
    // ---------------------------------------------------------------------------
    // Create the QFunction and Operator that computes the jacobian of the PDE.
    // ---------------------------------------------------------------------------
    // -- QFunction
    CeedQFunctionCreateInterior(ceed, 1, problem_data->jacobian, problem_data->jacobian_loc, &qf_jacobian);
    CeedQFunctionSetContext(qf_jacobian, problem_data->jacobian_qfunction_ctx);
    CeedQFunctionContextDestroy(&problem_data->jacobian_qfunction_ctx);
    CeedQFunctionAddInput(qf_jacobian, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_jacobian, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_jacobian, "du", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_jacobian, "div_du", 1, CEED_EVAL_DIV);
    CeedQFunctionAddInput(qf_jacobian, "dp", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_jacobian, "x", num_comp_x, CEED_EVAL_INTERP);
    // CeedQFunctionAddInput(qf_jacobian, "u", dim, CEED_EVAL_INTERP);
    // CeedQFunctionAddInput(qf_jacobian, "p", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_jacobian, "dv", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_jacobian, "div_dv", 1, CEED_EVAL_DIV);
    CeedQFunctionAddOutput(qf_jacobian, "dq", 1, CEED_EVAL_INTERP);
    // -- Operator
    CeedOperatorCreate(ceed, qf_jacobian, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_jacobian);
    CeedOperatorSetField(op_jacobian, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_jacobian, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_jacobian, "du", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "div_du", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "dp", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "x", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    // CeedOperatorSetField(op_jacobian, "u", ceed_data->elem_restr_u,
    //                      ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    // CeedOperatorSetField(op_jacobian, "p", ceed_data->elem_restr_p,
    //                      ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "dv", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "div_dv", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_jacobian, "dq", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in matops.c
    ceed_data->qf_jacobian = qf_jacobian;
    ceed_data->op_jacobian = op_jacobian;
  }
  // ---------------------------------------------------------------------------
  // Setup Error Qfunction
  // ---------------------------------------------------------------------------
  // Create the q-function that sets up the error
  CeedQFunctionCreateInterior(ceed, 1, problem_data->error, problem_data->error_loc, &qf_error);
  CeedQFunctionSetContext(qf_error, problem_data->error_qfunction_ctx);
  CeedQFunctionContextDestroy(&problem_data->error_qfunction_ctx);
  CeedQFunctionAddInput(qf_error, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_error, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_error, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "p", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true solution", dim + 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", dim + 1, CEED_EVAL_NONE);
  // Create the operator that builds the error
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_error, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true solution", ceed_data->elem_restr_U_i, CEED_BASIS_COLLOCATED, true_vec);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_U_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_error = qf_error;
  ceed_data->op_error = op_error;

  if (app_ctx->view_solution) {
    // -- Post processing
    PetscCall(CreateRestrictionFromPlex(ceed, dm_H1, height, domain_label, value, &ceed_data->elem_restr_u_H1));
    // ---------------------------------------------------------------------------
    // Setup RHS for post processing
    // ---------------------------------------------------------------------------
    // -- Operator action variables: we use them in post-processing.c
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->u_ceed, NULL);
    CeedQFunction qf_rhs_H1;
    CeedOperator  op_rhs_H1;
    // Create the q-function that sets up the RHS
    CeedQFunctionCreateInterior(ceed, 1, problem_data->post_rhs, problem_data->post_rhs_loc, &qf_rhs_H1);
    CeedQFunctionAddInput(qf_rhs_H1, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_rhs_H1, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_rhs_H1, "u_post", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_rhs_H1, "rhs_post", dim, CEED_EVAL_INTERP);
    // Create the operator that builds the RHS
    CeedOperatorCreate(ceed, qf_rhs_H1, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_rhs_H1);
    CeedOperatorSetField(op_rhs_H1, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_rhs_H1, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_rhs_H1, "u_post", ceed_data->elem_restr_u, ceed_data->basis_u, ceed_data->u_ceed);
    CeedOperatorSetField(op_rhs_H1, "rhs_post", ceed_data->elem_restr_u_H1, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in post-processing.c
    ceed_data->qf_rhs_H1 = qf_rhs_H1;
    ceed_data->op_rhs_H1 = op_rhs_H1;
    // ---------------------------------------------------------------------------
    // Setup qfunction for initial conditions u0
    // ---------------------------------------------------------------------------
    CeedQFunction qf_post_mass;
    CeedOperator  op_post_mass;
    CeedQFunctionCreateInterior(ceed, 1, problem_data->post_mass, problem_data->post_mass_loc, &qf_post_mass);
    CeedQFunctionAddInput(qf_post_mass, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddInput(qf_post_mass, "dx", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_post_mass, "u", dim, CEED_EVAL_INTERP);
    CeedQFunctionAddOutput(qf_post_mass, "v", dim, CEED_EVAL_INTERP);
    // Create the operator that builds the initial conditions
    CeedOperatorCreate(ceed, qf_post_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_post_mass);
    CeedOperatorSetField(op_post_mass, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
    CeedOperatorSetField(op_post_mass, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, ceed_data->x_coord);
    CeedOperatorSetField(op_post_mass, "u", ceed_data->elem_restr_u_H1, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_post_mass, "v", ceed_data->elem_restr_u_H1, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in post-processing.c
    ceed_data->qf_post_mass = qf_post_mass;
    ceed_data->op_post_mass = op_post_mass;
    // -- Operator action variables: we use them in post-processing.c
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_u_H1, &ceed_data->up_ceed, NULL);
    CeedElemRestrictionCreateVector(ceed_data->elem_restr_u_H1, &ceed_data->vp_ceed, NULL);
  }
  // -- Cleanup
  CeedVectorDestroy(&true_vec);
  CeedVectorDestroy(&true_force);
  CeedQFunctionDestroy(&qf_true);
  CeedOperatorDestroy(&op_true);

  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------