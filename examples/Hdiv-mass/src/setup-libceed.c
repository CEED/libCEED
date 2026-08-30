#include "../include/setup-libceed.h"

#include <stdio.h>

#include "../basis/Hdiv-hex.h"
#include "../basis/Hdiv-quad.h"
#include "../basis/L2-P0.h"

// -----------------------------------------------------------------------------
// Destroy libCEED objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedData ceed_data) {
  PetscFunctionBegin;

  // Vectors
  CeedVectorDestroy(&ceed_data->x_ceed);
  CeedVectorDestroy(&ceed_data->y_ceed);
  // Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u_i);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_p);
  // Bases
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_u);
  CeedBasisDestroy(&ceed_data->basis_p);
  // QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_residual);
  CeedQFunctionDestroy(&ceed_data->qf_error);
  // Operators
  CeedOperatorDestroy(&ceed_data->op_residual);
  CeedOperatorDestroy(&ceed_data->op_error);
  PetscCall(PetscFree(ceed_data));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED on the fine grid for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data, CeedVector rhs_ceed) {
  CeedInt P = app_ctx->degree + 1;
  // Number of quadratures in 1D, q_extra is set in cl-options.c
  CeedInt            Q = P + 1 + app_ctx->q_extra;
  CeedInt            dim, num_comp_x, num_comp_u;
  DM                 dm_coord;
  Vec                coords;
  PetscInt           c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedVector         x_coord;
  CeedQFunction      qf_setup_rhs, qf_residual, qf_error;
  CeedOperator       op_setup_rhs, op_residual, op_error;

  PetscFunctionBeginUser;
  // ---------------------------------------------------------------------------
  // libCEED bases:Hdiv basis_u and Lagrange basis_x
  // ---------------------------------------------------------------------------
  dim        = problem_data->dim;
  num_comp_x = dim;
  num_comp_u = 1;  // one vector dof
  // Number of quadratures per element
  CeedInt    num_qpts = PetscPowInt(Q, dim);
  CeedInt    P_u      = dim * PetscPowInt(P, dim);  // dof per element
  CeedScalar q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar div[P_u * num_qpts], interp[dim * P_u * num_qpts], interp_p[num_qpts], *grad = NULL;

  if (dim == 2) {
    HdivBasisQuad(Q, q_ref, q_weights, interp, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp_u, P_u, num_qpts, interp, div, q_ref, q_weights, &ceed_data->basis_u);
    L2BasisP0(dim, Q, q_ref, q_weights, interp_p, problem_data->quadrature_mode);
    CeedBasisCreateH1(ceed, CEED_TOPOLOGY_QUAD, 1, 1, num_qpts, interp_p, grad, q_ref, q_weights, &ceed_data->basis_p);
  } else {
    HdivBasisHex(Q, q_ref, q_weights, interp, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_HEX, num_comp_u, P_u, num_qpts, interp, div, q_ref, q_weights, &ceed_data->basis_u);
    L2BasisP0(dim, Q, q_ref, q_weights, interp_p, problem_data->quadrature_mode);
    CeedBasisCreateH1(ceed, CEED_TOPOLOGY_HEX, 1, 1, num_qpts, interp_p, grad, q_ref, q_weights, &ceed_data->basis_p);
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
  // -- Solution restriction, Error restriction
  PetscCall(CreateRestrictionFromPlexOriented(ceed, dm, P, &ceed_data->elem_restr_u, &ceed_data->elem_restr_p));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  // -- Target restriction for MMS
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, dim, dim * num_elem * num_qpts, CEED_STRIDES_BACKEND, &ceed_data->elem_restr_u_i);
  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coordArray));
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coordArray);
  PetscCall(VecRestoreArrayRead(coords, &coordArray));

  // ---------------------------------------------------------------------------
  // Setup RHS and true solution
  // ---------------------------------------------------------------------------
  CeedVector target;
  CeedVectorCreate(ceed, num_elem * num_qpts * dim, &target);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_rhs, problem_data->setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_rhs, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_rhs, "true_soln", dim, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", dim, CEED_EVAL_INTERP);
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_rhs, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "true_soln", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup_rhs, "rhs", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Setup RHS and true solution
  CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables: we use them in setup-solvers.c
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->x_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->y_ceed, NULL);

  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual, problem_data->residual_loc, &qf_residual);
  CeedQFunctionAddInput(qf_residual, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_residual, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_residual, "v", dim, CEED_EVAL_INTERP);

  // -- Operator
  CeedOperatorCreate(ceed, qf_residual, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_residual);
  CeedOperatorSetField(op_residual, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_residual, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, x_coord);
  CeedOperatorSetField(op_residual, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "v", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_residual = qf_residual;
  ceed_data->op_residual = op_residual;

  // ---------------------------------------------------------------------------
  // Setup Error Qfunction
  // ---------------------------------------------------------------------------
  // Create the q-function that sets up the error
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_error, problem_data->setup_error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_error, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", dim, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "weight", 1, CEED_EVAL_WEIGHT);
  // CeedQFunctionAddOutput(qf_error, "error", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", 1, CEED_EVAL_INTERP);
  // Create the operator that builds the error
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, x_coord);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  // CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_e_i, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_error = qf_error;
  ceed_data->op_error = op_error;

  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedVectorDestroy(&x_coord);
  CeedVectorDestroy(&target);

  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------