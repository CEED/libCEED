#include "../include/setup-libceed.h"

#include <stdio.h>

#include "../include/setup-fe.h"

// -----------------------------------------------------------------------------
// Destroy libCEED operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedData ceed_data, ProblemData problem_data) {
  PetscFunctionBeginUser;

  CeedVectorDestroy(&ceed_data->q_data);
  CeedVectorDestroy(&ceed_data->x_coord);
  CeedVectorDestroy(&ceed_data->x_ceed);
  CeedVectorDestroy(&ceed_data->y_ceed);
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_qdata);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u_i);
  CeedQFunctionDestroy(&ceed_data->qf_residual);
  CeedQFunctionDestroy(&ceed_data->qf_error_u);
  CeedOperatorDestroy(&ceed_data->op_residual);
  CeedOperatorDestroy(&ceed_data->op_error_u);
  if (problem_data->mixed) {
    CeedBasisDestroy(&ceed_data->basis_p);
    CeedElemRestrictionDestroy(&ceed_data->elem_restr_p);
    CeedQFunctionDestroy(&ceed_data->qf_error_p);
    CeedOperatorDestroy(&ceed_data->op_error_p);
  }
  PetscCall(PetscFree(ceed_data));

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Setup libCEED operators for a given FE space
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx, ProblemData problem_data, CeedData ceed_data, CeedVector rhs_ceed) {
  DM                 dm_coord;
  Vec                coords;
  const PetscScalar *coord_array;
  CeedQFunction      qf_setup_geo, qf_setup_rhs, qf_residual, qf_error_u;
  CeedOperator       op_setup_geo, op_setup_rhs, op_residual, op_error_u;
  CeedInt            dim, num_comp_x, num_comp_u, num_qpts, c_start, c_end, num_elem, q_data_size = problem_data->q_data_size;
  CeedInt            num_comp_p = 0;  // we change it to 1 for the mixed problem
  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  num_comp_x = dim, num_comp_u = dim;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  // CEED bases
  PetscCall(CreateBasisFromPlex(ceed, dm_coord, 0, 0, 0, 0, problem_data, &ceed_data->basis_x));
  PetscCall(CreateBasisFromPlex(ceed, dm, 0, 0, 0, 0, problem_data, &ceed_data->basis_u));

  // CEED restrictions
  CeedInt  height       = 0;  // 0 means no boundary conditions
  DMLabel  domain_label = 0;
  PetscInt value = 0, dm_field = 0;  // field 0 is for u field
  PetscCall(CreateRestrictionFromPlex(ceed, dm_coord, domain_label, value, height, dm_field, &ceed_data->elem_restr_x));
  PetscCall(CreateRestrictionFromPlex(ceed, dm, domain_label, value, height, dm_field, &ceed_data->elem_restr_u));

  if (problem_data->mixed) {
    num_comp_p += 1;  // one scalar dof for pressure field
    CeedInt Q;
    CeedBasisGetNumQuadraturePoints1D(ceed_data->basis_u, &Q);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_p, app_ctx->p_order + 1, Q, problem_data->quadrature_mode, &ceed_data->basis_p);
    PetscCall(CreateRestrictionFromPlex(ceed, dm, domain_label, value, height, 1, &ceed_data->elem_restr_p));
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  CeedBasisGetNumQuadraturePoints(ceed_data->basis_u, &num_qpts);

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, (num_comp_u + num_comp_p), (num_comp_u + num_comp_p) * num_elem * num_qpts,
                                   CEED_STRIDES_BACKEND, &ceed_data->elem_restr_u_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size, q_data_size * num_elem * num_qpts, CEED_STRIDES_BACKEND,
                                   &ceed_data->elem_restr_qdata);

  // Element coordinates vector
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coord_array));
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &ceed_data->x_coord, NULL);
  CeedVectorSetArray(ceed_data->x_coord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *)coord_array);
  PetscCall(VecRestoreArrayRead(coords, &coord_array));

  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the quadrature data q_data returns dXdx and w * detJ
  // ---------------------------------------------------------------------------
  // Create the persistent vectors that will be needed in setup and apply
  CeedVectorCreate(ceed, num_elem * num_qpts * q_data_size, &ceed_data->q_data);
  // Create the QFunction that builds the context data
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_geo, problem_data->setup_geo_loc, &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);
  // Create the operator that builds the quadrature data
  CeedOperatorCreate(ceed, qf_setup_geo, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "dx", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "qdata", ceed_data->elem_restr_qdata, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // Setup q_data
  CeedOperatorApply(op_setup_geo, ceed_data->x_coord, ceed_data->q_data, CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Setup RHS
  // ---------------------------------------------------------------------------
  CeedVector target;
  CeedVectorCreate(ceed, num_elem * num_qpts * (num_comp_u + num_comp_p), &target);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_rhs, problem_data->setup_rhs_loc, &qf_setup_rhs);
  if (!problem_data->bp4) {
    CeedQFunctionSetContext(qf_setup_rhs, problem_data->rhs_qfunction_ctx);
    CeedQFunctionContextDestroy(&problem_data->rhs_qfunction_ctx);
  }
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "true solution", num_comp_u + num_comp_p, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs_u", num_comp_u, CEED_EVAL_INTERP);
  if (problem_data->mixed) {
    CeedQFunctionAddOutput(qf_setup_rhs, "rhs_p", num_comp_p, CEED_EVAL_INTERP);
  }
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", ceed_data->elem_restr_x, ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "qdata", ceed_data->elem_restr_qdata, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_setup_rhs, "true solution", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup_rhs, "rhs_u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  if (problem_data->mixed) {
    CeedOperatorSetField(op_setup_rhs, "rhs_p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  }
  // Setup RHS and target
  CeedOperatorApply(op_setup_rhs, ceed_data->x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables: we use them in setup-solvers.c
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->x_ceed, NULL);
  CeedElemRestrictionCreateVector(ceed_data->elem_restr_u, &ceed_data->y_ceed, NULL);

  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual, problem_data->residual_loc, &qf_residual);
  if (!problem_data->bp4) {
    CeedQFunctionSetContext(qf_residual, problem_data->residual_qfunction_ctx);
    CeedQFunctionContextDestroy(&problem_data->residual_qfunction_ctx);
  }
  CeedQFunctionAddInput(qf_residual, "du", num_comp_u * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "qdata", q_data_size, CEED_EVAL_NONE);
  if (problem_data->mixed) {
    CeedQFunctionAddInput(qf_residual, "p", num_comp_p, CEED_EVAL_INTERP);
  }
  CeedQFunctionAddOutput(qf_residual, "dv", num_comp_u * dim, CEED_EVAL_GRAD);
  if (problem_data->mixed) {
    CeedQFunctionAddOutput(qf_residual, "q", num_comp_p, CEED_EVAL_INTERP);
  }

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qf_residual, NULL, NULL, &op_residual);
  CeedOperatorSetField(op_residual, "du", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "qdata", ceed_data->elem_restr_qdata, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  if (problem_data->mixed) {
    CeedOperatorSetField(op_residual, "p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  }
  CeedOperatorSetField(op_residual, "dv", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  if (problem_data->mixed) {
    CeedOperatorSetField(op_residual, "q", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
  }
  // -- Save libCEED data to apply operator in setop-solvers.c
  ceed_data->qf_residual = qf_residual;
  ceed_data->op_residual = op_residual;

  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes error.
  // ---------------------------------------------------------------------------
  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->error_u, problem_data->error_u_loc, &qf_error_u);
  CeedQFunctionAddInput(qf_error_u, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error_u, "true solution", num_comp_u + num_comp_p, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error_u, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error_u, "error u", num_comp_u, CEED_EVAL_INTERP);
  // Create the error operator
  CeedOperatorCreate(ceed, qf_error_u, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error_u);
  CeedOperatorSetField(op_error_u, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error_u, "true solution", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error_u, "qdata", ceed_data->elem_restr_qdata, CEED_BASIS_COLLOCATED, ceed_data->q_data);
  CeedOperatorSetField(op_error_u, "error u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data to apply operator in setop-solvers.c
  ceed_data->qf_error_u = qf_error_u;
  ceed_data->op_error_u = op_error_u;
  if (problem_data->mixed) {
    CeedQFunction qf_error_p;
    CeedOperator  op_error_p;
    // Create the error QFunction
    CeedQFunctionCreateInterior(ceed, 1, problem_data->error_p, problem_data->error_p_loc, &qf_error_p);
    CeedQFunctionAddInput(qf_error_p, "p", num_comp_p, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_error_p, "true solution", num_comp_u + num_comp_p, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf_error_p, "qdata", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_error_p, "error p", num_comp_p, CEED_EVAL_INTERP);
    // Create the error operator
    CeedOperatorCreate(ceed, qf_error_p, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error_p);
    CeedOperatorSetField(op_error_p, "p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_error_p, "true solution", ceed_data->elem_restr_u_i, CEED_BASIS_COLLOCATED, target);
    CeedOperatorSetField(op_error_p, "qdata", ceed_data->elem_restr_qdata, CEED_BASIS_COLLOCATED, ceed_data->q_data);
    CeedOperatorSetField(op_error_p, "error p", ceed_data->elem_restr_p, ceed_data->basis_p, CEED_VECTOR_ACTIVE);
    // -- Save libCEED data to apply operator in setop-solvers.c
    ceed_data->qf_error_p = qf_error_p;
    ceed_data->op_error_p = op_error_p;
  }

  // Cleanup
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);
  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);

  PetscFunctionReturn(0);
};
