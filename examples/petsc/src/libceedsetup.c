#include <stdio.h>
#include "../include/libceedsetup.h"
#include "../include/petscutils.h"

// -----------------------------------------------------------------------------
// Destroy libCEED operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data) {
  int ierr;

  CeedVectorDestroy(&data->q_data);
  CeedVectorDestroy(&data->x_ceed);
  CeedVectorDestroy(&data->y_ceed);
  CeedBasisDestroy(&data->basis_x);
  CeedBasisDestroy(&data->basis_u);
  CeedElemRestrictionDestroy(&data->elem_restr_u);
  CeedElemRestrictionDestroy(&data->elem_restr_x);
  CeedElemRestrictionDestroy(&data->elem_restr_u_i);
  CeedElemRestrictionDestroy(&data->elem_restr_qd_i);
  CeedQFunctionDestroy(&data->qf_apply);
  CeedOperatorDestroy(&data->op_apply);
  if (i > 0) {
    CeedOperatorDestroy(&data->op_prolong);
    CeedBasisDestroy(&data->basis_c_to_f);
    CeedOperatorDestroy(&data->op_restrict);
  }
  ierr = PetscFree(data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                    CeedInt topo_dim, CeedInt q_extra,
                                    PetscInt num_comp_x, PetscInt num_comp_u,
                                    PetscInt g_size, PetscInt xl_size,
                                    BPData bp_data, CeedData data,
                                    PetscBool setup_rhs, CeedVector rhs_ceed,
                                    CeedVector *target) {
  int ierr;
  DM dm_coord;
  Vec coords;
  const PetscScalar *coord_array;
  CeedBasis basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction qf_setup_geo, qf_apply;
  CeedOperator op_setup_geo, op_apply;
  CeedVector x_coord, q_data, x_ceed, y_ceed;
  CeedInt P, Q, num_qpts, c_start, c_end, num_elem,
          q_data_size = bp_data.q_data_size;
  CeedScalar R = 1,                      // radius of the sphere
             l = 1.0/PetscSqrtReal(3.0); // half edge of the inscribed cube

  // CEED bases
  P = degree + 1;
  Q = P + q_extra;
  CeedBasisCreateTensorH1Lagrange(ceed, topo_dim, num_comp_u, P, Q,
                                  bp_data.q_mode,
                                  &basis_u);
  CeedBasisCreateTensorH1Lagrange(ceed, topo_dim, num_comp_x, 2, Q,
                                  bp_data.q_mode,
                                  &basis_x);
  CeedBasisGetNumQuadraturePoints(basis_u, &num_qpts);

  // CEED restrictions
  ierr = DMSetCoordinateDim(dm, topo_dim); CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, 0, 0, 0, &elem_restr_x);
  CHKERRQ(ierr);
  ierr = CreateRestrictionFromPlex(ceed, dm, 0, 0, 0, &elem_restr_u);
  CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, num_comp_u,
                                   num_comp_u*num_elem*num_qpts,
                                   CEED_STRIDES_BACKEND, &elem_restr_u_i);
  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, q_data_size,
                                   q_data_size*num_elem*num_qpts,
                                   CEED_STRIDES_BACKEND, &elem_restr_qd_i);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coord_array); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coord_array);
  ierr = VecRestoreArrayRead(coords, &coord_array);

  // Create the persistent vectors that will be needed in setup and apply
  CeedVectorCreate(ceed, q_data_size*num_elem*num_qpts, &q_data);
  CeedVectorCreate(ceed, xl_size, &x_ceed);
  CeedVectorCreate(ceed, xl_size, &y_ceed);

  // Create the QFunction that builds the context data
  CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_geo, bp_data.setup_geo_loc,
                              &qf_setup_geo);
  CeedQFunctionAddInput(qf_setup_geo, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_geo, "dx", num_comp_x*topo_dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup_geo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup_geo, "qdata", q_data_size, CEED_EVAL_NONE);

  // Create the operator that builds the quadrature data
  CeedOperatorCreate(ceed, qf_setup_geo, NULL, NULL, &op_setup_geo);
  CeedOperatorSetField(op_setup_geo, "x", elem_restr_x, basis_x,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "dx", elem_restr_x, basis_x,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_geo, "weight", CEED_ELEMRESTRICTION_NONE, basis_x,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_geo, "qdata", elem_restr_qd_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Setup q_data
  CeedOperatorApply(op_setup_geo, x_coord, q_data, CEED_REQUEST_IMMEDIATE);

  // Set up PDE operator
  CeedInt in_scale = bp_data.in_mode == CEED_EVAL_GRAD ? topo_dim : 1;
  CeedInt out_scale = bp_data.out_mode == CEED_EVAL_GRAD ? topo_dim : 1;
  CeedQFunctionCreateInterior(ceed, 1, bp_data.apply, bp_data.apply_loc,
                              &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", num_comp_u*in_scale, bp_data.in_mode);
  CeedQFunctionAddInput(qf_apply, "qdata", q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", num_comp_u*out_scale, bp_data.out_mode);

  // Create the mass or diff operator
  CeedOperatorCreate(ceed, qf_apply, NULL, NULL, &op_apply);
  CeedOperatorSetField(op_apply, "u", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", elem_restr_qd_i, CEED_BASIS_COLLOCATED,
                       q_data);
  CeedOperatorSetField(op_apply, "v", elem_restr_u, basis_u, CEED_VECTOR_ACTIVE);

  // Set up RHS if needed
  if (setup_rhs) {
    CeedQFunction qf_setup_rhs;
    CeedOperator op_setup_rhs;
    CeedVectorCreate(ceed, num_elem*num_qpts*num_comp_u, target);

    // Create the q-function that sets up the RHS and true solution
    CeedQFunctionCreateInterior(ceed, 1, bp_data.setup_rhs, bp_data.setup_rhs_loc,
                                &qf_setup_rhs);
    CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(qf_setup_rhs, "qdata", q_data_size, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "true solution", num_comp_u,
                           CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf_setup_rhs, "rhs", num_comp_u, CEED_EVAL_INTERP);

    // Create the operator that builds the RHS and true solution
    CeedOperatorCreate(ceed, qf_setup_rhs, NULL, NULL, &op_setup_rhs);
    CeedOperatorSetField(op_setup_rhs, "x", elem_restr_x, basis_x,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_setup_rhs, "qdata", elem_restr_qd_i,
                         CEED_BASIS_COLLOCATED, q_data);
    CeedOperatorSetField(op_setup_rhs, "true solution", elem_restr_u_i,
                         CEED_BASIS_COLLOCATED, *target);
    CeedOperatorSetField(op_setup_rhs, "rhs", elem_restr_u, basis_u,
                         CEED_VECTOR_ACTIVE);

    // Set up the libCEED context
    CeedQFunctionContext ctx_rhs_setup;
    CeedQFunctionContextCreate(ceed, &ctx_rhs_setup);
    CeedScalar rhs_setup_data[2] = {R, l};
    CeedQFunctionContextSetData(ctx_rhs_setup, CEED_MEM_HOST, CEED_COPY_VALUES,
                                sizeof rhs_setup_data, &rhs_setup_data);
    CeedQFunctionSetContext(qf_setup_rhs, ctx_rhs_setup);
    CeedQFunctionContextDestroy(&ctx_rhs_setup);

    // Setup RHS and target
    CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

    // Cleanup
    CeedQFunctionDestroy(&qf_setup_rhs);
    CeedOperatorDestroy(&op_setup_rhs);
  }

  // Cleanup
  CeedQFunctionDestroy(&qf_setup_geo);
  CeedOperatorDestroy(&op_setup_geo);
  CeedVectorDestroy(&x_coord);

  // Save libCEED data required for level
  data->basis_x = basis_x; data->basis_u = basis_u;
  data->elem_restr_x = elem_restr_x;
  data->elem_restr_u = elem_restr_u;
  data->elem_restr_u_i = elem_restr_u_i;
  data->elem_restr_qd_i = elem_restr_qd_i;
  data->qf_apply = qf_apply;
  data->op_apply = op_apply;
  data->q_data = q_data;
  data->x_ceed = x_ceed;
  data->y_ceed = y_ceed;

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Setup libCEED level transfer operator objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt num_levels,
                                      CeedInt num_comp_u, CeedData *data,
                                      CeedInt *level_degrees,
                                      CeedQFunction qf_restrict, CeedQFunction qf_prolong) {
  // Return early if num_levels=1
  if (num_levels == 1)
    PetscFunctionReturn(0);

  // Set up each level
  for (CeedInt i=1; i<num_levels; i++) {
    // P coarse and P fine
    CeedInt Pc = level_degrees[i-1] + 1;
    CeedInt Pf = level_degrees[i] + 1;

    // Restriction - Fine to corse
    CeedBasis basis_c_to_f;
    CeedOperator op_restrict;

    // Basis
    CeedBasisCreateTensorH1Lagrange(ceed, 3, num_comp_u, Pc, Pf,
                                    CEED_GAUSS_LOBATTO, &basis_c_to_f);

    // Create the restriction operator
    CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_restrict);
    CeedOperatorSetField(op_restrict, "input", data[i]->elem_restr_u,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_restrict, "output", data[i-1]->elem_restr_u,
                         basis_c_to_f, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->basis_c_to_f = basis_c_to_f;
    data[i]->op_restrict = op_restrict;

    // Interpolation - Corse to fine
    CeedOperator op_prolong;

    // Create the prolongation operator
    CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &op_prolong);
    CeedOperatorSetField(op_prolong, "input", data[i-1]->elem_restr_u,
                         basis_c_to_f, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_prolong, "output", data[i]->elem_restr_u,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // Save libCEED data required for level
    data[i]->op_prolong = op_prolong;
  }

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
