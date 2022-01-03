#include "../include/setup-libceed.h"
#include "../include/petsc-macros.h"
#include "../basis/Hdiv-quad.h"
#include "../basis/Hdiv-hex.h"

// -----------------------------------------------------------------------------
// Convert PETSc MemType to libCEED MemType
// -----------------------------------------------------------------------------
CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}
// -----------------------------------------------------------------------------
// Destroy libCEED objects
// -----------------------------------------------------------------------------
PetscErrorCode CeedDataDestroy(CeedData ceed_data) {
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Vectors
  CeedVectorDestroy(&ceed_data->x_ceed);
  CeedVectorDestroy(&ceed_data->y_ceed);
  // Restrictions
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_x);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u);
  CeedElemRestrictionDestroy(&ceed_data->elem_restr_u_i);
  // Bases
  CeedBasisDestroy(&ceed_data->basis_x);
  CeedBasisDestroy(&ceed_data->basis_u);
  // QFunctions
  CeedQFunctionDestroy(&ceed_data->qf_residual);
  CeedQFunctionDestroy(&ceed_data->qf_error);
  // Operators
  CeedOperatorDestroy(&ceed_data->op_residual);
  CeedOperatorDestroy(&ceed_data->op_error);
  ierr = PetscFree(ceed_data); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
// -----------------------------------------------------------------------------
PetscInt Involute(PetscInt i) {
  return i >= 0 ? i : -(i + 1);
};

// -----------------------------------------------------------------------------
// Get CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height,
    DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr) {
  PetscInt num_elem, elem_size, num_dof, num_comp, *elem_restr_offsets;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = DMPlexGetLocalOffsets(dm, domain_label, value, height, 0, &num_elem,
                               &elem_size, &num_comp, &num_dof, &elem_restr_offsets);
  CHKERRQ(ierr);

  CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp,
                            1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                            elem_restr_offsets, elem_restr);
  ierr = PetscFree(elem_restr_offsets); CHKERRQ(ierr);

  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Get Oriented CEED restriction data from DMPlex
// -----------------------------------------------------------------------------
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm,
    CeedInt P, CeedElemRestriction *elem_restr_oriented) {
  PetscSection section;
  PetscInt p, num_elem, num_dof, *restr_indices, elem_offset, num_fields,
           dim, c_start, c_end;
  Vec U_loc;
  PetscErrorCode ierr;
  const PetscInt *ornt; // this is for orientation of dof
  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim); CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &num_fields); CHKERRQ(ierr);
  PetscInt num_comp[num_fields], field_offsets[num_fields+1];
  field_offsets[0] = 0;
  for (PetscInt f = 0; f < num_fields; f++) {
    ierr = PetscSectionGetFieldComponents(section, f, &num_comp[f]); CHKERRQ(ierr);
    field_offsets[f+1] = field_offsets[f] + num_comp[f];
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;
  ierr = PetscMalloc1(num_elem*dim*PetscPowInt(P, dim),
                      &restr_indices); CHKERRQ(ierr);
  bool *orient_indices; // to flip the dof
  ierr = PetscMalloc1(num_elem*dim*PetscPowInt(P, dim), &orient_indices);
  CHKERRQ(ierr);
  for (p = 0, elem_offset = 0; p < num_elem; p++) {
    PetscInt num_indices, *indices, faces_per_elem, dofs_per_face;
    ierr = DMPlexGetClosureIndices(dm, section, section, p, PETSC_TRUE,
                                   &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);

    ierr = DMPlexGetConeOrientation(dm, p, &ornt); CHKERRQ(ierr);
    // Get number of faces per element
    ierr = DMPlexGetConeSize(dm, p, &faces_per_elem); CHKERRQ(ierr);
    dofs_per_face = faces_per_elem - 2;
    for (PetscInt f = 0; f < faces_per_elem; f++) {
      for (PetscInt i = 0; i < dofs_per_face; i++) {
        PetscInt ii = dofs_per_face*f + i;
        // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
        PetscInt loc = Involute(indices[ii*num_comp[0]]);
        restr_indices[elem_offset] = loc;
        // Set orientation
        orient_indices[elem_offset] = ornt[f] < 0;
        elem_offset++;
      }
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, p, PETSC_TRUE,
                                       &num_indices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  //if (elem_offset != num_elem*dim*PetscPowInt(P, dim))
  //  SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_LIB,
  //           "ElemRestriction of size (%" PetscInt_FMT ",%" PetscInt_FMT ")
  //            initialized %" PetscInt_FMT "nodes", num_elem,
  //            dim*PetscPowInt(P, dim),elem_offset);
  ierr = DMGetLocalVector(dm, &U_loc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(U_loc, &num_dof); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &U_loc); CHKERRQ(ierr);
  // dof per element in Hdiv is dim*P^dim, for linear element P=2
  CeedElemRestrictionCreateOriented(ceed, num_elem, dim*PetscPowInt(P, dim),
                                    field_offsets[num_fields],
                                    1, num_dof, CEED_MEM_HOST, CEED_COPY_VALUES,
                                    restr_indices, orient_indices,
                                    elem_restr_oriented);
  ierr = PetscFree(restr_indices); CHKERRQ(ierr);
  ierr = PetscFree(orient_indices); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

// -----------------------------------------------------------------------------
// Set up libCEED on the fine grid for a given degree
// -----------------------------------------------------------------------------
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx,
                            ProblemData *problem_data, PetscInt U_g_size,
                            PetscInt U_loc_size, CeedData ceed_data,
                            CeedVector rhs_ceed, CeedVector *target) {
  int           ierr;
  CeedInt       P = app_ctx->degree + 1;
  // Number of quadratures in 1D, q_extra is set in cl-options.c
  CeedInt       Q = P + 1 + app_ctx->q_extra;
  CeedInt       dim, num_comp_x, num_comp_u;
  //CeedInt       elem_node = problem_data->elem_node;
  DM            dm_coord;
  Vec           coords;
  PetscInt      c_start, c_end, num_elem;
  const PetscScalar *coordArray;
  CeedVector    x_coord;
  CeedQFunction qf_setup_rhs, qf_residual, qf_error;
  CeedOperator  op_setup_rhs, op_residual, op_error;

  PetscFunctionBeginUser;
  // ---------------------------------------------------------------------------
  // libCEED bases:Hdiv basis_u and Lagrange basis_x
  // ---------------------------------------------------------------------------
  dim = problem_data->dim;
  num_comp_x = dim;
  num_comp_u = 1;   // one vector dof
  // Number of quadratures per element
  CeedInt       num_qpts = PetscPowInt(Q, dim);
  CeedInt       P_u = dim*PetscPowInt(P, dim); // dof per element
  CeedScalar    q_ref[dim*num_qpts], q_weights[num_qpts];
  CeedScalar    div[P_u*num_qpts], interp[dim*P_u*num_qpts];

  if (dim == 2) {
    HdivBasisQuad(Q, q_ref, q_weights, interp, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp_u, P_u, num_qpts,
                        interp, div, q_ref, q_weights, &ceed_data->basis_u);
  } else {
    HdivBasisHex(Q, q_ref, q_weights, interp, div, problem_data->quadrature_mode);
    CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_HEX, num_comp_u, P_u, num_qpts,
                        interp, div, q_ref, q_weights, &ceed_data->basis_u);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp_x, 2, Q,
                                  problem_data->quadrature_mode, &ceed_data->basis_x);

  // ---------------------------------------------------------------------------
  // libCEED restrictions
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinateDM(dm, &dm_coord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm_coord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);
  CeedInt height = 0; // 0 means no boundary conditions
  DMLabel domain_label = 0;
  PetscInt value = 0;
  // -- Coordinate restriction
  ierr = CreateRestrictionFromPlex(ceed, dm_coord, height, domain_label,
                                   value, &ceed_data->elem_restr_x); CHKERRQ(ierr);
  // -- Solution and projected true solution restriction
  ierr = CreateRestrictionFromPlexOriented(ceed, dm,
         P, &ceed_data->elem_restr_u);
  CHKERRQ(ierr);
  // -- Geometric ceed_data restriction
  ierr = DMPlexGetHeightStratum(dm, 0, &c_start, &c_end); CHKERRQ(ierr);
  num_elem = c_end - c_start;

  CeedElemRestrictionCreateStrided(ceed, num_elem, num_qpts, dim,
                                   dim*num_elem*num_qpts,
                                   CEED_STRIDES_BACKEND, &ceed_data->elem_restr_u_i);

  // ---------------------------------------------------------------------------
  // Element coordinates
  // ---------------------------------------------------------------------------
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);

  CeedElemRestrictionCreateVector(ceed_data->elem_restr_x, &x_coord, NULL);
  CeedVectorSetArray(x_coord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Setup RHS and true solution
  // ---------------------------------------------------------------------------
  CeedVectorCreate(ceed, num_elem*num_qpts*dim, target);
  // Create the q-function that sets up the RHS and true solution
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_rhs,
                              problem_data->setup_rhs_loc, &qf_setup_rhs);
  CeedQFunctionAddInput(qf_setup_rhs, "x", num_comp_x, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup_rhs, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup_rhs, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup_rhs, "true_soln", dim, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup_rhs, "rhs", dim, CEED_EVAL_INTERP);
  // Create the operator that builds the RHS and true solution
  CeedOperatorCreate(ceed, qf_setup_rhs, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_setup_rhs);
  CeedOperatorSetField(op_setup_rhs, "x", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_rhs, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_rhs, "true_soln", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, *target);
  CeedOperatorSetField(op_setup_rhs, "rhs", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Setup RHS and true solution
  CeedOperatorApply(op_setup_rhs, x_coord, rhs_ceed, CEED_REQUEST_IMMEDIATE);

  // ---------------------------------------------------------------------------
  // Persistent libCEED vectors
  // ---------------------------------------------------------------------------
  // -- Operator action variables
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->x_ceed);
  CeedVectorCreate(ceed, U_loc_size, &ceed_data->y_ceed);

  // Local residual evaluator
  // ---------------------------------------------------------------------------
  // Create the QFunction and Operator that computes the residual of the PDE.
  // ---------------------------------------------------------------------------
  // -- QFunction
  CeedQFunctionCreateInterior(ceed, 1, problem_data->residual,
                              problem_data->residual_loc, &qf_residual);
  CeedQFunctionAddInput(qf_residual, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_residual, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_residual, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_residual, "v", dim, CEED_EVAL_INTERP);

  // -- Operator
  CeedOperatorCreate(ceed, qf_residual, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_residual);
  CeedOperatorSetField(op_residual, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_residual, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, x_coord);
  CeedOperatorSetField(op_residual, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_residual, "v", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_residual = qf_residual;
  ceed_data->op_residual = op_residual;

  // ---------------------------------------------------------------------------
  // Setup Error Qfunction
  // ---------------------------------------------------------------------------
  // Create the q-function that sets up the error
  CeedQFunctionCreateInterior(ceed, 1, problem_data->setup_error,
                              problem_data->setup_error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "dx", dim*dim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_error, "u", dim, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", dim, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_error, "error", dim, CEED_EVAL_NONE);
  // Create the operator that builds the error
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_error);
  CeedOperatorSetField(op_error, "dx", ceed_data->elem_restr_x,
                       ceed_data->basis_x, x_coord);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u,
                       ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, *target);
  CeedOperatorSetField(op_error, "weight", CEED_ELEMRESTRICTION_NONE,
                       ceed_data->basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u_i,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  // -- Save libCEED data to apply operator in matops.c
  ceed_data->qf_error = qf_error;
  ceed_data->op_error = op_error;

  CeedQFunctionDestroy(&qf_setup_rhs);
  CeedOperatorDestroy(&op_setup_rhs);
  CeedVectorDestroy(&x_coord);

  PetscFunctionReturn(0);
};
// -----------------------------------------------------------------------------