#include "../include/setup-boundary.h"

// ---------------------------------------------------------------------------
// Create boundary label
// ---------------------------------------------------------------------------
PetscErrorCode CreateBCLabel(DM dm, const char name[]) {
  DMLabel        label;

  PetscFunctionBeginUser;

  PetscCall( DMCreateLabel(dm, name) );
  PetscCall( DMGetLabel(dm, name, &label) );
  PetscCall( DMPlexMarkBoundaryFaces(dm, PETSC_DETERMINE, label) );
  PetscCall( DMPlexLabelComplete(dm, label) );

  PetscFunctionReturn(0);
};

// ---------------------------------------------------------------------------
// Add Dirichlet boundaries to DM
// ---------------------------------------------------------------------------
PetscErrorCode DMAddBoundariesDirichlet(DM dm) {

  PetscFunctionBeginUser;

  // BCs given by manufactured solution
  PetscBool  has_label;
  const char *name = "MMS Face Sets";
  PetscInt   face_ids[1] = {1};
  PetscCall( DMHasLabel(dm, name, &has_label) );
  if (!has_label) {
    PetscCall( CreateBCLabel(dm, name) );
  }
  DMLabel label;
  PetscCall( DMGetLabel(dm, name, &label) );
  PetscCall( DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, 1, face_ids, 0, 0,
                           NULL,
                           (void(*)(void))BoundaryDirichletMMS, NULL, NULL, NULL) );


  PetscFunctionReturn(0);
}

// ---------------------------------------------------------------------------
// Add Neumann boundaries to DM
// ---------------------------------------------------------------------------
PetscErrorCode DMAddBoundariesPressure(Ceed ceed, CeedData ceed_data,
                                       AppCtx app_ctx, ProblemData problem_data, DM dm,
                                       CeedVector bc_pressure) {
  PetscInt             dim;
  CeedQFunction        qf_pressure;
  CeedOperator         op_pressure;

  PetscFunctionBeginUser;

  PetscCall( DMGetDimension(dm, &dim) );

  if (app_ctx->bc_pressure_count > 0) {
    DMLabel domain_label;
    PetscCall(DMGetLabel(dm, "Face Sets", &domain_label));
    // Compute contribution on each boundary face
    for (CeedInt i = 0; i < app_ctx->bc_pressure_count; i++) {

      CeedQFunctionCreateInterior(ceed, 1, problem_data->bc_pressure,
                                  problem_data->bc_pressure_loc, &qf_pressure);

      CeedQFunctionAddInput(qf_pressure, "weight", 1, CEED_EVAL_WEIGHT);
      CeedQFunctionAddOutput(qf_pressure, "v", dim, CEED_EVAL_INTERP);
      // -- Apply operator
      CeedOperatorCreate(ceed, qf_pressure, NULL, NULL,
                         &op_pressure);
      CeedOperatorSetField(op_pressure, "weight", CEED_ELEMRESTRICTION_NONE,
                           ceed_data->basis_x, CEED_VECTOR_NONE);
      CeedOperatorSetField(op_pressure, "v", ceed_data->elem_restr_u,
                           ceed_data->basis_u_face, CEED_VECTOR_ACTIVE);
      // ---- Compute pressure on face
      CeedOperatorApplyAdd(op_pressure, ceed_data->x_coord, bc_pressure,
                           CEED_REQUEST_IMMEDIATE);

      // -- Cleanup
      CeedQFunctionDestroy(&qf_pressure);
      CeedOperatorDestroy(&op_pressure);
    }
  }

  PetscFunctionReturn(0);
}

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif
// ---------------------------------------------------------------------------
// Boundary function for manufactured solution
// ---------------------------------------------------------------------------
PetscErrorCode BoundaryDirichletMMS(PetscInt dim, PetscReal t,
                                    const PetscReal coords[],
                                    PetscInt num_comp_u, PetscScalar *u, void *ctx) {
  PetscScalar x = coords[0];
  PetscScalar y = coords[1];
  PetscScalar z = coords[1];

  PetscFunctionBeginUser;

  if (dim == 2) {
    u[0] = -M_PI*cos(M_PI*x) *sin(M_PI*y) - M_PI*y;
    u[1] = -M_PI*sin(M_PI*x) *cos(M_PI*y) - M_PI*x;
  } else {
    u[0] = -M_PI*cos(M_PI*x) *sin(M_PI*y) *sin(M_PI*z) - M_PI*y*z;
    u[1] = -M_PI*sin(M_PI*x) *cos(M_PI*y) *sin(M_PI*z) - M_PI*x*z;
    u[2] = -M_PI*sin(M_PI*x) *sin(M_PI*y) *cos(M_PI*z) - M_PI*x*y;
  }

  PetscFunctionReturn(0);
}
