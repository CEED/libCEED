#include <ceed.h>
#include <petsc.h>

// -----------------------------------------------------------------------------
// Boundary Conditions
// -----------------------------------------------------------------------------

// Diff boundary condition function
PetscErrorCode BCsDiff(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt num_comp_u, PetscScalar *u, void *ctx) {
  // *INDENT-OFF*
  #ifndef M_PI
  #define M_PI    3.14159265358979323846
  #endif
  // *INDENT-ON*
  const CeedScalar c[3] = { 0, 1., 2. };
  const CeedScalar k[3] = { 1., 2., 3. };

  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < num_comp_u; i++)
    u[i] = sin(M_PI*(c[0] + k[0]*x[0])) *
           sin(M_PI*(c[1] + k[1]*x[1])) *
           sin(M_PI*(c[2] + k[2]*x[2]));

  PetscFunctionReturn(0);
}

// Mass boundary condition function
PetscErrorCode BCsMass(PetscInt dim, PetscReal time, const PetscReal x[],
                       PetscInt num_comp_u, PetscScalar *u, void *ctx) {
  PetscFunctionBeginUser;

  for (PetscInt i = 0; i < num_comp_u; i++)
    u[i] = PetscSqrtScalar(PetscSqr(x[0]) + PetscSqr(x[1]) +
                           PetscSqr(x[2]));

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
