#ifndef neo_hookean_h
#define neo_hookean_h

#include <petsc.h>
#include "../include/structs.h"

#ifndef PHYSICS_STRUCT_NH
#define PHYSICS_STRUCT_NH
typedef struct Physics_NH_ *Physics_NH;
struct Physics_NH_ {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif // PHYSICS_STRUCT_NH

// Create context object
PetscErrorCode PhysicsContext_NH(MPI_Comm comm, Ceed ceed, Units *units,
                                 CeedQFunctionContext *ctx);
PetscErrorCode PhysicsSmootherContext_NH(MPI_Comm comm, Ceed ceed,
    CeedQFunctionContext ctx, CeedQFunctionContext *ctx_smoother);

// Process physics options
PetscErrorCode ProcessPhysics_NH(MPI_Comm comm, Physics_NH phys, Units units);

#endif // neo_hookean_h
