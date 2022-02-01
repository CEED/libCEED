#ifndef libceed_petsc_examples_structs_h
#define libceed_petsc_examples_structs_h

#include <ceed.h>
#include <petsc.h>

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct UserO_ *UserO;
struct UserO_ {
  MPI_Comm comm;
  DM dm;
  Vec X_loc, Y_loc, diag;
  CeedVector x_ceed, y_ceed;
  CeedOperator op;
  Ceed ceed;
};

// Data for PETSc Prolong/Restrict Matshells
typedef struct UserProlongRestr_ *UserProlongRestr;
struct UserProlongRestr_ {
  MPI_Comm comm;
  DM dmc, dmf;
  Vec loc_vec_c, loc_vec_f, mult_vec;
  CeedVector ceed_vec_c, ceed_vec_f;
  CeedOperator op_prolong, op_restrict;
  Ceed ceed;
};

// -----------------------------------------------------------------------------
// libCEED Data Structs
// -----------------------------------------------------------------------------

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed ceed;
  CeedBasis basis_x, basis_u, basis_c_to_f;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction qf_apply;
  CeedOperator op_apply, op_restrict, op_prolong;
  CeedVector q_data, x_ceed, y_ceed;
};

// BP specific data
typedef struct {
  CeedInt num_comp_x, num_comp_u, topo_dim, q_data_size, q_extra;
  CeedQFunctionUser setup_geo, setup_rhs, apply, error;
  const char *setup_geo_loc, *setup_rhs_loc, *apply_loc, *error_loc;
  CeedEvalMode in_mode, out_mode;
  CeedQuadMode q_mode;
  PetscBool enforce_bc;
  PetscErrorCode (*bc_func)(PetscInt, PetscReal, const PetscReal *,
                            PetscInt, PetscScalar *, void *);
} BPData;

#endif // libceed_petsc_examples_structs_h
