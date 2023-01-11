#ifndef libceed_petsc_examples_structs_h
#define libceed_petsc_examples_structs_h

#include <ceed.h>
#include <petsc.h>

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm     comm;
  DM           dm;
  Vec          X_loc, Y_loc, diag;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op;
  Ceed         ceed;
};

// Data for PETSc Prolong/Restrict Matshells
typedef struct ProlongRestrContext_ *ProlongRestrContext;
struct ProlongRestrContext_ {
  MPI_Comm     comm;
  DM           dmc, dmf;
  Vec          loc_vec_c, loc_vec_f, mult_vec;
  CeedVector   ceed_vec_c, ceed_vec_f;
  CeedOperator op_prolong, op_restrict;
  Ceed         ceed;
};

// -----------------------------------------------------------------------------
// libCEED Data Structs
// -----------------------------------------------------------------------------

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed                ceed;
  CeedBasis           basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_qd_i;
  CeedQFunction       qf_apply;
  CeedOperator        op_apply, op_restrict, op_prolong;
  CeedVector          q_data, x_ceed, y_ceed;
  CeedInt             q_data_size;
};

// BP specific data
typedef struct {
  CeedInt           num_comp_x, num_comp_u, topo_dim, q_data_size, q_extra;
  CeedQFunctionUser setup_geo, setup_rhs, apply, error;
  const char       *setup_geo_loc, *setup_rhs_loc, *apply_loc, *error_loc;
  CeedEvalMode      in_mode, out_mode;
  CeedQuadMode      q_mode;
  PetscBool         enforce_bc;
} BPData;

// BP options
typedef enum { CEED_BP1 = 0, CEED_BP2 = 1, CEED_BP3 = 2, CEED_BP4 = 3, CEED_BP5 = 4, CEED_BP6 = 5 } BPType;

// -----------------------------------------------------------------------------
// Parameter structure for running problems
// -----------------------------------------------------------------------------
typedef struct RunParams_ *RunParams;
struct RunParams_ {
  MPI_Comm      comm;
  PetscBool     test_mode, read_mesh, user_l_nodes, write_solution, simplex;
  char         *filename, *hostname;
  PetscInt      local_nodes, degree, q_extra, dim, num_comp_u, *mesh_elem;
  PetscInt      ksp_max_it_clip[2];
  PetscMPIInt   ranks_per_node;
  BPType        bp_choice;
  PetscLogStage solve_stage;
};

#endif  // libceed_petsc_examples_structs_h
