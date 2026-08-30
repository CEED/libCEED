#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// PETSc operator contexts
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm     comm;
  DM           dm;
  Vec          X_loc, Y_loc;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op_apply;
  Ceed         ceed;
};

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  MPI_Comm comm;
  // libCEED arguments
  PetscInt degree;
  PetscInt q_extra;
  // Problem type arguments
  PetscFunctionList    problems;
  char                 problem_name[PETSC_MAX_PATH_LEN];
  OperatorApplyContext ctx_residual, ctx_error_u;
};

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis           basis_x, basis_u, basis_p;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_u_i, elem_restr_p;
  CeedQFunction       qf_residual, qf_error;
  CeedOperator        op_residual, op_error;
  CeedVector          x_ceed, y_ceed;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser setup_rhs, residual, setup_error;
  const char       *setup_rhs_loc, *residual_loc, *setup_error_loc;
  CeedQuadMode      quadrature_mode;
  CeedInt           elem_node, dim;
};

#endif  // structs_h