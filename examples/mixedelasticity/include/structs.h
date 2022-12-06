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

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis           basis_x, basis_u, basis_p;
  CeedElemRestriction elem_restr_x, elem_restr_u, elem_restr_qdata, elem_restr_u_i, elem_restr_p;
  CeedQFunction       qf_residual, qf_jacobian, qf_error_u, qf_error_p;
  CeedOperator        op_residual, op_jacobian, op_error_u, op_error_p;
  CeedVector          q_data, x_ceed, y_ceed, x_coord;
};

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  MPI_Comm comm;
  // Degree of basis for u and p fields, and quadrature space
  PetscInt u_order, p_order, q_order;
  PetscInt q_extra;  // extra quadrature pts if needed
  // Problem type arguments
  PetscFunctionList    problems;
  char                 problem_name[PETSC_MAX_PATH_LEN];
  OperatorApplyContext ctx_residual, ctx_jacobian, ctx_error_u, ctx_error_p;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser    setup_geo, setup_rhs, residual, jacobian, error_u, error_p;
  const char          *setup_geo_loc, *setup_rhs_loc, *residual_loc, *jacobian_loc, *error_u_loc, *error_p_loc;
  CeedQuadMode         quadrature_mode;
  CeedInt              q_data_size;
  CeedQFunctionContext residual_qfunction_ctx, jacobian_qfunction_ctx, rhs_qfunction_ctx;
};

#endif  // structs_h