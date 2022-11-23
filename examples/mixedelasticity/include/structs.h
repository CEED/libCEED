#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// PETSc operator contexts
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm     comm;
  DM           dm;
  Vec          X_loc, Y_loc, diag;
  CeedVector   x_ceed, y_ceed;
  CeedOperator op_apply;
  Ceed         ceed;
};

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis           basis_x, basis_u;
  CeedElemRestriction elem_restr_x, elem_restr_u;
  CeedQFunction       qf_residual;
  CeedOperator        op_residual;
  CeedVector          q_data, x_ceed, y_ceed, x_coord;
};

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  char     ceed_resource[PETSC_MAX_PATH_LEN];  // libCEED backend
  MPI_Comm comm;
  // Degree of polynomial, extra quadrature pts
  PetscInt order;
  PetscInt q_extra;
  // Problem type arguments
  PetscFunctionList    problems;
  char                 problem_name[PETSC_MAX_PATH_LEN];
  OperatorApplyContext ctx_residual;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser    true_solution, residual;
  const char          *true_solution_loc, *residual_loc;
  CeedQuadMode         quadrature_mode;
  CeedInt              q_data_size_face;
  CeedQFunctionContext residual_qfunction_ctx;
};

#endif  // structs_h