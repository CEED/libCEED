#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  MPI_Comm          comm;
  // Degree of polynomial (1 only), extra quadrature pts
  PetscInt          degree;
  PetscInt          q_extra;
  // For applying traction BCs
  PetscInt          bc_pressure_count;
  PetscInt          bc_faces[16]; // face ID
  PetscScalar       bc_pressure_value[16];
  // Problem type arguments
  PetscFunctionList problems;
  char              problem_name[PETSC_MAX_PATH_LEN];
};

// 2) richard
// We have 3 experiment parameters as described in Table 1:P1, P2, P3
// Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872

#ifndef PHYSICS_RICHARDP2_STRUCT
#define PHYSICS_RICHARDP2_STRUCT
typedef struct RICHARDP2Context_ *RICHARDP2Context;
struct RICHARDP2Context_ {
  CeedScalar K_star;
  CeedScalar theta_s;
  CeedScalar theta_r;
  CeedScalar alpha_v;
  CeedScalar n_v;
  CeedScalar m_v;
  CeedScalar m_r;
  CeedScalar rho_0;
  CeedScalar beta;
};
#endif

#ifndef PHYSICS_RICHARDP3_STRUCT
#define PHYSICS_RICHARDP3_STRUCT
typedef struct RICHARDP3Context_ *RICHARDP3Context;
struct RICHARDP3Context_ {
  CeedScalar K_star;
  CeedScalar theta_s;
  CeedScalar theta_r;
  CeedScalar alpha_star_v;
  CeedScalar n_v;
  CeedScalar m_v;
  CeedScalar rho_0;
  CeedScalar beta;
};
#endif

// Struct that contains all enums and structs used for the physics of all problems
typedef struct Physics_ *Physics;
struct Physics_ {
  RICHARDP2Context        richard_p2_ctx;
  RICHARDP3Context        richard_p3_ctx;
};

// PETSc operator contexts
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm        comm;
  Vec             X_loc, Y_loc, X_t_loc;
  CeedVector      x_ceed, y_ceed, x_t_ceed;
  CeedOperator    op_apply;
  DM              dm;
  Ceed            ceed;
};

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis            basis_x, basis_u, basis_p, basis_u_face;
  CeedElemRestriction  elem_restr_x, elem_restr_u, elem_restr_U_i,
                       elem_restr_p, elem_restr_p_i;
  CeedQFunction        qf_residual, qf_jacobian, qf_error, qf_ics;
  CeedOperator         op_residual, op_jacobian, op_error, op_ics;
  CeedVector           x_ceed, y_ceed, x_coord, U0_ceed, x_t_ceed;
  OperatorApplyContext ctx_residual, ctx_jacobian, ctx_error, ctx_residual_ut;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser true_solution, residual, jacobian, error, ics,
                    bc_pressure;
  const char        *true_solution_loc, *residual_loc, *jacobian_loc,
        *error_loc, *bc_pressure_loc, *ics_loc;
  CeedQuadMode      quadrature_mode;
  CeedInt           elem_node, dim, q_data_size_face;
  CeedQFunctionContext qfunction_context;
  PetscBool         has_ts;
};

#endif // structs_h