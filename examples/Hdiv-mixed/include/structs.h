#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// PETSc operator contexts
typedef struct OperatorApplyContext_ *OperatorApplyContext;
struct OperatorApplyContext_ {
  MPI_Comm        comm;
  Vec             X_loc, Y_loc, X_t_loc;
  CeedVector      x_ceed, y_ceed, x_t_ceed, x_coord, rhs_ceed_H1;
  CeedOperator    op_apply, op_rhs_H1;
  DM              dm;
  Ceed            ceed;
  CeedScalar      t, dt;
  CeedContextFieldLabel solution_time_label, final_time_label,
                        timestep_label;
  CeedElemRestriction  elem_restr_u_H1;
  VecType         vec_type;
};

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  CeedBasis            basis_x, basis_u, basis_p, basis_u_face;
  CeedElemRestriction  elem_restr_x, elem_restr_u, elem_restr_U_i,
                       elem_restr_p, elem_restr_p_i, elem_restr_u0,
                       elem_restr_p0, elem_restr_u_H1;
  CeedQFunction        qf_residual, qf_jacobian, qf_error, qf_ics_u, qf_ics_p,
                       qf_rhs_u0, qf_rhs_p0, qf_rhs_H1, qf_post_mass;
  CeedOperator         op_residual, op_jacobian, op_error, op_ics_u, op_ics_p,
                       op_rhs_u0, op_rhs_p0, op_rhs_H1, op_post_mass;
  CeedVector           x_ceed, y_ceed, x_coord, x_t_ceed, rhs_u0_ceed,
                       u0_ceed, v0_ceed, rhs_p0_ceed, p0_ceed, q0_ceed,
                       rhs_ceed_H1, u_ceed, up_ceed, vp_ceed;
  CeedInt              num_elem;
};

// Application context from user command line options
typedef struct AppCtx_ *AppCtx;
struct AppCtx_ {
  char              ceed_resource[PETSC_MAX_PATH_LEN]; // libCEED backend
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
  CeedScalar        t_final, t;
  PetscBool         view_solution, quartic;
  char              output_dir[PETSC_MAX_PATH_LEN];
  PetscInt          output_freq;
  OperatorApplyContext ctx_residual, ctx_jacobian, ctx_error, ctx_residual_ut,
                       ctx_initial_u0, ctx_initial_p0, ctx_Hdiv, ctx_H1;
};

// Problem specific data
typedef struct ProblemData_ *ProblemData;
struct ProblemData_ {
  CeedQFunctionUser true_solution, residual, jacobian, error, ics_u, ics_p,
                    bc_pressure, rhs_u0, rhs_p0, post_rhs, post_mass;
  const char        *true_solution_loc, *residual_loc, *jacobian_loc,
        *error_loc, *bc_pressure_loc, *ics_u_loc, *ics_p_loc, *rhs_u0_loc,
        *rhs_p0_loc, *post_rhs_loc, *post_mass_loc;
  CeedQuadMode      quadrature_mode;
  CeedInt           elem_node, dim, q_data_size_face;
  CeedQFunctionContext true_qfunction_ctx, error_qfunction_ctx,
                       residual_qfunction_ctx, jacobian_qfunction_ctx,
                       rhs_u0_qfunction_ctx ;
  PetscBool         has_ts, view_solution, quartic;
};

#endif // structs_h