#include "../include/post-processing.h"

// -----------------------------------------------------------------------------
// This function print the output
// -----------------------------------------------------------------------------
PetscErrorCode PrintOutput(DM dm, Ceed ceed, AppCtx app_ctx, KSP ksp, Vec X, CeedScalar l2_error_u) {
  PetscFunctionBeginUser;

  const char *used_resource;
  CeedMemType mem_type_backend;
  CeedGetResource(ceed, &used_resource);
  CeedGetPreferredMemType(ceed, &mem_type_backend);
  char hostname[PETSC_MAX_PATH_LEN];
  PetscCall(PetscGetHostName(hostname, sizeof hostname));
  PetscMPIInt comm_size;
  PetscCall(MPI_Comm_size(app_ctx->comm, &comm_size));
  PetscCall(PetscPrintf(app_ctx->comm,
                        "\n-- Mixed-Elasticity Example - libCEED + PETSc --\n"
                        "  MPI:\n"
                        "    Hostname                                : %s\n"
                        "    Total ranks                             : %d\n"
                        "  libCEED:\n"
                        "    libCEED Backend                         : %s\n"
                        "    libCEED Backend MemType                 : %s\n",
                        hostname, comm_size, used_resource, CeedMemTypes[mem_type_backend]));

  MatType mat_type;
  VecType vec_type;
  PetscCall(DMGetMatType(dm, &mat_type));
  PetscCall(DMGetVecType(dm, &vec_type));
  PetscCall(PetscPrintf(app_ctx->comm,
                        "  PETSc:\n"
                        "    DM MatType                              : %s\n"
                        "    DM VecType                              : %s\n",
                        mat_type, vec_type));
  PetscInt X_l_size, X_g_size;
  PetscCall(VecGetSize(X, &X_g_size));
  PetscCall(VecGetLocalSize(X, &X_l_size));
  PetscInt c_start, c_end;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  DMPolytopeType cell_type;
  PetscCall(DMPlexGetCellType(dm, c_start, &cell_type));
  CeedElemTopology elem_topo = ElemTopologyP2C(cell_type);
  PetscCall(PetscPrintf(app_ctx->comm,
                        "  Problem:\n"
                        "    Problem Name                            : %s\n"
                        "  Mesh:\n"
                        "    Solution Order (P)                      : %" CeedInt_FMT "\n"
                        "    Quadrature  Order (Q)                   : %" CeedInt_FMT "\n"
                        "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                        "    Global nodes                            : %" PetscInt_FMT "\n"
                        "    Local Elements                          : %" PetscInt_FMT "\n"
                        "    Element topology                        : %s\n"
                        "    Owned nodes                             : %" PetscInt_FMT "\n"
                        "    DoF per node                            : %" PetscInt_FMT "\n",
                        app_ctx->problem_name, app_ctx->p_order, app_ctx->q_order, app_ctx->q_extra, X_g_size / 3, c_end - c_start,
                        CeedElemTopologies[elem_topo], X_l_size / 3, 3));
  PetscInt           ksp_its;
  KSPType            ksp_type;
  KSPConvergedReason ksp_reason;
  PetscReal          ksp_rnorm;
  PC                 pc;
  PCType             pc_type;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &pc_type));
  PetscCall(KSPGetType(ksp, &ksp_type));
  PetscCall(KSPGetConvergedReason(ksp, &ksp_reason));
  PetscCall(KSPGetIterationNumber(ksp, &ksp_its));
  PetscCall(KSPGetResidualNorm(ksp, &ksp_rnorm));
  PetscCall(PetscPrintf(app_ctx->comm,
                        "  KSP:\n"
                        "    KSP Type                                : %s\n"
                        "    PC Type                                 : %s\n"
                        "    KSP Convergence                         : %s\n"
                        "    Total KSP Iterations                    : %" PetscInt_FMT "\n"
                        "    Final rnorm                             : %e\n",
                        ksp_type, pc_type, KSPConvergedReasons[ksp_reason], ksp_its, (double)ksp_rnorm));

  PetscCall(PetscPrintf(app_ctx->comm,
                        "  L2 Error (MMS):\n"
                        "    L2 error of u                           : %e\n",
                        (double)l2_error_u));
  PetscFunctionReturn(0);
};
