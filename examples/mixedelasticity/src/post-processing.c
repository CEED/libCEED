#include "../include/post-processing.h"

// -----------------------------------------------------------------------------
// This function print the output
// -----------------------------------------------------------------------------
PetscErrorCode PrintOutput(DM dm, Ceed ceed, AppCtx app_ctx, SNES snes, KSP ksp, Vec X, CeedScalar l2_error_u, CeedScalar l2_error_p) {
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
  PetscInt c_start, c_end, num_elem, dim;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  num_elem = c_end - c_start;
  PetscCall(DMGetDimension(dm, &dim));
  DMPolytopeType cell_type;
  PetscCall(DMPlexGetCellType(dm, c_start, &cell_type));
  CeedElemTopology elem_topo = ElemTopologyP2C(cell_type);
  PetscCall(PetscPrintf(app_ctx->comm,
                        "  Problem:\n"
                        "    Problem Name                            : %s\n"
                        "  Mesh:\n"
                        "    Polynomial order of u field             : %" CeedInt_FMT "\n"
                        "    Polynomial order of p field             : %" CeedInt_FMT "\n"
                        "    Quadrature space order (Q)              : %" CeedInt_FMT "\n"
                        "    Additional quadrature points (q_extra)  : %" CeedInt_FMT "\n"
                        "    Global dofs (u + p)                     : %" PetscInt_FMT "\n"
                        "    Owned dofs (u + p)                      : %" PetscInt_FMT "\n"
                        "    Number of elements                      : %" PetscInt_FMT "\n"
                        "    Element topology                        : %s\n",
                        app_ctx->problem_name, app_ctx->u_order, app_ctx->p_order, app_ctx->q_order, app_ctx->q_extra, X_g_size, X_l_size, num_elem,
                        CeedElemTopologies[elem_topo]));
  // -- SNES
  PetscInt its, snes_its = 0;
  PetscCall(SNESGetIterationNumber(snes, &its));
  snes_its += its;
  SNESType            snes_type;
  SNESConvergedReason snes_reason;
  PetscReal           snes_rnorm;
  PetscCall(SNESGetType(snes, &snes_type));
  PetscCall(SNESGetConvergedReason(snes, &snes_reason));
  PetscCall(SNESGetFunctionNorm(snes, &snes_rnorm));
  PetscCall(PetscPrintf(app_ctx->comm,
                        "  SNES:\n"
                        "    SNES Type                               : %s\n"
                        "    SNES Convergence                        : %s\n"
                        "    Total SNES Iterations                   : %" PetscInt_FMT "\n"
                        "    Final rnorm                             : %e\n",
                        snes_type, SNESConvergedReasons[snes_reason], snes_its, (double)snes_rnorm));
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
                        "    L2 error of u and p                     : %e, %e\n",
                        (double)l2_error_u, (double)l2_error_p));
  PetscFunctionReturn(0);
};
