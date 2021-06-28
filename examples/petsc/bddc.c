// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

//                        libCEED + PETSc Example: CEED BPs 3-6 with BDDC
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP benchmark problems, see http://ceed.exascaleproject.org/bps.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make bddc [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bddc -problem bp3
//     bddc -problem bp4
//     bddc -problem bp5 -ceed /cpu/self
//     bddc -problem bp6 -ceed /gpu/cuda
//
//TESTARGS -ceed {ceed_resource} -test -problem bp3 -degree 3

/// @file
/// CEED BPs 1-6 BDDC example using PETSc
const char help[] = "Solve CEED BPs using BDDC with PETSc and DMPlex\n";

#include "bddc.h"

// The BDDC example uses vectors in three spaces
//
//  Fine mesh:       Broken mesh:      Vertex mesh:    Broken vertex mesh:
// x----x----x      x----x x----x       x    x    x      x    x x    x
// |    |    |      |    | |    |
// |    |    |      |    | |    |
// x----x----x      x----x x----x       x    x    x      x    x x    x
//
// Vectors are organized as follows
//  - *_Pi    : vector on the vertex mesh
//  - *_Pi_r  : vector on the broken vertex mesh
//  - *_r     : vector on the broken mesh, all points but vertices
//  - *_Gamma : vector on the broken mesh, face/vertex/edge points
//  - *_I     : vector on the broken mesh, interior points
//  - *       : all other vectors are on the fine mesh

int main(int argc, char **argv) {
  MPI_Comm      comm;
  char          filename[PETSC_MAX_PATH_LEN], ceed_resource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  double        my_rt_start, my_rt, rt_min, rt_max;
  PetscInt      degree = 3, q_extra, l_size, xl_size, g_size, l_Pi_size, xl_Pi_size, g_Pi_size, dim = 3, mesh_elem[3] = {3, 3, 3}, num_comp_u = 1;
  PetscScalar  *r;
  PetscScalar   eps = 1.0;
  PetscBool     test_mode, benchmark_mode, read_mesh, write_solution;
  PetscLogStage solve_stage;
  DM            dm, dm_Pi;
  SNES          snes_Pi, snes_Pi_r;
  KSP           ksp, ksp_S_Pi, ksp_S_Pi_r;
  PC            pc;
  Mat           mat_O, mat_S_Pi, mat_S_Pi_r;
  Vec           X, X_loc, X_Pi, X_Pi_loc, X_Pi_r_loc, rhs, rhs_loc;
  PetscMemType  mem_type;
  OperatorApplyContext apply_ctx, error_ctx;
  BDDCApplyContext     bddc_ctx;
  Ceed                 ceed;
  CeedData             ceed_data;
  CeedDataBDDC         ceed_data_bddc;
  CeedVector           rhs_ceed, target;
  CeedQFunction        qf_error, qf_restrict, qf_prolong;
  CeedOperator         op_error;
  BPType               bp_choice;
  InjectionType        injection;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  // Parse command line options
  PetscOptionsBegin(comm, NULL, "CEED BPs in PETSc", NULL);
  bp_choice = CEED_BP3;
  PetscCall(PetscOptionsEnum("-problem", "CEED benchmark problem to solve", NULL, bp_types, (PetscEnum)bp_choice, (PetscEnum *)&bp_choice, NULL));
  num_comp_u = bp_options[bp_choice].num_comp_u;
  test_mode  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-test", "Testing mode (do not print unless error is large)", NULL, test_mode, &test_mode, NULL));
  benchmark_mode = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-benchmark", "Benchmarking mode (prints benchmark statistics)", NULL, benchmark_mode, &benchmark_mode, NULL));
  write_solution = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-write_solution", "Write solution for visualization", NULL, write_solution, &write_solution, NULL));
  PetscCall(PetscOptionsScalar("-eps", "Epsilon parameter for Kershaw mesh transformation", NULL, eps, &eps, NULL));
  if (eps > 1 || eps <= 0) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-eps %g must be (0,1]", (double)PetscRealPart(eps));
  degree = test_mode ? 3 : 2;
  PetscCall(PetscOptionsInt("-degree", "Polynomial degree of tensor product basis", NULL, degree, &degree, NULL));
  if (degree < 2) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-degree %" PetscInt_FMT " must be at least 2", degree);
  q_extra = bp_options[bp_choice].q_extra;
  PetscCall(PetscOptionsInt("-q_extra", "Number of extra quadrature points", NULL, q_extra, &q_extra, NULL));
  PetscCall(PetscOptionsString("-ceed", "CEED resource specifier", NULL, ceed_resource, ceed_resource, sizeof(ceed_resource), NULL));
  injection = INJECTION_SCALED;
  PetscCall(PetscOptionsEnum("-injection", "Injection strategy to use", NULL, injection_types, (PetscEnum)injection, (PetscEnum *)&injection, NULL));
  read_mesh = PETSC_FALSE;
  PetscCall(PetscOptionsString("-mesh", "Read mesh from file", NULL, filename, filename, sizeof(filename), &read_mesh));
  if (!read_mesh) {
    PetscInt tmp = dim;
    PetscCall(PetscOptionsIntArray("-cells", "Number of cells per dimension", NULL, mesh_elem, &tmp, NULL));
  }
  PetscOptionsEnd();

  // Set up libCEED
  CeedInit(ceed_resource, &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);

  // Setup DMs
  if (read_mesh) {
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, NULL, PETSC_TRUE, &dm));
  } else {
    PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, PETSC_FALSE, mesh_elem, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_FALSE, &dm));
  }

  {
    DM               dm_dist = NULL;
    PetscPartitioner part;

    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm, 0, NULL, &dm_dist));
    if (dm_dist) {
      PetscCall(DMDestroy(&dm));
      dm = dm_dist;
    }
  }

  // Apply Kershaw mesh transformation
  PetscCall(Kershaw(dm, eps));

  VecType vec_type;
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      CeedGetResource(ceed, &resolved);
      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
      else vec_type = VECSTANDARD;
    }
  }
  PetscCall(DMSetVecType(dm, vec_type));

  // Setup DM
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(SetupDMByDegree(dm, degree, bp_options[bp_choice].q_extra, num_comp_u, dim, bp_options[bp_choice].enforce_bc));

  // Set up subdomain vertex DM
  PetscCall(DMClone(dm, &dm_Pi));
  PetscCall(DMSetVecType(dm_Pi, vec_type));
  PetscCall(SetupVertexDMFromDM(dm, dm_Pi, degree + bp_options[bp_choice].q_extra, num_comp_u, bp_options[bp_choice].enforce_bc));

  // Create vectors
  // -- Fine mesh
  PetscCall(DMCreateGlobalVector(dm, &X));
  PetscCall(VecGetLocalSize(X, &l_size));
  PetscCall(VecGetSize(X, &g_size));
  PetscCall(DMCreateLocalVector(dm, &X_loc));
  PetscCall(VecGetSize(X_loc, &xl_size));
  // -- Vertex mesh
  PetscCall(DMCreateGlobalVector(dm_Pi, &X_Pi));
  PetscCall(VecGetLocalSize(X_Pi, &l_Pi_size));
  PetscCall(VecGetSize(X_Pi, &g_Pi_size));
  PetscCall(DMCreateLocalVector(dm_Pi, &X_Pi_loc));
  PetscCall(VecGetSize(X_Pi_loc, &xl_Pi_size));

  // Operator
  PetscCall(PetscNew(&apply_ctx));
  PetscCall(MatCreateShell(comm, l_size, l_size, g_size, g_size, apply_ctx, &mat_O));
  PetscCall(MatShellSetOperation(mat_O, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  PetscCall(MatShellSetOperation(mat_O, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiag));
  PetscCall(MatShellSetVecType(mat_O, vec_type));

  // Print global grid information
  if (!test_mode) {
    PetscInt P = degree + 1, Q = P + q_extra;

    const char *used_resource;
    CeedGetResource(ceed, &used_resource);

    PetscCall(VecGetType(X, &vec_type));

    PetscCall(PetscPrintf(comm,
                          "\n-- CEED Benchmark Problem %" CeedInt_FMT " -- libCEED + PETSc + BDDC --\n"
                          "  PETSc:\n"
                          "    PETSc Vec Type                     : %s\n"
                          "  libCEED:\n"
                          "    libCEED Backend                    : %s\n"
                          "    libCEED Backend MemType            : %s\n"
                          "  Mesh:\n"
                          "    Number of 1D Basis Nodes (p)       : %" PetscInt_FMT "\n"
                          "    Number of 1D Quadrature Points (q) : %" PetscInt_FMT "\n"
                          "    Global Nodes                       : %" PetscInt_FMT "\n"
                          "    Owned Nodes                        : %" PetscInt_FMT "\n"
                          "    DoF per node                       : %" PetscInt_FMT "\n"
                          "  BDDC:\n"
                          "    Injection                          : %s\n"
                          "    Global Interface Nodes             : %" PetscInt_FMT "\n"
                          "    Owned Interface Nodes              : %" PetscInt_FMT "\n",
                          bp_choice + 1, vec_type, used_resource, CeedMemTypes[mem_type_backend], P, Q, g_size / num_comp_u, l_size / num_comp_u,
                          num_comp_u, injection_types[injection], g_Pi_size, l_Pi_size));
  }

  // Create RHS vector
  PetscCall(VecDuplicate(X, &rhs));
  PetscCall(VecDuplicate(X_loc, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(VecGetArrayAndMemType(rhs_loc, &r, &mem_type));
  CeedVectorCreate(ceed, xl_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // Set up libCEED operator
  PetscCall(PetscCalloc1(1, &ceed_data));
  PetscCall(SetupLibceedByDegree(dm, ceed, degree, dim, q_extra, dim, num_comp_u, g_size, xl_size, bp_options[bp_choice], ceed_data, true, true,
                                 rhs_ceed, &target));

  // Gather RHS
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  PetscCall(VecRestoreArrayAndMemType(rhs_loc, &r));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs));
  CeedVectorDestroy(&rhs_ceed);

  // Set up libCEED operator on interface vertices
  PetscCall(PetscNew(&ceed_data_bddc));
  PetscCall(SetupLibceedBDDC(dm_Pi, ceed_data, ceed_data_bddc, g_Pi_size, xl_Pi_size, bp_options[bp_choice]));

  // Create the injection/restriction QFunction
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_NONE, CEED_EVAL_INTERP, &qf_restrict);
  CeedQFunctionCreateIdentity(ceed, num_comp_u, CEED_EVAL_INTERP, CEED_EVAL_NONE, &qf_prolong);

  // Create the error QFunction
  CeedQFunctionCreateInterior(ceed, 1, bp_options[bp_choice].error, bp_options[bp_choice].error_loc, &qf_error);
  CeedQFunctionAddInput(qf_error, "u", num_comp_u, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", num_comp_u, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_error, "qdata", ceed_data->q_data_size, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", num_comp_u, CEED_EVAL_INTERP);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_error);
  CeedOperatorSetField(op_error, "u", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", ceed_data->elem_restr_u_i, CEED_BASIS_NONE, target);
  CeedOperatorSetField(op_error, "qdata", ceed_data->elem_restr_qd_i, CEED_BASIS_NONE, ceed_data->q_data);
  CeedOperatorSetField(op_error, "error", ceed_data->elem_restr_u, ceed_data->basis_u, CEED_VECTOR_ACTIVE);

  // Calculate multiplicity
  {
    PetscScalar *x;

    // CEED vector
    PetscCall(VecZeroEntries(X_loc));
    PetscCall(VecGetArray(X_loc, &x));
    CeedVectorSetArray(ceed_data->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Multiplicity
    CeedElemRestrictionGetMultiplicity(ceed_data->elem_restr_u, ceed_data->x_ceed);

    // Restore vector
    CeedVectorTakeArray(ceed_data->x_ceed, CEED_MEM_HOST, &x);
    PetscCall(VecRestoreArray(X_loc, &x));

    // Local-to-global
    PetscCall(VecZeroEntries(X));
    PetscCall(DMLocalToGlobal(dm, X_loc, ADD_VALUES, X));

    // Global-to-local
    PetscCall(DMGlobalToLocal(dm, X, INSERT_VALUES, X_loc));
    PetscCall(VecZeroEntries(X));

    // Multiplicity scaling
    PetscCall(VecReciprocal(X_loc));

    // CEED vector
    PetscCall(VecGetArray(X_loc, &x));
    CeedVectorSetArray(ceed_data->x_ceed, CEED_MEM_HOST, CEED_USE_POINTER, x);

    // Inject multiplicity
    CeedOperatorApply(ceed_data_bddc->op_inject_r, ceed_data->x_ceed, ceed_data_bddc->mult_ceed, CEED_REQUEST_IMMEDIATE);
    // Restore vector
    CeedVectorTakeArray(ceed_data->x_ceed, CEED_MEM_HOST, &x);
    PetscCall(VecRestoreArray(X_loc, &x));
    PetscCall(VecZeroEntries(X_loc));
  }

  // Setup dummy SNESs
  {
    // Schur compliment operator
    // -- Jacobian Matrix
    PetscCall(DMSetMatType(dm_Pi, MATAIJ));
    PetscCall(DMCreateMatrix(dm_Pi, &mat_S_Pi));

    // -- Dummy SNES
    PetscCall(SNESCreate(comm, &snes_Pi));
    PetscCall(SNESSetDM(snes_Pi, dm_Pi));
    PetscCall(SNESSetSolution(snes_Pi, X_Pi));

    // -- Jacobian function
    PetscCall(SNESSetJacobian(snes_Pi, mat_S_Pi, mat_S_Pi, NULL, NULL));

    // -- Residual evaluation function
    PetscCall(PetscNew(&bddc_ctx));
    PetscCall(SNESSetFunction(snes_Pi, X_Pi, FormResidual_BDDCSchur, bddc_ctx));
  }
  {
    // Element Schur compliment operator
    // -- Vectors
    CeedInt num_elem, elem_size;

    CeedElemRestrictionGetNumElements(ceed_data_bddc->elem_restr_Pi, &num_elem);
    CeedElemRestrictionGetElementSize(ceed_data_bddc->elem_restr_Pi, &elem_size);
    PetscCall(VecCreate(comm, &X_Pi_r_loc));
    PetscCall(VecSetSizes(X_Pi_r_loc, num_elem * elem_size, PETSC_DECIDE));
    PetscCall(VecSetType(X_Pi_r_loc, vec_type));

    // -- Jacobian Matrix
    PetscCall(MatCreateSeqAIJ(comm, elem_size * num_elem, elem_size * num_elem, elem_size, NULL, &mat_S_Pi_r));
    for (PetscInt e = 0; e < num_elem; e++) {
      for (PetscInt i = 0; i < elem_size; i++) {
        for (PetscInt j = 0; j < elem_size; j++) {
          PetscInt    row   = e * elem_size + i;
          PetscInt    col   = e * elem_size + j;
          PetscScalar value = i + j;
          PetscCall(MatSetValues(mat_S_Pi_r, 1, &row, 1, &col, &value, INSERT_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(mat_S_Pi_r, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat_S_Pi_r, MAT_FINAL_ASSEMBLY));

    // -- Dummy SNES
    PetscCall(SNESCreate(comm, &snes_Pi_r));
    PetscCall(SNESSetSolution(snes_Pi_r, X_Pi_r_loc));

    // -- Jacobian function
    PetscCall(SNESSetJacobian(snes_Pi_r, mat_S_Pi_r, mat_S_Pi_r, NULL, NULL));

    // -- Residual evaluation function
    PetscCall(SNESSetFunction(snes_Pi_r, X_Pi_r_loc, FormResidual_BDDCElementSchur, bddc_ctx));
  }

  // Set up MatShell user data
  {
    apply_ctx->comm  = comm;
    apply_ctx->dm    = dm;
    apply_ctx->X_loc = X_loc;
    PetscCall(VecDuplicate(X_loc, &apply_ctx->Y_loc));
    apply_ctx->x_ceed = ceed_data->x_ceed;
    apply_ctx->y_ceed = ceed_data->y_ceed;
    apply_ctx->op     = ceed_data->op_apply;
    apply_ctx->ceed   = ceed;
  }

  // Set up PCShell user data (also used for Schur operators)
  {
    bddc_ctx->comm  = comm;
    bddc_ctx->dm    = dm;
    bddc_ctx->dm_Pi = dm_Pi;
    bddc_ctx->X_loc = X_loc;
    bddc_ctx->Y_loc = apply_ctx->Y_loc;
    bddc_ctx->X_Pi  = X_Pi;
    PetscCall(VecDuplicate(X_Pi, &bddc_ctx->Y_Pi));
    bddc_ctx->X_Pi_loc = X_Pi_loc;
    PetscCall(VecDuplicate(X_Pi_loc, &bddc_ctx->Y_Pi_loc));
    bddc_ctx->X_Pi_r_loc = X_Pi_r_loc;
    PetscCall(VecDuplicate(X_Pi_r_loc, &bddc_ctx->Y_Pi_r_loc));
    bddc_ctx->ceed_data_bddc = ceed_data_bddc;
    bddc_ctx->mat_S_Pi       = mat_S_Pi;
    bddc_ctx->mat_S_Pi_r     = mat_S_Pi_r;
    PetscCall(KSPCreate(comm, &ksp_S_Pi));
    PetscCall(KSPCreate(comm, &ksp_S_Pi_r));
    bddc_ctx->ksp_S_Pi    = ksp_S_Pi;
    bddc_ctx->ksp_S_Pi_r  = ksp_S_Pi_r;
    bddc_ctx->snes_Pi     = snes_Pi;
    bddc_ctx->snes_Pi_r   = snes_Pi_r;
    bddc_ctx->is_harmonic = injection == INJECTION_HARMONIC;
  }

  // Set up KSP
  PetscCall(KSPCreate(comm, &ksp));
  {
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPSetNormType(ksp, KSP_NORM_NATURAL));
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetOperators(ksp, mat_O, mat_O));

  // Set up PCShell
  PetscCall(KSPGetPC(ksp, &pc));
  {
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetContext(pc, bddc_ctx));
    PetscCall(PCShellSetApply(pc, PCShellApply_BDDC));
    PetscCall(PCShellSetSetUp(pc, PCShellSetup_BDDC));

    // Set up Schur compilemnt solvers
    {
      // -- Vertex mesh AMG
      PC pc_S_Pi;
      PetscCall(KSPSetType(ksp_S_Pi, KSPPREONLY));
      PetscCall(KSPSetOperators(ksp_S_Pi, mat_S_Pi, mat_S_Pi));

      PetscCall(KSPGetPC(ksp_S_Pi, &pc_S_Pi));
      PetscCall(PCSetType(pc_S_Pi, PCGAMG));

      PetscCall(KSPSetOptionsPrefix(ksp_S_Pi, "S_Pi_"));
      PetscCall(PCSetOptionsPrefix(pc_S_Pi, "S_Pi_"));
      PetscCall(KSPSetFromOptions(ksp_S_Pi));
      PetscCall(PCSetFromOptions(pc_S_Pi));
    }
    {
      // -- Broken mesh AMG
      PC pc_S_Pi_r;
      PetscCall(KSPSetType(ksp_S_Pi_r, KSPPREONLY));
      PetscCall(KSPSetOperators(ksp_S_Pi_r, mat_S_Pi_r, mat_S_Pi_r));

      PetscCall(KSPGetPC(ksp_S_Pi_r, &pc_S_Pi_r));
      PetscCall(PCSetType(pc_S_Pi_r, PCGAMG));

      PetscCall(KSPSetOptionsPrefix(ksp_S_Pi_r, "S_Pi_r_"));
      PetscCall(PCSetOptionsPrefix(pc_S_Pi_r, "S_Pi_r_"));
      PetscCall(KSPSetFromOptions(ksp_S_Pi_r));
      PetscCall(PCSetFromOptions(pc_S_Pi_r));
    }
  }

  // First run, if benchmarking
  if (benchmark_mode) {
    PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 1));
    PetscCall(VecZeroEntries(X));
    my_rt_start = MPI_Wtime();
    PetscCall(KSPSolve(ksp, rhs, X));
    my_rt = MPI_Wtime() - my_rt_start;
    PetscCall(MPI_Allreduce(MPI_IN_PLACE, &my_rt, 1, MPI_DOUBLE, MPI_MIN, comm));
    // Set maxits based on first iteration timing
    if (my_rt > 0.02) {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 5));
    } else {
      PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 20));
    }
  }

  // Timed solve
  PetscCall(VecZeroEntries(X));
  PetscCall(PetscBarrier((PetscObject)ksp));

  // -- Performance logging
  PetscCall(PetscLogStageRegister("Solve Stage", &solve_stage));
  PetscCall(PetscLogStagePush(solve_stage));

  // -- Solve
  my_rt_start = MPI_Wtime();
  PetscCall(KSPSolve(ksp, rhs, X));
  my_rt = MPI_Wtime() - my_rt_start;

  // -- Performance logging
  PetscCall(PetscLogStagePop());

  // Output results
  {
    KSPType            ksp_type;
    KSPConvergedReason reason;
    PCType             pc_type;
    PetscReal          rnorm;
    PetscInt           its;
    PetscCall(KSPGetType(ksp, &ksp_type));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(PCGetType(pc, &pc_type));
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      PetscCall(PetscPrintf(comm,
                            "  KSP:\n"
                            "    KSP Type                           : %s\n"
                            "    KSP Convergence                    : %s\n"
                            "    Total KSP Iterations               : %" PetscInt_FMT "\n"
                            "    Final rnorm                        : %e\n",
                            ksp_type, KSPConvergedReasons[reason], its, (double)rnorm));
      PetscCall(PetscPrintf(comm,
                            "  BDDC:\n"
                            "    PC Type                            : %s\n",
                            pc_type));
    }
    if (!test_mode) {
      PetscCall(PetscPrintf(comm, "  Performance:\n"));
    }
    {
      // Set up error operator context
      PetscCall(PetscNew(&error_ctx));
      PetscCall(SetupErrorOperatorCtx(comm, dm, ceed, ceed_data, X_loc, op_error, error_ctx));
      PetscScalar l2_error;
      PetscCall(ComputeL2Error(X, &l2_error, error_ctx));
      PetscReal tol = 5e-2;
      if (!test_mode || l2_error > tol) {
        PetscCall(MPI_Allreduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, comm));
        PetscCall(MPI_Allreduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, comm));
        PetscCall(PetscPrintf(comm,
                              "    L2 Error                           : %e\n"
                              "    CG Solve Time                      : %g (%g) sec\n",
                              (double)l2_error, rt_max, rt_min));
      }
    }
    if (benchmark_mode && (!test_mode)) {
      PetscCall(PetscPrintf(comm, "    DoFs/Sec in CG                     : %g (%g) million\n", 1e-6 * g_size * its / rt_max,
                            1e-6 * g_size * its / rt_min));
    }
  }

  if (write_solution) {
    PetscViewer vtk_viewer_soln;

    PetscCall(PetscViewerCreate(comm, &vtk_viewer_soln));
    PetscCall(PetscViewerSetType(vtk_viewer_soln, PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetName(vtk_viewer_soln, "solution.vtu"));
    PetscCall(VecView(X, vtk_viewer_soln));
    PetscCall(PetscViewerDestroy(&vtk_viewer_soln));
  }

  // Cleanup
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X_loc));
  PetscCall(VecDestroy(&apply_ctx->Y_loc));
  PetscCall(VecDestroy(&X_Pi));
  PetscCall(VecDestroy(&bddc_ctx->Y_Pi));
  PetscCall(VecDestroy(&X_Pi_loc));
  PetscCall(VecDestroy(&bddc_ctx->Y_Pi_loc));
  PetscCall(VecDestroy(&X_Pi_r_loc));
  PetscCall(VecDestroy(&bddc_ctx->Y_Pi_r_loc));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&rhs_loc));
  PetscCall(MatDestroy(&mat_O));
  PetscCall(MatDestroy(&mat_S_Pi));
  PetscCall(MatDestroy(&mat_S_Pi_r));
  PetscCall(PetscFree(apply_ctx));
  PetscCall(PetscFree(error_ctx));
  PetscCall(PetscFree(bddc_ctx));
  PetscCall(CeedDataDestroy(0, ceed_data));
  PetscCall(CeedDataBDDCDestroy(ceed_data_bddc));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dm_Pi));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(KSPDestroy(&ksp_S_Pi));
  PetscCall(KSPDestroy(&ksp_S_Pi_r));
  PetscCall(SNESDestroy(&snes_Pi));
  PetscCall(SNESDestroy(&snes_Pi_r));
  CeedVectorDestroy(&target);
  CeedQFunctionDestroy(&qf_error);
  CeedQFunctionDestroy(&qf_restrict);
  CeedQFunctionDestroy(&qf_prolong);
  CeedOperatorDestroy(&op_error);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
