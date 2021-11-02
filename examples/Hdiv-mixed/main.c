/// @file
/// Test creation, use, and destruction of an element restriction for 2D quad Hdiv

// run with ./main
const char help[] = "Solve H(div)-mixed problem using PETSc and libCEED\n";

#include "main.h"

int main(int argc, char **argv) {
  // ---------------------------------------------------------------------------
  // Initialize PETSc
  // ---------------------------------------------------------------------------
  PetscInt ierr;
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  // ---------------------------------------------------------------------------
  // Create structs
  // ---------------------------------------------------------------------------
  AppCtx app_ctx;
  ierr = PetscCalloc1(1, &app_ctx); CHKERRQ(ierr);

  ProblemData *problem_data = NULL;
  ierr = PetscCalloc1(1, &problem_data); CHKERRQ(ierr);

  User user;
  ierr = PetscCalloc1(1, &user); CHKERRQ(ierr);

  CeedData ceed_data;
  ierr = PetscCalloc1(1, &ceed_data); CHKERRQ(ierr);

  Physics phys_ctx;
  ierr = PetscCalloc1(1, &phys_ctx); CHKERRQ(ierr);

  user->app_ctx = app_ctx;
  user->phys    = phys_ctx;

  // ---------------------------------------------------------------------------
  // Process command line options
  // ---------------------------------------------------------------------------
  // -- Register problems to be available on the command line
  ierr = RegisterProblems_Hdiv(app_ctx); CHKERRQ(ierr);

  // -- Process general command line options
  MPI_Comm comm = PETSC_COMM_WORLD;
  ierr = ProcessCommandLineOptions(comm, app_ctx); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Choose the problem from the list of registered problems
  // ---------------------------------------------------------------------------
  {
    PetscErrorCode (*p)(ProblemData *, void *);
    ierr = PetscFunctionListFind(app_ctx->problems, app_ctx->problem_name, &p);
    CHKERRQ(ierr);
    if (!p) SETERRQ1(PETSC_COMM_SELF, 1, "Problem '%s' not found",
                       app_ctx->problem_name);
    ierr = (*p)(problem_data, &user); CHKERRQ(ierr);
  }

  // ---------------------------------------------------------------------------
  // Initialize libCEED
  // ---------------------------------------------------------------------------
  // -- Initialize backend
  Ceed ceed;
  CeedInit("/cpu/self/ref/serial", &ceed);
  CeedMemType mem_type_backend;
  CeedGetPreferredMemType(ceed, &mem_type_backend);
  // ---------------------------------------------------------------------------
  // Set-up DM
  // ---------------------------------------------------------------------------
  // PETSc objects
  DM             dm;
  VecType        vec_type;
  ierr = CreateDistributedDM(comm, &dm); CHKERRQ(ierr);
  ierr = DMGetVecType(dm, &vec_type); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Create global, local solution, local rhs vector
  // ---------------------------------------------------------------------------
  Vec            U_g, U_loc;
  PetscInt       U_l_size, U_g_size, U_loc_size;
  // Create global and local solution vectors
  ierr = DMCreateGlobalVector(dm, &U_g); CHKERRQ(ierr);
  ierr = VecGetSize(U_g, &U_g_size); CHKERRQ(ierr);
  // Local size for matShell
  ierr = VecGetLocalSize(U_g, &U_l_size); CHKERRQ(ierr);
  // Create local unknown vector U_loc
  ierr = DMCreateLocalVector(dm, &U_loc); CHKERRQ(ierr);
  // Local size for libCEED
  ierr = VecGetSize(U_loc, &U_loc_size); CHKERRQ(ierr);

  // Operator
  Mat mat;
  ierr = MatCreateShell(comm, U_l_size, U_l_size, U_g_size, U_g_size,
                        user, &mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_MULT,
                              (void(*)(void))MatMult_Ceed); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_GET_DIAGONAL,
                              (void(*)(void))MatGetDiag); CHKERRQ(ierr);
  ierr = MatShellSetVecType(mat, vec_type); CHKERRQ(ierr);
  // Get RHS vector
  Vec rhs_loc;
  PetscScalar *r;
  CeedVector rhs_ceed, target;
  PetscMemType mem_type;
  ierr = VecDuplicate(U_loc, &rhs_loc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs_loc); CHKERRQ(ierr);
  ierr = VecGetArrayAndMemType(rhs_loc, &r, &mem_type); CHKERRQ(ierr);
  CeedVectorCreate(ceed, U_l_size, &rhs_ceed);
  CeedVectorSetArray(rhs_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, r);

  // ---------------------------------------------------------------------------
  // Setup libCEED
  // ---------------------------------------------------------------------------
  // -- Set up libCEED objects
  ierr = SetupLibceed(dm, ceed, app_ctx, problem_data, U_g_size,
                      U_loc_size, ceed_data, rhs_ceed, &target);
  CHKERRQ(ierr);
  // -- Set up context for QFunctions
  ierr = problem_data->setup_ctx(ceed, ceed_data, phys_ctx);
  CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Gather RHS
  // ---------------------------------------------------------------------------
  Vec rhs;
  ierr = VecDuplicate(U_g, &rhs); CHKERRQ(ierr);
  CeedVectorTakeArray(rhs_ceed, MemTypeP2C(mem_type), NULL);
  ierr = VecRestoreArrayAndMemType(rhs_loc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, rhs_loc, ADD_VALUES, rhs); CHKERRQ(ierr);
  // ---------------------------------------------------------------------------
  // Setup Mat, KSP
  // ---------------------------------------------------------------------------
  user->comm = comm;
  user->dm = dm;
  user->X_loc = U_loc;
  ierr = VecDuplicate(U_loc, &user->Y_loc); CHKERRQ(ierr);
  user->x_ceed = ceed_data->x_ceed;
  user->y_ceed = ceed_data->y_ceed;
  user->op = ceed_data->op_residual;
  user->ceed = ceed;
  // Ksp
  KSP ksp;
  PC pc;
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
  ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
  ierr = KSPSetNormType(ksp, KSP_NORM_NATURAL); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                          PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, U_g); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // Output results
  KSPType ksp_type;
  KSPConvergedReason reason;
  PetscReal rnorm;
  PetscInt its;
  ierr = KSPGetType(ksp, &ksp_type); CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,
                     "  KSP:\n"
                     "    KSP Type                           : %s\n"
                     "    KSP Convergence                    : %s\n"
                     "    Total KSP Iterations               : %D\n"
                     "    Final rnorm                        : %e\n",
                     ksp_type, KSPConvergedReasons[reason], its,
                     (double)rnorm); CHKERRQ(ierr);

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------

  // Free PETSc objects
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&U_g); CHKERRQ(ierr);
  ierr = VecDestroy(&U_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Y_loc); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhs_loc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  // -- Function list
  ierr = PetscFunctionListDestroy(&app_ctx->problems); CHKERRQ(ierr);

  // -- Structs
  ierr = PetscFree(app_ctx); CHKERRQ(ierr);
  ierr = PetscFree(problem_data); CHKERRQ(ierr);
  ierr = PetscFree(user); CHKERRQ(ierr);
  ierr = PetscFree(phys_ctx->pq2d_ctx); CHKERRQ(ierr);
  ierr = PetscFree(phys_ctx); CHKERRQ(ierr);
  // Free libCEED objects

  CeedVectorDestroy(&rhs_ceed);
  CeedVectorDestroy(&target);
  ierr = CeedDataDestroy(ceed_data); CHKERRQ(ierr);
  CeedDestroy(&ceed);

  return PetscFinalize();
}
