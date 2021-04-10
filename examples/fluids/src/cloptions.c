
#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Process command line options
// -----------------------------------------------------------------------------
// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx) {

  PetscErrorCode ierr;
  PetscBool ceed_flag = PETSC_FALSE;
  PetscBool problem_flag = PETSC_FALSE;

  PetscFunctionBeginUser;

  ierr = PetscOptionsBegin(comm, NULL, "Navier-Stokes in PETSc with libCEED",
                           NULL); CHKERRQ(ierr);

  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, app_ctx->ceedresource, app_ctx->ceedresource,
                            sizeof(app_ctx->ceedresource), &ceed_flag); CHKERRQ(ierr);
  
  app_ctx->test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test", "Run in test mode",
                          NULL, app_ctx->test_mode, &app_ctx->test_mode, NULL); CHKERRQ(ierr);
  
  app_ctx->test_tol = 1E-11;
  ierr = PetscOptionsScalar("-compare_final_state_atol",
                            "Test absolute tolerance",
                            NULL, app_ctx->test_tol, &app_ctx->test_tol, NULL); CHKERRQ(ierr);
  
  ierr = PetscOptionsString("-compare_final_state_filename", "Test filename",
                            NULL, app_ctx->filepath, app_ctx->filepath,
                            sizeof(app_ctx->filepath), NULL); CHKERRQ(ierr);
  
  ierr = PetscOptionsFList("-problem", "Problem to solve", NULL, app_ctx->problems,
                           app_ctx->problemName, app_ctx->problemName, sizeof(app_ctx->problemName),
                           &problem_flag); CHKERRQ(ierr);
  
  app_ctx->viz_refine = 0;
  ierr = PetscOptionsInt("-viz_refine",
                         "Regular refinement levels for visualization",
                         NULL, app_ctx->viz_refine, &app_ctx->viz_refine, NULL); CHKERRQ(ierr);
  
  app_ctx->outputfreq = 10;
  ierr = PetscOptionsInt("-output_freq",
                         "Frequency of output, in number of steps",
                         NULL, app_ctx->outputfreq, &app_ctx->outputfreq, NULL); CHKERRQ(ierr);

  app_ctx->contsteps = 0;
  ierr = PetscOptionsInt("-continue", "Continue from previous solution",
                         NULL, app_ctx->contsteps, &app_ctx->contsteps, NULL); CHKERRQ(ierr);
  
  app_ctx->degree = 2;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of finite elements",
                         NULL, app_ctx->degree, &app_ctx->degree, NULL); CHKERRQ(ierr);
  
  app_ctx->qextra = 2;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, app_ctx->qextra, &app_ctx->qextra, NULL); CHKERRQ(ierr);
  
  app_ctx->qextraSur = 2;
  ierr = PetscOptionsInt("-qextra_boundary",
                         "Number of extra quadrature points on in/outflow faces",
                         NULL, app_ctx->qextraSur, &app_ctx->qextraSur, NULL);
  CHKERRQ(ierr);

  ierr = PetscStrncpy(app_ctx->outputdir, ".", 2); CHKERRQ(ierr);
  ierr = PetscOptionsString("-output_dir", "Output directory",
                            NULL, app_ctx->outputdir, app_ctx->outputdir,
                            sizeof(app_ctx->outputdir), NULL); CHKERRQ(ierr);

  // Provide default ceed resource if not specified
  if (!ceed_flag) {
    const char *ceedresource = "/cpu/self";
    strncpy(app_ctx->ceedresource, ceedresource, 10);
  }

  // Provide default problem if not specified
  if (!problem_flag) {
    const char *problemName = "density_current";
    strncpy(app_ctx->problemName, problemName, 16);
  }

  // todo
  //ierr = PetscOptionsEnum("-memtype",
  //                        "CEED MemType requested", NULL,
  //                        memTypes, (PetscEnum)app_ctx->memtyperequested,
  //                        (PetscEnum *)&app_ctx->memtyperequested, &app_ctx->setmemtyperequest);
  //CHKERRQ(ierr);
  
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}