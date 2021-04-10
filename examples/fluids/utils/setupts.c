#include "../navierstokes.h"

// -----------------------------------------------------------------------------
// Time-stepping functions
// -----------------------------------------------------------------------------
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm,
                                       CeedElemRestriction restrictq, CeedBasis basisq,
                                       CeedElemRestriction restrictqdi, CeedVector qdata, Vec M) {
  PetscErrorCode ierr;
  CeedQFunction qf_mass;
  CeedOperator op_mass;
  CeedVector mceed;
  Vec Mloc;
  CeedInt ncompq, qdatasize;

  PetscFunctionBeginUser;
  CeedElemRestrictionGetNumComponents(restrictq, &ncompq);
  CeedElemRestrictionGetNumComponents(restrictqdi, &qdatasize);
  // Create the Q-function that defines the action of the mass operator
  CeedQFunctionCreateInterior(ceed, 1, Mass, Mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", ncompq, CEED_EVAL_INTERP);

  // Create the mass operator
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "q", restrictq, basisq, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "qdata", restrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_mass, "v", restrictq, basisq, CEED_VECTOR_ACTIVE);

  ierr = DMGetLocalVector(dm, &Mloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Mloc); CHKERRQ(ierr);
  CeedElemRestrictionCreateVector(restrictq, &mceed, NULL);
  ierr = VectorPlacePetscVec(mceed, Mloc); CHKERRQ(ierr);

  {
    // Compute a lumped mass matrix
    CeedVector onesvec;
    CeedElemRestrictionCreateVector(restrictq, &onesvec, NULL);
    CeedVectorSetValue(onesvec, 1.0);
    CeedOperatorApply(op_mass, onesvec, mceed, CEED_REQUEST_IMMEDIATE);
    CeedVectorDestroy(&onesvec);
    CeedOperatorDestroy(&op_mass);
    CeedVectorDestroy(&mceed);
  }
  CeedQFunctionDestroy(&qf_mass);

  ierr = VecZeroEntries(M); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Mloc, ADD_VALUES, M); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Mloc); CHKERRQ(ierr);

  // Invert diagonally lumped mass vector for RHS function
  ierr = VecReciprocal(M); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// RHS (Explicit time-stepper) function setup
//   This is the RHS of the ODE, given as u_t = G(t,u)
//   This function takes in a state vector Q and writes into G
PetscErrorCode RHS_NS(TS ts, PetscReal t, Vec Q, Vec G, void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  PetscScalar *q, *g;
  Vec Qloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  if (user->phys->hasCurrentTime) user->phys->ctxEulerData->currentTime = t;
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_rhs, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G); CHKERRQ(ierr);

  // Inverse of the lumped mass matrix
  ierr = VecPointwiseMult(G, G, user->M); // M is Minv
  CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Implicit time-stepper function setup
PetscErrorCode IFunction_NS(TS ts, PetscReal t, Vec Q, Vec Qdot, Vec G,
                            void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  const PetscScalar *q, *qdot;
  PetscScalar *g;
  Vec Qloc, Qdotloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  if (user->phys->hasCurrentTime) user->phys->ctxEulerData->currentTime = t;
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(user->dm, PETSC_TRUE, Qloc, 0.0,
                                    NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qdotloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Qdot, INSERT_VALUES, Qdotloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->qdotceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)qdot);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_ifunction, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecRestoreArray(Gloc, &g); CHKERRQ(ierr);

  ierr = VecZeroEntries(G); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Gloc, ADD_VALUES, G); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// User provided TS Monitor
PetscErrorCode TSMonitor_NS(TS ts, PetscInt stepno, PetscReal time,
                            Vec Q, void *ctx) {
  User user = ctx;
  Vec Qloc;
  char file_path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  // Set up output
  PetscFunctionBeginUser;
  // Print every 'output_freq' steps
  if (stepno % user->app_ctx->output_freq != 0)
    PetscFunctionReturn(0);
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Qloc, "StateVec"); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-%03D.vtu",
                       user->app_ctx->output_dir, stepno + user->app_ctx->cont_steps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), file_path,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(Qloc, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  if (user->dmviz) {
    Vec Qrefined, Qrefined_loc;
    char file_path_refined[PETSC_MAX_PATH_LEN];
    PetscViewer viewer_refined;

    ierr = DMGetGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Qrefined_loc, "Refined");
    CHKERRQ(ierr);
    ierr = MatInterpolate(user->interpviz, Q, Qrefined); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qrefined_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dmviz, Qrefined, INSERT_VALUES, Qrefined_loc);
    CHKERRQ(ierr);
    ierr = PetscSNPrintf(file_path_refined, sizeof file_path_refined,
                         "%s/nsrefined-%03D.vtu",
                         user->app_ctx->output_dir, stepno + user->app_ctx->cont_steps);
    CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Qrefined),
                              file_path_refined,
                              FILE_MODE_WRITE, &viewer_refined); CHKERRQ(ierr);
    ierr = VecView(Qrefined_loc, viewer_refined); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer_refined); CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);

  // Save data in a binary file for continuation of simulations
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-solution.bin",
                       user->app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Q, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save time stamp
  // Dimensionalize time back
  time /= user->units->second;
  ierr = PetscSNPrintf(file_path, sizeof file_path, "%s/ns-time.bin",
                       user->app_ctx->output_dir); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, file_path, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  #if PETSC_VERSION_GE(3,13,0)
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL);
  #else
  ierr = PetscViewerBinaryWrite(viewer, &time, 1, PETSC_REAL, true);
  #endif
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics, CeedVector xcorners,
                                   CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
                                   CeedElemRestriction restrictq,
                                   CeedQFunctionContext ctxSetup, CeedScalar time) {
  PetscErrorCode ierr;
  CeedVector multlvec;
  Vec Multiplicity, MultiplicityLoc;

  SetupContext ctxSetupData;
  CeedQFunctionContextGetData(ctxSetup, CEED_MEM_HOST, (void **)&ctxSetupData);
  ctxSetupData->time = time;
  CeedQFunctionContextRestoreData(ctxSetup, (void **)&ctxSetupData);

  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(q0ceed, Qloc); CHKERRQ(ierr);
  CeedOperatorApply(op_ics, xcorners, q0ceed, CEED_REQUEST_IMMEDIATE);
  ierr = VecZeroEntries(Q); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, Qloc, ADD_VALUES, Q); CHKERRQ(ierr);

  // Fix multiplicity for output of ICs
  ierr = DMGetLocalVector(dm, &MultiplicityLoc); CHKERRQ(ierr);
  CeedElemRestrictionCreateVector(restrictq, &multlvec, NULL);
  ierr = VectorPlacePetscVec(multlvec, MultiplicityLoc); CHKERRQ(ierr);
  CeedElemRestrictionGetMultiplicity(restrictq, multlvec);
  CeedVectorDestroy(&multlvec);
  ierr = DMGetGlobalVector(dm, &Multiplicity); CHKERRQ(ierr);
  ierr = VecZeroEntries(Multiplicity); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(dm, MultiplicityLoc, ADD_VALUES, Multiplicity);
  CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Q, Q, Multiplicity); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Qloc, Qloc, MultiplicityLoc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &MultiplicityLoc); CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &Multiplicity); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
