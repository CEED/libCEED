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

/// @file
/// libCEED auxiliary wrapper functions for shallow-water example using PETSc

#include "../sw_headers.h"
#include "../qfunctions/mass.h"

// -----------------------------------------------------------------------------
// This froms the RHS of the IMEX ODE, given as F(t,Q,Q_t) = G(t,Q)
//   This function takes in a state vector Q and writes into G
// -----------------------------------------------------------------------------

PetscErrorCode FormRHSFunction_SW(TS ts, PetscReal t, Vec Q, Vec G,
                                  void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  PetscScalar *q, *g;
  Vec Qloc, Gloc;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Gloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Gloc); CHKERRQ(ierr);

  // TODO:
  // L-vector to E-vector:
  // Apply user-defined restriction with appropriate coordinate transformations to Qloc

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecGetArray(Gloc, &g); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->gceed, CEED_MEM_HOST, CEED_USE_POINTER, g);

  // Apply CEED operator
  CeedOperatorApply(user->op_explicit, user->qceed, user->gceed,
                    CEED_REQUEST_IMMEDIATE);

  // TODO:
  // E-vector to L-vector:
  // Apply transpose of user-defined restriction

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

// -----------------------------------------------------------------------------
// This forms the LHS of the IMEX ODE, given as F(t,Q,Qdot) = G(t,Q)
//   This function takes in the state vector Q and its derivative Qdot
//   and writes into F
// -----------------------------------------------------------------------------

PetscErrorCode FormIFunction_SW(TS ts, PetscReal t, Vec Q, Vec Qdot,
                                Vec F, void *userData) {
  PetscErrorCode ierr;
  User user = *(User *)userData;
  const PetscScalar *q, *qdot;
  PetscScalar *f;
  Vec Qloc, Qdotloc, Floc;

  // Global-to-local
  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Floc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qdotloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Qdot, INSERT_VALUES, Qdotloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Floc); CHKERRQ(ierr);

  // TODO:
  // L-vector to E-vector:
  // Apply user-defined restriction with appropriate coordinate transformations to Qloc

  // Ceed Vectors
  ierr = VecGetArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecGetArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecGetArray(Floc, &f); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)q);
  CeedVectorSetArray(user->qdotceed, CEED_MEM_HOST, CEED_USE_POINTER,
                     (PetscScalar *)qdot);
  CeedVectorSetArray(user->fceed, CEED_MEM_HOST, CEED_USE_POINTER, f);

  // Apply CEED operator
  CeedOperatorApply(user->op_implicit, user->qceed, user->fceed,
                    CEED_REQUEST_IMMEDIATE);

  // TODO:
  // E-vector to L-vector:
  // Apply transpose of user-defined restriction

  // Restore vectors
  ierr = VecRestoreArrayRead(Qloc, &q); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Qdotloc, &qdot); CHKERRQ(ierr);
  ierr = VecRestoreArray(Floc, &f); CHKERRQ(ierr);

  ierr = VecZeroEntries(F); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Floc, ADD_VALUES, F); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qdotloc); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Floc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// User provided Jacobian = dF/dQ + sigma dF/dQdot
// -----------------------------------------------------------------------------

PetscErrorCode FormJacobian_SW(TS ts, PetscReal t, Vec Q, Vec Qdot,
                               PetscReal sigma, Mat J, Mat Jpre,
                               void *userData) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  ierr = TSComputeIJacobianDefaultColor(ts, t, Q, Qdot, sigma, J, Jpre, NULL);
  CHKERRQ(ierr);

  // Jpre might be AIJ (e.g., when using coloring), so we need to assemble it
  ierr = MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// User provided wrapper function for MATOP_MULT MatShellOperation in PETSc
//   for action of Jacobian calculation. Uses libCEED to compute the
//   matrix-vector product:
//   Jvec = mat*Q

//   Input Parameters:
//   mat  - input matrix
//   Q    - input vector
//
//   Output Parameters:
//   Jvec - output vector
// -----------------------------------------------------------------------------

PetscErrorCode ApplyJacobian_SW(Mat mat, Vec Q, Vec JVec) {
  User user;
  PetscScalar *q, *j;
  Vec Qloc, Jloc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  MatShellGetContext(mat, &user);
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm, &Jloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  // Global-to-local
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(Jloc); CHKERRQ(ierr);

  // TODO:
  // L-vector to E-vector:
  // Apply user-defined restriction with appropriate coordinate transformations to Qloc

  // CEED vectors
  ierr = VecGetArrayRead(Qloc, (const PetscScalar **)&q); CHKERRQ(ierr);
  ierr = VecGetArray(Jloc, &j); CHKERRQ(ierr);
  CeedVectorSetArray(user->qceed, CEED_MEM_HOST, CEED_USE_POINTER, q);
  CeedVectorSetArray(user->jceed, CEED_MEM_HOST, CEED_USE_POINTER, j);

  // Apply the CEED operator for the dF/dQ terms
  CeedOperatorApply(user->op_jacobian, user->qceed, user->jceed,
                    CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(user->jceed, CEED_MEM_HOST);

  // TODO:
  // E-vector to L-vector:
  // Apply transpose of user-defined restriction

  // Restore PETSc vectors
  ierr = VecRestoreArrayRead(Qloc, (const PetscScalar **)&q);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(Jloc, &j); CHKERRQ(ierr);

  // Local-to-global
  ierr = VecZeroEntries(JVec); CHKERRQ(ierr);
  ierr = DMLocalToGlobal(user->dm, Jloc, ADD_VALUES, JVec); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


// -----------------------------------------------------------------------------
// TS Monitor to print output
// -----------------------------------------------------------------------------

PetscErrorCode TSMonitor_SW(TS ts, PetscInt stepno, PetscReal time,
                            Vec Q, void *ctx) {
  User user = ctx;
  Vec Qloc;
  char filepath[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  PetscErrorCode ierr;

  // Set up output
  PetscFunctionBeginUser;
  // Print every 'outputfreq' steps
  if (stepno % user->outputfreq != 0)
    PetscFunctionReturn(0);
  ierr = DMGetLocalVector(user->dm, &Qloc); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)Qloc, "StateVec"); CHKERRQ(ierr);
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = DMGlobalToLocal(user->dm, Q, INSERT_VALUES, Qloc); CHKERRQ(ierr);

  // Output
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/swe-%03D.vtu",
                       user->outputfolder, stepno + user->contsteps);
  CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Q), filepath,
                            FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  ierr = VecView(Qloc, viewer); CHKERRQ(ierr);
  if (user->dmviz) {
    Vec Qrefined, Qrefined_loc;
    char filepath_refined[PETSC_MAX_PATH_LEN];
    PetscViewer viewer_refined;

    ierr = DMGetGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = DMGetLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Qrefined_loc, "Refined");
    CHKERRQ(ierr);
    ierr = MatInterpolate(user->interpviz, Q, Qrefined); CHKERRQ(ierr);
    ierr = VecZeroEntries(Qrefined_loc); CHKERRQ(ierr);
    ierr = DMGlobalToLocal(user->dmviz, Qrefined, INSERT_VALUES, Qrefined_loc);
    CHKERRQ(ierr);
    ierr = PetscSNPrintf(filepath_refined, sizeof filepath_refined,
                         "%s/swe-refined-%03D.vtu",
                         user->outputfolder, stepno + user->contsteps);
    CHKERRQ(ierr);
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)Qrefined),
                              filepath_refined,
                              FILE_MODE_WRITE, &viewer_refined); CHKERRQ(ierr);
    ierr = VecView(Qrefined_loc, viewer_refined); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user->dmviz, &Qrefined_loc); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(user->dmviz, &Qrefined); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer_refined); CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm, &Qloc); CHKERRQ(ierr);

  // Save data in a binary file for continuation of simulations
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/swe-solution.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
  CHKERRQ(ierr);
  ierr = VecView(Q, viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

  // Save time stamp
  // Dimensionalize time back
  time /= user->units->second;
  ierr = PetscSNPrintf(filepath, sizeof filepath, "%s/swe-time.bin",
                       user->outputfolder); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(user->comm, filepath, FILE_MODE_WRITE, &viewer);
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

// -----------------------------------------------------------------------------
// Auxiliary function to apply the ICs and eliminate repeated values in initial
//   state vector, arising from restriction
// -----------------------------------------------------------------------------

PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics,
    CeedVector xcorners, CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
    CeedElemRestriction restrictq, PhysicsContext ctxSetup, CeedScalar time) {
  PetscErrorCode ierr;
  CeedVector multlvec;
  Vec Multiplicity, MultiplicityLoc;

  ctxSetup->time = time;
  ierr = VecZeroEntries(Qloc); CHKERRQ(ierr);
  ierr = VectorPlacePetscVec(q0ceed, Qloc); CHKERRQ(ierr);

  // Apply IC operator
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

// -----------------------------------------------------------------------------
// Auxiliary function to compute the lumped mass matrix
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
