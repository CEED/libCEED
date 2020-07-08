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
/// Helper setup functions for shallow-water example using PETSc

#include <stdbool.h>
#include <string.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <ceed.h>
#include "../sw_headers.h"               // Function prototytes
#include "../qfunctions/setup_geo.h"     // Geometric factors
#include "../qfunctions/advection.h"     // Physics point-wise functions
#include "../qfunctions/geostrophic.h"  // Physics point-wise functions

#if PETSC_VERSION_LT(3,14,0)
#  define DMPlexGetClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexGetClosureIndices(a,b,c,d,f,g,i)
#  define DMPlexRestoreClosureIndices(a,b,c,d,e,f,g,h,i) DMPlexRestoreClosureIndices(a,b,c,d,f,g,i)
#endif

problemData problemOptions[] = {
  [SWE_ADVECTION] = {
    .topodim                = 2,
    .qdatasize              = 11,
    .setup                  = SetupGeo,
    .setup_loc              = SetupGeo_loc,
    .ics                    = ICsSW_Advection,
    .ics_loc                = ICsSW_Advection_loc,
    .apply_explfunction     = SWExplicit_Advection,
    .apply_explfunction_loc = SWExplicit_Advection_loc,
    .apply_implfunction     = SWImplicit_Advection,
    .apply_implfunction_loc = SWImplicit_Advection_loc,
    .apply_jacobian         = SWJacobian_Advection,
    .apply_jacobian_loc     = SWJacobian_Advection_loc,
    .non_zero_time          = PETSC_TRUE
  },
  [SWE_GEOSTROPHIC] = {
    .topodim                = 2,
    .qdatasize              = 11,
    .setup                  = SetupGeo,
    .setup_loc              = SetupGeo_loc,
    .ics                    = ICsSW,
    .ics_loc                = ICsSW_loc,
    .apply_explfunction     = SWExplicit,
    .apply_explfunction_loc = SWExplicit_loc,
    .apply_implfunction     = SWImplicit,
    .apply_implfunction_loc = SWImplicit_loc,
    .apply_jacobian         = SWJacobian,
    .apply_jacobian_loc     = SWJacobian_loc,
    .non_zero_time          = PETSC_FALSE
  }
};

// -----------------------------------------------------------------------------
// Auxiliary function to create PETSc FE space for a given degree
// -----------------------------------------------------------------------------

PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool isSimplex,
                                     const char prefix[],
                                     PetscInt order, PetscFE *fem) {
  PetscQuadrature q, fq;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        quadPointsPerEdge;
  PetscBool       tensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create space */
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) dm), &P); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) P, prefix); CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor); CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc); CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim); CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order); CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P); CHKERRQ(ierr);
  ierr = PetscSpacePolynomialGetTensor(P, &tensor); CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) dm), &Q);
  CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q,PETSCDUALSPACELAGRANGE); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) Q, prefix); CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, isSimplex, &K); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K); CHKERRQ(ierr);
  ierr = DMDestroy(&K); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order); CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q); CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q); CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PetscObjectComm((PetscObject) dm), fem); CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) *fem, prefix); CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(*fem); CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fem, P); CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fem, Q); CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fem, Nc); CHKERRQ(ierr);
  ierr = PetscFESetUp(*fem); CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P); CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q); CHKERRQ(ierr);
  /* Create quadrature */
  quadPointsPerEdge = PetscMax(order + 1,1);
  if (isSimplex) {
    ierr = PetscDTStroudConicalQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0,
                                          &q); CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
                                          &fq); CHKERRQ(ierr);
  } else {
    ierr = PetscDTGaussTensorQuadrature(dim,   1, quadPointsPerEdge, -1.0, 1.0,
                                        &q); CHKERRQ(ierr);
    ierr = PetscDTGaussTensorQuadrature(dim-1, 1, quadPointsPerEdge, -1.0, 1.0,
                                        &fq); CHKERRQ(ierr);
  }
  ierr = PetscFESetQuadrature(*fem, q); CHKERRQ(ierr);
  ierr = PetscFESetFaceQuadrature(*fem, fq); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q); CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&fq); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Auxiliary function to setup DM FE space and info
// -----------------------------------------------------------------------------

PetscErrorCode SetupDM(DM dm, PetscInt degree, PetscInt ncompq,
                       PetscInt dim) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    // Configure the finite element space
    PetscFE fe;
    ierr = PetscFECreateByDegree(dm, dim, ncompq, PETSC_FALSE, NULL, degree,
                                 &fe);
    ierr = PetscObjectSetName((PetscObject)fe, "Q"); CHKERRQ(ierr);
    ierr = DMAddField(dm,NULL,(PetscObject)fe); CHKERRQ(ierr);
    ierr = DMCreateDS(dm); CHKERRQ(ierr);

    ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
    CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  }
  {
    // Empty name for conserved field (because there is only one field)
    PetscSection section;
    ierr = DMGetLocalSection(dm, &section); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(section, 0, ""); CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 0, "u_lambda");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 1, "u_theta");
    CHKERRQ(ierr);
    ierr = PetscSectionSetComponentName(section, 0, 2, "h");
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// PETSc sphere auxiliary function
// -----------------------------------------------------------------------------

// Utility function taken from petsc/src/dm/impls/plex/examples/tutorials/ex7.c
PetscErrorCode ProjectToUnitSphere(DM dm) {
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt       Nv, v, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetCoordinatesLocal(dm, &coordinates); CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordinates, &Nv); CHKERRQ(ierr);
  ierr = VecGetBlockSize(coordinates, &dim); CHKERRQ(ierr);
  Nv  /= dim;
  ierr = VecGetArray(coordinates, &coords); CHKERRQ(ierr);
  for (v = 0; v < Nv; ++v) {
    PetscReal r = 0.0;

    for (d = 0; d < dim; ++d) r += PetscSqr(PetscRealPart(coords[v*dim+d]));
    r = PetscSqrtReal(r);
    for (d = 0; d < dim; ++d) coords[v*dim+d] /= r;
  }
  ierr = VecRestoreArray(coordinates, &coords); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Auxiliary function to define CEED restrictions from DMPlex data
// -----------------------------------------------------------------------------

PetscErrorCode CreateRestrictionPlex(Ceed ceed, DM dm, CeedInt P,
                                     CeedInt ncomp,
                                     CeedElemRestriction *Erestrict) {
  PetscInt ierr;
  PetscInt c, cStart, cEnd, nelem, nnodes, *erestrict, eoffset;
  PetscSection section;
  Vec Uloc;

  PetscFunctionBegin;

  // Get Nelem
  ierr = DMGetSection(dm, &section); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart,& cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;

  // Get indices
  ierr = PetscMalloc1(nelem*P*P, &erestrict); CHKERRQ(ierr);
  for (c=cStart, eoffset = 0; c<cEnd; c++) {
    PetscInt numindices, *indices, i;
    ierr = DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE,
                                   &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i] + (PetscInt)(copysign(j, indices[i])))
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                   "Cell %D closure indices not interlaced", c);
      }
      // NO BC on closed surfaces
      PetscInt loc = indices[i];
      erestrict[eoffset++] = loc;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE,
                                       &numindices, &indices, NULL, NULL);
    CHKERRQ(ierr);
  }
  if (eoffset != nelem*P*P) SETERRQ3(PETSC_COMM_SELF,
        PETSC_ERR_LIB, "ElemRestriction of size (%D,%D) initialized %D nodes",
        nelem, P*P,eoffset);

  // Setup CEED restriction
  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &nnodes); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, nelem, P*P, ncomp, 1, nnodes,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Auxiliary function to create a CeedVector from PetscVec of same size
// -----------------------------------------------------------------------------

PetscErrorCode CreateVectorFromPetscVec(Ceed ceed, Vec p,
                                        CeedVector *v) {
  PetscErrorCode ierr;
  PetscInt m;

  PetscFunctionBeginUser;
  ierr = VecGetLocalSize(p, &m); CHKERRQ(ierr);
  ierr = CeedVectorCreate(ceed, m, v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Auxiliary function to place a PetscVec into a CeedVector of same size
// -----------------------------------------------------------------------------

PetscErrorCode VectorPlacePetscVec(CeedVector c, Vec p) {
  PetscErrorCode ierr;
  PetscInt mceed,mpetsc;
  PetscScalar *a;

  PetscFunctionBeginUser;
  ierr = CeedVectorGetLength(c, &mceed); CHKERRQ(ierr);
  ierr = VecGetLocalSize(p, &mpetsc); CHKERRQ(ierr);
  if (mceed != mpetsc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,
                                  "Cannot place PETSc Vec of length %D in CeedVector of length %D",
                                  mpetsc, mceed);
  ierr = VecGetArray(p, &a); CHKERRQ(ierr);
  CeedVectorSetArray(c, CEED_MEM_HOST, CEED_USE_POINTER, a);
  PetscFunctionReturn(0);
}

// -----------------------------------------------------------------------------
// Auxiliary function to set up libCEED objects for a given degree
// -----------------------------------------------------------------------------

PetscErrorCode SetupLibceed(DM dm, Ceed ceed, CeedInt degree, CeedInt qextra,
                            PetscInt ncompx, PetscInt ncompq, User user, 
                            CeedData data, problemData *problem,
                            PhysicsContext phys_ctx, ProblemContext probl_ctx) {
  int ierr;
  DM dmcoord;
  Vec Xloc;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction Erestrictx, Erestrictq, Erestrictqdi;
  CeedQFunction qf_setup, qf_ics, qf_explicit, qf_implicit,
                qf_jacobian;
  CeedOperator op_setup, op_ics, op_explicit, op_implicit,
               op_jacobian;
  CeedVector xcorners, qdata, q0ceed;
  CeedInt P, Q, cStart, cEnd, nelem, qdatasize = problem->qdatasize, 
          topodim = problem->topodim;;

  // CEED bases
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompq, P, Q,
                                  CEED_GAUSS, &basisq);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, Q,
                                  CEED_GAUSS, &basisx);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, P,
                                  CEED_GAUSS_LOBATTO, &basisxc);

  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // CEED restrictions
  ierr = CreateRestrictionPlex(ceed, dm, P, ncompq, &Erestrictq); CHKERRQ(ierr);
  ierr = CreateRestrictionPlex(ceed, dmcoord, 2, ncompx, &Erestrictx);
  CHKERRQ(ierr);

  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  nelem = cEnd - cStart;
  CeedElemRestrictionCreateStrided(ceed, nelem, Q*Q, qdatasize,
                                   qdatasize*nelem*Q*Q,
                                   CEED_STRIDES_BACKEND, &Erestrictqdi);

  // Element coordinates
  ierr = DMGetCoordinatesLocal(dm, &Xloc); CHKERRQ(ierr);
  ierr = CreateVectorFromPetscVec(ceed, Xloc, &xcorners); CHKERRQ(ierr);

  // Create the persistent vectors that will be needed in setup and apply
  CeedInt nqpts;
  CeedBasisGetNumQuadraturePoints(basisq, &nqpts);
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &qdata);
  CeedElemRestrictionCreateVector(Erestrictq, &q0ceed, NULL);
  user->q0ceed = q0ceed;

  // Create the Q-Function that builds the quadrature data
  CeedQFunctionCreateInterior(ceed, 1, problem->setup, problem->setup_loc, 
                              &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "dx", ncompx*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "qdata", qdatasize, CEED_EVAL_NONE);

  // Create the Q-Function that sets the ICs of the operator
  CeedQFunctionCreateInterior(ceed, 1, problem->ics, problem->ics_loc, &qf_ics);
  CeedQFunctionAddInput(qf_ics, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_ics, "q0", ncompq, CEED_EVAL_NONE);

  // Create the Q-Function that defines the explicit part of the PDE operator
  CeedQFunctionCreateInterior(ceed, 1, problem->apply_explfunction, 
                              problem->apply_explfunction_loc, &qf_explicit);
  CeedQFunctionAddInput(qf_explicit, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_explicit, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_explicit, "dq", ncompq*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_explicit, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_explicit, "v", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_explicit, "dv", ncompq*topodim, CEED_EVAL_GRAD);

  // Create the Q-Function that defines the implicit part of the PDE operator
  CeedQFunctionCreateInterior(ceed, 1, problem->apply_implfunction, 
                              problem->apply_implfunction_loc, &qf_implicit);
  CeedQFunctionAddInput(qf_implicit, "q", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_implicit, "dq", ncompq*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_implicit, "qdot", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_implicit, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_implicit, "x", ncompx, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_implicit, "v", ncompq, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_implicit, "dv", ncompq*topodim, CEED_EVAL_GRAD);

  // Create the Q-Function that defines the action of the Jacobian operator
  CeedQFunctionCreateInterior(ceed, 1, problem->apply_jacobian, 
                              problem->apply_jacobian_loc, &qf_jacobian);
  CeedQFunctionAddInput(qf_jacobian, "q", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_jacobian, "deltaq", 3, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_jacobian, "qdata", 10, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_jacobian, "dv", 3, CEED_EVAL_GRAD);

  // Create the operator that builds the quadrature data for the operator
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, basisx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, basisx,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basisx,
                       CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the operator that sets the ICs
  CeedOperatorCreate(ceed, qf_ics, NULL, NULL, &op_ics);
  CeedOperatorSetField(op_ics, "x", Erestrictx, basisxc, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_ics, "q0", Erestrictq,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedElemRestrictionCreateVector(Erestrictq, &user->qceed, NULL);
  CeedElemRestrictionCreateVector(Erestrictq, &user->qdotceed, NULL);
  CeedElemRestrictionCreateVector(Erestrictq, &user->gceed, NULL);
  CeedElemRestrictionCreateVector(Erestrictq, &user->fceed, NULL);
  CeedElemRestrictionCreateVector(Erestrictq, &user->jceed, NULL);

  // Create the explicit part of the PDE operator
  CeedOperatorCreate(ceed, qf_explicit, NULL, NULL, &op_explicit);
  CeedOperatorSetField(op_explicit, "x", Erestrictx, basisx, xcorners);
  CeedOperatorSetField(op_explicit, "q", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "dq", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_explicit, "v", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_explicit, "dv", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  user->op_explicit = op_explicit;

  // Create the implicit part of the PDE operator
  CeedOperatorCreate(ceed, qf_implicit, NULL, NULL, &op_implicit);
  CeedOperatorSetField(op_implicit, "q", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "dq", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "qdot", Erestrictq, basisq, user->qdotceed);
  CeedOperatorSetField(op_implicit, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_implicit, "x", Erestrictx, basisx, xcorners);
  CeedOperatorSetField(op_implicit, "v", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_implicit, "dv", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  user->op_implicit = op_implicit;

  // Create the Jacobian of the PDE operator
  CeedOperatorCreate(ceed, qf_jacobian, NULL, NULL, &op_jacobian);
  CeedOperatorSetField(op_jacobian, "q", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_jacobian, "deltaq", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE); // TODO: Check restriction and basis for delta q
  CeedOperatorSetField(op_jacobian, "qdata", Erestrictqdi,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_jacobian, "dv", Erestrictq, basisq,
                       CEED_VECTOR_ACTIVE);
  user->op_jacobian = op_jacobian;

  // Set up the libCEED context
  CeedQFunctionSetContext(qf_ics, phys_ctx, sizeof *phys_ctx);
  CeedQFunctionSetContext(qf_explicit, probl_ctx, sizeof *probl_ctx);
  CeedQFunctionSetContext(qf_implicit, probl_ctx, sizeof *probl_ctx);
  CeedQFunctionSetContext(qf_jacobian, probl_ctx, sizeof *probl_ctx);

  // Save libCEED data required for level // TODO: check how many of these are really needed outside
  data->basisx = basisx;
  data->basisq = basisq;
  data->Erestrictx = Erestrictx;
  data->Erestrictq = Erestrictq;
  data->Erestrictqdi = Erestrictqdi;
  data->qf_setup = qf_setup;
  data->qf_ics = qf_ics;
  data->qf_explicit = qf_explicit;
  data->qf_implicit = qf_implicit;
  data->qf_jacobian = qf_jacobian;
  data->op_setup = op_setup;
  data->op_ics = op_ics;
  data->op_explicit = op_explicit;
  data->op_implicit = op_implicit;
  data->op_jacobian = op_jacobian;
  data->qdata = qdata;
  data->xcorners = xcorners;

  PetscFunctionReturn(0);
}
