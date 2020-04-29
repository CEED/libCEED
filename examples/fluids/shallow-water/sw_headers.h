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

#ifndef sw_headers_h
#define sw_headers_h

#include <stdbool.h>
#include <string.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <ceed.h>

#ifndef physics_context_struct
#define physics_context_struct
typedef struct {
  CeedScalar u0;
  CeedScalar v0;
  CeedScalar h0;
  CeedScalar f;
  CeedScalar g;
  CeedScalar time;
} PhysicsContext_s;
typedef PhysicsContext_s *PhysicsContext;
#endif // physics_context_struct

// Problem specific data
typedef struct {
  CeedInt topodim, qdatasize;
  CeedQFunctionUser setup, ics, apply_explfunction, apply_implfunction;
  const char *setup_loc, *ics_loc, *apply_explfunction_loc,
  *apply_implfunction_loc;
  const bool non_zero_time;
} problemData;

// PETSc user data
typedef struct User_ *User;
typedef struct Units_ *Units;

struct User_ {
  MPI_Comm comm;
  PetscInt outputfreq;
  DM dm;
  DM dmviz;
  Mat interpviz;
  Ceed ceed;
  Units units;
  CeedVector qceed, q0ceed, qdotceed, fceed, gceed, jceed;
  CeedOperator op_explicit, op_implicit, op_jacobian;
  Vec M;
  char outputfolder[PETSC_MAX_PATH_LEN];
  PetscInt contsteps;
};

struct Units_ {
  // fundamental units
  PetscScalar meter;
  PetscScalar second;
  // derived unit
  PetscScalar mpersquareds;
};

// -----------------------------------------------------------------------------
// libCEED Data Struct
// -----------------------------------------------------------------------------

// libCEED data struct
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed ceed;
  CeedBasis basisx, basisxc, basisq;
  CeedElemRestriction Erestrictx, Erestrictq, Erestrictqdi;
  CeedQFunction qf_setup, qf_mass, qf_ics, qf_explicit, qf_implicit,
                qf_jacobian;
  CeedOperator op_setup, op_mass, op_ics, op_explicit, op_implicit, op_jacobian;
  CeedVector xcorners, xceed, qdata, q0ceed, mceed, hsceed, H0ceed;
};

// -----------------------------------------------------------------------------
// Setup DM functions
// -----------------------------------------------------------------------------

// Auxiliary function to create PETSc FE space for a given degree
PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool isSimplex, const char prefix[],
                                     PetscInt order, PetscFE *fem);

// Auxiliary function to setup DM FE space and info
PetscErrorCode SetupDM(DM dm, PetscInt degree, PetscInt ncompq, PetscInt dim);

// -----------------------------------------------------------------------------
// libCEED functions
// -----------------------------------------------------------------------------

// Auxiliary function to define CEED restrictions from DMPlex data
PetscErrorCode CreateRestrictionPlex(Ceed ceed, DM dm, CeedInt P, CeedInt ncomp,
                                     CeedElemRestriction *Erestrict);

// Auxiliary function to set up libCEED objects for a given degree
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, CeedInt degree,
                            CeedInt topodim, CeedInt qextra,
                            PetscInt ncompx, PetscInt ncompq,
                            User user, CeedData data,
                            PhysicsContext ctx);

// -----------------------------------------------------------------------------
// RHS (Explicit part in time-stepper) function setup
// -----------------------------------------------------------------------------

// This forms the RHS of the IMEX ODE, given as F(t,Q,Q_t) = G(t,Q)
PetscErrorCode FormRHSFunction_SW(TS ts, PetscReal t, Vec Q, Vec G,
                                  void *userData);

// -----------------------------------------------------------------------------
// Implicit part in time-stepper function setup
// -----------------------------------------------------------------------------

// This forms the LHS of the IMEX ODE, given as F(t,Q,Qdot) = G(t,Q)
PetscErrorCode FormIFunction_SW(TS ts, PetscReal t, Vec Q, Vec Qdot,
                                Vec F, void *userData);

// This forms the RHS of the IMEX ODE, given as F(t,Q,Q_t) = G(t,Q)
PetscErrorCode FormRHSFunction_SW(TS ts, PetscReal t, Vec Q, Vec G,
                                  void *userData);

// -----------------------------------------------------------------------------
// Jacobian setup and apply
// -----------------------------------------------------------------------------

PetscErrorCode FormJacobian_SW(TS ts, PetscReal t, Vec Q, Vec Qdot,
                               PetscReal sigma, Mat J, Mat Jpre,
                               void *userData);

PetscErrorCode ApplyJacobian_SW(Mat mat, Vec Q, Vec JVec);

// -----------------------------------------------------------------------------
// TS Monitor to print output
// -----------------------------------------------------------------------------

PetscErrorCode TSMonitor_SW(TS ts, PetscInt stepno, PetscReal time,
                            Vec Q, void *ctx);

// -----------------------------------------------------------------------------
// Miscellaneous utility functions
// -----------------------------------------------------------------------------

// Utility function to project refined discrete mesh points onto the unit sphere
PetscErrorCode ProjectToUnitSphere(DM dm);

// Auxiliary function to create a CeedVector from PetscVec of same size
PetscErrorCode CreateVectorFromPetscVec(Ceed ceed, Vec p, CeedVector *v);

// Auxiliary function to place a PetscVec into a CeedVector of same size
PetscErrorCode VectorPlacePetscVec(CeedVector c, Vec p);

// Auxiliary function to apply the ICs and eliminate repeated values in initial
//   state vector, arising from restriction
PetscErrorCode ICs_FixMultiplicity(CeedOperator op_ics,
    CeedVector xcorners, CeedVector q0ceed, DM dm, Vec Qloc, Vec Q,
    CeedElemRestriction restrictq, PhysicsContext ctx, CeedScalar time);

// Auxiliary function to compute the lumped mass matrix
PetscErrorCode ComputeLumpedMassMatrix(Ceed ceed, DM dm,
    CeedElemRestriction restrictq, CeedBasis basisq,
    CeedElemRestriction restrictqdi, CeedVector qdata, Vec M);

#endif // sw_headers_h
