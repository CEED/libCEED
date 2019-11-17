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

//                        libCEED + PETSc Example: Surface Area
//
// This example demonstrates a simple usage of libCEED with PETSc to calculate
// the surface area of a simple closed surface, such as the one of a cube
// via the mass operator.
//
// The code uses higher level communication protocols in DMPlex.
//
// Build with:
//
//     make area [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//   Sequential:
//
//     area -petscspace_degree 3
//
//   In parallel:
//
//     mpiexec -n 4 area -petscspace_degree 3
//
//TESTARGS -ceed {ceed_resource} -test -petscspace_degree 3

/// @file
/// libCEED example using the mass operator to compute surface area using PETSc with DMPlex
static const char help[] =
  "Compute surface area of a cube using DMPlex in PETSc\n";

#include <string.h>
#include <petscdmplex.h>
#include <ceed.h>
#include "qfunctions/area/area.h"

// Auxiliary function to define CEED restrictions from DMPlex data
static int CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
                                 CeedElemRestriction *Erestrict, DM dm) {
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
    ierr = DMPlexGetClosureIndices(dm, section, section, c, &numindices,
                                   &indices, NULL); CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i] + (PetscInt)(copysign(j, indices[i])))
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
                   "Cell %D closure indices not interlaced", c);
      }
      // NO BC on closed surfaces
      PetscInt loc = indices[i];
      erestrict[eoffset++] = loc/ncomp;
    }
    ierr = DMPlexRestoreClosureIndices(dm, section, section, c, &numindices,
                                       &indices, NULL); CHKERRQ(ierr);
  }

  // Setup CEED restriction
  ierr = DMGetLocalVector(dm, &Uloc); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &nnodes); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &Uloc); CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, nelem, P*P, nnodes/ncomp, ncomp,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict,
                            Erestrict);
  ierr = PetscFree(erestrict); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char filename[PETSC_MAX_PATH_LEN],
       ceedresource[PETSC_MAX_PATH_LEN] = "/cpu/self";
  PetscInt lsize, gsize, xlsize,
           qextra  = 1, // default number of extra quadrature points
           ncompx  = 3, // number of components of 3D physical coordinates
           ncompu  = 1, // dimension of field to which apply mass operator
           topodim = 2, // topological dimension of manifold
           degree  = 3; // default degree for finite element bases
  PetscBool read_mesh = PETSC_FALSE,
            test_mode = PETSC_FALSE;
  PetscSpace sp;
  PetscFE fe;
  Vec X, Xloc, V, Vloc;
  DM  dm, dmcoord;
  Ceed ceed;
  CeedInt P, Q;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  // Read CL options
  ierr = PetscOptionsBegin(comm, NULL, "CEED surface area problem with PETSc",
                           NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);
  CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Setup DM
  PetscScalar l = 1.0/PetscSqrtReal(3.0); // half edge of the cube
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);
    CHKERRQ(ierr);
  } else {
    // Create the mesh as a 0-refined sphere. This will create a cubic surface, not a box.
    PetscBool simplex = PETSC_FALSE;
    ierr = DMPlexCreateSphereMesh(PETSC_COMM_WORLD, topodim, simplex, &dm);
    CHKERRQ(ierr);
    // Set the object name
    ierr = PetscObjectSetName((PetscObject) dm, "Cube"); CHKERRQ(ierr);
    // Distribute mesh over processes
    {
      DM dmDist = NULL;
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(dm, &part); CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part); CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, 0, NULL, &dmDist); CHKERRQ(ierr);
      if (dmDist) {
        ierr = DMDestroy(&dm); CHKERRQ(ierr);
        dm  = dmDist;
      }
    }
    // View DMPlex via runtime option
    ierr = DMViewFromOptions(dm, NULL, "-dm_view"); CHKERRQ(ierr);
  }

  // Create FE
  ierr = PetscFECreateDefault(PETSC_COMM_SELF, topodim, ncompu, PETSC_FALSE, NULL,
                              PETSC_DETERMINE, &fe);
  CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm); CHKERRQ(ierr);
  ierr = DMAddField(dm, NULL, (PetscObject)fe); CHKERRQ(ierr);
  ierr = DMCreateDS(dm); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  // Get basis space degree
  ierr = PetscFEGetBasisSpace(fe, &sp); CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &degree, NULL); CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe); CHKERRQ(ierr);
  if (degree < 1) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                             "-petscspace_degree %D must be at least 1", degree);

  // Create vectors
  ierr = DMCreateGlobalVector(dm, &X); CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &lsize); CHKERRQ(ierr);
  ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &Xloc); CHKERRQ(ierr);
  ierr = VecGetSize(Xloc, &xlsize); CHKERRQ(ierr);
  ierr = VecDuplicate(X, &V); CHKERRQ(ierr);
  ierr = VecDuplicate(Xloc, &Vloc); CHKERRQ(ierr);

  // Set up libCEED
  CeedInit(ceedresource, &ceed);

  // Print summary
  P = degree + 1;
  Q = P + qextra;
  const char *usedresource;
  CeedGetResource(ceed, &usedresource);
  if (!test_mode) {
    ierr = PetscPrintf(comm,
                       "\n-- libCEED + PETSc Surface Area problem --\n"
                       "  libCEED:\n"
                       "    libCEED Backend                    : %s\n"
                       "  Mesh:\n"
                       "    Number of 1D Basis Nodes (p)       : %d\n"
                       "    Number of 1D Quadrature Points (q) : %d\n"
                       "    Global nodes                       : %D\n",
                       usedresource, P, Q,  gsize/ncompu);
    CHKERRQ(ierr);
  }

  // Setup libCEED's objects
  // Create CEED operators and API objects they need
  CeedOperator op_setupgeo, op_apply;
  CeedQFunction qf_setupgeo, qf_apply;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi,
                      Erestrictqdi;

  // Create bases
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompu, P, Q,
                                  CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, topodim, ncompx, 2, Q,
                                  CEED_GAUSS, &basisx);

  // CEED restrictions
  ierr = DMGetCoordinateDM(dm, &dmcoord); CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord, PETSC_DETERMINE, NULL);
  CHKERRQ(ierr);

  CreateRestrictionPlex(ceed, 2, ncompx, &Erestrictx, dmcoord); CHKERRQ(ierr);
  CreateRestrictionPlex(ceed, P, ncompu, &Erestrictu, dm); CHKERRQ(ierr);

  CeedInt cStart, cEnd;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  const CeedInt nelem = cEnd - cStart;

  // CEED identity restrictions
  const CeedInt qdatasize = 1;
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nelem*Q*Q,
                                    qdatasize, &Erestrictqdi);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q, nelem*Q*Q, 1,
                                    &Erestrictxi);

  // Element coordinates
  Vec coords;
  const PetscScalar *coordArray;
  PetscSection section;
  ierr = DMGetCoordinatesLocal(dm, &coords); CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords, &coordArray); CHKERRQ(ierr);
  ierr = DMGetSection(dmcoord, &section); CHKERRQ(ierr);

  CeedVector xcoord;
  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES,
                     (PetscScalar *)coordArray);
  ierr = VecRestoreArrayRead(coords, &coordArray);

  // Create the vectors that will be needed in setup and apply
  CeedVector uceed, vceed, qdata;
  CeedInt nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &nqpts);
  CeedVectorCreate(ceed, qdatasize*nelem*nqpts, &qdata);
  CeedVectorCreate(ceed, xlsize, &uceed);
  CeedVectorCreate(ceed, xlsize, &vceed);

  /* Create the Q-function that builds the operator for the geomteric factors
     (i.e., the quadrature data) */
  CeedQFunctionCreateInterior(ceed, 1, SetupMassGeo,
                              SetupMassGeo_loc, &qf_setupgeo);
  CeedQFunctionAddInput(qf_setupgeo, "dx", ncompx*topodim, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setupgeo, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setupgeo, "qdata", qdatasize, CEED_EVAL_NONE);

  // Set up the mass operator
  CeedQFunctionCreateInterior(ceed, 1, Mass, Mass_loc, &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", ncompu, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_apply, "qdata", qdatasize, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_apply, "v", ncompu, CEED_EVAL_INTERP);

  // Create the operator that builds the quadrature data for the operator
  CeedOperatorCreate(ceed, qf_setupgeo, CEED_QFUNCTION_NONE,
                     CEED_QFUNCTION_NONE, &op_setupgeo);
  CeedOperatorSetField(op_setupgeo, "dx", Erestrictx, CEED_TRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setupgeo, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setupgeo, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  // Create the mass operator
  CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE,
                     &op_apply);
  CeedOperatorSetField(op_apply, "u", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_apply, "qdata", Erestrictqdi, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata);
  CeedOperatorSetField(op_apply, "v", Erestrictu, CEED_TRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Compute the quadrature data for the mass operator
  CeedOperatorApply(op_setupgeo, xcoord, qdata, CEED_REQUEST_IMMEDIATE);

  PetscScalar *v;
  ierr = VecZeroEntries(Vloc); CHKERRQ(ierr);
  ierr = VecGetArray(Vloc, &v);
  CeedVectorSetArray(vceed, CEED_MEM_HOST, CEED_USE_POINTER, v);

  // Compute the mesh volume using the mass operator: vol = 1^T \cdot M \cdot 1
  if (!test_mode) {
    ierr = PetscPrintf(comm,
                       "Computing the mesh volume using the formula: vol = 1^T M 1\n");
    CHKERRQ(ierr);
  }

  // Initialize u and v with ones
  CeedVectorSetValue(uceed, 1.0);
  CeedVectorSetValue(vceed, 1.0);

  // Apply the mass operator: 'u' -> 'v'
  CeedOperatorApply(op_apply, uceed, vceed, CEED_REQUEST_IMMEDIATE);
  CeedVectorSyncArray(vceed, CEED_MEM_HOST);

  // Gather output vector
  ierr = VecRestoreArray(Vloc, &v); CHKERRQ(ierr);
  ierr = VecZeroEntries(V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, Vloc, ADD_VALUES, V); CHKERRQ(ierr);

  // Compute and print the sum of the entries of 'v' giving the mesh surface area
  PetscScalar area;
  ierr = VecSum(V, &area); CHKERRQ(ierr);

  // Compute the exact surface area and print the result
  CeedScalar exact_surfarea = 6 * (2*l) * (2*l);
  if (!test_mode) {
    ierr = PetscPrintf(comm, "Exact mesh surface area    : % .14g\n",
                       exact_surfarea); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Computed mesh surface area : % .14g\n", area);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Area error                 : % .14g\n",
                       fabs(area - exact_surfarea)); CHKERRQ(ierr);
  }

  // PETSc cleanup
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&V); CHKERRQ(ierr);
  ierr = VecDestroy(&Vloc); CHKERRQ(ierr);

  // libCEED cleanup
  CeedQFunctionDestroy(&qf_setupgeo);
  CeedOperatorDestroy(&op_setupgeo);
  CeedVectorDestroy(&xcoord);
  CeedVectorDestroy(&uceed);
  CeedVectorDestroy(&vceed);
  CeedVectorDestroy(&qdata);
  CeedBasisDestroy(&basisx);
  CeedBasisDestroy(&basisu);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedElemRestrictionDestroy(&Erestrictqdi);
  CeedQFunctionDestroy(&qf_apply);
  CeedOperatorDestroy(&op_apply);
  CeedDestroy(&ceed);
  return PetscFinalize();
}
