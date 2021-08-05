/// @file
/// Test creation, use, and destruction of an element restriction for 2D quad Hdiv

// run with ./MixedPoisson /cpu/self/ref/serial 
const char help[] = "Test creation, use, and destruction of an element restriction for 2D quad Hdiv\n";

#include <ceed.h>
#include <petscdmplex.h>
#include "2DQuadbasis.h"

int main(int argc, char **argv) {
  PetscInt       ierr;
  MPI_Comm       comm;
  // PETSc objects
  DM             dm;
  PetscSection   sec;
  PetscBool      interpolate = PETSC_TRUE;
  PetscInt       nx = 2, ny = 1, num_elem = nx * ny;
  PetscInt       num_nodes = (nx+1)*(ny+1);
  PetscInt       faces[2] = {nx, ny};
  PetscInt       dim = 2;
  PetscInt       pStart, pEnd;
  PetscInt       cStart, cEnd, c; // cells
  PetscInt       eStart, eEnd, e; // edges
  PetscInt       vStart, vEnd, v; // vertices
  PetscInt       dofs_per_face;
  const PetscInt *ornt;
  // libCEED objects
  Ceed           ceed;
  const CeedInt loc_node = 4, Q1d = 2, Q = Q1d*Q1d;
  CeedInt num_comp = dim;
  CeedInt dof_e = dim*num_nodes; // dof per element! dof is vector in Hdiv
  CeedBasis b;
  CeedScalar q_ref[dim*Q], q_weights[Q];
  CeedScalar div[dof_e*Q], interp[dim*dof_e*Q];
  
  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;

  comm = PETSC_COMM_WORLD;

  CeedInit(argv[1], &ceed);
  
  // ---------------------------------------------------------------------------
  // Build 2D Hdiv basis
  // ---------------------------------------------------------------------------
  buildmats(Q1d, q_ref, q_weights, interp, div);
  ierr = CeedBasisCreateHdiv(ceed, CEED_QUAD, num_comp, loc_node, Q,
                             interp, div, q_ref, q_weights, &b); CHKERRQ(ierr);
  //CeedBasisHdivView(b, stdout);

  // ---------------------------------------------------------------------------
  // Set-up DM
  // ---------------------------------------------------------------------------
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL, 
                             NULL, NULL, interpolate, &dm); CHKERRQ(ierr);
  // Get plex limits
  ierr = DMPlexGetChart(dm, &pStart, &pEnd); CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd); CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd); CHKERRQ(ierr);

  // Create section
  ierr = PetscSectionCreate(comm, &sec); CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(sec,1); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(sec,0,"Velocity"); CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(sec,0,1); CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec,pStart,pEnd); CHKERRQ(ierr);

  // Setup dofs_per_face
  for (e = eStart; e < eEnd; e++) {
    ierr = DMPlexGetConeSize(dm, e, &dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(sec, e, 0, dofs_per_face); CHKERRQ(ierr);
    ierr = PetscSectionSetDof     (sec, e, dofs_per_face); CHKERRQ(ierr);
  }
  printf("=============cell========\n");
  CeedInt cone_size;
  for (c = cStart; c< cEnd; c++) {
    ierr = DMPlexGetConeSize(dm, c, &cone_size); CHKERRQ(ierr);
    printf("cell number %d\n", c);
    ierr = DMPlexGetConeOrientation(dm, c, &ornt); CHKERRQ(ierr);
    for (CeedInt j = 0; j < cone_size; j++){
      printf("%d\n", ornt[j]);
    }
  }

  //================================== To Check Restrictions ===================
  CeedInt P = 2;
  CeedInt ind_x[num_elem*P*P];
  CeedScalar x[dim*num_nodes];
  CeedVector X, Y;
  CeedElemRestriction r;

  for (CeedInt i=0; i<num_nodes; i++) {
    x[i+0*num_nodes] = i + 10;
    x[i+1*num_nodes] = i + 50;
  }

  CeedVectorCreate(ceed, dim*num_nodes, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorView(X, "%12.8f", stdout);
  //================== Element Setup =================
  for (CeedInt i=0; i<num_elem; i++) {
    CeedInt col, row, offset;
    col = i % nx;
    row = i / nx;
    offset = col*(P-1) + row*(nx+1)*(P-1);
    for (CeedInt j=0; j<P; j++)
      for (CeedInt k=0; k<P; k++) {
        ind_x[P*(P*i+k)+j] = offset + k*(nx+1) + j;
        //fprintf(stdout, "%d\n", ind_x[P*(P*i+k)+j]);
      }
  }
  //================== Restrictions ==================
  CeedElemRestrictionCreate(ceed, num_elem, P*P, dim, num_nodes, dim*num_nodes,
                            CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &r);
  CeedElemRestrictionView(r, stdout);
  CeedVectorCreate(ceed, num_elem*P*P*2, &Y);
  CeedVectorSetValue(Y, 0); // Allocates array
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, X, Y, CEED_REQUEST_IMMEDIATE);
  CeedVectorView(Y, "%12.8f", stdout);
  printf("=====\n");

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedElemRestrictionDestroy(&r);


  //==================================

  // ---------------------------------------------------------------------------
  // Free objects
  // ---------------------------------------------------------------------------
  // Free libCEED objects
  CeedDestroy(&ceed);
  CeedBasisDestroy(&b);
  // Free PETSc objects
  ierr = DMDestroy(&dm); CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sec); CHKERRQ(ierr);
  
  return PetscFinalize();
}

/*
  CeedInt P = 2, dim = 2;
  CeedInt nx = 2, ny = 1, num_elem = nx * ny;
  CeedInt num_nodes = (nx+1)*(ny+1);
  CeedInt ind_x[num_elem*P*P];
  CeedScalar x[dim*num_nodes];
  CeedVector X, Y;
  CeedElemRestriction r;

  CeedInit(argv[1], &ceed);

  //============= Some dof in x and y direction ==============
  for (CeedInt i=0; i<num_nodes; i++) {
    x[i+0*num_nodes] = i + 10;
    x[i+1*num_nodes] = i + 50;
  }

  CeedVectorCreate(ceed, dim*num_nodes, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorView(X, "%12.8f", stdout);
  //================== Element Setup =================
  for (CeedInt i=0; i<num_elem; i++) {
    CeedInt col, row, offset;
    col = i % nx;
    row = i / nx;
    offset = col*(P-1) + row*(nx+1)*(P-1);
    for (CeedInt j=0; j<P; j++)
      for (CeedInt k=0; k<P; k++) {
        ind_x[P*(P*i+k)+j] = offset + k*(nx+1) + j;
        //fprintf(stdout, "%d\n", ind_x[P*(P*i+k)+j]);
      }
  }
  //================== Restrictions ==================
  CeedElemRestrictionCreate(ceed, num_elem, P*P, dim, num_nodes, dim*num_nodes,
                            CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &r);
  CeedVectorCreate(ceed, num_elem*P*P*2, &Y);
  CeedVectorSetValue(Y, 0); // Allocates array
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, X, Y, CEED_REQUEST_IMMEDIATE);
  CeedVectorView(Y, "%12.8f", stdout);
  printf("=====\n");

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedElemRestrictionDestroy(&r);
  */