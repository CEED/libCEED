//                        libCEED + PETSc Example: BP1
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP1 benchmark problem, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bp1 [PETSC_DIR=</path/to/petsc>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp1
//     bp1 -ceed /cpu/self
//     bp1 -ceed /gpu/occa
//     bp1 -ceed /cpu/occa
//     bp1 -ceed /omp/occa
//     bp1 -ceed /ocl/occa
//
//TESTARGS -ceed {ceed_resource} -test

/// @file
/// Mass operator example using PETSc
const char help[] = "Solve CEED BP1 using PETSc\n";

#include "bp4.h"
#include <petscdmplex.h>
#include <petscksp.h>

static int CreateRestrictionPlex(Ceed ceed, CeedInt P, CeedInt ncomp,
				 CeedElemRestriction *Erestrict, DM dm) {

  PetscSection   section;
  PetscInt       c, cStart, cEnd, Nelem, Ndof, *erestrict, eoffset;
  PetscInt       ierr;
  Vec Uloc;

  PetscFunctionBegin;
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  Nelem = cEnd - cStart;
  ierr = PetscMalloc1(Nelem*P*P*P, &erestrict);CHKERRQ(ierr);
  for (c=cStart,eoffset=0; c<cEnd; c++) {
    PetscInt numindices,*indices,i;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
    for (i=0; i<numindices; i+=ncomp) {
      for (PetscInt j=0; j<ncomp; j++) {
        if (indices[i+j] != indices[i]+j) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %D closure indices not interlaced",c);
      }
      // Essential boundary conditions are encoded as -(loc+1), but we don't care so we decode.
      PetscInt loc = indices[i] >= 0 ? indices[i] : -(indices[i] + 1);
      erestrict[eoffset++] = loc/ncomp;
    }
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
  }
  ierr = DMGetLocalVector(dm, &Uloc);CHKERRQ(ierr);
  ierr = VecGetLocalSize(Uloc, &Ndof);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc);CHKERRQ(ierr);
  CeedElemRestrictionCreate(ceed, Nelem, P*P*P, Ndof/ncomp, ncomp,
                            CEED_MEM_HOST, CEED_COPY_VALUES, erestrict, Erestrict);
  ierr = PetscFree(erestrict);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  CeedVector rho;
  Ceed ceed;
  DM   dm; 
};

// This function uses libCEED to compute the action of the mass matrix
static PetscErrorCode MatMult_Mass(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
 
  DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, user->Xloc);
  DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, user->Xloc);

  DMGlobalToLocalBegin(user->dm, Y, INSERT_VALUES, user->Yloc);
  DMGlobalToLocalEnd(user->dm, Y, INSERT_VALUES, user->Yloc);

  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  CeedOperatorApply(user->op, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(user->yceed, CEED_MEM_HOST); CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  ierr = VecZeroEntries(Y); CHKERRQ(ierr);
  DMLocalToGlobalBegin(user->dm, user->Yloc, ADD_VALUES, Y);
  DMLocalToGlobalEnd(user->dm, user->Yloc, ADD_VALUES, Y);
 
  PetscFunctionReturn(0);
}


static PetscErrorCode ComputeErrorMax(User user, CeedOperator op_error, Vec X,
                                      CeedVector target, PetscReal *maxerror) {
  PetscErrorCode ierr;
  PetscScalar *x;
  CeedVector collocated_error;
  CeedInt length;

  PetscFunctionBeginUser;
  CeedVectorGetLength(target, &length);
  CeedVectorCreate(user->ceed, length, &collocated_error);
  ierr = DMGlobalToLocal(user->dm, X, INSERT_VALUES, user->Xloc);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorApply(op_error, user->xceed, collocated_error,
                    CEED_REQUEST_IMMEDIATE);
  VecRestoreArrayRead(user->Xloc, (const PetscScalar **)&x); CHKERRQ(ierr);

  *maxerror = 0;
  const CeedScalar *e;
  CeedVectorGetArrayRead(collocated_error, CEED_MEM_HOST, &e);
  for (CeedInt i=0; i<length; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(e[i]));
  }
  CeedVectorRestoreArrayRead(collocated_error, &e);
  ierr = MPI_Allreduce(MPI_IN_PLACE, maxerror,
                       1, MPIU_REAL, MPIU_MAX, user->comm); CHKERRQ(ierr);
  CeedVectorDestroy(&collocated_error);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self/ref/serial";
  PetscInt degree, qextra, melem[3] = {2, 1, 1}, lsize, gsize;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode, read_mesh, enforce_bc;
  DM             dm, dmcoord;
  PetscInt       dim = 3;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi, Erestrictui;
  CeedQFunction qf_setup, qf_mass, qf_error;
  CeedOperator op_setup, op_mass, op_error;
  CeedVector xcoord, rho, rhsceed, target;
  CeedInt P, Q;
  Vec X, Xloc, rhs, rhsloc;
  Mat mat;
  KSP ksp;
  User user;
  double my_rt_start, my_rt, rt_min, rt_max;
  char filename[PETSC_MAX_PATH_LEN];

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "CEED BP1 in PETSc", NULL); CHKERRQ(ierr);
  test_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-test",
                          "Testing mode (do not print unless error is large)",
                          NULL, test_mode, &test_mode, NULL); CHKERRQ(ierr);
  benchmark_mode = PETSC_FALSE;
  ierr = PetscOptionsBool("-benchmark",
                          "Benchmarking mode (prints benchmark statistics)",
                          NULL, benchmark_mode, &benchmark_mode, NULL);
  CHKERRQ(ierr);
  //gotta change this eventually
  qextra = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Topological dimension",NULL,dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Read mesh from file", NULL,
                            filename, filename, sizeof(filename), &read_mesh);CHKERRQ(ierr);
  enforce_bc = PETSC_FALSE;
  ierr = PetscOptionsBool("-enforce_bc", "Enforce essential BCs", NULL,
                          enforce_bc, &enforce_bc, NULL);CHKERRQ(ierr);
  if (!read_mesh) {
    PetscInt tmp = dim;
    ierr = PetscOptionsIntArray("-cells","Number of cells per dimension",NULL,melem,&tmp,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CeedInit(ceedresource, &ceed);

  //needed for the geometry, eventually this should be just one function call
  PetscInt		vStart, vEnd, j, numindices, *indices, marker_ids[] = {1};
  PetscInt		ic, cStart, cEnd, nelem;
  const PetscScalar     *coordArray;
  Vec			coords;
  PetscSpace            sp;
  PetscFE               fe;
  PetscSection          section;


  //I keep for now both, different things initialized
  if (read_mesh) {
    ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, filename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,melem,NULL,NULL,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  }
  if (1) {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm  = dmDist;
    }
  }
  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  if (enforce_bc) {
    ierr = DMAddBoundary(dm,DM_BC_ESSENTIAL,"wall","marker",0,0,NULL,(void(*)(void))TrueSolution,1,marker_ids,NULL);CHKERRQ(ierr);
  }
  ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);

  ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp, &degree, NULL);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  if (degree < 1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"-petscspace_degree %D must be at least 1",degree);

  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, P, Q, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &basisx);

  //CeedInt ndof2,nqpt2;
  //ierr = CeedBasisGetNumNodes(basisu, &ndof2); //CeedChk(ierr);
  //ierr = CeedBasisGetNumQuadraturePoints(basisu, &nqpt2); //CeedChk(ierr);
  //printf("basis ndof %d, nqpt %d\n", ndof2, nqpt2);
  ierr = DMGetCoordinateDM(dm, &dmcoord);CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dmcoord,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
  CreateRestrictionPlex(ceed, 2, 3, &Erestrictx, dmcoord);
  CreateRestrictionPlex(ceed, P, 1, &Erestrictu, dm);

  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  nelem = cEnd - cStart;
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictui);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*Q*Q, nelem*Q*Q*Q, 1,
                                    &Erestrictxi);

  /*
   CeedInt  xdof, udof;
  CeedElemRestrictionGetNumComponents(Erestrictx,&xdof);
  CeedElemRestrictionGetNumComponents(Erestrictui,&udof);
  printf("restriction dofs, xdof %d, udof %d, nelem %d \n",xdof,udof, nelem);
   */

  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

        /*	Get Local Coordinates	*/
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coords,&coordArray);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dmcoord, &section);CHKERRQ(ierr);

  ierr = PetscPrintf(comm, " Total number vertices %d, cells %d \n", vEnd-vStart, cEnd-cStart);CHKERRQ(ierr);
  for (ic = cStart; ic < cEnd; ic++) {
    ierr = DMPlexGetClosureIndices(dmcoord,section,section,ic,&numindices,&indices,NULL);CHKERRQ(ierr);
    printf("coords %d\n",numindices);
    // writing this super explicitly, in case there are still issues
    for (j = 0; j < numindices; j+=dim) {
      PetscScalar xx,yy,zz;
      PetscInt    tx,ty,tz;
      tx = indices[j];
      ty = indices[j+1];
      tz = indices[j+2];
      xx = coordArray[tx];
      yy = coordArray[ty];
      zz = coordArray[tz];
      ierr = PetscPrintf(comm, "E %d.%d: xloc(%2d, %2d, %2d)=(%.2f,%.2f,%0.2f)\n", ic, j, tx, ty, tz, xx,yy,zz);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreClosureIndices(dmcoord,section,section,ic,&numindices,&indices,NULL);CHKERRQ(ierr);
  }
  CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL);
  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar*)coordArray);
  ierr = VecRestoreArrayRead(coords,&coordArray);CHKERRQ(ierr);

  DMCreateGlobalVector(dm,&X);
  DMCreateLocalVector(dm,&Xloc);
 
  PetscInt Xlocsize;
  VecGetSize(Xloc,&Xlocsize);
  VecGetLocalSize(X,&lsize);
  VecGetSize(X, &gsize); 
 
  // Create the Q-function that builds the mass operator (i.e. computes its
  // quadrature data) and set its context data.
  CeedQFunctionCreateInterior(ceed, 1,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionAddInput(qf_setup, "x", 3, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_setup, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_setup, "rhs", 1, CEED_EVAL_INTERP);

  // Create the Q-function that defines the action of the mass operator.
  CeedQFunctionCreateInterior(ceed, 1,
                              Mass, __FILE__ ":Mass", &qf_mass);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);

  // Create the error qfunction
  CeedQFunctionCreateInterior(ceed, 1,
                              Error, __FILE__ ":Error", &qf_error);
  CeedQFunctionAddInput(qf_error, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_error, "true_soln", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_error, "error", 1, CEED_EVAL_NONE);

  // Create the persistent vectors that will be needed in setup
  CeedInt Nqpts;
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, nelem*Nqpts, &rho);
  CeedVectorCreate(ceed, nelem*Nqpts, &target);
  CeedVectorCreate(ceed, Xlocsize, &rhsceed);
 
  // Create the operator that builds the quadrature data for the mass operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, CEED_TRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_TRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "weight", Erestrictxi, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_setup, "rhs", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, rhsceed);
  
    
  // Create the mass operator.
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorSetField(op_mass, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "rho", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, rho);
  CeedOperatorSetField(op_mass, "v", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);

  // Create the error operator
  CeedOperatorCreate(ceed, qf_error, NULL, NULL, &op_error);
  CeedOperatorSetField(op_error, "u", Erestrictu, CEED_NOTRANSPOSE,
                       basisu, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_error, "true_soln", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, target);
  CeedOperatorSetField(op_error, "error", Erestrictui, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);


  // Set up Mat
  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->comm = comm;
  user->dm =dm; 
  user->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &user->Yloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, Xlocsize, &user->xceed);
  CeedVectorCreate(ceed, Xlocsize, &user->yceed);
  user->op = op_mass;
  user->rho = rho;
  user->ceed = ceed;

  ierr = MatCreateShell(comm, lsize, lsize, gsize, gsize, user, &mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Mass);

  ierr = VecDuplicate(X, &rhs); CHKERRQ(ierr);

  // Get RHS vector
  ierr = VecDuplicate(Xloc, &rhsloc); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhsloc); CHKERRQ(ierr);
  ierr = VecGetArray(rhsloc, &r); CHKERRQ(ierr);
  CeedVectorSetArray(rhsceed, CEED_MEM_HOST, CEED_USE_POINTER, r);

  PetscInt ntest,ntestt;

  CeedVectorGetLength(xcoord,&ntest);
  CeedVectorGetLength(rho, &ntestt);
  printf("Length of rho is %d, length of xcoord %d, gsize %d, Xlocsize %d \n",ntestt, ntest,gsize,Xlocsize);
  CeedOperatorApply(op_setup, xcoord, rho, CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(rhsceed, CEED_MEM_HOST); CHKERRQ(ierr);
  CeedVectorDestroy(&xcoord);
  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);

  ierr = DMLocalToGlobal(dm, rhsloc, ADD_VALUES, rhs);CHKERRQ(ierr);

  CeedVectorDestroy(&rhsceed);
  
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  // Timed solve
  my_rt_start = MPI_Wtime();
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  my_rt = MPI_Wtime() - my_rt_start;
  {
    KSPType ksptype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    if (!test_mode || reason < 0 || rnorm > 1e-8) {
      ierr = PetscPrintf(comm, "KSP %s %s iterations %D rnorm %e\n", ksptype,
                         KSPConvergedReasons[reason], its, (double)rnorm); CHKERRQ(ierr);
    }
    if (benchmark_mode && (!test_mode)) {
      CeedInt gsize;
      ierr = VecGetSize(X, &gsize); CHKERRQ(ierr);
      MPI_Reduce(&my_rt, &rt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
      MPI_Reduce(&my_rt, &rt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      ierr = PetscPrintf(comm,
                         "CG solve time  : %g (%g) sec.\n"
                         "DOFs/sec in CG : %g (%g) million.\n",
                         rt_max, rt_min,
                         1e-6*gsize*its/rt_max, 1e-6*gsize*its/rt_min);
      CHKERRQ(ierr);
    }
  }
  {
    PetscReal maxerror;
    ierr = ComputeErrorMax(user, op_error, X, target, &maxerror); CHKERRQ(ierr);
    if (!test_mode || maxerror > 5e-3) {
      ierr = PetscPrintf(comm, "Pointwise error (max) %e\n", (double)maxerror);
      CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Yloc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = DMDestroy(&dm); CHKERRQ(ierr);

  CeedVectorDestroy(&user->xceed);
  CeedVectorDestroy(&user->yceed);
  CeedVectorDestroy(&user->rho);
  CeedVectorDestroy(&target);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
