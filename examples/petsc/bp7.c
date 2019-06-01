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

#include <stdbool.h>
#include "bp4.h"

static int CreateRestrictionPlex(Ceed ceed, const CeedInt melem[3], CeedInt P, CeedInt ncomp,
				 CeedElemRestriction *Erestrict, DM dm) {

  const PetscInt Nelem = melem[0]*melem[1]*melem[2];
  PetscInt mdof[3];//, *idx, *idxp;
  PetscSection   section;
  PetscInt       c, cStart, cEnd;
  PetscSpace     sp; 
  PetscFE        fe;
  PetscInt numindices,*indices;
  PetscInt       ierr;

  PetscInt       dim;

  DMGetDimension(dm, &dim);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
 
  ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(fe, &sp);
  ierr = PetscSpaceSetDegree(sp, P, P);
 
   for (c=cStart; c<cEnd; c++) {
    ierr = DMPlexSetClosurePermutationTensor(dm,c,section);CHKERRQ(ierr);
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
  }
  
  for (int d=0; d<3; d++) mdof[d] = melem[d]*(P-1) + 1;
  CeedElemRestrictionCreate(ceed, Nelem, P*P*P, mdof[0]*mdof[1]*mdof[2], ncomp,
                            CEED_MEM_HOST, CEED_OWN_POINTER, indices, Erestrict);
  //printf("Nelem %d, mdof %d, p %d, ncomp %d \n",Nelem, mdof[0]*mdof[1]*mdof[2], P, ncomp);
  PetscFunctionReturn(0);
}

typedef struct User_ *User;
struct User_ {
  MPI_Comm comm;
  VecScatter ltog;
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
  DMLocalToGlobalBegin(user->dm, user->Yloc, INSERT_VALUES, Y);
  DMLocalToGlobalEnd(user->dm, user->Yloc, INSERT_VALUES, Y);
 
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
  ierr = VecScatterBegin(user->ltog, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, X, user->Xloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
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
  ierr = MPI_Allreduce(MPI_IN_PLACE, &maxerror,
                       1, MPIU_SCALAR, MPIU_MAX, user->comm); CHKERRQ(ierr);
  CeedVectorDestroy(&collocated_error);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self/ref/serial";
  PetscInt degree, qextra, localelem, melem[3] = {2, 1, 1}, lsize, gsize;
  PetscScalar *r;
  PetscBool test_mode, benchmark_mode;
  DM             dm;
  PetscInt       dim = 3,tmp;
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
  degree = 1;// test_mode ? 3 : 1;
  ierr = PetscOptionsInt("-petscspace_degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  //gotta change this eventually
  qextra = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Topological dimension",NULL,dim,&dim,NULL);CHKERRQ(ierr);
  tmp = dim;
  ierr = PetscOptionsIntArray("-cells","Number of cells per dimension",NULL,melem,&tmp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-local_elem", "Target number of locally owned elements per process",
                         NULL, localelem, &localelem, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CeedInit(ceedresource, &ceed);

  //needed for the geometry, eventually this should be just one function call
  IS			verts, cells, bcPointsIS;
  PetscInt		vStart, vEnd, j, counter = 0, numFields, numBC, numindices, *indices;
  PetscInt		ic, cStart, cEnd;
  PetscScalar		*coordArray;
  PetscInt		numComp[1], numDOF[1], bcField[1];
  const PetscInt	*vertids, *cellids;
  Vec			coords;
  PetscSpace            sp; 
  PetscBool		dmInterp = PETSC_TRUE;
  PetscFE               fe;
  PetscSection          section;


  //I keep for now both, different things initialized
  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,melem,NULL,NULL,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, P, Q, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &basisx);
   
  //CeedInt ndof2,nqpt2;
  //ierr = CeedBasisGetNumNodes(basisu, &ndof2); //CeedChk(ierr);
  //ierr = CeedBasisGetNumQuadraturePoints(basisu, &nqpt2); //CeedChk(ierr);
  //printf("basis ndof %d, nqpt %d\n", ndof2, nqpt2);
  
  CreateRestrictionPlex(ceed, melem, 2, 3, &Erestrictx, dm);
  CreateRestrictionPlex(ceed, melem, P, 1, &Erestrictu, dm);
 
  CeedInt nelem = melem[0]*melem[1]*melem[2];
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

  //ierr = DMPlexCreateFromFile(comm, "3Dbrick4els.exo", dmInterp, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  numFields = 1;
  numComp[0] = 1;
  for (PetscInt k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
  numDOF[0] = 1;
  numBC = 0;

  // Please note that bcField stays uninitialized because numBC = 0,
  // therefore having a trash value. This is probably handled internally
  // within DMPlexCreateSection but idk how exactly.
  ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
  ierr = DMSetSection(dm, section);CHKERRQ(ierr);

  DMGetDimension(dm, &dim);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);

  PetscInt Pgrid = 2;
  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(fe, &sp); // fake FE I should be able to get rid of it
  ierr = PetscSpaceSetDegree(sp, Pgrid, Pgrid);
 
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "depth", 0, &verts);CHKERRQ(ierr);
  ierr = ISGetIndices(verts, &vertids);CHKERRQ(ierr);

  ierr = DMPlexGetDepthStratum(dm, 3, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "depth", 3, &cells);CHKERRQ(ierr);
  ierr = ISGetIndices(cells, &cellids);CHKERRQ(ierr);

        /*	Get Local Coordinates	*/
  ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);
 
  CeedInt  len3; 
  ierr = VecGetSize(coords,&len3);CHKERRQ(ierr);
  PetscInt len=floor(len3/dim);
  CeedScalar *xloc;
  CeedInt lenloc = pow(2,dim);
  printf("len %d len3 %d\n", len, len3);
  PetscScalar xx,yy,zz;
  PetscInt    ix,iy,iz, tx,ty,tz;
  PetscInt offset=cEnd-cStart;
  xloc = malloc(len3*sizeof(xloc[0]));
  ierr = PetscPrintf(comm, " Total number vertices %d, cells %d \n", vEnd-vStart, cEnd-cStart);CHKERRQ(ierr);
    for (ic = 0; ic < cEnd-cStart; ic++) 
	{ 	ierr = DMPlexSetClosurePermutationTensor(dm,ic,section);CHKERRQ(ierr);
                ierr = DMPlexGetClosureIndices(dm,section,section,ic,&numindices,&indices,NULL);CHKERRQ(ierr);
                printf("numindices %d, cellids[%d] %d \n",numindices, ic, cellids[ic]);
                // writing this super explicitly, in case there are still issues
                for (j = 0; j < lenloc; j++){
                   tx=dim*(indices[j]);
                   ty=dim*(indices[j])+1;
                   tz=dim*(indices[j])+2;
                   xx=coordArray[dim*(indices[j])];
                   yy=coordArray[dim*(indices[j])+1];
                   zz=coordArray[dim*(indices[j])+2];
                   ix=j+len*0;
		   iy=j+len*1;
	           iz=j+len*2;
                   xloc[ix]=xx;
                   xloc[iy]=yy;
                   xloc[iz]=zz;
                   ierr = PetscPrintf(comm, "xloc(%2d, %2d, %2d)=(%.2f,%.2f,%0.2f)  ind(%d)=%d, cell %d \n", ix, iy,iz, xx,yy,zz, j, indices[j], ic);CHKERRQ(ierr); 
                }
               ierr = DMPlexRestoreClosureIndices(dm,section,section,ic,&numindices,&indices,NULL);CHKERRQ(ierr);
               counter++;
        }
   
  CeedVectorCreate(ceed, len3, &xcoord);
  CeedInt     nnn;  
  CeedVectorGetLength(xcoord, &nnn);
  printf("Length of xcoord is %d \n",nnn);

  CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);

  //ierr = PetscFEGetBasisSpace(fe, &sp);
  //ierr = PetscSpaceSetDegree(sp, P, P);

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
  CeedInt Nqpts, Nelem = melem[0]*melem[1]*melem[2];
  CeedBasisGetNumQuadraturePoints(basisu, &Nqpts);
  CeedVectorCreate(ceed, Nelem*Nqpts, &rho);
  CeedVectorCreate(ceed, Nelem*Nqpts, &target);
  CeedVectorCreate(ceed, lsize, &rhsceed);
 
  // Create the operator that builds the quadrature data for the mass operator.
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorSetField(op_setup, "x", Erestrictx, CEED_NOTRANSPOSE,
                       basisx, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "dx", Erestrictx, CEED_NOTRANSPOSE,
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
  CeedVectorCreate(ceed, lsize, &user->xceed);
  CeedVectorCreate(ceed, lsize, &user->yceed);
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
  CeedScalar *testo;

  CeedVectorGetLength(xcoord,&ntest);
  CeedVectorGetLength(rho, &ntestt);
  //CeedVectorGetArrayRead(rho, CEED_MEM_HOST, &testo);
  printf("Length of rho is %d, length of xcoord %d, gsize %d, Xlocsize %d \n",ntestt, ntest,gsize,Xlocsize);
  //for (CeedInt i=0; i<ntestt; i++) {
  //printf("rho[%d] is %f, and length is %d \n",i,testo[i],ntestt);
  //}
  //CeedVectorRestoreArrayRead(rho, &testo);
  //exit(1);
  // Setup rho, rhs, and target
  CeedOperatorApply(op_setup, xcoord, rho, CEED_REQUEST_IMMEDIATE);
  ierr = CeedVectorSyncArray(rhsceed, CEED_MEM_HOST); CHKERRQ(ierr);
  CeedVectorDestroy(&xcoord);
  // Gather RHS
  ierr = VecRestoreArray(rhsloc, &r); CHKERRQ(ierr);
  ierr = VecZeroEntries(rhs); CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm, rhsloc, ADD_VALUES, rhs);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm, rhsloc, ADD_VALUES, rhs);CHKERRQ(ierr);

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
/*
  {
    PetscReal maxerror;
    ierr = ComputeErrorMax(user, op_error, X, target, &maxerror); CHKERRQ(ierr);
    if (!test_mode || maxerror > 5e-3) {
      ierr = PetscPrintf(comm, "Pointwise error (max) %e\n", (double)maxerror);
      CHKERRQ(ierr);
    }
  }
*/
  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&rhsloc); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Yloc); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

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
