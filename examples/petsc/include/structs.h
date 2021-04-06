#ifndef structs_h
#define structs_h

#include <ceed.h>
#include <petsc.h>

// -----------------------------------------------------------------------------
// PETSc Operator Structs
// -----------------------------------------------------------------------------

// Data for PETSc Matshell
typedef struct UserO_ *UserO;
struct UserO_ {
  MPI_Comm comm;
  DM dm;
  Vec Xloc, Yloc, diag;
  CeedVector Xceed, Yceed;
  CeedOperator op;
  Ceed ceed;
};

// Data for PETSc Prolong/Restrict Matshells
typedef struct UserProlongRestr_ *UserProlongRestr;
struct UserProlongRestr_ {
  MPI_Comm comm;
  DM dmc, dmf;
  Vec locvecc, locvecf, multvec;
  CeedVector ceedvecc, ceedvecf;
  CeedOperator opProlong, opRestrict;
  Ceed ceed;
};

// -----------------------------------------------------------------------------
// libCEED Data Structs
// -----------------------------------------------------------------------------

// libCEED data struct for level
typedef struct CeedData_ *CeedData;
struct CeedData_ {
  Ceed ceed;
  CeedBasis basisx, basisu, basisctof;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictui, Erestrictqdi;
  CeedQFunction qfApply;
  CeedOperator opApply, opRestrict, opProlong;
  CeedVector qdata, Xceed, Yceed;
};

// BP specific data
typedef struct {
  CeedInt ncompx, ncompu, topodim, qdatasize, qextra;
  CeedQFunctionUser setupgeo, setuprhs, apply, error;
  const char *setupgeofname, *setuprhsfname, *applyfname, *errorfname;
  CeedEvalMode inmode, outmode;
  CeedQuadMode qmode;
  PetscBool enforcebc;
  PetscErrorCode (*bcsfunc)(PetscInt, PetscReal, const PetscReal *,
                            PetscInt, PetscScalar *, void *);
} bpData;

#endif // structs_h
