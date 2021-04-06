#ifndef petscutils_h
#define petscutils_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>

CeedMemType MemTypeP2C(PetscMemType mtype);
PetscErrorCode PetscFECreateByDegree(DM dm, PetscInt dim, PetscInt Nc,
                                     PetscBool isSimplex, const char prefix[],
                                     PetscInt order, PetscFE *fem);
PetscErrorCode ProjectToUnitSphere(DM dm);
PetscErrorCode kershaw(DM dmorig, PetscScalar eps);
typedef PetscErrorCode (*BCFunction)(PetscInt dim, PetscReal time,
                                     const PetscReal x[],
                                     PetscInt ncompu, PetscScalar *u, void *ctx);
PetscErrorCode SetupDMByDegree(DM dm, PetscInt degree, PetscInt ncompu,
                               PetscInt topodim,
                               bool enforcebc,  BCFunction bcsfunc);
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt P,
    CeedInt topodim, CeedInt height, DMLabel domainLabel, CeedInt value,
    CeedElemRestriction *Erestrict);

#endif // petscutils_h
