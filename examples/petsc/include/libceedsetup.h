#ifndef libceedsetup_h
#define libceedsetup_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data);
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                    CeedInt topodim, CeedInt qextra,
                                    PetscInt ncompx, PetscInt ncompu,
                                    PetscInt gsize, PetscInt xlsize,
                                    bpData bpData, CeedData data,
                                    PetscBool setup_rhs, CeedVector rhsceed,
                                    CeedVector *target);
PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt numlevels,
                                      CeedInt ncompu, CeedData *data, CeedInt *leveldegrees,
                                      CeedQFunction qfrestrict, CeedQFunction qfprolong);

#endif // libceedsetup_h
