#ifndef setupfe_h
#define setupfe_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

#include "structs.h"

// ---------------------------------------------------------------------------
// Setup FE
// ---------------------------------------------------------------------------
CeedMemType    MemTypeP2C(PetscMemType mtype);
PetscErrorCode SetupFEHdiv(MPI_Comm comm, DM dm, DM dm_u0, DM dm_p0);
PetscErrorCode SetupFEH1(ProblemData problem_data, AppCtx app_ctx, DM dm_H1);
PetscInt       Involute(PetscInt i);
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
// Utility function to create local CEED Oriented restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, DM dm_u0, DM dm_p0, CeedInt P, CeedElemRestriction *elem_restr_u,
                                                 CeedElemRestriction *elem_restr_p, CeedElemRestriction *elem_restr_u0,
                                                 CeedElemRestriction *elem_restr_p0);
#endif  // setupfe_h
