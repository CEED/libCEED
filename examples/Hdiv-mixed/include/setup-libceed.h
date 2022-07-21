#ifndef setuplibceed_h
#define setuplibceed_h

#include "structs.h"

// Convert PETSc MemType to libCEED MemType
CeedMemType MemTypeP2C(PetscMemType mtype);
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedData ceed_data, ProblemData problem_data);
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i);
// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm,
    CeedInt height, DMLabel domain_label, CeedInt value,
    CeedElemRestriction *elem_restr);
// Utility function to create local CEED Oriented restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, DM dm_u0,
    DM dm_p0, CeedInt P,
    CeedElemRestriction *elem_restr_u, CeedElemRestriction *elem_restr_p,
    CeedElemRestriction *elem_restr_u0, CeedElemRestriction *elem_restr_p0);
// Set up libCEED for a given degree
PetscErrorCode SetupLibceed(DM dm, DM dm_u0, DM dm_p0, Ceed ceed,
                            AppCtx app_ctx,
                            ProblemData problem_data,
                            CeedData ceed_data);
#endif // setuplibceed_h
