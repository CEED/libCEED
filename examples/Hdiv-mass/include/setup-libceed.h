#ifndef setuplibceed_h
#define setuplibceed_h

#include "../include/structs.h"

// Convert PETSc MemType to libCEED MemType
CeedMemType MemTypeP2C(PetscMemType mtype);
// Destroy libCEED objects
PetscErrorCode CeedDataDestroy(CeedData ceed_data);
// Utility function - essential BC dofs are encoded in closure indices as -(i+1)
PetscInt Involute(PetscInt i);
// Utility function to create local CEED restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm,
    CeedInt height, DMLabel domain_label, CeedInt value, CeedInt P,
    CeedElemRestriction *elem_restr);
// Utility function to create local CEED Oriented restriction from DMPlex
PetscErrorCode CreateRestrictionFromPlexOriented(Ceed ceed, DM dm,
    CeedInt height, DMLabel domain_label, CeedInt value, CeedInt P,
    CeedElemRestriction *elem_restr_oriented);
// Set up libCEED for a given degree
PetscErrorCode SetupLibceed(DM dm, Ceed ceed, AppCtx app_ctx,
                            ProblemData *problem_data, PetscInt U_g_size,
                            PetscInt U_loc_size, CeedData ceed_data,
                            CeedVector rhs_ceed, CeedVector *target,
                            CeedVector true_ceed);
#endif // setuplibceed_h
