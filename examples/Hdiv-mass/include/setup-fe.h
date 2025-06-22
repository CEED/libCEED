#ifndef setupfe_h
#define setupfe_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

#include "structs.h"

// ---------------------------------------------------------------------------
// Setup H(div) FE space
// ---------------------------------------------------------------------------
CeedMemType      MemTypeP2C(PetscMemType mtype);
PetscErrorCode   SetupFEHdiv(AppCtx app_ctx, ProblemData problem_data, DM dm);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscInt         Involute(PetscInt i);
PetscErrorCode   CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
PetscErrorCode   CreateRestrictionFromPlexOriented(Ceed ceed, DM dm, CeedInt P, CeedElemRestriction *elem_restr_u, CeedElemRestriction *elem_restr_p);
#endif  // setupfe_h
