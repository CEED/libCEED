#ifndef setupfe_h
#define setupfe_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscsys.h>

#include "setup-boundary.h"
#include "structs.h"

// ---------------------------------------------------------------------------
// Setup FE
// ---------------------------------------------------------------------------
CeedMemType      MemTypeP2C(PetscMemType mtype);
PetscErrorCode   SetupFEByOrder(AppCtx app_ctx, ProblemData problem_data, DM dm);
PetscErrorCode   CreateRestrictionFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt value, CeedInt height, PetscInt dm_field,
                                           CeedElemRestriction *elem_restr);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscErrorCode   DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field);
PetscErrorCode   BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                           PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis);
PetscErrorCode   CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field,
                                     ProblemData problem_data, CeedBasis *basis);
#endif  // setupfe_h
