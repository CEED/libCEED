#ifndef libceed_petsc_examples_utils_h
#define libceed_petsc_examples_utils_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>

#include "structs.h"

CeedMemType      MemTypeP2C(PetscMemType mtype);
PetscErrorCode   Kershaw(DM dm_orig, PetscScalar eps);
PetscErrorCode   SetupDMByDegree(DM dm, PetscInt p_degree, PetscInt q_extra, PetscInt num_comp_u, PetscInt topo_dim, bool enforce_bc);
PetscErrorCode   CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height, DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscErrorCode   DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field, PetscInt *ds_field);
PetscErrorCode   BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label, PetscInt label_value, PetscInt height, PetscInt face, PetscFE fe,
                                           PetscTabulation basis_tabulation, PetscQuadrature quadrature, CeedBasis *basis);
PetscErrorCode   CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label, CeedInt label_value, CeedInt height, CeedInt dm_field, BPData bp_data,
                                     CeedBasis *basis);
PetscErrorCode   CreateDistributedDM(RunParams rp, DM *dm);

#endif  // libceed_petsc_examples_utils_h
