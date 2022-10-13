#ifndef libceed_petsc_examples_utils_h
#define libceed_petsc_examples_utils_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>

CeedMemType MemTypeP2C(PetscMemType mtype);
PetscErrorCode Kershaw(DM dm_orig, PetscScalar eps);
typedef PetscErrorCode (*BCFunction)(PetscInt dim, PetscReal time,
                                     const PetscReal x[],
                                     PetscInt num_comp_u, PetscScalar *u, void *ctx);
PetscErrorCode SetupDMByDegree(DM dm, PetscInt p_degree, PetscInt q_extra,
                               PetscInt num_comp_u,
                               PetscInt dim, bool enforce_bc, BCFunction bc_func);
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height,
    DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
CeedElemTopology ElemTopologyP2C(DMPolytopeType cell_type);
PetscErrorCode DMFieldToDSField(DM dm, DMLabel domain_label, PetscInt dm_field,
                                PetscInt *ds_field);
PetscErrorCode BasisCreateFromTabulation(Ceed ceed, DM dm, DMLabel domain_label,
    PetscInt label_value, PetscInt height, PetscInt face,
    PetscFE fe, PetscTabulation basis_tabulation, PetscQuadrature quadrature,
    CeedBasis *basis);
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label,
                                   CeedInt label_value, CeedInt height,
                                   CeedInt dm_field, CeedBasis *basis);
#endif // libceed_petsc_examples_utils_h
