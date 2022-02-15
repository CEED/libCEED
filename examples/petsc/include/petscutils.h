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
PetscErrorCode SetupDMByDegree(DM dm, PetscInt degree, PetscInt num_comp_u,
                               PetscInt topo_dim,
                               bool enforce_bc,  BCFunction bc_func);
PetscErrorCode CreateRestrictionFromPlex(Ceed ceed, DM dm, CeedInt height,
    DMLabel domain_label, CeedInt value, CeedElemRestriction *elem_restr);
PetscErrorCode CreateBasisFromPlex(Ceed ceed, DM dm, DMLabel domain_label,
                                   CeedInt label_value, CeedInt height,
                                   CeedInt dm_field, CeedBasis *basis);
#endif // libceed_petsc_examples_utils_h
