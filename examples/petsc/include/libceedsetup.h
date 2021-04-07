#ifndef libceedsetup_h
#define libceedsetup_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data);
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree,
                                    CeedInt topo_dim, CeedInt q_extra,
                                    PetscInt num_comp_x, PetscInt num_comp_u,
                                    PetscInt g_size, PetscInt xl_size,
                                    BPData bp_data, CeedData data,
                                    PetscBool setup_rhs, CeedVector rhs_ceed,
                                    CeedVector *target);
PetscErrorCode CeedLevelTransferSetup(Ceed ceed, CeedInt num_levels,
                                      CeedInt num_comp_u, CeedData *data, CeedInt *leveldegrees,
                                      CeedQFunction qf_restrict, CeedQFunction qf_prolong);

#endif // libceedsetup_h
