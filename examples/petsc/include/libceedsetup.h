#ifndef libceed_petsc_examples_setup_h
#define libceed_petsc_examples_setup_h

#include <ceed.h>
#include <petsc.h>

#include "structs.h"

PetscErrorCode CeedDataDestroy(CeedInt i, CeedData data);
PetscErrorCode SetupLibceedByDegree(DM dm, Ceed ceed, CeedInt degree, CeedInt topo_dim, CeedInt q_extra, PetscInt num_comp_x, PetscInt num_comp_u,
                                    PetscInt g_size, PetscInt xl_size, BPData bp_data, CeedData data, PetscBool setup_rhs, CeedVector rhs_ceed,
                                    CeedVector *target);
PetscErrorCode CeedLevelTransferSetup(DM dm, Ceed ceed, CeedInt level, CeedInt num_comp_u, CeedData *data, BPData bp_data, Vec fine_mult);

#endif  // libceed_petsc_examples_setup_h
