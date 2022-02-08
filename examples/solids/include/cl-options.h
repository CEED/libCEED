#ifndef libceed_solids_examples_cl_options_h
#define libceed_solids_examples_cl_options_h

#include <petsc.h>
#include "../include/structs.h"

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

#endif // libceed_solids_examples_cl_options_h
