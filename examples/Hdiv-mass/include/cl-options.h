#ifndef cloptions_h
#define cloptions_h

#include "../include/structs.h"

// Register problems to be available on the command line
PetscErrorCode RegisterProblems_Hdiv(AppCtx app_ctx);

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

#endif // cloptions_h
