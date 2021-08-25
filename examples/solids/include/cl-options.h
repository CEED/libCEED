#ifndef cloptions_h
#define cloptions_h

#include <petsc.h>
#include "../include/structs.h"

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

#endif // cloptions_h
