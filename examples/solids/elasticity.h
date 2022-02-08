#ifndef libceed_solids_examples_setup_h
#define libceed_solids_examples_setup_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <petscksp.h>
#include <stdbool.h>
#include <string.h>
#include "problems/problems.h"
#include "include/cl-options.h"
#include "include/matops.h"
#include "include/misc.h"
#include "include/structs.h"
#include "include/setup-dm.h"
#include "include/setup-libceed.h"
#include "include/utils.h"

#if PETSC_VERSION_LT(3,17,0)
#error "PETSc v3.17 or later is required"
#endif

#endif // libceed_solids_examples_setup_h
