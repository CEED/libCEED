#ifndef libceed_solids_examples_utils_h
#define libceed_solids_examples_utils_h

#include <ceed.h>
#include <petsc.h>

// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

#endif // libceed_solids_examples_utils_h
