#ifndef utils_h
#define utils_h

#include <ceed.h>
#include <petsc.h>

// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) {
  return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST;
}

#endif // utils_h