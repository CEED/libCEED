#ifndef CEED_UTILS_H
#define CEED_UTILS_H

#include <ceed.h>
#include <petscdm.h>

#define PetscCallCeed(ceed, ...)                                    \
  do {                                                              \
    int ierr = __VA_ARGS__;                                         \
    if (ierr != CEED_ERROR_SUCCESS) {                               \
      const char *error_message;                                    \
      CeedGetErrorMessage(ceed, &error_message);                    \
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", error_message); \
    }                                                               \
  } while (0)

/**
  @brief Translate PetscMemType to CeedMemType

  @param[in]  mem_type  PetscMemType

  @return Equivalent CeedMemType
**/
/// @ingroup RatelInternal
static inline CeedMemType MemTypePetscToCeed(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

/**
  @brief Translate array of `PetscInt` to `CeedInt`.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  Not collective across MPI processes.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of `PetscInt`
  @param[out]     array_ceed   Array of `CeedInt`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode IntArrayCeedToPetsc(PetscInt num_entries, CeedInt **array_ceed, PetscInt **array_petsc) {
  const CeedInt  int_c = 0;
  const PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_petsc = (PetscInt *)*array_ceed;
  } else {
    *array_petsc = malloc(num_entries * sizeof(PetscInt));
    for (PetscInt i = 0; i < num_entries; i++) (*array_petsc)[i] = (*array_ceed)[i];
    free(*array_ceed);
  }
  *array_ceed = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Translate array of `PetscInt` to `CeedInt`.
    If the types differ, `array_petsc` is freed with `PetscFree()` and `array_ceed` is allocated with `PetscMalloc1()`.
    Caller is responsible for freeing `array_ceed` with `PetscFree()`.

  Not collective across MPI processes.

  @param[in]      num_entries  Number of array entries
  @param[in,out]  array_petsc  Array of `PetscInt`
  @param[out]     array_ceed   Array of `CeedInt`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode IntArrayPetscToCeed(PetscInt num_entries, PetscInt **array_petsc, CeedInt **array_ceed) {
  const CeedInt  int_c = 0;
  const PetscInt int_p = 0;

  PetscFunctionBeginUser;
  if (sizeof(int_c) == sizeof(int_p)) {
    *array_ceed = (CeedInt *)*array_petsc;
  } else {
    PetscCall(PetscMalloc1(num_entries, array_ceed));
    for (PetscInt i = 0; i < num_entries; i++) (*array_ceed)[i] = (*array_petsc)[i];
    PetscCall(PetscFree(*array_petsc));
  }
  *array_petsc = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Transfer array from PETSc `Vec` to `CeedVector`.

  Collective across MPI processes.

  @param[in]   X_petsc   PETSc `Vec`
  @param[out]  mem_type  PETSc `MemType`
  @param[out]  x_ceed    `CeedVector`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode VecPetscToCeed(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayAndMemType(X_petsc, &x, mem_type));
  PetscCallCeed(CeedVectorReturnCeed(x_ceed), CeedVectorSetArray(x_ceed, MemTypePetscToCeed(*mem_type), CEED_USE_POINTER, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Transfer array from `CeedVector` to PETSc `Vec`.

  Collective across MPI processes.

  @param[in]   x_ceed    `CeedVector`
  @param[in]   mem_type  PETSc `MemType`
  @param[out]  X_petsc   PETSc `Vec`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode VecCeedToPetsc(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCallCeed(CeedVectorReturnCeed(x_ceed), CeedVectorTakeArray(x_ceed, MemTypePetscToCeed(mem_type), &x));
  PetscCall(VecRestoreArrayAndMemType(X_petsc, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Transfer read only array from PETSc `Vec` to `CeedVector`.

  Collective across MPI processes.

  @param[in]   X_petsc   PETSc `Vec`
  @param[out]  mem_type  PETSc `MemType`
  @param[out]  x_ceed    `CeedVector`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode VecReadPetscToCeed(Vec X_petsc, PetscMemType *mem_type, CeedVector x_ceed) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, mem_type));
  PetscCallCeed(CeedVectorReturnCeed(x_ceed), CeedVectorSetArray(x_ceed, MemTypePetscToCeed(*mem_type), CEED_USE_POINTER, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Transfer read only array from `CeedVector` to PETSc `Vec`.

  Collective across MPI processes.

  @param[in]   x_ceed    `CeedVector`
  @param[in]   mem_type  PETSc `MemType`
  @param[out]  X_petsc   PETSc `Vec`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode VecReadCeedToPetsc(CeedVector x_ceed, PetscMemType mem_type, Vec X_petsc) {
  PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCallCeed(CeedVectorReturnCeed(x_ceed), CeedVectorTakeArray(x_ceed, MemTypePetscToCeed(mem_type), &x));
  PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Copy PETSc `Vec` data into `CeedVector`

  @param[in]   X_petsc PETSc `Vec`
  @param[out]  x_ceed  `CeedVector`

  @return An error code: 0 - success, otherwise - failure
**/
static inline PetscErrorCode VecCopyPetscToCeed(Vec X_petsc, CeedVector x_ceed) {
  PetscScalar *x;
  PetscMemType mem_type;
  PetscInt     X_size;
  CeedSize     x_size;
  Ceed         ceed;

  PetscFunctionBeginUser;
  PetscCall(CeedVectorGetCeed(x_ceed, &ceed));
  PetscCall(VecGetLocalSize(X_petsc, &X_size));
  PetscCallCeed(ceed, CeedVectorGetLength(x_ceed, &x_size));
  PetscCheck(X_size == x_size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "X_petsc (%" PetscInt_FMT ") and x_ceed (%" CeedSize_FMT ") must be same size",
             X_size, x_size);

  PetscCall(VecGetArrayReadAndMemType(X_petsc, (const PetscScalar **)&x, &mem_type));
  PetscCallCeed(ceed, CeedVectorSetArray(x_ceed, MemTypePetscToCeed(mem_type), CEED_COPY_VALUES, x));
  PetscCall(VecRestoreArrayReadAndMemType(X_petsc, (const PetscScalar **)&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Return the quadrature size from the eval mode, dimension, and number of components

  @param[in]  eval_mode       The basis evaluation mode
  @param[in]  dim             The basis dimension
  @param[in]  num_components  The basis number of components

  @return The maximum of the two integers

**/
/// @ingroup RatelInternal
static inline CeedInt GetCeedQuadratureSize(CeedEvalMode eval_mode, CeedInt dim, CeedInt num_components) {
  switch (eval_mode) {
    case CEED_EVAL_INTERP:
      return num_components;
    case CEED_EVAL_GRAD:
      return dim * num_components;
    default:
      return -1;
  }
}

/**
  @brief Convert from DMPolytopeType to CeedElemTopology

  @param[in]  cell_type  DMPolytopeType for the cell

  @return CeedElemTopology, or 0 if no equivelent CeedElemTopology was found
**/
static inline CeedElemTopology PolytopeTypePetscToCeed(DMPolytopeType cell_type) {
  switch (cell_type) {
    case DM_POLYTOPE_TRIANGLE:
      return CEED_TOPOLOGY_TRIANGLE;
    case DM_POLYTOPE_QUADRILATERAL:
      return CEED_TOPOLOGY_QUAD;
    case DM_POLYTOPE_TETRAHEDRON:
      return CEED_TOPOLOGY_TET;
    case DM_POLYTOPE_HEXAHEDRON:
      return CEED_TOPOLOGY_HEX;
    default:
      return 0;
  }
}

#endif  // CEED_UTILS_H
