// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define _POSIX_C_SOURCE 200112
#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @cond DOXYGEN_SKIP
static CeedRequest ceed_request_immediate;
static CeedRequest ceed_request_ordered;

static struct {
  char prefix[CEED_MAX_RESOURCE_LEN];
  int (*init)(const char *resource, Ceed f);
  unsigned int priority;
} backends[32];
static size_t num_backends;

#define CEED_FTABLE_ENTRY(class, method) {#class #method, offsetof(struct class##_private, method)}
/// @endcond

/// @file
/// Implementation of core components of Ceed library

/// @addtogroup CeedUser
/// @{

/**
  @brief Request immediate completion

  This predefined constant is passed as the @ref CeedRequest argument to interfaces when the caller wishes for the operation to be performed immediately.
  The code

  @code
    CeedOperatorApply(op, ..., CEED_REQUEST_IMMEDIATE);
  @endcode

  is semantically equivalent to

  @code
    CeedRequest request;
    CeedOperatorApply(op, ..., &request);
    CeedRequestWait(&request);
  @endcode

  @sa CEED_REQUEST_ORDERED
**/
CeedRequest *const CEED_REQUEST_IMMEDIATE = &ceed_request_immediate;

/**
  @brief Request ordered completion

  This predefined constant is passed as the @ref CeedRequest argument to interfaces when the caller wishes for the operation to be completed in the order that it is submitted to the device.
  It is typically used in a construct such as:

  @code
    CeedRequest request;
    CeedOperatorApply(op1, ..., CEED_REQUEST_ORDERED);
    CeedOperatorApply(op2, ..., &request);
    // other optional work
    CeedRequestWait(&request);
  @endcode

  which allows the sequence to complete asynchronously but does not start `op2` until `op1` has completed.

  @todo The current implementation is overly strict, offering equivalent semantics to @ref CEED_REQUEST_IMMEDIATE.

  @sa CEED_REQUEST_IMMEDIATE
 */
CeedRequest *const CEED_REQUEST_ORDERED = &ceed_request_ordered;

/**
  @brief Wait for a @ref CeedRequest to complete.

  Calling @ref CeedRequestWait() on a `NULL` request is a no-op.

  @param[in,out] req Address of @ref CeedRequest to wait for; zeroed on completion.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedRequestWait(CeedRequest *req) {
  if (!*req) return CEED_ERROR_SUCCESS;
  return CeedError(NULL, CEED_ERROR_UNSUPPORTED, "CeedRequestWait not implemented");
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedDeveloper
/// @{

/**
  @brief Register a Ceed backend internally.

  Note: Backends should call @ref CeedRegister() instead.

  @param[in] prefix   Prefix of resources for this backend to respond to.
                        For example, the reference backend responds to "/cpu/self".
  @param[in] init     Initialization function called by @ref CeedInit() when the backend is selected to drive the requested resource
  @param[in] priority Integer priority.
                        Lower values are preferred in case the resource requested by @ref CeedInit() has non-unique best prefix match.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedRegisterImpl(const char *prefix, int (*init)(const char *, Ceed), unsigned int priority) {
  int ierr = 0;

  CeedPragmaCritical(CeedRegisterImpl) {
    if (num_backends < sizeof(backends) / sizeof(backends[0])) {
      strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
      backends[num_backends].prefix[CEED_MAX_RESOURCE_LEN - 1] = 0;
      backends[num_backends].init                              = init;
      backends[num_backends].priority                          = priority;
      num_backends++;
    } else {
      ierr = 1;
    }
  }
  CeedCheck(ierr == 0, NULL, CEED_ERROR_MAJOR, "Too many backends");
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a work vector space for a `ceed`

  @param[in,out] ceed `Ceed` to create work vector space for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedWorkVectorsCreate(Ceed ceed) {
  CeedCall(CeedCalloc(1, &ceed->work_vectors));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a work vector space for a `ceed`

  @param[in,out] ceed `Ceed` to destroy work vector space for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedWorkVectorsDestroy(Ceed ceed) {
  if (!ceed->work_vectors) return CEED_ERROR_SUCCESS;
  for (CeedSize i = 0; i < ceed->work_vectors->num_vecs; i++) {
    CeedCheck(!ceed->work_vectors->is_in_use[i], ceed, CEED_ERROR_ACCESS, "Work vector %" CeedSize_FMT " checked out but not returned");
    ceed->ref_count += 2;  // Note: increase ref_count to prevent Ceed destructor from triggering again
    CeedCall(CeedVectorDestroy(&ceed->work_vectors->vecs[i]));
    ceed->ref_count -= 1;  // Note: restore ref_count
  }
  CeedCall(CeedFree(&ceed->work_vectors->is_in_use));
  CeedCall(CeedFree(&ceed->work_vectors->vecs));
  CeedCall(CeedFree(&ceed->work_vectors));
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBackend
/// @{

/**
  @brief Return value of `CEED_DEBUG` environment variable

  @param[in] ceed `Ceed` context

  @return Boolean value: true  - debugging mode enabled
                         false - debugging mode disabled

  @ref Backend
**/
// LCOV_EXCL_START
bool CeedDebugFlag(const Ceed ceed) { return ceed->is_debug; }
// LCOV_EXCL_STOP

/**
  @brief Return value of `CEED_DEBUG` environment variable

  @return Boolean value: true  - debugging mode enabled
                         false - debugging mode disabled

  @ref Backend
**/
// LCOV_EXCL_START
bool CeedDebugFlagEnv(void) { return getenv("CEED_DEBUG") || getenv("DEBUG") || getenv("DBG"); }
// LCOV_EXCL_STOP

/**
  @brief Print debugging information in color

  @param[in] color  Color to print
  @param[in] format Printing format

  @ref Backend
**/
// LCOV_EXCL_START
void CeedDebugImpl256(const unsigned char color, const char *format, ...) {
  va_list args;
  va_start(args, format);
  fflush(stdout);
  if (color != CEED_DEBUG_COLOR_NONE) fprintf(stdout, "\033[38;5;%dm", color);
  vfprintf(stdout, format, args);
  if (color != CEED_DEBUG_COLOR_NONE) fprintf(stdout, "\033[m");
  fprintf(stdout, "\n");
  fflush(stdout);
  va_end(args);
}
// LCOV_EXCL_STOP

/**
  @brief Allocate an array on the host; use @ref CeedMalloc().

  Memory usage can be tracked by the library.
  This ensures sufficient alignment for vectorization and should be used for large allocations.

  @param[in]  n    Number of units to allocate
  @param[in]  unit Size of each unit
  @param[out] p    Address of pointer to hold the result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend

  @sa CeedFree()
**/
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n * unit);
  CeedCheck(ierr == 0, NULL, CEED_ERROR_MAJOR, "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Allocate a cleared (zeroed) array on the host; use @ref CeedCalloc().

  Memory usage can be tracked by the library.

  @param[in]  n    Number of units to allocate
  @param[in]  unit Size of each unit
  @param[out] p    Address of pointer to hold the result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend

  @sa CeedFree()
**/
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  CeedCheck(!n || !unit || *(void **)p, NULL, CEED_ERROR_MAJOR, "calloc failed to allocate %zd members of size %zd\n", n, unit);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Reallocate an array on the host; use @ref CeedRealloc().

  Memory usage can be tracked by the library.

  @param[in]  n    Number of units to allocate
  @param[in]  unit Size of each unit
  @param[out] p    Address of pointer to hold the result

  @return An error code: 0 - success, otherwise - failure

  @ref Backend

  @sa CeedFree()
**/
int CeedReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n * unit);
  CeedCheck(!n || !unit || *(void **)p, NULL, CEED_ERROR_MAJOR, "realloc failed to allocate %zd members of size %zd\n", n, unit);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Allocate a cleared string buffer on the host.

  Memory usage can be tracked by the library.

  @param[in]  source Pointer to string to be copied
  @param[out] copy   Pointer to variable to hold newly allocated string copy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend

  @sa CeedFree()
**/
int CeedStringAllocCopy(const char *source, char **copy) {
  size_t len = strlen(source);
  CeedCall(CeedCalloc(len + 1, copy));
  memcpy(*copy, source, len);
  return CEED_ERROR_SUCCESS;
}

/** Free memory allocated using @ref CeedMalloc() or @ref CeedCalloc()

  @param[in,out] p Address of pointer to memory.
                     This argument is of type `void*` to avoid needing a cast, but is the address of the pointer (which is zeroed) rather than the pointer.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return CEED_ERROR_SUCCESS;
}

/** Internal helper to manage handoff of user `source_array` to backend with proper @ref CeedCopyMode behavior.

  @param[in]     source_array          Source data provided by user
  @param[in]     copy_mode             Copy mode for the data
  @param[in]     num_values            Number of values to handle
  @param[in]     size_unit             Size of array element in bytes
  @param[in,out] target_array_owned    Pointer to location to allocated or hold owned data, may be freed if already allocated
  @param[out]    target_array_borrowed Pointer to location to hold borrowed data
  @param[out]    target_array          Pointer to location for data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
static inline int CeedSetHostGenericArray(const void *source_array, CeedCopyMode copy_mode, size_t size_unit, CeedSize num_values,
                                          void *target_array_owned, void *target_array_borrowed, void *target_array) {
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      if (!*(void **)target_array) {
        if (*(void **)target_array_borrowed) {
          *(void **)target_array = *(void **)target_array_borrowed;
        } else {
          if (!*(void **)target_array_owned) CeedCall(CeedCallocArray(num_values, size_unit, target_array_owned));
          *(void **)target_array = *(void **)target_array_owned;
        }
      }
      if (source_array) memcpy(*(void **)target_array, source_array, size_unit * num_values);
      break;
    case CEED_OWN_POINTER:
      CeedCall(CeedFree(target_array_owned));
      *(void **)target_array_owned    = (void *)source_array;
      *(void **)target_array_borrowed = NULL;
      *(void **)target_array          = *(void **)target_array_owned;
      break;
    case CEED_USE_POINTER:
      CeedCall(CeedFree(target_array_owned));
      *(void **)target_array_owned    = NULL;
      *(void **)target_array_borrowed = (void *)source_array;
      *(void **)target_array          = *(void **)target_array_borrowed;
  }
  return CEED_ERROR_SUCCESS;
}

/** Manage handoff of user `bool` `source_array` to backend with proper @ref CeedCopyMode behavior.

  @param[in]     source_array          Source data provided by user
  @param[in]     copy_mode             Copy mode for the data
  @param[in]     num_values            Number of values to handle
  @param[in,out] target_array_owned    Pointer to location to allocated or hold owned data, may be freed if already allocated
  @param[out]    target_array_borrowed Pointer to location to hold borrowed data
  @param[out]    target_array          Pointer to location for data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetHostBoolArray(const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values, const bool **target_array_owned,
                         const bool **target_array_borrowed, const bool **target_array) {
  CeedCall(CeedSetHostGenericArray(source_array, copy_mode, sizeof(bool), num_values, target_array_owned, target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

/** Manage handoff of user `CeedInt8` `source_array` to backend with proper @ref CeedCopyMode behavior.

  @param[in]     source_array          Source data provided by user
  @param[in]     copy_mode             Copy mode for the data
  @param[in]     num_values            Number of values to handle
  @param[in,out] target_array_owned    Pointer to location to allocated or hold owned data, may be freed if already allocated
  @param[out]    target_array_borrowed Pointer to location to hold borrowed data
  @param[out]    target_array          Pointer to location for data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetHostCeedInt8Array(const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values, const CeedInt8 **target_array_owned,
                             const CeedInt8 **target_array_borrowed, const CeedInt8 **target_array) {
  CeedCall(CeedSetHostGenericArray(source_array, copy_mode, sizeof(CeedInt8), num_values, target_array_owned, target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

/** Manage handoff of user `CeedInt` `source_array` to backend with proper @ref CeedCopyMode behavior.

  @param[in]     source_array          Source data provided by user
  @param[in]     copy_mode             Copy mode for the data
  @param[in]     num_values            Number of values to handle
  @param[in,out] target_array_owned    Pointer to location to allocated or hold owned data, may be freed if already allocated
  @param[out]    target_array_borrowed Pointer to location to hold borrowed data
  @param[out]    target_array          Pointer to location for data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetHostCeedIntArray(const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values, const CeedInt **target_array_owned,
                            const CeedInt **target_array_borrowed, const CeedInt **target_array) {
  CeedCall(CeedSetHostGenericArray(source_array, copy_mode, sizeof(CeedInt), num_values, target_array_owned, target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

/** Manage handoff of user `CeedScalar` `source_array` to backend with proper @ref CeedCopyMode behavior.

  @param[in]     source_array          Source data provided by user
  @param[in]     copy_mode             Copy mode for the data
  @param[in]     num_values            Number of values to handle
  @param[in,out] target_array_owned    Pointer to location to allocated or hold owned data, may be freed if already allocated
  @param[out]    target_array_borrowed Pointer to location to hold borrowed data
  @param[out]    target_array          Pointer to location for data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetHostCeedScalarArray(const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values, const CeedScalar **target_array_owned,
                               const CeedScalar **target_array_borrowed, const CeedScalar **target_array) {
  CeedCall(CeedSetHostGenericArray(source_array, copy_mode, sizeof(CeedScalar), num_values, target_array_owned, target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register a `Ceed` backend

  @param[in] prefix   Prefix of resources for this backend to respond to.
                        For example, the reference backend responds to "/cpu/self".
  @param[in] init     Initialization function called by @ref CeedInit() when the backend is selected to drive the requested resource
  @param[in] priority Integer priority.
                        Lower values are preferred in case the resource requested by @ref CeedInit() has non-unique best prefix match.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedRegister(const char *prefix, int (*init)(const char *, Ceed), unsigned int priority) {
  CeedDebugEnv("Backend Register: %s", prefix);
  CeedRegisterImpl(prefix, init, priority);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return debugging status flag

  @param[in]  ceed     `Ceed` context to get debugging flag
  @param[out] is_debug Variable to store debugging flag

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedIsDebug(Ceed ceed, bool *is_debug) {
  *is_debug = ceed->is_debug;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the root of the requested resource.

  Note: Caller is responsible for calling @ref CeedFree() on the `resource_root`.

  @param[in]  ceed          `Ceed` context to get resource name of
  @param[in]  resource      Full user specified resource
  @param[in]  delineator    Delineator to break `resource_root` and `resource_spec`
  @param[out] resource_root Variable to store resource root

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetResourceRoot(Ceed ceed, const char *resource, const char *delineator, char **resource_root) {
  char  *device_spec       = strstr(resource, delineator);
  size_t resource_root_len = device_spec ? (size_t)(device_spec - resource) + 1 : strlen(resource) + 1;

  CeedCall(CeedCalloc(resource_root_len, resource_root));
  memcpy(*resource_root, resource, resource_root_len - 1);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a parent `Ceed` context

  @param[in]  ceed   `Ceed` context to retrieve parent of
  @param[out] parent Address to save the parent to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetParent(Ceed ceed, Ceed *parent) {
  if (ceed->parent) {
    CeedCall(CeedGetParent(ceed->parent, parent));
    return CEED_ERROR_SUCCESS;
  }
  *parent = NULL;
  CeedCall(CeedReferenceCopy(ceed, parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate `Ceed` context

  @param[in]  ceed     `Ceed` context to retrieve delegate of
  @param[out] delegate Address to save the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetDelegate(Ceed ceed, Ceed *delegate) {
  *delegate = NULL;
  if (ceed->delegate) CeedCall(CeedReferenceCopy(ceed->delegate, delegate));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a delegate `Ceed` context

  This function allows a `Ceed` context to set a delegate `Ceed` context.
  All backend implementations default to the delegate `Ceed` context, unless overridden.

  @param[in]  ceed     `Ceed` context to set delegate of
  @param[out] delegate Address to set the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetDelegate(Ceed ceed, Ceed delegate) {
  CeedCall(CeedReferenceCopy(delegate, &ceed->delegate));
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate `Ceed` context for a specific object type

  @param[in]  ceed     `Ceed` context to retrieve delegate of
  @param[out] delegate Address to save the delegate to
  @param[in]  obj_name Name of the object type to retrieve delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate, const char *obj_name) {
  // Check for object delegate
  for (CeedInt i = 0; i < ceed->obj_delegate_count; i++) {
    if (!strcmp(obj_name, ceed->obj_delegates->obj_name)) {
      *delegate = NULL;
      CeedCall(CeedReferenceCopy(ceed->obj_delegates->delegate, delegate));
      return CEED_ERROR_SUCCESS;
    }
  }

  // Use default delegate if no object delegate
  CeedCall(CeedGetDelegate(ceed, delegate));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a delegate `Ceed` context for a specific object type

  This function allows a `Ceed` context to set a delegate `Ceed` context for a given type of `Ceed` object.
  All backend implementations default to the delegate `Ceed` context for this object.
  For example, `CeedSetObjectDelegate(ceed, delegate, "Basis")` uses delegate implementations for all `CeedBasis` backend functions.

  @param[in,out] ceed     `Ceed` context to set delegate of
  @param[in]     delegate `Ceed` context to use for delegation
  @param[in]     obj_name Name of the object type to set delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetObjectDelegate(Ceed ceed, Ceed delegate, const char *obj_name) {
  CeedInt count = ceed->obj_delegate_count;

  // Malloc or Realloc
  if (count) {
    CeedCall(CeedRealloc(count + 1, &ceed->obj_delegates));
  } else {
    CeedCall(CeedCalloc(1, &ceed->obj_delegates));
  }
  ceed->obj_delegate_count++;

  // Set object delegate
  CeedCall(CeedReferenceCopy(delegate, &ceed->obj_delegates[count].delegate));
  CeedCall(CeedStringAllocCopy(obj_name, &ceed->obj_delegates[count].obj_name));

  // Set delegate parent
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the fallback resource for `CeedOperator`

  @param[in]  ceed     `Ceed` context
  @param[out] resource Variable to store fallback resource

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetOperatorFallbackResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->op_fallback_resource;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the fallback `Ceed` for `CeedOperator`

  @param[in]  ceed          `Ceed` context
  @param[out] fallback_ceed Variable to store fallback `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetOperatorFallbackCeed(Ceed ceed, Ceed *fallback_ceed) {
  if (ceed->has_valid_op_fallback_resource) {
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- CeedOperator Fallback ----------\n");
    CeedDebug(ceed, "Getting fallback from %s to %s\n", ceed->resource, ceed->op_fallback_resource);
  }

  // Create fallback Ceed if uninitalized
  if (!ceed->op_fallback_ceed && ceed->has_valid_op_fallback_resource) {
    CeedDebug(ceed, "Creating fallback Ceed");

    Ceed        fallback_ceed;
    const char *fallback_resource;

    CeedCall(CeedGetOperatorFallbackResource(ceed, &fallback_resource));
    CeedCall(CeedInit(fallback_resource, &fallback_ceed));
    fallback_ceed->op_fallback_parent = ceed;
    fallback_ceed->Error              = ceed->Error;
    ceed->op_fallback_ceed            = fallback_ceed;
    {
      const char **jit_source_roots;
      CeedInt      num_jit_source_roots = 0;

      CeedCall(CeedGetJitSourceRoots(ceed, &num_jit_source_roots, &jit_source_roots));
      for (CeedInt i = 0; i < num_jit_source_roots; i++) {
        CeedCall(CeedAddJitSourceRoot(fallback_ceed, jit_source_roots[i]));
      }
      CeedCall(CeedRestoreJitSourceRoots(ceed, &jit_source_roots));
    }
    {
      const char **jit_defines;
      CeedInt      num_jit_defines = 0;

      CeedCall(CeedGetJitDefines(ceed, &num_jit_defines, &jit_defines));
      for (CeedInt i = 0; i < num_jit_defines; i++) {
        CeedCall(CeedAddJitSourceRoot(fallback_ceed, jit_defines[i]));
      }
      CeedCall(CeedRestoreJitDefines(ceed, &jit_defines));
    }
  }
  *fallback_ceed = NULL;
  if (ceed->op_fallback_ceed) CeedCall(CeedReferenceCopy(ceed->op_fallback_ceed, fallback_ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the fallback resource for `CeedOperator`.

  The current resource, if any, is freed by calling this function.
  This string is freed upon the destruction of the `Ceed` context.

  @param[in,out] ceed     `Ceed` context
  @param[in]     resource Fallback resource to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetOperatorFallbackResource(Ceed ceed, const char *resource) {
  // Free old
  CeedCall(CeedFree(&ceed->op_fallback_resource));

  // Set new
  CeedCall(CeedStringAllocCopy(resource, (char **)&ceed->op_fallback_resource));

  // Check validity
  ceed->has_valid_op_fallback_resource = ceed->op_fallback_resource && ceed->resource && strcmp(ceed->op_fallback_resource, ceed->resource);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Flag `Ceed` context as deterministic

  @param[in]  ceed             `Ceed` to flag as deterministic
  @param[out] is_deterministic Deterministic status to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetDeterministic(Ceed ceed, bool is_deterministic) {
  ceed->is_deterministic = is_deterministic;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a backend function.

  This function is used for a backend to set the function associated with the Ceed objects.
  For example, `CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", BackendVectorCreate)` sets the backend implementation of @ref CeedVectorCreate() and `CeedSetBackendFunction(ceed, "Basis", basis, "Apply", BackendBasisApply)` sets the backend implementation of @ref CeedBasisApply().
  Note, the prefix 'Ceed' is not required for the object type ("Basis" vs "CeedBasis").

  @param[in]  ceed      `Ceed` context for error handling
  @param[in]  type      Type of Ceed object to set function for
  @param[out] object    Ceed object to set function for
  @param[in]  func_name Name of function to set
  @param[in]  f         Function to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetBackendFunctionImpl(Ceed ceed, const char *type, void *object, const char *func_name, void (*f)(void)) {
  char lookup_name[CEED_MAX_RESOURCE_LEN + 1] = "";

  // Build lookup name
  if (strcmp(type, "Ceed")) strncat(lookup_name, "Ceed", CEED_MAX_RESOURCE_LEN);
  strncat(lookup_name, type, CEED_MAX_RESOURCE_LEN);
  strncat(lookup_name, func_name, CEED_MAX_RESOURCE_LEN);

  // Find and use offset
  for (CeedInt i = 0; ceed->f_offsets[i].func_name; i++) {
    if (!strcmp(ceed->f_offsets[i].func_name, lookup_name)) {
      size_t offset          = ceed->f_offsets[i].offset;
      int (**fpointer)(void) = (int (**)(void))((char *)object + offset);  // *NOPAD*

      *fpointer = (int (*)(void))f;
      return CEED_ERROR_SUCCESS;
    }
  }

  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Requested function '%s' was not found for CEED object '%s'", func_name, type);
  // LCOV_EXCL_STOP
}

/**
  @brief Retrieve backend data for a `Ceed` context

  @param[in]  ceed `Ceed` context to retrieve data of
  @param[out] data Address to save data to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetData(Ceed ceed, void *data) {
  *(void **)data = ceed->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data for a `Ceed` context

  @param[in,out] ceed `Ceed` context to set data of
  @param[in]     data Address of data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetData(Ceed ceed, void *data) {
  ceed->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `Ceed` context

  @param[in,out] ceed `Ceed` context to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedReference(Ceed ceed) {
  ceed->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a `CeedVector` for scratch work from a `Ceed` context.

  Note: This vector must be restored with @ref CeedRestoreWorkVector().

  @param[in]  ceed `Ceed` context
  @param[in]  len  Minimum length of work vector
  @param[out] vec  Address of the variable where `CeedVector` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetWorkVector(Ceed ceed, CeedSize len, CeedVector *vec) {
  CeedInt i = 0;

  if (!ceed->work_vectors) CeedCall(CeedWorkVectorsCreate(ceed));

  // Search for big enough work vector
  for (i = 0; i < ceed->work_vectors->num_vecs; i++) {
    if (!ceed->work_vectors->is_in_use[i]) {
      CeedSize work_len;

      CeedCall(CeedVectorGetLength(ceed->work_vectors->vecs[i], &work_len));
      if (work_len >= len) break;
    }
  }
  // Long enough vector was not found
  if (i == ceed->work_vectors->num_vecs) {
    if (ceed->work_vectors->max_vecs == 0) {
      ceed->work_vectors->max_vecs = 1;
      CeedCall(CeedCalloc(ceed->work_vectors->max_vecs, &ceed->work_vectors->vecs));
      CeedCall(CeedCalloc(ceed->work_vectors->max_vecs, &ceed->work_vectors->is_in_use));
    } else if (ceed->work_vectors->max_vecs == i) {
      ceed->work_vectors->max_vecs *= 2;
      CeedCall(CeedRealloc(ceed->work_vectors->max_vecs, &ceed->work_vectors->vecs));
      CeedCall(CeedRealloc(ceed->work_vectors->max_vecs, &ceed->work_vectors->is_in_use));
    }
    ceed->work_vectors->num_vecs++;
    CeedCallBackend(CeedVectorCreate(ceed, len, &ceed->work_vectors->vecs[i]));
    ceed->ref_count--;  // Note: ref_count manipulation to prevent a ref-loop
  }
  // Return pointer to work vector
  ceed->work_vectors->is_in_use[i] = true;
  *vec                             = NULL;
  CeedCall(CeedVectorReferenceCopy(ceed->work_vectors->vecs[i], vec));
  ceed->ref_count++;  // Note: bump ref_count to account for external access
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore a `CeedVector` for scratch work from a `Ceed` context from @ref CeedGetWorkVector()

  @param[in]  ceed `Ceed` context
  @param[out] vec  `CeedVector` to restore

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedRestoreWorkVector(Ceed ceed, CeedVector *vec) {
  for (CeedInt i = 0; i < ceed->work_vectors->num_vecs; i++) {
    if (*vec == ceed->work_vectors->vecs[i]) {
      CeedCheck(ceed->work_vectors->is_in_use[i], ceed, CEED_ERROR_ACCESS, "Work vector %" CeedSize_FMT " was not checked out but is being returned");
      CeedCall(CeedVectorDestroy(vec));
      ceed->work_vectors->is_in_use[i] = false;
      ceed->ref_count--;  // Note: reduce ref_count again to prevent a ref-loop
      return CEED_ERROR_SUCCESS;
    }
  }
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_MAJOR, "vec was not checked out via CeedGetWorkVector()");
  // LCOV_EXCL_STOP
}

/**
  @brief Retrieve list of additional JiT source roots from `Ceed` context.

  Note: The caller is responsible for restoring `jit_source_roots` with @ref CeedRestoreJitSourceRoots().

  @param[in]  ceed             `Ceed` context
  @param[out] num_source_roots Number of JiT source directories
  @param[out] jit_source_roots Absolute paths to additional JiT source directories

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitSourceRoots(Ceed ceed, CeedInt *num_source_roots, const char ***jit_source_roots) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  *num_source_roots = ceed_parent->num_jit_source_roots;
  *jit_source_roots = (const char **)ceed_parent->jit_source_roots;
  ceed_parent->num_jit_source_roots_readers++;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore list of additional JiT source roots from with @ref CeedGetJitSourceRoots()

  @param[in]  ceed             `Ceed` context
  @param[out] jit_source_roots Absolute paths to additional JiT source directories

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedRestoreJitSourceRoots(Ceed ceed, const char ***jit_source_roots) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  *jit_source_roots = NULL;
  ceed_parent->num_jit_source_roots_readers--;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve list of additional JiT defines from `Ceed` context.

  Note: The caller is responsible for restoring `jit_defines` with @ref CeedRestoreJitDefines().

  @param[in]  ceed            `Ceed` context
  @param[out] num_jit_defines Number of JiT defines
  @param[out] jit_defines     Strings such as `foo=bar`, used as `-Dfoo=bar` in JiT

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetJitDefines(Ceed ceed, CeedInt *num_jit_defines, const char ***jit_defines) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  *num_jit_defines = ceed_parent->num_jit_defines;
  *jit_defines     = (const char **)ceed_parent->jit_defines;
  ceed_parent->num_jit_defines_readers++;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore list of additional JiT defines from with @ref CeedGetJitDefines()

  @param[in]  ceed        `Ceed` context
  @param[out] jit_defines String such as `foo=bar`, used as `-Dfoo=bar` in JiT

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedRestoreJitDefines(Ceed ceed, const char ***jit_defines) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  *jit_defines = NULL;
  ceed_parent->num_jit_defines_readers--;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedUser
/// @{

/**
  @brief Get the list of available resource names for `Ceed` contexts

  Note: The caller is responsible for `free()`ing the resources and priorities arrays, but should not `free()` the contents of the resources array.

  @param[out] n          Number of available resources
  @param[out] resources  List of available resource names
  @param[out] priorities Resource name prioritization values, lower is better

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
// LCOV_EXCL_START
int CeedRegistryGetList(size_t *n, char ***const resources, CeedInt **priorities) {
  *n         = 0;
  *resources = malloc(num_backends * sizeof(**resources));
  CeedCheck(resources, NULL, CEED_ERROR_MAJOR, "malloc() failure");
  if (priorities) {
    *priorities = malloc(num_backends * sizeof(**priorities));
    CeedCheck(priorities, NULL, CEED_ERROR_MAJOR, "malloc() failure");
  }
  for (size_t i = 0; i < num_backends; i++) {
    // Only report compiled backends
    if (backends[i].priority < CEED_MAX_BACKEND_PRIORITY) {
      *resources[i] = backends[i].prefix;
      if (priorities) *priorities[i] = backends[i].priority;
      *n += 1;
    }
  }
  CeedCheck(*n, NULL, CEED_ERROR_MAJOR, "No backends installed");
  *resources = realloc(*resources, *n * sizeof(**resources));
  CeedCheck(resources, NULL, CEED_ERROR_MAJOR, "realloc() failure");
  if (priorities) {
    *priorities = realloc(*priorities, *n * sizeof(**priorities));
    CeedCheck(priorities, NULL, CEED_ERROR_MAJOR, "realloc() failure");
  }
  return CEED_ERROR_SUCCESS;
}
// LCOV_EXCL_STOP

/**
  @brief Initialize a `Ceed` context to use the specified resource.

  Note: Prefixing the resource with "help:" (e.g. "help:/cpu/self") will result in @ref CeedInt() printing the current libCEED version number and a list of current available backend resources to `stderr`.

  @param[in]  resource Resource to use, e.g., "/cpu/self"
  @param[out] ceed     The library context

  @return An error code: 0 - success, otherwise - failure

  @ref User

  @sa CeedRegister() CeedDestroy()
**/
int CeedInit(const char *resource, Ceed *ceed) {
  size_t match_len = 0, match_index = UINT_MAX, match_priority = CEED_MAX_BACKEND_PRIORITY, priority;

  // Find matching backend
  CeedCheck(resource, NULL, CEED_ERROR_MAJOR, "No resource provided");
  CeedCall(CeedRegisterAll());

  // Check for help request
  const char *help_prefix = "help";
  size_t      match_help  = 0;
  while (match_help < 4 && resource[match_help] == help_prefix[match_help]) match_help++;
  if (match_help == 4) {
    fprintf(stderr, "libCEED version: %d.%d%d%s\n", CEED_VERSION_MAJOR, CEED_VERSION_MINOR, CEED_VERSION_PATCH,
            CEED_VERSION_RELEASE ? "" : "+development");
    fprintf(stderr, "Available backend resources:\n");
    for (size_t i = 0; i < num_backends; i++) {
      // Only report compiled backends
      if (backends[i].priority < CEED_MAX_BACKEND_PRIORITY) fprintf(stderr, "  %s\n", backends[i].prefix);
    }
    fflush(stderr);
    match_help = 5;  // Delineating character expected
  } else {
    match_help = 0;
  }

  // Find best match, computed as number of matching characters from requested resource stem
  size_t stem_length = 0;
  while (resource[stem_length + match_help] && resource[stem_length + match_help] != ':') stem_length++;
  for (size_t i = 0; i < num_backends; i++) {
    size_t      n      = 0;
    const char *prefix = backends[i].prefix;
    while (prefix[n] && prefix[n] == resource[n + match_help]) n++;
    priority = backends[i].priority;
    if (n > match_len || (n == match_len && match_priority > priority)) {
      match_len      = n;
      match_priority = priority;
      match_index    = i;
    }
  }
  // Using Levenshtein distance to find closest match
  if (match_len <= 1 || match_len != stem_length) {
    // LCOV_EXCL_START
    size_t lev_dis   = UINT_MAX;
    size_t lev_index = UINT_MAX, lev_priority = CEED_MAX_BACKEND_PRIORITY;
    for (size_t i = 0; i < num_backends; i++) {
      const char *prefix        = backends[i].prefix;
      size_t      prefix_length = strlen(backends[i].prefix);
      size_t      min_len       = (prefix_length < stem_length) ? prefix_length : stem_length;
      size_t      column[min_len + 1];
      for (size_t j = 0; j <= min_len; j++) column[j] = j;
      for (size_t j = 1; j <= min_len; j++) {
        column[0] = j;
        for (size_t k = 1, last_diag = j - 1; k <= min_len; k++) {
          size_t old_diag = column[k];
          size_t min_1    = (column[k] < column[k - 1]) ? column[k] + 1 : column[k - 1] + 1;
          size_t min_2    = last_diag + (resource[k - 1] == prefix[j - 1] ? 0 : 1);
          column[k]       = (min_1 < min_2) ? min_1 : min_2;
          last_diag       = old_diag;
        }
      }
      size_t n = column[min_len];
      priority = backends[i].priority;
      if (n < lev_dis || (n == lev_dis && lev_priority > priority)) {
        lev_dis      = n;
        lev_priority = priority;
        lev_index    = i;
      }
    }
    const char *prefix_lev = backends[lev_index].prefix;
    size_t      lev_length = 0;
    while (prefix_lev[lev_length] && prefix_lev[lev_length] != '\0') lev_length++;
    size_t m = (lev_length < stem_length) ? lev_length : stem_length;
    if (lev_dis + 1 >= m) return CeedError(NULL, CEED_ERROR_MAJOR, "No suitable backend: %s", resource);
    else return CeedError(NULL, CEED_ERROR_MAJOR, "No suitable backend: %s\nClosest match: %s", resource, backends[lev_index].prefix);
    // LCOV_EXCL_STOP
  }

  // Setup Ceed
  CeedCall(CeedCalloc(1, ceed));
  CeedCall(CeedCalloc(1, &(*ceed)->jit_source_roots));
  const char *ceed_error_handler = getenv("CEED_ERROR_HANDLER");
  if (!ceed_error_handler) ceed_error_handler = "abort";
  if (!strcmp(ceed_error_handler, "exit")) (*ceed)->Error = CeedErrorExit;
  else if (!strcmp(ceed_error_handler, "store")) (*ceed)->Error = CeedErrorStore;
  else (*ceed)->Error = CeedErrorAbort;
  memcpy((*ceed)->err_msg, "No error message stored", 24);
  (*ceed)->ref_count = 1;
  (*ceed)->data      = NULL;

  // Set lookup table
  FOffset f_offsets[] = {
      CEED_FTABLE_ENTRY(Ceed, Error),
      CEED_FTABLE_ENTRY(Ceed, SetStream),
      CEED_FTABLE_ENTRY(Ceed, GetPreferredMemType),
      CEED_FTABLE_ENTRY(Ceed, Destroy),
      CEED_FTABLE_ENTRY(Ceed, VectorCreate),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreate),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreateAtPoints),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreateBlocked),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateTensorH1),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateH1),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateHdiv),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateHcurl),
      CEED_FTABLE_ENTRY(Ceed, TensorContractCreate),
      CEED_FTABLE_ENTRY(Ceed, QFunctionCreate),
      CEED_FTABLE_ENTRY(Ceed, QFunctionContextCreate),
      CEED_FTABLE_ENTRY(Ceed, OperatorCreate),
      CEED_FTABLE_ENTRY(Ceed, OperatorCreateAtPoints),
      CEED_FTABLE_ENTRY(Ceed, CompositeOperatorCreate),
      CEED_FTABLE_ENTRY(CeedVector, HasValidArray),
      CEED_FTABLE_ENTRY(CeedVector, HasBorrowedArrayOfType),
      CEED_FTABLE_ENTRY(CeedVector, CopyStrided),
      CEED_FTABLE_ENTRY(CeedVector, SetArray),
      CEED_FTABLE_ENTRY(CeedVector, TakeArray),
      CEED_FTABLE_ENTRY(CeedVector, SetValue),
      CEED_FTABLE_ENTRY(CeedVector, SetValueStrided),
      CEED_FTABLE_ENTRY(CeedVector, SyncArray),
      CEED_FTABLE_ENTRY(CeedVector, GetArray),
      CEED_FTABLE_ENTRY(CeedVector, GetArrayRead),
      CEED_FTABLE_ENTRY(CeedVector, GetArrayWrite),
      CEED_FTABLE_ENTRY(CeedVector, RestoreArray),
      CEED_FTABLE_ENTRY(CeedVector, RestoreArrayRead),
      CEED_FTABLE_ENTRY(CeedVector, Norm),
      CEED_FTABLE_ENTRY(CeedVector, Scale),
      CEED_FTABLE_ENTRY(CeedVector, AXPY),
      CEED_FTABLE_ENTRY(CeedVector, AXPBY),
      CEED_FTABLE_ENTRY(CeedVector, PointwiseMult),
      CEED_FTABLE_ENTRY(CeedVector, Reciprocal),
      CEED_FTABLE_ENTRY(CeedVector, Destroy),
      CEED_FTABLE_ENTRY(CeedElemRestriction, Apply),
      CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyUnsigned),
      CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyUnoriented),
      CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyAtPointsInElement),
      CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyBlock),
      CEED_FTABLE_ENTRY(CeedElemRestriction, GetOffsets),
      CEED_FTABLE_ENTRY(CeedElemRestriction, GetOrientations),
      CEED_FTABLE_ENTRY(CeedElemRestriction, GetCurlOrientations),
      CEED_FTABLE_ENTRY(CeedElemRestriction, GetAtPointsElementOffset),
      CEED_FTABLE_ENTRY(CeedElemRestriction, Destroy),
      CEED_FTABLE_ENTRY(CeedBasis, Apply),
      CEED_FTABLE_ENTRY(CeedBasis, ApplyAdd),
      CEED_FTABLE_ENTRY(CeedBasis, ApplyAtPoints),
      CEED_FTABLE_ENTRY(CeedBasis, ApplyAddAtPoints),
      CEED_FTABLE_ENTRY(CeedBasis, Destroy),
      CEED_FTABLE_ENTRY(CeedTensorContract, Apply),
      CEED_FTABLE_ENTRY(CeedTensorContract, Destroy),
      CEED_FTABLE_ENTRY(CeedQFunction, Apply),
      CEED_FTABLE_ENTRY(CeedQFunction, SetCUDAUserFunction),
      CEED_FTABLE_ENTRY(CeedQFunction, SetHIPUserFunction),
      CEED_FTABLE_ENTRY(CeedQFunction, Destroy),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, HasValidData),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, HasBorrowedDataOfType),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, SetData),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, TakeData),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, GetData),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, GetDataRead),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, RestoreData),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, RestoreDataRead),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, DataDestroy),
      CEED_FTABLE_ENTRY(CeedQFunctionContext, Destroy),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleQFunction),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleQFunctionUpdate),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleDiagonal),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleAddDiagonal),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssemblePointBlockDiagonal),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleAddPointBlockDiagonal),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleSymbolic),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssemble),
      CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleSingle),
      CEED_FTABLE_ENTRY(CeedOperator, CreateFDMElementInverse),
      CEED_FTABLE_ENTRY(CeedOperator, Apply),
      CEED_FTABLE_ENTRY(CeedOperator, ApplyComposite),
      CEED_FTABLE_ENTRY(CeedOperator, ApplyAdd),
      CEED_FTABLE_ENTRY(CeedOperator, ApplyAddComposite),
      CEED_FTABLE_ENTRY(CeedOperator, ApplyJacobian),
      CEED_FTABLE_ENTRY(CeedOperator, Destroy),
      {NULL, 0}  // End of lookup table - used in SetBackendFunction loop
  };

  CeedCall(CeedCalloc(sizeof(f_offsets), &(*ceed)->f_offsets));
  memcpy((*ceed)->f_offsets, f_offsets, sizeof(f_offsets));

  // Set fallback for advanced CeedOperator functions
  const char fallback_resource[] = "";
  CeedCall(CeedSetOperatorFallbackResource(*ceed, fallback_resource));

  // Record env variables CEED_DEBUG or DBG
  (*ceed)->is_debug = getenv("CEED_DEBUG") || getenv("DEBUG") || getenv("DBG");

  // Copy resource prefix, if backend setup successful
  CeedCall(CeedStringAllocCopy(backends[match_index].prefix, (char **)&(*ceed)->resource));

  // Set default JiT source root
  // Note: there will always be the default root for every Ceed but all additional paths are added to the top-most parent
  CeedCall(CeedAddJitSourceRoot(*ceed, (char *)CeedJitSourceRootDefault));

  // Backend specific setup
  CeedCall(backends[match_index].init(&resource[match_help], *ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the GPU stream for a `Ceed` context

  @param[in,out] ceed   `Ceed` context to set the stream
  @param[in]     handle Handle to GPU stream

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedSetStream(Ceed ceed, void *handle) {
  CeedCheck(handle, ceed, CEED_ERROR_INCOMPATIBLE, "Stream handle must be non-NULL");
  if (ceed->SetStream) {
    CeedCall(ceed->SetStream(ceed, handle));
  } else {
    Ceed delegate;
    CeedCall(CeedGetDelegate(ceed, &delegate));

    if (delegate) CeedCall(CeedSetStream(delegate, handle));
    else return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support setting stream");
    CeedCall(CeedDestroy(&delegate));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `Ceed` context.

  Both pointers should be destroyed with @ref CeedDestroy().

  Note: If the value of `*ceed_copy` passed to this function is non-`NULL`, then it is assumed that `*ceed_copy` is a pointer to a `Ceed` context.
        This `Ceed` context will be destroyed if `*ceed_copy` is the only reference to this `Ceed` context.

  @param[in]     ceed      `Ceed` context to copy reference to
  @param[in,out] ceed_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedReferenceCopy(Ceed ceed, Ceed *ceed_copy) {
  CeedCall(CeedReference(ceed));
  CeedCall(CeedDestroy(ceed_copy));
  *ceed_copy = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the full resource name for a `Ceed` context

  @param[in]  ceed     `Ceed` context to get resource name of
  @param[out] resource Variable to store resource name

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedGetResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->resource;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return `Ceed` context preferred memory type

  @param[in]  ceed     `Ceed` context to get preferred memory type of
  @param[out] mem_type Address to save preferred memory type to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedGetPreferredMemType(Ceed ceed, CeedMemType *mem_type) {
  if (ceed->GetPreferredMemType) {
    CeedCall(ceed->GetPreferredMemType(mem_type));
  } else {
    Ceed delegate;
    CeedCall(CeedGetDelegate(ceed, &delegate));

    if (delegate) {
      CeedCall(CeedGetPreferredMemType(delegate, mem_type));
    } else {
      *mem_type = CEED_MEM_HOST;
    }
    CeedCall(CeedDestroy(&delegate));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get deterministic status of `Ceed` context

  @param[in]  ceed             `Ceed` context
  @param[out] is_deterministic Variable to store deterministic status

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedIsDeterministic(Ceed ceed, bool *is_deterministic) {
  *is_deterministic = ceed->is_deterministic;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set additional JiT source root for `Ceed` context

  @param[in,out] ceed            `Ceed` context
  @param[in]     jit_source_root Absolute path to additional JiT source directory

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedAddJitSourceRoot(Ceed ceed, const char *jit_source_root) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  CeedCheck(!ceed_parent->num_jit_source_roots_readers, ceed, CEED_ERROR_ACCESS, "Cannot add JiT source root, read access has not been restored");

  CeedInt index       = ceed_parent->num_jit_source_roots;
  size_t  path_length = strlen(jit_source_root);

  if (ceed_parent->num_jit_source_roots == ceed_parent->max_jit_source_roots) {
    if (ceed_parent->max_jit_source_roots == 0) ceed_parent->max_jit_source_roots = 1;
    ceed_parent->max_jit_source_roots *= 2;
    CeedCall(CeedRealloc(ceed_parent->max_jit_source_roots, &ceed_parent->jit_source_roots));
  }
  CeedCall(CeedCalloc(path_length + 1, &ceed_parent->jit_source_roots[index]));
  memcpy(ceed_parent->jit_source_roots[index], jit_source_root, path_length);
  ceed_parent->num_jit_source_roots++;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set additional JiT compiler define for `Ceed` context

  @param[in,out] ceed       `Ceed` context
  @param[in]     jit_define String such as `foo=bar`, used as `-Dfoo=bar` in JiT

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedAddJitDefine(Ceed ceed, const char *jit_define) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));
  CeedCheck(!ceed_parent->num_jit_defines_readers, ceed, CEED_ERROR_ACCESS, "Cannot add JiT define, read access has not been restored");

  CeedInt index         = ceed_parent->num_jit_defines;
  size_t  define_length = strlen(jit_define);

  if (ceed_parent->num_jit_defines == ceed_parent->max_jit_defines) {
    if (ceed_parent->max_jit_defines == 0) ceed_parent->max_jit_defines = 1;
    ceed_parent->max_jit_defines *= 2;
    CeedCall(CeedRealloc(ceed_parent->max_jit_defines, &ceed_parent->jit_defines));
  }
  CeedCall(CeedCalloc(define_length + 1, &ceed_parent->jit_defines[index]));
  memcpy(ceed_parent->jit_defines[index], jit_define, define_length);
  ceed_parent->num_jit_defines++;
  CeedCall(CeedDestroy(&ceed_parent));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a `Ceed`

  @param[in] ceed   `Ceed` to view
  @param[in] stream Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedView(Ceed ceed, FILE *stream) {
  CeedMemType mem_type;

  CeedCall(CeedGetPreferredMemType(ceed, &mem_type));

  fprintf(stream,
          "Ceed\n"
          "  Ceed Resource: %s\n"
          "  Preferred MemType: %s\n",
          ceed->resource, CeedMemTypes[mem_type]);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `Ceed`

  @param[in,out] ceed Address of `Ceed` context to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedDestroy(Ceed *ceed) {
  if (!*ceed || --(*ceed)->ref_count > 0) {
    *ceed = NULL;
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(!(*ceed)->num_jit_source_roots_readers, *ceed, CEED_ERROR_ACCESS,
            "Cannot destroy ceed context, read access for JiT source roots has been granted");
  CeedCheck(!(*ceed)->num_jit_defines_readers, *ceed, CEED_ERROR_ACCESS, "Cannot add JiT source root, read access for JiT defines has been granted");

  if ((*ceed)->delegate) CeedCall(CeedDestroy(&(*ceed)->delegate));

  if ((*ceed)->obj_delegate_count > 0) {
    for (CeedInt i = 0; i < (*ceed)->obj_delegate_count; i++) {
      CeedCall(CeedDestroy(&((*ceed)->obj_delegates[i].delegate)));
      CeedCall(CeedFree(&(*ceed)->obj_delegates[i].obj_name));
    }
    CeedCall(CeedFree(&(*ceed)->obj_delegates));
  }

  if ((*ceed)->Destroy) CeedCall((*ceed)->Destroy(*ceed));

  for (CeedInt i = 0; i < (*ceed)->num_jit_source_roots; i++) {
    CeedCall(CeedFree(&(*ceed)->jit_source_roots[i]));
  }
  CeedCall(CeedFree(&(*ceed)->jit_source_roots));

  for (CeedInt i = 0; i < (*ceed)->num_jit_defines; i++) {
    CeedCall(CeedFree(&(*ceed)->jit_defines[i]));
  }
  CeedCall(CeedFree(&(*ceed)->jit_defines));

  CeedCall(CeedFree(&(*ceed)->f_offsets));
  CeedCall(CeedFree(&(*ceed)->resource));
  CeedCall(CeedDestroy(&(*ceed)->op_fallback_ceed));
  CeedCall(CeedFree(&(*ceed)->op_fallback_resource));
  CeedCall(CeedWorkVectorsDestroy(*ceed));
  CeedCall(CeedFree(ceed));
  return CEED_ERROR_SUCCESS;
}

// LCOV_EXCL_START
const char *CeedErrorFormat(Ceed ceed, const char *format, va_list *args) {
  if (ceed->parent) return CeedErrorFormat(ceed->parent, format, args);
  if (ceed->op_fallback_parent) return CeedErrorFormat(ceed->op_fallback_parent, format, args);
  // Using pointer to va_list for better FFI, but clang-tidy can't verify va_list is initalized
  vsnprintf(ceed->err_msg, CEED_MAX_RESOURCE_LEN, format, *args);  // NOLINT
  return ceed->err_msg;
}
// LCOV_EXCL_STOP

/**
  @brief Error handling implementation; use @ref CeedError() instead.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func, int ecode, const char *format, ...) {
  va_list args;
  int     ret_val;

  va_start(args, format);
  if (ceed) {
    ret_val = ceed->Error(ceed, filename, lineno, func, ecode, format, &args);
  } else {
    // LCOV_EXCL_START
    const char *ceed_error_handler = getenv("CEED_ERROR_HANDLER");
    if (!ceed_error_handler) ceed_error_handler = "abort";
    if (!strcmp(ceed_error_handler, "return")) {
      ret_val = CeedErrorReturn(ceed, filename, lineno, func, ecode, format, &args);
    } else {
      // This function will not return
      ret_val = CeedErrorAbort(ceed, filename, lineno, func, ecode, format, &args);
    }
  }
  va_end(args);
  return ret_val;
  // LCOV_EXCL_STOP
}

/**
  @brief Error handler that returns without printing anything.

  Pass this to @ref CeedSetErrorHandler() to obtain this error handling behavior.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorReturn(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  return err_code;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that stores the error message for future use and returns the error.

  Pass this to @ref CeedSetErrorHandler() to obtain this error handling behavior.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorStore(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  if (ceed->parent) return CeedErrorStore(ceed->parent, filename, line_no, func, err_code, format, args);
  if (ceed->op_fallback_parent) return CeedErrorStore(ceed->op_fallback_parent, filename, line_no, func, err_code, format, args);

  // Build message
  int len = snprintf(ceed->err_msg, CEED_MAX_RESOURCE_LEN, "%s:%d in %s(): ", filename, line_no, func);
  // Using pointer to va_list for better FFI, but clang-tidy can't verify va_list is initalized
  vsnprintf(ceed->err_msg + len, CEED_MAX_RESOURCE_LEN - len, format, *args);  // NOLINT
  return err_code;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that prints to `stderr` and aborts

  Pass this to @ref CeedSetErrorHandler() to obtain this error handling behavior.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorAbort(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, line_no, func);
  vfprintf(stderr, format, *args);
  fprintf(stderr, "\n");
  abort();
  return err_code;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that prints to `stderr` and exits.

  Pass this to @ref CeedSetErrorHandler() to obtain this error handling behavior.

  In contrast to @ref CeedErrorAbort(), this exits without a signal, so `atexit()` handlers (e.g., as used by gcov) are run.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedErrorExit(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, line_no, func);
  // Using pointer to va_list for better FFI, but clang-tidy can't verify va_list is initalized
  vfprintf(stderr, format, *args);  // NOLINT
  fprintf(stderr, "\n");
  exit(err_code);
  return err_code;
}

/**
  @brief Set error handler

  A default error handler is set in @ref CeedInit().
  Use this function to change the error handler to @ref CeedErrorReturn(), @ref CeedErrorAbort(), or a user-defined error handler.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedSetErrorHandler(Ceed ceed, CeedErrorHandler handler) {
  ceed->Error = handler;
  if (ceed->delegate) CeedSetErrorHandler(ceed->delegate, handler);
  for (CeedInt i = 0; i < ceed->obj_delegate_count; i++) CeedSetErrorHandler(ceed->obj_delegates[i].delegate, handler);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get error message

  The error message is only stored when using the error handler @ref CeedErrorStore()

  @param[in]  ceed    `Ceed` context to retrieve error message
  @param[out] err_msg Char pointer to hold error message

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedGetErrorMessage(Ceed ceed, const char **err_msg) {
  if (ceed->parent) return CeedGetErrorMessage(ceed->parent, err_msg);
  if (ceed->op_fallback_parent) return CeedGetErrorMessage(ceed->op_fallback_parent, err_msg);
  *err_msg = ceed->err_msg;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore error message.

  The error message is only stored when using the error handler @ref CeedErrorStore().

  @param[in]  ceed    `Ceed` context to restore error message
  @param[out] err_msg Char pointer that holds error message

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedResetErrorMessage(Ceed ceed, const char **err_msg) {
  if (ceed->parent) return CeedResetErrorMessage(ceed->parent, err_msg);
  if (ceed->op_fallback_parent) return CeedResetErrorMessage(ceed->op_fallback_parent, err_msg);
  *err_msg = NULL;
  memcpy(ceed->err_msg, "No error message stored", 24);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get libCEED library version information.

  libCEED version numbers have the form major.minor.patch.
  Non-release versions may contain unstable interfaces.

  @param[out] major   Major version of the library
  @param[out] minor   Minor version of the library
  @param[out] patch   Patch (subminor) version of the library
  @param[out] release True for releases; false for development branches

  The caller may pass `NULL` for any arguments that are not needed.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer

  @sa CEED_VERSION_GE() CeedGetGitVersion() CeedGetBuildConfiguration()
*/
int CeedGetVersion(int *major, int *minor, int *patch, bool *release) {
  if (major) *major = CEED_VERSION_MAJOR;
  if (minor) *minor = CEED_VERSION_MINOR;
  if (patch) *patch = CEED_VERSION_PATCH;
  if (release) *release = CEED_VERSION_RELEASE;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get libCEED scalar type, such as F64 or F32

  @param[out] scalar_type Type of libCEED scalars

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
*/
int CeedGetScalarType(CeedScalarType *scalar_type) {
  *scalar_type = CEED_SCALAR_TYPE;
  return CEED_ERROR_SUCCESS;
}

/// @}
