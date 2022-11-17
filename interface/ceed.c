// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#define _POSIX_C_SOURCE 200112
#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <limits.h>
#include <stdarg.h>
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

#define CEED_FTABLE_ENTRY(class, method) \
  { #class #method, offsetof(struct class##_private, method) }
/// @endcond

/// @file
/// Implementation of core components of Ceed library

/// @addtogroup CeedUser
/// @{

/**
  @brief Request immediate completion

  This predefined constant is passed as the \ref CeedRequest argument to
  interfaces when the caller wishes for the operation to be performed
  immediately. The code

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

  This predefined constant is passed as the \ref CeedRequest argument to
  interfaces when the caller wishes for the operation to be completed in the
  order that it is submitted to the device. It is typically used in a construct
  such as

  @code
    CeedRequest request;
    CeedOperatorApply(op1, ..., CEED_REQUEST_ORDERED);
    CeedOperatorApply(op2, ..., &request);
    // other optional work
    CeedRequestWait(&request);
  @endcode

  which allows the sequence to complete asynchronously but does not start
  `op2` until `op1` has completed.

  @todo The current implementation is overly strict, offering equivalent
  semantics to @ref CEED_REQUEST_IMMEDIATE.

  @sa CEED_REQUEST_IMMEDIATE
 */
CeedRequest *const CEED_REQUEST_ORDERED = &ceed_request_ordered;

/**
  @brief Wait for a CeedRequest to complete.

  Calling CeedRequestWait on a NULL request is a no-op.

  @param req Address of CeedRequest to wait for; zeroed on completion.

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
           Note: Backends should call `CeedRegister` instead.

  @param prefix    Prefix of resources for this backend to respond to.  For
                     example, the reference backend responds to "/cpu/self".
  @param init      Initialization function called by CeedInit() when the backend
                     is selected to drive the requested resource.
  @param priority  Integer priority.  Lower values are preferred in case the
                     resource requested by CeedInit() has non-unique best prefix
                     match.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedRegisterImpl(const char *prefix, int (*init)(const char *, Ceed), unsigned int priority) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0]))
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "Too many backends");
  // LCOV_EXCL_STOP

  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].prefix[CEED_MAX_RESOURCE_LEN - 1] = 0;
  backends[num_backends].init                              = init;
  backends[num_backends].priority                          = priority;
  num_backends++;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBackend
/// @{

/**
  @brief Return value of CEED_DEBUG environment variable

  @param ceed    Ceed context

  @return boolean value: true  - debugging mode enabled
                         false - debugging mode disabled

  @ref Backend
**/
// LCOV_EXCL_START
bool CeedDebugFlag(const Ceed ceed) { return ceed->is_debug; }
// LCOV_EXCL_STOP

/**
  @brief Return value of CEED_DEBUG environment variable

  @return boolean value: true  - debugging mode enabled
                         false - debugging mode disabled

  @ref Backend
**/
// LCOV_EXCL_START
bool CeedDebugFlagEnv(void) { return !!getenv("CEED_DEBUG") || !!getenv("DEBUG") || !!getenv("DBG"); }
// LCOV_EXCL_STOP

/**
  @brief Print debugging information in color

  @param color   Color to print
  @param format  Printing format

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
  @brief Allocate an array on the host; use CeedMalloc()

  Memory usage can be tracked by the library.  This ensures sufficient
    alignment for vectorization and should be used for large allocations.

  @param n     Number of units to allocate
  @param unit  Size of each unit
  @param p     Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n * unit);
  if (ierr) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Allocate a cleared (zeroed) array on the host; use CeedCalloc()

  Memory usage can be tracked by the library.

  @param n     Number of units to allocate
  @param unit  Size of each unit
  @param p     Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "calloc failed to allocate %zd members of size %zd\n", n, unit);
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Reallocate an array on the host; use CeedRealloc()

  Memory usage can be tracked by the library.

  @param n     Number of units to allocate
  @param unit  Size of each unit
  @param p     Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n * unit);
  if (n && unit && !*(void **)p) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "realloc failed to allocate %zd members of size %zd\n", n, unit);
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Allocate a cleared string buffer on the host

  Memory usage can be tracked by the library.

  @param source Pointer to string to be copied
  @param copy   Pointer to variable to hold newly allocated string copy

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedStringAllocCopy(const char *source, char **copy) {
  size_t len = strlen(source);
  CeedCall(CeedCalloc(len + 1, copy));
  memcpy(*copy, source, len);
  return CEED_ERROR_SUCCESS;
}

/** Free memory allocated using CeedMalloc() or CeedCalloc()

  @param p  address of pointer to memory.  This argument is of type void* to
              avoid needing a cast, but is the address of the pointer (which is
              zeroed) rather than the pointer.
**/
int CeedFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Register a Ceed backend

  @param prefix    Prefix of resources for this backend to respond to.  For
                     example, the reference backend responds to "/cpu/self".
  @param init      Initialization function called by CeedInit() when the backend
                     is selected to drive the requested resource.
  @param priority  Integer priority.  Lower values are preferred in case the
                     resource requested by CeedInit() has non-unique best prefix
                     match.

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

  @param ceed      Ceed context to get debugging flag
  @param is_debug  Variable to store debugging flag

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedIsDebug(Ceed ceed, bool *is_debug) {
  *is_debug = ceed->is_debug;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a parent Ceed context

  @param ceed         Ceed context to retrieve parent of
  @param[out] parent  Address to save the parent to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetParent(Ceed ceed, Ceed *parent) {
  if (ceed->parent) {
    CeedCall(CeedGetParent(ceed->parent, parent));
    return CEED_ERROR_SUCCESS;
  }
  *parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate Ceed context

  @param ceed           Ceed context to retrieve delegate of
  @param[out] delegate  Address to save the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetDelegate(Ceed ceed, Ceed *delegate) {
  *delegate = ceed->delegate;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a delegate Ceed context

  This function allows a Ceed context to set a delegate Ceed context. All
    backend implementations default to the delegate Ceed context, unless
    overridden.

  @param ceed           Ceed context to set delegate of
  @param[out] delegate  Address to set the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetDelegate(Ceed ceed, Ceed delegate) {
  ceed->delegate   = delegate;
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate Ceed context for a specific object type

  @param ceed           Ceed context to retrieve delegate of
  @param[out] delegate  Address to save the delegate to
  @param[in] obj_name   Name of the object type to retrieve delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate, const char *obj_name) {
  // Check for object delegate
  for (CeedInt i = 0; i < ceed->obj_delegate_count; i++) {
    if (!strcmp(obj_name, ceed->obj_delegates->obj_name)) {
      *delegate = ceed->obj_delegates->delegate;
      return CEED_ERROR_SUCCESS;
    }
  }

  // Use default delegate if no object delegate
  CeedCall(CeedGetDelegate(ceed, delegate));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a delegate Ceed context for a specific object type

  This function allows a Ceed context to set a delegate Ceed context for a
    given type of Ceed object. All backend implementations default to the
    delegate Ceed context for this object. For example,
    CeedSetObjectDelegate(ceed, refceed, "Basis")
  uses refceed implementations for all CeedBasis backend functions.

  @param ceed           Ceed context to set delegate of
  @param[out] delegate  Address to set the delegate to
  @param[in] obj_name   Name of the object type to set delegate for

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
  ceed->obj_delegates[count].delegate = delegate;
  CeedCall(CeedStringAllocCopy(obj_name, &ceed->obj_delegates[count].obj_name));

  // Set delegate parent
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the fallback resource for CeedOperators

  @param ceed           Ceed context
  @param[out] resource  Variable to store fallback resource

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedGetOperatorFallbackResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->op_fallback_resource;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the fallback Ceed for CeedOperators

  @param ceed                Ceed context
  @param[out] fallback_ceed  Variable to store fallback Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedGetOperatorFallbackCeed(Ceed ceed, Ceed *fallback_ceed) {
  if (ceed->has_valid_op_fallback_resource) {
    CeedDebug256(ceed, 1, "---------- CeedOperator Fallback ----------\n");
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
  }
  *fallback_ceed = ceed->op_fallback_ceed;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the fallback resource for CeedOperators. The current resource, if
           any, is freed by calling this function. This string is freed upon the
           destruction of the Ceed context.

  @param[out] ceed Ceed context
  @param resource  Fallback resource to set

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
  @brief Get the parent Ceed context associated with a fallback Ceed context
           for a CeedOperator

  @param ceed         Ceed context
  @param[out] parent  Variable to store parent Ceed context

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedGetOperatorFallbackParentCeed(Ceed ceed, Ceed *parent) {
  *parent = ceed->op_fallback_parent;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Flag Ceed context as deterministic

  @param ceed                   Ceed to flag as deterministic
  @param[out] is_deterministic  Deterministic status to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedSetDeterministic(Ceed ceed, bool is_deterministic) {
  ceed->is_deterministic = is_deterministic;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a backend function

  This function is used for a backend to set the function associated with
  the Ceed objects. For example,
    CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", BackendVectorCreate)
  sets the backend implementation of 'CeedVectorCreate' and
    CeedSetBackendFunction(ceed, "Basis", basis, "Apply", BackendBasisApply)
  sets the backend implementation of 'CeedBasisApply'. Note, the prefix 'Ceed'
  is not required for the object type ("Basis" vs "CeedBasis").

  @param ceed         Ceed context for error handling
  @param type         Type of Ceed object to set function for
  @param[out] object  Ceed object to set function for
  @param func_name    Name of function to set
  @param f            Function to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetBackendFunction(Ceed ceed, const char *type, void *object, const char *func_name, int (*f)()) {
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
      *fpointer              = f;
      return CEED_ERROR_SUCCESS;
    }
  }

  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Requested function '%s' was not found for CEED object '%s'", func_name, type);
  // LCOV_EXCL_STOP
}

/**
  @brief Retrieve backend data for a Ceed context

  @param ceed       Ceed context to retrieve data of
  @param[out] data  Address to save data to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetData(Ceed ceed, void *data) {
  *(void **)data = ceed->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data for a Ceed context

  @param ceed  Ceed context to set data of
  @param data  Address of data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetData(Ceed ceed, void *data) {
  ceed->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a Ceed context

  @param ceed  Ceed context to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedReference(Ceed ceed) {
  ceed->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedUser
/// @{

/**
  @brief Get the list of available resource names for Ceed contexts
  Note: The caller is responsible for `free()`ing the resources and priorities arrays,
          but should not `free()` the contents of the resources array.

  @param[out] n           Number of available resources
  @param[out] resources   List of available resource names
  @param[out] priorities  Resource name prioritization values, lower is better

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
// LCOV_EXCL_START
int CeedRegistryGetList(size_t *n, char ***const resources, CeedInt **priorities) {
  *n         = 0;
  *resources = malloc(num_backends * sizeof(**resources));
  if (!resources) return CeedError(NULL, CEED_ERROR_MAJOR, "malloc() failure");
  if (priorities) {
    *priorities = malloc(num_backends * sizeof(**priorities));
    if (!priorities) return CeedError(NULL, CEED_ERROR_MAJOR, "malloc() failure");
  }
  for (size_t i = 0; i < num_backends; i++) {
    // Only report compiled backends
    if (backends[i].priority < CEED_MAX_BACKEND_PRIORITY) {
      *resources[i] = backends[i].prefix;
      if (priorities) *priorities[i] = backends[i].priority;
      *n += 1;
    }
  }
  if (*n == 0) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "No backends installed");
    // LCOV_EXCL_STOP
  }
  *resources = realloc(*resources, *n * sizeof(**resources));
  if (!resources) return CeedError(NULL, CEED_ERROR_MAJOR, "realloc() failure");
  if (priorities) {
    *priorities = realloc(*priorities, *n * sizeof(**priorities));
    if (!priorities) return CeedError(NULL, CEED_ERROR_MAJOR, "realloc() failure");
  }
  return CEED_ERROR_SUCCESS;
}
// LCOV_EXCL_STOP

/**
  @brief Initialize a \ref Ceed context to use the specified resource.
  Note: Prefixing the resource with "help:" (e.g. "help:/cpu/self")
    will result in CeedInt printing the current libCEED version number
    and a list of current available backend resources to stderr.

  @param resource  Resource to use, e.g., "/cpu/self"
  @param ceed      The library context
  @sa CeedRegister() CeedDestroy()

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedInit(const char *resource, Ceed *ceed) {
  size_t match_len = 0, match_index = UINT_MAX, match_priority = CEED_MAX_BACKEND_PRIORITY, priority;

  // Find matching backend
  if (!resource) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "No resource provided");
    // LCOV_EXCL_STOP
  }
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

  // Find best match, computed as number of matching characters
  //   from requested resource stem
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
    if (lev_dis + 1 >= m) {
      return CeedError(NULL, CEED_ERROR_MAJOR, "No suitable backend: %s", resource);
    } else {
      return CeedError(NULL, CEED_ERROR_MAJOR,
                       "No suitable backend: %s\n"
                       "Closest match: %s",
                       resource, backends[lev_index].prefix);
    }
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
      CEED_FTABLE_ENTRY(Ceed, GetPreferredMemType),
      CEED_FTABLE_ENTRY(Ceed, Destroy),
      CEED_FTABLE_ENTRY(Ceed, VectorCreate),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreate),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreateOriented),
      CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreateBlocked),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateTensorH1),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateH1),
      CEED_FTABLE_ENTRY(Ceed, BasisCreateHdiv),
      CEED_FTABLE_ENTRY(Ceed, TensorContractCreate),
      CEED_FTABLE_ENTRY(Ceed, QFunctionCreate),
      CEED_FTABLE_ENTRY(Ceed, QFunctionContextCreate),
      CEED_FTABLE_ENTRY(Ceed, OperatorCreate),
      CEED_FTABLE_ENTRY(Ceed, CompositeOperatorCreate),
      CEED_FTABLE_ENTRY(CeedVector, HasValidArray),
      CEED_FTABLE_ENTRY(CeedVector, HasBorrowedArrayOfType),
      CEED_FTABLE_ENTRY(CeedVector, SetArray),
      CEED_FTABLE_ENTRY(CeedVector, TakeArray),
      CEED_FTABLE_ENTRY(CeedVector, SetValue),
      CEED_FTABLE_ENTRY(CeedVector, SyncArray),
      CEED_FTABLE_ENTRY(CeedVector, GetArray),
      CEED_FTABLE_ENTRY(CeedVector, GetArrayRead),
      CEED_FTABLE_ENTRY(CeedVector, GetArrayWrite),
      CEED_FTABLE_ENTRY(CeedVector, RestoreArray),
      CEED_FTABLE_ENTRY(CeedVector, RestoreArrayRead),
      CEED_FTABLE_ENTRY(CeedVector, Norm),
      CEED_FTABLE_ENTRY(CeedVector, Scale),
      CEED_FTABLE_ENTRY(CeedVector, AXPY),
      CEED_FTABLE_ENTRY(CeedVector, PointwiseMult),
      CEED_FTABLE_ENTRY(CeedVector, Reciprocal),
      CEED_FTABLE_ENTRY(CeedVector, Destroy),
      CEED_FTABLE_ENTRY(CeedElemRestriction, Apply),
      CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyBlock),
      CEED_FTABLE_ENTRY(CeedElemRestriction, GetOffsets),
      CEED_FTABLE_ENTRY(CeedElemRestriction, Destroy),
      CEED_FTABLE_ENTRY(CeedBasis, Apply),
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
  const char fallbackresource[] = "";
  CeedCall(CeedSetOperatorFallbackResource(*ceed, fallbackresource));

  // Record env variables CEED_DEBUG or DBG
  (*ceed)->is_debug = !!getenv("CEED_DEBUG") || !!getenv("DEBUG") || !!getenv("DBG");

  // Copy resource prefix, if backend setup successful
  CeedCall(CeedStringAllocCopy(backends[match_index].prefix, (char **)&(*ceed)->resource));

  // Set default JiT source root
  // Note: there will always be the default root for every Ceed
  // but all additional paths are added to the top-most parent
  CeedCall(CeedAddJitSourceRoot(*ceed, (char *)CeedJitSourceRootDefault));

  // Backend specific setup
  CeedCall(backends[match_index].init(&resource[match_help], *ceed));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a Ceed context. Both pointers should
           be destroyed with `CeedDestroy()`;
           Note: If `*ceed_copy` is non-NULL, then it is assumed that
           `*ceed_copy` is a pointer to a Ceed context. This Ceed
           context will be destroyed if `*ceed_copy` is the only
           reference to this Ceed context.

  @param ceed            Ceed context to copy reference to
  @param[out] ceed_copy  Variable to store copied reference

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
  @brief Get the full resource name for a Ceed context

  @param ceed           Ceed context to get resource name of
  @param[out] resource  Variable to store resource name

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedGetResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->resource;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return Ceed context preferred memory type

  @param ceed           Ceed context to get preferred memory type of
  @param[out] mem_type  Address to save preferred memory type to

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
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get deterministic status of Ceed

  @param[in] ceed               Ceed
  @param[out] is_deterministic  Variable to store deterministic status

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedIsDeterministic(Ceed ceed, bool *is_deterministic) {
  *is_deterministic = ceed->is_deterministic;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set additional JiT source root for Ceed

  @param[in] ceed            Ceed
  @param[in] jit_source_root Absolute path to additional JiT source directory

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedAddJitSourceRoot(Ceed ceed, const char *jit_source_root) {
  Ceed ceed_parent;

  CeedCall(CeedGetParent(ceed, &ceed_parent));

  CeedInt index       = ceed_parent->num_jit_source_roots;
  size_t  path_length = strlen(jit_source_root);
  CeedCall(CeedRealloc(index + 1, &ceed_parent->jit_source_roots));
  CeedCall(CeedCalloc(path_length + 1, &ceed_parent->jit_source_roots[index]));
  memcpy(ceed_parent->jit_source_roots[index], jit_source_root, path_length);
  ceed_parent->num_jit_source_roots++;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a Ceed

  @param[in] ceed    Ceed to view
  @param[in] stream  Filestream to write to

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
  @brief Destroy a Ceed context

  @param ceed  Address of Ceed context to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedDestroy(Ceed *ceed) {
  if (!*ceed || --(*ceed)->ref_count > 0) return CEED_ERROR_SUCCESS;
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

  CeedCall(CeedFree(&(*ceed)->f_offsets));
  CeedCall(CeedFree(&(*ceed)->resource));
  CeedCall(CeedDestroy(&(*ceed)->op_fallback_ceed));
  CeedCall(CeedFree(&(*ceed)->op_fallback_resource));
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
  @brief Error handling implementation; use \ref CeedError instead.

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
    if (!strcmp(ceed_error_handler, "return")) ret_val = CeedErrorReturn(ceed, filename, lineno, func, ecode, format, &args);
    else
      // This function will not return
      ret_val = CeedErrorAbort(ceed, filename, lineno, func, ecode, format, &args);
  }
  va_end(args);
  return ret_val;
  // LCOV_EXCL_STOP
}

/**
  @brief Error handler that returns without printing anything.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorReturn(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  return err_code;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that stores the error message for future use and returns
           the error.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorStore(Ceed ceed, const char *filename, int line_no, const char *func, int err_code, const char *format, va_list *args) {
  if (ceed->parent) return CeedErrorStore(ceed->parent, filename, line_no, func, err_code, format, args);
  if (ceed->op_fallback_parent) return CeedErrorStore(ceed->op_fallback_parent, filename, line_no, func, err_code, format, args);

  // Build message
  int len;
  len = snprintf(ceed->err_msg, CEED_MAX_RESOURCE_LEN, "%s:%d in %s(): ", filename, line_no, func);
  // Using pointer to va_list for better FFI, but clang-tidy can't verify va_list is initalized
  // *INDENT-OFF*
  vsnprintf(ceed->err_msg + len, CEED_MAX_RESOURCE_LEN - len, format, *args);  // NOLINT
  // *INDENT-ON*
  return err_code;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that prints to stderr and aborts

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

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
  @brief Error handler that prints to stderr and exits

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  In contrast to CeedErrorAbort(), this exits without a signal, so atexit()
  handlers (e.g., as used by gcov) are run.

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

  A default error handler is set in CeedInit().  Use this function to change
  the error handler to CeedErrorReturn(), CeedErrorAbort(), or a user-defined
  error handler.

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

  The error message is only stored when using the error handler
    CeedErrorStore()

  @param[in] ceed      Ceed contex to retrieve error message
  @param[out] err_msg  Char pointer to hold error message

  @ref Developer
**/
int CeedGetErrorMessage(Ceed ceed, const char **err_msg) {
  if (ceed->parent) return CeedGetErrorMessage(ceed->parent, err_msg);
  if (ceed->op_fallback_parent) return CeedGetErrorMessage(ceed->op_fallback_parent, err_msg);
  *err_msg = ceed->err_msg;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore error message

  The error message is only stored when using the error handler
    CeedErrorStore()

  @param[in] ceed      Ceed contex to restore error message
  @param[out] err_msg  Char pointer that holds error message

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
  @brief Get libCEED library version info

  libCEED version numbers have the form major.minor.patch. Non-release versions
  may contain unstable interfaces.

  @param[out] major    Major version of the library
  @param[out] minor    Minor version of the library
  @param[out] patch    Patch (subminor) version of the library
  @param[out] release  True for releases; false for development branches.

  The caller may pass NULL for any arguments that are not needed.

  @sa CEED_VERSION_GE()

  @ref Developer
*/
int CeedGetVersion(int *major, int *minor, int *patch, bool *release) {
  if (major) *major = CEED_VERSION_MAJOR;
  if (minor) *minor = CEED_VERSION_MINOR;
  if (patch) *patch = CEED_VERSION_PATCH;
  if (release) *release = CEED_VERSION_RELEASE;
  return 0;
}

int CeedGetScalarType(CeedScalarType *scalar_type) {
  *scalar_type = CEED_SCALAR_TYPE;
  return 0;
}

/// @}
