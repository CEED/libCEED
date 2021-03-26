// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#define _POSIX_C_SOURCE 200112
#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
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
  {#class #method, offsetof(struct class ##_private, method)}
/// @endcond

/// @file
/// Implementation of core components of Ceed library

/// @addtogroup CeedUser
/// @{

/**
  @brief Request immediate completion

  This predefined constant is passed as the \ref CeedRequest argument to
  interfaces when the caller wishes for the operation to be performed
  immediately.  The code

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
  order that it is submitted to the device.  It is typically used in a construct
  such as

  @code
    CeedRequest request;
    CeedOperatorApply(op1, ..., CEED_REQUEST_ORDERED);
    CeedOperatorApply(op2, ..., &request);
    // other optional work
    CeedWait(&request);
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
  if (!*req)
    return CEED_ERROR_SUCCESS;
  return CeedError(NULL, CEED_ERROR_UNSUPPORTED,
                   "CeedRequestWait not implemented");
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedDeveloper
/// @{

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBackend
/// @{

/**
  @brief Print Ceed debugging information

  @param ceed    Ceed context
  @param format  Printing format

  @return None

  @ref Backend
**/
// LCOV_EXCL_START
void CeedDebugImpl(const Ceed ceed, const char *format,...) {
  if (!ceed->debug) return;
  va_list args;
  va_start(args, format);
  CeedDebugImpl256(ceed, 0, format, args);
  va_end(args);
}
// LCOV_EXCL_STOP

/**
  @brief Print Ceed debugging information in color

  @param ceed    Ceed context
  @param color   Color to print
  @param format  Printing format

  @return None

  @ref Backend
**/
// LCOV_EXCL_START
void CeedDebugImpl256(const Ceed ceed, const unsigned char color,
                      const char *format,...) {
  if (!ceed->debug) return;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout, "\033[38;5;%dm", color);
  vfprintf(stdout, format, args);
  fprintf(stdout, "\033[m");
  fprintf(stdout, "\n");
  fflush(stdout);
  va_end(args);
}
// LCOV_EXCL_STOP

/**
  @brief Allocate an array on the host; use CeedMalloc()

  Memory usage can be tracked by the library.  This ensures sufficient
    alignment for vectorization and should be used for large allocations.

  @param n Number of units to allocate
  @param unit Size of each unit
  @param p Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n*unit);
  if (ierr)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR,
                     "posix_memalign failed to allocate %zd "
                     "members of size %zd\n", n, unit);
  // LCOV_EXCL_STOP
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Allocate a cleared (zeroed) array on the host; use CeedCalloc()

  Memory usage can be tracked by the library.

  @param n    Number of units to allocate
  @param unit Size of each unit
  @param p    Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR,
                     "calloc failed to allocate %zd members of size "
                     "%zd\n", n, unit);
  // LCOV_EXCL_STOP
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Reallocate an array on the host; use CeedRealloc()

  Memory usage can be tracked by the library.

  @param n    Number of units to allocate
  @param unit Size of each unit
  @param p    Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Backend
**/
int CeedReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n*unit);
  if (n && unit && !*(void **)p)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR,
                     "realloc failed to allocate %zd members of size "
                     "%zd\n", n, unit);
  // LCOV_EXCL_STOP
  return CEED_ERROR_SUCCESS;
}

/** Free memory allocated using CeedMalloc() or CeedCalloc()

  @param p address of pointer to memory.  This argument is of type void* to
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

  @param prefix   Prefix of resources for this backend to respond to.  For
                    example, the reference backend responds to "/cpu/self".
  @param init     Initialization function called by CeedInit() when the backend
                    is selected to drive the requested resource.
  @param priority Integer priority.  Lower values are preferred in case the
                    resource requested by CeedInit() has non-unique best prefix
                    match.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedRegister(const char *prefix, int (*init)(const char *, Ceed),
                 unsigned int priority) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0]))
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "Too many backends");
  // LCOV_EXCL_STOP

  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].prefix[CEED_MAX_RESOURCE_LEN-1] = 0;
  backends[num_backends].init = init;
  backends[num_backends].priority = priority;
  num_backends++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return debugging status flag

  @param ceed     Ceed context to get debugging flag
  @param isDebug  Variable to store debugging flag

  @return An error code: 0 - success, otherwise - failure

  @ref Bcakend
**/
int CeedIsDebug(Ceed ceed, bool *isDebug) {
  *isDebug = ceed->debug;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a parent Ceed context

  @param ceed        Ceed context to retrieve parent of
  @param[out] parent Address to save the parent to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetParent(Ceed ceed, Ceed *parent) {
  int ierr;
  if (ceed->parent) {
    ierr = CeedGetParent(ceed->parent, parent); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }
  *parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate Ceed context

  @param ceed          Ceed context to retrieve delegate of
  @param[out] delegate Address to save the delegate to

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
  ceed->delegate = delegate;
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve a delegate Ceed context for a specific object type

  @param ceed           Ceed context to retrieve delegate of
  @param[out] delegate  Address to save the delegate to
  @param[in] objname    Name of the object type to retrieve delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate, const char *objname) {
  CeedInt ierr;

  // Check for object delegate
  for (CeedInt i=0; i<ceed->objdelegatecount; i++)
    if (!strcmp(objname, ceed->objdelegates->objname)) {
      *delegate = ceed->objdelegates->delegate;
      return CEED_ERROR_SUCCESS;
    }

  // Use default delegate if no object delegate
  ierr = CeedGetDelegate(ceed, delegate); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a delegate Ceed context for a specific object type

  This function allows a Ceed context to set a delegate Ceed context for a
    given type of Ceed object. All backend implementations default to the
    delegate Ceed context for this object. For example,
    CeedSetObjectDelegate(ceed, refceed, "Basis")
  uses refceed implementations for all CeedBasis backend functions.

  @param ceed          Ceed context to set delegate of
  @param[out] delegate Address to set the delegate to
  @param[in] objname   Name of the object type to set delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetObjectDelegate(Ceed ceed, Ceed delegate, const char *objname) {
  CeedInt ierr;
  CeedInt count = ceed->objdelegatecount;

  // Malloc or Realloc
  if (count) {
    ierr = CeedRealloc(count+1, &ceed->objdelegates); CeedChk(ierr);
  } else {
    ierr = CeedCalloc(1, &ceed->objdelegates); CeedChk(ierr);
  }
  ceed->objdelegatecount++;

  // Set object delegate
  ceed->objdelegates[count].delegate = delegate;
  size_t slen = strlen(objname) + 1;
  ierr = CeedMalloc(slen, &ceed->objdelegates[count].objname); CeedChk(ierr);
  memcpy(ceed->objdelegates[count].objname, objname, slen);

  // Set delegate parent
  delegate->parent = ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the fallback resource for CeedOperators

  @param ceed          Ceed context
  @param[out] resource Variable to store fallback resource

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedGetOperatorFallbackResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->opfallbackresource;
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
  int ierr;

  // Free old
  ierr = CeedFree(&ceed->opfallbackresource); CeedChk(ierr);

  // Set new
  size_t len = strlen(resource);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, resource, len+1);
  ceed->opfallbackresource = tmp;
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
  *parent = ceed->opfallbackparent;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Flag Ceed context as deterministic

  @param ceed     Ceed to flag as deterministic

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedSetDeterministic(Ceed ceed, bool isDeterministic) {
  ceed->isDeterministic = isDeterministic;
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

  @param ceed           Ceed context for error handling
  @param type           Type of Ceed object to set function for
  @param[out] object    Ceed object to set function for
  @param fname          Name of function to set
  @param f              Function to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetBackendFunction(Ceed ceed, const char *type, void *object,
                           const char *fname, int (*f)()) {
  char lookupname[CEED_MAX_RESOURCE_LEN+1] = "";

  // Build lookup name
  if (strcmp(type, "Ceed"))
    strncat (lookupname, "Ceed", CEED_MAX_RESOURCE_LEN);
  strncat(lookupname, type, CEED_MAX_RESOURCE_LEN);
  strncat(lookupname, fname, CEED_MAX_RESOURCE_LEN);

  // Find and use offset
  for (CeedInt i = 0; ceed->foffsets[i].fname; i++)
    if (!strcmp(ceed->foffsets[i].fname, lookupname)) {
      size_t offset = ceed->foffsets[i].offset;
      int (**fpointer)(void) = (int (**)(void))((char *)object + offset); // *NOPAD*
      *fpointer = f;
      return CEED_ERROR_SUCCESS;
    }

  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                   "Requested function '%s' was not found for CEED "
                   "object '%s'", fname, type);
  // LCOV_EXCL_STOP
}

/**
  @brief Retrieve backend data for a Ceed context

  @param ceed      Ceed context to retrieve data of
  @param[out] data Address to save data to

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedGetData(Ceed ceed, void *data) {
  *(void **)data = ceed->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data for a Ceed context

  @param ceed           Ceed context to set data of
  @param data           Address of data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedSetData(Ceed ceed, void *data) {
  ceed->data = data;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedUser
/// @{

/**
  @brief Initialize a \ref Ceed context to use the specified resource.

  @param resource  Resource to use, e.g., "/cpu/self"
  @param ceed      The library context
  @sa CeedRegister() CeedDestroy()

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedInit(const char *resource, Ceed *ceed) {
  int ierr;
  size_t matchlen = 0, matchidx = UINT_MAX, matchpriority = UINT_MAX, priority;

  // Find matching backend
  if (!resource)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "No resource provided");
  // LCOV_EXCL_STOP
  ierr = CeedRegisterAll(); CeedChk(ierr);

  for (size_t i=0; i<num_backends; i++) {
    size_t n;
    const char *prefix = backends[i].prefix;
    for (n = 0; prefix[n] && prefix[n] == resource[n]; n++) {}
    priority = backends[i].priority;
    if (n > matchlen || (n == matchlen && matchpriority > priority)) {
      matchlen = n;
      matchpriority = priority;
      matchidx = i;
    }
  }
  if (matchlen <= 1)
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "No suitable backend: %s",
                     resource);
  // LCOV_EXCL_STOP

  // Setup Ceed
  ierr = CeedCalloc(1, ceed); CeedChk(ierr);
  const char *ceed_error_handler = getenv("CEED_ERROR_HANDLER");
  if (!ceed_error_handler)
    ceed_error_handler = "abort";
  if (!strcmp(ceed_error_handler, "exit"))
    (*ceed)->Error = CeedErrorExit;
  else if (!strcmp(ceed_error_handler, "store"))
    (*ceed)->Error = CeedErrorStore;
  else
    (*ceed)->Error = CeedErrorAbort;
  memcpy((*ceed)->errmsg, "No error message stored", 24);
  (*ceed)->refcount = 1;
  (*ceed)->data = NULL;

  // Set lookup table
  foffset foffsets[] = {
    CEED_FTABLE_ENTRY(Ceed, Error),
    CEED_FTABLE_ENTRY(Ceed, GetPreferredMemType),
    CEED_FTABLE_ENTRY(Ceed, Destroy),
    CEED_FTABLE_ENTRY(Ceed, VectorCreate),
    CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreate),
    CEED_FTABLE_ENTRY(Ceed, ElemRestrictionCreateBlocked),
    CEED_FTABLE_ENTRY(Ceed, BasisCreateTensorH1),
    CEED_FTABLE_ENTRY(Ceed, BasisCreateH1),
    CEED_FTABLE_ENTRY(Ceed, TensorContractCreate),
    CEED_FTABLE_ENTRY(Ceed, QFunctionCreate),
    CEED_FTABLE_ENTRY(Ceed, QFunctionContextCreate),
    CEED_FTABLE_ENTRY(Ceed, OperatorCreate),
    CEED_FTABLE_ENTRY(Ceed, CompositeOperatorCreate),
    CEED_FTABLE_ENTRY(CeedVector, SetArray),
    CEED_FTABLE_ENTRY(CeedVector, TakeArray),
    CEED_FTABLE_ENTRY(CeedVector, SetValue),
    CEED_FTABLE_ENTRY(CeedVector, GetArray),
    CEED_FTABLE_ENTRY(CeedVector, GetArrayRead),
    CEED_FTABLE_ENTRY(CeedVector, RestoreArray),
    CEED_FTABLE_ENTRY(CeedVector, RestoreArrayRead),
    CEED_FTABLE_ENTRY(CeedVector, Norm),
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
    CEED_FTABLE_ENTRY(CeedQFunctionContext, SetData),
    CEED_FTABLE_ENTRY(CeedQFunctionContext, GetData),
    CEED_FTABLE_ENTRY(CeedQFunctionContext, RestoreData),
    CEED_FTABLE_ENTRY(CeedQFunctionContext, Destroy),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleQFunction),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleDiagonal),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleAddDiagonal),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssemblePointBlockDiagonal),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleAddPointBlockDiagonal),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssembleSymbolic),
    CEED_FTABLE_ENTRY(CeedOperator, LinearAssemble),
    CEED_FTABLE_ENTRY(CeedOperator, CreateFDMElementInverse),
    CEED_FTABLE_ENTRY(CeedOperator, Apply),
    CEED_FTABLE_ENTRY(CeedOperator, ApplyComposite),
    CEED_FTABLE_ENTRY(CeedOperator, ApplyAdd),
    CEED_FTABLE_ENTRY(CeedOperator, ApplyAddComposite),
    CEED_FTABLE_ENTRY(CeedOperator, ApplyJacobian),
    CEED_FTABLE_ENTRY(CeedOperator, Destroy),
    {NULL, 0} // End of lookup table - used in SetBackendFunction loop
  };

  ierr = CeedCalloc(sizeof(foffsets), &(*ceed)->foffsets); CeedChk(ierr);
  memcpy((*ceed)->foffsets, foffsets, sizeof(foffsets));

  // Set fallback for advanced CeedOperator functions
  const char fallbackresource[] = "";
  ierr = CeedSetOperatorFallbackResource(*ceed, fallbackresource);
  CeedChk(ierr);

  // Record env variables CEED_DEBUG or DBG
  (*ceed)->debug = !!getenv("CEED_DEBUG") || !!getenv("DBG");

  // Backend specific setup
  ierr = backends[matchidx].init(resource, *ceed); CeedChk(ierr);

  // Copy resource prefix, if backend setup sucessful
  size_t len = strlen(backends[matchidx].prefix);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, backends[matchidx].prefix, len+1);
  (*ceed)->resource = tmp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the full resource name for a Ceed context

  @param ceed            Ceed context to get resource name of
  @param[out] resource   Variable to store resource name

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/

int CeedGetResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->resource;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return Ceed context preferred memory type

  @param ceed      Ceed context to get preferred memory type of
  @param[out] type Address to save preferred memory type to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedGetPreferredMemType(Ceed ceed, CeedMemType *type) {
  int ierr;

  if (ceed->GetPreferredMemType) {
    ierr = ceed->GetPreferredMemType(type); CeedChk(ierr);
  } else {
    Ceed delegate;
    ierr = CeedGetDelegate(ceed, &delegate); CeedChk(ierr);

    if (delegate) {
      ierr = CeedGetPreferredMemType(delegate, type); CeedChk(ierr);
    } else {
      *type = CEED_MEM_HOST;
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get deterministic status of Ceed

  @param[in] ceed              Ceed
  @param[out] isDeterministic  Variable to store deterministic status

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedIsDeterministic(Ceed ceed, bool *isDeterministic) {
  *isDeterministic = ceed->isDeterministic;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a Ceed

  @param[in] ceed          Ceed to view
  @param[in] stream        Filestream to write to

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedView(Ceed ceed, FILE *stream) {
  int ierr;
  CeedMemType memtype;

  ierr = CeedGetPreferredMemType(ceed, &memtype); CeedChk(ierr);

  fprintf(stream, "Ceed\n"
          "  Ceed Resource: %s\n"
          "  Preferred MemType: %s\n",
          ceed->resource, CeedMemTypes[memtype]);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a Ceed context

  @param ceed Address of Ceed context to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedDestroy(Ceed *ceed) {
  int ierr;
  if (!*ceed || --(*ceed)->refcount > 0) return CEED_ERROR_SUCCESS;
  if ((*ceed)->delegate) {
    ierr = CeedDestroy(&(*ceed)->delegate); CeedChk(ierr);
  }

  if ((*ceed)->objdelegatecount > 0) {
    for (int i=0; i<(*ceed)->objdelegatecount; i++) {
      ierr = CeedDestroy(&((*ceed)->objdelegates[i].delegate)); CeedChk(ierr);
      ierr = CeedFree(&(*ceed)->objdelegates[i].objname); CeedChk(ierr);
    }
    ierr = CeedFree(&(*ceed)->objdelegates); CeedChk(ierr);
  }

  if ((*ceed)->Destroy) {
    ierr = (*ceed)->Destroy(*ceed); CeedChk(ierr);
  }

  ierr = CeedFree(&(*ceed)->foffsets); CeedChk(ierr);
  ierr = CeedFree(&(*ceed)->resource); CeedChk(ierr);
  ierr = CeedDestroy(&(*ceed)->opfallbackceed); CeedChk(ierr);
  ierr = CeedFree(&(*ceed)->opfallbackresource); CeedChk(ierr);
  ierr = CeedFree(ceed); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

// LCOV_EXCL_START
const char *CeedErrorFormat(Ceed ceed, const char *format, va_list *args) {
  if (ceed->parent)
    return CeedErrorFormat(ceed->parent, format, args);
  if (ceed->opfallbackparent)
    return CeedErrorFormat(ceed->opfallbackparent, format, args);
  vsnprintf(ceed->errmsg, CEED_MAX_RESOURCE_LEN, format, *args);
  return ceed->errmsg;
}
// LCOV_EXCL_STOP

/**
  @brief Error handling implementation; use \ref CeedError instead.

  @ref Developer
**/
int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func,
                  int ecode, const char *format, ...) {
  va_list args;
  int retval;
  va_start(args, format);
  if (ceed) {
    retval = ceed->Error(ceed, filename, lineno, func, ecode, format, &args);
  } else {
    // LCOV_EXCL_START
    const char *ceed_error_handler = getenv("CEED_ERROR_HANDLER");
    if (!ceed_error_handler)
      ceed_error_handler = "abort";
    if (!strcmp(ceed_error_handler, "return"))
      retval = CeedErrorReturn(ceed, filename, lineno, func, ecode, format, &args);
    else
      // This function will not return
      retval = CeedErrorAbort(ceed, filename, lineno, func, ecode, format, &args);
  }
  va_end(args);
  return retval;
  // LCOV_EXCL_STOP
}

/**
  @brief Error handler that returns without printing anything.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorReturn(Ceed ceed, const char *filename, int lineno,
                    const char *func, int ecode, const char *format,
                    va_list *args) {
  return ecode;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that stores the error message for future use and returns
           the error.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorStore(Ceed ceed, const char *filename, int lineno,
                   const char *func, int ecode, const char *format,
                   va_list *args) {
  if (ceed->parent)
    return CeedErrorStore(ceed->parent, filename, lineno, func, ecode, format,
                          args);
  if (ceed->opfallbackparent)
    return CeedErrorStore(ceed->opfallbackparent, filename, lineno, func, ecode,
                          format, args);

  // Build message
  CeedInt len;
  len = snprintf(ceed->errmsg, CEED_MAX_RESOURCE_LEN, "%s:%d in %s(): ",
                 filename, lineno, func);
  vsnprintf(ceed->errmsg + len, CEED_MAX_RESOURCE_LEN - len, format, *args);
  return ecode;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that prints to stderr and aborts

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorAbort(Ceed ceed, const char *filename, int lineno,
                   const char *func, int ecode, const char *format,
                   va_list *args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, *args);
  fprintf(stderr, "\n");
  abort();
  return ecode;
}
// LCOV_EXCL_STOP

/**
  @brief Error handler that prints to stderr and exits

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  In contrast to CeedErrorAbort(), this exits without a signal, so atexit()
  handlers (e.g., as used by gcov) are run.

  @ref Developer
**/
int CeedErrorExit(Ceed ceed, const char *filename, int lineno, const char *func,
                  int ecode, const char *format, va_list *args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, *args);
  fprintf(stderr, "\n");
  exit(ecode);
  return ecode;
}

/**
  @brief Set error handler

  A default error handler is set in CeedInit().  Use this function to change
  the error handler to CeedErrorReturn(), CeedErrorAbort(), or a user-defined
  error handler.

  @ref Developer
**/
int CeedSetErrorHandler(Ceed ceed, CeedErrorHandler eh) {
  ceed->Error = eh;
  if (ceed->delegate) CeedSetErrorHandler(ceed->delegate, eh);
  for (int i=0; i<ceed->objdelegatecount; i++)
    CeedSetErrorHandler(ceed->objdelegates[i].delegate, eh);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get error message

  The error message is only stored when using the error handler
    CeedErrorStore()

  @param[in] ceed     Ceed contex to retrieve error message
  @param[out] errmsg  Char pointer to hold error message

  @ref Developer
**/
int CeedGetErrorMessage(Ceed ceed, const char **errmsg) {
  if (ceed->parent)
    return CeedGetErrorMessage(ceed->parent, errmsg);
  if (ceed->opfallbackparent)
    return CeedGetErrorMessage(ceed->opfallbackparent, errmsg);
  *errmsg = ceed->errmsg;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore error message

  The error message is only stored when using the error handler
    CeedErrorStore()

  @param[in] ceed     Ceed contex to restore error message
  @param[out] errmsg  Char pointer that holds error message

  @ref Developer
**/
int CeedResetErrorMessage(Ceed ceed, const char **errmsg) {
  if (ceed->parent)
    return CeedResetErrorMessage(ceed->parent, errmsg);
  if (ceed->opfallbackparent)
    return CeedResetErrorMessage(ceed->opfallbackparent, errmsg);
  *errmsg = NULL;
  memcpy(ceed->errmsg, "No error message stored", 24);
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

/// @}
