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
#include <ceed-impl.h>
#include <ceed-backend.h>
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
///
/// @addtogroup Ceed
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
  semantics to CEED_REQUEST_IMMEDIATE.

  @sa CEED_REQUEST_IMMEDIATE
 */
CeedRequest *const CEED_REQUEST_ORDERED = &ceed_request_ordered;

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
    retval = ceed->Error(ceed, filename, lineno, func, ecode, format, args);
  } else {
    // This function doesn't actually return
    retval = CeedErrorAbort(ceed, filename, lineno, func, ecode, format, args);
  }
  va_end(args);
  return retval;
}

/**
  @brief Error handler that returns without printing anything.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
// LCOV_EXCL_START
int CeedErrorReturn(Ceed ceed, const char *filename, int lineno,
                    const char *func, int ecode, const char *format,
                    va_list args) {
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
                   va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args);
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
                  int ecode, const char *format, va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args);
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
int CeedSetErrorHandler(Ceed ceed,
                        int (eh)(Ceed, const char *, int, const char *,
                                 int, const char *, va_list)) {
  ceed->Error = eh;
  return 0;
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

  @ref Advanced
**/
int CeedRegister(const char *prefix, int (*init)(const char *, Ceed),
                 unsigned int priority) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0]))
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "Too many backends");
  // LCOV_EXCL_STOP

  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].prefix[CEED_MAX_RESOURCE_LEN-1] = 0;
  backends[num_backends].init = init;
  backends[num_backends].priority = priority;
  num_backends++;
  return 0;
}

/**
  @brief Allocate an array on the host; use CeedMalloc()

  Memory usage can be tracked by the library.  This ensures sufficient
    alignment for vectorization and should be used for large allocations.

  @param n Number of units to allocate
  @param unit Size of each unit
  @param p Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Advanced
**/
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n*unit);
  if (ierr)
    // LCOV_EXCL_START
    return CeedError(NULL, ierr, "posix_memalign failed to allocate %zd "
                     "members of size %zd\n", n, unit);
  // LCOV_EXCL_STOP

  return 0;
}

/**
  @brief Allocate a cleared (zeroed) array on the host; use CeedCalloc()

  Memory usage can be tracked by the library.

  @param n    Number of units to allocate
  @param unit Size of each unit
  @param p    Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Advanced
**/
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "calloc failed to allocate %zd members of size "
                     "%zd\n", n, unit);
  // LCOV_EXCL_STOP

  return 0;
}

/**
  @brief Reallocate an array on the host; use CeedRealloc()

  Memory usage can be tracked by the library.

  @param n    Number of units to allocate
  @param unit Size of each unit
  @param p    Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Advanced
**/
int CeedReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n*unit);
  if (n && unit && !*(void **)p)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "realloc failed to allocate %zd members of size "
                     "%zd\n", n, unit);
  // LCOV_EXCL_STOP

  return 0;
}

/** Free memory allocated using CeedMalloc() or CeedCalloc()

  @param p address of pointer to memory.  This argument is of type void* to
             avoid needing a cast, but is the address of the pointer (which is
             zeroed) rather than the pointer.
**/
int CeedFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return 0;
}

/**
  @brief Wait for a CeedRequest to complete.

  Calling CeedRequestWait on a NULL request is a no-op.

  @param req Address of CeedRequest to wait for; zeroed on completion.

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedRequestWait(CeedRequest *req) {
  if (!*req)
    return 0;
  return CeedError(NULL, 2, "CeedRequestWait not implemented");
}

/**
  @brief Initialize a \ref Ceed context to use the specified resource.

  @param resource  Resource to use, e.g., "/cpu/self"
  @param ceed      The library context
  @sa CeedRegister() CeedDestroy()

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedInit(const char *resource, Ceed *ceed) {
  int ierr;
  size_t matchlen = 0, matchidx = UINT_MAX, matchpriority = UINT_MAX, priority;

  // Find matching backend
  if (!resource)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "No resource provided");
  // LCOV_EXCL_STOP

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
  if (!matchlen)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "No suitable backend");
  // LCOV_EXCL_STOP

  // Setup Ceed
  ierr = CeedCalloc(1, ceed); CeedChk(ierr);
  const char *ceed_error_handler = getenv("CEED_ERROR_HANDLER");
  if (!ceed_error_handler)
    ceed_error_handler = "abort";
  if (!strcmp(ceed_error_handler, "exit"))
    (*ceed)->Error = CeedErrorExit;
  else
    (*ceed)->Error = CeedErrorAbort;
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
    CEED_FTABLE_ENTRY(Ceed, OperatorCreate),
    CEED_FTABLE_ENTRY(Ceed, CompositeOperatorCreate),
    CEED_FTABLE_ENTRY(CeedVector, SetArray),
    CEED_FTABLE_ENTRY(CeedVector, SetValue),
    CEED_FTABLE_ENTRY(CeedVector, GetArray),
    CEED_FTABLE_ENTRY(CeedVector, GetArrayRead),
    CEED_FTABLE_ENTRY(CeedVector, RestoreArray),
    CEED_FTABLE_ENTRY(CeedVector, RestoreArrayRead),
    CEED_FTABLE_ENTRY(CeedVector, Destroy),
    CEED_FTABLE_ENTRY(CeedElemRestriction, Apply),
    CEED_FTABLE_ENTRY(CeedElemRestriction, ApplyBlock),
    CEED_FTABLE_ENTRY(CeedElemRestriction, Destroy),
    CEED_FTABLE_ENTRY(CeedBasis, Apply),
    CEED_FTABLE_ENTRY(CeedBasis, Destroy),
    CEED_FTABLE_ENTRY(CeedTensorContract, Apply),
    CEED_FTABLE_ENTRY(CeedTensorContract, Destroy),
    CEED_FTABLE_ENTRY(CeedQFunction, Apply),
    CEED_FTABLE_ENTRY(CeedQFunction, Destroy),
    CEED_FTABLE_ENTRY(CeedOperator, AssembleLinearQFunction),
    CEED_FTABLE_ENTRY(CeedOperator, AssembleLinearDiagonal),
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
  const char fallbackresource[] = "/cpu/self/ref/serial";
  ierr = CeedSetOperatorFallbackResource(*ceed, fallbackresource);
  CeedChk(ierr);

  // Backend specific setup
  ierr = backends[matchidx].init(resource, *ceed); CeedChk(ierr);

  // Copy resource prefix, if backend setup sucessful
  size_t len = strlen(backends[matchidx].prefix);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, backends[matchidx].prefix, len+1);
  (*ceed)->resource = tmp;

  return 0;
}

/**
  @brief Retrieve a parent Ceed context

  @param ceed        Ceed context to retrieve parent of
  @param[out] parent Address to save the parent to

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedGetParent(Ceed ceed, Ceed *parent) {
  int ierr;
  if (ceed->parent) {
    ierr = CeedGetParent(ceed->parent, parent); CeedChk(ierr);
    return 0;
  }
  *parent = ceed;
  return 0;
}

/**
  @brief Retrieve a delegate Ceed context

  @param ceed          Ceed context to retrieve delegate of
  @param[out] delegate Address to save the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedGetDelegate(Ceed ceed, Ceed *delegate) {
  *delegate = ceed->delegate;
  return 0;
}

/**
  @brief Set a delegate Ceed context

  This function allows a Ceed context to set a delegate Ceed context. All
    backend implementations default to the delegate Ceed context, unless
    overridden.

  @param ceed           Ceed context to set delegate of
  @param[out] delegate  Address to set the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedSetDelegate(Ceed ceed, Ceed delegate) {
  ceed->delegate = delegate;
  delegate->parent = ceed;
  return 0;
}

/**
  @brief Retrieve a delegate Ceed context for a specific object type

  @param ceed           Ceed context to retrieve delegate of
  @param[out] delegate  Address to save the delegate to
  @param[in] objname    Name of the object type to retrieve delegate for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate, const char *objname) {
  CeedInt ierr;

  // Check for object delegate
  for (CeedInt i=0; i<ceed->objdelegatecount; i++)
    if (!strcmp(objname, ceed->objdelegates->objname)) {
      *delegate = ceed->objdelegates->delegate;
      return 0;
    }

  // Use default delegate if no object delegate
  ierr = CeedGetDelegate(ceed, delegate); CeedChk(ierr);

  return 0;
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

  @ref Advanced
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

  return 0;
}

/**
  @brief Set the fallback resource for CeedOperators. The current resource, if
           any, is freed by calling this function. This string is freed upon the
           destruction of the Ceed context.

  @param[out] ceed Ceed context
  @param resource  Fallback resource to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
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

  return 0;
}

/**
  @brief Get the fallback resource for CeedOperators

  @param ceed          Ceed context
  @param[out] resource Variable to store fallback resource

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedGetOperatorFallbackResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->opfallbackresource;
  return 0;
}

/**
  @brief Get the parent Ceed context associated with a fallback Ceed context
           for a CeedOperator

  @param ceed            Ceed context
  @param[out] ceed       Variable to store parent Ceed context

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedGetOperatorFallbackParentCeed(Ceed ceed, Ceed *parent) {
  *parent = ceed->opfallbackparent;
  return 0;
}

/**
  @brief Return Ceed context preferred memory type

  @param ceed      Ceed context to get preferred memory type of
  @param[out] type Address to save preferred memory type to

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
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

  return 0;
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

  @ref Advanced
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
      return 0;
    }

  // LCOV_EXCL_START
  return CeedError(ceed, 1, "Requested function '%s' was not found for CEED "
                   "object '%s'", fname, type);
  // LCOV_EXCL_STOP
}

/**
  @brief Retrieve backend data for a Ceed context

  @param ceed      Ceed context to retrieve data of
  @param[out] data Address to save data to

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedGetData(Ceed ceed, void **data) {
  *data = ceed->data;
  return 0;
}

/**
  @brief Set backend data for a Ceed context

  @param ceed           Ceed context to set data of
  @param data           Address of data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedSetData(Ceed ceed, void **data) {
  ceed->data = *data;
  return 0;
}

/**
  @brief Get the full resource name for a Ceed context

  @param ceed            Ceed context to get resource name of
  @param[out] resource   Variable to store resource name

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/

int CeedGetResource(Ceed ceed, const char **resource) {
  *resource = (const char *)ceed->resource;
  return 0;
}

/**
  @brief Destroy a Ceed context

  @param ceed Address of Ceed context to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedDestroy(Ceed *ceed) {
  int ierr;

  if (!*ceed || --(*ceed)->refcount > 0)
    return 0;
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
  return 0;
}

/// @}
