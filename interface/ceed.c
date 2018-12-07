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
#include <stdarg.h>
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

#define ceedoffsetof(st, m) \
    ((size_t) ( (char *)&((st)(0))->m - (char *)0 ))
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

  @fixme The current implementation is overly strict, offering equivalent
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
  va_start(args, format);
  if (ceed) return ceed->Error(ceed, filename, lineno, func, ecode, format, args);
  return CeedErrorAbort(ceed, filename, lineno, func, ecode, format, args);
}

/**
  @brief Error handler that returns without printing anything.

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
int CeedErrorReturn(Ceed ceed, const char *filename, int lineno,
                    const char *func, int ecode, const char *format,
                    va_list args) {
  return ecode;
}

/**
  @brief Error handler that prints to stderr and aborts

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  @ref Developer
**/
int CeedErrorAbort(Ceed ceed, const char *filename, int lineno,
                   const char *func, int ecode,
                   const char *format, va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  abort();
  return ecode;
}

/**
  @brief Error handler that prints to stderr and exits

  Pass this to CeedSetErrorHandler() to obtain this error handling behavior.

  In contrast to CeedErrorAbort(), this exits without a signal, so atexit()
  handlers (e.g., as used by gcov) are run.

  @ref Developer
**/
int CeedErrorExit(Ceed ceed, const char *filename, int lineno,
                  const char *func, int ecode,
                  const char *format, va_list args) {
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
int CeedRegister(const char *prefix,
                 int (*init)(const char *, Ceed), unsigned int priority) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0])) {
    return CeedError(NULL, 1, "Too many backends");
  }
  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
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
    return CeedError(NULL, ierr,
                     "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

/**
  @brief Allocate a cleared (zeroed) array on the host; use CeedCalloc()

  Memory usage can be tracked by the library.

  @param n Number of units to allocate
  @param unit Size of each unit
  @param p Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Advanced
**/
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p)
    return CeedError(NULL, 1, "calloc failed to allocate %zd members of size %zd\n",
                     n, unit);
  return 0;
}

/**
  @brief Reallocate an array on the host; use CeedRealloc()

  Memory usage can be tracked by the library.

  @param n Number of units to allocate
  @param unit Size of each unit
  @param p Address of pointer to hold the result.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedFree()

  @ref Advanced
**/
int CeedReallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = realloc(*(void **)p, n*unit);
  if (n && unit && !*(void **)p)
    return CeedError(NULL, 1,
                     "realloc failed to allocate %zd members of size %zd\n",
                     n, unit);
  return 0;
}

/// Free memory allocated using CeedMalloc() or CeedCalloc()
///
/// @param p address of pointer to memory.  This argument is of type void* to
/// avoid needing a cast, but is the address of the pointer (which is zeroed)
/// rather than the pointer.
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
  if (!*req) return 0;
  return CeedError(NULL, 2, "CeedRequestWait not implemented");
}

/**
  @brief Initialize a \ref Ceed to use the specified resource.

  @param resource  Resource to use, e.g., "/cpu/self"
  @param ceed The library context
  @sa CeedRegister() CeedDestroy()

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedInit(const char *resource, Ceed *ceed) {
  int ierr;
  size_t matchlen = 0, matchidx;
  unsigned int matchpriority = 100, priority;

  // Find matching backend
  if (!resource) return CeedError(NULL, 1, "No resource provided");
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
  if (!matchlen) return CeedError(NULL, 1, "No suitable backend");

  // Setup Ceed
  ierr = CeedCalloc(1,ceed); CeedChk(ierr);
  const char * ceed_error_handler = getenv("CEED_ERROR_HANDLER");
  if (!ceed_error_handler) ceed_error_handler = "abort";
  if (!strcmp(ceed_error_handler, "exit"))
    (*ceed)->Error = CeedErrorExit;
  else
    (*ceed)->Error = CeedErrorAbort;
  (*ceed)->refcount = 1;
  (*ceed)->data = NULL;

  // Set lookup table
  foffset foffsets[CEED_NUM_BACKEND_FUNCTIONS] = {
    {"CeedError",                 ceedoffsetof(Ceed, Error)},
    {"CeedDestroy",               ceedoffsetof(Ceed, Destroy)},
    {"CeedVecCreate",             ceedoffsetof(Ceed, VecCreate)},
    {"CeedElemRestrictionCreate", ceedoffsetof(Ceed, ElemRestrictionCreate)},
    {"CeedElemRestrictionCreateBlocked",
      ceedoffsetof(Ceed, ElemRestrictionCreateBlocked)},
    {"CeedBasisCreateTensorH1",    ceedoffsetof(Ceed, BasisCreateTensorH1)},
    {"CeedBasisCreateH1",          ceedoffsetof(Ceed, BasisCreateH1)},
    {"CeedQFunctionCreate",        ceedoffsetof(Ceed, QFunctionCreate)},
    {"CeedOperatorCreate",         ceedoffsetof(Ceed, OperatorCreate)},
    {"VectorSetArray",             ceedoffsetof(CeedVector, SetArray)},
    {"VectorSetValue",             ceedoffsetof(CeedVector, SetValue)},
    {"VectorGetArray",             ceedoffsetof(CeedVector, GetArray)},
    {"VectorGetArrayRead",         ceedoffsetof(CeedVector, GetArrayRead)},
    {"VectorRestoreArray",         ceedoffsetof(CeedVector, RestoreArray)},
    {"VectorRestoreArrayRead",     ceedoffsetof(CeedVector, RestoreArrayRead)},
    {"VectorDestroy",              ceedoffsetof(CeedVector, Destroy)},
    {"ElemRestrictionApply",       ceedoffsetof(CeedElemRestriction, Apply)},
    {"ElemRestrictionDestroy",     ceedoffsetof(CeedElemRestriction, Destroy)},
    {"BasisApply",                 ceedoffsetof(CeedBasis, Apply)},
    {"BasisDestroy",               ceedoffsetof(CeedBasis, Destroy)},
    {"QFunctionApply",             ceedoffsetof(CeedQFunction, Apply)},
    {"QFunctionDestroy",           ceedoffsetof(CeedQFunction, Destroy)},
    {"OperatorApply",              ceedoffsetof(CeedOperator, Apply)},
    {"OperatorApplyJacobian",      ceedoffsetof(CeedOperator, ApplyJacobian)},
    {"OperatorDestroy",            ceedoffsetof(CeedOperator, Destroy)} };

  memcpy((*ceed)->foffsets, foffsets,
         CEED_NUM_BACKEND_FUNCTIONS*sizeof(foffset));

  // Backend specific setup
  ierr = backends[matchidx].init(resource, *ceed); CeedChk(ierr);

  return 0;
}

/**
  @brief Retrieve a delegate CEED

  @param ceed           Ceed to retrieve delegate of
  @param[out] delegate  Address to save the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedGetDelegate(Ceed ceed, Ceed *delegate) {
  *delegate = ceed->delegate;
  return 0;
}

/**
  @brief Set a delegate CEED

  @param ceed           Ceed to set delegate of
  @param[out] delegate  Address to set the delegate to

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedSetDelegate(Ceed ceed, Ceed *delegate) {
  ceed->delegate = *delegate;
  return 0;
}

/**
  @brief Set a backend function

  @param ceed           Ceed for error handling
  @param type           Type of Ceed object to set function for
  @param[out] object    Ceed object to set function for
  @param fname          Name of function to set
  @param f              Function to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedSetBackendFunction(Ceed ceed,
                           const char *type, void *object,
                           const char *fname, int (*f)()) {
  char lookupname[100];
  strcpy(lookupname, "");

  // Build lookup name
  strcat(strcat(lookupname, type), fname);

  // Find and use offset
  for (CeedInt i = 0; i < CEED_NUM_BACKEND_FUNCTIONS; i++) {
    if (!strcmp(ceed->foffsets[i].fname, lookupname)) {
      size_t offset = ceed->foffsets[i].offset;
      size_t *fpointer;
      fpointer = (size_t *)(object + offset);
      *fpointer = (size_t) f;
      return 0;
    }
  }

  return CeedError(ceed, 1, "Requested function '%s' was not found for CEED object '%s'", fname, type);
}

/**
  @brief Retrieve backend data for a CEED

  @param ceed           Ceed to retrieve data of
  @param[out] data      Address to save data to

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedGetData(Ceed ceed, void* *data) {
  *data = ceed->data;
  return 0;
}

/**
  @brief Set backend data for a CEED

  @param ceed           Ceed to set data of
  @param data           Address of data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedSetData(Ceed ceed, void* *data) {
  ceed->data = *data;
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

  if (!*ceed || --(*ceed)->refcount > 0) return 0;
  if ((*ceed)->delegate) {
    ierr = CeedDestroy(&(*ceed)->delegate); CeedChk(ierr);
  }
  if ((*ceed)->Destroy) {
    ierr = (*ceed)->Destroy(*ceed); CeedChk(ierr);
  }
  ierr = CeedFree(ceed); CeedChk(ierr);
  return 0;
}

/// @}
