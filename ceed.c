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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @cond DOXYGEN_SKIP
static CeedRequest ceed_request_immediate;

static struct {
  char prefix[CEED_MAX_RESOURCE_LEN];
  int (*init)(const char *resource, Ceed f);
} backends[32];
static size_t num_backends;
/// @endcond

/// @file
/// Implementation of core components of Ceed library
///
/// @defgroup Ceed Ceed: core components
/// @{

/// Request immediate completion
///
/// This predefined constant is passed as the \ref CeedRequest argument to
/// interfaces when the caller wishes for the operation to be performed
/// immediately.  The code
///
/// @code
///   CeedOperatorApply(op, ..., CEED_REQUEST_IMMEDIATE);
/// @endcode
///
/// is semantically equivalent to
///
/// @code
///   CeedRequest request;
///   CeedOperatorApply(op, ..., &request);
///   CeedRequestWait(&request);
/// @endcode
CeedRequest *const CEED_REQUEST_IMMEDIATE = &ceed_request_immediate;

/// Error handling implementation; use \ref CeedError instead.
int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func,
                  int ecode, const char *format, ...) {
  va_list args;
  va_start(args, format);
  if (ceed) return ceed->Error(ceed, filename, lineno, func, ecode, format, args);
  return CeedErrorAbort(ceed, filename, lineno, func, ecode, format, args);
}

/// Error handler that returns without printing anything.
///
/// Pass this to CeedSetErrorHandler() to obtain this error handling behavior.
///
/// @sa CeedErrorAbort
int CeedErrorReturn(Ceed ceed, const char *filename, int lineno,
                    const char *func, int ecode, const char *format,
                    va_list args) {
  return ecode;
}

/// Error handler that prints to stderr and aborts
///
/// Pass this to CeedSetErrorHandler() to obtain this error handling behavior.
///
/// @sa CeedErrorReturn
int CeedErrorAbort(Ceed ceed, const char *filename, int lineno,
                   const char *func, int ecode,
                   const char *format, va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  abort();
  return ecode;
}

/// Set error handler
///
/// A default error handler is set in CeedInit().  Use this function to change
/// the error handler to CeedErrorReturn(), CeedErrorAbort(), or a user-defined
/// error handler.
int CeedSetErrorHandler(Ceed ceed,
                        int (eh)(Ceed, const char *, int, const char *,
                                 int, const char *, va_list)) {
  ceed->Error = eh;
  return 0;
}

/// Register a Ceed backend
///
/// @param prefix Prefix of resources for this backend to respond to.  For
///               example, the reference backend responds to "/cpu/self".
/// @param init   Initialization function called by CeedInit() when the backend
///               is selected to drive the requested resource.
int CeedRegister(const char *prefix, int (*init)(const char *resource,
                 Ceed f)) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0])) {
    return CeedError(NULL, 1, "Too many backends");
  }
  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].init = init;
  num_backends++;
  return 0;
}

/// Allocate an array on the host; use CeedMalloc()
///
/// Memory usage can be tracked by the library.  This ensures sufficient
/// alignment for vectorization and should be used for large allocations.
///
/// @param n Number of units to allocate
/// @param unit Size of each unit
/// @param p Address of pointer to hold the result.
/// @sa CeedFree()
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n*unit);
  if (ierr)
    return CeedError(NULL, ierr,
                     "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
  return 0;
}

/// Allocate a cleared (zeroed) array on the host; use CeedCalloc()
///
/// Memory usage can be tracked by the library.
///
/// @param n Number of units to allocate
/// @param unit Size of each unit
/// @param p Address of pointer to hold the result.
/// @sa CeedFree()
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p)
    return CeedError(NULL, 1, "calloc failed to allocate %zd members of size %zd\n",
                     n, unit);
  return 0;
}

/// Free memory allocated using CeedMalloc() or CeedCalloc()
///
/// @param p address of pointer to memory.  This argument is of type void* to avoid needing a cast, but is the address of the pointer (which is zeroed) rather than the pointer.
int CeedFree(void *p) {
  free(*(void **)p);
  *(void **)p = NULL;
  return 0;
}

/// Initialize a \ref Ceed to use the specified resource.
///
/// @param resource  Resource to use, e.g., "/cpu/self"
/// @param ceed The library context
/// @sa CeedRegister() CeedDestroy()
int CeedInit(const char *resource, Ceed *ceed) {
  int ierr;
  size_t matchlen = 0, matchidx;

  if (!resource) return CeedError(NULL, 1, "No resource provided");
  for (size_t i=0; i<num_backends; i++) {
    size_t n;
    const char *prefix = backends[i].prefix;
    for (n = 0; prefix[n] && prefix[n] == resource[n]; n++) {}
    if (n > matchlen) {
      matchlen = n;
      matchidx = i;
    }
  }
  if (!matchlen) return CeedError(NULL, 1, "No suitable backend");
  ierr = CeedCalloc(1,ceed); CeedChk(ierr);
  (*ceed)->Error = CeedErrorAbort;
  (*ceed)->data = NULL;
  ierr = backends[matchidx].init(resource, *ceed); CeedChk(ierr);
  return 0;
}

/// Destroy a Ceed context
int CeedDestroy(Ceed *ceed) {
  int ierr;

  if (!*ceed) return 0;
  if ((*ceed)->Destroy) {
    ierr = (*ceed)->Destroy(*ceed); CeedChk(ierr);
  }
  ierr = CeedFree(ceed); CeedChk(ierr);
  return 0;
}

/// @}

void CeedDebug(const char *format,...) {
  // real slow, should use NDEBUG to ifdef the body
  if (!getenv("CEED_DEBUG")) return;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout,"\033[32m");
  vfprintf(stdout,format,args);
  fprintf(stdout,"\033[m");
  fprintf(stdout,"\n");
  fflush(stdout);
  va_end(args);
}

// Fortran interface
static Ceed **Ceed_dict = NULL;
static int Ceed_count = 0;
static int Ceed_count_max = 0;

void fCeedInit(const char* resource, CeedInt *ceed, CeedInt *err) {
  if (Ceed_count == Ceed_count_max)
    Ceed_count_max += Ceed_count_max/2 + 1,
    Ceed_dict = realloc(Ceed_dict, sizeof(Ceed*)*Ceed_count_max);

  Ceed *ceed_ = Ceed_dict[Ceed_count] = (Ceed*) malloc(sizeof(Ceed));
  *err = CeedInit(resource, ceed_);

  *ceed = Ceed_count++;
}

static CeedVector **CeedVector_dict = NULL;
static int CeedVector_count = 0;
static int CeedVector_count_max = 0;

void fCeedVectorCreate(CeedInt *ceed, CeedInt *length, CeedInt *vec, CeedInt *err) {
  if (CeedVector_count == CeedVector_count_max)
    CeedVector_count_max += CeedVector_count_max/2 + 1,
    CeedVector_dict =
        realloc(CeedVector_dict, sizeof(CeedVector*)*CeedVector_count_max);

  CeedVector *vec_ = CeedVector_dict[CeedVector_count] =
                         (CeedVector*) malloc(sizeof(CeedVector));

  *err = CeedVectorCreate(*Ceed_dict[*ceed], *length, vec_);

  *vec = CeedVector_count++;
}

static CeedElemRestriction **CeedElemRestriction_dict = NULL;
static int CeedElemRestriction_count = 0;
static int CeedElemRestriction_count_max = 0;

void fCeedElemRestrictionCreate(CeedInt *ceed, CeedInt *nelements,
    CeedInt *esize, CeedInt *ndof, CeedInt *memtype, CeedInt *copymode,
    const CeedInt *indices, CeedInt *elemrestriction, CeedInt *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max)
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1,
    CeedElemRestriction_dict =
        realloc(CeedElemRestriction_dict, \
        sizeof(CeedElemRestriction*)*CeedElemRestriction_count_max);

  CeedElemRestriction *elemrestriction_ =
      CeedElemRestriction_dict[CeedElemRestriction_count] =
          (CeedElemRestriction*) malloc(sizeof(CeedElemRestriction));

  *err = CeedElemRestrictionCreate(*Ceed_dict[*ceed], *nelements, *esize, *ndof,
             *memtype, *copymode, indices, elemrestriction_);

  *elemrestriction = CeedElemRestriction_count++;
}

static CeedBasis **CeedBasis_dict = NULL;
static int CeedBasis_count = 0;
static int CeedBasis_count_max = 0;

void fCeedBasisCreateTensorH1Lagrange(CeedInt *ceed, CeedInt *dim,
    CeedInt *ndof, CeedInt *P, CeedInt *Q, CeedInt *quadmode, CeedInt *basis,
    CeedInt *err) {
  if (CeedBasis_count == CeedBasis_count_max)
    CeedBasis_count_max += CeedBasis_count_max/2 + 1,
    CeedBasis_dict = realloc(CeedBasis_dict, sizeof(CeedBasis*)*CeedBasis_count_max);

  CeedBasis *basis_ = CeedBasis_dict[CeedBasis_count] =
                          (CeedBasis*) malloc(sizeof(CeedBasis));

  *err = CeedBasisCreateTensorH1Lagrange(*Ceed_dict[*ceed], *dim, *ndof, *P, *Q,
             *quadmode, basis_);

  *basis = CeedBasis_count++;
}

static CeedQFunction **CeedQFunction_dict = NULL;
static int CeedQFunction_count = 0;
static int CeedQFunction_count_max = 0;

void fCeedQFunctionCreateInterior(CeedInt* ceed, CeedInt* vlength,
    CeedInt* nfields, size_t qdatasize, CeedInt* inmode, CeedInt* outmode,
    int (*f)(void *ctx, void *qdata, CeedInt nq, const CeedScalar *const *u,
             CeedScalar *const *v), const char *focca, CeedInt *qf) {
}
