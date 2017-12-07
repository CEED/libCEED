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

#ifndef _ceed_impl_h
#define _ceed_impl_h

#include <ceed.h>

#define CEED_INTERN CEED_EXTERN __attribute__((visibility ("hidden")))

#define CEED_MAX_RESOURCE_LEN 1024
#define CEED_ALIGN 64

struct Ceed_private {
  int (*Error)(Ceed, const char *, int, const char *, int, const char *, va_list);
  int (*Destroy)(Ceed);
  int (*VecCreate)(Ceed, CeedInt, CeedVector);
  int (*ElemRestrictionCreate)(CeedElemRestriction, CeedMemType, CeedCopyMode,
                               const CeedInt *);
  int (*BasisCreateTensorH1)(Ceed, CeedInt, CeedInt, CeedInt, const CeedScalar *,
                             const CeedScalar *, const CeedScalar *, const CeedScalar *, CeedBasis);
  int (*QFunctionCreate)(CeedQFunction);
  int (*OperatorCreate)(CeedOperator);
};

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p
   has to be a pointer to a pointer. */
CEED_INTERN int CeedMallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedCallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedFree(void *p);

#define CeedChk(ierr) do { if (ierr) return ierr; } while (0)
#define CeedMalloc(n, p) CeedMallocArray((n), sizeof(**(p)), p)
#define CeedCalloc(n, p) CeedCallocArray((n), sizeof(**(p)), p)

struct CeedVector_private {
  Ceed ceed;
  int (*SetArray)(CeedVector, CeedMemType, CeedCopyMode, CeedScalar *);
  int (*GetArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArrayRead)(CeedVector, CeedMemType, const CeedScalar **);
  int (*RestoreArray)(CeedVector, CeedScalar **);
  int (*RestoreArrayRead)(CeedVector, const CeedScalar **);
  int (*Destroy)(CeedVector);
  CeedInt length;
  void *data;
};

struct CeedElemRestriction_private {
  Ceed ceed;
  int (*Apply)(CeedElemRestriction, CeedTransposeMode, CeedVector, CeedVector,
               CeedRequest *);
  int (*Destroy)(CeedElemRestriction);
  CeedInt nelem;    /* number of elements */
  CeedInt elemsize; /* number of dofs per element */
  CeedInt ndof;     /* size of the L-vector, can be used for checking for
                       correct vector sizes */
  void *data;       /* place for the backend to store any data */
};

struct CeedBasis_private {
  Ceed ceed;
  int (*Apply)(CeedBasis, CeedTransposeMode, CeedEvalMode, const CeedScalar *,
               CeedScalar *);
  int (*Destroy)(CeedBasis);
  CeedInt dim;
  CeedInt ndof;
  CeedInt P1d;
  CeedInt Q1d;
  CeedScalar *qref1d;
  CeedScalar *qweight1d;
  CeedScalar *interp1d;
  CeedScalar *grad1d;
};

/* FIXME: The number of in-fields and out-fields may be different? */
/* FIXME: Shouldn't inmode and outmode be per-in-field and per-out-field,
   respectively? */
struct CeedQFunction_private {
  Ceed ceed;
  int (*Destroy)(CeedQFunction);
  CeedInt vlength;    // Number of quadrature points must be padded to a multiple of vlength
  CeedInt nfields;
  size_t qdatasize;   // Number of bytes of qdata per quadrature point
  CeedEvalMode inmode, outmode;
  int (*function)(void*, void*, CeedInt, const CeedScalar *const*,
                  CeedScalar *const*);
  const char *focca;
  void *ctx;      /* user context for function */
  size_t ctxsize; /* size of user context; may be used to copy to a device */
  void *data;     /* backend data */
};

struct CeedOperator_private {
  Ceed ceed;
  int (*Apply)(CeedOperator, CeedVector, CeedVector, CeedVector, CeedRequest*);
  int (*ApplyJacobian)(CeedOperator, CeedVector, CeedVector, CeedVector,
                       CeedVector, CeedRequest*);
  int (*Destroy)(CeedOperator);
  CeedElemRestriction Erestrict;
  CeedBasis basis;
  CeedQFunction qf;
  CeedQFunction dqf;
  CeedQFunction dqfT;
  void *data;
};

#endif
