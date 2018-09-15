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
#include <stdbool.h>

#define CEED_INTERN CEED_EXTERN __attribute__((visibility ("hidden")))

#define CEED_MAX_RESOURCE_LEN 1024
#define CEED_ALIGN 64

struct Ceed_private {
  Ceed delegate;
  int (*Error)(Ceed, const char *, int, const char *, int, const char *, va_list);
  int (*Destroy)(Ceed);
  int (*VecCreate)(CeedInt, CeedVector);
  int (*ElemRestrictionCreate)(CeedMemType, CeedCopyMode,
                               const CeedInt *, CeedElemRestriction);
  int (*ElemRestrictionCreateBlocked)(CeedMemType, CeedCopyMode,
                                      const CeedInt *, CeedElemRestriction);
  int (*BasisCreateTensorH1)(CeedInt, CeedInt, CeedInt, const CeedScalar *,
                             const CeedScalar *, const CeedScalar *, const CeedScalar *, CeedBasis);
  int (*BasisCreateH1)(CeedElemTopology, CeedInt, CeedInt, CeedInt,
                       const CeedScalar *,
                       const CeedScalar *, const CeedScalar *, const CeedScalar *, CeedBasis);
  int (*QFunctionCreate)(CeedQFunction);
  int (*OperatorCreate)(CeedOperator);
  int refcount;
  void *data;
};

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p
   has to be a pointer to a pointer. */
CEED_INTERN int CeedMallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedCallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedReallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedFree(void *p);

#define CeedChk(ierr) do { if (ierr) return ierr; } while (0)
/* Note that CeedMalloc and CeedCalloc will, generally, return pointers with
   different memory alignments: CeedMalloc returns pointers aligned at
   CEED_ALIGN bytes, while CeedCalloc uses the alignment of calloc. */
#define CeedMalloc(n, p) CeedMallocArray((n), sizeof(**(p)), p)
#define CeedCalloc(n, p) CeedCallocArray((n), sizeof(**(p)), p)
#define CeedRealloc(n, p) CeedReallocArray((n), sizeof(**(p)), p)

struct CeedVector_private {
  Ceed ceed;
  int (*SetArray)(CeedVector, CeedMemType, CeedCopyMode, CeedScalar *);
  int (*SetValue)(CeedVector, CeedScalar);
  int (*GetArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArrayRead)(CeedVector, CeedMemType, const CeedScalar **);
  int (*RestoreArray)(CeedVector, CeedScalar **);
  int (*RestoreArrayRead)(CeedVector, const CeedScalar **);
  int (*Destroy)(CeedVector);
  int refcount;
  CeedInt length;
  uint64_t state;
  void *data;
};

struct CeedElemRestriction_private {
  Ceed ceed;
  int (*Apply)(CeedElemRestriction, CeedTransposeMode, CeedTransposeMode,
               CeedVector, CeedVector, CeedRequest *);
  int (*Destroy)(CeedElemRestriction);
  int refcount;
  CeedInt nelem;    /* number of elements */
  CeedInt elemsize; /* number of dofs per element */
  CeedInt ndof;     /* size of the L-vector, can be used for checking for
                      correct vector sizes */
  CeedInt ncomp;    /* number of components */
  CeedInt blksize;  /* number of elements in a batch */
  CeedInt nblk;     /* number of blocks of elements */
  void *data;       /* place for the backend to store any data */
};

struct CeedBasis_private {
  Ceed ceed;
  int (*Apply)(CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode,
               const CeedScalar *,
               CeedScalar *);
  int (*Destroy)(CeedBasis);
  int refcount;
  bool tensorbasis;      /* flag for tensor basis */
  CeedInt dim;           /* topological dimension */
  CeedInt ncomp;         /* number of field components (1 for scalar fields) */
  CeedInt P1d;           /* number of nodes in one dimension */
  CeedInt Q1d;           /* number of quadrature points in one dimension */
  CeedInt P;             /* total number of nodes */
  CeedInt Q;             /* total number of quadrature points */
  CeedScalar *qref1d;    /* Array of length Q1d holding the locations of
                            quadrature points on the 1D reference element [-1, 1] */
  CeedScalar *qweight1d; /* array of length Q1d holding the quadrature weights on
                            the reference element */
  CeedScalar
  *interp1d;  /* row-major matrix of shape [Q1d, P1d] expressing the values of
                            nodal basis functions at quadrature points */
  CeedScalar
  *grad1d;    /* row-major matrix of shape [Q1d, P1d] matrix expressing derivatives of
                            nodal basis functions at quadrature points */
  void *data;            /* place for the backend to store any data */
};

struct CeedQFunctionField {
  const char *fieldname;
  CeedInt ncomp;
  CeedEvalMode emode;
};

struct CeedQFunction_private {
  Ceed ceed;
  int (*Apply)(CeedQFunction, CeedInt, const CeedScalar *const *,
               CeedScalar *const *);
  int (*Destroy)(CeedQFunction);
  int refcount;
  CeedInt vlength;    // Number of quadrature points must be padded to a multiple of vlength
  struct CeedQFunctionField inputfields[16];
  struct CeedQFunctionField outputfields[16];
  CeedInt numinputfields, numoutputfields;
  int (*function)(void*, CeedInt, const CeedScalar *const*, CeedScalar *const*);
  const char *focca;
  void *ctx;      /* user context for function */
  size_t ctxsize; /* size of user context; may be used to copy to a device */
  void *data;     /* backend data */
};

struct CeedOperatorField {
  CeedElemRestriction Erestrict; /// Restriction from L-vector or NULL if identity
  CeedBasis basis;               /// Basis or NULL for collocated fields
  CeedVector
  vec;                /// State vector for passive fields, NULL for active fields
};

struct CeedOperator_private {
  Ceed ceed;
  int refcount;
  int (*Apply)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyJacobian)(CeedOperator, CeedVector, CeedVector, CeedVector,
                       CeedVector, CeedRequest *);
  int (*GetQData)(CeedOperator, CeedVector *);
  int (*Destroy)(CeedOperator);
  struct CeedOperatorField inputfields[16];
  struct CeedOperatorField outputfields[16];
  CeedInt numelements; /// Number of elements
  CeedInt numqpoints;  /// Number of quadrature points over all elements
  CeedInt nfields;     /// Number of fields that have been set
  CeedQFunction qf;
  CeedQFunction dqf;
  CeedQFunction dqfT;
  bool setupdone;
  void *data;
};

#endif
