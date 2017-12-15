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

typedef struct CeedBasisScalarGeneric_private *CeedBasisScalarGeneric;
typedef struct CeedBasisScalarTensor_private *CeedBasisScalarTensor;

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p
   has to be a pointer to a pointer. */
CEED_INTERN int CeedMallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedCallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedFree(void *p);

#define CeedChk(ierr) do { if (ierr) return ierr; } while (0)
/* Note that CeedMalloc and CeedCalloc will, generally, return pointers with
   different memory alignments: CeedMalloc returns pointers aligned at
   CEED_ALIGN bytes, while CeedCalloc uses the alignment of calloc. */
#define CeedMalloc(n, p) CeedMallocArray((n), sizeof(**(p)), p)
#define CeedCalloc(n, p) CeedCallocArray((n), sizeof(**(p)), p)

CEED_INTERN int CeedBasisScalarGenericCreate(CeedInt ndof, CeedInt nqpt,
    CeedInt dim, CeedBasisScalarGeneric *basis);
CEED_INTERN int CeedBasisScalarGenericDestroy(CeedBasisScalarGeneric *basis);
CEED_INTERN int CeedBasisScalarTensorCreate(CeedInt ndof1d, CeedInt nqpt1d,
    CeedInt dim, CeedBasisScalarTensor *basis);
CEED_INTERN int CeedBasisScalarTensorDestroy(CeedBasisScalarTensor *basis);

struct Ceed_private {
  int (*Error)(Ceed, const char *, int, const char *, int, const char *, va_list);
  int (*Destroy)(Ceed);
  int (*VecCreate)(Ceed, CeedInt, CeedVector);
  int (*ElemRestrictionCreate)(CeedElemRestriction, CeedMemType, CeedCopyMode,
                               const CeedInt *);
  int (*BasisCreateScalarGeneric)(CeedBasis, CeedBasisScalarGeneric);
  int (*BasisCreateScalarTensor)(CeedBasis, CeedBasisScalarTensor);
  int (*QFunctionCreate)(CeedQFunction);
  int (*OperatorCreate)(CeedOperator);
};

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

typedef enum {
  CEED_BASIS_NO_DATA = 0,     /**< Initial value: host_data is not set */
  CEED_BASIS_SCALAR_GENERIC,  /**< Corresponds to CeedBasisScalarGeneric */
  CEED_BASIS_SCALAR_TENSOR,   /**< Corresponds to CeedBasisScalarTensor */
} CeedBasisDataType;

struct CeedBasis_private {
  Ceed ceed; // associated Ceed object
  // The function pointers Apply and Destroy will be set by the backend.
  int (*Apply)(CeedBasis, CeedTransposeMode, CeedEvalMode, const CeedScalar *,
               CeedScalar *);
  int (*Destroy)(CeedBasis);
  CeedGeometry geom;  // type of the reference element
  CeedBasisType btype;
  CeedInt degree; // polynomial degree of the basis functions: k in Pk/Qk
  CeedQuadMode node_locations; // node = point where a DOF is defined
  CeedInt qorder; // quadrature rule order; -1 = unknown (custom)
  CeedQuadMode qmode;
  CeedInt dim;  // dimension of the reference element space
  CeedInt ndof; // number of degrees of freedom / number of basis functions
  CeedInt nqpt; // number of quadrature points
  CeedInt ncomp; // number of components
  /* TODO: move ncomp to the restriction; also, add it as an argument to
           CeedBasisApply. */
  CeedBasisDataType dtype; // data type describing 'host_data'
  void *host_data; // host representation of the basis data
  void *be_data; // backend data
};

struct CeedBasisScalarGeneric_private {
  CeedInt ndof; // number of degrees of freedom
  CeedInt nqpt; // number of quadrature points
  CeedInt dim;  // dimension of the reference element space
  CeedScalar *interp; // nqpt x ndof (column-major layout)
  CeedScalar *grad;   // nqpt x dim x ndof (column-major layout)
  CeedScalar *qweights; // quadrature point weights, size = nqpt
};

struct CeedBasisScalarTensor_private {
  CeedInt P1d; // = (degree + 1) from CeedBasis_private; ndof = P1d^dim
  CeedInt Q1d; // number of quadrature points in 1D; nqpt = Q1d^dim
  CeedInt dim; // dimension of the reference space
  CeedScalar *qref1d;  // locations of the 1D quadrature points, size = Q1d
  CeedScalar *qweight1d; // weights of the 1D quadrature, size = Q1d
  CeedScalar *interp1d;  // Q1d x P1d (column-major layout)
  CeedScalar *grad1d;    // Q1d x P1d (column-major layout)
};

/* FIXME: The number of in-fields and out-fields may be different? */
/* FIXME: Shouldn't inmode and outmode be per-in-field and per-out-field,
   respectively? */
struct CeedQFunction_private {
  Ceed ceed;
  int (*Apply)(CeedQFunction, void *, CeedInt, const CeedScalar *const *,
               CeedScalar *const *);
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
