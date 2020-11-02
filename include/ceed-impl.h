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

/// @file
/// Private header for frontend components of libCEED
#ifndef _ceed_impl_h
#define _ceed_impl_h

#include <ceed.h>
#include <ceed-backend.h>
#include <stdbool.h>

/** @defgroup CeedUser Public API for Ceed
    @ingroup Ceed
*/
/** @defgroup CeedBackend Backend API for Ceed
    @ingroup Ceed
*/
/** @defgroup CeedDeveloper Internal library functions for Ceed
    @ingroup Ceed
*/
/** @defgroup CeedVectorUser Public API for CeedVector
    @ingroup CeedVector
*/
/** @defgroup CeedVectorBackend Backend API for CeedVector
    @ingroup CeedVector
*/
/** @defgroup CeedVectorDeveloper Internal library functions for CeedVector
    @ingroup CeedVector
*/
/** @defgroup CeedElemRestrictionUser Public API for CeedElemRestriction
    @ingroup CeedElemRestriction
*/
/** @defgroup CeedElemRestrictionBackend Backend API for CeedElemRestriction
    @ingroup CeedElemRestriction
*/
/** @defgroup CeedElemRestrictionDeveloper Internal library functions for CeedElemRestriction
    @ingroup CeedElemRestriction
*/
/** @defgroup CeedBasisUser Public API for CeedBasis
    @ingroup CeedBasis
*/
/** @defgroup CeedBasisBackend Backend API for CeedBasis
    @ingroup CeedBasis
*/
/** @defgroup CeedBasisDeveloper Internal library functions for CeedBasis
    @ingroup CeedBasis
*/
/** @defgroup CeedQFunctionUser Public API for CeedQFunction
    @ingroup CeedQFunction
*/
/** @defgroup CeedQFunctionBackend Backend API for CeedQFunction
    @ingroup CeedQFunction
*/
/** @defgroup CeedQFunctionDeveloper Internal library functions for CeedQFunction
    @ingroup CeedQFunction
*/
/** @defgroup CeedOperatorUser Public API for CeedOperator
    @ingroup CeedOperator
*/
/** @defgroup CeedOperatorBackend Backend API for CeedOperator
    @ingroup CeedOperator
*/
/** @defgroup CeedOperatorDeveloper Internal library functions for CeedOperator
    @ingroup CeedOperator
*/

// Lookup table field for backend functions
typedef struct {
  const char *fname;
  size_t offset;
} foffset;

// Lookup table field for object delegates
typedef struct {
  char *objname;
  Ceed delegate;
} objdelegate;

struct Ceed_private {
  const char *resource;
  Ceed delegate;
  Ceed parent;
  objdelegate *objdelegates;
  int objdelegatecount;
  Ceed opfallbackceed, opfallbackparent;
  const char *opfallbackresource;
  int (*Error)(Ceed, const char *, int, const char *, int, const char *,
               va_list *);
  int (*GetPreferredMemType)(CeedMemType *);
  int (*Destroy)(Ceed);
  int (*VectorCreate)(CeedInt, CeedVector);
  int (*ElemRestrictionCreate)(CeedMemType, CeedCopyMode,
                               const CeedInt *, CeedElemRestriction);
  int (*ElemRestrictionCreateBlocked)(CeedMemType, CeedCopyMode,
                                      const CeedInt *, CeedElemRestriction);
  int (*BasisCreateTensorH1)(CeedInt, CeedInt, CeedInt, const CeedScalar *,
                             const CeedScalar *, const CeedScalar *,
                             const CeedScalar *, CeedBasis);
  int (*BasisCreateH1)(CeedElemTopology, CeedInt, CeedInt, CeedInt,
                       const CeedScalar *,
                       const CeedScalar *, const CeedScalar *,
                       const CeedScalar *, CeedBasis);
  int (*TensorContractCreate)(CeedBasis, CeedTensorContract);
  int (*QFunctionCreate)(CeedQFunction);
  int (*QFunctionContextCreate)(CeedQFunctionContext);
  int (*OperatorCreate)(CeedOperator);
  int (*CompositeOperatorCreate)(CeedOperator);
  int refcount;
  bool isDeterministic;
  void *data;
  bool debug;
  char errmsg[CEED_MAX_RESOURCE_LEN];
  foffset *foffsets;
};

struct CeedVector_private {
  Ceed ceed;
  int (*SetArray)(CeedVector, CeedMemType, CeedCopyMode, CeedScalar *);
  int (*SetValue)(CeedVector, CeedScalar);
  int (*SyncArray)(CeedVector, CeedMemType);
  int (*TakeArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArrayRead)(CeedVector, CeedMemType, const CeedScalar **);
  int (*RestoreArray)(CeedVector);
  int (*RestoreArrayRead)(CeedVector);
  int (*Norm)(CeedVector, CeedNormType, CeedScalar *);
  int (*Reciprocal)(CeedVector);
  int (*Destroy)(CeedVector);
  int refcount;
  CeedInt length;
  uint64_t state;
  uint64_t numreaders;
  void *data;
};

struct CeedElemRestriction_private {
  Ceed ceed;
  int (*Apply)(CeedElemRestriction, CeedTransposeMode, CeedVector, CeedVector,
               CeedRequest *);
  int (*ApplyBlock)(CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector,
                    CeedVector, CeedRequest *);
  int (*GetOffsets)(CeedElemRestriction, CeedMemType, const CeedInt **);
  int (*Destroy)(CeedElemRestriction);
  int refcount;
  CeedInt nelem;            /* number of elements */
  CeedInt elemsize;         /* number of nodes per element */
  CeedInt ncomp;            /* number of components */
  CeedInt compstride;       /* Component stride for L-vector ordering */
  CeedInt lsize;            /* size of the L-vector, can be used for checking
                                 for correct vector sizes */
  CeedInt blksize;          /* number of elements in a batch */
  CeedInt nblk;             /* number of blocks of elements */
  CeedInt *strides;         /* strides between [nodes, components, elements] */
  CeedInt layout[3];        /* E-vector layout [nodes, components, elements] */
  uint64_t numreaders;      /* number of instances of offset read only access */
  void *data;               /* place for the backend to store any data */
};

struct CeedBasis_private {
  Ceed ceed;
  int (*Apply)(CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode,
               CeedVector, CeedVector);
  int (*Destroy)(CeedBasis);
  int refcount;
  bool tensorbasis;      /* flag for tensor basis */
  CeedInt dim;           /* topological dimension */
  CeedElemTopology topo; /* element topology */
  CeedInt ncomp;         /* number of field components (1 for scalar fields) */
  CeedInt P1d;           /* number of nodes in one dimension */
  CeedInt Q1d;           /* number of quadrature points in one dimension */
  CeedInt P;             /* total number of nodes */
  CeedInt Q;             /* total number of quadrature points */
  CeedScalar *qref1d;    /* Array of length Q1d holding the locations of
                              quadrature points on the 1D reference
                              element [-1, 1] */
  CeedScalar *qweight1d; /* array of length Q1d holding the quadrature weights on
                              the reference element */
  CeedScalar
  *interp;    /* row-major matrix of shape [Q, P] expressing the values of
                   nodal basis functions at quadrature points */
  CeedScalar
  *interp1d;  /* row-major matrix of shape [Q1d, P1d] expressing the values of
                   nodal basis functions at quadrature points */
  CeedScalar
  *grad;      /* row-major matrix of shape [dim*Q, P] matrix expressing
                   derivatives of nodal basis functions at quadrature points */
  CeedScalar
  *grad1d;    /* row-major matrix of shape [Q1d, P1d] matrix expressing
                   derivatives of nodal basis functions at quadrature points */
  CeedTensorContract contract; /* tensor contraction object */
  void *data;                  /* place for the backend to store any data */
};

struct CeedTensorContract_private {
  Ceed ceed;
  int (*Apply)(CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt,
               const CeedScalar *restrict, CeedTransposeMode, const CeedInt,
               const CeedScalar *restrict, CeedScalar *restrict);
  int (*Destroy)(CeedTensorContract);
  int refcount;
  void *data;
};

struct CeedQFunctionField_private {
  const char *fieldname;
  CeedInt size;
  CeedEvalMode emode;
};

struct CeedQFunction_private {
  Ceed ceed;
  int (*Apply)(CeedQFunction, CeedInt, CeedVector *, CeedVector *);
  int (*SetCUDAUserFunction)(CeedQFunction, void *);
  int (*SetHIPUserFunction)(CeedQFunction, void *);
  int (*Destroy)(CeedQFunction);
  int refcount;
  CeedInt vlength;    /* Number of quadrature points must be padded to a
                           multiple of vlength */
  CeedQFunctionField *inputfields;
  CeedQFunctionField *outputfields;
  CeedInt numinputfields, numoutputfields;
  CeedQFunctionUser function;
  const char *sourcepath;
  const char *qfname;
  bool identity;
  bool fortranstatus;
  CeedQFunctionContext ctx; /* user context for function */
  void *data;          /* place for the backend to store any data */
};

struct CeedQFunctionContext_private {
  Ceed ceed;
  int refcount;
  int (*SetData)(CeedQFunctionContext, CeedMemType, CeedCopyMode, void *);
  int (*GetData)(CeedQFunctionContext, CeedMemType, void *);
  int (*RestoreData)(CeedQFunctionContext);
  int (*Destroy)(CeedQFunctionContext);
  uint64_t state;
  size_t ctxsize;
  void *data;
};

/// Struct to handle the context data to use the Fortran QFunction stub
/// @ingroup CeedQFunction
struct CeedFortranContext_private {
  CeedQFunctionContext innerctx;
  void (*f)(void *ctx, int *nq,
            const CeedScalar *u,const CeedScalar *u1,
            const CeedScalar *u2,const CeedScalar *u3,
            const CeedScalar *u4,const CeedScalar *u5,
            const CeedScalar *u6,const CeedScalar *u7,
            const CeedScalar *u8,const CeedScalar *u9,
            const CeedScalar *u10,const CeedScalar *u11,
            const CeedScalar *u12,const CeedScalar *u13,
            const CeedScalar *u14,const CeedScalar *u15,
            CeedScalar *v,CeedScalar *v1,CeedScalar *v2,
            CeedScalar *v3,CeedScalar *v4,CeedScalar *v5,
            CeedScalar *v6,CeedScalar *v7,CeedScalar *v8,
            CeedScalar *v9, CeedScalar *v10,CeedScalar *v11,
            CeedScalar *v12,CeedScalar *v13,CeedScalar *v14,
            CeedScalar *v15, int *err);
};
typedef struct CeedFortranContext_private *CeedFortranContext;

struct CeedOperatorField_private {
  CeedElemRestriction Erestrict; /* Restriction from L-vector */
  CeedBasis basis;               /* Basis or CEED_BASIS_COLLOCATED for
                                      collocated fields */
  CeedVector vec;                /* State vector for passive fields or
                                      CEED_VECTOR_NONE for no vector */
  const char *fieldname;         /* matching QFunction field name */
};

struct CeedOperator_private {
  Ceed ceed;
  CeedOperator opfallback;
  CeedQFunction qffallback;
  int refcount;
  int (*LinearAssembleQFunction)(CeedOperator, CeedVector *,
                                 CeedElemRestriction *, CeedRequest *);
  int (*LinearAssembleDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssembleAddDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssemblePointBlockDiagonal)(CeedOperator, CeedVector,
                                          CeedRequest *);
  int (*LinearAssembleAddPointBlockDiagonal)(CeedOperator, CeedVector,
      CeedRequest *);
  int (*CreateFDMElementInverse)(CeedOperator, CeedOperator *, CeedRequest *);
  int (*Apply)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyComposite)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyAdd)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyAddComposite)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyJacobian)(CeedOperator, CeedVector, CeedVector, CeedVector,
                       CeedVector, CeedRequest *);
  int (*Destroy)(CeedOperator);
  CeedOperatorField *inputfields;
  CeedOperatorField *outputfields;
  CeedInt numelements; /// Number of elements
  CeedInt numqpoints;  /// Number of quadrature points over all elements
  CeedInt nfields;     /// Number of fields that have been set
  CeedQFunction qf;
  CeedQFunction dqf;
  CeedQFunction dqfT;
  bool setupdone;
  bool composite;
  bool hasrestriction;
  CeedOperator *suboperators;
  CeedInt numsub;
  void *data;
};

#endif
