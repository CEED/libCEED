// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Private header for frontend components of libCEED
#ifndef _ceed_impl_h
#define _ceed_impl_h

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>

CEED_INTERN const char *CeedJitSourceRootDefault;

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
  const char *func_name;
  size_t      offset;
} FOffset;

// Lookup table field for object delegates
typedef struct {
  char *obj_name;
  Ceed  delegate;
} ObjDelegate;

struct Ceed_private {
  const char  *resource;
  Ceed         delegate;
  Ceed         parent;
  ObjDelegate *obj_delegates;
  int          obj_delegate_count;
  Ceed         op_fallback_ceed, op_fallback_parent;
  const char  *op_fallback_resource;
  char       **jit_source_roots;
  CeedInt      num_jit_source_roots;
  int (*Error)(Ceed, const char *, int, const char *, int, const char *, va_list *);
  int (*GetPreferredMemType)(CeedMemType *);
  int (*Destroy)(Ceed);
  int (*VectorCreate)(CeedSize, CeedVector);
  int (*ElemRestrictionCreate)(CeedMemType, CeedCopyMode, const CeedInt *, CeedElemRestriction);
  int (*ElemRestrictionCreateOriented)(CeedMemType, CeedCopyMode, const CeedInt *, const bool *, CeedElemRestriction);
  int (*ElemRestrictionCreateBlocked)(CeedMemType, CeedCopyMode, const CeedInt *, CeedElemRestriction);
  int (*BasisCreateTensorH1)(CeedInt, CeedInt, CeedInt, const CeedScalar *, const CeedScalar *, const CeedScalar *, const CeedScalar *, CeedBasis);
  int (*BasisCreateH1)(CeedElemTopology, CeedInt, CeedInt, CeedInt, const CeedScalar *, const CeedScalar *, const CeedScalar *, const CeedScalar *,
                       CeedBasis);
  int (*BasisCreateHdiv)(CeedElemTopology, CeedInt, CeedInt, CeedInt, const CeedScalar *, const CeedScalar *, const CeedScalar *, const CeedScalar *,
                         CeedBasis);
  int (*TensorContractCreate)(CeedBasis, CeedTensorContract);
  int (*QFunctionCreate)(CeedQFunction);
  int (*QFunctionContextCreate)(CeedQFunctionContext);
  int (*OperatorCreate)(CeedOperator);
  int (*CompositeOperatorCreate)(CeedOperator);
  int      ref_count;
  void    *data;
  bool     is_debug;
  bool     has_valid_op_fallback_resource;
  bool     is_deterministic;
  char     err_msg[CEED_MAX_RESOURCE_LEN];
  FOffset *f_offsets;
};

struct CeedVector_private {
  Ceed ceed;
  int (*HasValidArray)(CeedVector, bool *);
  int (*HasBorrowedArrayOfType)(CeedVector, CeedMemType, bool *);
  int (*SetArray)(CeedVector, CeedMemType, CeedCopyMode, CeedScalar *);
  int (*SetValue)(CeedVector, CeedScalar);
  int (*SyncArray)(CeedVector, CeedMemType);
  int (*TakeArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArray)(CeedVector, CeedMemType, CeedScalar **);
  int (*GetArrayRead)(CeedVector, CeedMemType, const CeedScalar **);
  int (*GetArrayWrite)(CeedVector, CeedMemType, CeedScalar **);
  int (*RestoreArray)(CeedVector);
  int (*RestoreArrayRead)(CeedVector);
  int (*Norm)(CeedVector, CeedNormType, CeedScalar *);
  int (*Scale)(CeedVector, CeedScalar);
  int (*AXPY)(CeedVector, CeedScalar, CeedVector);
  int (*PointwiseMult)(CeedVector, CeedVector, CeedVector);
  int (*Reciprocal)(CeedVector);
  int (*Destroy)(CeedVector);
  int      ref_count;
  CeedSize length;
  uint64_t state;
  uint64_t num_readers;
  void    *data;
};

struct CeedElemRestriction_private {
  Ceed ceed;
  int (*Apply)(CeedElemRestriction, CeedTransposeMode, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyBlock)(CeedElemRestriction, CeedInt, CeedTransposeMode, CeedVector, CeedVector, CeedRequest *);
  int (*GetOffsets)(CeedElemRestriction, CeedMemType, const CeedInt **);
  int (*Destroy)(CeedElemRestriction);
  int      ref_count;
  CeedInt  num_elem;    /* number of elements */
  CeedInt  elem_size;   /* number of nodes per element */
  CeedInt  num_comp;    /* number of components */
  CeedInt  comp_stride; /* Component stride for L-vector ordering */
  CeedSize l_size;      /* size of the L-vector, can be used for checking
                             for correct vector sizes */
  CeedInt  blk_size;    /* number of elements in a batch */
  CeedInt  num_blk;     /* number of blocks of elements */
  CeedInt *strides;     /* strides between [nodes, components, elements] */
  CeedInt  layout[3];   /* E-vector layout [nodes, components, elements] */
  uint64_t num_readers; /* number of instances of offset read only access */
  bool     is_oriented; /* flag for oriented restriction */
  void    *data;        /* place for the backend to store any data */
};

struct CeedBasis_private {
  Ceed ceed;
  int (*Apply)(CeedBasis, CeedInt, CeedTransposeMode, CeedEvalMode, CeedVector, CeedVector);
  int (*Destroy)(CeedBasis);
  int              ref_count;
  bool             tensor_basis;  /* flag for tensor basis */
  CeedInt          dim;           /* topological dimension */
  CeedElemTopology topo;          /* element topology */
  CeedInt          num_comp;      /* number of field components (1 for scalar fields) */
  CeedInt          Q_comp;        /* number of Q-vector components (1 for H^1, dim for H(div)) */
  CeedInt          P_1d;          /* number of nodes in one dimension */
  CeedInt          Q_1d;          /* number of quadrature points in one dimension */
  CeedInt          P;             /* total number of nodes */
  CeedInt          Q;             /* total number of quadrature points */
  CeedScalar      *q_ref_1d;      /* Array of length Q1d holding the locations of
                                       quadrature points on the 1D reference element [-1, 1] */
  CeedScalar *q_weight_1d;        /* array of length Q1d holding the quadrature weights on
                                                 the reference element */
  CeedScalar *interp;             /* row-major matrix of shape [Q_comp*Q, P] expressing the values of
                                       nodal basis functions at quadrature points */
  CeedScalar *interp_1d;          /* row-major matrix of shape [Q1d, P1d] expressing the values of
                                       nodal basis functions at quadrature points */
  CeedScalar *grad;               /* row-major matrix of shape [dim*Q_comp*Q, P] matrix expressing
                                       derivatives of nodal basis functions at quadrature points */
  CeedScalar *grad_1d;            /* row-major matrix of shape [Q1d, P1d] matrix expressing
                                       derivatives of nodal basis functions at quadrature points */
  CeedTensorContract contract;    /* tensor contraction object */
  CeedFESpace        basis_space; /* Initialize in basis constructor
                                       with 1,2 for H^1, H(div) FE space */
  CeedScalar *div;                /* row-major matrix of shape [Q, P] expressing
                                      the divergence of nodal basis functions
                                      at quadrature points for H(div) discretizations */
  void *data;                     /* place for the backend to store any data */
};

struct CeedTensorContract_private {
  Ceed ceed;
  int (*Apply)(CeedTensorContract, CeedInt, CeedInt, CeedInt, CeedInt, const CeedScalar *restrict, CeedTransposeMode, const CeedInt,
               const CeedScalar *restrict, CeedScalar *restrict);
  int (*Destroy)(CeedTensorContract);
  int   ref_count;
  void *data;
};

struct CeedQFunctionField_private {
  const char  *field_name;
  CeedInt      size;
  CeedEvalMode eval_mode;
};

struct CeedQFunction_private {
  Ceed ceed;
  int (*Apply)(CeedQFunction, CeedInt, CeedVector *, CeedVector *);
  int (*SetCUDAUserFunction)(CeedQFunction, void *);
  int (*SetHIPUserFunction)(CeedQFunction, void *);
  int (*Destroy)(CeedQFunction);
  int     ref_count;
  CeedInt vec_length; /* Number of quadrature points must be padded to a
                           multiple of vec_length */
  CeedQFunctionField  *input_fields;
  CeedQFunctionField  *output_fields;
  CeedInt              num_input_fields, num_output_fields;
  CeedQFunctionUser    function;
  CeedInt              user_flop_estimate;
  const char          *user_source;
  const char          *source_path;
  const char          *kernel_name;
  const char          *gallery_name;
  bool                 is_gallery;
  bool                 is_identity;
  bool                 is_fortran;
  bool                 is_immutable;
  bool                 is_context_writable;
  CeedQFunctionContext ctx;  /* user context for function */
  void                *data; /* place for the backend to store any data */
};

struct CeedQFunctionContext_private {
  Ceed ceed;
  int  ref_count;
  int (*HasValidData)(CeedQFunctionContext, bool *);
  int (*HasBorrowedDataOfType)(CeedQFunctionContext, CeedMemType, bool *);
  int (*SetData)(CeedQFunctionContext, CeedMemType, CeedCopyMode, void *);
  int (*TakeData)(CeedQFunctionContext, CeedMemType, void *);
  int (*GetData)(CeedQFunctionContext, CeedMemType, void *);
  int (*GetDataRead)(CeedQFunctionContext, CeedMemType, void *);
  int (*RestoreData)(CeedQFunctionContext);
  int (*RestoreDataRead)(CeedQFunctionContext);
  int (*DataDestroy)(CeedQFunctionContext);
  int (*Destroy)(CeedQFunctionContext);
  CeedQFunctionContextDataDestroyUser data_destroy_function;
  CeedMemType                         data_destroy_mem_type;
  CeedInt                             num_fields;
  CeedInt                             max_fields;
  CeedContextFieldLabel              *field_labels;
  uint64_t                            state;
  uint64_t                            num_readers;
  size_t                              ctx_size;
  void                               *data;
};

/// Struct to handle the context data to use the Fortran QFunction stub
/// @ingroup CeedQFunction
struct CeedFortranContext_private {
  CeedQFunctionContext inner_ctx;
  void (*f)(void *ctx, int *nq, const CeedScalar *u, const CeedScalar *u1, const CeedScalar *u2, const CeedScalar *u3, const CeedScalar *u4,
            const CeedScalar *u5, const CeedScalar *u6, const CeedScalar *u7, const CeedScalar *u8, const CeedScalar *u9, const CeedScalar *u10,
            const CeedScalar *u11, const CeedScalar *u12, const CeedScalar *u13, const CeedScalar *u14, const CeedScalar *u15, CeedScalar *v,
            CeedScalar *v1, CeedScalar *v2, CeedScalar *v3, CeedScalar *v4, CeedScalar *v5, CeedScalar *v6, CeedScalar *v7, CeedScalar *v8,
            CeedScalar *v9, CeedScalar *v10, CeedScalar *v11, CeedScalar *v12, CeedScalar *v13, CeedScalar *v14, CeedScalar *v15, int *err);
};
typedef struct CeedFortranContext_private *CeedFortranContext;

struct CeedContextFieldLabel_private {
  const char            *name;
  const char            *description;
  CeedContextFieldType   type;
  size_t                 size;
  size_t                 num_values;
  size_t                 offset;
  CeedInt                num_sub_labels;
  CeedContextFieldLabel *sub_labels;
};

struct CeedOperatorField_private {
  CeedElemRestriction elem_restr; /* Restriction from L-vector */
  CeedBasis           basis;      /* Basis or CEED_BASIS_COLLOCATED for
                                       collocated fields */
  CeedVector vec;                 /* State vector for passive fields or
                                       CEED_VECTOR_NONE for no vector */
  const char *field_name;         /* matching QFunction field name */
};

struct CeedQFunctionAssemblyData_private {
  Ceed                ceed;
  int                 ref_count;
  bool                is_setup;
  bool                reuse_data;
  bool                needs_data_update;
  CeedVector          vec;
  CeedElemRestriction rstr;
};

struct CeedOperatorAssemblyData_private {
  Ceed          ceed;
  CeedInt       num_eval_mode_in, num_eval_mode_out;
  CeedEvalMode *eval_mode_in, *eval_mode_out;
  CeedScalar   *B_in, *B_out;
  CeedBasis     basis_in, basis_out;
};

struct CeedOperator_private {
  Ceed         ceed;
  CeedOperator op_fallback;
  int          ref_count;
  int (*LinearAssembleQFunction)(CeedOperator, CeedVector *, CeedElemRestriction *, CeedRequest *);
  int (*LinearAssembleQFunctionUpdate)(CeedOperator, CeedVector, CeedElemRestriction, CeedRequest *);
  int (*LinearAssembleDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssembleAddDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssemblePointBlockDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssembleAddPointBlockDiagonal)(CeedOperator, CeedVector, CeedRequest *);
  int (*LinearAssembleSymbolic)(CeedOperator, CeedSize *, CeedInt **, CeedInt **);
  int (*LinearAssemble)(CeedOperator, CeedVector);
  int (*LinearAssembleSingle)(CeedOperator, CeedInt, CeedVector);
  int (*CreateFDMElementInverse)(CeedOperator, CeedOperator *, CeedRequest *);
  int (*Apply)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyComposite)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyAdd)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyAddComposite)(CeedOperator, CeedVector, CeedVector, CeedRequest *);
  int (*ApplyJacobian)(CeedOperator, CeedVector, CeedVector, CeedVector, CeedVector, CeedRequest *);
  int (*Destroy)(CeedOperator);
  CeedOperatorField        *input_fields;
  CeedOperatorField        *output_fields;
  CeedSize                  input_size, output_size;
  CeedInt                   num_elem;   /* Number of elements */
  CeedInt                   num_qpts;   /* Number of quadrature points over all elements */
  CeedInt                   num_fields; /* Number of fields that have been set */
  CeedQFunction             qf;
  CeedQFunction             dqf;
  CeedQFunction             dqfT;
  const char               *name;
  bool                      is_immutable;
  bool                      is_interface_setup;
  bool                      is_backend_setup;
  bool                      is_composite;
  bool                      has_restriction;
  CeedQFunctionAssemblyData qf_assembled;
  CeedOperatorAssemblyData  op_assembled;
  CeedOperator             *sub_operators;
  CeedInt                   num_suboperators;
  void                     *data;
  CeedInt                   num_context_labels;
  CeedInt                   max_context_labels;
  CeedContextFieldLabel    *context_labels;
};

CEED_INTERN int CeedOperatorGetFallback(CeedOperator op, CeedOperator *op_fallback);

#endif
