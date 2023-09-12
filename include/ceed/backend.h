// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for backend components of libCEED
#ifndef CEED_BACKEND_H
#define CEED_BACKEND_H

#include <ceed.h>
#include <limits.h>
#include <stdbool.h>

#if defined(__clang_analyzer__)
#define CEED_INTERN
#elif defined(__cplusplus)
#define CEED_INTERN extern "C" CEED_VISIBILITY(hidden)
#else
#define CEED_INTERN extern CEED_VISIBILITY(hidden)
#endif

#define CEED_UNUSED __attribute__((unused))

#define CEED_MAX_RESOURCE_LEN 1024
#define CEED_MAX_BACKEND_PRIORITY UINT_MAX
#define CEED_ALIGN 64
#define CEED_COMPOSITE_MAX 16
#define CEED_FIELD_MAX 16

#ifndef CeedPragmaOptimizeOff
#if defined(__clang__)
/// This macro provides the ability to disable optimization flags for functions that are sensitive to floating point optimizations.
/// @ingroup Ceed
#define CeedPragmaOptimizeOff _Pragma("clang optimize off")
#elif defined(__GNUC__)
#define CeedPragmaOptimizeOff _Pragma("GCC push_options") _Pragma("GCC optimize 0")
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define CeedPragmaOptimizeOff _Pragma("optimize('', off)")
#else
#define CeedPragmaOptimizeOff
#endif
#endif

#ifndef CeedPragmaOptimizeOn
#if defined(__clang__)
/// This macro restores previously set optimization flags after CeedPragmaOptimizeOff.
/// @ingroup Ceed
#define CeedPragmaOptimizeOn _Pragma("clang optimize on")
#elif defined(__GNUC__)
#define CeedPragmaOptimizeOn _Pragma("GCC pop_options")
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define CeedPragmaOptimizeOff _Pragma("optimize('', on)")
#else
#define CeedPragmaOptimizeOn
#endif
#endif

/**
  This enum supples common colors for CeedDebug256 debugging output.
  Set the environment variable `CEED_DEBUG = 1` to activate debugging output.

  @ingroup Ceed
  @ref     Backend
**/
typedef enum {
  /// Success color
  CEED_DEBUG_COLOR_SUCCESS = 108,
  /// Warning color
  CEED_DEBUG_COLOR_WARNING = 208,
  /// Error color
  CEED_DEBUG_COLOR_ERROR = 196,
  /// Use native terminal coloring
  CEED_DEBUG_COLOR_NONE = 255,
} CeedDebugColor;

CEED_EXTERN void CeedDebugImpl256(const unsigned char, const char *, ...);
CEED_EXTERN bool CeedDebugFlag(const Ceed ceed);
CEED_EXTERN bool CeedDebugFlagEnv(void);
/**
  Print debugging information in color

  @param[in] ceed  Ceed
  @param[in] color Color to print with

  @ingroup Ceed
  @ref     Backend
**/
#define CeedDebug256(ceed, color, ...)                               \
  {                                                                  \
    if (CeedDebugFlag(ceed)) CeedDebugImpl256(color, ##__VA_ARGS__); \
  }
/**
  Print debugging information to terminal

  @param[in] ceed  Ceed

  @ingroup Ceed
  @ref     Backend
**/
#define CeedDebug(ceed, ...) CeedDebug256(ceed, (unsigned char)CEED_DEBUG_COLOR_NONE, ##__VA_ARGS__)
/**
  Print debugging information in color without Ceed to reference

  @param[in] color Color to print with

  @ingroup Ceed
  @ref     Backend
**/
#define CeedDebugEnv256(color, ...)                                 \
  {                                                                 \
    if (CeedDebugFlagEnv()) CeedDebugImpl256(color, ##__VA_ARGS__); \
  }
/**
  Print debugging information to terminal without Ceed to reference

  @ingroup Ceed
  @ref     Backend
**/
#define CeedDebugEnv(...) CeedDebugEnv256((unsigned char)CEED_DEBUG_COLOR_NONE, ##__VA_ARGS__)
/**
  Print warning information in color

  @ingroup Ceed
  @ref     Backend
**/
#define CeedWarn(...) \
  { CeedDebugImpl256(CEED_DEBUG_COLOR_WARNING, ##__VA_ARGS__); }

/**
  Swap the values of two CeedScalars

  @param[in,out] a  First CeedScalar
  @param[in,out] b  Second CeedScalar

  @ingroup Ceed
  @ref     Backend
**/
#define CeedScalarSwap(a, b)        \
  {                                 \
    const CeedScalar temp_ = a;     \
    a                      = b;     \
    b                      = temp_; \
  }

/// Handle for object handling TensorContraction
/// @ingroup CeedBasis
typedef struct CeedTensorContract_private *CeedTensorContract;

/// Handle for object handling assembled QFunction data
/// @ingroup CeedOperator
typedef struct CeedQFunctionAssemblyData_private *CeedQFunctionAssemblyData;

/// Handle for object handling assembled Operator data
/// @ingroup CeedOperator
typedef struct CeedOperatorAssemblyData_private *CeedOperatorAssemblyData;

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p has to be a pointer to a pointer. */
CEED_INTERN int CeedMallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedCallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedReallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedStringAllocCopy(const char *source, char **copy);
CEED_INTERN int CeedFree(void *p);

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then the error handler is called and the call from the current function with the error code.

  @ref Developer
**/
#define CeedCall(...)        \
  do {                       \
    int ierr_ = __VA_ARGS__; \
    if (ierr_) return ierr_; \
  } while (0)

/**
  @brief Calls a libCEED function and then checks the resulting error code.
  If the error code is non-zero, then the error handler is called and the call from the current function with the error code.
  All interface level error codes are upgraded to `CEED_ERROR_BACKEND`.

  @ref Developer
**/
#define CeedCallBackend(...)                                                     \
  do {                                                                           \
    int ierr_ = __VA_ARGS__;                                                     \
    if (ierr_) return (ierr_ > CEED_ERROR_SUCCESS) ? CEED_ERROR_BACKEND : ierr_; \
  } while (0)

/**
  @brief Check that a particular condition is true and returns a `CeedError` if not.

  @ref Developer
**/
#define CeedCheck(cond, ceed, ecode, ...)                    \
  do {                                                       \
    if (!(cond)) return CeedError(ceed, ecode, __VA_ARGS__); \
  } while (0)

/* Note - these are legacy macros that should be removed eventually */
#define CeedChk(...) CeedCall(__VA_ARGS__)
#define CeedChkBackend(...) CeedCallBackend(__VA_ARGS__)

/* Note that CeedMalloc and CeedCalloc will, generally, return pointers with different memory alignments:
   CeedMalloc returns pointers aligned at CEED_ALIGN bytes, while CeedCalloc uses the alignment of calloc. */
#define CeedMalloc(n, p) CeedMallocArray((n), sizeof(**(p)), p)
#define CeedCalloc(n, p) CeedCallocArray((n), sizeof(**(p)), p)
#define CeedRealloc(n, p) CeedReallocArray((n), sizeof(**(p)), p)

CEED_EXTERN int CeedRegister(const char *prefix, int (*init)(const char *, Ceed), unsigned int priority);
CEED_EXTERN int CeedRegisterImpl(const char *prefix, int (*init)(const char *, Ceed), unsigned int priority);

CEED_EXTERN int CeedIsDebug(Ceed ceed, bool *is_debug);
CEED_EXTERN int CeedGetResourceRoot(Ceed ceed, const char *resource, const char *delineator, char **resource_root);
CEED_EXTERN int CeedGetParent(Ceed ceed, Ceed *parent);
CEED_EXTERN int CeedGetDelegate(Ceed ceed, Ceed *delegate);
CEED_EXTERN int CeedSetDelegate(Ceed ceed, Ceed delegate);
CEED_EXTERN int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate, const char *obj_name);
CEED_EXTERN int CeedSetObjectDelegate(Ceed ceed, Ceed delegate, const char *obj_name);
CEED_EXTERN int CeedGetOperatorFallbackResource(Ceed ceed, const char **resource);
CEED_EXTERN int CeedGetOperatorFallbackCeed(Ceed ceed, Ceed *fallback_ceed);
CEED_EXTERN int CeedSetOperatorFallbackResource(Ceed ceed, const char *resource);
CEED_EXTERN int CeedSetDeterministic(Ceed ceed, bool is_deterministic);
CEED_EXTERN int CeedSetBackendFunction(Ceed ceed, const char *type, void *object, const char *func_name, int (*f)());
CEED_EXTERN int CeedGetData(Ceed ceed, void *data);
CEED_EXTERN int CeedSetData(Ceed ceed, void *data);
CEED_EXTERN int CeedReference(Ceed ceed);

CEED_EXTERN int CeedVectorHasValidArray(CeedVector vec, bool *has_valid_array);
CEED_EXTERN int CeedVectorHasBorrowedArrayOfType(CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type);
CEED_EXTERN int CeedVectorHasValidArray(CeedVector vec, bool *has_valid_array);
CEED_EXTERN int CeedVectorGetState(CeedVector vec, uint64_t *state);
CEED_EXTERN int CeedVectorGetData(CeedVector vec, void *data);
CEED_EXTERN int CeedVectorSetData(CeedVector vec, void *data);
CEED_EXTERN int CeedVectorReference(CeedVector vec);

/// Type of element restriction;
/// @ingroup CeedElemRestriction
typedef enum {
  /// Standard element restriction with offsets
  CEED_RESTRICTION_STANDARD = 1,
  /// Oriented element restriction
  CEED_RESTRICTION_ORIENTED = 2,
  /// Curl-oriented element restriction
  CEED_RESTRICTION_CURL_ORIENTED = 3,
  /// Strided element restriction
  CEED_RESTRICTION_STRIDED = 4,
  /// Point-in-cell element restriction
  CEED_RESTRICTION_POINTS = 5,
} CeedRestrictionType;

CEED_EXTERN int CeedElemRestrictionGetType(CeedElemRestriction rstr, CeedRestrictionType *rstr_type);
CEED_EXTERN int CeedElemRestrictionIsStrided(CeedElemRestriction rstr, bool *is_strided);
CEED_EXTERN int CeedElemRestrictionIsPoints(CeedElemRestriction rstr, bool *is_points);
CEED_EXTERN int CeedElemRestrictionGetStrides(CeedElemRestriction rstr, CeedInt (*strides)[3]);
CEED_EXTERN int CeedElemRestrictionHasBackendStrides(CeedElemRestriction rstr, bool *has_backend_strides);
CEED_EXTERN int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets);
CEED_EXTERN int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr, const CeedInt **offsets);
CEED_EXTERN int CeedElemRestrictionGetOrientations(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients);
CEED_EXTERN int CeedElemRestrictionRestoreOrientations(CeedElemRestriction rstr, const bool **orients);
CEED_EXTERN int CeedElemRestrictionGetCurlOrientations(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients);
CEED_EXTERN int CeedElemRestrictionRestoreCurlOrientations(CeedElemRestriction rstr, const CeedInt8 **curl_orients);
CEED_EXTERN int CeedElemRestrictionGetNumPointsInElement(CeedElemRestriction rstr, CeedInt elem, CeedInt *num_points);
CEED_EXTERN int CeedElemRestrictionGetELayout(CeedElemRestriction rstr, CeedInt (*layout)[3]);
CEED_EXTERN int CeedElemRestrictionSetELayout(CeedElemRestriction rstr, CeedInt layout[3]);
CEED_EXTERN int CeedElemRestrictionGetData(CeedElemRestriction rstr, void *data);
CEED_EXTERN int CeedElemRestrictionSetData(CeedElemRestriction rstr, void *data);
CEED_EXTERN int CeedElemRestrictionReference(CeedElemRestriction rstr);
CEED_EXTERN int CeedElemRestrictionGetFlopsEstimate(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedSize *flops);

/// Type of FE space;
/// @ingroup CeedBasis
typedef enum {
  /// H^1 FE space
  CEED_FE_SPACE_H1 = 1,
  /// H(div) FE space
  CEED_FE_SPACE_HDIV = 2,
  /// H(curl) FE space
  CEED_FE_SPACE_HCURL = 3,
} CeedFESpace;
CEED_EXTERN const char *const CeedFESpaces[];

CEED_EXTERN int CeedBasisGetCollocatedGrad(CeedBasis basis, CeedScalar *colo_grad_1d);
CEED_EXTERN int CeedBasisIsTensor(CeedBasis basis, bool *is_tensor);
CEED_EXTERN int CeedBasisGetData(CeedBasis basis, void *data);
CEED_EXTERN int CeedBasisSetData(CeedBasis basis, void *data);
CEED_EXTERN int CeedBasisReference(CeedBasis basis);
CEED_EXTERN int CeedBasisGetNumQuadratureComponents(CeedBasis basis, CeedEvalMode eval_mode, CeedInt *q_comp);
CEED_EXTERN int CeedBasisGetFlopsEstimate(CeedBasis basis, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedSize *flops);
CEED_EXTERN int CeedBasisGetFESpace(CeedBasis basis, CeedFESpace *fe_space);
CEED_EXTERN int CeedBasisGetTopologyDimension(CeedElemTopology topo, CeedInt *dim);
CEED_EXTERN int CeedBasisGetTensorContract(CeedBasis basis, CeedTensorContract *contract);
CEED_EXTERN int CeedBasisSetTensorContract(CeedBasis basis, CeedTensorContract contract);

CEED_EXTERN int CeedTensorContractCreate(Ceed ceed, CeedBasis basis, CeedTensorContract *contract);
CEED_EXTERN int CeedTensorContractApply(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *__restrict__ t,
                                        CeedTransposeMode t_mode, const CeedInt Add, const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v);
CEED_EXTERN int CeedTensorContractStridedApply(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt D, CeedInt J,
                                               const CeedScalar *__restrict__ t, CeedTransposeMode t_mode, const CeedInt add,
                                               const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v);
CEED_EXTERN int CeedTensorContractGetCeed(CeedTensorContract contract, Ceed *ceed);
CEED_EXTERN int CeedTensorContractGetData(CeedTensorContract contract, void *data);
CEED_EXTERN int CeedTensorContractSetData(CeedTensorContract contract, void *data);
CEED_EXTERN int CeedTensorContractReference(CeedTensorContract contract);
CEED_EXTERN int CeedTensorContractReferenceCopy(CeedTensorContract tensor, CeedTensorContract *tensor_copy);
CEED_EXTERN int CeedTensorContractDestroy(CeedTensorContract *contract);

CEED_EXTERN int CeedQFunctionRegister(const char *name, const char *source, CeedInt vec_length, CeedQFunctionUser f,
                                      int (*init)(Ceed, const char *, CeedQFunction));
CEED_EXTERN int CeedQFunctionSetFortranStatus(CeedQFunction qf, bool status);
CEED_EXTERN int CeedQFunctionGetVectorLength(CeedQFunction qf, CeedInt *vec_length);
CEED_EXTERN int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *num_input_fields, CeedInt *num_output_fields);
CEED_EXTERN int CeedQFunctionGetKernelName(CeedQFunction qf, char **kernel_name);
CEED_EXTERN int CeedQFunctionGetSourcePath(CeedQFunction qf, char **source_path);
CEED_EXTERN int CeedQFunctionLoadSourceToBuffer(CeedQFunction qf, char **source_buffer);
CEED_EXTERN int CeedQFunctionGetUserFunction(CeedQFunction qf, CeedQFunctionUser *f);
CEED_EXTERN int CeedQFunctionGetContext(CeedQFunction qf, CeedQFunctionContext *ctx);
CEED_EXTERN int CeedQFunctionGetContextData(CeedQFunction qf, CeedMemType mem_type, void *data);
CEED_EXTERN int CeedQFunctionRestoreContextData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionGetInnerContext(CeedQFunction qf, CeedQFunctionContext *ctx);
CEED_EXTERN int CeedQFunctionGetInnerContextData(CeedQFunction qf, CeedMemType mem_type, void *data);
CEED_EXTERN int CeedQFunctionRestoreInnerContextData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionIsIdentity(CeedQFunction qf, bool *is_identity);
CEED_EXTERN int CeedQFunctionIsContextWritable(CeedQFunction qf, bool *is_writable);
CEED_EXTERN int CeedQFunctionGetData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionSetData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionReference(CeedQFunction qf);
CEED_EXTERN int CeedQFunctionGetFlopsEstimate(CeedQFunction qf, CeedSize *flops);

CEED_EXTERN int CeedQFunctionContextGetCeed(CeedQFunctionContext ctx, Ceed *ceed);
CEED_EXTERN int CeedQFunctionContextHasValidData(CeedQFunctionContext ctx, bool *has_valid_data);
CEED_EXTERN int CeedQFunctionContextHasBorrowedDataOfType(CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type);
CEED_EXTERN int CeedQFunctionContextGetState(CeedQFunctionContext ctx, uint64_t *state);
CEED_EXTERN int CeedQFunctionContextGetBackendData(CeedQFunctionContext ctx, void *data);
CEED_EXTERN int CeedQFunctionContextSetBackendData(CeedQFunctionContext ctx, void *data);
CEED_EXTERN int CeedQFunctionContextGetFieldLabel(CeedQFunctionContext ctx, const char *field_name, CeedContextFieldLabel *field_label);
CEED_EXTERN int CeedQFunctionContextSetGeneric(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type,
                                               void *value);
CEED_EXTERN int CeedQFunctionContextGetGenericRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type,
                                                   size_t *num_values, void *value);
CEED_EXTERN int CeedQFunctionContextRestoreGenericRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, CeedContextFieldType field_type,
                                                       void *value);
CEED_EXTERN int CeedQFunctionContextSetDouble(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, double *values);
CEED_EXTERN int CeedQFunctionContextGetDoubleRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, size_t *num_values,
                                                  const double **values);
CEED_EXTERN int CeedQFunctionContextRestoreDoubleRead(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, const double **values);
CEED_EXTERN int CeedQFunctionContextSetInt32(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, int *values);
CEED_EXTERN int CeedQFunctionContextGetInt32Read(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, size_t *num_values, const int **values);
CEED_EXTERN int CeedQFunctionContextRestoreInt32Read(CeedQFunctionContext ctx, CeedContextFieldLabel field_label, const int **values);
CEED_EXTERN int CeedQFunctionContextGetDataDestroy(CeedQFunctionContext ctx, CeedMemType *f_mem_type, CeedQFunctionContextDataDestroyUser *f);
CEED_EXTERN int CeedQFunctionContextReference(CeedQFunctionContext ctx);

CEED_EXTERN int CeedQFunctionAssemblyDataCreate(Ceed ceed, CeedQFunctionAssemblyData *data);
CEED_EXTERN int CeedQFunctionAssemblyDataReference(CeedQFunctionAssemblyData data);
CEED_EXTERN int CeedQFunctionAssemblyDataSetReuse(CeedQFunctionAssemblyData data, bool reuse_assembly_data);
CEED_EXTERN int CeedQFunctionAssemblyDataSetUpdateNeeded(CeedQFunctionAssemblyData data, bool needs_data_update);
CEED_EXTERN int CeedQFunctionAssemblyDataIsUpdateNeeded(CeedQFunctionAssemblyData data, bool *is_update_needed);
CEED_EXTERN int CeedQFunctionAssemblyDataReferenceCopy(CeedQFunctionAssemblyData data, CeedQFunctionAssemblyData *data_copy);
CEED_EXTERN int CeedQFunctionAssemblyDataIsSetup(CeedQFunctionAssemblyData data, bool *is_setup);
CEED_EXTERN int CeedQFunctionAssemblyDataSetObjects(CeedQFunctionAssemblyData data, CeedVector vec, CeedElemRestriction rstr);
CEED_EXTERN int CeedQFunctionAssemblyDataGetObjects(CeedQFunctionAssemblyData data, CeedVector *vec, CeedElemRestriction *rstr);
CEED_EXTERN int CeedQFunctionAssemblyDataDestroy(CeedQFunctionAssemblyData *data);

CEED_EXTERN int CeedOperatorAssemblyDataCreate(Ceed ceed, CeedOperator op, CeedOperatorAssemblyData *data);
CEED_EXTERN int CeedOperatorAssemblyDataGetEvalModes(CeedOperatorAssemblyData data, CeedInt *num_active_bases, CeedInt **num_eval_modes_in,
                                                     const CeedEvalMode ***eval_modes_in, CeedSize ***eval_mode_offsets_in,
                                                     CeedInt **num_eval_modes_out, const CeedEvalMode ***eval_modes_out,
                                                     CeedSize ***eval_mode_offsets_out, CeedSize *num_output_components);
CEED_EXTERN int CeedOperatorAssemblyDataGetBases(CeedOperatorAssemblyData data, CeedInt *num_active_bases, CeedBasis **active_bases,
                                                 const CeedScalar ***assembled_bases_in, const CeedScalar ***assembled_bases_out);
CEED_EXTERN int CeedOperatorAssemblyDataGetElemRestrictions(CeedOperatorAssemblyData data, CeedInt *num_active_elem_rstrs,
                                                            CeedElemRestriction **active_elem_rstrs);
CEED_EXTERN int CeedOperatorAssemblyDataDestroy(CeedOperatorAssemblyData *data);

CEED_EXTERN int CeedOperatorGetOperatorAssemblyData(CeedOperator op, CeedOperatorAssemblyData *data);
CEED_EXTERN int CeedOperatorGetActiveBasis(CeedOperator op, CeedBasis *active_basis);
CEED_EXTERN int CeedOperatorGetActiveElemRestriction(CeedOperator op, CeedElemRestriction *active_rstr);
CEED_EXTERN int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *num_args);
CEED_EXTERN int CeedOperatorIsSetupDone(CeedOperator op, bool *is_setup_done);
CEED_EXTERN int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf);
CEED_EXTERN int CeedOperatorIsComposite(CeedOperator op, bool *is_composite);
CEED_EXTERN int CeedOperatorGetData(CeedOperator op, void *data);
CEED_EXTERN int CeedOperatorSetData(CeedOperator op, void *data);
CEED_EXTERN int CeedOperatorReference(CeedOperator op);
CEED_EXTERN int CeedOperatorGetFallback(CeedOperator op, CeedOperator *op_fallback);
CEED_EXTERN int CeedOperatorGetFallbackParent(CeedOperator op, CeedOperator *parent);
CEED_EXTERN int CeedOperatorGetFallbackParentCeed(CeedOperator op, Ceed *parent);
CEED_EXTERN int CeedOperatorSetSetupDone(CeedOperator op);

CEED_INTERN int CeedMatrixMatrixMultiply(Ceed ceed, const CeedScalar *mat_A, const CeedScalar *mat_B, CeedScalar *mat_C, CeedInt m, CeedInt n,
                                         CeedInt kk);
CEED_EXTERN int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau, CeedInt m, CeedInt n);
CEED_EXTERN int CeedHouseholderApplyQ(CeedScalar *mat_A, const CeedScalar *mat_Q, const CeedScalar *tau, CeedTransposeMode t_mode, CeedInt m,
                                      CeedInt n, CeedInt k, CeedInt row, CeedInt col);
CEED_EXTERN int CeedSymmetricSchurDecomposition(Ceed ceed, CeedScalar *mat, CeedScalar *lambda, CeedInt n);
CEED_EXTERN int CeedSimultaneousDiagonalization(Ceed ceed, CeedScalar *mat_A, CeedScalar *mat_B, CeedScalar *x, CeedScalar *lambda, CeedInt n);

#endif  // CEED_BACKEND_H
