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
/// Public header for backend components of libCEED
#ifndef _ceed_backend_h
#define _ceed_backend_h

#include <ceed/ceed.h>
#include <stdbool.h>

#define CEED_INTERN CEED_EXTERN __attribute__((visibility ("hidden")))
#define CEED_UNUSED __attribute__((unused))

#define CEED_MAX_RESOURCE_LEN 1024
#define CEED_ALIGN 64
#define CEED_COMPOSITE_MAX 16
#define CEED_EPSILON 1E-16

/// CEED_DEBUG_COLOR default value, forward CeedDebug* declarations & macros
#ifndef CEED_DEBUG_COLOR
#define CEED_DEBUG_COLOR 0
#endif
CEED_EXTERN void CeedDebugImpl(const Ceed,const char *,...);
CEED_EXTERN void CeedDebugImpl256(const Ceed,const unsigned char,const char *,
                                  ...);
#define CeedDebug1(ceed,format, ...) CeedDebugImpl(ceed,format, ## __VA_ARGS__)
#define CeedDebug256(ceed,color, ...) CeedDebugImpl256(ceed,color, ## __VA_ARGS__)
#define CeedDebug(...) CeedDebug256(ceed,(unsigned char)CEED_DEBUG_COLOR, ## __VA_ARGS__)

/// Handle for object handling TensorContraction
/// @ingroup CeedBasis
typedef struct CeedTensorContract_private *CeedTensorContract;

/* In the next 3 functions, p has to be the address of a pointer type, i.e. p
   has to be a pointer to a pointer. */
CEED_INTERN int CeedMallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedCallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedReallocArray(size_t n, size_t unit, void *p);
CEED_INTERN int CeedFree(void *p);

#define CeedChk(ierr) do { int ierr_ = ierr; if (ierr_) return ierr_; } while (0)
#define CeedChkBackend(ierr) do { int ierr_ = ierr; if (ierr_) { if (ierr_ > CEED_ERROR_SUCCESS) return CEED_ERROR_BACKEND; else return ierr_; } } while (0)
/* Note that CeedMalloc and CeedCalloc will, generally, return pointers with
   different memory alignments: CeedMalloc returns pointers aligned at
   CEED_ALIGN bytes, while CeedCalloc uses the alignment of calloc. */
#define CeedMalloc(n, p) CeedMallocArray((n), sizeof(**(p)), p)
#define CeedCalloc(n, p) CeedCallocArray((n), sizeof(**(p)), p)
#define CeedRealloc(n, p) CeedReallocArray((n), sizeof(**(p)), p)

/// Handle for object describing CeedQFunction fields
/// @ingroup CeedQFunctionBackend
typedef struct CeedQFunctionField_private *CeedQFunctionField;
/// Handle for object describing CeedOperator fields
/// @ingroup CeedOperatorBackend
typedef struct CeedOperatorField_private *CeedOperatorField;

CEED_EXTERN int CeedRegister(const char *prefix,
                             int (*init)(const char *, Ceed),
                             unsigned int priority);

CEED_EXTERN int CeedIsDebug(Ceed ceed, bool *isDebug);
CEED_EXTERN int CeedGetParent(Ceed ceed, Ceed *parent);
CEED_EXTERN int CeedGetDelegate(Ceed ceed, Ceed *delegate);
CEED_EXTERN int CeedSetDelegate(Ceed ceed, Ceed delegate);
CEED_EXTERN int CeedGetObjectDelegate(Ceed ceed, Ceed *delegate,
                                      const char *objname);
CEED_EXTERN int CeedSetObjectDelegate(Ceed ceed, Ceed delegate,
                                      const char *objname);
CEED_EXTERN int CeedGetOperatorFallbackResource(Ceed ceed,
    const char **resource);
CEED_EXTERN int CeedSetOperatorFallbackResource(Ceed ceed,
    const char *resource);
CEED_EXTERN int CeedGetOperatorFallbackParentCeed(Ceed ceed, Ceed *parent);
CEED_EXTERN int CeedSetDeterministic(Ceed ceed, bool isDeterministic);
CEED_EXTERN int CeedSetBackendFunction(Ceed ceed,
                                       const char *type, void *object,
                                       const char *fname, int (*f)());
CEED_EXTERN int CeedGetData(Ceed ceed, void *data);
CEED_EXTERN int CeedSetData(Ceed ceed, void *data);

CEED_EXTERN int CeedVectorGetCeed(CeedVector vec, Ceed *ceed);
CEED_EXTERN int CeedVectorGetState(CeedVector vec, uint64_t *state);
CEED_EXTERN int CeedVectorAddReference(CeedVector vec);
CEED_EXTERN int CeedVectorGetData(CeedVector vec, void *data);
CEED_EXTERN int CeedVectorSetData(CeedVector vec, void *data);

CEED_EXTERN int CeedElemRestrictionGetCeed(CeedElemRestriction rstr,
    Ceed *ceed);
CEED_EXTERN int CeedElemRestrictionGetStrides(CeedElemRestriction rstr,
    CeedInt (*strides)[3]);
CEED_EXTERN int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr,
    CeedMemType mtype, const CeedInt **offsets);
CEED_EXTERN int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr,
    const CeedInt **offsets);
CEED_EXTERN int CeedElemRestrictionIsStrided(CeedElemRestriction rstr,
    bool *isstrided);
CEED_EXTERN int CeedElemRestrictionHasBackendStrides( CeedElemRestriction rstr,
    bool *hasbackendstrides);
CEED_EXTERN int CeedElemRestrictionGetELayout(CeedElemRestriction rstr,
    CeedInt (*layout)[3]);
CEED_EXTERN int CeedElemRestrictionSetELayout(CeedElemRestriction rstr,
    CeedInt layout[3]);
CEED_EXTERN int CeedElemRestrictionGetData(CeedElemRestriction rstr,
    void *data);
CEED_EXTERN int CeedElemRestrictionSetData(CeedElemRestriction rstr,
    void *data);

CEED_EXTERN int CeedBasisGetCollocatedGrad(CeedBasis basis,
    CeedScalar *colograd1d);
CEED_EXTERN int CeedHouseholderApplyQ(CeedScalar *A, const CeedScalar *Q,
                                      const CeedScalar *tau, CeedTransposeMode tmode, CeedInt m, CeedInt n,
                                      CeedInt k, CeedInt row, CeedInt col);
CEED_EXTERN int CeedBasisGetCeed(CeedBasis basis, Ceed *ceed);
CEED_EXTERN int CeedBasisIsTensor(CeedBasis basis, bool *istensor);
CEED_EXTERN int CeedBasisGetData(CeedBasis basis, void *data);
CEED_EXTERN int CeedBasisSetData(CeedBasis basis, void *data);

CEED_EXTERN int CeedBasisGetTopologyDimension(CeedElemTopology topo,
    CeedInt *dim);

CEED_EXTERN int CeedBasisGetTensorContract(CeedBasis basis,
    CeedTensorContract *contract);
CEED_EXTERN int CeedBasisSetTensorContract(CeedBasis basis,
    CeedTensorContract *contract);
CEED_EXTERN int CeedTensorContractCreate(Ceed ceed, CeedBasis basis,
    CeedTensorContract *contract);
CEED_EXTERN int CeedTensorContractApply(CeedTensorContract contract, CeedInt A,
                                        CeedInt B, CeedInt C, CeedInt J,
                                        const CeedScalar *__restrict__ t,
                                        CeedTransposeMode tmode,
                                        const CeedInt Add,
                                        const CeedScalar *__restrict__ u,
                                        CeedScalar *__restrict__ v);
CEED_EXTERN int CeedTensorContractGetCeed(CeedTensorContract contract,
    Ceed *ceed);
CEED_EXTERN int CeedTensorContractGetData(CeedTensorContract contract,
    void *data);
CEED_EXTERN int CeedTensorContractSetData(CeedTensorContract contract,
    void *data);
CEED_EXTERN int CeedTensorContractDestroy(CeedTensorContract *contract);

CEED_EXTERN int CeedQFunctionRegister(const char *, const char *, CeedInt,
                                      CeedQFunctionUser, int (*init)(Ceed, const char *, CeedQFunction));
CEED_EXTERN int CeedQFunctionSetFortranStatus(CeedQFunction qf, bool status);
CEED_EXTERN int CeedQFunctionGetCeed(CeedQFunction qf, Ceed *ceed);
CEED_EXTERN int CeedQFunctionGetVectorLength(CeedQFunction qf,
    CeedInt *vlength);
CEED_EXTERN int CeedQFunctionGetNumArgs(CeedQFunction qf,
                                        CeedInt *numinputfields,
                                        CeedInt *numoutputfields);
CEED_EXTERN int CeedQFunctionGetSourcePath(CeedQFunction qf, char **source);
CEED_EXTERN int CeedQFunctionGetUserFunction(CeedQFunction qf,
    CeedQFunctionUser *f);
CEED_EXTERN int CeedQFunctionGetContext(CeedQFunction qf,
                                        CeedQFunctionContext *ctx);
CEED_EXTERN int CeedQFunctionGetInnerContext(CeedQFunction qf,
    CeedQFunctionContext *ctx);
CEED_EXTERN int CeedQFunctionIsIdentity(CeedQFunction qf, bool *isidentity);
CEED_EXTERN int CeedQFunctionGetData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionSetData(CeedQFunction qf, void *data);
CEED_EXTERN int CeedQFunctionGetFields(CeedQFunction qf,
                                       CeedQFunctionField **inputfields,
                                       CeedQFunctionField **outputfields);
CEED_EXTERN int CeedQFunctionFieldGetName(CeedQFunctionField qffield,
    char **fieldname);
CEED_EXTERN int CeedQFunctionFieldGetSize(CeedQFunctionField qffield,
    CeedInt *size);
CEED_EXTERN int CeedQFunctionFieldGetEvalMode(CeedQFunctionField qffield,
    CeedEvalMode *emode);

CEED_EXTERN int CeedQFunctionContextGetCeed(CeedQFunctionContext cxt,
    Ceed *ceed);
CEED_EXTERN int CeedQFunctionContextGetState(CeedQFunctionContext ctx,
    uint64_t *state);
CEED_EXTERN int CeedQFunctionContextGetContextSize(CeedQFunctionContext ctx,
    size_t *ctxsize);
CEED_EXTERN int CeedQFunctionContextGetBackendData(CeedQFunctionContext ctx,
    void *data);
CEED_EXTERN int CeedQFunctionContextSetBackendData(CeedQFunctionContext ctx,
    void *data);

CEED_EXTERN int CeedOperatorGetCeed(CeedOperator op, Ceed *ceed);
CEED_EXTERN int CeedOperatorGetNumElements(CeedOperator op, CeedInt *numelem);
CEED_EXTERN int CeedOperatorGetNumQuadraturePoints(CeedOperator op,
    CeedInt *numqpts);
CEED_EXTERN int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *numargs);
CEED_EXTERN int CeedOperatorIsSetupDone(CeedOperator op, bool *issetupdone);
CEED_EXTERN int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf);
CEED_EXTERN int CeedOperatorIsComposite(CeedOperator op, bool *iscomposite);
CEED_EXTERN int CeedOperatorGetNumSub(CeedOperator op, CeedInt *numsub);
CEED_EXTERN int CeedOperatorGetSubList(CeedOperator op,
                                       CeedOperator **suboperators);
CEED_EXTERN int CeedOperatorGetData(CeedOperator op, void *data);
CEED_EXTERN int CeedOperatorSetData(CeedOperator op, void *data);
CEED_EXTERN int CeedOperatorSetSetupDone(CeedOperator op);

CEED_EXTERN int CeedOperatorGetFields(CeedOperator op,
                                      CeedOperatorField **inputfields,
                                      CeedOperatorField **outputfields);
CEED_EXTERN int CeedOperatorFieldGetElemRestriction(CeedOperatorField opfield,
    CeedElemRestriction *rstr);
CEED_EXTERN int CeedOperatorFieldGetBasis(CeedOperatorField opfield,
    CeedBasis *basis);
CEED_EXTERN int CeedOperatorFieldGetVector(CeedOperatorField opfield,
    CeedVector *vec);

CEED_INTERN int CeedMatrixMultiply(Ceed ceed, const CeedScalar *matA,
                                   const CeedScalar *matB, CeedScalar *matC,
                                   CeedInt m, CeedInt n, CeedInt kk);

#endif
