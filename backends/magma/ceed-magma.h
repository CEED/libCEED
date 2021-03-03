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

// magma functions specific to ceed
#ifndef _ceed_magma_h
#define _ceed_magma_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <magma_v2.h>

typedef enum {
  MAGMA_KERNEL_DIM_GENERIC=101,
  MAGMA_KERNEL_DIM_SPECIFIC=102
} magma_kernel_mode_t;

typedef struct {
  magma_kernel_mode_t basis_kernel_mode;
  magma_device_t device;
  magma_queue_t queue;
} Ceed_Magma;

typedef struct {
  CeedScalar *dqref1d;
  CeedScalar *dinterp1d;
  CeedScalar *dgrad1d;
  CeedScalar *dqweight1d;
} CeedBasis_Magma;

typedef struct {
  CeedScalar *dqref;
  CeedScalar *dinterp;
  CeedScalar *dgrad;
  CeedScalar *dqweight;
} CeedBasisNonTensor_Magma;

typedef struct {
  CeedInt *offsets;
  CeedInt *doffsets;
  int  own_;
  int down_;            // cover a case where we own Device memory
} CeedElemRestriction_Magma;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setupdone;
} CeedQFunction_Magma;

#define USE_MAGMA_BATCH
#define USE_MAGMA_BATCH2
#define USE_MAGMA_BATCH3
#define USE_MAGMA_BATCH4

#ifdef __cplusplus
CEED_INTERN {
#endif

  magma_int_t magma_interp_1d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_interp_2d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_interp_3d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_interp_generic(magma_int_t P, magma_int_t Q,
                                   magma_int_t dim, magma_int_t ncomp,
                                   const double *dT, CeedTransposeMode tmode,
                                   const double *dU, magma_int_t u_elemstride,
                                   magma_int_t cstrdU,
                                   double *dV, magma_int_t v_elemstride,
                                   magma_int_t cstrdV,
                                   magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_interp(
    magma_int_t P, magma_int_t Q,
    magma_int_t dim, magma_int_t ncomp,
    const double *dT, CeedTransposeMode tmode,
    const double *dU, magma_int_t estrdU, magma_int_t cstrdU,
    double *dV, magma_int_t estrdV, magma_int_t cstrdV,
    magma_int_t nelem, magma_kernel_mode_t kernel_mode, magma_queue_t queue);

  magma_int_t magma_grad_1d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dTinterp, const CeedScalar *dTgrad, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_gradn_2d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_gradt_2d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_gradn_3d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_gradt_3d(
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_grad_generic(
    magma_int_t P, magma_int_t Q, magma_int_t dim, magma_int_t ncomp,
    const CeedScalar* dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_grad(
    magma_int_t P, magma_int_t Q, magma_int_t dim, magma_int_t ncomp,
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t u_elemstride, magma_int_t cstrdU, magma_int_t dstrdU,
    CeedScalar *dV, magma_int_t v_elemstride, magma_int_t cstrdV, magma_int_t dstrdV,
    magma_int_t nelem, magma_kernel_mode_t kernel_mode, magma_queue_t queue);

  magma_int_t magma_weight_1d(
    magma_int_t Q, const CeedScalar *dqweight1d,
    CeedScalar *dV, magma_int_t v_stride,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_weight_2d(
    magma_int_t Q, const CeedScalar *dqweight1d,
    CeedScalar *dV, magma_int_t v_stride,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_weight_3d(
    magma_int_t Q, const CeedScalar *dqweight1d,
    CeedScalar *dV, magma_int_t v_stride,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_weight_generic(
    magma_int_t Q, magma_int_t dim,
    const CeedScalar *dqweight1d,
    CeedScalar *dV, magma_int_t vstride,
    magma_int_t nelem, magma_queue_t queue);

  magma_int_t magma_weight(
    magma_int_t Q, magma_int_t dim,
    const CeedScalar *dqweight1d,
    CeedScalar *dV, magma_int_t v_stride,
    magma_int_t nelem, magma_kernel_mode_t kernel_mode, magma_queue_t queue);

  void magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t nelem,
                              magma_int_t Q,
                              double *dqweight, double *dv, magma_queue_t queue);

  void magma_readDofsOffset(const magma_int_t NCOMP,
                            const magma_int_t compstride,
                            const magma_int_t esize, const magma_int_t nelem,
                            magma_int_t *offsets, const double *du, double *dv,
                            magma_queue_t queue);

  void magma_readDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                             const magma_int_t nelem, magma_int_t *strides,
                             const double *du, double *dv,
                             magma_queue_t queue);

  void magma_writeDofsOffset(const magma_int_t NCOMP,
                             const magma_int_t compstride,
                             const magma_int_t esize, const magma_int_t nelem,
                             magma_int_t *offsets,const double *du, double *dv,
                             magma_queue_t queue);

  void magma_writeDofsStrided(const magma_int_t NCOMP, const magma_int_t esize,
                              const magma_int_t nelem, magma_int_t *strides,
                              const double *du, double *dv,
                              magma_queue_t queue);

  int magma_dgemm_nontensor(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha, const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,  double *dC, magma_int_t lddc,
    magma_queue_t queue );

  magma_int_t
  magma_isdevptr(const void *A);

  int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d,
                                    CeedInt Q1d,
                                    const CeedScalar *interp1d,
                                    const CeedScalar *grad1d,
                                    const CeedScalar *qref1d,
                                    const CeedScalar *qweight1d,
                                    CeedBasis basis);

  int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim,
                              CeedInt ndof, CeedInt nqpts,
                              const CeedScalar *interp,
                              const CeedScalar *grad,
                              const CeedScalar *qref,
                              const CeedScalar *qweight,
                              CeedBasis basis);

  int CeedElemRestrictionCreate_Magma(CeedMemType mtype,
                                      CeedCopyMode cmode,
                                      const CeedInt *offsets,
                                      CeedElemRestriction r);

  int CeedElemRestrictionCreateBlocked_Magma(const CeedMemType mtype,
      const CeedCopyMode cmode,
      const CeedInt *offsets,
      const CeedElemRestriction res);

  int CeedOperatorCreate_Magma(CeedOperator op);

  #ifdef __cplusplus
}
  #endif

// comment the line below to use the default magma_is_devptr function
#define magma_is_devptr magma_isdevptr

// if magma and cuda/ref are using the null stream, then ceed_magma_queue_sync
// should do nothing
#define ceed_magma_queue_sync(...)

// batch stride, override using -DMAGMA_BATCH_STRIDE=<desired-value>
#ifndef MAGMA_BATCH_STRIDE
#define MAGMA_BATCH_STRIDE (1000)
#endif

#endif  // _ceed_magma_h
