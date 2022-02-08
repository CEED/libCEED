// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

#ifndef _ceed_cuda_h
#define _ceed_cuda_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda.h>
#include "../cuda/ceed-cuda-common.h"

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_borrowed;
  CeedScalar *h_array_owned;
  CeedScalar *d_array;
  CeedScalar *d_array_borrowed;
  CeedScalar *d_array_owned;
} CeedVector_Cuda;

typedef struct {
  CUmodule module;
  CUfunction StridedTranspose;
  CUfunction StridedNoTranspose;
  CUfunction OffsetTranspose;
  CUfunction OffsetNoTranspose;
  CeedInt num_nodes;
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  CeedInt *d_ind;
  CeedInt *d_ind_allocated;
  CeedInt *d_t_offsets;
  CeedInt *d_t_indices;
  CeedInt *d_l_vec_indices;
} CeedElemRestriction_Cuda;

typedef struct {
  CUmodule module;
  CUfunction Interp;
  CUfunction Grad;
  CUfunction Weight;
  CeedScalar *d_interp_1d;
  CeedScalar *d_grad_1d;
  CeedScalar *d_q_weight_1d;
} CeedBasis_Cuda;

typedef struct {
  CUmodule module;
  CUfunction Interp;
  CUfunction Grad;
  CUfunction Weight;
  CeedScalar *d_interp;
  CeedScalar *d_grad;
  CeedScalar *d_q_weight;
} CeedBasisNonTensor_Cuda;

// We use a struct to avoid having to memCpy the array of pointers
// __global__ copies by value the struct.
typedef struct {
  const CeedScalar *inputs[CEED_FIELD_MAX];
  CeedScalar *outputs[CEED_FIELD_MAX];
} Fields_Cuda;

typedef struct {
  CUmodule module;
  char *qfunction_name;
  char *qfunction_source;
  CUfunction QFunction;
  Fields_Cuda fields;
  void *d_c;
} CeedQFunction_Cuda;

typedef struct {
  void *h_data;
  void *h_data_borrowed;
  void *h_data_owned;
  void *d_data;
  void *d_data_borrowed;
  void *d_data_owned;
} CeedQFunctionContext_Cuda;

typedef struct {
  CUmodule module;
  CUfunction linearDiagonal;
  CUfunction linearPointBlock;
  CeedBasis basisin, basisout;
  CeedElemRestriction diagrstr, pbdiagrstr;
  CeedVector elemdiag, pbelemdiag;
  CeedInt numemodein, numemodeout, nnodes;
  CeedEvalMode *h_emodein, *h_emodeout;
  CeedEvalMode *d_emodein, *d_emodeout;
  CeedScalar *d_identity, *d_interpin, *d_interpout, *d_gradin, *d_gradout;
} CeedOperatorDiag_Cuda;

typedef struct {
  CeedVector *evecs;   // E-vectors, inputs followed by outputs
  CeedVector *qvecsin;    // Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   // Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    qfnumactivein, qfnumactiveout;
  CeedVector *qfactivein;
  CeedOperatorDiag_Cuda *diag;
} CeedOperator_Cuda;

CEED_INTERN int CeedCudaGetCublasHandle(Ceed ceed, cublasHandle_t *handle);

CEED_INTERN int CeedVectorCreate_Cuda(CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type,
    CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedElemRestrictionCreateBlocked_Cuda(const CeedMemType
    mem_type,
    const CeedCopyMode copy_mode, const CeedInt *indices,
    const CeedElemRestriction res);

CEED_INTERN int CeedBasisApplyElems_Cuda(CeedBasis basis,
    const CeedInt num_elem,
    CeedTransposeMode t_mode, CeedEvalMode eval_mode, const CeedVector u,
    CeedVector v);

CEED_INTERN int CeedQFunctionApplyElems_Cuda(CeedQFunction qf, const CeedInt Q,
    const CeedVector *const u, const CeedVector *v);

CEED_INTERN int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d,
    CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
    const CeedScalar *qref_1d, const CeedScalar *qweight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Cuda(CeedElemTopology, CeedInt, CeedInt,
                                       CeedInt, const CeedScalar *,
                                       const CeedScalar *, const CeedScalar *,
                                       const CeedScalar *, CeedBasis);

CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);

CEED_INTERN int CeedCompositeOperatorCreate_Cuda(CeedOperator op);
#endif
