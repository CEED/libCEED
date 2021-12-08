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

#ifndef _ceed_ref_h
#define _ceed_ref_h

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  CeedScalar *collograd1d;
  bool collo_interp;
} CeedBasis_Ref;

typedef struct {
  CeedScalar *array;
  CeedScalar *array_allocated;
} CeedVector_Ref;

typedef struct {
  const CeedInt *offsets;
  CeedInt *offsets_allocated;
  int (*Apply)(CeedElemRestriction, const CeedInt, const CeedInt,
               const CeedInt, CeedInt, CeedInt, CeedTransposeMode, CeedVector,
               CeedVector, CeedRequest *);
} CeedElemRestriction_Ref;

typedef struct {
  const CeedScalar **inputs;
  CeedScalar **outputs;
  bool setup_done;
} CeedQFunction_Ref;

typedef struct {
  void *data;
  void *data_allocated;
} CeedQFunctionContext_Ref;

typedef struct {
  bool is_identity_qf, is_identity_restr_op;
  CeedVector *e_vecs;      /* All E-vectors, inputs followed by outputs */
  uint64_t *input_state;   /* State counter of inputs */
  CeedVector *e_vecs_in;   /* Input E-vectors needed to apply operator */
  CeedVector *e_vecs_out;  /* Output E-vectors needed to apply operator */
  CeedVector *q_vecs_in;   /* Input Q-vectors needed to apply operator */
  CeedVector *q_vecs_out;  /* Output Q-vectors needed to apply operator */
  CeedInt    num_e_vecs_in;
  CeedInt    num_e_vecs_out;
  CeedInt    qf_num_active_in, qf_num_active_out;
  CeedVector *qf_active_in;
} CeedOperator_Ref;

CEED_INTERN int CeedVectorCreate_Ref(CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_Ref(CeedMemType mem_type,
    CeedCopyMode copy_mode, const CeedInt *indices, CeedElemRestriction r);

CEED_INTERN int CeedBasisCreateTensorH1_Ref(CeedInt dim, CeedInt P_1d,
    CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
    const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis basis);

CEED_INTERN int CeedBasisCreateH1_Ref(CeedElemTopology topo,
                                      CeedInt dim, CeedInt num_dof, CeedInt num_qpts,
                                      const CeedScalar *interp,
                                      const CeedScalar *grad,
                                      const CeedScalar *q_ref,
                                      const CeedScalar *q_weight,
                                      CeedBasis basis);

CEED_INTERN int CeedTensorContractCreate_Ref(CeedBasis basis,
    CeedTensorContract contract);

CEED_INTERN int CeedQFunctionCreate_Ref(CeedQFunction qf);

CEED_INTERN int CeedQFunctionContextCreate_Ref(CeedQFunctionContext ctx);

CEED_INTERN int CeedOperatorCreate_Ref(CeedOperator op);

#endif // _ceed_ref_h
