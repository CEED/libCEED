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

#include <ceed-impl.h>

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed       Ceed library context
  @param vlength    Vector length.  Caller must ensure that number of quadrature
                    points is a multiple of vlength.
  @param nfields    Number of fields/components in input and output arrays
  @param qdatasize  Size in bytes of quadrature data at each quadrature point.
  @param inmode     Bitwise OR of evaluation modes for input fields
  @param outmode    Bitwise OR of evaluation modes for output fields
  @param f          Function pointer to evaluate action at quadrature points.
                    See below.
  @param focca      OCCA identifier "file.c:function_name" for definition of `f`
  @param qf         constructed QFunction
  @return 0 on success, otherwise failure

  The arguments of the call-back 'function' are:

   1. [void *ctx][in/out] - user data, this is the 'ctx' pointer stored in
              the CeedQFunction, set by calling CeedQFunctionSetContext

   2. [void *qdata][in/out] - quadrature points data corresponding to the
              batch-of-points being processed in this call; the quadrature
              point index has a stride of 1

   3. [CeedInt nq][in] - number of quadrature points to process

   4. [const CeedScalar *const *u][in] - input fields data at quadrature pts:
       u[0] - CEED_EVAL_INTERP data: field values at quadrature points in
              reference space; the quadrature point index has a stride of 1,
              vector component index has a stride of nq (see argument 3) and
              values for multiple fields are consequitive in memory (strides
              will generally vary with the number of components in a field);
              fields that do not specify CEED_EVAL_INTERP mode, use no memory.
       u[1] - CEED_EVAL_GRAD data: field gradients at quadrature points in
              reference space; the quadrature point index has a stride of 1,
              the derivative direction index has a stride of nq (see argument
              3), vector component index has a stride of (rdim x nq) where
              rdim is the dimension of the reference element, and values for
              multiple fields are consequitive in memory (strides will
              generally vary with the number of components in a field);
              fields that do not specify CEED_EVAL_GRAD mode, use no memory.
       u[2] - CEED_EVAL_DIV data: field divergences ... <same as above>?
       u[3] - CEED_EVAL_CURL data: field curl ... <same as above>?

   5. [CeedScalar *const *v][out] - output fields data at quadrature points:
       v[0], v[1], ..., v[3] - use similar layouts as u[] but use the output
              CeedEvalMode.

*/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength, CeedInt nfields,
                                size_t qdatasize, CeedEvalMode inmode,
                                CeedEvalMode outmode,
                                int (*f)(void*, void*, CeedInt, const CeedScalar *const*, CeedScalar *const*),
                                const char *focca, CeedQFunction *qf) {
  int ierr;

  if (!ceed->QFunctionCreate)
    return CeedError(ceed, 1, "Backend does not support QFunctionCreate");
  ierr = CeedCalloc(1,qf); CeedChk(ierr);
  (*qf)->ceed = ceed;
  (*qf)->vlength = vlength;
  (*qf)->nfields = nfields;
  (*qf)->qdatasize = qdatasize;
  (*qf)->inmode = inmode;
  (*qf)->outmode = outmode;
  (*qf)->function = f;
  (*qf)->focca = focca;
  ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
  return 0;
}

int CeedQFunctionSetContext(CeedQFunction qf, void *ctx, size_t ctxsize) {
  qf->ctx = ctx;
  qf->ctxsize = ctxsize;
  return 0;
}

int CeedQFunctionApply(CeedQFunction qf, void *qdata, CeedInt Q,
                       const CeedScalar *const *u,
                       CeedScalar *const *v) {
  int ierr;
  if (!qf->Apply)
    return CeedError(qf->ceed, 1, "Backend does not support QFunctionApply");
  if (Q % qf->vlength)
    return CeedError(qf->ceed, 2,
                     "Number of quadrature points %d must be a multiple of %d",
                     Q, qf->vlength);
  ierr = qf->Apply(qf, qdata, Q, u, v); CeedChk(ierr);
  return 0;
}

int CeedQFunctionDestroy(CeedQFunction *qf) {
  int ierr;

  if (!*qf) return 0;
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf); CeedChk(ierr);
  }
  ierr = CeedFree(qf); CeedChk(ierr);
  return 0;
}
