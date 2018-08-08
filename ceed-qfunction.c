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
#include <string.h>

/**
  @file
  Implementation of public CeedQFunction interfaces

  @defgroup CeedQFunction CeedQFunction: independent operations at quadrature points
  @{
 */

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed       Ceed library context
  @param vlength    Vector length.  Caller must ensure that number of quadrature
                    points is a multiple of vlength.
  @param f          Function pointer to evaluate action at quadrature points.
                    See below.
  @param focca      OCCA identifier "file.c:function_name" for definition of `f`
  @param qf         constructed QFunction
  @return 0 on success, otherwise failure

  The arguments of the call-back 'function' are:

   1. [void *ctx][in/out] - user data, this is the 'ctx' pointer stored in
              the CeedQFunction, set by calling CeedQFunctionSetContext

   2. [CeedInt nq][in] - number of quadrature points to process

   3. [const CeedScalar *const *u][in] - input fields data at quadrature pts, listed in the order given by the user

   4. [CeedScalar *const *v][out] - output fields data at quadrature points, again listed in order given by the user

*/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
                                CeedQFunctionCallback f, const char *fcuda,
                                const char *focca, CeedQFunction *qf) {
  int ierr;
  char *focca_copy;

  if (!ceed->QFunctionCreate)
    return CeedError(ceed, 1, "Backend does not support QFunctionCreate");
  ierr = CeedCalloc(1,qf); CeedChk(ierr);
  (*qf)->ceed = ceed;
  ceed->refcount++;
  (*qf)->refcount = 1;
  (*qf)->vlength = vlength;
  (*qf)->function = f;
  (*qf)->fcuda = fcuda;
  ierr = CeedCalloc(strlen(focca)+1, &focca_copy); CeedChk(ierr);
  strcpy(focca_copy, focca);
  (*qf)->focca = focca_copy;
  ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
  return 0;
}

static int CeedQFunctionFieldSet(struct CeedQFunctionField *f,
                                 const char *fieldname, CeedInt ncomp,
                                 CeedEvalMode emode) {
  size_t len = strlen(fieldname);
  char *tmp;
  int ierr =  CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, fieldname, len+1);
  f->fieldname = tmp;
  f->ncomp = ncomp;
  f->emode = emode;
  return 0;
}

int CeedQFunctionAddInput(CeedQFunction qf, const char *fieldname,
                          CeedInt ncomp, CeedEvalMode emode) {
  int ierr = CeedQFunctionFieldSet(&qf->inputfields[qf->numinputfields++],
                                   fieldname, ncomp, emode); CeedChk(ierr);
  return 0;
}

int CeedQFunctionAddOutput(CeedQFunction qf, const char *fieldname,
                           CeedInt ncomp, CeedEvalMode emode) {
  if (emode == CEED_EVAL_WEIGHT)
    return CeedError(qf->ceed, 1,
                     "Cannot create qfunction output with CEED_EVAL_WEIGHT");
  int ierr = CeedQFunctionFieldSet(&qf->outputfields[qf->numoutputfields++],
                                   fieldname, ncomp, emode); CeedChk(ierr);
  return 0;
}

int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *numinput,
                            CeedInt *numoutput) {
  CeedInt nin = 0, nout = 0;
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    CeedEvalMode emode = qf->inputfields[i].emode;
    if (emode == CEED_EVAL_NONE) nin++;  // Colocated field is input directly
    if (emode & CEED_EVAL_INTERP) nin++; // Interpolate to quadrature points
    if (emode & CEED_EVAL_GRAD) nin++;   // Gradients at quadrature points
  }
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    CeedEvalMode emode = qf->outputfields[i].emode;
    if (emode == CEED_EVAL_NONE) nout++;
    if (emode & CEED_EVAL_INTERP) nout++;
    if (emode & CEED_EVAL_GRAD) nout++;
  }
  if (numinput) *numinput = nin;
  if (numoutput) *numoutput = nout;
  return 0;
}

/**
  Set global context for a quadrature function
 */
int CeedQFunctionSetContext(CeedQFunction qf, void *ctx, size_t ctxsize) {
  qf->ctx = ctx;
  qf->ctxsize = ctxsize;
  return 0;
}

/** Apply the action of a CeedQFunction
 */
int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                       const CeedScalar *const *u,
                       CeedScalar *const *v) {
  int ierr;
  if (!qf->Apply)
    return CeedError(qf->ceed, 1, "Backend does not support QFunctionApply");
  if (Q % qf->vlength)
    return CeedError(qf->ceed, 2,
                     "Number of quadrature points %d must be a multiple of %d",
                     Q, qf->vlength);
  ierr = qf->Apply(qf, Q, u, v); CeedChk(ierr);
  return 0;
}

/** Destroy a CeedQFunction
 */
int CeedQFunctionDestroy(CeedQFunction *qf) {
  int ierr;

  if (!*qf || --(*qf)->refcount > 0) return 0;
  // Free field names
  for (int i=0; i<(*qf)->numinputfields; i++) {
    ierr = CeedFree(&(*qf)->inputfields[i].fieldname); CeedChk(ierr);
  }
  for (int i=0; i<(*qf)->numoutputfields; i++) {
    ierr = CeedFree(&(*qf)->outputfields[i].fieldname); CeedChk(ierr);
  }
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf); CeedChk(ierr);
  }
  ierr = CeedFree(&(*qf)->focca); CeedChk(ierr);
  ierr = CeedDestroy(&(*qf)->ceed); CeedChk(ierr);
  ierr = CeedFree(qf); CeedChk(ierr);
  return 0;
}
