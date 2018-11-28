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
#include <ceed-backend.h>
#include <string.h>

/// @file
/// Implementation of public CeedQFunction interfaces
///
/// @addtogroup CeedQFunction
/// @{

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed       A Ceed object where the CeedQFunction will be created
  @param vlength    Vector length.  Caller must ensure that number of quadrature
                    points is a multiple of vlength.
  @param f          Function pointer to evaluate action at quadrature points.
                    See below.
  @param focca      OCCA identifier "file.c:function_name" for definition of `f`
  @param[out] qf    Address of the variable where the newly created
                     CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  The arguments of the call-back 'function' are:

   1. [void *ctx][in/out] - user data, this is the 'ctx' pointer stored in
              the CeedQFunction, set by calling CeedQFunctionSetContext

   2. [CeedInt nq][in] - number of quadrature points to process

   3. [const CeedScalar *const *u][in] - input fields data at quadrature pts, listed in the order given by the user

   4. [CeedScalar *const *v][out] - output fields data at quadrature points, again listed in order given by the user

  @ref Basic
**/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
                                int (*f)(void*, CeedInt, const CeedScalar *const*, CeedScalar *const*),
                                const char *focca, CeedQFunction *qf) {
  int ierr;
  char *focca_copy;

  if (!ceed->QFunctionCreate) {
    Ceed delegate;
    ierr = CeedGetDelegate(ceed, &delegate); CeedChk(ierr);

    if (!delegate)
    return CeedError(ceed, 1, "Backend does not support QFunctionCreate");

    ierr = CeedQFunctionCreateInterior(delegate, vlength, f, focca, qf); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,qf); CeedChk(ierr);
  (*qf)->ceed = ceed;
  ceed->refcount++;
  (*qf)->refcount = 1;
  (*qf)->vlength = vlength;
  (*qf)->function = f;
  ierr = CeedCalloc(strlen(focca)+1, &focca_copy); CeedChk(ierr);
  strcpy(focca_copy, focca);
  (*qf)->focca = focca_copy;
  ierr = CeedCalloc(16, &(*qf)->inputfields); CeedChk(ierr);
  ierr = CeedCalloc(16, &(*qf)->outputfields); CeedChk(ierr);
  ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
  return 0;
}

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed       A Ceed object where the CeedQFunction will be created
  @param vlength    Vector length.  Caller must ensure that number of quadrature
                    points is a multiple of vlength.
  @param spec       The specification of the desired qFunction
  @param[out] qf    Address of the variable where the newly created
                     CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure
**/
int CeedQFunctionCreateInteriorFromGallery(Ceed ceed, CeedInt vlength, char* spec, CeedQFunction *qf) {
  int ierr;
  if (!strcmp(spec,"elliptic"))
  {

    ierr = CeedCalloc(1,qf); CeedChk(ierr);
    (*qf)->ceed = ceed;
    ceed->refcount++;
    (*qf)->refcount = 1;
    (*qf)->vlength = vlength;
    (*qf)->function = NULL;//TODO give a default implementation

    ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);

    CeedQFunctionAddInput (*qf, "ggeo", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "D", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "S", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "MM", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "lambda", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "q", 1, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(*qf, "Aq", 1, CEED_EVAL_NONE);

    return 0;
  }

  return 1;
}

/**
  @brief Set a CeedQFunction field, used by CeedQFunctionAddInput/Output

  @param f          CeedQFunctionField
  @param fieldname  Name of QFunction field
  @param ncomp      Number of components per quadrature node
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionFieldSet(CeedQFunctionField *f,const char *fieldname, 
                                 CeedInt ncomp, CeedEvalMode emode) {
  size_t len = strlen(fieldname);
  char *tmp;
  int ierr;
  ierr = CeedCalloc(1,f); CeedChk(ierr);

  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, fieldname, len+1);
  (*f)->fieldname = tmp;
  (*f)->ncomp = ncomp;
  (*f)->emode = emode;
  return 0;
}

/**
  @brief Add a CeedQFunction input

  @param qf         CeedQFunction
  @param fieldname  Name of QFunction field
  @param ncomp      Number of components per quadrature node
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionAddInput(CeedQFunction qf, const char *fieldname,
                          CeedInt ncomp, CeedEvalMode emode) {
  int ierr = CeedQFunctionFieldSet(&qf->inputfields[qf->numinputfields],
                                   fieldname, ncomp, emode);
  CeedChk(ierr);
  qf->numinputfields++;
  return 0;
}

/**
  @brief Add a CeedQFunction output

  @param qf         CeedQFunction
  @param fieldname  Name of QFunction field
  @param ncomp      Number of components per quadrature node
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionAddOutput(CeedQFunction qf, const char *fieldname,
                           CeedInt ncomp, CeedEvalMode emode) {
  if (emode == CEED_EVAL_WEIGHT)
    return CeedError(qf->ceed, 1,
                     "Cannot create qfunction output with CEED_EVAL_WEIGHT");
  int ierr = CeedQFunctionFieldSet(&qf->outputfields[qf->numoutputfields],
                                   fieldname, ncomp, emode);
  CeedChk(ierr);
  qf->numoutputfields++;
  return 0;
}

/**
  @brief Get the Ceed associated with a CeedQFunction

  @param qf              CeedQFunction
  @param[out] ceed       Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetCeed(CeedQFunction qf, Ceed *ceed) {
  *ceed = qf->ceed;
  return 0;
}

/**
  @brief Get the vector length of a CeedQFunction

  @param qf              CeedQFunction
  @param[out] veclength  Variable to store vector length

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetVectorLength(CeedQFunction qf, CeedInt *vlength) {
  *vlength = qf->vlength;
  return 0;
}

/**
  @brief Get the number of inputs and outputs to a CeedQFunction

  @param qf              CeedQFunction
  @param[out] numinput   Variable to store number of input fields
  @param[out] numoutput  Variable to store number of output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *numinput,
                            CeedInt *numoutput) {
  if (numinput) *numinput = qf->numinputfields;
  if (numoutput) *numoutput = qf->numoutputfields;
  return 0;
}

/**
  @brief Get the FOCCA string for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] focca      Variable to store focca string

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetFOCCA(CeedQFunction qf, char* *focca) {
  *focca = (char*) qf->focca;
  return 0;
}

/**
  @brief Get the User Function for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] f          Variable to store user function

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetUserFunction(CeedQFunction qf, int (**f)()) {
  *f = (int (*)())qf->function;
  return 0;
}

/**
  @brief Get global context size for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] ctxsize    Variable to store size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetContextSize(CeedQFunction qf, size_t *ctxsize) {
  *ctxsize = qf->ctxsize;
  return 0;
}

/**
  @brief Get global context for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] ctx        Variable to store context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetContext(CeedQFunction qf, void* *ctx) {
  *ctx = qf->ctx;
  return 0;
}

/**
  @brief Get backend data of a CeedQFunction

  @param qf              CeedQFunction
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetData(CeedQFunction qf, void* *data) {
  *data = qf->data;
  return 0;
}

/**
  @brief Set backend data of a CeedQFunction

  @param[out] qf         CeedQFunction
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionSetData(CeedQFunction qf, void* *data) {
  qf->data = *data;
  return 0;
}

/**
  @brief Set global context for a CeedQFunction

  @param qf       CeedQFunction
  @param ctx      Context data to set
  @param ctxsize  Size of context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionSetContext(CeedQFunction qf, void *ctx, size_t ctxsize) {
  qf->ctx = ctx;
  qf->ctxsize = ctxsize;
  return 0;
}

/**
  @brief Apply the action of a CeedQFunction

  @param qf      CeedQFunction
  @param Q       Number of quadrature points
  @param[in] u   Array of input data arrays
  @param[out] v  Array of output data arrays

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                       CeedVector* u, CeedVector* v) {
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

/**
  @brief Get the CeedQFunctionFields of a CeedQFunction

  @param qf                 CeedQFunction
  @param[out] inputfields   Variable to store inputfields
  @param[out] outputfields  Variable to store outputfields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetFields(CeedQFunction qf,
                           CeedQFunctionField* *inputfields,
                           CeedQFunctionField* *outputfields) {
  if (inputfields) *inputfields = qf->inputfields;
  if (outputfields) *outputfields = qf->outputfields;
  return 0;
}

/**
  @brief Get the name of a CeedQFunctionField

  @param qffield         CeedQFunctionField
  @param[out] fieldname  Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionFieldGetName(CeedQFunctionField qffield,
                              char* *fieldname) {
  *fieldname = (char *)qffield->fieldname;
  return 0;
}

/**
  @brief Get the number of components of a CeedQFunctionField

  @param qffield         CeedQFunctionField
  @param[out] numcomp    Variable to store the number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionFieldGetNumComponents(CeedQFunctionField qffield,
                                       CeedInt *numcomp) {
  *numcomp = qffield->ncomp;
  return 0;
}

/**
  @brief Get the CeedEvalMode of a CeedQFunctionField

  @param qffield         CeedQFunctionField
  @param[out] vec        Variable to store the number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionFieldGetEvalMode(CeedQFunctionField qffield,
                               CeedEvalMode *emode) {
  *emode = qffield->emode;
  return 0;
}

/**
  @brief Destroy a CeedQFunction

  @param qf CeedQFunction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionDestroy(CeedQFunction *qf) {
  int ierr;

  if (!*qf || --(*qf)->refcount > 0) return 0;
  // Backend destroy
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf); CeedChk(ierr);
  }
  // Free fields
  for (int i=0; i<(*qf)->numinputfields; i++) {
    ierr = CeedFree(&(*(*qf)->inputfields[i]).fieldname); CeedChk(ierr);
    ierr = CeedFree(&(*qf)->inputfields[i]); CeedChk(ierr);
  }
  for (int i=0; i<(*qf)->numoutputfields; i++) {
    ierr = CeedFree(&(*(*qf)->outputfields[i]).fieldname); CeedChk(ierr);
    ierr = CeedFree(&(*qf)->outputfields[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&(*qf)->inputfields); CeedChk(ierr);
  ierr = CeedFree(&(*qf)->outputfields); CeedChk(ierr);

  ierr = CeedFree(&(*qf)->focca); CeedChk(ierr);
  ierr = CeedDestroy(&(*qf)->ceed); CeedChk(ierr);
  ierr = CeedFree(qf); CeedChk(ierr);
  return 0;
}

/// @}
