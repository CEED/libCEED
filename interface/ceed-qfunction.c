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
#include <limits.h>

/// @cond DOXYGEN_SKIP
static struct CeedQFunction_private ceed_qfunction_none;
/// @endcond

/// @cond DOXYGEN_SKIP
static struct {
  char name[CEED_MAX_RESOURCE_LEN];
  char source[CEED_MAX_RESOURCE_LEN];
  CeedInt vlength;
  CeedQFunctionUser f;
  int (*init)(Ceed ceed, const char *name, CeedQFunction qf);
} qfunctions[1024];
static size_t num_qfunctions;
/// @endcond

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
                      See \ref CeedQFunctionUser.
  @param source     Absolute path to source of QFunction,
                      "\abs_path\file.h:function_name"
  @param[out] qf    Address of the variable where the newly created
                      CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  See \ref CeedQFunctionUser for details on the call-back function @a f's
    arguments.

  @ref Basic
**/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength, CeedQFunctionUser f,
                                const char *source, CeedQFunction *qf) {
  int ierr;
  char *source_copy;

  if (!ceed->QFunctionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "QFunction"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support QFunctionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedQFunctionCreateInterior(delegate, vlength, f, source, qf);
    CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, qf); CeedChk(ierr);
  (*qf)->ceed = ceed;
  ceed->refcount++;
  (*qf)->refcount = 1;
  (*qf)->vlength = vlength;
  (*qf)->identity = 0;
  (*qf)->function = f;
  size_t slen = strlen(source) + 1;
  ierr = CeedMalloc(slen, &source_copy); CeedChk(ierr);
  memcpy(source_copy, source, slen);
  (*qf)->sourcepath = source_copy;
  ierr = CeedCalloc(16, &(*qf)->inputfields); CeedChk(ierr);
  ierr = CeedCalloc(16, &(*qf)->outputfields); CeedChk(ierr);
  ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
  return 0;
}

/**
  @brief Register a gallery QFunction

  @param name    Name for this backend to respond to
  @param source  Absolute path to source of QFunction,
                   "\path\CEED_DIR\gallery\folder\file.h:function_name"
  @param vlength Vector length.  Caller must ensure that number of quadrature
                   points is a multiple of vlength.
  @param f       Function pointer to evaluate action at quadrature points.
                   See \ref CeedQFunctionUser.
  @param init    Initialization function called by CeedQFunctionInit() when the
                   QFunction is selected.

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionRegister(const char *name, const char *source,
                          CeedInt vlength, CeedQFunctionUser f,
                          int (*init)(Ceed, const char *, CeedQFunction)) {
  if (num_qfunctions >= sizeof(qfunctions) / sizeof(qfunctions[0]))
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "Too many gallery QFunctions");
  // LCOV_EXCL_STOP

  strncpy(qfunctions[num_qfunctions].name, name, CEED_MAX_RESOURCE_LEN);
  qfunctions[num_qfunctions].name[CEED_MAX_RESOURCE_LEN-1] = 0;
  strncpy(qfunctions[num_qfunctions].source, source, CEED_MAX_RESOURCE_LEN);
  qfunctions[num_qfunctions].source[CEED_MAX_RESOURCE_LEN-1] = 0;
  qfunctions[num_qfunctions].vlength = vlength;
  qfunctions[num_qfunctions].f = f;
  qfunctions[num_qfunctions].init = init;
  num_qfunctions++;
  return 0;
}

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms by name.

  @param ceed       A Ceed object where the CeedQFunction will be created
  @param name       Name of QFunction to use from gallery
  @param[out] qf    Address of the variable where the newly created
                      CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionCreateInteriorByName(Ceed ceed,  const char *name,
                                      CeedQFunction *qf) {
  int ierr;
  size_t matchlen = 0, matchidx = UINT_MAX;
  char *name_copy;

  // Find matching backend
  if (!name) return CeedError(NULL, 1, "No QFunction name provided");
  for (size_t i=0; i<num_qfunctions; i++) {
    size_t n;
    const char *currname = qfunctions[i].name;
    for (n = 0; currname[n] && currname[n] == name[n]; n++) {}
    if (n > matchlen) {
      matchlen = n;
      matchidx = i;
    }
  }
  if (!matchlen)
    // LCOV_EXCL_START
    return CeedError(NULL, 1, "No suitable gallery QFunction");
  // LCOV_EXCL_STOP

  // Create QFunction
  ierr = CeedQFunctionCreateInterior(ceed, qfunctions[matchidx].vlength,
                                     qfunctions[matchidx].f,
                                     qfunctions[matchidx].source, qf);
  CeedChk(ierr);

  // QFunction specific setup
  ierr = qfunctions[matchidx].init(ceed, name, *qf); CeedChk(ierr);

  // Copy name
  size_t slen = strlen(name) + 1;
  ierr = CeedMalloc(slen, &name_copy); CeedChk(ierr);
  memcpy(name_copy, name, slen);
  (*qf)->qfname = name_copy;

  return 0;
}

/**
  @brief Create an identity CeedQFunction. Inputs are written into outputs in
           the order given. This is useful for CeedOperators that can be
           represented with only the action of a CeedRestriction and CeedBasis,
           such as restriction and prolongation operators for p-multigrid.
           Backends may optimize CeedOperators with this CeedQFunction to avoid
           the copy of input data to output fields by using the same memory
           location for both.

  @param ceed        A Ceed object where the CeedQFunction will be created
  @param[in] size    Size of the qfunction fields
  @param[in] inmode  CeedEvalMode for input to CeedQFunction
  @param[in] outmode CeedEvalMode for output to CeedQFunction
  @param[out] qf     Address of the variable where the newly created
                       CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionCreateIdentity(Ceed ceed, CeedInt size, CeedEvalMode inmode,
                                CeedEvalMode outmode, CeedQFunction *qf) {
  int ierr;

  if (inmode == CEED_EVAL_NONE && outmode == CEED_EVAL_NONE)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "CEED_EVAL_NONE for a both the input and "
                     "output does not make sense with an identity QFunction");
  // LCOV_EXCL_STOP

  ierr = CeedQFunctionCreateInteriorByName(ceed, "Identity", qf); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(*qf, "input", 1, inmode); CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(*qf, "output", 1, outmode); CeedChk(ierr);

  (*qf)->identity = 1;
  if (size > 1) {
    CeedInt *ctx;
    ierr = CeedCalloc(1, &ctx); CeedChk(ierr);
    ctx[0] = size;
    ierr = CeedQFunctionSetContext(*qf, ctx, sizeof(ctx)); CeedChk(ierr);
    (*qf)->inputfields[0]->size = size;
    (*qf)->outputfields[0]->size = size;
  }

  return 0;
}

/**
  @brief Set a CeedQFunction field, used by CeedQFunctionAddInput/Output

  @param f          CeedQFunctionField
  @param fieldname  Name of QFunction field
  @param size       Size of QFunction field, ncomp * (dim for CEED_EVAL_GRAD or
                      1 for CEED_EVAL_NONE, CEED_EVAL_INTERP, and CEED_EVAL_WEIGHT)
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients,
                      \ref CEED_EVAL_WEIGHT to use quadrature weights.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionFieldSet(CeedQFunctionField *f,const char *fieldname,
                                 CeedInt size, CeedEvalMode emode) {
  size_t len = strlen(fieldname);
  char *tmp;
  int ierr;
  ierr = CeedCalloc(1,f); CeedChk(ierr);

  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, fieldname, len+1);
  (*f)->fieldname = tmp;
  (*f)->size = size;
  (*f)->emode = emode;
  return 0;
}

/**
  @brief Add a CeedQFunction input

  @param qf         CeedQFunction
  @param fieldname  Name of QFunction field
  @param size       Size of QFunction field, ncomp * (dim for CEED_EVAL_GRAD or
                      1 for CEED_EVAL_NONE and CEED_EVAL_INTERP)
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionAddInput(CeedQFunction qf, const char *fieldname, CeedInt size,
                          CeedEvalMode emode) {
  int ierr = CeedQFunctionFieldSet(&qf->inputfields[qf->numinputfields],
                                   fieldname, size, emode);
  CeedChk(ierr);
  qf->numinputfields++;
  return 0;
}

/**
  @brief Add a CeedQFunction output

  @param qf         CeedQFunction
  @param fieldname  Name of QFunction field
  @param size       Size of QFunction field, ncomp * (dim for CEED_EVAL_GRAD or
                      1 for CEED_EVAL_NONE and CEED_EVAL_INTERP)
  @param emode      \ref CEED_EVAL_NONE to use values directly,
                      \ref CEED_EVAL_INTERP to use interpolated values,
                      \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedQFunctionAddOutput(CeedQFunction qf, const char *fieldname,
                           CeedInt size, CeedEvalMode emode) {
  if (emode == CEED_EVAL_WEIGHT)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, 1, "Cannot create QFunction output with "
                     "CEED_EVAL_WEIGHT");
  // LCOV_EXCL_STOP
  int ierr = CeedQFunctionFieldSet(&qf->outputfields[qf->numoutputfields],
                                   fieldname, size, emode);
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

  @param qf            CeedQFunction
  @param[out] vlength  Variable to store vector length

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
  @brief Get the source path string for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] source     Variable to store source path string

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetSourcePath(CeedQFunction qf, char **source) {
  *source = (char *) qf->sourcepath;
  return 0;
}

/**
  @brief Get the User Function for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] f          Variable to store user function

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetUserFunction(CeedQFunction qf, CeedQFunctionUser *f) {
  *f = qf->function;
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
  if (qf->fortranstatus) {
    fContext *fctx = qf->ctx;
    *ctxsize = fctx->innerctxsize;
  } else {
    *ctxsize = qf->ctxsize;
  }
  return 0;
}

/**
  @brief Get global context for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] ctx        Variable to store context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetContext(CeedQFunction qf, void **ctx) {
  *ctx = qf->ctx;
  return 0;
}

/**
  @brief Determine if Fortran interface was used

  @param qf                  CeedQFunction
  @param[out] fortranstatus  Variable to store Fortran status

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetFortranStatus(CeedQFunction qf, bool *fortranstatus) {
  *fortranstatus = qf->fortranstatus;
  return 0;
}

/**
  @brief Determine if QFunction is identity

  @param qf               CeedQFunction
  @param[out] identity    Variable to store identity status

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetIdentityStatus(CeedQFunction qf, bool *identity) {
  *identity = qf->identity;
  return 0;
}

/**
  @brief Get true user context for a CeedQFunction

  @param qf              CeedQFunction
  @param[out] ctx        Variable to store context data values

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetInnerContext(CeedQFunction qf, void **ctx) {
  if (qf->fortranstatus) {
    fContext *fctx = qf->ctx;
    *ctx = fctx->innerctx;
  } else {
    *ctx = qf->ctx;
  }


  return 0;
}

/**
  @brief Get backend data of a CeedQFunction

  @param qf              CeedQFunction
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionGetData(CeedQFunction qf, void **data) {
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

int CeedQFunctionSetData(CeedQFunction qf, void **data) {
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
  @brief View a field of a CeedQFunction

  @param[in] field       QFunction field to view
  @param[in] fieldnumber Number of field being viewed
  @param[in] stream      Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedQFunctionFieldView(CeedQFunctionField field, CeedInt fieldnumber,
                                  bool in, FILE *stream) {
  const char *inout = in ? "Input" : "Output";
  fprintf(stream, "    %s Field [%d]:\n"
          "      Name: \"%s\"\n"
          "      Size: %d\n"
          "      EvalMode: \"%s\"\n",
          inout, fieldnumber, field->fieldname, field->size,
          CeedEvalModes[field->emode]);

  return 0;
}

/**
  @brief View a CeedQFunction

  @param[in] qf     CeedQFunction to view
  @param[in] stream Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedQFunctionView(CeedQFunction qf, FILE *stream) {
  int ierr;

  fprintf(stream, "%sCeedQFunction %s\n",
          qf->qfname ? "Gallery " : "User ", qf->qfname ? qf->qfname : "");

  fprintf(stream, "  %d Input Field%s:\n", qf->numinputfields,
          qf->numinputfields>1 ? "s" : "");
  for (CeedInt i=0; i<qf->numinputfields; i++) {
    ierr = CeedQFunctionFieldView(qf->inputfields[i], i, 1, stream);
    CeedChk(ierr);
  }

  fprintf(stream, "  %d Output Field%s:\n", qf->numoutputfields,
          qf->numoutputfields>1 ? "s" : "");
  for (CeedInt i=0; i<qf->numoutputfields; i++) {
    ierr = CeedQFunctionFieldView(qf->outputfields[i], i, 0, stream);
    CeedChk(ierr);
  }
  return 0;
}

/**
  @brief Apply the action of a CeedQFunction

  @param qf      CeedQFunction
  @param Q       Number of quadrature points
  @param[in] u   Array of input CeedVectors
  @param[out] v  Array of output CeedVectors

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                       CeedVector *u, CeedVector *v) {
  int ierr;
  if (!qf->Apply)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, 1, "Backend does not support QFunctionApply");
  // LCOV_EXCL_STOP
  if (Q % qf->vlength)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, 2, "Number of quadrature points %d must be a "
                     "multiple of %d", Q, qf->vlength);
  // LCOV_EXCL_STOP
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

int CeedQFunctionGetFields(CeedQFunction qf, CeedQFunctionField **inputfields,
                           CeedQFunctionField **outputfields) {
  if (inputfields)
    *inputfields = qf->inputfields;
  if (outputfields)
    *outputfields = qf->outputfields;
  return 0;
}

/**
  @brief Get the name of a CeedQFunctionField

  @param qffield         CeedQFunctionField
  @param[out] fieldname  Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionFieldGetName(CeedQFunctionField qffield, char **fieldname) {
  *fieldname = (char *)qffield->fieldname;
  return 0;
}

/**
  @brief Get the number of components of a CeedQFunctionField

  @param qffield    CeedQFunctionField
  @param[out] size  Variable to store the size of the field

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedQFunctionFieldGetSize(CeedQFunctionField qffield, CeedInt *size) {
  *size = qffield->size;
  return 0;
}

/**
  @brief Get the CeedEvalMode of a CeedQFunctionField

  @param qffield         CeedQFunctionField
  @param[out] emode      Variable to store the field evaluation mode

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

  if (!*qf || --(*qf)->refcount > 0)
    return 0;
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
  // Free ctx if identity
  if ((*qf)->identity) {
    ierr = CeedFree(&(*qf)->ctx); CeedChk(ierr);
  }

  ierr = CeedFree(&(*qf)->sourcepath); CeedChk(ierr);
  ierr = CeedFree(&(*qf)->qfname); CeedChk(ierr);
  ierr = CeedDestroy(&(*qf)->ceed); CeedChk(ierr);
  ierr = CeedFree(qf); CeedChk(ierr);
  return 0;
}

/// @cond DOXYGEN_SKIP
// Indicate that no QFunction is provided by the user
CeedQFunction CEED_QFUNCTION_NONE = &ceed_qfunction_none;
/// @endcond
/// @}
