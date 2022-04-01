// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <ceed-impl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @file
/// Implementation of public CeedQFunction interfaces

/// @cond DOXYGEN_SKIP
static struct CeedQFunction_private ceed_qfunction_none;
/// @endcond

/// @addtogroup CeedQFunctionUser
/// @{

// Indicate that no QFunction is provided by the user
const CeedQFunction CEED_QFUNCTION_NONE = &ceed_qfunction_none;

/// @}

/// @cond DOXYGEN_SKIP
static struct {
  char name[CEED_MAX_RESOURCE_LEN];
  char source[CEED_MAX_RESOURCE_LEN];
  CeedInt vec_length;
  CeedQFunctionUser f;
  int (*init)(Ceed ceed, const char *name, CeedQFunction qf);
} gallery_qfunctions[1024];
static size_t num_qfunctions;
/// @endcond

/// ----------------------------------------------------------------------------
/// CeedQFunction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionDeveloper
/// @{

/**
  @brief Register a gallery QFunction

  @param name        Name for this backend to respond to
  @param source      Absolute path to source of QFunction,
                       "\path\CEED_DIR\gallery\folder\file.h:function_name"
  @param vec_length  Vector length.  Caller must ensure that number of quadrature
                       points is a multiple of vec_length.
  @param f           Function pointer to evaluate action at quadrature points.
                       See \ref CeedQFunctionUser.
  @param init        Initialization function called by CeedQFunctionInit() when the
                       QFunction is selected.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionRegister(const char *name, const char *source,
                          CeedInt vec_length, CeedQFunctionUser f,
                          int (*init)(Ceed, const char *, CeedQFunction)) {
  int ierr;

  if (num_qfunctions >= sizeof(gallery_qfunctions) / sizeof(
        gallery_qfunctions[0]))
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_MAJOR, "Too many gallery QFunctions");
  // LCOV_EXCL_STOP

  CeedDebugEnv("Gallery Register: %s", name);

  const char *relative_file_path;
  ierr = CeedGetJitRelativePath(source, &relative_file_path); CeedChk(ierr);

  strncpy(gallery_qfunctions[num_qfunctions].name, name, CEED_MAX_RESOURCE_LEN);
  gallery_qfunctions[num_qfunctions].name[CEED_MAX_RESOURCE_LEN-1] = 0;
  strncpy(gallery_qfunctions[num_qfunctions].source, relative_file_path,
          CEED_MAX_RESOURCE_LEN);
  gallery_qfunctions[num_qfunctions].source[CEED_MAX_RESOURCE_LEN-1] = 0;
  gallery_qfunctions[num_qfunctions].vec_length = vec_length;
  gallery_qfunctions[num_qfunctions].f = f;
  gallery_qfunctions[num_qfunctions].init = init;
  num_qfunctions++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a CeedQFunction field, used by CeedQFunctionAddInput/Output

  @param f           CeedQFunctionField
  @param field_name  Name of QFunction field
  @param size        Size of QFunction field, (num_comp * dim) for @ref CEED_EVAL_GRAD or
                       (num_comp * 1) for @ref CEED_EVAL_NONE, @ref CEED_EVAL_INTERP, and @ref CEED_EVAL_WEIGHT
  @param eval_mode   \ref CEED_EVAL_NONE to use values directly,
                       \ref CEED_EVAL_INTERP to use interpolated values,
                       \ref CEED_EVAL_GRAD to use gradients,
                       \ref CEED_EVAL_WEIGHT to use quadrature weights.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionFieldSet(CeedQFunctionField *f, const char *field_name,
                                 CeedInt size, CeedEvalMode eval_mode) {
  int ierr;

  ierr = CeedCalloc(1, f); CeedChk(ierr);
  ierr = CeedStringAllocCopy(field_name, (char **)&(*f)->field_name);
  CeedChk(ierr);
  (*f)->size = size;
  (*f)->eval_mode = eval_mode;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a CeedQFunction

  @param[in] field         QFunction field to view
  @param[in] field_number  Number of field being viewed
  @param[in] in            true for input field, false for output
  @param[in] stream        Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedQFunctionFieldView(CeedQFunctionField field,
                                  CeedInt field_number,
                                  bool in, FILE *stream) {
  int ierr;
  const char *inout = in ? "Input" : "Output";
  char *field_name;
  ierr = CeedQFunctionFieldGetName(field, &field_name); CeedChk(ierr);
  CeedInt size;
  ierr = CeedQFunctionFieldGetSize(field, &size); CeedChk(ierr);
  CeedEvalMode eval_mode;
  ierr = CeedQFunctionFieldGetEvalMode(field, &eval_mode); CeedChk(ierr);
  fprintf(stream, "    %s Field [%d]:\n"
          "      Name: \"%s\"\n"
          "      Size: %d\n"
          "      EvalMode: \"%s\"\n",
          inout, field_number, field_name, size, CeedEvalModes[eval_mode]);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set flag to determine if Fortran interface is used

  @param qf      CeedQFunction
  @param status  Boolean value to set as Fortran status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionSetFortranStatus(CeedQFunction qf, bool status) {
  qf->is_fortran = status;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedQFunction Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionBackend
/// @{

/**
  @brief Get the vector length of a CeedQFunction

  @param qf               CeedQFunction
  @param[out] vec_length  Variable to store vector length

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetVectorLength(CeedQFunction qf, CeedInt *vec_length) {
  *vec_length = qf->vec_length;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of inputs and outputs to a CeedQFunction

  @param qf               CeedQFunction
  @param[out] num_input   Variable to store number of input fields
  @param[out] num_output  Variable to store number of output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *num_input,
                            CeedInt *num_output) {
  if (num_input) *num_input = qf->num_input_fields;
  if (num_output) *num_output = qf->num_output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of the user function for a CeedQFunction

  @param qf                CeedQFunction
  @param[out] kernel_name  Variable to store source path string

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetKernelName(CeedQFunction qf, char **kernel_name) {
  *kernel_name = (char *) qf->kernel_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the source path string for a CeedQFunction

  @param qf                CeedQFunction
  @param[out] source_path  Variable to store source path string

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetSourcePath(CeedQFunction qf, char **source_path) {
  *source_path = (char *) qf->source_path;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Initalize and load QFunction source file into string buffer, including
           full text of local files in place of `#include "local.h"`.
           The `buffer` is set to `NULL` if there is no QFunction source file.
         Note: Caller is responsible for freeing the string buffer with `CeedFree()`.

  @param qf                  CeedQFunction
  @param[out] source_buffer  String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionLoadSourceToBuffer(CeedQFunction qf, char **source_buffer) {
  int ierr;
  char *source_path;

  ierr = CeedQFunctionGetSourcePath(qf, &source_path); CeedChk(ierr);
  *source_buffer = NULL;
  if (source_path) {
    ierr = CeedLoadSourceToBuffer(qf->ceed, source_path, source_buffer);
    CeedChk(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the User Function for a CeedQFunction

  @param qf      CeedQFunction
  @param[out] f  Variable to store user function

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetUserFunction(CeedQFunction qf, CeedQFunctionUser *f) {
  *f = qf->function;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get global context for a CeedQFunction.
           Note: For QFunctions from the Fortran interface, this
             function will return the Fortran context
             CeedQFunctionContext.

  @param qf        CeedQFunction
  @param[out] ctx  Variable to store CeedQFunctionContext

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetContext(CeedQFunction qf, CeedQFunctionContext *ctx) {
  *ctx = qf->ctx;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get context data of a CeedQFunction

  @param qf         CeedQFunction
  @param mem_type   Memory type on which to access the data. If the backend
                      uses a different memory type, this will perform a copy.
  @param[out] data  Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetContextData(CeedQFunction qf, CeedMemType mem_type,
                                void *data) {
  int ierr;
  bool is_writable;
  CeedQFunctionContext ctx;

  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);
  if (ctx) {
    ierr = CeedQFunctionIsContextWritable(qf, &is_writable); CeedChk(ierr);
    if (is_writable) {
      ierr = CeedQFunctionContextGetData(ctx, mem_type, data); CeedChk(ierr);
    } else {
      ierr = CeedQFunctionContextGetDataRead(ctx, mem_type, data); CeedChk(ierr);
    }
  } else {
    *(void **)data = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore context data of a CeedQFunction

  @param qf    CeedQFunction
  @param data  Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionRestoreContextData(CeedQFunction qf, void *data) {
  int ierr;
  bool is_writable;
  CeedQFunctionContext ctx;

  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);
  if (ctx) {
    ierr = CeedQFunctionIsContextWritable(qf, &is_writable); CeedChk(ierr);
    if (is_writable) {
      ierr = CeedQFunctionContextRestoreData(ctx, data); CeedChk(ierr);
    } else {
      ierr = CeedQFunctionContextRestoreDataRead(ctx, data); CeedChk(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get true user context for a CeedQFunction
           Note: For all QFunctions this function will return the user
             CeedQFunctionContext and not interface context
             CeedQFunctionContext, if any such object exists.

  @param qf        CeedQFunction
  @param[out] ctx  Variable to store CeedQFunctionContext

  @return An error code: 0 - success, otherwise - failure
  @ref Backend
**/
int CeedQFunctionGetInnerContext(CeedQFunction qf, CeedQFunctionContext *ctx) {
  int ierr;
  if (qf->is_fortran) {
    CeedFortranContext fortran_ctx = NULL;
    ierr = CeedQFunctionContextGetData(qf->ctx, CEED_MEM_HOST, &fortran_ctx);
    CeedChk(ierr);
    *ctx = fortran_ctx->inner_ctx;
    ierr = CeedQFunctionContextRestoreData(qf->ctx, (void *)&fortran_ctx);
    CeedChk(ierr);
  } else {
    *ctx = qf->ctx;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get inner context data of a CeedQFunction

  @param qf         CeedQFunction
  @param mem_type   Memory type on which to access the data. If the backend
                      uses a different memory type, this will perform a copy.
  @param[out] data  Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetInnerContextData(CeedQFunction qf, CeedMemType mem_type,
                                     void *data) {
  int ierr;
  bool is_writable;
  CeedQFunctionContext ctx;

  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
  if (ctx) {
    ierr = CeedQFunctionIsContextWritable(qf, &is_writable); CeedChk(ierr);
    if (is_writable) {
      ierr = CeedQFunctionContextGetData(ctx, mem_type, data); CeedChk(ierr);
    } else {
      ierr = CeedQFunctionContextGetDataRead(ctx, mem_type, data); CeedChk(ierr);
    }
  } else {
    *(void **)data = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore inner context data of a CeedQFunction

  @param qf    CeedQFunction
  @param data  Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionRestoreInnerContextData(CeedQFunction qf, void *data) {
  int ierr;
  bool is_writable;
  CeedQFunctionContext ctx;

  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
  if (ctx) {
    ierr = CeedQFunctionIsContextWritable(qf, &is_writable); CeedChk(ierr);
    if (is_writable) {
      ierr = CeedQFunctionContextRestoreData(ctx, data); CeedChk(ierr);
    } else {
      ierr = CeedQFunctionContextRestoreDataRead(ctx, data); CeedChk(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if QFunction is identity

  @param qf                CeedQFunction
  @param[out] is_identity  Variable to store identity status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionIsIdentity(CeedQFunction qf, bool *is_identity) {
  *is_identity = qf->is_identity;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if QFunctionContext is writable

  @param qf                CeedQFunction
  @param[out] is_writable  Variable to store context writeable staus

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionIsContextWritable(CeedQFunction qf, bool *is_writable) {
  *is_writable = qf->is_context_writable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a CeedQFunction

  @param qf         CeedQFunction
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetData(CeedQFunction qf, void *data) {
  *(void **)data = qf->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a CeedQFunction

  @param[out] qf  CeedQFunction
  @param data     Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionSetData(CeedQFunction qf, void *data) {
  qf->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedQFunction

  @param qf  CeedQFunction to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionReference(CeedQFunction qf) {
  qf->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs per quadrature required to apply QFunction

  @param qf    QFunction to estimate FLOPs for
  @param flops Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedQFunctionGetFlopsEstimate(CeedQFunction qf, CeedSize *flops) {
  if (qf->user_flop_estimate == -1)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_INCOMPLETE,
                     "Must set FLOPs estimate with CeedQFunctionSetUserFlopsEstimate");
  // LCOV_EXCL_STOP
  *flops = qf->user_flop_estimate;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedQFunction Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedQFunctionUser
/// @{

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed        A Ceed object where the CeedQFunction will be created
  @param vec_length  Vector length. Caller must ensure that number of quadrature
                       points is a multiple of vec_length.
  @param f           Function pointer to evaluate action at quadrature points.
                       See \ref CeedQFunctionUser.
  @param source      Absolute path to source of QFunction,
                       "\abs_path\file.h:function_name".
                       For support across all backends, this source must only
                       contain constructs supported by C99, C++11, and CUDA.
  @param[out] qf     Address of the variable where the newly created
                       CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  See \ref CeedQFunctionUser for details on the call-back function @a f's
    arguments.

  @ref User
**/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vec_length,
                                CeedQFunctionUser f,
                                const char *source, CeedQFunction *qf) {
  int ierr;
  char *source_copy, *kernel_name_copy;

  if (!ceed->QFunctionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "QFunction"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support QFunctionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedQFunctionCreateInterior(delegate, vec_length, f, source, qf);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  if (strlen(source) && !strrchr(source, ':'))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                     "Provided path to source does not include function name. "
                     "Provided: \"%s\"\nRequired: \"\\abs_path\\file.h:function_name\"",
                     source);
  // LCOV_EXCL_STOP

  ierr = CeedCalloc(1, qf); CeedChk(ierr);
  (*qf)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*qf)->ref_count = 1;
  (*qf)->vec_length = vec_length;
  (*qf)->is_identity = false;
  (*qf)->is_context_writable = true;
  (*qf)->function = f;
  (*qf)->user_flop_estimate = -1;
  if (strlen(source)) {
    const char *kernel_name = strrchr(source, ':') + 1;
    size_t kernel_name_len = strlen(kernel_name);
    ierr = CeedCalloc(kernel_name_len + 1, &kernel_name_copy); CeedChk(ierr);
    strncpy(kernel_name_copy, kernel_name, kernel_name_len);
    (*qf)->kernel_name = kernel_name_copy;

    size_t source_len = strlen(source) - kernel_name_len - 1;
    ierr = CeedCalloc(source_len + 1, &source_copy); CeedChk(ierr);
    strncpy(source_copy, source, source_len);
    (*qf)->source_path = source_copy;
  }
  ierr = CeedCalloc(CEED_FIELD_MAX, &(*qf)->input_fields); CeedChk(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &(*qf)->output_fields); CeedChk(ierr);
  ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms by name.

  @param ceed     A Ceed object where the CeedQFunction will be created
  @param name     Name of QFunction to use from gallery
  @param[out] qf  Address of the variable where the newly created
                    CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionCreateInteriorByName(Ceed ceed,  const char *name,
                                      CeedQFunction *qf) {
  int ierr;
  size_t match_len = 0, match_index = UINT_MAX;

  ierr = CeedQFunctionRegisterAll(); CeedChk(ierr);
  // Find matching backend
  if (!name) return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                                "No QFunction name provided");
  for (size_t i=0; i<num_qfunctions; i++) {
    size_t n;
    const char *curr_name = gallery_qfunctions[i].name;
    for (n = 0; curr_name[n] && curr_name[n] == name[n]; n++) {}
    if (n > match_len) {
      match_len = n;
      match_index = i;
    }
  }
  if (!match_len)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "No suitable gallery QFunction");
  // LCOV_EXCL_STOP

  // Build source path
  char *gallery_qfunction_source_path;

  // Create QFunction
  ierr = CeedGetInstalledJitPath(ceed, gallery_qfunctions[match_index].source,
                                 &gallery_qfunction_source_path); CeedChk(ierr);
  ierr = CeedQFunctionCreateInterior(ceed,
                                     gallery_qfunctions[match_index].vec_length,
                                     gallery_qfunctions[match_index].f,
                                     gallery_qfunction_source_path, qf);
  CeedChk(ierr);
  ierr = CeedFree(&gallery_qfunction_source_path); CeedChkBackend(ierr);

  // QFunction specific setup
  ierr = gallery_qfunctions[match_index].init(ceed, name, *qf); CeedChk(ierr);

  // Copy name
  ierr = CeedStringAllocCopy(name, (char **)&(*qf)->gallery_name); CeedChk(ierr);
  (*qf)->is_gallery = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create an identity CeedQFunction. Inputs are written into outputs in
           the order given. This is useful for CeedOperators that can be
           represented with only the action of a CeedRestriction and CeedBasis,
           such as restriction and prolongation operators for p-multigrid.
           Backends may optimize CeedOperators with this CeedQFunction to avoid
           the copy of input data to output fields by using the same memory
           location for both.

  @param ceed          A Ceed object where the CeedQFunction will be created
  @param[in] size      Size of the QFunction fields
  @param[in] in_mode   CeedEvalMode for input to CeedQFunction
  @param[in] out_mode  CeedEvalMode for output to CeedQFunction
  @param[out] qf       Address of the variable where the newly created
                         CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionCreateIdentity(Ceed ceed, CeedInt size, CeedEvalMode in_mode,
                                CeedEvalMode out_mode, CeedQFunction *qf) {
  int ierr;

  ierr = CeedQFunctionCreateInteriorByName(ceed, "Identity", qf); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(*qf, "input", size, in_mode); CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(*qf, "output", size, out_mode); CeedChk(ierr);

  (*qf)->is_identity = true;

  CeedQFunctionContext ctx;
  CeedContextFieldLabel size_label;
  ierr = CeedQFunctionGetContext(*qf, &ctx); CeedChk(ierr);
  ierr = CeedQFunctionContextGetFieldLabel(ctx, "size", &size_label);
  CeedChk(ierr);
  ierr = CeedQFunctionContextSetInt32(ctx, size_label, &size); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedQFunction. Both pointers should
           be destroyed with `CeedQFunctionDestroy()`;
           Note: If `*qf_copy` is non-NULL, then it is assumed that
           `*qf_copy` is a pointer to a CeedQFunction. This
           CeedQFunction will be destroyed if `*qf_copy` is the only
           reference to this CeedQFunction.

  @param qf            CeedQFunction to copy reference to
  @param[out] qf_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionReferenceCopy(CeedQFunction qf, CeedQFunction *qf_copy) {
  int ierr;

  ierr = CeedQFunctionReference(qf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(qf_copy); CeedChk(ierr);
  *qf_copy = qf;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a CeedQFunction input

  @param qf          CeedQFunction
  @param field_name  Name of QFunction field
  @param size        Size of QFunction field, (num_comp * dim) for @ref CEED_EVAL_GRAD or
                       (num_comp * 1) for @ref CEED_EVAL_NONE and @ref CEED_EVAL_INTERP
  @param eval_mode   \ref CEED_EVAL_NONE to use values directly,
                       \ref CEED_EVAL_INTERP to use interpolated values,
                       \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionAddInput(CeedQFunction qf, const char *field_name,
                          CeedInt size,
                          CeedEvalMode eval_mode) {
  if (qf->is_immutable)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_MAJOR,
                     "QFunction cannot be changed after set as immutable");
  // LCOV_EXCL_STOP
  if ((eval_mode == CEED_EVAL_WEIGHT) && (size != 1))
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_DIMENSION,
                     "CEED_EVAL_WEIGHT should have size 1");
  // LCOV_EXCL_STOP
  int ierr = CeedQFunctionFieldSet(&qf->input_fields[qf->num_input_fields],
                                   field_name, size, eval_mode);
  CeedChk(ierr);
  qf->num_input_fields++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a CeedQFunction output

  @param qf          CeedQFunction
  @param field_name  Name of QFunction field
  @param size        Size of QFunction field, (num_comp * dim) for @ref CEED_EVAL_GRAD or
                       (num_comp * 1) for @ref CEED_EVAL_NONE and @ref CEED_EVAL_INTERP
  @param eval_mode   \ref CEED_EVAL_NONE to use values directly,
                       \ref CEED_EVAL_INTERP to use interpolated values,
                       \ref CEED_EVAL_GRAD to use gradients.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionAddOutput(CeedQFunction qf, const char *field_name,
                           CeedInt size, CeedEvalMode eval_mode) {
  if (qf->is_immutable)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_MAJOR,
                     "QFunction cannot be changed after set as immutable");
  // LCOV_EXCL_STOP
  if (eval_mode == CEED_EVAL_WEIGHT)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_DIMENSION,
                     "Cannot create QFunction output with "
                     "CEED_EVAL_WEIGHT");
  // LCOV_EXCL_STOP
  int ierr = CeedQFunctionFieldSet(&qf->output_fields[qf->num_output_fields],
                                   field_name, size, eval_mode);
  CeedChk(ierr);
  qf->num_output_fields++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedQFunctionFields of a CeedQFunction

  Note: Calling this function asserts that setup is complete
          and sets the CeedQFunction as immutable.

  @param qf                      CeedQFunction
  @param[out] num_input_fields   Variable to store number of input fields
  @param[out] input_fields       Variable to store input fields
  @param[out] num_output_fields  Variable to store number of output fields
  @param[out] output_fields      Variable to store output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionGetFields(CeedQFunction qf, CeedInt *num_input_fields,
                           CeedQFunctionField **input_fields,
                           CeedInt *num_output_fields,
                           CeedQFunctionField **output_fields) {
  qf->is_immutable = true;
  if (num_input_fields) *num_input_fields = qf->num_input_fields;
  if (input_fields) *input_fields = qf->input_fields;
  if (num_output_fields) *num_output_fields = qf->num_output_fields;
  if (output_fields) *output_fields = qf->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of a CeedQFunctionField

  @param qf_field         CeedQFunctionField
  @param[out] field_name  Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetName(CeedQFunctionField qf_field, char **field_name) {
  *field_name = (char *)qf_field->field_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components of a CeedQFunctionField

  @param qf_field   CeedQFunctionField
  @param[out] size  Variable to store the size of the field

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetSize(CeedQFunctionField qf_field, CeedInt *size) {
  *size = qf_field->size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedEvalMode of a CeedQFunctionField

  @param qf_field        CeedQFunctionField
  @param[out] eval_mode  Variable to store the field evaluation mode

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetEvalMode(CeedQFunctionField qf_field,
                                  CeedEvalMode *eval_mode) {
  *eval_mode = qf_field->eval_mode;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set global context for a CeedQFunction

  @param qf   CeedQFunction
  @param ctx  Context data to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetContext(CeedQFunction qf, CeedQFunctionContext ctx) {
  int ierr;
  ierr = CeedQFunctionContextDestroy(&qf->ctx); CeedChk(ierr);
  qf->ctx = ctx;
  ierr = CeedQFunctionContextReference(ctx); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set writability of CeedQFunctionContext when calling the `CeedQFunctionUser`.
           The default value is 'is_writable == true'.

           Setting `is_writable == true` indicates the `CeedQFunctionUser` writes
           into the CeedQFunctionContextData and requires memory syncronization
           after calling `CeedQFunctionApply()`.

           Setting 'is_writable == false' asserts that `CeedQFunctionUser` does not
           modify the CeedQFunctionContextData. Violating this assertion may lead
           to inconsistent data.

           Setting `is_writable == false` may offer a performance improvement on GPU backends.

  @param qf           CeedQFunction
  @param is_writable  Writability status

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetContextWritable(CeedQFunction qf, bool is_writable) {
  qf->is_context_writable = is_writable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set estimated number of FLOPs per quadrature required to apply QFunction

  @param qf    QFunction to estimate FLOPs for
  @param flops FLOPs per quadrature point estimate

  @ref Backend
**/
int CeedQFunctionSetUserFlopsEstimate(CeedQFunction qf, CeedSize flops) {
  if (flops < 0)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_INCOMPATIBLE,
                     "Must set non-negative FLOPs estimate");
  // LCOV_EXCL_STOP
  qf->user_flop_estimate = flops;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedQFunction

  @param[in] qf      CeedQFunction to view
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionView(CeedQFunction qf, FILE *stream) {
  int ierr;

  fprintf(stream, "%sCeedQFunction %s\n",
          qf->is_gallery ? "Gallery " : "User ",
          qf->is_gallery ? qf->gallery_name : qf->kernel_name);

  fprintf(stream, "  %d Input Field%s:\n", qf->num_input_fields,
          qf->num_input_fields>1 ? "s" : "");
  for (CeedInt i=0; i<qf->num_input_fields; i++) {
    ierr = CeedQFunctionFieldView(qf->input_fields[i], i, 1, stream);
    CeedChk(ierr);
  }

  fprintf(stream, "  %d Output Field%s:\n", qf->num_output_fields,
          qf->num_output_fields>1 ? "s" : "");
  for (CeedInt i=0; i<qf->num_output_fields; i++) {
    ierr = CeedQFunctionFieldView(qf->output_fields[i], i, 0, stream);
    CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
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
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply the action of a CeedQFunction

  Note: Calling this function asserts that setup is complete
          and sets the CeedQFunction as immutable.

  @param qf      CeedQFunction
  @param Q       Number of quadrature points
  @param[in] u   Array of input CeedVectors
  @param[out] v  Array of output CeedVectors

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionApply(CeedQFunction qf, CeedInt Q,
                       CeedVector *u, CeedVector *v) {
  int ierr;
  if (!qf->Apply)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support QFunctionApply");
  // LCOV_EXCL_STOP
  if (Q % qf->vec_length)
    // LCOV_EXCL_START
    return CeedError(qf->ceed, CEED_ERROR_DIMENSION,
                     "Number of quadrature points %d must be a "
                     "multiple of %d", Q, qf->vec_length);
  // LCOV_EXCL_STOP
  qf->is_immutable = true;
  ierr = qf->Apply(qf, Q, u, v); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedQFunction

  @param qf  CeedQFunction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionDestroy(CeedQFunction *qf) {
  int ierr;

  if (!*qf || --(*qf)->ref_count > 0) return CEED_ERROR_SUCCESS;
  // Backend destroy
  if ((*qf)->Destroy) {
    ierr = (*qf)->Destroy(*qf); CeedChk(ierr);
  }
  // Free fields
  for (int i=0; i<(*qf)->num_input_fields; i++) {
    ierr = CeedFree(&(*(*qf)->input_fields[i]).field_name); CeedChk(ierr);
    ierr = CeedFree(&(*qf)->input_fields[i]); CeedChk(ierr);
  }
  for (int i=0; i<(*qf)->num_output_fields; i++) {
    ierr = CeedFree(&(*(*qf)->output_fields[i]).field_name); CeedChk(ierr);
    ierr = CeedFree(&(*qf)->output_fields[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&(*qf)->input_fields); CeedChk(ierr);
  ierr = CeedFree(&(*qf)->output_fields); CeedChk(ierr);

  // User context data object
  ierr = CeedQFunctionContextDestroy(&(*qf)->ctx); CeedChk(ierr);

  ierr = CeedFree(&(*qf)->source_path); CeedChk(ierr);
  ierr = CeedFree(&(*qf)->gallery_name); CeedChk(ierr);
  ierr = CeedFree(&(*qf)->kernel_name); CeedChk(ierr);
  ierr = CeedDestroy(&(*qf)->ceed); CeedChk(ierr);
  ierr = CeedFree(qf); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
