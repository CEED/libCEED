// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of public CeedQFunction interfaces

/// @cond DOXYGEN_SKIP
static struct CeedQFunction_private ceed_qfunction_none;
/// @endcond

/// @addtogroup CeedQFunctionUser
/// @{

// Indicate that no `CeedQFunction` is provided by the user
const CeedQFunction CEED_QFUNCTION_NONE = &ceed_qfunction_none;

/// @}

/// @cond DOXYGEN_SKIP
static struct {
  char              name[CEED_MAX_RESOURCE_LEN];
  char              source[CEED_MAX_RESOURCE_LEN];
  CeedInt           vec_length;
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
  @brief Register a gallery `CeedQFunction`

  @param[in] name       Name for this backend to respond to
  @param[in] source     Absolute path to source of `CeedQFunction`, "\path\CEED_DIR\gallery\folder\file.h:function_name"
  @param[in] vec_length Vector length.
                          Caller must ensure that number of quadrature points is a multiple of `vec_length`.
  @param[in] f          Function pointer to evaluate action at quadrature points.
                          See `CeedQFunctionUser`.
  @param[in] init       Initialization function called by @ref CeedQFunctionCreateInteriorByName() when the `CeedQFunction` is selected.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedQFunctionRegister(const char *name, const char *source, CeedInt vec_length, CeedQFunctionUser f,
                          int (*init)(Ceed, const char *, CeedQFunction)) {
  const char *relative_file_path;
  int         ierr = 0;

  CeedDebugEnv("Gallery Register: %s", name);
  CeedCall(CeedGetJitRelativePath(source, &relative_file_path));
  CeedPragmaCritical(CeedQFunctionRegister) {
    if (num_qfunctions < sizeof(gallery_qfunctions) / sizeof(gallery_qfunctions[0])) {
      strncpy(gallery_qfunctions[num_qfunctions].name, name, CEED_MAX_RESOURCE_LEN);
      gallery_qfunctions[num_qfunctions].name[CEED_MAX_RESOURCE_LEN - 1] = 0;
      strncpy(gallery_qfunctions[num_qfunctions].source, relative_file_path, CEED_MAX_RESOURCE_LEN);
      gallery_qfunctions[num_qfunctions].source[CEED_MAX_RESOURCE_LEN - 1] = 0;
      gallery_qfunctions[num_qfunctions].vec_length                        = vec_length;
      gallery_qfunctions[num_qfunctions].f                                 = f;
      gallery_qfunctions[num_qfunctions].init                              = init;
      num_qfunctions++;
    } else {
      ierr = 1;
    }
  }
  CeedCheck(ierr == 0, NULL, CEED_ERROR_MAJOR, "Too many gallery CeedQFunctions");
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set a `CeedQFunction` field, used by @ref CeedQFunctionAddInput() and @ref CeedQFunctionAddOutput()

  @param[out] f           `CeedQFunctionField`
  @param[in]  field_name  Name of `CeedQFunction` field
  @param[in]  size        Size of `CeedQFunction` field, (`num_comp * 1`) for @ref CEED_EVAL_NONE and @ref CEED_EVAL_WEIGHT, (`num_comp * 1`) for @ref CEED_EVAL_INTERP for an \f$H^1\f$ space or (`num_comp * dim`) for an \f$H(\mathrm{div})\f$ or \f$H(\mathrm{curl})\f$ space, (`num_comp * dim`) for @ref CEED_EVAL_GRAD, or (num_comp * 1) for @ref CEED_EVAL_DIV, and (`num_comp * curl_dim`) with `curl_dim = 1` if `dim < 3` and `curl_dim = dim` for @ref CEED_EVAL_CURL.
  @param[in]  eval_mode   @ref CEED_EVAL_NONE to use values directly,
                            @ref CEED_EVAL_WEIGHT to use quadrature weights,
                            @ref CEED_EVAL_INTERP to use interpolated values,
                            @ref CEED_EVAL_GRAD to use gradients,
                            @ref CEED_EVAL_DIV to use divergence,
                            @ref CEED_EVAL_CURL to use curl

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionFieldSet(CeedQFunctionField *f, const char *field_name, CeedInt size, CeedEvalMode eval_mode) {
  CeedCall(CeedCalloc(1, f));
  CeedCall(CeedStringAllocCopy(field_name, (char **)&(*f)->field_name));
  (*f)->size      = size;
  (*f)->eval_mode = eval_mode;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a `CeedQFunction`

  @param[in] field        `CeedQFunction` field to view
  @param[in] field_number Number of field being viewed
  @param[in] in           true for input field, false for output
  @param[in] stream       Stream to view to, e.g., `stdout`

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedQFunctionFieldView(CeedQFunctionField field, CeedInt field_number, bool in, FILE *stream) {
  const char  *inout = in ? "Input" : "Output";
  const char  *field_name;
  CeedInt      size;
  CeedEvalMode eval_mode;

  CeedCall(CeedQFunctionFieldGetData(field, &field_name, &size, &eval_mode));
  fprintf(stream,
          "    %s field %" CeedInt_FMT
          ":\n"
          "      Name: \"%s\"\n"
          "      Size: %" CeedInt_FMT
          "\n"
          "      EvalMode: \"%s\"\n",
          inout, field_number, field_name, size, CeedEvalModes[eval_mode]);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set flag to determine if Fortran interface is used

  @param[in,out] qf     CeedQFunction
  @param[in]     status Boolean value to set as Fortran status

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
  @brief Get the vector length of a `CeedQFunction`

  @param[in]  qf         `CeedQFunction`
  @param[out] vec_length Variable to store vector length

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetVectorLength(CeedQFunction qf, CeedInt *vec_length) {
  *vec_length = qf->vec_length;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of inputs and outputs to a `CeedQFunction`

  @param[in]  qf         `CeedQFunction`
  @param[out] num_input  Variable to store number of input fields
  @param[out] num_output Variable to store number of output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetNumArgs(CeedQFunction qf, CeedInt *num_input, CeedInt *num_output) {
  if (num_input) *num_input = qf->num_input_fields;
  if (num_output) *num_output = qf->num_output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of the user function for a `CeedQFunction`

  @param[in]  qf          `CeedQFunction`
  @param[out] kernel_name Variable to store source path string

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetKernelName(CeedQFunction qf, const char **kernel_name) {
  if (!qf->kernel_name) {
    char *kernel_name_copy;

    if (qf->user_source) {
      const char *kernel_name     = strrchr(qf->user_source, ':') + 1;
      size_t      kernel_name_len = strlen(kernel_name);

      CeedCall(CeedCalloc(kernel_name_len + 1, &kernel_name_copy));
      memcpy(kernel_name_copy, kernel_name, kernel_name_len);
    } else {
      CeedCall(CeedCalloc(1, &kernel_name_copy));
    }
    qf->kernel_name = kernel_name_copy;
  }

  *kernel_name = qf->kernel_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the source path string for a `CeedQFunction`

  @param[in]  qf          `CeedQFunction`
  @param[out] source_path Variable to store source path string

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetSourcePath(CeedQFunction qf, const char **source_path) {
  if (!qf->source_path && qf->user_source) {
    Ceed        ceed;
    bool        is_absolute_path;
    char       *source_path_copy;
    const char *absolute_path;
    const char *kernel_name     = strrchr(qf->user_source, ':') + 1;
    size_t      kernel_name_len = strlen(kernel_name);

    CeedCall(CeedQFunctionGetCeed(qf, &ceed));
    CeedCall(CeedCheckFilePath(ceed, qf->user_source, &is_absolute_path));
    if (is_absolute_path) {
      absolute_path = (char *)qf->user_source;
    } else {
      CeedCall(CeedGetJitAbsolutePath(ceed, qf->user_source, &absolute_path));
    }
    CeedCall(CeedDestroy(&ceed));

    size_t source_len = strlen(absolute_path) - kernel_name_len - 1;

    CeedCall(CeedCalloc(source_len + 1, &source_path_copy));
    memcpy(source_path_copy, absolute_path, source_len);
    qf->source_path = source_path_copy;

    if (!is_absolute_path) CeedCall(CeedFree(&absolute_path));
  }

  *source_path = (char *)qf->source_path;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Initialize and load `CeedQFunction` source file into string buffer, including full text of local files in place of `#include "local.h"`.

  The `buffer` is set to `NULL` if there is no `CeedQFunction` source file.

  Note: This function may as well return a mutable buffer, but all current uses
  do not modify it. (This is just a downside of `const` semantics with output
  arguments instead of returns.)

  Note: Caller is responsible for freeing the string buffer with @ref CeedFree().

  @param[in]  qf            `CeedQFunction`
  @param[out] source_buffer String buffer for source file contents

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionLoadSourceToBuffer(CeedQFunction qf, const char **source_buffer) {
  const char *source_path;

  CeedCall(CeedQFunctionGetSourcePath(qf, &source_path));
  *source_buffer = NULL;
  if (source_path) {
    Ceed  ceed;
    char *buffer = NULL;

    CeedCall(CeedQFunctionGetCeed(qf, &ceed));
    CeedCall(CeedLoadSourceToBuffer(ceed, source_path, &buffer));
    CeedCall(CeedDestroy(&ceed));
    *source_buffer = buffer;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the User Function for a `CeedQFunction`

  @param[in]  qf `CeedQFunction`
  @param[out] f  Variable to store user function

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetUserFunction(CeedQFunction qf, CeedQFunctionUser *f) {
  *f = qf->function;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get global context for a `CeedQFunction`.

  Note: For `CeedQFunction` from the Fortran interface, this function will return the Fortran context `CeedQFunctionContext`.

  @param[in]  qf  CeedQFunction
  @param[out] ctx Variable to store CeedQFunctionContext

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetContext(CeedQFunction qf, CeedQFunctionContext *ctx) {
  *ctx = NULL;
  if (qf->ctx) CeedCall(CeedQFunctionContextReferenceCopy(qf->ctx, ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get context data of a `CeedQFunction`

  @param[in]  qf       `CeedQFunction`
  @param[in]  mem_type Memory type on which to access the data.
                         If the backend uses a different memory type, this will perform a copy.
  @param[out] data     Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetContextData(CeedQFunction qf, CeedMemType mem_type, void *data) {
  bool                 is_writable;
  CeedQFunctionContext ctx;

  CeedCall(CeedQFunctionGetContext(qf, &ctx));
  if (ctx) {
    CeedCall(CeedQFunctionIsContextWritable(qf, &is_writable));
    if (is_writable) {
      CeedCall(CeedQFunctionContextGetData(ctx, mem_type, data));
    } else {
      CeedCall(CeedQFunctionContextGetDataRead(ctx, mem_type, data));
    }
  } else {
    *(void **)data = NULL;
  }
  CeedCall(CeedQFunctionContextDestroy(&ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore context data of a `CeedQFunction`

  @param[in]     qf   `CeedQFunction`
  @param[in,out] data Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionRestoreContextData(CeedQFunction qf, void *data) {
  bool                 is_writable;
  CeedQFunctionContext ctx;

  CeedCall(CeedQFunctionGetContext(qf, &ctx));
  if (ctx) {
    CeedCall(CeedQFunctionIsContextWritable(qf, &is_writable));
    if (is_writable) {
      CeedCall(CeedQFunctionContextRestoreData(ctx, data));
    } else {
      CeedCall(CeedQFunctionContextRestoreDataRead(ctx, data));
    }
  }
  CeedCall(CeedQFunctionContextDestroy(&ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get true user context for a `CeedQFunction`

  Note: For all `CeedQFunction` this function will return the user `CeedQFunctionContext` and not interface context `CeedQFunctionContext`, if any such object exists.

  @param[in]  qf  `CeedQFunction`
  @param[out] ctx Variable to store `CeedQFunctionContext`

  @return An error code: 0 - success, otherwise - failure
  @ref Backend
**/
int CeedQFunctionGetInnerContext(CeedQFunction qf, CeedQFunctionContext *ctx) {
  CeedQFunctionContext qf_ctx;

  CeedCall(CeedQFunctionGetContext(qf, &qf_ctx));
  if (qf->is_fortran) {
    CeedFortranContext fortran_ctx = NULL;

    CeedCall(CeedQFunctionContextGetData(qf_ctx, CEED_MEM_HOST, &fortran_ctx));
    *ctx = fortran_ctx->inner_ctx;
    CeedCall(CeedQFunctionContextRestoreData(qf_ctx, &fortran_ctx));
  } else {
    *ctx = qf_ctx;
  }
  CeedCall(CeedQFunctionContextDestroy(&qf_ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get inner context data of a `CeedQFunction`

  @param[in]  qf       `CeedQFunction`
  @param[in]  mem_type Memory type on which to access the data.
                         If the backend uses a different memory type, this will perform a copy.
  @param[out] data     Data on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetInnerContextData(CeedQFunction qf, CeedMemType mem_type, void *data) {
  bool                 is_writable;
  CeedQFunctionContext ctx;

  CeedCall(CeedQFunctionGetInnerContext(qf, &ctx));
  if (ctx) {
    CeedCall(CeedQFunctionIsContextWritable(qf, &is_writable));
    if (is_writable) {
      CeedCall(CeedQFunctionContextGetData(ctx, mem_type, data));
    } else {
      CeedCall(CeedQFunctionContextGetDataRead(ctx, mem_type, data));
    }
  } else {
    *(void **)data = NULL;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore inner context data of a `CeedQFunction`

  @param[in]     qf   `CeedQFunction`
  @param[in,out] data Data to restore

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionRestoreInnerContextData(CeedQFunction qf, void *data) {
  bool                 is_writable;
  CeedQFunctionContext ctx;

  CeedCall(CeedQFunctionGetInnerContext(qf, &ctx));
  if (ctx) {
    CeedCall(CeedQFunctionIsContextWritable(qf, &is_writable));
    if (is_writable) {
      CeedCall(CeedQFunctionContextRestoreData(ctx, data));
    } else {
      CeedCall(CeedQFunctionContextRestoreDataRead(ctx, data));
    }
  }
  *(void **)data = NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if `CeedQFunction` is identity

  @param[in]  qf          `CeedQFunction`
  @param[out] is_identity Variable to store identity status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionIsIdentity(CeedQFunction qf, bool *is_identity) {
  *is_identity = qf->is_identity;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if `CeedQFunctionContext` is writable

  @param[in]  qf          `CeedQFunction`
  @param[out] is_writable Variable to store context writeable status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionIsContextWritable(CeedQFunction qf, bool *is_writable) {
  *is_writable = qf->is_context_writable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a `CeedQFunction`

  @param[in]  qf   `CeedQFunction`
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionGetData(CeedQFunction qf, void *data) {
  *(void **)data = qf->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a `CeedQFunction`

  @param[in,out] qf   `CeedQFunction`
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionSetData(CeedQFunction qf, void *data) {
  qf->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the `CeedQFunction` is immutable

  @param[in]  qf           `CeedOperator`
  @param[out] is_immutable Variable to store immutability status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionIsImmutable(CeedQFunction qf, bool *is_immutable) {
  *is_immutable = qf->is_immutable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the immutable flag of a `CeedQFunction` to `true`

  @param[in,out] qf `CeedQFunction`

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionSetImmutable(CeedQFunction qf) {
  qf->is_immutable = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedQFunction`

  @param[in,out] qf `CeedQFunction` to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionReference(CeedQFunction qf) {
  qf->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs per quadrature required to apply `CeedQFunction`

  @param[in]  qf    `CeedQFunction` to estimate FLOPs for
  @param[out] flops Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedQFunctionGetFlopsEstimate(CeedQFunction qf, CeedSize *flops) {
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
  @brief Create a `CeedQFunction` for evaluating interior (volumetric) terms

  @param[in]  ceed       `Ceed` object used to create the `CeedQFunction`
  @param[in]  vec_length Vector length.
                           Caller must ensure that number of quadrature points is a multiple of `vec_length`.
  @param[in]  f          Function pointer to evaluate action at quadrature points.
                           See `CeedQFunctionUser`.
  @param[in]  source     Absolute path to source of `CeedQFunctionUser`, "\abs_path\file.h:function_name".
                           The entire source file must only contain constructs supported by all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).
                           The entire contents of this file and all locally included files are used during JiT compilation for GPU backends.
                           The header `ceed/types.h` is preferred over `ceed.h` or `ceed/ceed.h` for `CeedQFunction` source files.
                           The macro `CEED_RUNNING_JIT_PASS` is set during JiT and can be used to guard include statements that JiT compilers cannot use, such as `math.h` or `std*.h`.
                           All source files must be at the provided filepath at runtime for JiT to function.
  @param[out] qf         Address of the variable where the newly created `CeedQFunction` will be stored

  @return An error code: 0 - success, otherwise - failure

  See \ref CeedQFunctionUser for details on the call-back function `f` arguments.

  @ref User
**/
int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vec_length, CeedQFunctionUser f, const char *source, CeedQFunction *qf) {
  char *user_source_copy;

  if (!ceed->QFunctionCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "QFunction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedQFunctionCreateInterior");
    CeedCall(CeedQFunctionCreateInterior(delegate, vec_length, f, source, qf));
    CeedCall(CeedDestroy(&delegate));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(!strlen(source) || strrchr(source, ':'), ceed, CEED_ERROR_INCOMPLETE,
            "Provided path to source does not include function name. Provided: \"%s\"\nRequired: \"\\abs_path\\file.h:function_name\"", source);

  CeedCall(CeedCalloc(1, qf));
  CeedCall(CeedReferenceCopy(ceed, &(*qf)->ceed));
  (*qf)->ref_count           = 1;
  (*qf)->vec_length          = vec_length;
  (*qf)->is_identity         = false;
  (*qf)->is_context_writable = true;
  (*qf)->function            = f;
  (*qf)->user_flop_estimate  = -1;
  if (strlen(source)) {
    size_t user_source_len = strlen(source);

    CeedCall(CeedCalloc(user_source_len + 1, &user_source_copy));
    memcpy(user_source_copy, source, user_source_len);
    (*qf)->user_source = user_source_copy;
  }
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*qf)->input_fields));
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*qf)->output_fields));
  CeedCall(ceed->QFunctionCreate(*qf));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a `CeedQFunction` for evaluating interior (volumetric) terms by name

  @param[in]  ceed `Ceed` object used to create the `CeedQFunction`
  @param[in]  name Name of `CeedQFunction` to use from gallery
  @param[out] qf   Address of the variable where the newly created `CeedQFunction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionCreateInteriorByName(Ceed ceed, const char *name, CeedQFunction *qf) {
  size_t match_len = 0, match_index = UINT_MAX;

  CeedCall(CeedQFunctionRegisterAll());
  // Find matching backend
  CeedCheck(name, ceed, CEED_ERROR_INCOMPLETE, "No CeedQFunction name provided");
  for (size_t i = 0; i < num_qfunctions; i++) {
    size_t      n;
    const char *curr_name = gallery_qfunctions[i].name;
    for (n = 0; curr_name[n] && curr_name[n] == name[n]; n++) {
    }
    if (n > match_len) {
      match_len   = n;
      match_index = i;
    }
  }
  CeedCheck(match_len > 0, ceed, CEED_ERROR_UNSUPPORTED, "No suitable gallery CeedQFunction");

  // Create QFunction
  CeedCall(CeedQFunctionCreateInterior(ceed, gallery_qfunctions[match_index].vec_length, gallery_qfunctions[match_index].f,
                                       gallery_qfunctions[match_index].source, qf));

  // QFunction specific setup
  CeedCall(gallery_qfunctions[match_index].init(ceed, name, *qf));

  // Copy name
  CeedCall(CeedStringAllocCopy(name, (char **)&(*qf)->gallery_name));
  (*qf)->is_gallery = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create an identity `CeedQFunction`.

  Inputs are written into outputs in the order given.
  This is useful for `CeedOperator that can be represented with only the action of a `CeedElemRestriction` and `CeedBasis`, such as restriction and prolongation operators for p-multigrid.
  Backends may optimize `CeedOperator` with this `CeedQFunction` to avoid the copy of input data to output fields by using the same memory location for both.

  @param[in]  ceed     `Ceed` object used to create the `CeedQFunction`
  @param[in]  size     Size of the `CeedQFunction` fields
  @param[in]  in_mode  @ref CeedEvalMode for input to `CeedQFunction`
  @param[in]  out_mode @ref CeedEvalMode for output to `CeedQFunction`
  @param[out] qf       Address of the variable where the newly created `CeedQFunction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionCreateIdentity(Ceed ceed, CeedInt size, CeedEvalMode in_mode, CeedEvalMode out_mode, CeedQFunction *qf) {
  CeedQFunctionContext  ctx;
  CeedContextFieldLabel size_label;

  CeedCall(CeedQFunctionCreateInteriorByName(ceed, "Identity", qf));
  CeedCall(CeedQFunctionAddInput(*qf, "input", size, in_mode));
  CeedCall(CeedQFunctionAddOutput(*qf, "output", size, out_mode));

  (*qf)->is_identity = true;

  CeedCall(CeedQFunctionGetContext(*qf, &ctx));
  CeedCall(CeedQFunctionContextGetFieldLabel(ctx, "size", &size_label));
  CeedCall(CeedQFunctionContextSetInt32(ctx, size_label, &size));
  CeedCall(CeedQFunctionContextDestroy(&ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedQFunction`.

  Both pointers should be destroyed with @ref CeedQFunctionDestroy().

  Note: If the value of `*qf_copy` passed to this function is non-NULL, then it is assumed that `*qf_copy` is a pointer to a `CeedQFunction`.
        This `CeedQFunction` will be destroyed if `*qf_copy` is the only reference to this `CeedQFunction`.

  @param[in]  qf      `CeedQFunction` to copy reference to
  @param[out] qf_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionReferenceCopy(CeedQFunction qf, CeedQFunction *qf_copy) {
  CeedCall(CeedQFunctionReference(qf));
  CeedCall(CeedQFunctionDestroy(qf_copy));
  *qf_copy = qf;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a `CeedQFunction` input

  @param[in,out] qf         `CeedQFunction`
  @param[in]     field_name Name of `CeedQFunction` field
  @param[in]     size       Size of `CeedQFunction` field, (`num_comp * 1`) for @ref CEED_EVAL_NONE, (`num_comp * 1`) for @ref CEED_EVAL_INTERP for an \f$H^1\f$ space or (`num_comp * dim`) for an \f$H(\mathrm{div})\f$ or \f$H(\mathrm{curl})\f$ space, (`num_comp * dim`) for @ref CEED_EVAL_GRAD, or (`num_comp * 1`) for @ref CEED_EVAL_DIV, and (`num_comp * curl_dim`) with `curl_dim = 1` if `dim < 3` otherwise `curl_dim = dim` for @ref CEED_EVAL_CURL.
  @param[in]     eval_mode  @ref CEED_EVAL_NONE to use values directly,
                              @ref CEED_EVAL_INTERP to use interpolated values,
                              @ref CEED_EVAL_GRAD to use gradients,
                              @ref CEED_EVAL_DIV to use divergence,
                              @ref CEED_EVAL_CURL to use curl

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionAddInput(CeedQFunction qf, const char *field_name, CeedInt size, CeedEvalMode eval_mode) {
  bool is_immutable;

  CeedCall(CeedQFunctionIsImmutable(qf, &is_immutable));
  CeedCheck(!is_immutable, CeedQFunctionReturnCeed(qf), CEED_ERROR_MAJOR, "QFunction cannot be changed after set as immutable");
  CeedCheck(eval_mode != CEED_EVAL_WEIGHT || size == 1, CeedQFunctionReturnCeed(qf), CEED_ERROR_DIMENSION, "CEED_EVAL_WEIGHT should have size 1");
  for (CeedInt i = 0; i < qf->num_input_fields; i++) {
    CeedCheck(strcmp(field_name, qf->input_fields[i]->field_name), CeedQFunctionReturnCeed(qf), CEED_ERROR_MINOR,
              "CeedQFunction field names must be unique. Duplicate name: %s", field_name);
  }
  for (CeedInt i = 0; i < qf->num_output_fields; i++) {
    CeedCheck(strcmp(field_name, qf->output_fields[i]->field_name), CeedQFunctionReturnCeed(qf), CEED_ERROR_MINOR,
              "CeedQFunction field names must be unique. Duplicate name: %s", field_name);
  }
  CeedCall(CeedQFunctionFieldSet(&qf->input_fields[qf->num_input_fields], field_name, size, eval_mode));
  qf->num_input_fields++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a `CeedQFunction` output

  @param[in,out] qf         `CeedQFunction`
  @param[in]     field_name Name of `CeedQFunction` field
  @param[in]     size       Size of `CeedQFunction` field, (`num_comp * 1`) for @ref CEED_EVAL_NONE, (`num_comp * 1`) for @ref CEED_EVAL_INTERP for an \f$H^1\f$ space or (`num_comp * dim`) for an \f$H(\mathrm{div})\f$ or \f$H(\mathrm{curl})\f$ space, (`num_comp * dim`) for @ref CEED_EVAL_GRAD, or (`num_comp * 1`) for @ref CEED_EVAL_DIV, and (`num_comp * curl_dim`) with `curl_dim = 1` if `dim < 3` else dim for @ref CEED_EVAL_CURL.
  @param[in]     eval_mode  @ref CEED_EVAL_NONE to use values directly,
                              @ref CEED_EVAL_INTERP to use interpolated values,
                              @ref CEED_EVAL_GRAD to use gradients,
                              @ref CEED_EVAL_DIV to use divergence,
                              @ref CEED_EVAL_CURL to use curl.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionAddOutput(CeedQFunction qf, const char *field_name, CeedInt size, CeedEvalMode eval_mode) {
  bool is_immutable;

  CeedCall(CeedQFunctionIsImmutable(qf, &is_immutable));
  CeedCheck(!is_immutable, CeedQFunctionReturnCeed(qf), CEED_ERROR_MAJOR, "CeedQFunction cannot be changed after set as immutable");
  CeedCheck(eval_mode != CEED_EVAL_WEIGHT, CeedQFunctionReturnCeed(qf), CEED_ERROR_DIMENSION,
            "Cannot create CeedQFunction output with CEED_EVAL_WEIGHT");
  for (CeedInt i = 0; i < qf->num_input_fields; i++) {
    CeedCheck(strcmp(field_name, qf->input_fields[i]->field_name), CeedQFunctionReturnCeed(qf), CEED_ERROR_MINOR,
              "CeedQFunction field names must be unique");
  }
  for (CeedInt i = 0; i < qf->num_output_fields; i++) {
    CeedCheck(strcmp(field_name, qf->output_fields[i]->field_name), CeedQFunctionReturnCeed(qf), CEED_ERROR_MINOR,
              "CeedQFunction field names must be unique");
  }
  CeedCall(CeedQFunctionFieldSet(&qf->output_fields[qf->num_output_fields], field_name, size, eval_mode));
  qf->num_output_fields++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedQFunctionField` of a `CeedQFunction`

  Note: Calling this function asserts that setup is complete and sets the `CeedQFunction` as immutable.

  @param[in]  qf                `CeedQFunction`
  @param[out] num_input_fields  Variable to store number of input fields
  @param[out] input_fields      Variable to store input fields
  @param[out] num_output_fields Variable to store number of output fields
  @param[out] output_fields     Variable to store output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionGetFields(CeedQFunction qf, CeedInt *num_input_fields, CeedQFunctionField **input_fields, CeedInt *num_output_fields,
                           CeedQFunctionField **output_fields) {
  CeedCall(CeedQFunctionSetImmutable(qf));
  if (num_input_fields) *num_input_fields = qf->num_input_fields;
  if (input_fields) *input_fields = qf->input_fields;
  if (num_output_fields) *num_output_fields = qf->num_output_fields;
  if (output_fields) *output_fields = qf->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of a `CeedQFunctionField`

  @param[in]  qf_field   `CeedQFunctionField`
  @param[out] field_name Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetName(CeedQFunctionField qf_field, const char **field_name) {
  *field_name = qf_field->field_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components of a `CeedQFunctionField`

  @param[in]  qf_field `CeedQFunctionField`
  @param[out] size     Variable to store the size of the field

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetSize(CeedQFunctionField qf_field, CeedInt *size) {
  *size = qf_field->size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the @ref CeedEvalMode of a `CeedQFunctionField`

  @param[in]  qf_field  `CeedQFunctionField`
  @param[out] eval_mode Variable to store the field evaluation mode

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetEvalMode(CeedQFunctionField qf_field, CeedEvalMode *eval_mode) {
  *eval_mode = qf_field->eval_mode;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the data of a `CeedQFunctionField`.

  Any arguments set as `NULL` are ignored.

  @param[in]  qf_field   `CeedQFunctionField`
  @param[out] field_name Variable to store the field name
  @param[out] size       Variable to store the size of the field
  @param[out] eval_mode  Variable to store the field evaluation mode

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionFieldGetData(CeedQFunctionField qf_field, const char **field_name, CeedInt *size, CeedEvalMode *eval_mode) {
  if (field_name) CeedCall(CeedQFunctionFieldGetName(qf_field, field_name));
  if (size) CeedCall(CeedQFunctionFieldGetSize(qf_field, size));
  if (eval_mode) CeedCall(CeedQFunctionFieldGetEvalMode(qf_field, eval_mode));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set global context for a `CeedQFunction`

  @param[in,out] qf  `CeedQFunction`
  @param[in]     ctx Context data to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetContext(CeedQFunction qf, CeedQFunctionContext ctx) {
  CeedCall(CeedQFunctionContextDestroy(&qf->ctx));
  qf->ctx = ctx;
  if (ctx) CeedCall(CeedQFunctionContextReference(ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set writability of `CeedQFunctionContext` when calling the `CeedQFunctionUser`.

  The default value is `is_writable == true`.

  Setting `is_writable == true` indicates the `CeedQFunctionUser` writes into the `CeedQFunctionContext` and requires memory synchronization after calling @ref CeedQFunctionApply().

  Setting 'is_writable == false' asserts that `CeedQFunctionUser` does not modify the `CeedQFunctionContext`.
  Violating this assertion may lead to inconsistent data.

  Setting `is_writable == false` may offer a performance improvement on GPU backends.

  @param[in,out] qf          `CeedQFunction`
  @param[in]     is_writable Boolean flag for writability status

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionSetContextWritable(CeedQFunction qf, bool is_writable) {
  qf->is_context_writable = is_writable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set estimated number of FLOPs per quadrature required to apply `CeedQFunction`

  @param[in]  qf    `CeedQFunction` to estimate FLOPs for
  @param[out] flops FLOPs per quadrature point estimate

  @ref Backend
**/
int CeedQFunctionSetUserFlopsEstimate(CeedQFunction qf, CeedSize flops) {
  CeedCheck(flops >= 0, CeedQFunctionReturnCeed(qf), CEED_ERROR_INCOMPATIBLE, "Must set non-negative FLOPs estimate");
  qf->user_flop_estimate = flops;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a `CeedQFunction`

  @param[in] qf     `CeedQFunction` to view
  @param[in] stream Stream to write; typically `stdout` or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionView(CeedQFunction qf, FILE *stream) {
  const char *kernel_name;

  CeedCall(CeedQFunctionGetKernelName(qf, &kernel_name));
  fprintf(stream, "%sCeedQFunction - %s\n", qf->is_gallery ? "Gallery " : "User ", qf->is_gallery ? qf->gallery_name : kernel_name);

  fprintf(stream, "  %" CeedInt_FMT " input field%s:\n", qf->num_input_fields, qf->num_input_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < qf->num_input_fields; i++) {
    CeedCall(CeedQFunctionFieldView(qf->input_fields[i], i, 1, stream));
  }

  fprintf(stream, "  %" CeedInt_FMT " output field%s:\n", qf->num_output_fields, qf->num_output_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < qf->num_output_fields; i++) {
    CeedCall(CeedQFunctionFieldView(qf->output_fields[i], i, 0, stream));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `Ceed` associated with a `CeedQFunction`

  @param[in]  qf   `CeedQFunction`
  @param[out] ceed Variable to store`Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedQFunctionGetCeed(CeedQFunction qf, Ceed *ceed) {
  *ceed = NULL;
  CeedCall(CeedReferenceCopy(CeedQFunctionReturnCeed(qf), ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedQFunction`

  @param[in]  qf   `CeedQFunction`

  @return `Ceed` associated with the `qf`

  @ref Advanced
**/
Ceed CeedQFunctionReturnCeed(CeedQFunction qf) { return qf->ceed; }

/**
  @brief Apply the action of a `CeedQFunction`

  Note: Calling this function asserts that setup is complete and sets the `CeedQFunction` as immutable.

  @param[in]  qf `CeedQFunction`
  @param[in]  Q  Number of quadrature points
  @param[in]  u  Array of input `CeedVector`
  @param[out] v  Array of output `CeedVector`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionApply(CeedQFunction qf, CeedInt Q, CeedVector *u, CeedVector *v) {
  CeedInt vec_length;

  CeedCheck(qf->Apply, CeedQFunctionReturnCeed(qf), CEED_ERROR_UNSUPPORTED, "Backend does not support CeedQFunctionApply");
  CeedCall(CeedQFunctionGetVectorLength(qf, &vec_length));
  CeedCheck(Q % vec_length == 0, CeedQFunctionReturnCeed(qf), CEED_ERROR_DIMENSION,
            "Number of quadrature points %" CeedInt_FMT " must be a multiple of %" CeedInt_FMT, Q, qf->vec_length);
  CeedCall(CeedQFunctionSetImmutable(qf));
  CeedCall(qf->Apply(qf, Q, u, v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `CeedQFunction`

  @param[in,out] qf `CeedQFunction` to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedQFunctionDestroy(CeedQFunction *qf) {
  if (!*qf || --(*qf)->ref_count > 0) {
    *qf = NULL;
    return CEED_ERROR_SUCCESS;
  }
  // Backend destroy
  if ((*qf)->Destroy) {
    CeedCall((*qf)->Destroy(*qf));
  }
  // Free fields
  for (CeedInt i = 0; i < (*qf)->num_input_fields; i++) {
    CeedCall(CeedFree(&(*(*qf)->input_fields[i]).field_name));
    CeedCall(CeedFree(&(*qf)->input_fields[i]));
  }
  for (CeedInt i = 0; i < (*qf)->num_output_fields; i++) {
    CeedCall(CeedFree(&(*(*qf)->output_fields[i]).field_name));
    CeedCall(CeedFree(&(*qf)->output_fields[i]));
  }
  CeedCall(CeedFree(&(*qf)->input_fields));
  CeedCall(CeedFree(&(*qf)->output_fields));

  // User context data object
  CeedCall(CeedQFunctionContextDestroy(&(*qf)->ctx));

  CeedCall(CeedFree(&(*qf)->user_source));
  CeedCall(CeedFree(&(*qf)->source_path));
  CeedCall(CeedFree(&(*qf)->gallery_name));
  CeedCall(CeedFree(&(*qf)->kernel_name));
  CeedCall(CeedDestroy(&(*qf)->ceed));
  CeedCall(CeedFree(qf));
  return CEED_ERROR_SUCCESS;
}

/// @}
