/// @file
/// MatCEED implementation

#include <ceed.h>
#include <ceed/backend.h>
#include <mat-ceed-impl.h>
#include <mat-ceed.h>
#include <petsc-ceed-utils.h>
#include <petsc-ceed.h>
#include <petscdmplex.h>
#include <stdlib.h>
#include <string.h>

PetscClassId  MATCEED_CLASSID;
PetscLogEvent MATCEED_MULT, MATCEED_MULT_TRANSPOSE;

/**
  @brief Register MATCEED log events.

  Not collective across MPI processes.

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatCeedRegisterLogEvents() {
  static PetscBool registered = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (registered) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscClassIdRegister("MATCEED", &MATCEED_CLASSID));
  PetscCall(PetscLogEventRegister("MATCEED Mult", MATCEED_CLASSID, &MATCEED_MULT));
  PetscCall(PetscLogEventRegister("MATCEED Mult Transpose", MATCEED_CLASSID, &MATCEED_MULT_TRANSPOSE));
  registered = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Assemble the point block diagonal of a `MATCEED` into a `MATAIJ` or similar.
         The `mat_coo` preallocation is set to match the sparsity pattern of `mat_ceed`.
         The caller is responsible for assuring the global and local sizes are compatible, otherwise this function will fail.

  Collective across MPI processes.

  @param[in]      mat_ceed  `MATCEED` to assemble
  @param[in,out]  mat_coo   `MATAIJ` or similar to assemble into

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatCeedAssemblePointBlockDiagonalCOO(Mat mat_ceed, Mat mat_coo) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  // Check if COO pattern set
  {
    PetscInt index = -1;

    for (PetscInt i = 0; i < ctx->num_mats_assembled_pbd; i++) {
      if (ctx->mats_assembled_pbd[i] == mat_coo) index = i;
    }
    if (index == -1) {
      PetscInt     *rows_petsc = NULL, *cols_petsc = NULL;
      CeedInt      *rows_ceed, *cols_ceed;
      PetscCount    num_entries;
      PetscLogStage stage_amg_setup;

      // -- Assemble sparsity pattern if mat hasn't been assembled before
      PetscCall(PetscLogStageGetId("MATCEED Assembly Setup", &stage_amg_setup));
      if (stage_amg_setup == -1) {
        PetscCall(PetscLogStageRegister("MATCEED Assembly Setup", &stage_amg_setup));
      }
      PetscCall(PetscLogStagePush(stage_amg_setup));
      PetscCallCeed(ctx->ceed, CeedOperatorLinearAssemblePointBlockDiagonalSymbolic(ctx->op_mult, &num_entries, &rows_ceed, &cols_ceed));
      PetscCall(IntArrayCeedToPetsc(num_entries, &rows_ceed, &rows_petsc));
      PetscCall(IntArrayCeedToPetsc(num_entries, &cols_ceed, &cols_petsc));
      PetscCall(MatSetPreallocationCOOLocal(mat_coo, num_entries, rows_petsc, cols_petsc));
      free(rows_petsc);
      free(cols_petsc);
      if (!ctx->coo_values_pbd) PetscCallCeed(ctx->ceed, CeedVectorCreate(ctx->ceed, num_entries, &ctx->coo_values_pbd));
      PetscCall(PetscRealloc(++ctx->num_mats_assembled_pbd * sizeof(Mat), &ctx->mats_assembled_pbd));
      ctx->mats_assembled_pbd[ctx->num_mats_assembled_pbd - 1] = mat_coo;
      PetscCall(PetscLogStagePop());
    }
  }

  // Assemble mat_ceed
  PetscCall(MatAssemblyBegin(mat_coo, MAT_FINAL_ASSEMBLY));
  {
    const CeedScalar *values;
    MatType           mat_type;
    CeedMemType       mem_type = CEED_MEM_HOST;
    PetscBool         is_spd, is_spd_known;

    PetscCall(MatGetType(mat_coo, &mat_type));
    if (strstr(mat_type, "cusparse")) mem_type = CEED_MEM_DEVICE;
    else if (strstr(mat_type, "kokkos")) mem_type = CEED_MEM_DEVICE;
    else mem_type = CEED_MEM_HOST;

    PetscCallCeed(ctx->ceed, CeedOperatorLinearAssemblePointBlockDiagonal(ctx->op_mult, ctx->coo_values_pbd, CEED_REQUEST_IMMEDIATE));
    PetscCallCeed(ctx->ceed, CeedVectorGetArrayRead(ctx->coo_values_pbd, mem_type, &values));
    PetscCall(MatSetValuesCOO(mat_coo, values, INSERT_VALUES));
    PetscCall(MatIsSPDKnown(mat_ceed, &is_spd_known, &is_spd));
    if (is_spd_known) PetscCall(MatSetOption(mat_coo, MAT_SPD, is_spd));
    PetscCallCeed(ctx->ceed, CeedVectorRestoreArrayRead(ctx->coo_values_pbd, &values));
  }
  PetscCall(MatAssemblyEnd(mat_coo, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Assemble inner `Mat` for diagonal `PC` operations

  Collective across MPI processes.

  @param[in]   mat_ceed      `MATCEED` to invert
  @param[in]   use_ceed_pbd  Boolean flag to use libCEED PBD assembly
  @param[out]  mat_inner     Inner `Mat` for diagonal operations

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatCeedAssembleInnerBlockDiagonalMat(Mat mat_ceed, PetscBool use_ceed_pbd, Mat *mat_inner) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));
  if (use_ceed_pbd) {
    // Check if COO pattern set
    if (!ctx->mat_assembled_pbd_internal) PetscCall(MatCeedCreateMatCOO(mat_ceed, &ctx->mat_assembled_pbd_internal));

    // Assemble mat_assembled_full_internal
    PetscCall(MatCeedAssemblePointBlockDiagonalCOO(mat_ceed, ctx->mat_assembled_pbd_internal));
    if (mat_inner) *mat_inner = ctx->mat_assembled_pbd_internal;
  } else {
    // Check if COO pattern set
    if (!ctx->mat_assembled_full_internal) PetscCall(MatCeedCreateMatCOO(mat_ceed, &ctx->mat_assembled_full_internal));

    // Assemble mat_assembled_full_internal
    PetscCall(MatCeedAssembleCOO(mat_ceed, ctx->mat_assembled_full_internal));
    if (mat_inner) *mat_inner = ctx->mat_assembled_full_internal;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get `MATCEED` diagonal block for Jacobi.

  Collective across MPI processes.

  @param[in]   mat_ceed   `MATCEED` to invert
  @param[out]  mat_block  The diagonal block matrix

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatGetDiagonalBlock_Ceed(Mat mat_ceed, Mat *mat_block) {
  Mat            mat_inner = NULL;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  // Assemble inner mat if needed
  PetscCall(MatCeedAssembleInnerBlockDiagonalMat(mat_ceed, ctx->is_ceed_pbd_valid, &mat_inner));

  // Get block diagonal
  PetscCall(MatGetDiagonalBlock(mat_inner, mat_block));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Invert `MATCEED` diagonal block for Jacobi.

  Collective across MPI processes.

  @param[in]   mat_ceed  `MATCEED` to invert
  @param[out]  values    The block inverses in column major order

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatInvertBlockDiagonal_Ceed(Mat mat_ceed, const PetscScalar **values) {
  Mat            mat_inner = NULL;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  // Assemble inner mat if needed
  PetscCall(MatCeedAssembleInnerBlockDiagonalMat(mat_ceed, ctx->is_ceed_pbd_valid, &mat_inner));

  // Invert PB diagonal
  PetscCall(MatInvertBlockDiagonal(mat_inner, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Invert `MATCEED` variable diagonal block for Jacobi.

  Collective across MPI processes.

  @param[in]   mat_ceed     `MATCEED` to invert
  @param[in]   num_blocks   The number of blocks on the process
  @param[in]   block_sizes  The size of each block on the process
  @param[out]  values       The block inverses in column major order

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatInvertVariableBlockDiagonal_Ceed(Mat mat_ceed, PetscInt num_blocks, const PetscInt *block_sizes, PetscScalar *values) {
  Mat            mat_inner = NULL;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  // Assemble inner mat if needed
  PetscCall(MatCeedAssembleInnerBlockDiagonalMat(mat_ceed, ctx->is_ceed_vpbd_valid, &mat_inner));

  // Invert PB diagonal
  PetscCall(MatInvertVariableBlockDiagonal(mat_inner, num_blocks, block_sizes, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief View `MATCEED`.

  Collective across MPI processes.

  @param[in]   mat_ceed  `MATCEED` to view
  @param[in]   viewer    The visualization context

  @return An error code: 0 - success, otherwise - failure
**/
static PetscErrorCode MatView_Ceed(Mat mat_ceed, PetscViewer viewer) {
  PetscBool         is_ascii;
  PetscViewerFormat format;
  PetscMPIInt       size;
  MatCeedContext    ctx;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(MatShellGetContext(mat_ceed, &ctx));
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mat_ceed), &viewer));

  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat_ceed), &size));
  if (size == 1 && format == PETSC_VIEWER_LOAD_BALANCE) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  {
    FILE *file;

    PetscCall(PetscViewerASCIIPrintf(viewer, "MatCEED:\n  Default COO MatType:%s\n", ctx->coo_mat_type));
    PetscCall(PetscViewerASCIIGetPointer(viewer, &file));
    PetscCall(PetscViewerASCIIPrintf(viewer, " libCEED Operator:\n"));
    PetscCallCeed(ctx->ceed, CeedOperatorView(ctx->op_mult, file));
    if (ctx->op_mult_transpose) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  libCEED Transpose Operator:\n"));
      PetscCallCeed(ctx->ceed, CeedOperatorView(ctx->op_mult_transpose, file));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// MatCeed
// -----------------------------------------------------------------------------

/**
  @brief Create PETSc `Mat` from libCEED operators.

  Collective across MPI processes.

  @param[in]   dm_x                      Input `DM`
  @param[in]   dm_y                      Output `DM`
  @param[in]   op_mult                   `CeedOperator` for forward evaluation
  @param[in]   op_mult_transpose         `CeedOperator` for transpose evaluation
  @param[out]  mat                        New MatCeed

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedCreate(DM dm_x, DM dm_y, CeedOperator op_mult, CeedOperator op_mult_transpose, Mat *mat) {
  PetscInt       X_l_size, X_g_size, Y_l_size, Y_g_size;
  VecType        vec_type;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatCeedRegisterLogEvents());

  // Collect context data
  PetscCall(DMGetVecType(dm_x, &vec_type));
  {
    Vec X;

    PetscCall(DMGetGlobalVector(dm_x, &X));
    PetscCall(VecGetSize(X, &X_g_size));
    PetscCall(VecGetLocalSize(X, &X_l_size));
    PetscCall(DMRestoreGlobalVector(dm_x, &X));
  }
  if (dm_y) {
    Vec Y;

    PetscCall(DMGetGlobalVector(dm_y, &Y));
    PetscCall(VecGetSize(Y, &Y_g_size));
    PetscCall(VecGetLocalSize(Y, &Y_l_size));
    PetscCall(DMRestoreGlobalVector(dm_y, &Y));
  } else {
    dm_y     = dm_x;
    Y_g_size = X_g_size;
    Y_l_size = X_l_size;
  }

  // Create context
  {
    Vec X_loc, Y_loc_transpose = NULL;

    PetscCall(DMCreateLocalVector(dm_x, &X_loc));
    PetscCall(VecZeroEntries(X_loc));
    if (op_mult_transpose) {
      PetscCall(DMCreateLocalVector(dm_y, &Y_loc_transpose));
      PetscCall(VecZeroEntries(Y_loc_transpose));
    }
    PetscCall(MatCeedContextCreate(dm_x, dm_y, X_loc, Y_loc_transpose, op_mult, op_mult_transpose, MATCEED_MULT, MATCEED_MULT_TRANSPOSE, &ctx));
    PetscCall(VecDestroy(&X_loc));
    PetscCall(VecDestroy(&Y_loc_transpose));
  }

  // Create mat
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)dm_x), Y_l_size, X_l_size, Y_g_size, X_g_size, ctx, mat));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*mat, MATCEED));
  // -- Set block and variable block sizes
  if (dm_x == dm_y) {
    MatType dm_mat_type, dm_mat_type_copy;
    Mat     temp_mat;

    PetscCall(DMGetMatType(dm_x, &dm_mat_type));
    PetscCall(PetscStrallocpy(dm_mat_type, (char **)&dm_mat_type_copy));
    PetscCall(DMSetMatType(dm_x, MATAIJ));
    PetscCall(DMCreateMatrix(dm_x, &temp_mat));
    PetscCall(DMSetMatType(dm_x, dm_mat_type_copy));
    PetscCall(PetscFree(dm_mat_type_copy));

    {
      PetscInt        block_size, num_blocks, max_vblock_size = PETSC_INT_MAX;
      const PetscInt *vblock_sizes;

      // -- Get block sizes
      PetscCall(MatGetBlockSize(temp_mat, &block_size));
      PetscCall(MatGetVariableBlockSizes(temp_mat, &num_blocks, &vblock_sizes));
      {
        PetscInt local_min_max[2] = {0}, global_min_max[2] = {0, PETSC_INT_MAX};

        for (PetscInt i = 0; i < num_blocks; i++) local_min_max[1] = PetscMax(local_min_max[1], vblock_sizes[i]);
        PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm_x), local_min_max, global_min_max));
        max_vblock_size = global_min_max[1];
      }

      // -- Copy block sizes
      if (block_size > 1) PetscCall(MatSetBlockSize(*mat, block_size));
      if (num_blocks) PetscCall(MatSetVariableBlockSizes(*mat, num_blocks, (PetscInt *)vblock_sizes));

      // -- Check libCEED compatibility
      {
        bool is_composite;

        ctx->is_ceed_pbd_valid  = PETSC_TRUE;
        ctx->is_ceed_vpbd_valid = PETSC_TRUE;
        PetscCallCeed(ctx->ceed, CeedOperatorIsComposite(op_mult, &is_composite));
        if (is_composite) {
          CeedInt       num_sub_operators;
          CeedOperator *sub_operators;

          PetscCallCeed(ctx->ceed, CeedCompositeOperatorGetNumSub(op_mult, &num_sub_operators));
          PetscCallCeed(ctx->ceed, CeedCompositeOperatorGetSubList(op_mult, &sub_operators));
          for (CeedInt i = 0; i < num_sub_operators; i++) {
            CeedInt                  num_bases, num_comp;
            CeedBasis               *active_bases;
            CeedOperatorAssemblyData assembly_data;

            PetscCallCeed(ctx->ceed, CeedOperatorGetOperatorAssemblyData(sub_operators[i], &assembly_data));
            PetscCallCeed(ctx->ceed, CeedOperatorAssemblyDataGetBases(assembly_data, &num_bases, &active_bases, NULL, NULL, NULL, NULL));
            PetscCallCeed(ctx->ceed, CeedBasisGetNumComponents(active_bases[0], &num_comp));
            if (num_bases > 1) {
              ctx->is_ceed_pbd_valid  = PETSC_FALSE;
              ctx->is_ceed_vpbd_valid = PETSC_FALSE;
            }
            if (num_comp != block_size) ctx->is_ceed_pbd_valid = PETSC_FALSE;
            if (num_comp < max_vblock_size) ctx->is_ceed_vpbd_valid = PETSC_FALSE;
          }
        } else {
          // LCOV_EXCL_START
          CeedInt                  num_bases, num_comp;
          CeedBasis               *active_bases;
          CeedOperatorAssemblyData assembly_data;

          PetscCallCeed(ctx->ceed, CeedOperatorGetOperatorAssemblyData(op_mult, &assembly_data));
          PetscCallCeed(ctx->ceed, CeedOperatorAssemblyDataGetBases(assembly_data, &num_bases, &active_bases, NULL, NULL, NULL, NULL));
          PetscCallCeed(ctx->ceed, CeedBasisGetNumComponents(active_bases[0], &num_comp));
          if (num_bases > 1) {
            ctx->is_ceed_pbd_valid  = PETSC_FALSE;
            ctx->is_ceed_vpbd_valid = PETSC_FALSE;
          }
          if (num_comp != block_size) ctx->is_ceed_pbd_valid = PETSC_FALSE;
          if (num_comp < max_vblock_size) ctx->is_ceed_vpbd_valid = PETSC_FALSE;
          // LCOV_EXCL_STOP
        }
        {
          PetscInt local_is_valid[2], global_is_valid[2];

          local_is_valid[0] = local_is_valid[1] = ctx->is_ceed_pbd_valid;
          PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm_x), local_is_valid, global_is_valid));
          ctx->is_ceed_pbd_valid = global_is_valid[0];
          local_is_valid[0] = local_is_valid[1] = ctx->is_ceed_vpbd_valid;
          PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)dm_x), local_is_valid, global_is_valid));
          ctx->is_ceed_vpbd_valid = global_is_valid[0];
        }
      }
    }
    PetscCall(MatDestroy(&temp_mat));
  }
  // -- Set internal mat type
  {
    VecType vec_type;
    MatType coo_mat_type;

    PetscCall(VecGetType(ctx->X_loc, &vec_type));
    if (strstr(vec_type, VECCUDA)) coo_mat_type = MATAIJCUSPARSE;
    else if (strstr(vec_type, VECKOKKOS)) coo_mat_type = MATAIJKOKKOS;
    else coo_mat_type = MATAIJ;
    PetscCall(PetscStrallocpy(coo_mat_type, &ctx->coo_mat_type));
  }
  // -- Set mat operations
  PetscCall(MatShellSetContextDestroy(*mat, (PetscErrorCode(*)(void *))MatCeedContextDestroy));
  PetscCall(MatShellSetOperation(*mat, MATOP_VIEW, (void (*)(void))MatView_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_MULT, (void (*)(void))MatMult_Ceed));
  if (op_mult_transpose) PetscCall(MatShellSetOperation(*mat, MATOP_MULT_TRANSPOSE, (void (*)(void))MatMultTranspose_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_GET_DIAGONAL_BLOCK, (void (*)(void))MatGetDiagonalBlock_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_INVERT_BLOCK_DIAGONAL, (void (*)(void))MatInvertBlockDiagonal_Ceed));
  PetscCall(MatShellSetOperation(*mat, MATOP_INVERT_VBLOCK_DIAGONAL, (void (*)(void))MatInvertVariableBlockDiagonal_Ceed));
  PetscCall(MatShellSetVecType(*mat, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Copy `MATCEED` into a compatible `Mat` with type `MatShell` or `MATCEED`.

  Collective across MPI processes.

  @param[in]   mat_ceed   `MATCEED` to copy from
  @param[out]  mat_other  `MatShell` or `MATCEED` to copy into

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedCopy(Mat mat_ceed, Mat mat_other) {
  PetscFunctionBeginUser;
  PetscCall(MatCeedRegisterLogEvents());

  // Check type compatibility
  {
    PetscBool is_matceed = PETSC_FALSE, is_matshell = PETSC_FALSE;
    MatType   mat_type_ceed, mat_type_other;

    PetscCall(MatGetType(mat_ceed, &mat_type_ceed));
    PetscCall(PetscStrcmp(mat_type_ceed, MATCEED, &is_matceed));
    PetscCheck(is_matceed, PETSC_COMM_SELF, PETSC_ERR_LIB, "mat_ceed must have type " MATCEED);
    PetscCall(MatGetType(mat_other, &mat_type_other));
    PetscCall(PetscStrcmp(mat_type_other, MATCEED, &is_matceed));
    PetscCall(PetscStrcmp(mat_type_other, MATSHELL, &is_matceed));
    PetscCheck(is_matceed || is_matshell, PETSC_COMM_SELF, PETSC_ERR_LIB, "mat_other must have type " MATCEED " or " MATSHELL);
  }

  // Check dimension compatibility
  {
    PetscInt X_l_ceed_size, X_g_ceed_size, Y_l_ceed_size, Y_g_ceed_size, X_l_other_size, X_g_other_size, Y_l_other_size, Y_g_other_size;

    PetscCall(MatGetSize(mat_ceed, &Y_g_ceed_size, &X_g_ceed_size));
    PetscCall(MatGetLocalSize(mat_ceed, &Y_l_ceed_size, &X_l_ceed_size));
    PetscCall(MatGetSize(mat_ceed, &Y_g_other_size, &X_g_other_size));
    PetscCall(MatGetLocalSize(mat_ceed, &Y_l_other_size, &X_l_other_size));
    PetscCheck((Y_g_ceed_size == Y_g_other_size) && (X_g_ceed_size == X_g_other_size) && (Y_l_ceed_size == Y_l_other_size) &&
                   (X_l_ceed_size == X_l_other_size),
               PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
               "mat_ceed and mat_other must have compatible sizes; found mat_ceed (Global: %" PetscInt_FMT ", %" PetscInt_FMT
               "; Local: %" PetscInt_FMT ", %" PetscInt_FMT ") mat_other (Global: %" PetscInt_FMT ", %" PetscInt_FMT "; Local: %" PetscInt_FMT
               ", %" PetscInt_FMT ")",
               Y_g_ceed_size, X_g_ceed_size, Y_l_ceed_size, X_l_ceed_size, Y_g_other_size, X_g_other_size, Y_l_other_size, X_l_other_size);
  }

  // Convert
  {
    VecType        vec_type;
    MatCeedContext ctx;

    PetscCall(PetscObjectChangeTypeName((PetscObject)mat_other, MATCEED));
    PetscCall(MatShellGetContext(mat_ceed, &ctx));
    PetscCall(MatCeedContextReference(ctx));
    PetscCall(MatShellSetContext(mat_other, ctx));
    PetscCall(MatShellSetContextDestroy(mat_other, (PetscErrorCode(*)(void *))MatCeedContextDestroy));
    PetscCall(MatShellSetOperation(mat_other, MATOP_VIEW, (void (*)(void))MatView_Ceed));
    PetscCall(MatShellSetOperation(mat_other, MATOP_MULT, (void (*)(void))MatMult_Ceed));
    if (ctx->op_mult_transpose) PetscCall(MatShellSetOperation(mat_other, MATOP_MULT_TRANSPOSE, (void (*)(void))MatMultTranspose_Ceed));
    PetscCall(MatShellSetOperation(mat_other, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_Ceed));
    PetscCall(MatShellSetOperation(mat_other, MATOP_GET_DIAGONAL_BLOCK, (void (*)(void))MatGetDiagonalBlock_Ceed));
    PetscCall(MatShellSetOperation(mat_other, MATOP_INVERT_BLOCK_DIAGONAL, (void (*)(void))MatInvertBlockDiagonal_Ceed));
    PetscCall(MatShellSetOperation(mat_other, MATOP_INVERT_VBLOCK_DIAGONAL, (void (*)(void))MatInvertVariableBlockDiagonal_Ceed));
    {
      PetscInt block_size;

      PetscCall(MatGetBlockSize(mat_ceed, &block_size));
      if (block_size > 1) PetscCall(MatSetBlockSize(mat_other, block_size));
    }
    {
      PetscInt        num_blocks;
      const PetscInt *block_sizes;

      PetscCall(MatGetVariableBlockSizes(mat_ceed, &num_blocks, &block_sizes));
      if (num_blocks) PetscCall(MatSetVariableBlockSizes(mat_other, num_blocks, (PetscInt *)block_sizes));
    }
    PetscCall(DMGetVecType(ctx->dm_x, &vec_type));
    PetscCall(MatShellSetVecType(mat_other, vec_type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup a `Mat` with the same COO pattern as a `MatCEED`.

  Collective across MPI processes.

  @param[in]   mat_ceed  `MATCEED`
  @param[out]  mat_coo   Sparse `Mat` with same COO pattern

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedCreateMatCOO(Mat mat_ceed, Mat *mat_coo) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  PetscCheck(ctx->dm_x == ctx->dm_y, PetscObjectComm((PetscObject)mat_ceed), PETSC_ERR_SUP, "COO assembly only supported for MATCEED on a single DM");

  // Check cl mat type
  {
    PetscBool is_coo_mat_type_cl = PETSC_FALSE;
    char      coo_mat_type_cl[64];

    // Check for specific CL coo mat type for this Mat
    {
      const char *mat_ceed_prefix = NULL;

      PetscCall(MatGetOptionsPrefix(mat_ceed, &mat_ceed_prefix));
      PetscOptionsBegin(PetscObjectComm((PetscObject)mat_ceed), mat_ceed_prefix, "", NULL);
      PetscCall(PetscOptionsFList("-ceed_coo_mat_type", "Default MATCEED COO assembly MatType", NULL, MatList, coo_mat_type_cl, coo_mat_type_cl,
                                  sizeof(coo_mat_type_cl), &is_coo_mat_type_cl));
      PetscOptionsEnd();
      if (is_coo_mat_type_cl) {
        PetscCall(PetscFree(ctx->coo_mat_type));
        PetscCall(PetscStrallocpy(coo_mat_type_cl, &ctx->coo_mat_type));
      }
    }
  }

  // Create sparse matrix
  {
    MatType dm_mat_type, dm_mat_type_copy;

    PetscCall(DMGetMatType(ctx->dm_x, &dm_mat_type));
    PetscCall(PetscStrallocpy(dm_mat_type, (char **)&dm_mat_type_copy));
    PetscCall(DMSetMatType(ctx->dm_x, ctx->coo_mat_type));
    PetscCall(DMCreateMatrix(ctx->dm_x, mat_coo));
    PetscCall(DMSetMatType(ctx->dm_x, dm_mat_type_copy));
    PetscCall(PetscFree(dm_mat_type_copy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Setup the COO preallocation `MATCEED` into a `MATAIJ` or similar.
         The caller is responsible for assuring the global and local sizes are compatible, otherwise this function will fail.

  Collective across MPI processes.

  @param[in]      mat_ceed  `MATCEED` to assemble
  @param[in,out]  mat_coo   `MATAIJ` or similar to assemble into

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetPreallocationCOO(Mat mat_ceed, Mat mat_coo) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  {
    PetscInt     *rows_petsc = NULL, *cols_petsc = NULL;
    CeedInt      *rows_ceed, *cols_ceed;
    PetscCount    num_entries;
    PetscLogStage stage_amg_setup;

    // -- Assemble sparsity pattern if mat hasn't been assembled before
    PetscCall(PetscLogStageGetId("MATCEED Assembly Setup", &stage_amg_setup));
    if (stage_amg_setup == -1) {
      PetscCall(PetscLogStageRegister("MATCEED Assembly Setup", &stage_amg_setup));
    }
    PetscCall(PetscLogStagePush(stage_amg_setup));
    PetscCallCeed(ctx->ceed, CeedOperatorLinearAssembleSymbolic(ctx->op_mult, &num_entries, &rows_ceed, &cols_ceed));
    PetscCall(IntArrayCeedToPetsc(num_entries, &rows_ceed, &rows_petsc));
    PetscCall(IntArrayCeedToPetsc(num_entries, &cols_ceed, &cols_petsc));
    PetscCall(MatSetPreallocationCOOLocal(mat_coo, num_entries, rows_petsc, cols_petsc));
    free(rows_petsc);
    free(cols_petsc);
    if (!ctx->coo_values_full) PetscCallCeed(ctx->ceed, CeedVectorCreate(ctx->ceed, num_entries, &ctx->coo_values_full));
    PetscCall(PetscRealloc(++ctx->num_mats_assembled_full * sizeof(Mat), &ctx->mats_assembled_full));
    ctx->mats_assembled_full[ctx->num_mats_assembled_full - 1] = mat_coo;
    PetscCall(PetscLogStagePop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Assemble a `MATCEED` into a `MATAIJ` or similar.
         The `mat_coo` preallocation is set to match the sparsity pattern of `mat_ceed`.
         The caller is responsible for assuring the global and local sizes are compatible, otherwise this function will fail.

  Collective across MPI processes.

  @param[in]      mat_ceed  `MATCEED` to assemble
  @param[in,out]  mat_coo   `MATAIJ` or similar to assemble into

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedAssembleCOO(Mat mat_ceed, Mat mat_coo) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat_ceed, &ctx));

  // Set COO pattern if needed
  {
    CeedInt index = -1;

    for (PetscInt i = 0; i < ctx->num_mats_assembled_full; i++) {
      if (ctx->mats_assembled_full[i] == mat_coo) index = i;
    }
    if (index == -1) PetscCall(MatCeedSetPreallocationCOO(mat_ceed, mat_coo));
  }

  // Assemble mat_ceed
  PetscCall(MatAssemblyBegin(mat_coo, MAT_FINAL_ASSEMBLY));
  {
    const CeedScalar *values;
    MatType           mat_type;
    CeedMemType       mem_type = CEED_MEM_HOST;
    PetscBool         is_spd, is_spd_known;

    PetscCall(MatGetType(mat_coo, &mat_type));
    if (strstr(mat_type, "cusparse")) mem_type = CEED_MEM_DEVICE;
    else if (strstr(mat_type, "kokkos")) mem_type = CEED_MEM_DEVICE;
    else mem_type = CEED_MEM_HOST;

    PetscCallCeed(ctx->ceed, CeedOperatorLinearAssemble(ctx->op_mult, ctx->coo_values_full));
    PetscCallCeed(ctx->ceed, CeedVectorGetArrayRead(ctx->coo_values_full, mem_type, &values));
    PetscCall(MatSetValuesCOO(mat_coo, values, INSERT_VALUES));
    PetscCall(MatIsSPDKnown(mat_ceed, &is_spd_known, &is_spd));
    if (is_spd_known) PetscCall(MatSetOption(mat_coo, MAT_SPD, is_spd));
    PetscCallCeed(ctx->ceed, CeedVectorRestoreArrayRead(ctx->coo_values_full, &values));
  }
  PetscCall(MatAssemblyEnd(mat_coo, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set the current value of a context field for a `MatCEED`.

  Not collective across MPI processes.

  @param[in,out]  mat    `MatCEED`
  @param[in]      name   Name of the context field
  @param[in]      value  New context field value

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetContextDouble(Mat mat, const char *name, double value) {
  PetscBool      was_updated = PETSC_FALSE;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  {
    CeedContextFieldLabel label = NULL;

    PetscCallCeed(ctx->ceed, CeedOperatorGetContextFieldLabel(ctx->op_mult, name, &label));
    if (label) {
      PetscCallCeed(ctx->ceed, CeedOperatorSetContextDouble(ctx->op_mult, label, &value));
      was_updated = PETSC_TRUE;
    }
    if (ctx->op_mult_transpose) {
      label = NULL;
      PetscCallCeed(ctx->ceed, CeedOperatorGetContextFieldLabel(ctx->op_mult_transpose, name, &label));
      if (label) {
        PetscCallCeed(ctx->ceed, CeedOperatorSetContextDouble(ctx->op_mult_transpose, label, &value));
        was_updated = PETSC_TRUE;
      }
    }
  }
  if (was_updated) {
    PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set the current `PetscReal` value of a context field for a `MatCEED`.

  Not collective across MPI processes.

  @param[in,out]  mat    `MatCEED`
  @param[in]      name   Name of the context field
  @param[in]      value  New context field value

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetContextReal(Mat mat, const char *name, PetscReal value) {
  double value_double = value;

  PetscFunctionBeginUser;
  PetscCall(MatCeedSetContextDouble(mat, name, value_double));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the current value of a context field for a `MatCEED`.

  Not collective across MPI processes.

  @param[in]   mat    `MatCEED`
  @param[in]   name   Name of the context field
  @param[out]  value  Current context field value

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetContextDouble(Mat mat, const char *name, double *value) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  {
    CeedContextFieldLabel label = NULL;
    CeedOperator          op    = ctx->op_mult;

    PetscCallCeed(ctx->ceed, CeedOperatorGetContextFieldLabel(op, name, &label));
    if (!label && ctx->op_mult_transpose) {
      op = ctx->op_mult_transpose;
      PetscCallCeed(ctx->ceed, CeedOperatorGetContextFieldLabel(op, name, &label));
    }
    if (label) {
      PetscSizeT    num_values;
      const double *values_ceed;

      PetscCallCeed(ctx->ceed, CeedOperatorGetContextDoubleRead(op, label, &num_values, &values_ceed));
      *value = values_ceed[0];
      PetscCallCeed(ctx->ceed, CeedOperatorRestoreContextDoubleRead(op, label, &values_ceed));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get the current `PetscReal` value of a context field for a `MatCEED`.

  Not collective across MPI processes.

  @param[in]   mat    `MatCEED`
  @param[in]   name   Name of the context field
  @param[out]  value  Current context field value

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetContextReal(Mat mat, const char *name, PetscReal *value) {
  double value_double;

  PetscFunctionBeginUser;
  PetscCall(MatCeedGetContextDouble(mat, name, &value_double));
  *value = value_double;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set user context for a `MATCEED`.

  Collective across MPI processes.

  @param[in,out]  mat  `MATCEED`
  @param[in]      f    The context destroy function, or NULL
  @param[in]      ctx  User context, or NULL to unset

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetContext(Mat mat, PetscErrorCode (*f)(void *), void *ctx) {
  PetscContainer user_ctx = NULL;

  PetscFunctionBeginUser;
  if (ctx) {
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)mat), &user_ctx));
    PetscCall(PetscContainerSetPointer(user_ctx, ctx));
    PetscCall(PetscContainerSetUserDestroy(user_ctx, f));
  }
  PetscCall(PetscObjectCompose((PetscObject)mat, "MatCeed user context", (PetscObject)user_ctx));
  PetscCall(PetscContainerDestroy(&user_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Retrieve the user context for a `MATCEED`.

  Collective across MPI processes.

  @param[in,out]  mat  `MATCEED`
  @param[in]      ctx  User context

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetContext(Mat mat, void *ctx) {
  PetscContainer user_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)mat, "MatCeed user context", (PetscObject *)&user_ctx));
  if (user_ctx) PetscCall(PetscContainerGetPointer(user_ctx, (void **)ctx));
  else *(void **)ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/**
  @brief Set a user defined matrix operation for a `MATCEED` matrix.

  Within each user-defined routine, the user should call `MatCeedGetContext()` to obtain the user-defined context that was set by
`MatCeedSetContext()`.

  Collective across MPI processes.

  @param[in,out]  mat  `MATCEED`
  @param[in]      op   Name of the `MatOperation`
  @param[in]      g    Function that provides the operation

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetOperation(Mat mat, MatOperation op, void (*g)(void)) {
  PetscFunctionBeginUser;
  PetscCall(MatShellSetOperation(mat, op, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Sets the default COO matrix type as a string from the `MATCEED`.

  Collective across MPI processes.

  @param[in,out]  mat   `MATCEED`
  @param[in]      type  COO `MatType` to set

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetCOOMatType(Mat mat, MatType type) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  // Check if same
  {
    size_t    len_old, len_new;
    PetscBool is_same = PETSC_FALSE;

    PetscCall(PetscStrlen(ctx->coo_mat_type, &len_old));
    PetscCall(PetscStrlen(type, &len_new));
    if (len_old == len_new) PetscCall(PetscStrncmp(ctx->coo_mat_type, type, len_old, &is_same));
    if (is_same) PetscFunctionReturn(PETSC_SUCCESS);
  }
  // Clean up old mats in different format
  // LCOV_EXCL_START
  if (ctx->mat_assembled_full_internal) {
    for (PetscInt i = 0; i < ctx->num_mats_assembled_full; i++) {
      if (ctx->mats_assembled_full[i] == ctx->mat_assembled_full_internal) {
        for (PetscInt j = i + 1; j < ctx->num_mats_assembled_full; j++) {
          ctx->mats_assembled_full[j - 1] = ctx->mats_assembled_full[j];
        }
        ctx->num_mats_assembled_full--;
        // Note: we'll realloc this array again, so no need to shrink the allocation
        PetscCall(MatDestroy(&ctx->mat_assembled_full_internal));
      }
    }
  }
  if (ctx->mat_assembled_pbd_internal) {
    for (PetscInt i = 0; i < ctx->num_mats_assembled_pbd; i++) {
      if (ctx->mats_assembled_pbd[i] == ctx->mat_assembled_pbd_internal) {
        for (PetscInt j = i + 1; j < ctx->num_mats_assembled_pbd; j++) {
          ctx->mats_assembled_pbd[j - 1] = ctx->mats_assembled_pbd[j];
        }
        // Note: we'll realloc this array again, so no need to shrink the allocation
        ctx->num_mats_assembled_pbd--;
        PetscCall(MatDestroy(&ctx->mat_assembled_pbd_internal));
      }
    }
  }
  PetscCall(PetscFree(ctx->coo_mat_type));
  PetscCall(PetscStrallocpy(type, &ctx->coo_mat_type));
  PetscFunctionReturn(PETSC_SUCCESS);
  // LCOV_EXCL_STOP
}

/**
  @brief Gets the default COO matrix type as a string from the `MATCEED`.

  Collective across MPI processes.

  @param[in,out]  mat   `MATCEED`
  @param[in]      type  COO `MatType`

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetCOOMatType(Mat mat, MatType *type) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  *type = ctx->coo_mat_type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set input local vectors for `MATCEED` `MatMult()` and `MatMultTranspose()` operations.

  Not collective across MPI processes.

  @param[in,out]  mat              `MATCEED`
  @param[in]      X_loc            Input PETSc local vector, or NULL
  @param[in]      Y_loc_transpose  Input PETSc local vector for transpose operation, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetLocalVectors(Mat mat, Vec X_loc, Vec Y_loc_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (X_loc) {
    PetscInt len_old, len_new;

    PetscCall(VecGetSize(ctx->X_loc, &len_old));
    PetscCall(VecGetSize(X_loc, &len_new));
    PetscCheck(len_old == len_new, PETSC_COMM_SELF, PETSC_ERR_LIB, "new X_loc length %" PetscInt_FMT " should match old X_loc length %" PetscInt_FMT,
               len_new, len_old);
    PetscCall(VecReferenceCopy(X_loc, &ctx->X_loc));
  }
  if (Y_loc_transpose) {
    PetscInt len_old, len_new;

    PetscCall(VecGetSize(ctx->Y_loc_transpose, &len_old));
    PetscCall(VecGetSize(Y_loc_transpose, &len_new));
    PetscCheck(len_old == len_new, PETSC_COMM_SELF, PETSC_ERR_LIB,
               "new Y_loc_transpose length %" PetscInt_FMT " should match old Y_loc_transpose length %" PetscInt_FMT, len_new, len_old);
    PetscCall(VecReferenceCopy(Y_loc_transpose, &ctx->Y_loc_transpose));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get input local vectors for `MATCEED` `MatMult()` and `MatMultTranspose()` operations.

  Not collective across MPI processes.

  @param[in,out]  mat              `MATCEED`
  @param[out]     X_loc            Input PETSc local vector, or NULL
  @param[out]     Y_loc_transpose  Input PETSc local vector for transpose operation, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (X_loc) {
    *X_loc = NULL;
    PetscCall(VecReferenceCopy(ctx->X_loc, X_loc));
  }
  if (Y_loc_transpose) {
    *Y_loc_transpose = NULL;
    PetscCall(VecReferenceCopy(ctx->Y_loc_transpose, Y_loc_transpose));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Restore input local vectors for `MATCEED` `MatMult()` and `MatMultTranspose()` operations.

  Not collective across MPI processes.

  @param[in,out]  mat              MatCeed
  @param[out]     X_loc            Input PETSc local vector, or NULL
  @param[out]     Y_loc_transpose  Input PETSc local vector for transpose operation, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedRestoreLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose) {
  PetscFunctionBeginUser;
  if (X_loc) PetscCall(VecDestroy(X_loc));
  if (Y_loc_transpose) PetscCall(VecDestroy(Y_loc_transpose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get libCEED `CeedOperator` for `MATCEED` `MatMult()` and `MatMultTranspose()` operations.

  Not collective across MPI processes.

  @param[in,out]  mat                MatCeed
  @param[out]     op_mult            libCEED `CeedOperator` for `MatMult()`, or NULL
  @param[out]     op_mult_transpose  libCEED `CeedOperator` for `MatMultTranspose()`, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (op_mult) {
    *op_mult = NULL;
    PetscCallCeed(ctx->ceed, CeedOperatorReferenceCopy(ctx->op_mult, op_mult));
  }
  if (op_mult_transpose) {
    *op_mult_transpose = NULL;
    PetscCallCeed(ctx->ceed, CeedOperatorReferenceCopy(ctx->op_mult_transpose, op_mult_transpose));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Restore libCEED `CeedOperator` for `MATCEED` `MatMult()` and `MatMultTranspose()` operations.

  Not collective across MPI processes.

  @param[in,out]  mat                MatCeed
  @param[out]     op_mult            libCEED `CeedOperator` for `MatMult()`, or NULL
  @param[out]     op_mult_transpose  libCEED `CeedOperator` for `MatMultTranspose()`, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedRestoreCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (op_mult) PetscCallCeed(ctx->ceed, CeedOperatorDestroy(op_mult));
  if (op_mult_transpose) PetscCallCeed(ctx->ceed, CeedOperatorDestroy(op_mult_transpose));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Set `PetscLogEvent` for `MATCEED` `MatMult()` and `MatMultTranspose()` operators.

  Not collective across MPI processes.

  @param[in,out]  mat                       MatCeed
  @param[out]     log_event_mult            `PetscLogEvent` for forward evaluation, or NULL
  @param[out]     log_event_mult_transpose  `PetscLogEvent` for transpose evaluation, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedSetLogEvents(Mat mat, PetscLogEvent log_event_mult, PetscLogEvent log_event_mult_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (log_event_mult) ctx->log_event_mult = log_event_mult;
  if (log_event_mult_transpose) ctx->log_event_mult_transpose = log_event_mult_transpose;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Get `PetscLogEvent` for `MATCEED` `MatMult()` and `MatMultTranspose()` operators.

  Not collective across MPI processes.

  @param[in,out]  mat                       MatCeed
  @param[out]     log_event_mult            `PetscLogEvent` for forward evaluation, or NULL
  @param[out]     log_event_mult_transpose  `PetscLogEvent` for transpose evaluation, or NULL

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedGetLogEvents(Mat mat, PetscLogEvent *log_event_mult, PetscLogEvent *log_event_mult_transpose) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat, &ctx));
  if (log_event_mult) *log_event_mult = ctx->log_event_mult;
  if (log_event_mult_transpose) *log_event_mult_transpose = ctx->log_event_mult_transpose;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// -----------------------------------------------------------------------------
// Operator context data
// -----------------------------------------------------------------------------

/**
  @brief Setup context data for operator application.

  Collective across MPI processes.

  @param[in]   dm_x                      Input `DM`
  @param[in]   dm_y                      Output `DM`
  @param[in]   X_loc                     Input PETSc local vector, or NULL
  @param[in]   Y_loc_transpose           Input PETSc local vector for transpose operation, or NULL
  @param[in]   op_mult                   `CeedOperator` for forward evaluation
  @param[in]   op_mult_transpose         `CeedOperator` for transpose evaluation
  @param[in]   log_event_mult            `PetscLogEvent` for forward evaluation
  @param[in]   log_event_mult_transpose  `PetscLogEvent` for transpose evaluation
  @param[out]  ctx                       Context data for operator evaluation

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedContextCreate(DM dm_x, DM dm_y, Vec X_loc, Vec Y_loc_transpose, CeedOperator op_mult, CeedOperator op_mult_transpose,
                                    PetscLogEvent log_event_mult, PetscLogEvent log_event_mult_transpose, MatCeedContext *ctx) {
  CeedSize x_loc_len, y_loc_len;

  PetscFunctionBeginUser;

  // Allocate
  PetscCall(PetscNew(ctx));
  (*ctx)->ref_count = 1;

  // Logging
  (*ctx)->log_event_mult           = log_event_mult;
  (*ctx)->log_event_mult_transpose = log_event_mult_transpose;

  // PETSc objects
  PetscCall(DMReferenceCopy(dm_x, &(*ctx)->dm_x));
  PetscCall(DMReferenceCopy(dm_y, &(*ctx)->dm_y));
  if (X_loc) PetscCall(VecReferenceCopy(X_loc, &(*ctx)->X_loc));
  if (Y_loc_transpose) PetscCall(VecReferenceCopy(Y_loc_transpose, &(*ctx)->Y_loc_transpose));

  // Memtype
  {
    const PetscScalar *x;
    Vec                X;

    PetscCall(DMGetLocalVector(dm_x, &X));
    PetscCall(VecGetArrayReadAndMemType(X, &x, &(*ctx)->mem_type));
    PetscCall(VecRestoreArrayReadAndMemType(X, &x));
    PetscCall(DMRestoreLocalVector(dm_x, &X));
  }

  // libCEED objects
  PetscCheck(CeedOperatorGetCeed(op_mult, &(*ctx)->ceed) == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB,
             "retrieving Ceed context object failed");
  PetscCallCeed((*ctx)->ceed, CeedReference((*ctx)->ceed));
  PetscCallCeed((*ctx)->ceed, CeedOperatorGetActiveVectorLengths(op_mult, &x_loc_len, &y_loc_len));
  PetscCallCeed((*ctx)->ceed, CeedOperatorReferenceCopy(op_mult, &(*ctx)->op_mult));
  if (op_mult_transpose) PetscCallCeed((*ctx)->ceed, CeedOperatorReferenceCopy(op_mult_transpose, &(*ctx)->op_mult_transpose));
  PetscCallCeed((*ctx)->ceed, CeedVectorCreate((*ctx)->ceed, x_loc_len, &(*ctx)->x_loc));
  PetscCallCeed((*ctx)->ceed, CeedVectorCreate((*ctx)->ceed, y_loc_len, &(*ctx)->y_loc));

  // Flop counting
  {
    CeedSize ceed_flops_estimate = 0;

    PetscCallCeed((*ctx)->ceed, CeedOperatorGetFlopsEstimate(op_mult, &ceed_flops_estimate));
    (*ctx)->flops_mult = ceed_flops_estimate;
    if (op_mult_transpose) {
      PetscCallCeed((*ctx)->ceed, CeedOperatorGetFlopsEstimate(op_mult_transpose, &ceed_flops_estimate));
      (*ctx)->flops_mult_transpose = ceed_flops_estimate;
    }
  }

  // Check sizes
  if (x_loc_len > 0 || y_loc_len > 0) {
    CeedSize ctx_x_loc_len, ctx_y_loc_len;
    PetscInt X_loc_len, dm_x_loc_len, Y_loc_len, dm_y_loc_len;
    Vec      dm_X_loc, dm_Y_loc;

    // -- Input
    PetscCall(DMGetLocalVector(dm_x, &dm_X_loc));
    PetscCall(VecGetLocalSize(dm_X_loc, &dm_x_loc_len));
    PetscCall(DMRestoreLocalVector(dm_x, &dm_X_loc));
    PetscCallCeed((*ctx)->ceed, CeedVectorGetLength((*ctx)->x_loc, &ctx_x_loc_len));
    if (X_loc) {
      PetscCall(VecGetLocalSize(X_loc, &X_loc_len));
      PetscCheck(X_loc_len == dm_x_loc_len, PETSC_COMM_SELF, PETSC_ERR_LIB,
                 "X_loc (%" PetscInt_FMT ") must match dm_x (%" PetscInt_FMT ") dimensions", X_loc_len, dm_x_loc_len);
    }
    PetscCheck(x_loc_len == dm_x_loc_len, PETSC_COMM_SELF, PETSC_ERR_LIB, "op (%" CeedSize_FMT ") must match dm_x (%" PetscInt_FMT ") dimensions",
               x_loc_len, dm_x_loc_len);
    PetscCheck(x_loc_len == ctx_x_loc_len, PETSC_COMM_SELF, PETSC_ERR_LIB, "x_loc (%" CeedSize_FMT ") must match op dimensions (%" CeedSize_FMT ")",
               x_loc_len, ctx_x_loc_len);

    // -- Output
    PetscCall(DMGetLocalVector(dm_y, &dm_Y_loc));
    PetscCall(VecGetLocalSize(dm_Y_loc, &dm_y_loc_len));
    PetscCall(DMRestoreLocalVector(dm_y, &dm_Y_loc));
    PetscCallCeed((*ctx)->ceed, CeedVectorGetLength((*ctx)->y_loc, &ctx_y_loc_len));
    PetscCheck(ctx_y_loc_len == dm_y_loc_len, PETSC_COMM_SELF, PETSC_ERR_LIB, "op (%" CeedSize_FMT ") must match dm_y (%" PetscInt_FMT ") dimensions",
               ctx_y_loc_len, dm_y_loc_len);

    // -- Transpose
    if (Y_loc_transpose) {
      PetscCall(VecGetLocalSize(Y_loc_transpose, &Y_loc_len));
      PetscCheck(Y_loc_len == dm_y_loc_len, PETSC_COMM_SELF, PETSC_ERR_LIB,
                 "Y_loc_transpose (%" PetscInt_FMT ") must match dm_y (%" PetscInt_FMT ") dimensions", Y_loc_len, dm_y_loc_len);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Increment reference counter for `MATCEED` context.

  Not collective across MPI processes.

  @param[in,out]  ctx  Context data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedContextReference(MatCeedContext ctx) {
  PetscFunctionBeginUser;
  ctx->ref_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Copy reference for `MATCEED`.
         Note: If `ctx_copy` is non-null, it is assumed to be a valid pointer to a `MatCeedContext`.

  Not collective across MPI processes.

  @param[in]   ctx       Context data
  @param[out]  ctx_copy  Copy of pointer to context data

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedContextReferenceCopy(MatCeedContext ctx, MatCeedContext *ctx_copy) {
  PetscFunctionBeginUser;
  PetscCall(MatCeedContextReference(ctx));
  PetscCall(MatCeedContextDestroy(*ctx_copy));
  *ctx_copy = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Destroy context data for operator application.

  Collective across MPI processes.

  @param[in,out]  ctx  Context data for operator evaluation

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatCeedContextDestroy(MatCeedContext ctx) {
  PetscFunctionBeginUser;
  if (!ctx || --ctx->ref_count > 0) PetscFunctionReturn(PETSC_SUCCESS);

  // PETSc objects
  PetscCall(DMDestroy(&ctx->dm_x));
  PetscCall(DMDestroy(&ctx->dm_y));
  PetscCall(VecDestroy(&ctx->X_loc));
  PetscCall(VecDestroy(&ctx->Y_loc_transpose));
  PetscCall(MatDestroy(&ctx->mat_assembled_full_internal));
  PetscCall(MatDestroy(&ctx->mat_assembled_pbd_internal));
  PetscCall(PetscFree(ctx->coo_mat_type));
  PetscCall(PetscFree(ctx->mats_assembled_full));
  PetscCall(PetscFree(ctx->mats_assembled_pbd));

  // libCEED objects
  PetscCallCeed(ctx->ceed, CeedVectorDestroy(&ctx->x_loc));
  PetscCallCeed(ctx->ceed, CeedVectorDestroy(&ctx->y_loc));
  PetscCallCeed(ctx->ceed, CeedVectorDestroy(&ctx->coo_values_full));
  PetscCallCeed(ctx->ceed, CeedVectorDestroy(&ctx->coo_values_pbd));
  PetscCallCeed(ctx->ceed, CeedOperatorDestroy(&ctx->op_mult));
  PetscCallCeed(ctx->ceed, CeedOperatorDestroy(&ctx->op_mult_transpose));
  PetscCheck(CeedDestroy(&ctx->ceed) == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "destroying libCEED context object failed");

  // Deallocate
  ctx->is_destroyed = PETSC_TRUE;  // Flag as destroyed in case someone has stale ref
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute the diagonal of an operator via libCEED.

  Collective across MPI processes.

  @param[in]   A  `MATCEED`
  @param[out]  D  Vector holding operator diagonal

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatGetDiagonal_Ceed(Mat A, Vec D) {
  PetscMemType   mem_type;
  Vec            D_loc;
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ctx));

  // Place PETSc vector in libCEED vector
  PetscCall(DMGetLocalVector(ctx->dm_x, &D_loc));
  PetscCall(VecPetscToCeed(D_loc, &mem_type, ctx->x_loc));

  // Compute Diagonal
  PetscCallCeed(ctx->ceed, CeedOperatorLinearAssembleDiagonal(ctx->op_mult, ctx->x_loc, CEED_REQUEST_IMMEDIATE));

  // Restore PETSc vector
  PetscCall(VecCeedToPetsc(ctx->x_loc, mem_type, D_loc));

  // Local-to-Global
  PetscCall(VecZeroEntries(D));
  PetscCall(DMLocalToGlobal(ctx->dm_x, D_loc, ADD_VALUES, D));
  PetscCall(DMRestoreLocalVector(ctx->dm_x, &D_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute `A X = Y` for a `MATCEED`.

  Collective across MPI processes.

  @param[in]   A  `MATCEED`
  @param[in]   X  Input PETSc vector
  @param[out]  Y  Output PETSc vector

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatMult_Ceed(Mat A, Vec X, Vec Y) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(PetscLogEventBegin(ctx->log_event_mult, A, X, Y, 0));

  {
    PetscMemType x_mem_type, y_mem_type;
    Vec          X_loc = ctx->X_loc, Y_loc;

    // Get local vectors
    if (!ctx->X_loc) PetscCall(DMGetLocalVector(ctx->dm_x, &X_loc));
    PetscCall(DMGetLocalVector(ctx->dm_y, &Y_loc));

    // Global-to-local
    PetscCall(DMGlobalToLocal(ctx->dm_x, X, INSERT_VALUES, X_loc));

    // Setup libCEED vectors
    PetscCall(VecReadPetscToCeed(X_loc, &x_mem_type, ctx->x_loc));
    PetscCall(VecZeroEntries(Y_loc));
    PetscCall(VecPetscToCeed(Y_loc, &y_mem_type, ctx->y_loc));

    // Apply libCEED operator
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCeed(ctx->ceed, CeedOperatorApplyAdd(ctx->op_mult, ctx->x_loc, ctx->y_loc, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());

    // Restore PETSc vectors
    PetscCall(VecReadCeedToPetsc(ctx->x_loc, x_mem_type, X_loc));
    PetscCall(VecCeedToPetsc(ctx->y_loc, y_mem_type, Y_loc));

    // Local-to-global
    PetscCall(VecZeroEntries(Y));
    PetscCall(DMLocalToGlobal(ctx->dm_y, Y_loc, ADD_VALUES, Y));

    // Restore local vectors, as needed
    if (!ctx->X_loc) PetscCall(DMRestoreLocalVector(ctx->dm_x, &X_loc));
    PetscCall(DMRestoreLocalVector(ctx->dm_y, &Y_loc));
  }

  // Log flops
  if (PetscMemTypeDevice(ctx->mem_type)) PetscCall(PetscLogGpuFlops(ctx->flops_mult));
  else PetscCall(PetscLogFlops(ctx->flops_mult));

  PetscCall(PetscLogEventEnd(ctx->log_event_mult, A, X, Y, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Compute `A^T Y = X` for a `MATCEED`.

  Collective across MPI processes.

  @param[in]   A  `MATCEED`
  @param[in]   Y  Input PETSc vector
  @param[out]  X  Output PETSc vector

  @return An error code: 0 - success, otherwise - failure
**/
PetscErrorCode MatMultTranspose_Ceed(Mat A, Vec Y, Vec X) {
  MatCeedContext ctx;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(PetscLogEventBegin(ctx->log_event_mult_transpose, A, Y, X, 0));

  {
    PetscMemType x_mem_type, y_mem_type;
    Vec          X_loc, Y_loc = ctx->Y_loc_transpose;

    // Get local vectors
    if (!ctx->Y_loc_transpose) PetscCall(DMGetLocalVector(ctx->dm_y, &Y_loc));
    PetscCall(DMGetLocalVector(ctx->dm_x, &X_loc));

    // Global-to-local
    PetscCall(DMGlobalToLocal(ctx->dm_y, Y, INSERT_VALUES, Y_loc));

    // Setup libCEED vectors
    PetscCall(VecReadPetscToCeed(Y_loc, &y_mem_type, ctx->y_loc));
    PetscCall(VecZeroEntries(X_loc));
    PetscCall(VecPetscToCeed(X_loc, &x_mem_type, ctx->x_loc));

    // Apply libCEED operator
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCeed(ctx->ceed, CeedOperatorApplyAdd(ctx->op_mult_transpose, ctx->y_loc, ctx->x_loc, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());

    // Restore PETSc vectors
    PetscCall(VecReadCeedToPetsc(ctx->y_loc, y_mem_type, Y_loc));
    PetscCall(VecCeedToPetsc(ctx->x_loc, x_mem_type, X_loc));

    // Local-to-global
    PetscCall(VecZeroEntries(X));
    PetscCall(DMLocalToGlobal(ctx->dm_x, X_loc, ADD_VALUES, X));

    // Restore local vectors, as needed
    if (!ctx->Y_loc_transpose) PetscCall(DMRestoreLocalVector(ctx->dm_y, &Y_loc));
    PetscCall(DMRestoreLocalVector(ctx->dm_x, &X_loc));
  }

  // Log flops
  if (PetscMemTypeDevice(ctx->mem_type)) PetscCall(PetscLogGpuFlops(ctx->flops_mult_transpose));
  else PetscCall(PetscLogFlops(ctx->flops_mult_transpose));

  PetscCall(PetscLogEventEnd(ctx->log_event_mult_transpose, A, Y, X, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
