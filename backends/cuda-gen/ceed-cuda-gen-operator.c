// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stddef.h>

#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-gen-operator-build.h"
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
  CeedOperator_Cuda_gen *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

static int Waste(int threads_per_sm, int warp_size, int threads_per_elem, int elems_per_block) {
  int useful_threads_per_block = threads_per_elem * elems_per_block;
  // round up to nearest multiple of warp_size
  int block_size    = ((useful_threads_per_block + warp_size - 1) / warp_size) * warp_size;
  int blocks_per_sm = threads_per_sm / block_size;
  return threads_per_sm - useful_threads_per_block * blocks_per_sm;
}

// Choose the least wasteful block size constrained by blocks_per_sm of
// max_threads_per_block.
//
// The x and y part of block[] contains per-element sizes (specified on input)
// while the z part is number of elements.
//
// Problem setting: we'd like to make occupancy high with relatively few
// inactive threads. CUDA (cuOccupancyMaxPotentialBlockSize) can tell us how
// many threads can run.
//
// Note that full occupancy sometimes can't be achieved by one thread block. For
// example, an SM might support 1536 threads in total, but only 1024 within a
// single thread block. So cuOccupancyMaxPotentialBlockSize may suggest a block
// size of 768 so that two blocks can run, versus one block of 1024 will prevent
// a second block from running. The cuda-gen kernels are pretty heavy with lots
// of instruction-level parallelism (ILP) so we'll generally be okay with
// relatvely low occupancy and smaller thread blocks, but we solve a reasonably
// general problem here. Empirically, we find that blocks bigger than about 256
// have higher latency and worse load balancing when the number of elements is
// modest.
//
// cuda-gen can't choose block sizes arbitrarily; they need to be a multiple of
// the number of quadrature points (or number of basis functions). They also
// have a lot of __syncthreads(), which is another point against excessively
// large thread blocks. Suppose I have elements with 7x7x7 quadrature points.
// This will loop over the last dimension, so we have 7*7=49 threads per
// element. Suppose we have two elements = 2*49=98 useful threads. CUDA
// schedules in units of full warps (32 threads), so 128 CUDA hardware threads
// are effectively committed to that block. Now suppose
// cuOccupancyMaxPotentialBlockSize returned 352. We can schedule 2 blocks of
// size 98 (196 useful threads using 256 hardware threads), but not a third
// block (which would need a total of 384 hardware threads).
//
// If instead, we had packed 3 elements, we'd have 3*49=147 useful threads
// occupying 160 slots, and could schedule two blocks. Alternatively, we could
// pack a single block of 7 elements (2*49=343 useful threads) into the 354
// slots. The latter has the least "waste", but __syncthreads()
// over-synchronizes and it might not pay off relative to smaller blocks.
static int BlockGridCalculate(CeedInt num_elem, int blocks_per_sm, int max_threads_per_block, int max_threads_z, int warp_size, int block[3],
                              int *grid) {
  const int threads_per_sm   = blocks_per_sm * max_threads_per_block;
  const int threads_per_elem = block[0] * block[1];
  int       elems_per_block  = 1;
  int       waste            = Waste(threads_per_sm, warp_size, threads_per_elem, 1);
  for (int i = 2; i <= CeedIntMin(max_threads_per_block / threads_per_elem, num_elem); i++) {
    int i_waste = Waste(threads_per_sm, warp_size, threads_per_elem, i);
    // We want to minimize waste, but smaller kernels have lower latency and
    // less __syncthreads() overhead so when a larger block size has the same
    // waste as a smaller one, go ahead and prefer the smaller block.
    if (i_waste < waste || (i_waste == waste && threads_per_elem * i <= 128)) {
      elems_per_block = i;
      waste           = i_waste;
    }
  }
  // In low-order elements, threads_per_elem may be sufficiently low to give
  // an elems_per_block greater than allowable for the device, so we must check
  // before setting the z-dimension size of the block.
  block[2] = CeedIntMin(elems_per_block, max_threads_z);
  *grid    = (num_elem + elems_per_block - 1) / elems_per_block;
  return CEED_ERROR_SUCCESS;
}

// callback for cuOccupancyMaxPotentialBlockSize, providing the amount of
// dynamic shared memory required for a thread block of size threads.
static size_t dynamicSMemSize(int threads) { return threads * sizeof(CeedScalar); }

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cuda_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Cuda *cuda_data;
  CeedCallBackend(CeedGetData(ceed, &cuda_data));
  CeedOperator_Cuda_gen *data;
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedQFunction           qf;
  CeedQFunction_Cuda_gen *qf_data;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedInt num_elem, num_input_fields, num_output_fields;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedEvalMode eval_mode;
  CeedVector   vec, output_vecs[CEED_FIELD_MAX] = {};

  // Creation of the operator
  CeedCallBackend(CeedCudaGenOperatorBuild(op));

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.inputs[i] = NULL;
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
    }
  }

  // Output vectors

  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.outputs[i] = NULL;
    } else {
      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
      output_vecs[i] = vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == output_vecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.outputs[i]));
      } else {
        data->fields.outputs[i] = data->fields.outputs[index];
      }
    }
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c));

  // Apply operator

  void         *opargs[]  = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W};
  const CeedInt dim       = data->dim;
  const CeedInt Q_1d      = data->Q_1d;
  const CeedInt P_1d      = data->max_P_1d;
  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  int           max_threads_per_block, min_grid_size;
  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, data->op, dynamicSMemSize, 0, 0x10000));
  int block[3] =
      {
          thread_1d,
          dim < 2 ? 1 : thread_1d,
          -1,
      },
      grid;
  CeedChkBackend(BlockGridCalculate(num_elem, min_grid_size / cuda_data->device_prop.multiProcessorCount, max_threads_per_block,
                                    cuda_data->device_prop.maxThreadsDim[2], cuda_data->device_prop.warpSize, block, &grid));
  CeedInt shared_mem = block[0] * block[1] * block[2] * sizeof(CeedScalar);
  CeedCallBackend(CeedRunKernelDimSharedCuda(ceed, data->op, grid, block[0], block[1], block[2], shared_mem, opargs));

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == output_vecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorRestoreArray(vec, &data->fields.outputs[i]));
      }
    }
  }

  // Restore context data
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Cuda_gen *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda_gen));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
