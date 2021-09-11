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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stddef.h>
#include "ceed-cuda-gen.h"
#include "ceed-cuda-gen-operator-build.h"
#include "../cuda/ceed-cuda.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
  int ierr;
  CeedOperator_Cuda_gen *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

static int Waste(int threads_per_sm, int warp_size, int threads_per_elem,
                 int elems_per_block) {
  int useful_threads_per_block = threads_per_elem * elems_per_block;
  // round up to nearest multiple of warp_size
  int block_size = ((useful_threads_per_block + warp_size - 1) / warp_size) *
                   warp_size;
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
static int BlockGridCalculate(CeedInt nelem, int blocks_per_sm,
                              int max_threads_per_block, int max_threads_z,
                              int warp_size, int block[3], int *grid) {
  const int threads_per_sm = blocks_per_sm * max_threads_per_block;
  const int threads_per_elem = block[0] * block[1];
  int elems_per_block = 1;
  int waste = Waste(threads_per_sm, warp_size, threads_per_elem, 1);
  for (int i=2;
       i <= CeedIntMin(max_threads_per_block / threads_per_elem, nelem);
       i++) {
    int i_waste = Waste(threads_per_sm, warp_size, threads_per_elem, i);
    // We want to minimize waste, but smaller kernels have lower latency and
    // less __syncthreads() overhead so when a larger block size has the same
    // waste as a smaller one, go ahead and prefer the smaller block.
    if (i_waste < waste || (i_waste == waste && threads_per_elem * i <= 128)) {
      elems_per_block = i;
      waste = i_waste;
    }
  }
  // In low-order elements, threads_per_elem may be sufficiently low to give
  // an elems_per_block greater than allowable for the device, so we must check
  // before setting the z-dimension size of the block.
  block[2] = CeedIntMin(elems_per_block, max_threads_z);
  *grid = (nelem + elems_per_block - 1) / elems_per_block;
  return CEED_ERROR_SUCCESS;
}

// callback for cuOccupancyMaxPotentialBlockSize, providing the amount of
// dynamic shared memory required for a thread block of size threads.
static size_t dynamicSMemSize(int threads) { return threads * sizeof(double); }

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Cuda_gen(CeedOperator op, CeedVector invec,
    CeedVector outvec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Cuda *cuda_data;
  ierr = CeedGetData(ceed, &cuda_data); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Cuda_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedInt nelem, numinputfields, numoutputfields;
  ierr = CeedOperatorGetNumElements(op, &nelem); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  CeedOperatorField *opinputfields, *opoutputfields;
  ierr = CeedOperatorGetFields(op, &opinputfields, &opoutputfields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  ierr = CeedQFunctionGetFields(qf, &qfinputfields, &qfoutputfields);
  CeedChkBackend(ierr);
  CeedEvalMode emode;
  CeedVector vec, outvecs[16] = {};

  // Creation of the operator
  ierr = CeedCudaGenOperatorBuild(op); CeedChkBackend(ierr);

  // Input vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.in[i] = NULL;
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.in[i]);
      CeedChkBackend(ierr);
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.out[i] = NULL;
    } else {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      outvecs[i] = vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        ierr = CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.out[i]);
        CeedChkBackend(ierr);
      } else {
        data->fields.out[i] = data->fields.out[index];
      }
    }
  }

  // Get context data
  CeedQFunctionContext ctx;
  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChkBackend(ierr);
  if (ctx) {
    ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_DEVICE, &qf_data->d_c);
    CeedChkBackend(ierr);
  }

  // Apply operator
  void *opargs[] = {(void *) &nelem, &qf_data->d_c, &data->indices,
                    &data->fields, &data->B, &data->G, &data->W
                   };
  const CeedInt dim = data->dim;
  const CeedInt Q1d = data->Q1d;
  const CeedInt P1d = data->maxP1d;
  const CeedInt thread1d = CeedIntMax(Q1d, P1d);
  int max_threads_per_block, min_grid_size;
  CeedChk_Cu(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size,
             &max_threads_per_block, data->op, dynamicSMemSize, 0, 0x10000));
  int block[3] = {thread1d, dim < 2 ? 1 : thread1d, -1,}, grid;
  CeedChkBackend(BlockGridCalculate(nelem,
                                    min_grid_size/ cuda_data->deviceProp.multiProcessorCount, max_threads_per_block,
                                    cuda_data->deviceProp.maxThreadsDim[2],
                                    cuda_data->deviceProp.warpSize, block, &grid));
  CeedInt shared_mem = block[0] * block[1] * block[2] * sizeof(double);
  ierr = CeedRunKernelDimSharedCuda(ceed, data->op, grid, block[0], block[1],
                                    block[2], shared_mem, opargs);
  CeedChkBackend(ierr);

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opinputfields[i], &vec); CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      ierr = CeedVectorRestoreArrayRead(vec, &data->fields.in[i]);
      CeedChkBackend(ierr);
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode);
    CeedChkBackend(ierr);
    if (emode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(opoutputfields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        ierr = CeedVectorRestoreArray(vec, &data->fields.out[i]);
        CeedChkBackend(ierr);
      }
    }
  }

  // Restore context data
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &qf_data->d_c);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Cuda_gen *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Cuda_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
