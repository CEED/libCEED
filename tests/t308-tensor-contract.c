/// @file
/// Test tensor contraction layouts, batches, and accumulation
/// \test Test tensor contraction layouts, batches, and accumulation

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int CheckTensorContract(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, CeedTransposeMode t_mode, CeedInt add) {
  const CeedSize   num_t = (CeedSize)J * B, num_u = (CeedSize)A * B * C, num_v = (CeedSize)A * J * C;
  const CeedScalar guard = 12345;
  CeedScalar     *t_storage = malloc((num_t + 2) * sizeof(*t_storage)), *t_copy = malloc((num_t + 2) * sizeof(*t_copy));
  CeedScalar     *u_storage = malloc((num_u + 2) * sizeof(*u_storage)), *u_copy = malloc((num_u + 2) * sizeof(*u_copy));
  CeedScalar     *v_storage = malloc((num_v + 2) * sizeof(*v_storage));
  long double    *v_ref = malloc(num_v * sizeof(*v_ref));
  int             error = 0;
  CeedInt         call = 0;

  if (!t_storage || !t_copy || !u_storage || !u_copy || !v_storage || !v_ref) {
    fprintf(stderr, "Tensor contraction test allocation failed\n");
    error = 1;
    goto cleanup;
  }

  // Offset each array by one scalar to exercise unaligned vector accesses.
  CeedScalar *t = t_storage + 1, *u = u_storage + 1, *v = v_storage + 1;

  t_storage[0] = t_storage[num_t + 1] = u_storage[0] = u_storage[num_u + 1] = v_storage[0] = v_storage[num_v + 1] = guard;
  // Store the same mathematical matrix in both coefficient layouts.
  for (CeedInt j = 0; j < J; j++) {
    for (CeedInt b = 0; b < B; b++) {
      const CeedSize index = t_mode == CEED_TRANSPOSE ? (CeedSize)b * J + j : (CeedSize)j * B + b;

      t[index] = (CeedScalar)((j * 7 + b * 3 + 1) % 17 - 8) / 10;
    }
  }
  memcpy(t_copy, t_storage, (num_t + 2) * sizeof(*t_copy));
  for (CeedSize i = 0; i < num_v; i++) {
    v_ref[i] = (CeedScalar)((CeedInt)(i % 7) - 3) / 4;
    // Overwrite must not depend on the previous output, including NaNs.
    v[i] = add ? (CeedScalar)v_ref[i] : NAN;
  }

  for (call = 0; call < 3; call++) {
    // Change the input between calls to expose stale output or cached data.
    for (CeedSize i = 0; i < num_u; i++) u[i] = (CeedScalar)((CeedInt)((i * 5 + 3 + call * 7) % 19) - 9) / 10;
    memcpy(u_copy, u_storage, (num_u + 2) * sizeof(*u_copy));

    error = CeedTensorContractApply(contract, A, B, C, J, t, t_mode, add, u, v);
    if (error) goto cleanup;
    if (memcmp(t_storage, t_copy, (num_t + 2) * sizeof(*t_copy)) || memcmp(u_storage, u_copy, (num_u + 2) * sizeof(*u_copy))) {
      printf("Tensor contraction modified an input or its guards\n");
      error = 1;
      goto cleanup;
    }
    if (v_storage[0] != guard || v_storage[num_v + 1] != guard) {
      printf("Tensor contraction modified an output guard\n");
      error = 1;
      goto cleanup;
    }

    // Independent scalar oracle, accumulated in higher precision from saved inputs.
    for (CeedInt a = 0; a < A; a++) {
      for (CeedInt j = 0; j < J; j++) {
        for (CeedInt c = 0; c < C; c++) {
          const CeedSize out = ((CeedSize)a * J + j) * C + c;
          long double    expected = add ? v_ref[out] : 0, magnitude = fabsl(expected);

          for (CeedInt b = 0; b < B; b++) {
            const CeedSize    t_index = t_mode == CEED_TRANSPOSE ? (CeedSize)b * J + j : (CeedSize)j * B + b;
            const CeedSize    u_index = ((CeedSize)a * B + b) * C + c;
            const long double product = (long double)t_copy[t_index + 1] * u_copy[u_index + 1];

            expected += product;
            magnitude += fabsl(product);
          }
          // Allow rounding from each reduction and repeated accumulation in either precision.
          const long double tolerance = 10.L * CEED_EPSILON * (B + 1) * (call + 1) * (1 + magnitude);

          if (!isfinite(v[out]) || !isfinite(expected) || fabsl((long double)v[out] - expected) > tolerance) {
            printf("Tensor contraction v[%" CeedSize_FMT "] = %.17g, expected %.17Lg (tolerance %.3Lg)\n", out, (double)v[out], expected, tolerance);
            error = 1;
            goto cleanup;
          }
          v_ref[out] = expected;
        }
      }
    }
  }

cleanup:
  if (error) {
    printf("A=%" CeedInt_FMT " B=%" CeedInt_FMT " C=%" CeedInt_FMT " J=%" CeedInt_FMT " mode=%s add=%" CeedInt_FMT " call=%" CeedInt_FMT "\n",
           A, B, C, J, t_mode == CEED_TRANSPOSE ? "transpose" : "notranspose", add, call);
  }
  free(t_storage);
  free(t_copy);
  free(u_storage);
  free(u_copy);
  free(v_storage);
  free(v_ref);
  return error;
}

int main(int argc, char **argv) {
  Ceed                    ceed;
  CeedTensorContract      contract;
  const CeedTransposeMode t_modes[] = {CEED_NOTRANSPOSE, CEED_TRANSPOSE};
  const CeedInt           add_modes[] = {0, 1, -1};
  // A, B, C, J: singleton dimensions, rectangular shapes, long reductions, and vector/tile tails.
  const CeedInt dims[][4] = {
      {1, 1,  1,  1 },
      {1, 5,  1,  7 },
      {1, 7,  9,  1 },
      {2, 1,  9,  5 },
      {3, 5,  9,  7 },
      {2, 7,  19, 17},
      {1, 33, 7,  9 },
      {2, 5,  8,  4 },
      {2, 3,  65, 67}
  };
  int error = 0;

  CeedCall(CeedInit(argv[1], &ceed));
  CeedCall(CeedTensorContractCreate(ceed, &contract));
  for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
    for (size_t mode = 0; mode < sizeof(t_modes) / sizeof(t_modes[0]); mode++) {
      for (size_t add = 0; add < sizeof(add_modes) / sizeof(add_modes[0]); add++) {
        error = CheckTensorContract(contract, dims[i][0], dims[i][1], dims[i][2], dims[i][3], t_modes[mode], add_modes[add]);
        if (error) goto cleanup;
      }
    }
  }

cleanup:
  CeedCall(CeedTensorContractDestroy(&contract));
  CeedCall(CeedDestroy(&ceed));
  return error;
}
