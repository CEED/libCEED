/// @file
/// Test tensor contraction across SME streaming and private-ZA call boundaries
/// \test Test tensor contraction across SME streaming and private-ZA call boundaries

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CEED_TEST_SME
#include <arm_sme.h>
#include <arm_sve.h>

static inline uint64_t ReadSVCR(void) __arm_streaming_compatible {
  uint64_t value;

  __asm__ volatile("mrs %0, svcr" : "=r"(value) : : "memory");
  return value & 3;
}

// The ordinary libCEED call must preserve this caller's live ZA, even when
// the contraction creates its own private ZA and switches streaming mode.
__arm_new("za") static int ApplyWithZA(CeedTensorContract contract, CeedInt n, const CeedScalar *t, CeedTransposeMode t_mode, CeedInt add,
                                       const CeedScalar *u, CeedScalar *v, const unsigned char *pattern, unsigned char *saved,
                                       uint64_t *streaming_state) __arm_streaming {
  const uint64_t bytes = svcntb();
  const svbool_t all   = svptrue_b8();

  svzero_za();
  for (uint32_t row = 0; row < bytes; row++) svld1_hor_za8(0, row, all, pattern + row * bytes);
  int error = CeedTensorContractApply(contract, 2, 3, n, n, t, t_mode, add, u, v);

  for (uint32_t row = 0; row < bytes; row++) svst1_hor_za8(0, row, all, saved + row * bytes);
  *streaming_state = ReadSVCR();
  return error;
}

static int CheckState(CeedTensorContract contract) {
  const uint64_t bytes    = svcntsb();
  const CeedInt  n        = bytes / sizeof(CeedScalar) + 1;
  const size_t   za_bytes = bytes * bytes;
  CeedScalar    *t = malloc(3 * n * sizeof(*t)), *u = malloc(2 * 3 * n * sizeof(*u)), *v = malloc(2 * n * n * sizeof(*v));
  unsigned char *pattern = malloc(za_bytes), *saved = malloc(za_bytes);
  int            error = 0;

  if (!t || !u || !v || !pattern || !saved) {
    fprintf(stderr, "SME state test allocation failed\n");
    error = 1;
    goto cleanup;
  }
  for (CeedInt mode = 0; mode < 2; mode++) {
    const CeedTransposeMode t_mode = mode ? CEED_TRANSPOSE : CEED_NOTRANSPOSE;

    for (CeedInt j = 0; j < n; j++) {
      for (CeedInt b = 0; b < 3; b++) t[mode ? b * n + j : j * 3 + b] = (CeedScalar)(j + b + 1) / 8;
    }
    for (CeedInt add = 0; add < 2; add++) {
      for (CeedInt nested = 0; nested < 2; nested++) {
        for (CeedInt repeat = 0; repeat < 2; repeat++) {
          // Powers of two keep every product and sum exact in both precisions.
          for (CeedInt a = 0; a < 2; a++) {
            for (CeedInt b = 0; b < 3; b++) {
              for (CeedInt c = 0; c < n; c++) u[(a * 3 + b) * n + c] = (CeedScalar)(a + b + c + repeat + 1) / 16;
            }
          }
          for (CeedInt i = 0; i < 2 * n * n; i++) v[i] = add ? (CeedScalar)repeat / 4 : NAN;
          for (size_t i = 0; i < za_bytes; i++) pattern[i] = (unsigned char)((i * 13 + i / bytes * 7 + repeat * 19 + 1) % 251);
          memset(saved, 0, za_bytes);
          const uint64_t state_before    = ReadSVCR();
          uint64_t       streaming_state = 0;

          if (nested) {
            error = ApplyWithZA(contract, n, t, t_mode, add, u, v, pattern, saved, &streaming_state);
          } else {
            error = CeedTensorContractApply(contract, 2, 3, n, n, t, t_mode, add, u, v);
          }
          if (error) goto cleanup;
          if (ReadSVCR() != state_before || (nested && streaming_state != 3)) {
            printf("Tensor contraction did not restore SME streaming/ZA state (transpose=%d add=%d nested=%d repeat=%d)\n", mode, add, nested,
                   repeat);
            error = 1;
            goto cleanup;
          }
          if (nested && memcmp(pattern, saved, za_bytes)) {
            printf("Tensor contraction changed the caller's ZA (transpose=%d add=%d repeat=%d)\n", mode, add, repeat);
            error = 1;
            goto cleanup;
          }
          for (CeedInt a = 0; a < 2; a++) {
            for (CeedInt j = 0; j < n; j++) {
              for (CeedInt c = 0; c < n; c++) {
                CeedScalar expected = add ? (CeedScalar)repeat / 4 : 0;

                for (CeedInt b = 0; b < 3; b++) expected += (CeedScalar)(j + b + 1) * (a + b + c + repeat + 1) / 128;
                if (v[(a * n + j) * n + c] != expected) {
                  printf("Incorrect contraction across SME call boundary (transpose=%d add=%d nested=%d repeat=%d)\n", mode, add, nested, repeat);
                  error = 1;
                  goto cleanup;
                }
              }
            }
          }
        }
      }
    }
  }

cleanup:
  free(t);
  free(u);
  free(v);
  free(pattern);
  free(saved);
  return error;
}
#endif

int main(int argc, char **argv) {
  Ceed ceed;

  CeedCall(CeedInit(argv[1], &ceed));
#ifdef CEED_TEST_SME
  CeedTensorContract contract;
  int                error;

  CeedCall(CeedTensorContractCreate(ceed, &contract));
  error = CheckState(contract);
  CeedCall(CeedTensorContractDestroy(&contract));
  CeedCall(CeedDestroy(&ceed));
  return error;
#else
  return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test requires SME support");
#endif
}
