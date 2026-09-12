/// @file
/// Test tensor contraction semantics and vector-length tails
/// \test Test tensor contraction semantics and vector-length tails
//TESTARGS(only="cpu") {ceed_resource}
#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(__clang__)
#define CEED_TEST_SCALAR_ORACLE __attribute__((optnone))
#else
#define CEED_TEST_SCALAR_ORACLE
#endif

typedef struct {
  CeedInt A, B, C, J;
} TensorCase;

typedef struct {
  CeedScalar *data;
#if defined(__linux__)
  void  *mapping;
  size_t mapping_size;
#endif
} GuardedArray;

static int GuardedArrayCreate(CeedSize length, GuardedArray *array) {
  if (length <= 0 || (uintmax_t)length > SIZE_MAX / sizeof(*array->data)) return 1;
#if defined(__linux__)
  const long page_size = sysconf(_SC_PAGESIZE);

  if (page_size <= 0) return 1;
  const size_t bytes = (size_t)length * sizeof(*array->data), num_pages = (bytes + page_size - 1) / page_size;

  if (!bytes || num_pages >= SIZE_MAX / (size_t)page_size) return 1;
  array->mapping_size = (num_pages + 1) * page_size;
  array->mapping      = mmap(NULL, array->mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (array->mapping == MAP_FAILED) return 1;
  void *guard = (char *)array->mapping + num_pages * page_size;

  if (mprotect(guard, page_size, PROT_NONE)) {
    munmap(array->mapping, array->mapping_size);
    array->mapping = NULL;
    return 1;
  }
  array->data = (CeedScalar *)((char *)guard - bytes);
#else
  array->data = malloc((size_t)length * sizeof(*array->data));
  if (!array->data) return 1;
#endif
  return 0;
}

static void GuardedArrayDestroy(GuardedArray *array) {
#if defined(__linux__)
  if (array->mapping && array->mapping != MAP_FAILED) munmap(array->mapping, array->mapping_size);
#else
  free(array->data);
#endif
  array->data = NULL;
}

static CeedInt GetVectorLength(void) {
#if defined(__ARM_FEATURE_SVE)
#ifdef CEED_SCALAR_IS_FP64
  return svcntd();
#else
  return svcntw();
#endif
#else
  return 16 / sizeof(CeedScalar);
#endif
}

// Keep the scalar oracle independent of Clang's scalable-vector lowering when QEMU varies the runtime SVE vector length.
CEED_TEST_SCALAR_ORACLE static int RunCase(CeedTensorContract contract, TensorCase test, CeedTransposeMode t_mode, CeedInt add) {
  const CeedSize             t_size = (CeedSize)test.B * test.J, u_size = (CeedSize)test.A * test.B * test.C;
  const CeedSize             v_size = (CeedSize)test.A * test.J * test.C;
  GuardedArray               u = {0}, v = {0};
  CeedScalar                *t = malloc(t_size * sizeof(*t)), *expected = malloc(v_size * sizeof(*expected));
  const volatile CeedScalar *t_ref = t, *u_ref = u.data;

  if (!t || !expected || GuardedArrayCreate(u_size, &u) || GuardedArrayCreate(v_size, &v)) {
    free(t);
    free(expected);
    GuardedArrayDestroy(&u);
    GuardedArrayDestroy(&v);
    return 1;
  }
  u_ref = u.data;
  for (CeedSize i = 0; i < t_size; i++) t[i] = (CeedScalar)(((CeedInt)(7 * i % 17) - 8) / 16.0);
  for (CeedSize i = 0; i < u_size; i++) u.data[i] = (CeedScalar)(((CeedInt)(5 * i % 19) - 9) / 32.0);
  for (CeedSize i = 0; i < v_size; i++) {
    v.data[i] = add ? (CeedScalar)(((CeedInt)(3 * i % 11) - 5) / 8.0) : (CeedScalar)NAN;
  }

  for (CeedInt a = 0; a < test.A; a++) {
    for (CeedInt j = 0; j < test.J; j++) {
      for (CeedInt c = 0; c < test.C; c++) {
        const CeedSize index = ((CeedSize)a * test.J + j) * test.C + c;
        CeedScalar     value = add ? v.data[index] : 0.0;

        for (CeedInt b = 0; b < test.B; b++) {
          const CeedSize t_index = t_mode == CEED_TRANSPOSE ? (CeedSize)b * test.J + j : (CeedSize)j * test.B + b;

          value += t_ref[t_index] * u_ref[((CeedSize)a * test.B + b) * test.C + c];
        }
        expected[index] = value;
      }
    }
  }
  int failed = CeedTensorContractApply(contract, test.A, test.B, test.C, test.J, t, t_mode, add, u.data, v.data);

  if (!failed) {
    for (CeedSize i = 0; i < v_size; i++) {
      const CeedScalar tolerance = 100 * CEED_EPSILON * (test.B + 1) * (1 + fabs(expected[i]));

      if (!isfinite(v.data[i]) || fabs(v.data[i] - expected[i]) > tolerance) {
        printf("Error in tensor contraction at index %" CeedSize_FMT ": %g != %g\n", i, (double)v.data[i], (double)expected[i]);
        failed = 1;
        break;
      }
    }
  }
  free(t);
  free(expected);
  GuardedArrayDestroy(&u);
  GuardedArrayDestroy(&v);
  return failed;
}

int main(int argc, char **argv) {
  Ceed               ceed;
  CeedTensorContract contract;
  const CeedInt      vector_length = GetVectorLength();
  TensorCase         tests[]       = {
      {1, 1, 1,                     1 },
      {2, 3, vector_length - 1,     4 },
      {3, 4, vector_length,         2 },
      {2, 5, vector_length + 1,     5 },
      {4, 2, 2 * vector_length - 1, 3 },
      {2, 7, 2 * vector_length,     6 },
      {3, 3, 2 * vector_length + 1, 7 },
      {1, 6, 3 * vector_length - 1, 8 },
      {2, 5, 3 * vector_length,     9 },
      {3, 4, 3 * vector_length + 1, 10},
      {2, 7, 4 * vector_length - 1, 11},
      {1, 6, 4 * vector_length,     12},
      {3, 5, 4 * vector_length + 1, 13},
      {2, 3, 8 * vector_length - 1, 14},
      {1, 7, 8 * vector_length + 1, 15},
  };

  CeedInit(argv[1], &ceed);
  CeedTensorContractCreate(ceed, &contract);
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    for (CeedTransposeMode t_mode = CEED_NOTRANSPOSE; t_mode <= CEED_TRANSPOSE; t_mode++) {
      for (CeedInt add = 0; add <= 1; add++) {
        if (RunCase(contract, tests[i], t_mode, add)) return 1;
      }
    }
  }
  CeedTensorContractDestroy(&contract);
  CeedDestroy(&ceed);
  return 0;
}
