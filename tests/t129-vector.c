/// @file
/// Test Filter function: Filters or clips a CeedVector using a threshold value.
#include <ceed.h>
#include <stdio.h>

typedef struct {
  CeedScalar threshold;
  CeedScalar input[10];
  CeedScalar expected[10];
} FilterTest;

static int VerifyFilter(CeedVector x, const CeedScalar *expected, CeedSize len) {
  const CeedScalar *read_array;
  int errors = 0;

  CeedCall(CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array));
  for (CeedInt i = 0; i < len; i++) {
    if (read_array[i] != expected[i]) errors++;
  }
  CeedCall(CeedVectorRestoreArrayRead(x, &read_array));
  return errors;
}

int main(int argc, char **argv) {
  Ceed ceed;
  int  test_errors = 0;

  // Initialize libCEED
  CeedInit(argv[1], &ceed);

  // Test Data
  FilterTest tests[] = {
    // Filter some values
    { .threshold = 0.5,
      .input    = {1.0, -0.2, 0.5, -1.0, 0.0, 0.51, -0.49, 10.0, -0.5, 0.1},
      .expected = {1.0,  0.0, 0.0, -1.0, 0.0, 0.51,  0.0,  10.0,  0.0, 0.0} 
    },

    // Filter all of the values
    { .threshold = 6.0,
      .input    = {1.0, -2.0, 3.0, -4.0, 5.0, -5.0, 5.1, -5.1, 0.0, 1.1},
      .expected = {0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0} 
    },

    // Filter none of the values
    { .threshold = 1.2E-14,
      .input    = {1.0, -1.0, 0.5, -0.5, 0.0001, -0.0001, 0.0, 2.2, -2.2, 3.3},
      .expected = {1.0, -1.0, 0.5, -0.5, 0.0001, -0.0001, 0.0, 2.2, -2.2, 3.3}
    },

    // Filter some values
    { .threshold = 1E-9,
      .input    = {0.11, -1.2E-10, 0.1, -0.1, 9.0E-8, -0.09, 0.0, 7.0E-12, -1.0, 5.1E-6},
      .expected = {0.11, 0.0, 0.1, -0.1, 9.0E-8, -0.09, 0.0, 0.0, -1.0, 5.1E-6} 
    }
  };

  // Execution Loop
  for (int i = 0; i < 4; i++) {
    CeedVector x;
    CeedInt    len = 10;
    
    printf("Running Test Case %d (Threshold: %g)...\n", i + 1, (double)tests[i].threshold);
    CeedCall(CeedVectorCreate(ceed, len, &x));
    CeedCall(CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, tests[i].input));

    CeedCall(CeedVectorFilter(x, tests[i].threshold));

    if (VerifyFilter(x, tests[i].expected, len) == 0) {
      printf("  Result: PASS\n");
    } else {
      printf("  Result: FAIL\n");
      test_errors++;
    }
    CeedVectorDestroy(&x);
  }

  // Cleanup and Exit
  CeedDestroy(&ceed);
  return test_errors;
}