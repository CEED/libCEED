/// @file
/// Test delegated non-tensor basis application across contraction boundaries
/// \test Test delegated non-tensor basis application across contraction boundaries
//TESTARGS(only="cpu") {ceed_resource}

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

#if defined(__clang__)
#define CEED_TEST_SCALAR_ORACLE __attribute__((optnone))
#else
#define CEED_TEST_SCALAR_ORACLE
#endif

typedef struct {
  CeedInt num_qpts, num_elem, num_comp;
} BasisCase;

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

static CeedScalar InputValue(CeedSize i) { return (CeedScalar)(((CeedInt)(5 * i % 23) - 11) / 32.0); }

static CeedScalar OutputValue(CeedSize i) { return (CeedScalar)(((CeedInt)(3 * i % 17) - 8) / 16.0); }

// Keep the scalar oracle independent of Clang's scalable-vector lowering when QEMU varies the runtime SVE vector length.
CEED_TEST_SCALAR_ORACLE static int RunBasisCase(Ceed ceed, BasisCase test, CeedTransposeMode t_mode, CeedInt add) {
  const CeedInt              dim = 2, num_nodes = 5, q_comp = dim;
  const CeedSize             interp_size = (CeedSize)q_comp * test.num_qpts * num_nodes;
  const CeedSize             input_size  = t_mode == CEED_NOTRANSPOSE ? (CeedSize)test.num_comp * num_nodes * test.num_elem
                                                                      : (CeedSize)q_comp * test.num_comp * test.num_qpts * test.num_elem;
  const CeedSize             output_size = t_mode == CEED_NOTRANSPOSE ? (CeedSize)q_comp * test.num_comp * test.num_qpts * test.num_elem
                                                                      : (CeedSize)test.num_comp * num_nodes * test.num_elem;
  CeedBasis                  basis;
  CeedVector                 input, output;
  CeedScalar                *interp       = malloc(interp_size * sizeof(*interp));
  CeedScalar                *div          = calloc((CeedSize)test.num_qpts * num_nodes, sizeof(*div));
  CeedScalar                *q_ref        = calloc((CeedSize)dim * test.num_qpts, sizeof(*q_ref));
  CeedScalar                *q_weight     = calloc(test.num_qpts, sizeof(*q_weight));
  CeedScalar                *input_array  = malloc(input_size * sizeof(*input_array));
  CeedScalar                *output_array = malloc(output_size * sizeof(*output_array));
  CeedScalar                *expected     = malloc(output_size * sizeof(*expected));
  const volatile CeedScalar *interp_ref = interp, *input_ref = input_array;
  int                        failed = !interp || !div || !q_ref || !q_weight || !input_array || !output_array || !expected;

  if (failed) goto cleanup_arrays;
  for (CeedSize i = 0; i < interp_size; i++) interp[i] = (CeedScalar)(((CeedInt)(7 * i % 29) - 14) / 64.0);
  for (CeedSize i = 0; i < input_size; i++) input_array[i] = InputValue(i);
  for (CeedSize i = 0; i < output_size; i++) output_array[i] = add ? OutputValue(i) : (CeedScalar)NAN;

  if (t_mode == CEED_NOTRANSPOSE) {
    for (CeedInt q_comp_i = 0; q_comp_i < q_comp; q_comp_i++) {
      for (CeedInt comp = 0; comp < test.num_comp; comp++) {
        for (CeedInt q = 0; q < test.num_qpts; q++) {
          for (CeedInt elem = 0; elem < test.num_elem; elem++) {
            const CeedSize output_index = (((CeedSize)q_comp_i * test.num_comp + comp) * test.num_qpts + q) * test.num_elem + elem;
            CeedScalar     value        = add ? output_array[output_index] : 0.0;

            for (CeedInt node = 0; node < num_nodes; node++) {
              const CeedSize input_index  = ((CeedSize)comp * num_nodes + node) * test.num_elem + elem;
              const CeedSize interp_index = ((CeedSize)q_comp_i * test.num_qpts + q) * num_nodes + node;

              value += interp_ref[interp_index] * input_ref[input_index];
            }
            expected[output_index] = value;
          }
        }
      }
    }
  } else {
    for (CeedInt comp = 0; comp < test.num_comp; comp++) {
      for (CeedInt node = 0; node < num_nodes; node++) {
        for (CeedInt elem = 0; elem < test.num_elem; elem++) {
          const CeedSize output_index = ((CeedSize)comp * num_nodes + node) * test.num_elem + elem;
          CeedScalar     value        = add ? output_array[output_index] : 0.0;

          for (CeedInt q_comp_i = 0; q_comp_i < q_comp; q_comp_i++) {
            for (CeedInt q = 0; q < test.num_qpts; q++) {
              const CeedSize input_index  = (((CeedSize)q_comp_i * test.num_comp + comp) * test.num_qpts + q) * test.num_elem + elem;
              const CeedSize interp_index = ((CeedSize)q_comp_i * test.num_qpts + q) * num_nodes + node;

              value += interp_ref[interp_index] * input_ref[input_index];
            }
          }
          expected[output_index] = value;
        }
      }
    }
  }

  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, test.num_comp, num_nodes, test.num_qpts, interp, div, q_ref, q_weight, &basis);
  CeedVectorCreate(ceed, input_size, &input);
  CeedVectorCreate(ceed, output_size, &output);
  CeedVectorSetArray(input, CEED_MEM_HOST, CEED_COPY_VALUES, input_array);
  CeedVectorSetArray(output, CEED_MEM_HOST, CEED_COPY_VALUES, output_array);
  if (add) {
    failed = CeedBasisApplyAdd(basis, test.num_elem, t_mode, CEED_EVAL_INTERP, input, output);
  } else {
    failed = CeedBasisApply(basis, test.num_elem, t_mode, CEED_EVAL_INTERP, input, output);
  }
  if (!failed) {
    const CeedScalar *result;

    CeedVectorGetArrayRead(output, CEED_MEM_HOST, &result);
    for (CeedSize i = 0; i < output_size; i++) {
      const CeedScalar tolerance = 200 * CEED_EPSILON * (test.num_qpts + num_nodes + 1) * (1 + fabs(expected[i]));

      if (!isfinite(result[i]) || fabs(result[i] - expected[i]) > tolerance) {
        printf("Error in non-tensor basis at index %" CeedSize_FMT ": %g != %g\n", i, (double)result[i], (double)expected[i]);
        failed = 1;
        break;
      }
    }
    CeedVectorRestoreArrayRead(output, &result);
  }
  CeedVectorDestroy(&input);
  CeedVectorDestroy(&output);
  CeedBasisDestroy(&basis);

cleanup_arrays:
  free(interp);
  free(div);
  free(q_ref);
  free(q_weight);
  free(input_array);
  free(output_array);
  free(expected);
  return failed;
}

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt vector_length = GetVectorLength();
  BasisCase     tests[]       = {
      {127, 4 * vector_length - 1, 1},
      {128, 4 * vector_length,     1},
      {129, 4 * vector_length - 1, 1},
      {129, 4 * vector_length,     1},
      {129, 4 * vector_length + 1, 1},
      {129, 4 * vector_length + 1, 2},
  };

  CeedInit(argv[1], &ceed);
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    if (RunBasisCase(ceed, tests[i], CEED_NOTRANSPOSE, 0)) return 1;
    for (CeedInt add = 0; add <= 1; add++) {
      if (RunBasisCase(ceed, tests[i], CEED_TRANSPOSE, add)) return 1;
    }
  }
  CeedDestroy(&ceed);
  return 0;
}
