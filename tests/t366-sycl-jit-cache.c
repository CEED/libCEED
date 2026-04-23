/// @file
/// Test that tensor basis with JIT compilation works across repeated
/// CeedInit/CeedDestroy cycles (regression test for SYCL kernel bundle
/// caching bug where reloaded native binaries lost kernel IDs).
/// \test Test repeated CeedInit/Destroy with tensor basis apply
#include <ceed.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

static int run_basis_apply(const char *resource) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector u, v;
  int        dim = 2, p = 4, q = 4, len = (int)(pow((CeedScalar)(q), dim) + 0.4);

  CeedInit(resource, &ceed);
  CeedVectorCreate(ceed, len, &u);
  CeedVectorCreate(ceed, len, &v);

  {
    CeedScalar u_array[len];
    for (int i = 0; i < len; i++) u_array[i] = 1.0;
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS_LOBATTO, &basis);
  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  {
    const CeedScalar *v_array;
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (int i = 0; i < len; i++) {
      if (fabs(v_array[i] - 1.) > 10. * CEED_EPSILON) {
        printf("v[%d] = %f != 1.\n", i, v_array[i]);
        CeedVectorRestoreArrayRead(v, &v_array);
        CeedBasisDestroy(&basis);
        CeedVectorDestroy(&u);
        CeedVectorDestroy(&v);
        CeedDestroy(&ceed);
        return 1;
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}

int main(int argc, char **argv) {
  // First run: JIT compiles from source, may populate cache
  if (run_basis_apply(argv[1])) return 1;

  // Unset SYCL_CACHE_DIR to exercise the no-cache-dir code path.
  // This caught a bug where CeedBuildBundleCached_Sycl loaded a cached
  // native binary via zeModuleCreate + make_kernel_bundle, which lost
  // SYCL kernel IDs and crashed with "kernel bundle does not contain
  // the kernel" at dispatch time.
  unsetenv("SYCL_CACHE_DIR");

  // Second run: must still work without SYCL_CACHE_DIR
  if (run_basis_apply(argv[1])) return 1;
  return 0;
}
