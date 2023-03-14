# libCEED Feedback

### t108-vector

Test failure:
```
./t108-vector /gpu/hip/gen
Error: Max norm 8.000000 != 9.
```

The test initializes the array {0, -1, 2, -3, -4, 5, -6, 7, -8, 9} and then computes the max norm. BLAS correctly returns the index which, if unchanged, would return 9 and the test would pass.
Why is the index decremented?

```
    case CEED_NORM_MAX: {
      CeedInt indx;
      if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
        CeedCallHipblas(ceed, hipblasIsamax(handle, length, (float *)d_array, 1, &indx));
      } else {
        CeedCallHipblas(ceed, hipblasIdamax(handle, length, (double *)d_array, 1, &indx));
      }
      CeedScalar normNoAbs;
      CeedCallHip(ceed, hipMemcpy(&normNoAbs, impl->d_array + indx /* why? - 1 */, sizeof(CeedScalar), hipMemcpyDeviceToHost));
      *norm = fabs(normNoAbs);
      break;
    }
```

### Resource Constraints
#### Shared Local Memory
In the default configuration, libCEED will un in fp64 mode which causes both Iris and Arcticus to run out of shared local memory in some tests such as t313-basis.
For this reason, fp32 is now forced:
```
// #include "ceed-f64.h"
include "ceed-f32.h"
```

#### Block Size
The default block size is 512 which is too large for Iris. The block size is now set to 256 so we can test on all hardware available.

### hipBLAS Failures
two tests fail when run on OpenCL backend: t108-vector and t540-operator. The causes for these errors is not yet known but hipBLAS hasn't been tested much with OpenCL compared to Level Zero.

t108-vector
````
# $ build/t108-vector /gpu/hip/ref
not ok 1 - FAIL: stderr
/gpfs/jlse-fs0/users/pvelesko/libCEED/backends/hip-ref/ceed-hip-ref.c:30 in CeedHipGetHipblasHandle(): HIPBLAS_STATUS_INTERNAL_ERROR
````

t540-operator
````
# $ build/t540-operator /gpu/hip/ref
not ok 1 - FAIL: stderr
/gpfs/jlse-fs0/users/pvelesko/libCEED/backends/hip-ref/ceed-hip-ref.c:30 in CeedHipGetHipblasHandle(): HIPBLAS_STATUS_INTERNAL_ERROR
````