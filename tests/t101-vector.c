/// @file
/// Test CeedVectorSetValue
/// \test Test CeedVectorSetValue

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdio.h>
#include <stdlib.h>

static int CheckValues(Ceed ceed, CeedVector x, CeedScalar value) {
  const CeedScalar *read_array;
  CeedSize          len;

  CeedVectorGetLength(x, &len);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
  for (CeedInt i = 0; i < len; i++) {
    if (read_array[i] != value) printf("Error reading array[%" CeedInt_FMT "] = %f", i, (CeedScalar)read_array[i]);
  }
  CeedVectorRestoreArrayRead(x, &read_array);
  return 0;
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedVector x;
  CeedInt    len = 10;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  {
    CeedScalar array[len];

    CeedVectorCreate(ceed, len, &x);
    for (CeedInt i = 0; i < len; i++) array[i] = len + i;
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }

  // Sync memtype to device for GPU backends
  {
    CeedMemType type = CEED_MEM_HOST;

    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }

  {
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      if (read_array[i] != len + i) printf("Error reading array[%" CeedInt_FMT "] = %f\n", i, (CeedScalar)read_array[i]);
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  // Set all entries to same value and check
  CeedVectorSetValue(x, 3.0);
  CheckValues(ceed, x, 3.0);
  CeedVectorDestroy(&x);

  // Set value before setting or getting the array
  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 5.0);
  CheckValues(ceed, x, 5.0);
  CeedVectorDestroy(&x);

  CeedDestroy(&ceed);
  return 0;
}
