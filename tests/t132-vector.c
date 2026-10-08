/// @file
/// Test overwriting a vector
/// \test Test overwriting a vector

#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

// Add a value to an entry the way a backend does, storing it on the first write while the vector is overwritten
static void AddValue(CeedScalar *array, uint64_t *overwrite_mask, CeedSize i, CeedScalar value) {
  if (overwrite_mask) {
    array[i] = ((overwrite_mask[i / 64] >> (i % 64)) & 1 ? array[i] : 0.0) + value;
    overwrite_mask[i / 64] |= (uint64_t)1 << (i % 64);
  } else {
    array[i] += value;
  }
}

static void CheckValues(const char *label, CeedVector x, CeedScalar value) {
  CeedSize          len;
  const CeedScalar *array;

  CeedVectorGetLength(x, &len);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &array);
  for (CeedSize i = 0; i < len; i++) {
    if (array[i] != value) printf("%s: entry %" CeedSize_FMT " is %f, not %f\n", label, i, array[i], value);
  }
  CeedVectorRestoreArrayRead(x, &array);
}

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedInt    len = 70;  // More than one mask word
  CeedVector x;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, len, &x);

  // Entries not written through the mask are zero once it is applied
  {
    uint64_t          state, cleared_state;
    uint64_t         *overwrite_mask;
    CeedScalar       *array;
    const CeedScalar *read_array;

    CeedVectorSetValue(x, 5.0);
    CeedVectorGetState(x, &state);
    CeedVectorClearOverwriteMask(x);
    CeedVectorGetState(x, &cleared_state);
    if (cleared_state == state) printf("Clearing the overwrite mask did not change the vector state\n");

    CeedVectorGetArrayOverwrite(x, CEED_MEM_HOST, &array, &overwrite_mask);
    if (!overwrite_mask) printf("No overwrite mask after clearing it\n");
    AddValue(array, overwrite_mask, 3, 1.0);
    AddValue(array, overwrite_mask, 3, 2.0);
    AddValue(array, overwrite_mask, 65, 4.0);
    CeedVectorRestoreArray(x, &array);
    CeedVectorApplyOverwriteMask(x);

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      const CeedScalar value = i == 3 ? 3.0 : i == 65 ? 4.0 : 0.0;

      if (read_array[i] != value) printf("Overwrite: entry %" CeedInt_FMT " is %f, not %f\n", i, read_array[i], value);
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  // Applying the mask drops it, and without a mask nothing changes
  {
    uint64_t   *overwrite_mask;
    CeedScalar *array;

    CeedVectorSetValue(x, 5.0);
    CeedVectorGetArrayOverwrite(x, CEED_MEM_HOST, &array, &overwrite_mask);
    if (overwrite_mask) printf("Overwrite mask after applying it\n");
    CeedVectorRestoreArray(x, &array);
    CeedVectorApplyOverwriteMask(x);
    CheckValues("Apply without a mask", x, 5.0);
  }

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
