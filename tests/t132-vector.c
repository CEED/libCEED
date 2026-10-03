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
  CeedVector x, y;

  CeedInit(argv[1], &ceed);
  CeedVectorCreate(ceed, len, &x);
  CeedVectorCreate(ceed, len, &y);
  CeedVectorSetValue(y, 1.0);

  // Entries not written are zero after the overwrite ends
  {
    uint64_t          state, overwrite_state;
    uint64_t         *overwrite_mask;
    CeedScalar       *array;
    const CeedScalar *read_array;

    CeedVectorSetValue(x, 5.0);
    CeedVectorGetState(x, &state);
    CeedVectorBeginOverwrite(x);
    CeedVectorGetState(x, &overwrite_state);
    if (overwrite_state == state) printf("Beginning an overwrite did not change the vector state\n");

    CeedVectorGetArrayOverwrite(x, CEED_MEM_HOST, &array, &overwrite_mask);
    AddValue(array, overwrite_mask, 3, 1.0);
    AddValue(array, overwrite_mask, 3, 2.0);
    AddValue(array, overwrite_mask, 65, 4.0);
    CeedVectorRestoreArray(x, &array);
    CeedVectorEndOverwrite(x);

    CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
    for (CeedInt i = 0; i < len; i++) {
      const CeedScalar value = i == 3 ? 3.0 : i == 65 ? 4.0 : 0.0;

      if (read_array[i] != value) printf("Overwrite: entry %" CeedInt_FMT " is %f, not %f\n", i, read_array[i], value);
    }
    CeedVectorRestoreArrayRead(x, &read_array);
  }

  // Reading ends the overwrite, setting the entries not written to zero
  CeedVectorSetValue(x, 5.0);
  CeedVectorBeginOverwrite(x);
  CheckValues("Read during overwrite", x, 0.0);

  // So does a vector operation
  CeedVectorSetValue(x, 5.0);
  CeedVectorBeginOverwrite(x);
  CeedVectorAXPY(x, 2.0, y);
  CheckValues("AXPY during overwrite", x, 2.0);

  // Setting every entry discards the overwrite
  CeedVectorBeginOverwrite(x);
  CeedVectorSetValue(x, 4.0);
  CeedVectorEndOverwrite(x);
  CheckValues("SetValue during overwrite", x, 4.0);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
