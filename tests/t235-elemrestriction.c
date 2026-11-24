/// @file
/// Test overwriting an L-vector with the transpose of an element restriction
/// \test Test overwriting an L-vector with the transpose of an element restriction
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

// The transpose of ones gives 1 at the end nodes, 2 at the shared nodes, and 0 at the entries no element writes
static void Check(const char *label, CeedVector x, CeedInt num_elem) {
  CeedSize          len;
  const CeedScalar *x_array;

  CeedVectorGetLength(x, &len);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &x_array);
  for (CeedSize i = 0; i < len; i++) {
    const CeedScalar value = i > num_elem ? 0.0 : (i == 0 || i == num_elem) ? 1.0 : 2.0;

    if (x_array[i] != value) printf("%s: x[%" CeedSize_FMT "] = %f != %f\n", label, i, x_array[i], value);
  }
  CeedVectorRestoreArrayRead(x, &x_array);
}

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedVector          x, y;
  CeedInt             num_elem = 70, elem_size = 2, blk_size = 8, num_blk = 9;
  CeedInt             ind[elem_size * num_elem], strides[3] = {1, 2, 2};
  CeedElemRestriction elem_restriction, elem_restriction_blocked, elem_restriction_strided;

  CeedInit(argv[1], &ceed);

  // L-vector of 2 (num_elem + 1) entries, so entries past the nodes are never written
  CeedVectorCreate(ceed, 2 * (num_elem + 1), &x);
  for (CeedInt i = 0; i < num_elem; i++) {
    ind[2 * i + 0] = i;
    ind[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, elem_size, 1, 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);
  CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, 1, 1, 2 * (num_elem + 1), CEED_MEM_HOST, CEED_USE_POINTER, ind,
                                   &elem_restriction_blocked);
  CeedElemRestrictionCreateStrided(ceed, num_elem, elem_size, 1, 2 * (num_elem + 1), strides, &elem_restriction_strided);

  // Restrictions with offsets support overwriting, strided restrictions do not
  {
    bool supports_overwrite;

    CeedElemRestrictionSupportsOverwrite(elem_restriction, &supports_overwrite);
    if (!supports_overwrite) printf("Restriction does not support overwriting\n");
    CeedElemRestrictionSupportsOverwrite(elem_restriction_blocked, &supports_overwrite);
    if (!supports_overwrite) printf("Blocked restriction does not support overwriting\n");
    CeedElemRestrictionSupportsOverwrite(elem_restriction_strided, &supports_overwrite);
    if (supports_overwrite) printf("Strided restriction supports overwriting\n");
  }

  // Overwrite with the transpose of all elements at once, from an L-vector holding 5
  CeedVectorCreate(ceed, num_elem * elem_size, &y);
  CeedVectorSetValue(y, 1.0);
  CeedVectorSetValue(x, 5.0);
  CeedElemRestrictionApplyOverwrite(elem_restriction, y, x, CEED_REQUEST_IMMEDIATE);
  Check("Overwrite", x, num_elem);

  // Overwrite with the transpose of the blocks in order
  CeedVectorDestroy(&y);
  CeedVectorCreate(ceed, blk_size * elem_size, &y);
  CeedVectorSetValue(y, 1.0);
  CeedVectorSetValue(x, 5.0);
  for (CeedInt b = 0; b < num_blk; b++) {
    // The last block has 6 elements, so its padding lanes must not count
    CeedElemRestrictionApplyBlockOverwrite(elem_restriction_blocked, b, y, x, CEED_REQUEST_IMMEDIATE);
  }
  Check("Overwrite by block", x, num_elem);

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedElemRestrictionDestroy(&elem_restriction_blocked);
  CeedElemRestrictionDestroy(&elem_restriction_strided);
  CeedDestroy(&ceed);
  return 0;
}
