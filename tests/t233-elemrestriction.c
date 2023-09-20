/// @file
/// Test creation, transpose use, and destruction of an element restriction at points for single elements
/// \test Test creation, transpose use, and destruction of an element restriction at points for single elements
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem = 3, num_points = num_elem * 2;
  CeedInt             ind[(num_elem + 1) + num_points];
  CeedVector          x, y;
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_points, &x);
  CeedVectorSetValue(x, 0.0);

  {
    CeedInt offset      = num_elem + 1;
    CeedInt point_index = num_elem;

    for (CeedInt i = 0; i < num_elem; i++) {
      CeedInt num_points_in_elem = (i + 1) % num_elem + 1;

      ind[i] = offset;
      for (CeedInt j = 0; j < num_points_in_elem; j++) {
        ind[offset + j] = point_index;
        point_index     = (point_index + 1) % num_points;
      }
      offset += num_points_in_elem;
    }
    ind[num_elem] = offset;
  }
  CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_COPY_VALUES, ind, &elem_restriction);

  {
    CeedInt max_points;

    CeedElemRestrictionGetMaxPointsInElement(elem_restriction, &max_points);
    CeedVectorCreate(ceed, max_points, &y);
    CeedVectorSetValue(y, 1.0);
  }

  {
    for (CeedInt i = 0; i < num_elem; i++) {
      CeedInt           point_index = num_elem;
      const CeedScalar *read_array;

      CeedVectorSetValue(x, 0.0);
      CeedElemRestrictionApplyAtPointsInElement(elem_restriction, i, CEED_TRANSPOSE, y, x, CEED_REQUEST_IMMEDIATE);

      CeedVectorGetArrayRead(x, CEED_MEM_HOST, &read_array);
      for (CeedInt j = 0; j < num_elem; j++) {
        CeedInt num_points_in_elem = (j + 1) % num_elem + 1;

        for (CeedInt k = 0; k < num_points_in_elem; k++) {
          if (fabs(read_array[point_index] - (i == j ? 1.0 : 0.0)) > 10 * CEED_EPSILON) {
            printf("Error in restricted array x[%" CeedInt_FMT "] = %f\n", point_index, (CeedScalar)read_array[point_index]);
          }
          point_index = (point_index + 1) % num_points;
        }
      }
      CeedVectorRestoreArrayRead(x, &read_array);
    }
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
