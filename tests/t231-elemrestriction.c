/// @file
/// Test creation, use, and destruction of an element restriction at points
/// \test Test creation, use, and destruction of an element restriction at points
#include <ceed.h>
#include <ceed/backend.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedInt             num_elem = 3, num_points = num_elem * 2;
  CeedInt             ind[(num_elem + 1) + num_points];
  CeedVector          x, y;
  CeedElemRestriction elem_restriction;

  CeedInit(argv[1], &ceed);

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
  CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points, 1, num_points, CEED_MEM_HOST, CEED_USE_POINTER, ind, &elem_restriction);

  CeedElemRestrictionCreateVector(elem_restriction, &x, &y);
  CeedVectorSetValue(y, 0.0);
  {
    CeedInt    point_index = num_elem;
    CeedScalar array[num_points];

    for (CeedInt i = 0; i < num_elem; i++) {
      CeedInt num_points_in_elem = (i + 1) % num_elem + 1;

      for (CeedInt j = 0; j < num_points_in_elem; j++) {
        array[point_index] = i;
        point_index        = (point_index + 1) % num_points;
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, array);
  }

  CeedElemRestrictionApply(elem_restriction, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  {
    CeedInt           e_layout[3];
    const CeedScalar *read_array;

    CeedVectorGetArrayRead(y, CEED_MEM_HOST, &read_array);
    CeedElemRestrictionGetELayout(elem_restriction, e_layout);

    for (CeedInt i = 0; i < num_elem; i++) {
      CeedSize      elem_offset        = 0;
      const CeedInt num_points_in_elem = (i + 1) % num_elem + 1;

      CeedElemRestrictionGetAtPointsElementOffset(elem_restriction, i, &elem_offset);
      for (CeedInt j = 0; j < num_points_in_elem; j++) {
        if (i != read_array[elem_offset + j * e_layout[0]]) {
          // LCOV_EXCL_START
          printf("Error in restricted array y[%" CeedInt_FMT "] = %f\n != %f\n", (CeedInt)elem_offset + j * e_layout[0],
                 (CeedScalar)read_array[elem_offset + j * e_layout[0]], (CeedScalar)i);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(y, &read_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&elem_restriction);
  CeedDestroy(&ceed);
  return 0;
}
