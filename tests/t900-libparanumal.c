/// @file
/// Test creation and destruction of a CEED object
/// \test Test creation and destruction of a CEED object
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedOperator op;

  CeedInit("/gpu/libparanumal", &ceed);
  CeedOperatorCreateFromGallery(ceed, "elliptic", &op);

  //CeedQFunctionDestroy(&qf);
  CeedDestroy(&ceed);
  return 0;
}
