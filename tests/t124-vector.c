/// @file
/// Test CeedVectorTakeFromDLPack
/// Test CeedVectorToDLPack
#include <ceed/ceed.h>
#include <ceed/dlpack.h>
#include <ceed/backend.h>

static int CheckValues(Ceed ceed, CeedVector x, CeedScalar value) {
  const CeedScalar *b;
  CeedInt n;
  CeedVectorGetLength(x, &n);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++) {
    if (b[i] != value)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f",i,
             (double)b[i]);
    // LCOV_EXCL_STOP
  }
  CeedVectorRestoreArrayRead(x, &b);
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x, y;
  DLManagedTensor *dl_tensor;
  CeedInt n;
  int ierr;
  CeedInit(argv[1], &ceed);
  n = 10;
  /* test fills a vector with a true value (5) and another vector with
     a signal value (-1). The "true" value is copied into the other vector
     via an intermediate DLManagedTensor */
  CeedVectorCreate(ceed, n, &x);
  CeedVectorSetValue(x, 5.0);
  
  CeedVectorCreate(ceed, n, &y);
  CeedVectorSetValue(y, -1.0);

  ierr = CeedVectorToDLPack(ceed, x, CEED_MEM_HOST,
			    &dl_tensor); CeedChk(ierr);

  ierr = CeedVectorTakeFromDLPack(ceed, y, dl_tensor,
				  CEED_USE_POINTER); CeedChk(ierr);
  CheckValues(ceed, y, 5.0);
  CeedVectorDestroy(&y); /* should only need to destroy this version, since it
			    uses the pointer from x */
  //CeedFree(&dl_tensor);
  return 0;
}
