#include <ceed.h>

int main(int argc, char** argv) {
  Ceed ceed;
  CeedVector x, y;
  CeedInt ne = 3;
  CeedInt ind[2*ne];
  CeedScalar a[ne+1];
  const CeedScalar* yy;
  CeedElemRestriction r;

  CeedInit("/cpu/occa", &ceed);
  CeedVectorCreate(ceed, ne+1, &x); // x[0..ne] size ne+1
  for (CeedInt i=0; i<ne+1; i++) a[i] = 10 + i; // a[0..ne] size ne+1 = 10..14
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<ne; i++) {  // ind[0..2*ne-1] = ..(i,i+1)..
    ind[2*i+0] = i;
    ind[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed,ne,2,ne+1,CEED_MEM_HOST,CEED_USE_POINTER,ind,&r);
  
  CeedVectorCreate(ceed, ne*2, &y); // y[0..2*ne-1]  
  CeedVectorSetArray(y, CEED_MEM_HOST, CEED_COPY_VALUES, NULL); // Allocates array
  
  CeedElemRestrictionApply(r, CEED_NOTRANSPOSE, x, y, CEED_REQUEST_IMMEDIATE);
  
  CeedVectorGetArrayRead(y, CEED_MEM_HOST, &yy);
  for (CeedInt i=0; i<ne*2; i++) {
    if (10+(i+1)/2 != yy[i])
      return CeedError(ceed, (int)i, "Error in restricted array y[%d] = %f",
                       i, (double)yy[i]);
  }
  CeedVectorRestoreArrayRead(y, &yy);
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedElemRestrictionDestroy(&r);
  CeedDestroy(&ceed);
  return 0;
}
