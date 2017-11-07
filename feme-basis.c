#include <feme-impl.h>
#include <stdio.h>
#include <stdlib.h>

int FemeBasisCreateTensorH1(Feme feme, FemeInt dim, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  FemeScalar temp;
  FemeScalar *bmat1d, *dmat1d;
  ierr = FemeCalloc(P1d*Q1d, &bmat1d); FemeChk(ierr);
  ierr = FemeCalloc(P1d*Q1d, &dmat1d); FemeChk(ierr);
  // Build B matrix
  for (i = 0; i <= Q1d; i++) {
    for (j = 0; j <= P1d; j++) {
      temp = 1.0;
      for (k = 0; k <= P1d; k++) {
        if (k != j) {
          temp *= (qref1d[i] - interp1d[k]) / (interp1d[j] - interp1d[k]);
        } }
      bmat1d[i + Q1d*j] = temp;
    } }
    // Build D matrix
    for (i = 0; i <= Q1d; i++) {
      for (j = 0; j <= P1d; j++) {
        temp = 1.0;
        for (k = 0; k <= P1d; k++) {
          if (k != j) {
            temp *= (qref1d[i] - grad1d[k]) / (grad1d[j] - grad1d[k]);
          } }
        dmat1d[i + Q1d*j] = temp;
      } }
  // Populate basis struct
  (*basis)->feme = feme;
  (*basis)->dim = dim;
  (*basis)->qref1d = qref1d;
  (*basis)->qweight1d = qweight1d;
  (*basis)->bmat1d = bmat1d;
  (*basis)->dmat1d = dmat1d;
  return 0;
}

int FemeBasisCreateTensorH1Lagrange(Feme feme, FemeInt dim, FemeInt degree, FemeInt Q, FemeBasis *basis) {
  // Allocate
  int ierr, i;
  FemeScalar *interp1d, *grad1d, *qref1d, *qweight1d;
  ierr = FemeCalloc(degree+1, &interp1d); FemeChk(ierr);
  ierr = FemeCalloc(degree+1, &grad1d); FemeChk(ierr);
  ierr = FemeCalloc(Q+1, &qref1d); FemeChk(ierr);
  ierr= FemeCalloc(Q+1, &qweight1d); FemeChk(ierr);
  // Build interp1d, grad1d
for (i = 0; i <= degree; i++) {
  interp1d[i] = -1 + 2/degree;
}
for (i = 0; i <= degree; i++) {
  grad1d[i] = 1.0;
}
  // Build qref1d, qweight1d

  // Pass to FemeBasisCreateTensorH1
  ierr = FemeBasisCreateTensorH1(feme, dim, degree, Q, interp1d, grad1d, qref1d, qweight1d, basis); FemeChk(ierr);
  return 0;
}

int FemeBasisDestroy(FemeBasis *basis) {
  int ierr;
  ierr = FemeFree(&(*basis)->bmat1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->dmat1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qref1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qweight1d); FemeChk(ierr);
  return 0;
}
