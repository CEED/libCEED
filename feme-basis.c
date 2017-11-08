#include <feme-impl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int FemeBasisCreateTensorH1(Feme feme, FemeInt dim, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis *basis) {
  // Populate basis struct
  (*basis)->feme = feme;
  (*basis)->dim = dim;
  memcpy((*basis)->qref1d, qref1d, Q1d);
  memcpy((*basis)->qweight1d, qweight1d, Q1d);
  memcpy((*basis)->interp1d, interp1d, P1d);
  memcpy((*basis)->grad1d, grad1d, P1d);
  // Tensor code here
  return 0;
}

int FemeBasisCreateTensorH1Lagrange(Feme feme, FemeInt dim, FemeInt degree, FemeInt Q, FemeQuadMode qmode, FemeBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  FemeScalar temp, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;
  ierr = FemeCalloc((degree+ 1)*(Q + 1), &interp1d); FemeChk(ierr);
  ierr = FemeCalloc((degree + 1)*(Q + 1), &grad1d); FemeChk(ierr);
  ierr = FemeCalloc(degree + 1, &nodes); FemeChk(ierr);
  ierr = FemeCalloc(Q + 1, &qref1d); FemeChk(ierr);
  ierr = FemeCalloc(Q + 1, &qweight1d); FemeChk(ierr);
  // Get Nodes and Weights
  if (qmode) {
    ierr = FemeLobattoQuadrature(degree, nodes, nodes); FemeChk(ierr);
    ierr = FemeLobattoQuadrature(Q, qref1d, qweight1d); FemeChk(ierr);
  } else {
    ierr = FemeLobattoQuadrature(degree, nodes, nodes); FemeChk(ierr);
    ierr = FemeGaussQuadrature(Q, qref1d, qweight1d); FemeChk(ierr);
  }
  // Build B matrix
  for (i = 0; i <= Q; i++) {
    for (j = 0; j <= degree; j++) {
      temp = 1.0;
      for (k = 0; k <= degree; k++) {
        if (k != j) {
          temp *= (qref1d[i] - nodes[k]) / (nodes[j] - nodes[k]);
        } }
      interp1d[i + Q*j] = temp;
    } }
    // Build D matrix
    for (i = 0; i <= Q; i++) {
      for (j = 0; j <= degree; j++) {
        temp = 1.0;
        for (k = 0; k <= degree; k++) {
          if (k != j) {
            temp *= (qref1d[i] -nodes[k]) / (nodes[j] - nodes[k]);
          } }
        grad1d[i + Q*j] = temp;
        temp = 1.0;
        for (k = 0; k <= degree; k++) {
          if (k != j) {
            temp += 1 / (nodes[k] - nodes[j]);
          }
        }
        grad1d[i + Q*j] *= temp;
      } }
   // Pass to FemeBasisCreateTensorH1
  ierr = FemeBasisCreateTensorH1(feme, dim, degree, Q, interp1d, grad1d, qref1d, qweight1d, basis); FemeChk(ierr);
  return 0;
}

int FemeGaussQuadrature(FemeInt degree, FemeScalar *qref1d, FemeScalar *qweight1d) {
  // Allocate
  int ierr, i, j, k;
  char s[50] = "";
  FemeScalar P0, P1, P2, dP2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  for (i = 0; i <= (degree + 1)/2; i++) {
    // Guess
    xi = cos(PI*(FemeScalar)(2*i+1)/((FemeScalar)(2*degree+2)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    for (j = 2; j <= degree + 1; j++) {
      P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi*P2 - P0)*(FemeScalar)(degree + 1)/(xi*xi-1.0);
    xi = xi-P2/dP2;
    k = 1;
    // Newton to convergence
    while (fabs(P2)>pow(10,-15) && k < 100) {
      P0 = 1.0;
      P1 = xi;
      for (j = 2; j <= degree + 1; j++) {
        P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(FemeScalar)(degree + 1)/(xi*xi-1.0);
      xi = xi-P2/dP2;
      k++;
    }
    // Save xi, wi
    wi = 2.0/((1.0-xi*xi)*dP2*dP2);
    qweight1d[i] = wi;
    qweight1d[degree-i] = wi;
    qref1d[i] = -xi;
    qref1d[degree-i]= xi;
  }
  return 0;
}

int FemeLobattoQuadrature(FemeInt degree, FemeScalar *qref1d, FemeScalar *qweight1d) {
  // Allocate
  int ierr, i, j, k;
  FemeScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0*atan(1.0);
  char s[2*sizeof(double)];
  // Build qref1d, qweight1d
  // Set endpoints
  wi = 2.0/((FemeScalar)(degree*(degree - 1)));
  qweight1d[0] = wi;
  qweight1d[degree] = wi;
  qref1d[0] = -1.0;
  qref1d[degree] = 1.0;
  // Interior
  for (i = 1; i <= degree/2; i++) {
    xi = cos(PI*(FemeScalar)(i)/((FemeScalar)(degree)));
    P0 = 1.0;
    P1 = xi;
    for (j = 2; j <= degree; j++) {
      P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    dP2 = (xi*P2 - P0)*(FemeScalar)(degree)/(xi*xi-1.0);
    d2P2 = (2*xi*dP2 - (FemeScalar)(degree*(degree-1))*P2)/(1.0-xi*xi);
    xi = xi-dP2/d2P2;
    k = 1;
    while (fabs(dP2)>pow(10,-15) && k < 100) {
      P0 = 1.0;
      P1 = xi;
      for (j = 2; j <= degree; j++) {
        P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(FemeScalar)(degree)/(xi*xi-1.0);
      d2P2 = (2*xi*dP2 - (FemeScalar)(degree*(degree-1))*P2)/(1.0-xi*xi);
      xi = xi-dP2/d2P2;
      k++;
    }

    sprintf(s, "%d", k);
    printf("%s\n", s);fflush(stdout);
    wi = 2.0/(((FemeScalar)(degree*(degree+1)))*P2*P2);
    qweight1d[i] = wi;
    qweight1d[degree-i] = wi;
    qref1d[i] = -xi;
    qref1d[degree-i]= xi;
  }
  return 0;
}

int FemeBasisDestroy(FemeBasis *basis) {
  int ierr;
  ierr = FemeFree(&(*basis)->interp1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->grad1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qref1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qweight1d); FemeChk(ierr);
  return 0;
}
