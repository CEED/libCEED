#include <feme-impl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int FemeBasisCreateTensorH1(Feme feme, FemeInt dim, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis *basis) {
  int ierr;

  if (!feme->BasisCreateTensorH1) return FemeError(feme, 1, "Backend does not support BasisCreateTensorH1");
  ierr = FemeCalloc(1,basis);FemeChk(ierr);
  (*basis)->feme = feme;
  (*basis)->dim = dim;
  (*basis)->P1d = P1d;
  (*basis)->Q1d = Q1d;
  ierr = FemeMalloc(Q1d,&(*basis)->qref1d);FemeChk(ierr);
  ierr = FemeMalloc(Q1d,&(*basis)->qweight1d);FemeChk(ierr);
  memcpy((*basis)->qref1d, qref1d, Q1d*sizeof(qref1d[0]));
  memcpy((*basis)->qweight1d, qweight1d, Q1d*sizeof(qweight1d[0]));
  ierr = FemeMalloc(Q1d*P1d,&(*basis)->interp1d);FemeChk(ierr);
  ierr = FemeMalloc(Q1d*P1d,&(*basis)->grad1d);FemeChk(ierr);
  memcpy((*basis)->interp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));
  memcpy((*basis)->grad1d, grad1d, Q1d*P1d*sizeof(interp1d[0]));
  ierr = feme->BasisCreateTensorH1(feme, dim, P1d, Q1d, interp1d, grad1d, qref1d, qweight1d, *basis);FemeChk(ierr);
  return 0;
}

int FemeBasisCreateTensorH1Lagrange(Feme feme, FemeInt dim, FemeInt degree, FemeInt Q, FemeQuadMode qmode, FemeBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  FemeScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;
  FemeInt P = degree+1;
  ierr = FemeCalloc(P*Q, &interp1d); FemeChk(ierr);
  ierr = FemeCalloc(P*Q, &grad1d); FemeChk(ierr);
  ierr = FemeCalloc(P, &nodes); FemeChk(ierr);
  ierr = FemeCalloc(Q, &qref1d); FemeChk(ierr);
  ierr = FemeCalloc(Q, &qweight1d); FemeChk(ierr);
  // Get Nodes and Weights
  ierr = FemeLobattoQuadrature(degree, nodes, NULL); FemeChk(ierr);
  switch (qmode) {
  case FEME_GAUSS:
    ierr = FemeGaussQuadrature(Q-1, qref1d, qweight1d); FemeChk(ierr);
    break;
  case FEME_GAUSS_LOBATTO:
    ierr = FemeLobattoQuadrature(Q-1, qref1d, qweight1d); FemeChk(ierr);
    break;
  }
  // Build B, D matrix
  // Fornberg, 1998
  for (i = 0; i  < Q; i++) {
    c1 = 1.0;
    c3 = nodes[0] - qref1d[i];
    interp1d[i*P+0] = 1.0;
    for (j = 1; j < P; j++) {
      c2 = 1.0;
      c4 = c3;
      c3 = nodes[j] - qref1d[i];
      for (k = 0; k < j; k++) {
        dx = nodes[j] - nodes[k];
        c2 *= dx;
        if (k == j - 1) {
          grad1d[i*P + j] = c1*(interp1d[i*P + k] - c4*grad1d[i*P + k]) / c2;
          interp1d[i*P + j] = - c1*c4*interp1d[i*P + k] / c2;
        }
        grad1d[i*P + k] = (c3*grad1d[i*P + k] - interp1d[i*P + k]) / dx;
        interp1d[i*P + k] = c3*interp1d[i*P + k] / dx;
      }
      c1 = c2;
    } }
  //  // Pass to FemeBasisCreateTensorH1
  ierr = FemeBasisCreateTensorH1(feme, dim, P, Q, interp1d, grad1d, qref1d, qweight1d, basis); FemeChk(ierr);
  ierr = FemeFree(&interp1d);FemeChk(ierr);
  ierr = FemeFree(&grad1d);FemeChk(ierr);
  ierr = FemeFree(&nodes);FemeChk(ierr);
  ierr = FemeFree(&qref1d);FemeChk(ierr);
  ierr = FemeFree(&qweight1d);FemeChk(ierr);
  return 0;
}

int FemeGaussQuadrature(FemeInt degree, FemeScalar *qref1d, FemeScalar *qweight1d) {
  // Allocate
  int i, j, k;
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
    while (fabs(P2)>1e-15 && k < 100) {
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
  int i, j, k;
  FemeScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  // Set endpoints
  wi = 2.0/((FemeScalar)(degree*(degree + 1)));
  if (qweight1d) {
    qweight1d[0] = wi;
    qweight1d[degree] = wi;
  }
  qref1d[0] = -1.0;
  qref1d[degree] = 1.0;
  // Interior
  for (i = 1; i <= degree/2; i++) {
    // Guess
    xi = cos(PI*(FemeScalar)(i)/((FemeScalar)(degree)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    for (j = 2; j <= degree ; j++) {
      P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton step
    dP2 = (xi*P2 - P0)*(FemeScalar)(degree + 1)/(xi*xi-1.0);
    d2P2 = (2*xi*dP2 - (FemeScalar)(degree*(degree + 1))*P2)/(1.0-xi*xi);
    xi = xi-dP2/d2P2;
    k = 1;
    // Newton to convergence
    while (fabs(dP2)>1e-15 && k < 100) {
      P0 = 1.0;
      P1 = xi;
      for (j = 2; j <= degree; j++) {
        P2 = (((FemeScalar)(2*j-1))*xi*P1-((FemeScalar)(j-1))*P0)/((FemeScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(FemeScalar)(degree + 1)/(xi*xi-1.0);
      d2P2 = (2*xi*dP2 - (FemeScalar)(degree*(degree + 1))*P2)/(1.0-xi*xi);
      xi = xi-dP2/d2P2;
      k++;
    }
    // Save xi, wi
    wi = 2.0/(((FemeScalar)(degree*(degree + 1)))*P2*P2);
    if (qweight1d) {
      qweight1d[i] = wi;
      qweight1d[degree-i] = wi;
    }
    qref1d[i] = -xi;
    qref1d[degree-i]= xi;
  }
  return 0;
}

static int FemeScalarView(const char *name, const char *fpformat, FemeInt m, FemeInt n, const FemeScalar *a, FILE *stream) {
  for (int i=0; i<m; i++) {
    if (m > 1) fprintf(stream, "%12s[%d]:", name, i);
    else fprintf(stream, "%12s:", name);
    for (int j=0; j<n; j++) fprintf(stream, fpformat, a[i*n+j]);
    fputs("\n", stream);
  }
  return 0;
}

int FemeBasisView(FemeBasis basis, FILE *stream) {
  int ierr;

  fprintf(stream, "FemeBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P1d, basis->Q1d);
  ierr = FemeScalarView("qref1d", "\t% 12.8f", 1, basis->Q1d, basis->qref1d, stream);FemeChk(ierr);
  ierr = FemeScalarView("qweight1d", "\t% 12.8f", 1, basis->Q1d, basis->qweight1d, stream);FemeChk(ierr);
  ierr = FemeScalarView("interp1d", "\t% 12.8f", basis->Q1d, basis->P1d, basis->interp1d, stream);FemeChk(ierr);
  ierr = FemeScalarView("grad1d", "\t% 12.8f", basis->Q1d, basis->P1d, basis->grad1d, stream);FemeChk(ierr);
  return 0;
}

int FemeBasisApply(FemeBasis basis, FemeTransposeMode tmode, FemeEvalMode emode, const FemeScalar *u, FemeScalar *v) {
  int ierr;
  if (!basis->Apply) return FemeError(basis->feme, 1, "Backend does not support BasisApply");
  ierr = basis->Apply(basis, tmode, emode, u, v);FemeChk(ierr);
  return 0;
}

int FemeBasisDestroy(FemeBasis *basis) {
  int ierr;

  if (!*basis) return 0;
  if ((*basis)->Destroy) {
    ierr = (*basis)->Destroy(*basis);FemeChk(ierr);
  }
  ierr = FemeFree(&(*basis)->interp1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->grad1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qref1d); FemeChk(ierr);
  ierr = FemeFree(&(*basis)->qweight1d); FemeChk(ierr);
  ierr = FemeFree(basis);FemeChk(ierr);
  return 0;
}
