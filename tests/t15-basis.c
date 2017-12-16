#include <ceed.h>
#include <math.h>
// Test a scalar generic basis on a triangle - P2

// np     - number of points
// pts    - (2 x np)
// interp - (np x 6)
// grad   - (np x 2 x 6)
void EvalP2Basis(CeedInt np, const CeedScalar *pts, CeedScalar *interp,
                 CeedScalar *grad) {
  for (CeedInt i = 0; i < np; i++) {
    const CeedScalar x = pts[2*i], y = pts[2*i+1];
    const CeedScalar l1 = 1.-x-y, l2 = x, l3 = y;
    interp[0*np+i] = l1 * (2. * l1 - 1.);
    interp[1*np+i] = l2 * (2. * l2 - 1.);
    interp[2*np+i] = l3 * (2. * l3 - 1.);
    interp[3*np+i] = 4. * l1 * l2;
    interp[4*np+i] = 4. * l2 * l3;
    interp[5*np+i] = 4. * l3 * l1;
    grad[(0*2+0)*np+i] = 4. * (x + y) - 3.;
    grad[(0*2+1)*np+i] = 4. * (x + y) - 3.;
    grad[(1*2+0)*np+i] = 4. * x - 1.;
    grad[(1*2+1)*np+i] = 0.;
    grad[(2*2+0)*np+i] = 0.;
    grad[(2*2+1)*np+i] = 4. * y - 1.;
    grad[(3*2+0)*np+i] = -4. * (2. * x + y - 1.);
    grad[(3*2+1)*np+i] = -4. * x;
    grad[(4*2+0)*np+i] = 4. * y;
    grad[(4*2+1)*np+i] = 4. * x;
    grad[(5*2+0)*np+i] = -4. * y;
    grad[(5*2+1)*np+i] = -4. * (x + 2. * y - 1.);
  }
}

// np     - number of points
// pts    - (2 x np)
// interp - (np)
// grad   - (np x 2)
void EvalPoly(CeedInt np, const CeedScalar *pts, CeedScalar *interp,
              CeedScalar *grad) {
  const CeedScalar c[] = {2., -4., 3., 5., -6., 7.};
  for (CeedInt i = 0; i < np; i++) {
    const CeedScalar x = pts[2*i], y = pts[2*i+1];
    interp[i] = c[0]*x*x + c[1]*x*y + c[2]*y*y + c[3]*x + c[4]*y + c[5];
    grad[0*np+i] = 2.*c[0]*x +    c[1]*y + c[3];
    grad[1*np+i] =    c[1]*x + 2.*c[2]*y + c[4];
  }
}

int main(int argc, char **argv) {
  const CeedScalar a = 0.091576213509770743460, b = 1. - 2.*a;
  const CeedScalar c = 0.44594849091596488632, d = 1. - 2.*c;
  const CeedScalar qpts[] = {a, a, a, b, b, a, c, c, c, d, d, c};
  const CeedScalar wa = 0.054975871827660933819, wb = 1./6. - wa;
  const CeedScalar qwgt[] = {wa, wa, wa, wb, wb, wb};
  const CeedScalar nodes[] = {0., 0., 1., 0., 0., 1., .5, 0., .5, .5, 0., .5};
  const CeedInt ndof = 6, nqpt = 6;
  CeedScalar interp[nqpt*ndof], grad[nqpt*2*ndof];
  EvalP2Basis(nqpt, qpts, interp, grad);

  Ceed ceed;
  CeedBasis basis;

  CeedInit("/cpu/self", &ceed);
  CeedBasisCreateScalarGeneric(ceed, 2, ndof, nqpt, interp, grad, qwgt, &basis);
  CeedBasisView(basis, stdout);
  CeedScalar u[ndof], gu[ndof*2]; // EvalPoly @ nodes
  EvalPoly(ndof, nodes, u, gu);
  CeedScalar v[nqpt], gv[nqpt*2]; // EvalPoly @ qpts
  EvalPoly(nqpt, qpts, v, gv);
  CeedScalar w[nqpt], gw[nqpt*2]; // BasisApply(u) -- 2 EvalModes
  CeedBasisApply(basis, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, w);
  CeedBasisApply(basis, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, gw);
  const CeedScalar eps = 1e-14;
  for (CeedInt i = 0; i < nqpt; i++) {
    if (fabs(v[i]-w[i]) >= eps) {
      printf("%f != %f = p(%f,%f)\n", w[i], v[i], qpts[2*i], qpts[2*i+1]);
    }
  }
  for (CeedInt i = 0; i < nqpt; i++) {
    if (fabs(gv[i]-gw[i]) >= eps) {
      printf("%f != %f = dp/dx(%f,%f)\n", gw[i], gv[i], qpts[2*i], qpts[2*i+1]);
    }
  }
  for (CeedInt i = 0; i < nqpt; i++) {
    const CeedInt j = i+nqpt;
    if (fabs(gv[j]-gw[j]) >= eps) {
      printf("%f != %f = dp/dy(%f,%f)\n", gw[j], gv[j], qpts[2*i], qpts[2*i+1]);
    }
  }
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
