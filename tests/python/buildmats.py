import numpy as np

def buildmats(qref, qweight):
  P, Q, dim = 6, 4, 2
  interp = np.empty(P*Q, dtype="float64")
  grad = np.empty(dim*P*Q, dtype="float64")

  qref[0] = 0.2
  qref[1] = 0.6
  qref[2] = 1./3.
  qref[3] = 0.2
  qref[4] = 0.2
  qref[5] = 0.2
  qref[6] = 1./3.
  qref[7] = 0.6
  qweight[0] = 25./96.
  qweight[1] = 25./96.
  qweight[2] = -27./96.
  qweight[3] = 25./96.

  # Loop over quadrature points
  for i in range(Q):
    x1 = qref[0*Q+i]
    x2 = qref[1*Q+i]
    # Interp
    interp[i*P+0] =  2.*(x1+x2-1.)*(x1+x2-1./2.)
    interp[i*P+1] = -4.*x1*(x1+x2-1.)
    interp[i*P+2] =  2.*x1*(x1-1./2.)
    interp[i*P+3] = -4.*x2*(x1+x2-1.)
    interp[i*P+4] =  4.*x1*x2
    interp[i*P+5] =  2.*x2*(x2-1./2.)
    # Grad
    grad[(i+0)*P+0] =  2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.)
    grad[(i+Q)*P+0] =  2.*(1.*(x1+x2-1./2.)+(x1+x2-1.)*1.)
    grad[(i+0)*P+1] = -4.*(1.*(x1+x2-1.)+x1*1.)
    grad[(i+Q)*P+1] = -4.*(x1*1.)
    grad[(i+0)*P+2] =  2.*(1.*(x1-1./2.)+x1*1.)
    grad[(i+Q)*P+2] =  2.*0.
    grad[(i+0)*P+3] = -4.*(x2*1.)
    grad[(i+Q)*P+3] = -4.*(1.*(x1+x2-1.)+x2*1.)
    grad[(i+0)*P+4] =  4.*(1.*x2)
    grad[(i+Q)*P+4] =  4.*(x1*1.)
    grad[(i+0)*P+5] =  2.*0.
    grad[(i+Q)*P+5] =  2.*(1.*(x2-1./2.)+x2*1.)

  return interp, grad
