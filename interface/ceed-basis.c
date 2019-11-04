// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed-impl.h>
#include <ceed-backend.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @cond DOXYGEN_SKIP
static struct CeedBasis_private ceed_basis_collocated;
/// @endcond

/// @file
/// Implementation of public CeedBasis interfaces
///
/// @addtogroup CeedBasis
/// @{

/**
  @brief Create a tensor product basis for H^1 discretizations

  @param ceed       A Ceed object where the CeedBasis will be created
  @param dim        Topological dimension
  @param ncomp      Number of field components (1 for scalar fields)
  @param P1d        Number of nodes in one dimension
  @param Q1d        Number of quadrature points in one dimension
  @param interp1d   Row-major Q1d × P1d matrix expressing the values of nodal
                      basis functions at quadrature points
  @param grad1d     Row-major Q1d × P1d matrix expressing derivatives of nodal
                      basis functions at quadrature points
  @param qref1d     Array of length Q1d holding the locations of quadrature points
                      on the 1D reference element [-1, 1]
  @param qweight1d  Array of length Q1d holding the quadrature weights on the
                      reference element
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp, CeedInt P1d,
                            CeedInt Q1d, const CeedScalar *interp1d,
                            const CeedScalar *grad1d, const CeedScalar *qref1d,
                            const CeedScalar *qweight1d, CeedBasis *basis) {
  int ierr;

  if (dim<1)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Basis dimension must be a positive value");
  // LCOV_EXCL_STOP

  if (!ceed->BasisCreateTensorH1) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Basis"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support BasisCreateTensorH1");
    // LCOV_EXCL_STOP

    ierr = CeedBasisCreateTensorH1(delegate, dim, ncomp, P1d,
                                   Q1d, interp1d, grad1d, qref1d,
                                   qweight1d, basis); CeedChk(ierr);
    return 0;
  }
  ierr = CeedCalloc(1,basis); CeedChk(ierr);
  (*basis)->ceed = ceed;
  ceed->refcount++;
  (*basis)->refcount = 1;
  (*basis)->tensorbasis = 1;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  (*basis)->P1d = P1d;
  (*basis)->Q1d = Q1d;
  (*basis)->P = CeedIntPow(P1d, dim);
  (*basis)->Q = CeedIntPow(Q1d, dim);
  ierr = CeedMalloc(Q1d,&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d,&(*basis)->qweight1d); CeedChk(ierr);
  memcpy((*basis)->qref1d, qref1d, Q1d*sizeof(qref1d[0]));
  memcpy((*basis)->qweight1d, qweight1d, Q1d*sizeof(qweight1d[0]));
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*P1d,&(*basis)->grad1d); CeedChk(ierr);
  memcpy((*basis)->interp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));
  memcpy((*basis)->grad1d, grad1d, Q1d*P1d*sizeof(grad1d[0]));
  ierr = ceed->BasisCreateTensorH1(dim, P1d, Q1d, interp1d, grad1d, qref1d,
                                   qweight1d, *basis); CeedChk(ierr);
  return 0;
}

/**
  @brief Create a tensor product Lagrange basis

  @param ceed       A Ceed object where the CeedBasis will be created
  @param dim        Topological dimension of element
  @param ncomp      Number of field components
  @param P          Number of Gauss-Lobatto nodes in one dimension.  The
                      polynomial degree of the resulting Q_k element is k=P-1.
  @param Q          Number of quadrature points in one dimension.
  @param qmode      Distribution of the Q quadrature points (affects order of
                      accuracy for the quadrature)
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                    CeedInt P, CeedInt Q, CeedQuadMode qmode,
                                    CeedBasis *basis) {
  // Allocate
  int ierr, i, j, k;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp1d, *grad1d, *qref1d, *qweight1d;

  if (dim<1)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Basis dimension must be a positive value");
  // LCOV_EXCL_STOP

  ierr = CeedCalloc(P*Q, &interp1d); CeedChk(ierr);
  ierr = CeedCalloc(P*Q, &grad1d); CeedChk(ierr);
  ierr = CeedCalloc(P, &nodes); CeedChk(ierr);
  ierr = CeedCalloc(Q, &qref1d); CeedChk(ierr);
  ierr = CeedCalloc(Q, &qweight1d); CeedChk(ierr);
  // Get Nodes and Weights
  ierr = CeedLobattoQuadrature(P, nodes, NULL); CeedChk(ierr);
  switch (qmode) {
  case CEED_GAUSS:
    ierr = CeedGaussQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
    break;
  case CEED_GAUSS_LOBATTO:
    ierr = CeedLobattoQuadrature(Q, qref1d, qweight1d); CeedChk(ierr);
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
    }
  }
  //  // Pass to CeedBasisCreateTensorH1
  ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P, Q, interp1d, grad1d, qref1d,
                                 qweight1d, basis); CeedChk(ierr);
  ierr = CeedFree(&interp1d); CeedChk(ierr);
  ierr = CeedFree(&grad1d); CeedChk(ierr);
  ierr = CeedFree(&nodes); CeedChk(ierr);
  ierr = CeedFree(&qref1d); CeedChk(ierr);
  ierr = CeedFree(&qweight1d); CeedChk(ierr);
  return 0;
}

/**
  @brief Create a non tensor product basis for H^1 discretizations

  @param ceed       A Ceed object where the CeedBasis will be created
  @param topo       Topology of element, e.g. hypercube, simplex, ect
  @param ncomp      Number of field components (1 for scalar fields)
  @param nnodes     Total number of nodes
  @param nqpts      Total number of quadrature points
  @param interp     Row-major nqpts × nnodes matrix expressing the values of
                      nodal basis functions at quadrature points
  @param grad       Row-major (nqpts x dim) × nnodes matrix expressing
                      derivatives of nodal basis functions at quadrature points
  @param qref       Array of length nqpts holding the locations of quadrature
                      points on the reference element [-1, 1]
  @param qweight    Array of length nqpts holding the quadrature weights on the
                      reference element
  @param[out] basis Address of the variable where the newly created
                      CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo, CeedInt ncomp,
                      CeedInt nnodes, CeedInt nqpts, const CeedScalar *interp,
                      const CeedScalar *grad, const CeedScalar *qref,
                      const CeedScalar *qweight, CeedBasis *basis) {
  int ierr;
  CeedInt P = nnodes, Q = nqpts, dim = 0;

  if (!ceed->BasisCreateH1) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Basis"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support BasisCreateH1");
    // LCOV_EXCL_STOP

    ierr = CeedBasisCreateH1(delegate, topo, ncomp, nnodes,
                             nqpts, interp, grad, qref,
                             qweight, basis); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,basis); CeedChk(ierr);

  ierr = CeedBasisGetTopologyDimension(topo, &dim); CeedChk(ierr);

  (*basis)->ceed = ceed;
  ceed->refcount++;
  (*basis)->refcount = 1;
  (*basis)->tensorbasis = 0;
  (*basis)->dim = dim;
  (*basis)->ncomp = ncomp;
  (*basis)->P = P;
  (*basis)->Q = Q;
  ierr = CeedMalloc(Q*dim,&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedMalloc(Q,&(*basis)->qweight1d); CeedChk(ierr);
  memcpy((*basis)->qref1d, qref, Q*dim*sizeof(qref[0]));
  memcpy((*basis)->qweight1d, qweight, Q*sizeof(qweight[0]));
  ierr = CeedMalloc(Q*P,&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedMalloc(dim*Q*P,&(*basis)->grad1d); CeedChk(ierr);
  memcpy((*basis)->interp1d, interp, Q*P*sizeof(interp[0]));
  memcpy((*basis)->grad1d, grad, dim*Q*P*sizeof(grad[0]));
  ierr = ceed->BasisCreateH1(topo, dim, P, Q, interp, grad, qref,
                             qweight, *basis); CeedChk(ierr);
  return 0;
}

/**
  @brief Construct a Gauss-Legendre quadrature

  @param Q              Number of quadrature points (integrates polynomials of
                          degree 2*Q-1 exactly)
  @param[out] qref1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] qweight1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d, CeedScalar *qweight1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  for (int i = 0; i <= Q/2; i++) {
    // Guess
    xi = cos(PI*(CeedScalar)(2*i+1)/((CeedScalar)(2*Q)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j <= Q; j++) {
      P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
    xi = xi-P2/dP2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(P2)>10*CEED_EPSILON; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j <= Q; j++) {
        P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
      xi = xi-P2/dP2;
    }
    // Save xi, wi
    wi = 2.0/((1.0-xi*xi)*dP2*dP2);
    qweight1d[i] = wi;
    qweight1d[Q-1-i] = wi;
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}

/**
  @brief Construct a Gauss-Legendre-Lobatto quadrature

  @param Q              Number of quadrature points (integrates polynomials of
                          degree 2*Q-3 exactly)
  @param[out] qref1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] qweight1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                          CeedScalar *qweight1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0*atan(1.0);
  // Build qref1d, qweight1d
  // Set endpoints
  if (Q < 2)
    return CeedError(NULL, 1, "Cannot create Lobatto quadrature with Q=%d < 2 points", Q);
  wi = 2.0/((CeedScalar)(Q*(Q-1)));
  if (qweight1d) {
    qweight1d[0] = wi;
    qweight1d[Q-1] = wi;
  }
  qref1d[0] = -1.0;
  qref1d[Q-1] = 1.0;
  // Interior
  for (int i = 1; i <= (Q-1)/2; i++) {
    // Guess
    xi = cos(PI*(CeedScalar)(i)/(CeedScalar)(Q-1));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j < Q; j++) {
      P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton step
    dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
    d2P2 = (2*xi*dP2 - (CeedScalar)(Q*(Q-1))*P2)/(1.0-xi*xi);
    xi = xi-dP2/d2P2;
    // Newton to convergence
    for (int k=0; k<100 && fabs(dP2)>10*CEED_EPSILON; k++) {
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j < Q; j++) {
        P2 = (((CeedScalar)(2*j-1))*xi*P1-((CeedScalar)(j-1))*P0)/((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi*P2 - P0)*(CeedScalar)Q/(xi*xi-1.0);
      d2P2 = (2*xi*dP2 - (CeedScalar)(Q*(Q-1))*P2)/(1.0-xi*xi);
      xi = xi-dP2/d2P2;
    }
    // Save xi, wi
    wi = 2.0/(((CeedScalar)(Q*(Q-1)))*P2*P2);
    if (qweight1d) {
      qweight1d[i] = wi;
      qweight1d[Q-1-i] = wi;
    }
    qref1d[i] = -xi;
    qref1d[Q-1-i]= xi;
  }
  return 0;
}

/**
  @brief View an array stored in a CeedBasis

  @param name      Name of array
  @param fpformat  Printing format
  @param m         Number of rows in array
  @param n         Number of columns in array
  @param a         Array to be viewed
  @param stream    Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedScalarView(const char *name, const char *fpformat, CeedInt m,
                          CeedInt n, const CeedScalar *a, FILE *stream) {
  for (int i=0; i<m; i++) {
    if (m > 1)
      fprintf(stream, "%12s[%d]:", name, i);
    else
      fprintf(stream, "%12s:", name);
    for (int j=0; j<n; j++)
      fprintf(stream, fpformat, fabs(a[i*n+j]) > 1E-14 ? a[i*n+j] : 0);
    fputs("\n", stream);
  }
  return 0;
}

/**
  @brief View a CeedBasis

  @param basis  CeedBasis to view
  @param stream Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisView(CeedBasis basis, FILE *stream) {
  int ierr;

  if (basis->tensorbasis) {
    fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P1d,
            basis->Q1d);
    ierr = CeedScalarView("qref1d", "\t% 12.8f", 1, basis->Q1d, basis->qref1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("qweight1d", "\t% 12.8f", 1, basis->Q1d,
                          basis->qweight1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("interp1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                          basis->interp1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad1d", "\t% 12.8f", basis->Q1d, basis->P1d,
                          basis->grad1d, stream); CeedChk(ierr);
  } else {
    fprintf(stream, "CeedBasis: dim=%d P=%d Q=%d\n", basis->dim, basis->P,
            basis->Q);
    ierr = CeedScalarView("qref", "\t% 12.8f", 1, basis->Q*basis->dim,
                          basis->qref1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("qweight", "\t% 12.8f", 1, basis->Q, basis->qweight1d,
                          stream); CeedChk(ierr);
    ierr = CeedScalarView("interp", "\t% 12.8f", basis->Q, basis->P,
                          basis->interp1d, stream); CeedChk(ierr);
    ierr = CeedScalarView("grad", "\t% 12.8f", basis->dim*basis->Q, basis->P,
                          basis->grad1d, stream); CeedChk(ierr);
  }
  return 0;
}

/**
  @brief Compute Householder reflection

    Computes A = (I - b v v^T) A
    where A is an mxn matrix indexed as A[i*row + j*col]

  @param[in,out] A  Matrix to apply Householder reflection to, in place
  @param v          Householder vector
  @param b          Scaling factor
  @param m          Number of rows in A
  @param n          Number of columns in A
  @param row        Row stride
  @param col        Col stride

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedHouseholderReflect(CeedScalar *A, const CeedScalar *v,
                                  CeedScalar b, CeedInt m, CeedInt n,
                                  CeedInt row, CeedInt col) {
  for (CeedInt j=0; j<n; j++) {
    CeedScalar w = A[0*row + j*col];
    for (CeedInt i=1; i<m; i++)
      w += v[i] * A[i*row + j*col];
    A[0*row + j*col] -= b * w;
    for (CeedInt i=1; i<m; i++)
      A[i*row + j*col] -= b * w * v[i];
  }
  return 0;
}

/**
  @brief Apply Householder Q matrix

    Compute A = Q A where Q is mxm and A is mxn.

  @param[in,out] A  Matrix to apply Householder Q to, in place
  @param Q          Householder Q matrix
  @param tau        Householder scaling factors
  @param tmode      Transpose mode for application
  @param m          Number of rows in A
  @param n          Number of columns in A
  @param k          Number of elementary reflectors in Q, k<m
  @param row        Row stride in A
  @param col        Col stride in A

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedHouseholderApplyQ(CeedScalar *A, const CeedScalar *Q,
                                 const CeedScalar *tau, CeedTransposeMode tmode,
                                 CeedInt m, CeedInt n, CeedInt k,
                                 CeedInt row, CeedInt col) {
  CeedScalar v[m];
  for (CeedInt ii=0; ii<k; ii++) {
    CeedInt i = tmode == CEED_TRANSPOSE ? ii : k-1-ii;
    for (CeedInt j=i+1; j<m; j++)
      v[j] = Q[j*k+i];
    // Apply Householder reflector (I - tau v v^T) collograd1d^T
    CeedHouseholderReflect(&A[i*row], &v[i], tau[i], m-i, n, row, col);
  }
  return 0;
}

/**
  @brief Compute Givens rotation

    Computes A = G A (or G^T A in transpose mode)
    where A is an mxn matrix indexed as A[i*n + j*m]

  @param[in,out] A  Row major matrix to apply Givens rotation to, in place
  @param c          Cosine factor
  @param s          Sine factor
  @param i          First row/column to apply rotation
  @param k          Second row/column to apply rotation
  @param m          Number of rows in A
  @param n          Number of columns in A

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedGivensRotation(CeedScalar *A, CeedScalar c, CeedScalar s,
                              CeedTransposeMode tmode, CeedInt i, CeedInt k,
                              CeedInt m, CeedInt n) {
  CeedInt stridej = 1, strideik = m, numits = n;
  if (tmode == CEED_NOTRANSPOSE) {
    stridej = n; strideik = 1; numits = m;
  }

  // Apply rotation
  for (CeedInt j=0; j<numits; j++) {
    CeedScalar tau1 = A[i*strideik+j*stridej], tau2 = A[k*strideik+j*stridej];
    A[i*strideik+j*stridej] = c*tau1 - s*tau2;
    A[k*strideik+j*stridej] = s*tau1 + c*tau2;
  }

  return 0;
}

/**
  @brief Return QR Factorization of matrix

  @param ceed         A Ceed object currently in use
  @param[in,out] mat  Row-major matrix to be factorized in place
  @param[in,out] tau  Vector of length m of scaling factors
  @param m            Number of rows
  @param n            Number of columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau,
                        CeedInt m, CeedInt n) {
  CeedScalar v[m];

  // Check m >= n
  if (n > m)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Cannot compute QR factorization with n > m");
  // LCOV_EXCL_STOP

  for (CeedInt i=0; i<n; i++) {
    // Calculate Householder vector, magnitude
    CeedScalar sigma = 0.0;
    v[i] = mat[i+n*i];
    for (CeedInt j=i+1; j<m; j++) {
      v[j] = mat[i+n*j];
      sigma += v[j] * v[j];
    }
    CeedScalar norm = sqrt(v[i]*v[i] + sigma); // norm of v[i:m]
    CeedScalar Rii = -copysign(norm, v[i]);
    v[i] -= Rii;
    // norm of v[i:m] after modification above and scaling below
    //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
    //   tau = 2 / (norm*norm)
    tau[i] = 2 * v[i]*v[i] / (v[i]*v[i] + sigma);

    for (CeedInt j=i+1; j<m; j++)
      v[j] /= v[i];

    // Apply Householder reflector to lower right panel
    CeedHouseholderReflect(&mat[i*n+i+1], &v[i], tau[i], m-i, n-i-1, n, 1);
    // Save v
    mat[i+n*i] = Rii;
    for (CeedInt j=i+1; j<m; j++)
      mat[i+n*j] = v[j];
  }

  return 0;
}

/**
  @brief Return symmetric Schur decomposition of the symmetric matrix mat via
           symmetric QR factorization

  @param ceed         A Ceed object for error handling
  @param[in,out] mat  Row-major matrix to be factorized in place
  @param[out] lambda  Vector of length n of eigenvalues
  @param n            Number of rows/columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedSymmetricSchurDecomposition(Ceed ceed, CeedScalar *mat,
                                    CeedScalar *lambda, CeedInt n) {
  // Check bounds for clang-tidy
  if (n<2)
    // LCOV_EXCL_START
    return CeedError(ceed, 1,
                     "Cannot compute symmetric Schur decomposition of scalars");
  // LCOV_EXCL_STOP

  CeedScalar v[n-1], tau[n-1], matT[n*n];

  // Copy mat to matT and set mat to I
  memcpy(matT, mat, n*n*sizeof(mat[0]));
  for (CeedInt i=0; i<n; i++)
    for (CeedInt j=0; j<n; j++)
      mat[j+n*i] = (i==j) ? 1 : 0;

  // Reduce to tridiagonal
  for (CeedInt i=0; i<n-1; i++) {
    // Calculate Householder vector, magnitude
    CeedScalar sigma = 0.0;
    v[i] = matT[i+n*(i+1)];
    for (CeedInt j=i+1; j<n-1; j++) {
      v[j] = matT[i+n*(j+1)];
      sigma += v[j] * v[j];
    }
    CeedScalar norm = sqrt(v[i]*v[i] + sigma); // norm of v[i:n-1]
    CeedScalar Rii = -copysign(norm, v[i]);
    v[i] -= Rii;
    // norm of v[i:m] after modification above and scaling below
    //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
    //   tau = 2 / (norm*norm)
    if (sigma > 10*CEED_EPSILON)
      tau[i] = 2 * v[i]*v[i] / (v[i]*v[i] + sigma);
    else
      tau[i] = 0;

    for (CeedInt j=i+1; j<n-1; j++)
      v[j] /= v[i];

    // Update sub and super diagonal
    matT[i+n*(i+1)] = Rii;
    matT[(i+1)+n*i] = Rii;
    for (CeedInt j=i+2; j<n; j++) {
      matT[i+n*j] = 0; matT[j+n*i] = 0;
    }
    // Apply symmetric Householder reflector to lower right panel
    CeedHouseholderReflect(&matT[(i+1)+n*(i+1)], &v[i], tau[i],
                           n-(i+1), n-(i+1), n, 1);
    CeedHouseholderReflect(&matT[(i+1)+n*(i+1)], &v[i], tau[i],
                           n-(i+1), n-(i+1), 1, n);
    // Save v
    for (CeedInt j=i+1; j<n-1; j++) {
      matT[i+n*(j+1)] = v[j];
    }
  }
  // Backwards accumulation of Q
  for (CeedInt i=n-2; i>=0; i--) {
    v[i] = 1;
    for (CeedInt j=i+1; j<n-1; j++) {
      v[j] = matT[i+n*(j+1)];
      matT[i+n*(j+1)] = 0;
    }
    CeedHouseholderReflect(&mat[(i+1)+n*(i+1)], &v[i], tau[i],
                           n-(i+1), n-(i+1), n, 1);
  }

  // Reduce sub and super diagonal
  CeedInt p = 0, q = 0, itr = 0, maxitr = n*n*n;
  CeedScalar tol = 10*CEED_EPSILON;

  while (q < n && itr < maxitr) {
    // Update p, q, size of reduced portions of diagonal
    p = 0; q = 0;
    for (CeedInt i=n-2; i>=0; i--) {
      if (fabs(matT[i+n*(i+1)]) < tol)
        q += 1;
      else
        break;
    }
    for (CeedInt i=0; i<n-1-q; i++) {
      if (fabs(matT[i+n*(i+1)]) < tol)
        p += 1;
      else
        break;
    }
    if (q == n-1) break; // Finished reducing

    // Reduce tridiagonal portion
    CeedScalar tnn = matT[(n-1-q)+n*(n-1-q)],
               tnnm1 = matT[(n-2-q)+n*(n-1-q)];
    CeedScalar d = (matT[(n-2-q)+n*(n-2-q)] - tnn)/2;
    CeedScalar mu = tnn - tnnm1*tnnm1 /
                    (d + copysign(sqrt(d*d + tnnm1*tnnm1), d));
    CeedScalar x = matT[p+n*p] - mu;
    CeedScalar z = matT[p+n*(p+1)];
    for (CeedInt k=p; k<n-1-q; k++) {
      // Compute Givens rotation
      CeedScalar c = 1, s = 0;
      if (fabs(z) > tol) {
        if (fabs(z) > fabs(x)) {
          CeedScalar tau = -x/z;
          s = 1/sqrt(1+tau*tau), c = s*tau;
        } else {
          CeedScalar tau = -z/x;
          c = 1/sqrt(1+tau*tau), s = c*tau;
        }
      }

      // Apply Givens rotation to T
      CeedGivensRotation(matT, c, s, CEED_NOTRANSPOSE, k, k+1, n, n);
      CeedGivensRotation(matT, c, s, CEED_TRANSPOSE, k, k+1, n, n);

      // Apply Givens rotation to Q
      CeedGivensRotation(mat, c, s, CEED_NOTRANSPOSE, k, k+1, n, n);

      // Update x, z
      if (k < n-q-2) {
        x = matT[k+n*(k+1)];
        z = matT[k+n*(k+2)];
      }
    }
    itr++;
  }
  // Save eigenvalues
  for (CeedInt i=0; i<n; i++)
    lambda[i] = matT[i+n*i];

  // Check convergence
  if (itr == maxitr && q < n-1)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Symmetric QR failed to converge");
  // LCOV_EXCL_STOP

  return 0;
}

/**
  @brief Return C = A B

  @param[in] matA     Row-major matrix A
  @param[in] matB     Row-major matrix B
  @param[out] matC    Row-major output matrix C
  @param m            Number of rows of C
  @param n            Number of columns of C
  @param kk           Number of columns of A/rows of B

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedMatrixMultiply(Ceed ceed, CeedScalar *matA, CeedScalar *matB,
                       CeedScalar *matC, CeedInt m, CeedInt n, CeedInt kk) {
  for (CeedInt i=0; i<m; i++)
    for (CeedInt j=0; j<n; j++) {
      CeedScalar sum = 0;
      for (CeedInt k=0; k<kk; k++)
        sum += matA[k+i*kk]*matB[j+k*n];
      matC[j+i*n] = sum;
    }
  return 0;
}

/**
  @brief Return Simultaneous Diagonalization of two matrices. This solves the
           generalized eigenvalue problem A x = lambda B x, where A and B
           are symmetric and B is positive definite. We generate the matrix X
           and vector Lambda such that X^T A X = Lambda and X^T B X = I. This
           is equivalent to the LAPACK routine 'sygv' with TYPE = 1.

  @param ceed         A Ceed object for error handling
  @param[in] matA     Row-major matrix to be factorized with eigenvalues
  @param[in] matB     Row-major matrix to be factorized to identity
  @param[out] x       Row-major orthogonal matrix
  @param[out] lambda  Vector of length n of generalized eigenvalues
  @param n            Number of rows/columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedSimultaneousDiagonalization(Ceed ceed, CeedScalar *matA,
                                    CeedScalar *matB, CeedScalar *x,
                                    CeedScalar *lambda, CeedInt n) {
  int ierr;
  CeedScalar matC[n*n], matG[n*n], vecD[n];

  // Compute B = G D G^T
  memcpy(matG, matB, n*n*sizeof(matB[0]));
  ierr = CeedSymmetricSchurDecomposition(ceed, matG, vecD, n); CeedChk(ierr);
  for (CeedInt i=0; i<n; i++)
    vecD[i] = sqrt(vecD[i]);

  // Compute C = (G D^1/2)^-1 A (G D^1/2)^-T
  //           = D^-1/2 G^T A G D^-1/2
  for (CeedInt i=0; i<n; i++)
    for (CeedInt j=0; j<n; j++)
      matC[j+i*n] = matG[i+j*n] / vecD[i];
  CeedMatrixMultiply(ceed, matC, matA, x, n, n, n);
  for (CeedInt i=0; i<n; i++)
    for (CeedInt j=0; j<n; j++)
      matG[j+i*n] = matG[j+i*n] / vecD[j];
  CeedMatrixMultiply(ceed, x, matG, matC, n, n, n);

  // Compute Q^T C Q = lambda
  ierr = CeedSymmetricSchurDecomposition(ceed, matC, lambda, n); CeedChk(ierr);

  // Set x = (G D^1/2)^-T Q
  //       = G D^-1/2 Q
  CeedMatrixMultiply(ceed, matG, matC, x, n, n, n);

  return 0;
}

/**
  @brief Return collocated grad matrix

  @param basis           CeedBasis
  @param[out] collograd1d Row-major Q1d × Q1d matrix expressing derivatives of
                           basis functions at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCollocatedGrad(CeedBasis basis, CeedScalar *collograd1d) {
  int i, j, k;
  Ceed ceed;
  CeedInt ierr, P1d=(basis)->P1d, Q1d=(basis)->Q1d;
  CeedScalar *interp1d, *grad1d, tau[Q1d];

  ierr = CeedMalloc(Q1d*P1d, &interp1d); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*P1d, &grad1d); CeedChk(ierr);
  memcpy(interp1d, (basis)->interp1d, Q1d*P1d*sizeof(basis)->interp1d[0]);
  memcpy(grad1d, (basis)->grad1d, Q1d*P1d*sizeof(basis)->interp1d[0]);

  // QR Factorization, interp1d = Q R
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  ierr = CeedQRFactorization(ceed, interp1d, tau, Q1d, P1d); CeedChk(ierr);

  // Apply Rinv, collograd1d = grad1d Rinv
  for (i=0; i<Q1d; i++) { // Row i
    collograd1d[Q1d*i] = grad1d[P1d*i]/interp1d[0];
    for (j=1; j<P1d; j++) { // Column j
      collograd1d[j+Q1d*i] = grad1d[j+P1d*i];
      for (k=0; k<j; k++)
        collograd1d[j+Q1d*i] -= interp1d[j+P1d*k]*collograd1d[k+Q1d*i];
      collograd1d[j+Q1d*i] /= interp1d[j+P1d*j];
    }
    for (j=P1d; j<Q1d; j++)
      collograd1d[j+Q1d*i] = 0;
  }

  // Apply Qtranspose, collograd = collograd Qtranspose
  CeedHouseholderApplyQ(collograd1d, interp1d, tau, CEED_NOTRANSPOSE,
                        Q1d, Q1d, P1d, 1, Q1d);

  ierr = CeedFree(&interp1d); CeedChk(ierr);
  ierr = CeedFree(&grad1d); CeedChk(ierr);

  return 0;
}

/**
  @brief Apply basis evaluation from nodes to quadrature points or vice-versa

  @param basis  CeedBasis to evaluate
  @param nelem  The number of elements to apply the basis evaluation to;
                  the backend will specify the ordering in
                  ElemRestrictionCreateBlocked
  @param tmode  \ref CEED_NOTRANSPOSE to evaluate from nodes to quadrature
                  points, \ref CEED_TRANSPOSE to apply the transpose, mapping
                  from quadrature points to nodes
  @param emode  \ref CEED_EVAL_NONE to use values directly,
                  \ref CEED_EVAL_INTERP to use interpolated values,
                  \ref CEED_EVAL_GRAD to use gradients,
                  \ref CEED_EVAL_WEIGHT to use quadrature weights.
  @param[in] u  Input CeedVector
  @param[out] v Output CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisApply(CeedBasis basis, CeedInt nelem, CeedTransposeMode tmode,
                   CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  CeedInt ulength = 0, vlength, nnodes, nqpt;
  if (!basis->Apply)
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "Backend does not support BasisApply");
  // LCOV_EXCL_STOP

  // Check compatibility of topological and geometrical dimensions
  ierr = CeedBasisGetNumNodes(basis, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  ierr = CeedVectorGetLength(v, &vlength); CeedChk(ierr);

  if (u) {
    ierr = CeedVectorGetLength(u, &ulength); CeedChk(ierr);
  }

  if ((tmode == CEED_TRANSPOSE && (vlength%nnodes != 0 || ulength%nqpt != 0)) ||
      (tmode == CEED_NOTRANSPOSE && (ulength%nnodes != 0 || vlength%nqpt != 0)))
    return CeedError(basis->ceed, 1, "Length of input/output vectors "
                     "incompatible with basis dimensions");

  ierr = basis->Apply(basis, nelem, tmode, emode, u, v); CeedChk(ierr);
  return 0;
}

/**
  @brief Get Ceed associated with a CeedBasis

  @param basis      CeedBasis
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCeed(CeedBasis basis, Ceed *ceed) {
  *ceed = basis->ceed;
  return 0;
};

/**
  @brief Get dimension for given CeedBasis

  @param basis     CeedBasis
  @param[out] dim  Variable to store dimension of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDimension(CeedBasis basis, CeedInt *dim) {
  *dim = basis->dim;
  return 0;
};

/**
  @brief Get tensor status for given CeedBasis

  @param basis        CeedBasis
  @param[out] tensor  Variable to store tensor status

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTensorStatus(CeedBasis basis, bool *tensor) {
  *tensor = basis->tensorbasis;
  return 0;
};

/**
  @brief Get number of components for given CeedBasis

  @param basis        CeedBasis
  @param[out] numcomp Variable to store number of components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumComponents(CeedBasis basis, CeedInt *numcomp) {
  *numcomp = basis->ncomp;
  return 0;
};

/**
  @brief Get total number of nodes (in 1 dimension) of a CeedBasis

  @param basis     CeedBasis
  @param[out] P1d  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumNodes1D(CeedBasis basis, CeedInt *P1d) {
  if (!basis->tensorbasis)
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "Cannot supply P1d for non-tensor basis");
  // LCOV_EXCL_STOP

  *P1d = basis->P1d;
  return 0;
}

/**
  @brief Get total number of quadrature points (in 1 dimension) of a CeedBasis

  @param basis     CeedBasis
  @param[out] Q1d  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumQuadraturePoints1D(CeedBasis basis, CeedInt *Q1d) {
  if (!basis->tensorbasis)
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "Cannot supply Q1d for non-tensor basis");
  // LCOV_EXCL_STOP

  *Q1d = basis->Q1d;
  return 0;
}

/**
  @brief Get total number of nodes (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] P  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P) {
  *P = basis->P;
  return 0;
}

/**
  @brief Get total number of quadrature points (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] Q  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q) {
  *Q = basis->Q;
  return 0;
}

/**
  @brief Get reference coordinates of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis      CeedBasis
  @param[out] qref  Variable to store reference coordinates of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQRef(CeedBasis basis, CeedScalar **qref) {
  *qref = basis->qref1d;
  return 0;
}

/**
  @brief Get quadrature weights of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis         CeedBasis
  @param[out] qweight  Variable to store quadrature weights

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQWeights(CeedBasis basis, CeedScalar **qweight) {
  *qweight = basis->qweight1d;
  return 0;
}

/**
  @brief Get interpolation matrix of a CeedBasis

  @param basis       CeedBasis
  @param[out] interp Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetInterp(CeedBasis basis, CeedScalar **interp) {
  *interp = basis->interp1d;
  return 0;
}

/**
  @brief Get gradient matrix of a CeedBasis

  @param basis      CeedBasis
  @param[out] grad  Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad(CeedBasis basis, CeedScalar **grad) {
  *grad = basis->grad1d;
  return 0;
}

/**
  @brief Get value in CeedEvalMode matrix of a CeedBasis

  @param basis       CeedBasis
  @param[in] emode   CeedEvalMode to retrieve value
  @param[in] node    Node (column) to retrieve value
  @param[in] qpt     Quadrature point (row) to retrieve value
  @param[in] dim     Dimension to retrieve value for, for CEED_EVAL_GRAD
  @param[out] value  Variable to store value

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetValue(CeedBasis basis, CeedEvalMode emode, CeedInt qpt,
                      CeedInt node, CeedInt dim, CeedScalar *value) {
  bool tensor = basis->tensorbasis;

  switch (emode) {
  case CEED_EVAL_NONE:
    if (node == qpt)
      *value = 0.0;
    else
      *value = 1.0;
    break;
  case CEED_EVAL_INTERP: {
    CeedScalar *interp = basis->interp1d;

    if (tensor) {
      CeedInt n, q;

      *value = 1.0;
      for (CeedInt d=0; d<basis->dim; d++) {
        n = (node / CeedIntPow(basis->P1d, d)) % basis->P1d;
        q = (qpt / CeedIntPow(basis->Q1d, d)) % basis->Q1d;
        *value *= interp[q*(basis->P1d)+n];
      }
    } else {
      *value = interp[qpt*(basis->P)+node];
    }
  } break;
  case CEED_EVAL_GRAD: {
    CeedScalar *grad = basis->grad1d;

    if (tensor) {
      CeedInt n, q;
      CeedScalar *interp = basis->interp1d;

      *value = 1.0;
      for (CeedInt d=0; d<basis->dim; d++) {
        n = (node / CeedIntPow(basis->P1d, d)) % basis->P1d;
        q = (qpt / CeedIntPow(basis->Q1d, d)) % basis->Q1d;
        if (d == dim)
          *value *= grad[q*(basis->P1d)+n];
        else
          *value *= interp[q*(basis->P1d)+n];
      }
    } else {
      *value = grad[(dim*(basis->Q)+qpt)*(basis->P)+node];
    }
  } break;
  case CEED_EVAL_WEIGHT:
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "CEED_EVAL_WEIGHT does not make sense in "
                     "this context");
  // LCOV_EXCL_STOP
  case CEED_EVAL_DIV:
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "CEED_EVAL_DIV not supported");
  // LCOV_EXCL_STOP
  case CEED_EVAL_CURL:
    // LCOV_EXCL_START
    return CeedError(basis->ceed, 1, "CEED_EVAL_CURL not supported");
    // LCOV_EXCL_STOP
  }
  return 0;
}

/**
  @brief Get backend data of a CeedBasis

  @param basis      CeedBasis
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetData(CeedBasis basis, void **data) {
  *data = basis->data;
  return 0;
}

/**
  @brief Set backend data of a CeedBasis

  @param[out] basis CeedBasis
  @param data       Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisSetData(CeedBasis basis, void **data) {
  basis->data = *data;
  return 0;
}

/**
  @brief Get CeedTensorContract of a CeedBasis

  @param basis          CeedBasis
  @param[out] contract  Variable to store CeedTensorContract

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTensorContract(CeedBasis basis, CeedTensorContract *contract) {
  *contract = basis->contract;
  return 0;
}

/**
  @brief Set CeedTensorContract of a CeedBasis

  @param[out] basis     CeedBasis
  @param contract       CeedTensorContract to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisSetTensorContract(CeedBasis basis, CeedTensorContract *contract) {
  basis->contract = *contract;
  return 0;
}

/**
  @brief Get dimension for given CeedElemTopology

  @param topo      CeedElemTopology
  @param[out] dim  Variable to store dimension of topology

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTopologyDimension(CeedElemTopology topo, CeedInt *dim) {
  *dim = (CeedInt) topo >> 16;
  return 0;
};

/**
  @brief Destroy a CeedBasis

  @param basis CeedBasis to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedBasisDestroy(CeedBasis *basis) {
  int ierr;

  if (!*basis || --(*basis)->refcount > 0)
    return 0;
  if ((*basis)->Destroy) {
    ierr = (*basis)->Destroy(*basis); CeedChk(ierr);
  }
  ierr = CeedFree(&(*basis)->interp1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->grad1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qref1d); CeedChk(ierr);
  ierr = CeedFree(&(*basis)->qweight1d); CeedChk(ierr);
  ierr = CeedDestroy(&(*basis)->ceed); CeedChk(ierr);
  ierr = CeedFree(basis); CeedChk(ierr);
  return 0;
}

/// @cond DOXYGEN_SKIP
// Indicate that the quadrature points are collocated with the nodes
CeedBasis CEED_BASIS_COLLOCATED = &ceed_basis_collocated;
/// @endcond
/// @}
