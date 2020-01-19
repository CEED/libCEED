# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

from _ceed_cffi import ffi, lib
import tempfile
import numpy as np
from abc import ABC
from .ceed_constants import TRANSPOSE, NOTRANSPOSE

# ------------------------------------------------------------------------------
class Basis(ABC):
  """Ceed Basis: finite element basis objects."""

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Representation
  def __repr__(self):
    return "<CeedBasis instance at " + hex(id(self)) + ">"

  # String conversion for print() to stdout
  def __str__(self):
    """View a Basis via print()."""

    # libCEED call
    with tempfile.NamedTemporaryFile() as key_file:
      with open(key_file.name, 'r+') as stream_file:
        stream = ffi.cast("FILE *", stream_file)

        lib.CeedBasisView(self._pointer[0], stream)

        stream_file.seek(0)
        out_string = stream_file.read()

    return out_string

  # Apply Basis
  def apply(self, nelem, emode, u, v, tmode=NOTRANSPOSE):
    """Apply basis evaluation from nodes to quadrature points or vice versa.

       Args:
         nelem: the number of elements to apply the basis evaluation to;
                  the backend will specify the ordering in a
                  BlockedElemRestriction
         emode: basis evaluation mode
         u: input vector
         v: output vector
         **tmode: CEED_NOTRANSPOSE to evaluate from nodes to quadrature
                    points, CEED_TRANSPOSE to apply the transpose, mapping
                    from quadrature points to nodes; default CEED_NOTRANSPOSE"""

    # libCEED call
    lib.CeedBasisApply(self._pointer[0], nelem, tmode, emode,
                       u._pointer[0], v._pointer[0])

  # Transpose a Basis
  @property
  def T(self):
    """Transpose a Basis."""

    return TransposeBasis(self)

  # Transpose a Basis
  @property
  def transpose(self):
    """Transpose a Basis."""

    return TransposeBasis(self)

  # Get number of nodes
  def get_num_nodes(self):
    """Get total number of nodes (in dim dimensions) of a Basis.

       Returns:
         num_nodes: total number of nodes"""

    # Setup argument
    p_pointer = ffi.new("CeedInt *")

    # libCEED call
    lib.CeedBasisGetNumNodes(self._pointer[0], p_pointer)

    return p_pointer[0]

  # Get number of quadrature points
  def get_num_quadrature_points(self):
    """Get total number of quadrature points (in dim dimensions) of a Basis.

       Returns:
         num_qpts: total number of quadrature points"""

    # Setup argument
    q_pointer = ffi.new("CeedInt *")

    # libCEED call
    lib.CeedBasisGetNumQuadraturePoints(self._pointer[0], q_pointer)

    return q_pointer[0]

  # Gauss quadrature
  @staticmethod
  def gauss_quadrature(q):
    """Construct a Gauss-Legendre quadrature.

       Args:
         Q: number of quadrature points (integrates polynomials of
              degree 2*Q-1 exactly)

       Returns:
         (qref1d, qweight1d): array of length Q to hold the abscissa on [-1, 1]
                                and array of length Q to hold the weights"""

    # Setup arguments
    qref1d = np.empty(q, dtype="float64")
    qweight1d = np.empy(q, dtype="float64")

    qref1d_pointer = ffi.new("CeedScalar *")
    qref1d_pointer = ffi.cast("CeedScalar *", qref1d.__array_interface__['data'][0])

    qweight1d_pointer = ffi.new("CeedScalar *")
    qweight1d_pointer = ffi.cast("CeedScalar *", qweight1d.__array_interface__['data'][0])

    # libCEED call
    lib.CeedGaussQuadrature(q, qref1d_pointer, qweight1d_pointer)

    return qref1d, qweight1d

  # Lobatto quadrature
  @staticmethod
  def lobatto_quadrature(q):
    """Construct a Gauss-Legendre-Lobatto quadrature.

       Args:
         q: number of quadrature points (integrates polynomials of
              degree 2*Q-3 exactly)

       Returns:
         (qref1d, qweight1d): array of length Q to hold the abscissa on [-1, 1]
                                and array of length Q to hold the weights"""

    # Setup arguments
    qref1d = np.empty(q, dtype="float64")
    qref1d_pointer = ffi.new("CeedScalar *")
    qref1d_pointer = ffi.cast("CeedScalar *", qref1d.__array_interface__['data'][0])

    qweight1d = np.empy(q, dtype="float64")
    qweight1d_pointer = ffi.new("CeedScalar *")
    qweight1d_pointer = ffi.cast("CeedScalar *", qweight1d.__array_interface__['data'][0])

    # libCEED call
    lib.CeedLobattoQuadrature(q, qref1d_pointer, qweight1d_pointer)

    return qref1d, qweight1d

  # QR factorization
  @staticmethod
  def qr_factorization(ceed, mat, tau, m, n):
    """Return QR Factorization of a matrix.

       Args:
         ceed: Ceed context currently in use
         *mat: Numpy array holding the row-major matrix to be factorized in place
         *tau: Numpy array to hold the vector of lengt m of scaling factors
         m: number of rows
         n: numbef of columns"""

    # Setup arguments
    mat_pointer = ffi.new("CeedScalar *")
    mat_pointer = ffi.cast("CeedScalar *", mat.__array_interface__['data'][0])

    tau_pointer = ffi.new("CeedScalar *")
    tau_pointer = ffi.cast("CeedScalar *", tau.__array_interface__['data'][0])

    # libCEED call
    lib.CeedQRFactorization(ceed._pointer[0], mat_pointer, tau_pointer, m, n)

    return mat, tau

  # Symmetric Schur decomposition
  @staticmethod
  def symmetric_schur_decomposition(ceed, mat, n):
    """Return symmetric Schur decomposition of a symmetric matrix
         via symmetric QR factorization.

       Args:
         ceed: Ceed context currently in use
         *mat: Numpy array holding the row-major matrix to be factorized in place
         n: number of rows/columns

       Returns:
         lbda: Numpy array of length n holding eigenvalues"""

    # Setup arguments
    mat_pointer = ffi.new("CeedScalar *")
    mat_pointer = ffi.cast("CeedScalar *", mat.__array_interface__['data'][0])

    lbda = np.empty(n, dtype="float64")
    l_pointer = ffi.new("CeedScalar *")
    l_pointer = ffi.cast("CeedScalar *", lbda.__array_interface__['data'][0])

    # libCEED call
    lib.CeedSymmetricSchurDecomposition(ceed._pointer[0], mat_pointer, l_pointer, n)

    return lbda

  # Simultaneous Diagonalization
  @staticmethod
  def simultaneous_diagonalization(ceed, matA, matB, n):
    """Return Simultaneous Diagonalization of two matrices.

       Args:
         ceed: Ceed context currently in use
         *matA: Numpy array holding the row-major matrix to be factorized with
                  eigenvalues
         *matB: Numpy array holding the row-major matrix to be factorized to identity
         n: number of rows/columns

       Returns:
         (x, lbda): Numpy array holding the row-major orthogonal matrix and
                      Numpy array holding the vector of length n of generalized
                      eigenvalues"""

    # Setup arguments
    matA_pointer = ffi.new("CeedScalar *")
    matA_pointer = ffi.cast("CeedScalar *", matA.__array_interface__['data'][0])

    matB_pointer = ffi.new("CeedScalar *")
    matB_pointer = ffi.cast("CeedScalar *", matB.__array_interface__['data'][0])

    lbda = np.empty(n, dtype="float64")
    l_pointer = ffi.new("CeedScalar *")
    l_pointer = ffi.cast("CeedScalar *", lbda.__array_interface__['data'][0])

    x = np.empty(n*n, dtype="float64")
    x_pointer = ffi.new("CeedScalar *")
    x_pointer = ffi.cast("CeedScalar *", x.__array_interface__['data'][0])

    # libCEED call
    lib.CeedSimultaneousDiagonalization(ceed._pointer[0], matA_pointer, matB_pointer,
                                        x_pointer, l_pointer, n)

    return x, lbda

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedBasisDestroy(self._pointer)

# ------------------------------------------------------------------------------
class BasisTensorH1(Basis):
  """Ceed Tensor H1 Basis: finite element tensor-product basis objects for
       H^1 discretizations."""

  # Constructor
  def __init__(self, ceed, dim, ncomp, P1d, Q1d, interp1d, grad1d,
               qref1d, qweight1d):

    # Setup arguments
    self._pointer = ffi.new("CeedBasis *")

    self._ceed = ceed

    interp1d_pointer = ffi.new("CeedScalar *")
    interp1d_pointer = ffi.cast("CeedScalar *", interp1d.__array_interface__['data'][0])

    grad1d_pointer = ffi.new("CeedScalar *")
    grad1d_pointer = ffi.cast("CeedScalar *", grad1d.__array_interface__['data'][0])

    qref1d_pointer = ffi.new("CeedScalar *")
    qref1d_pointer = ffi.cast("CeedScalar *", qref1d.__array_interface__['data'][0])

    qweight1d_pointer = ffi.new("CeedScalar *")
    qweight1d_pointer = ffi.cast("CeedScalar *", qweight1d.__array_interface__['data'][0])

    # libCEED call
    lib.CeedBasisCreateTensorH1(self._ceed._pointer[0], dim, ncomp, P1d, Q1d,
                                interp1d_pointer, grad1d_pointer, qref1d_pointer,
                                qweight1d_pointer, self._pointer)

# ------------------------------------------------------------------------------
class BasisTensorH1Lagrange(Basis):
  """Ceed Tensor H1 Lagrange Basis: finite element tensor-product Lagrange basis
       objects for H^1 discretizations."""

  # Constructor
  def __init__(self, ceed, dim, ncomp, P, Q, qmode):

    # Setup arguments
    self._pointer = ffi.new("CeedBasis *")

    self._ceed = ceed

    # libCEED call
    lib.CeedBasisCreateTensorH1Lagrange(self._ceed._pointer[0], dim, ncomp, P,
                                        Q, qmode, self._pointer)

# ------------------------------------------------------------------------------
class BasisH1(Basis):
  """Ceed H1 Basis: finite element non tensor-product basis for H^1 discretizations."""

  # Constructor
  def __init__(self, ceed, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight):

    # Setup arguments
    self._pointer = ffi.new("CeedBasis *")

    self._ceed = ceed

    interp_pointer = ffi.new("CeedScalar *")
    interp_pointer = ffi.cast("CeedScalar *", interp.__array_interface__['data'][0])

    grad_pointer = ffi.new("CeedScalar *")
    grad_pointer = ffi.cast("CeedScalar *", grad.__array_interface__['data'][0])

    qref_pointer = ffi.new("CeedScalar *")
    qref_pointer = ffi.cast("CeedScalar *", qref.__array_interface__['data'][0])

    qweight_pointer = ffi.new("CeedScalar *")
    qweight_pointer = ffi.cast("CeedScalar *", qweight.__array_interface__['data'][0])

    # libCEED call
    lib.CeedBasisCreateH1(self._ceed._pointer[0], topo, ncomp, nnodes, nqpts,
                          interp_pointer, grad_pointer, qref_pointer,
                          qweight_pointer, self._pointer)

# ------------------------------------------------------------------------------
class TransposeBasis():
  """Transpose Ceed Basis: transpose of finite element tensor-product basis objects."""

  # Attributes
  _basis = None

  # Constructor
  def __init__(self, basis):

    # Reference basis
    self._basis = basis

  # Representation
  def __repr__(self):
    return "<Transpose CeedBasis instance at " + hex(id(self)) + ">"

  # Apply Transpose Basis
  def apply(self, nelem, emode, u, v):
    """Apply basis evaluation from quadrature points to nodes.

       Args:
         nelem: the number of elements to apply the basis evaluation to;
                  the backend will specify the ordering in a
                  Blocked ElemRestriction
         **emode: basis evaluation mode
         u: input vector
         v: output vector"""

    # libCEED call
    self._basis.apply(nelem, emode, u, v, tmode=TRANSPOSE)

# ------------------------------------------------------------------------------
