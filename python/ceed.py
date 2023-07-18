# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

from _ceed_cffi import ffi, lib
import sys
import os
import io
import numpy as np
import tempfile
from abc import ABC
from .ceed_vector import Vector
from .ceed_basis import BasisTensorH1, BasisTensorH1Lagrange, BasisH1, BasisHdiv, BasisHcurl
from .ceed_elemrestriction import ElemRestriction, OrientedElemRestriction, CurlOrientedElemRestriction, StridedElemRestriction
from .ceed_elemrestriction import BlockedElemRestriction, BlockedOrientedElemRestriction, BlockedCurlOrientedElemRestriction, BlockedStridedElemRestriction
from .ceed_qfunction import QFunction, QFunctionByName, IdentityQFunction
from .ceed_qfunctioncontext import QFunctionContext
from .ceed_operator import Operator, CompositeOperator
from .ceed_constants import *

# ------------------------------------------------------------------------------


class Ceed():
    """Ceed: core components."""

    # Constructor
    def __init__(self, resource="/cpu/self", on_error="store"):
        # libCEED object
        self._pointer = ffi.new("Ceed *")

        # libCEED call
        resourceAscii = ffi.new("char[]", resource.encode("ascii"))
        os.environ["CEED_ERROR_HANDLER"] = "return"
        err_code = lib.CeedInit(resourceAscii, self._pointer)
        if err_code:
            raise Exception("Error initializing backend resource: " + resource)
        error_handlers = dict(
            store="CeedErrorStore",
            abort="CeedErrorAbort",
        )
        lib.CeedSetErrorHandler(
            self._pointer[0], ffi.addressof(
                lib, error_handlers[on_error]))

    # Representation
    def __repr__(self):
        return "<Ceed instance at " + hex(id(self)) + ">"

    # String conversion for print() to stdout
    def __str__(self):
        """View a Ceed via print()."""

        # libCEED call
        with tempfile.NamedTemporaryFile() as key_file:
            with open(key_file.name, 'r+') as stream_file:
                stream = ffi.cast("FILE *", stream_file)

                err_code = lib.CeedView(self._pointer[0], stream)
                self._check_error(err_code)

                stream_file.seek(0)
                out_string = stream_file.read()

        return out_string

    # Error handler
    def _check_error(self, err_code):
        """Check return code and retrieve error message for non-zero code"""
        if (err_code != lib.CEED_ERROR_SUCCESS):
            message = ffi.new("char **")
            lib.CeedGetErrorMessage(self._pointer[0], message)
            raise Exception(ffi.string(message[0]).decode("UTF-8"))

    # Get Resource
    def get_resource(self):
        """Get the full resource name for a Ceed context.

           Returns:
             resource: resource name"""

        # libCEED call
        resource = ffi.new("char **")
        err_code = lib.CeedGetResource(self._pointer[0], resource)
        self._check_error(err_code)

        return ffi.string(resource[0]).decode("UTF-8")

    # Preferred MemType
    def get_preferred_memtype(self):
        """Return Ceed preferred memory type.

           Returns:
             memtype: Ceed preferred memory type"""

        # libCEED call
        memtype = ffi.new("CeedMemType *", MEM_HOST)
        err_code = lib.CeedGetPreferredMemType(self._pointer[0], memtype)
        self._check_error(err_code)

        return memtype[0]

    # Convenience function to get CeedScalar type
    def scalar_type(self):
        """Return dtype string for CeedScalar.

           Returns:
             np_dtype: String naming numpy data type corresponding to CeedScalar"""

        return scalar_types[lib.CEED_SCALAR_TYPE]

    # --- Basis utility functions ---

    # Gauss quadrature
    def gauss_quadrature(self, q):
        """Construct a Gauss-Legendre quadrature.

           Args:
             Q: number of quadrature points (integrates polynomials of
                  degree 2*Q-1 exactly)

           Returns:
             (qref1d, qweight1d): array of length Q to hold the abscissa on [-1, 1]
                                    and array of length Q to hold the weights"""

        # Setup arguments
        qref1d = np.empty(q, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
        qweight1d = np.empty(q, dtype=scalar_types[lib.CEED_SCALAR_TYPE])

        qref1d_pointer = ffi.new("CeedScalar *")
        qref1d_pointer = ffi.cast(
            "CeedScalar *",
            qref1d.__array_interface__['data'][0])

        qweight1d_pointer = ffi.new("CeedScalar *")
        qweight1d_pointer = ffi.cast(
            "CeedScalar *",
            qweight1d.__array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedGaussQuadrature(q, qref1d_pointer, qweight1d_pointer)
        self._check_error(err_code)

        return qref1d, qweight1d

    # Lobatto quadrature
    def lobatto_quadrature(self, q):
        """Construct a Gauss-Legendre-Lobatto quadrature.

           Args:
             q: number of quadrature points (integrates polynomials of
                  degree 2*Q-3 exactly)

           Returns:
             (qref1d, qweight1d): array of length Q to hold the abscissa on [-1, 1]
                                    and array of length Q to hold the weights"""

        # Setup arguments
        qref1d = np.empty(q, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
        qref1d_pointer = ffi.new("CeedScalar *")
        qref1d_pointer = ffi.cast(
            "CeedScalar *",
            qref1d.__array_interface__['data'][0])

        qweight1d = np.empty(q, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
        qweight1d_pointer = ffi.new("CeedScalar *")
        qweight1d_pointer = ffi.cast(
            "CeedScalar *",
            qweight1d.__array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedLobattoQuadrature(
            q, qref1d_pointer, qweight1d_pointer)
        self._check_error(err_code)

        return qref1d, qweight1d

    # --- libCEED objects ---

    # CeedVector
    def Vector(self, size):
        """Ceed Vector: storing and manipulating vectors.

           Args:
             size: length of vector

           Returns:
             vector: Ceed Vector"""

        return Vector(self, size)

    # CeedElemRestriction
    def ElemRestriction(self, nelem, elemsize, ncomp, compstride, lsize, offsets,
                        memtype=lib.CEED_MEM_HOST, cmode=lib.CEED_COPY_VALUES):
        """Ceed ElemRestriction: restriction from local vectors to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             compstride: Stride between components for the same L-vector "node".
                           Data for node i, component k can be found in the
                           L-vector at index [offsets[i] + k*compstride].
             lsize: The size of the L-vector. This vector may be larger than
                       the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1].
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed ElemRestiction"""

        return ElemRestriction(self, nelem, elemsize, ncomp, compstride, lsize,
                               offsets, memtype=memtype, cmode=cmode)

    def OrientedElemRestriction(self, nelem, elemsize, ncomp, compstride, lsize,
                                offsets, orients, memtype=lib.CEED_MEM_HOST, cmode=lib.CEED_COPY_VALUES):
        """Ceed Oriented ElemRestriction: oriented restriction from local vectors
             to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             compstride: Stride between components for the same L-vector "node".
                           Data for node i, component k can be found in the
                           L-vector at index [offsets[i] + k*compstride].
             lsize: The size of the L-vector. This vector may be larger than
                       the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1].
             *orients: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the orientations for the unknowns
                         corresponding to element i, with bool false used for
                         positively oriented and true to flip the orientation.
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed Oriented ElemRestiction"""

        return OrientedElemRestriction(self, nelem, elemsize, ncomp, compstride, lsize,
                                       offsets, orients, memtype=memtype, cmode=cmode)

    def CurlOrientedElemRestriction(self, nelem, elemsize, ncomp, compstride, lsize,
                                    offsets, curl_orients, memtype=lib.CEED_MEM_HOST, cmode=lib.CEED_COPY_VALUES):
        """Ceed Curl Oriented ElemRestriction: curl-oriented restriction from local
             vectors to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             compstride: Stride between components for the same L-vector "node".
                           Data for node i, component k can be found in the
                           L-vector at index [offsets[i] + k*compstride].
             lsize: The size of the L-vector. This vector may be larger than
                       the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1].
             *curl_orients: Numpy array of shape [nelem, 3 * elemsize]. Row i holds
                              a row-major tridiagonal matrix (curl_orients[i, 0] =
                              curl_orients[i, 3 * elemsize - 1] = 0, where 0 <= i < nelem)
                              which is applied to the element unknowns upon restriction.
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed Curl Oriented ElemRestiction"""

        return CurlOrientedElemRestriction(
            self, nelem, elemsize, ncomp, compstride, lsize, offsets, curl_orients, memtype=memtype, cmode=cmode)

    def StridedElemRestriction(self, nelem, elemsize, ncomp, lsize, strides):
        """Ceed Identity ElemRestriction: strided restriction from local vectors
             to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             lsize: The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
             *strides: Array for strides between [nodes, components, elements].
                         The data for node i, component j, element k in the
                         L-vector is given by
                           i*strides[0] + j*strides[1] + k*strides[2]

           Returns:
             elemrestriction: Ceed Strided ElemRestiction"""

        return StridedElemRestriction(
            self, nelem, elemsize, ncomp, lsize, strides)

    def BlockedElemRestriction(self, nelem, elemsize, blksize, ncomp, compstride,
                               lsize, offsets, memtype=lib.CEED_MEM_HOST,
                               cmode=lib.CEED_COPY_VALUES):
        """Ceed Blocked ElemRestriction: blocked restriction from local vectors
             to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             blksize: number of elements in a block
             ncomp: number of field components per interpolation node
                       (1 for scalar fields)
             lsize: The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1]. The backend will permute and pad this
                         array to the desired ordering for the blocksize, which is
                         typically given by the backend. The default reordering is
                         to interlace elements.
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed Blocked ElemRestiction"""

        return BlockedElemRestriction(self, nelem, elemsize, blksize, ncomp,
                                      compstride, lsize, offsets,
                                      memtype=memtype, cmode=cmode)

    def BlockedOrientedElemRestriction(self, nelem, elemsize, blksize, ncomp, compstride,
                                       lsize, offsets, orients, memtype=lib.CEED_MEM_HOST, cmode=lib.CEED_COPY_VALUES):
        """Ceed Blocked Oriented ElemRestriction: blocked oriented restriction
             from local vectors to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             blksize: number of elements in a block
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             compstride: Stride between components for the same L-vector "node".
                           Data for node i, component k can be found in the
                           L-vector at index [offsets[i] + k*compstride].
             lsize: The size of the L-vector. This vector may be larger than
                       the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1]. The backend will permute and pad this
                         array to the desired ordering for the blocksize, which is
                         typically given by the backend. The default reordering is
                         to interlace elements.
             *orients: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the orientations for the unknowns
                         corresponding to element i, with ool false is used for
                         positively oriented and true to flip the orientation.
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed Blocked Oriented ElemRestiction"""

        return BlockedOrientedElemRestriction(self, nelem, elemsize, blksize, ncomp, compstride, lsize,
                                              offsets, orients, memtype=memtype, cmode=cmode)

    def BlockedCurlOrientedElemRestriction(self, nelem, elemsize, blksize, ncomp, compstride,
                                           lsize, offsets, curl_orients, memtype=lib.CEED_MEM_HOST, cmode=lib.CEED_COPY_VALUES):
        """Ceed Blocked Curl Oriented ElemRestriction: blocked curl-oriented
             restriction from local vectors to elements.

           Args:
             nelem: number of elements described by the restriction
             elemsize: size (number of nodes) per element
             blksize: number of elements in a block
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             compstride: Stride between components for the same L-vector "node".
                           Data for node i, component k can be found in the
                           L-vector at index [offsets[i] + k*compstride].
             lsize: The size of the L-vector. This vector may be larger than
                       the elements and fields given by this restriction.
             *offsets: Numpy array of shape [nelem, elemsize]. Row i holds the
                         ordered list of the offsets (into the input Ceed Vector)
                         for the unknowns corresponding to element i, where
                         0 <= i < nelem. All offsets must be in the range
                         [0, lsize - 1]. The backend will permute and pad this
                         array to the desired ordering for the blocksize, which is
                         typically given by the backend. The default reordering is
                         to interlace elements.
             *curl_orients: Numpy array of shape [nelem, 3 * elemsize]. Row i holds
                              a row-major tridiagonal matrix (curl_orients[i, 0] =
                              curl_orients[i, 3 * elemsize - 1] = 0, where 0 <= i < nelem)
                              which is applied to the element unknowns upon restriction.
             **memtype: memory type of the offsets array, default CEED_MEM_HOST
             **cmode: copy mode for the offsets array, default CEED_COPY_VALUES

           Returns:
             elemrestriction: Ceed Blocked Curl Oriented ElemRestiction"""

        return BlockedCurlOrientedElemRestriction(
            self, nelem, elemsize, blksize, ncomp, compstride, lsize, offsets, curl_orients, memtype=memtype, cmode=cmode)

    def BlockedStridedElemRestriction(self, nelem, elemsize, blksize, ncomp,
                                      lsize, strides):
        """Ceed Blocked Strided ElemRestriction: blocked and strided restriction
             from local vectors to elements.

           Args:
             nelem: number of elements described in the restriction
             elemsize: size (number of nodes) per element
             blksize: number of elements in a block
             ncomp: number of field components per interpolation node
                      (1 for scalar fields)
             lsize: The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
             *strides: Array for strides between [nodes, components, elements].
                         The data for node i, component j, element k in the
                         L-vector is given by
                           i*strides[0] + j*strides[1] + k*strides[2]

           Returns:
             elemrestriction: Ceed Blocked Strided ElemRestiction"""

        return BlockedStridedElemRestriction(self, nelem, elemsize, blksize,
                                             ncomp, lsize, strides)

    # CeedBasis
    def BasisTensorH1(self, dim, ncomp, P1d, Q1d, interp1d, grad1d,
                      qref1d, qweight1d):
        """Ceed Tensor H1 Basis: finite element tensor-product basis objects for
             H^1 discretizations.

           Args:
             dim: topological dimension
             ncomp: number of field components (1 for scalar fields)
             P1d: number of nodes in one dimension
             Q1d: number of quadrature points in one dimension
             *interp1d: Numpy array holding the row-major (Q1d * P1d) matrix
                          expressing the values of nodal basis functions at
                          quadrature points
             *grad1d: Numpy array holding the row-major (Q1d * P1d) matrix
                        expressing the derivatives of nodal basis functions at
                        quadrature points
             *qref1d: Numpy array of length Q1d holding the locations of
                        quadrature points on the 1D reference element [-1, 1]
             *qweight1d: Numpy array of length Q1d holding the quadrature
                           weights on the reference element

           Returns:
             basis: Ceed Basis"""

        return BasisTensorH1(self, dim, ncomp, P1d, Q1d, interp1d, grad1d,
                             qref1d, qweight1d)

    def BasisTensorH1Lagrange(self, dim, ncomp, P, Q, qmode):
        """Ceed Tensor H1 Lagrange Basis: finite element tensor-product Lagrange
             basis objects for H^1 discretizations.

           Args:
             dim: topological dimension
             ncomp: number of field components (1 for scalar fields)
             P: number of Gauss-Lobatto nodes in one dimension.  The
                  polynomial degree of the resulting Q_k element is k=P-1.
             Q: number of quadrature points in one dimension
             qmode: distribution of the Q quadrature points (affects order of
                      accuracy for the quadrature)

           Returns:
             basis: Ceed Basis"""

        return BasisTensorH1Lagrange(self, dim, ncomp, P, Q, qmode)

    def BasisH1(self, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight):
        """Ceed H1 Basis: finite element non tensor-product basis for H^1
             discretizations.

           Args:
             topo: topology of the element, e.g. hypercube, simplex, etc
             ncomp: number of field components (1 for scalar fields)
             nnodes: total number of nodes
             nqpts: total number of quadrature points
             *interp: Numpy array holding the row-major (nqpts * nnodes) matrix
                       expressing the values of nodal basis functions at
                       quadrature points
             *grad: Numpy array holding the row-major (dim * nqpts * nnodes)
                     matrix expressing the derivatives of nodal basis functions
                     at quadrature points
             *qref: Numpy array of length (nqpts * dim) holding the locations of
                     quadrature points on the reference element [-1, 1]
             *qweight: Numpy array of length nnodes holding the quadrature
                        weights on the reference element

           Returns:
             basis: Ceed Basis"""

        return BasisH1(self, topo, ncomp, nnodes, nqpts,
                       interp, grad, qref, qweight)

    def BasisHdiv(self, topo, ncomp, nnodes, nqpts, interp, div, qref, qweight):
        """Ceed Hdiv Basis: finite element non tensor-product basis for H(div)
             discretizations.

           Args:
             topo: topology of the element, e.g. hypercube, simplex, etc
             ncomp: number of field components (1 for scalar fields)
             nnodes: total number of nodes
             nqpts: total number of quadrature points
             *interp: Numpy array holding the row-major (dim * nqpts * nnodes)
                       matrix expressing the values of basis functions at
                       quadrature points
             *div: Numpy array holding the row-major (nqpts * nnodes) matrix
                    expressing the divergence of basis functions at
                    quadrature points
             *qref: Numpy array of length (nqpts * dim) holding the locations of
                     quadrature points on the reference element [-1, 1]
             *qweight: Numpy array of length nnodes holding the quadrature
                        weights on the reference element

           Returns:
             basis: Ceed Basis"""

        return BasisHdiv(self, topo, ncomp, nnodes, nqpts,
                         interp, div, qref, qweight)

    def BasisHcurl(self, topo, ncomp, nnodes, nqpts,
                   interp, curl, qref, qweight):
        """Ceed Hcurl Basis: finite element non tensor-product basis for H(curl)
             discretizations.

           Args:
             topo: topology of the element, e.g. hypercube, simplex, etc
             ncomp: number of field components (1 for scalar fields)
             nnodes: total number of nodes
             nqpts: total number of quadrature points
             *interp: Numpy array holding the row-major (dim * nqpts * nnodes)
                       matrix expressing the values of basis functions at
                       quadrature points
             *curl: Numpy array holding the row-major (curlcomp * nqpts * nnodes),
                     curlcomp = 1 if dim < 3 else dim, matrix expressing the curl
                     of basis functions at quadrature points
             *qref: Numpy array of length (nqpts * dim) holding the locations of
                     quadrature points on the reference element [-1, 1]
             *qweight: Numpy array of length nnodes holding the quadrature
                        weights on the reference element

           Returns:
             basis: Ceed Basis"""

        return BasisHcurl(self, topo, ncomp, nnodes, nqpts,
                          interp, curl, qref, qweight)

    # CeedQFunction
    def QFunction(self, vlength, f, source):
        """Ceed QFunction: point-wise operation at quadrature points for
             evaluating volumetric terms.

           Args:
             vlength: vector length. Caller must ensure that number of quadrature
                        points is a multiple of vlength
             f: ctypes function pointer to evaluate action at quadrature points
             source: absolute path to source of QFunction,
               "\\abs_path\\file.h:function_name

           Returns:
             qfunction: Ceed QFunction"""

        return QFunction(self, vlength, f, source)

    def QFunctionByName(self, name):
        """Ceed QFunction By Name: point-wise operation at quadrature points
             from a given gallery, for evaluating volumetric terms.

           Args:
             name: name of QFunction to use from gallery

           Returns:
             qfunction: Ceed QFunction By Name"""

        return QFunctionByName(self, name)

    def IdentityQFunction(self, size, inmode, outmode):
        """Ceed Idenity QFunction: identity qfunction operation.

           Args:
             size: size of the qfunction fields
             **inmode: CeedEvalMode for input to Ceed QFunction
             **outmode: CeedEvalMode for output to Ceed QFunction

           Returns:
             qfunction: Ceed Identity QFunction"""

        return IdentityQFunction(self, size, inmode, outmode)

    def QFunctionContext(self):
        """Ceed QFunction Context: stores Ceed QFunction user context data.

           Returns:
             userContext: Ceed QFunction Context"""

        return QFunctionContext(self)

    # CeedOperator
    def Operator(self, qf, dqf=None, qdfT=None):
        """Ceed Operator: composed FE-type operations on vectors.

           Args:
             qf: QFunction defining the action of the operator at quadrature
                   points
             **dqf: QFunction defining the action of the Jacobian, default None
             **dqfT: QFunction defining the action of the transpose of the
                       Jacobian, default None

           Returns:
             operator: Ceed Operator"""

        return Operator(self, qf, dqf, qdfT)

    def CompositeOperator(self):
        """Composite Ceed Operator: composition of multiple CeedOperators.

           Returns:
             operator: Ceed Composite Operator"""

        return CompositeOperator(self)

    # Destructor
    def __del__(self):
        # libCEED call
        lib.CeedDestroy(self._pointer)

# ------------------------------------------------------------------------------
