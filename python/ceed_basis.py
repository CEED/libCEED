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

                err_code = lib.CeedBasisView(self._pointer[0], stream)
                self._ceed._check_error(err_code)

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
        err_code = lib.CeedBasisApply(self._pointer[0], nelem, tmode, emode,
                                      u._pointer[0], v._pointer[0])
        self._ceed._check_error(err_code)

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
        err_code = lib.CeedBasisGetNumNodes(self._pointer[0], p_pointer)
        self._ceed._check_error(err_code)

        return p_pointer[0]

    # Get number of quadrature points
    def get_num_quadrature_points(self):
        """Get total number of quadrature points (in dim dimensions) of a Basis.

           Returns:
             num_qpts: total number of quadrature points"""

        # Setup argument
        q_pointer = ffi.new("CeedInt *")

        # libCEED call
        err_code = lib.CeedBasisGetNumQuadraturePoints(
            self._pointer[0], q_pointer)
        self._ceed._check_error(err_code)

        return q_pointer[0]

    # Destructor
    def __del__(self):
        # libCEED call
        err_code = lib.CeedBasisDestroy(self._pointer)
        self._ceed._check_error(err_code)

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
        interp1d_pointer = ffi.cast(
            "CeedScalar *",
            interp1d.__array_interface__['data'][0])

        grad1d_pointer = ffi.new("CeedScalar *")
        grad1d_pointer = ffi.cast(
            "CeedScalar *",
            grad1d.__array_interface__['data'][0])

        qref1d_pointer = ffi.new("CeedScalar *")
        qref1d_pointer = ffi.cast(
            "CeedScalar *",
            qref1d.__array_interface__['data'][0])

        qweight1d_pointer = ffi.new("CeedScalar *")
        qweight1d_pointer = ffi.cast(
            "CeedScalar *",
            qweight1d.__array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedBasisCreateTensorH1(self._ceed._pointer[0], dim, ncomp,
                                               P1d, Q1d, interp1d_pointer,
                                               grad1d_pointer, qref1d_pointer,
                                               qweight1d_pointer, self._pointer)
        self._ceed._check_error(err_code)

    # Get 1D interpolation matrix
    def get_interp_1d(self):
        """Return 1D interpolation matrix of a tensor product Basis.

           Returns:
             *array: Numpy array"""

        # Compute the length of the array
        nnodes_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumNodes1D(self._pointer[0], nnodes_pointer)
        nqpts_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumQuadraturePoints1D(self._pointer[0], nqpts_pointer)
        length = nnodes_pointer[0] * nqpts_pointer[0]

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        lib.CeedBasisGetInterp1D(self._pointer[0], array_pointer)

        # Return array created from buffer
        # Create buffer object from returned pointer
        buff = ffi.buffer(
            array_pointer[0],
            ffi.sizeof("CeedScalar") *
            length)
        # return read only Numpy array
        ret = np.frombuffer(buff, dtype="float64")
        ret.flags['WRITEABLE'] = False
        return ret

    # Get 1D gradient matrix
    def get_grad_1d(self):
        """Return 1D gradient matrix of a tensor product Basis.

           Returns:
             *array: Numpy array"""

        # Compute the length of the array
        nnodes_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumNodes1D(self._pointer[0], nnodes_pointer)
        nqpts_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumQuadraturePoints1D(self._pointer[0], nqpts_pointer)
        length = nnodes_pointer[0] * nqpts_pointer[0]

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        lib.CeedBasisGetGrad1D(self._pointer[0], array_pointer)

        # Return array created from buffer
        # Create buffer object from returned pointer
        buff = ffi.buffer(
            array_pointer[0],
            ffi.sizeof("CeedScalar") *
            length)
        # return read only Numpy array
        ret = np.frombuffer(buff, dtype="float64")
        ret.flags['WRITEABLE'] = False
        return ret

    # Get 1D quadrature weights matrix
    def get_q_weight_1d(self):
        """Return 1D quadrature weights matrix of a tensor product Basis.

           Returns:
             *array: Numpy array"""

        # Compute the length of the array
        nqpts_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumQuadraturePoints1D(self._pointer[0], nqpts_pointer)
        length = nqpts_pointer[0]

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        lib.CeedBasisGetQWeights(self._pointer[0], array_pointer)

        # Return array created from buffer
        # Create buffer object from returned pointer
        buff = ffi.buffer(
            array_pointer[0],
            ffi.sizeof("CeedScalar") *
            length)
        # return read only Numpy array
        ret = np.frombuffer(buff, dtype="float64")
        ret.flags['WRITEABLE'] = False
        return ret

    # Get 1D quadrature points matrix
    def get_q_ref_1d(self):
        """Return 1D quadrature points matrix of a tensor product Basis.

           Returns:
             *array: Numpy array"""

        # Compute the length of the array
        nqpts_pointer = ffi.new("CeedInt *")
        lib.CeedBasisGetNumQuadraturePoints1D(self._pointer[0], nqpts_pointer)
        length = nqpts_pointer[0]

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        lib.CeedBasisGetQRef(self._pointer[0], array_pointer)

        # Return array created from buffer
        # Create buffer object from returned pointer
        buff = ffi.buffer(
            array_pointer[0],
            ffi.sizeof("CeedScalar") *
            length)
        # return read only Numpy array
        ret = np.frombuffer(buff, dtype="float64")
        ret.flags['WRITEABLE'] = False
        return ret


# ------------------------------------------------------------------------------


class BasisTensorH1Lagrange(BasisTensorH1):
    """Ceed Tensor H1 Lagrange Basis: finite element tensor-product Lagrange basis
         objects for H^1 discretizations."""

    # Constructor
    def __init__(self, ceed, dim, ncomp, P, Q, qmode):

        # Setup arguments
        self._pointer = ffi.new("CeedBasis *")

        self._ceed = ceed

        # libCEED call
        err_code = lib.CeedBasisCreateTensorH1Lagrange(self._ceed._pointer[0], dim,
                                                       ncomp, P, Q, qmode, self._pointer)
        self._ceed._check_error(err_code)

# ------------------------------------------------------------------------------


class BasisH1(Basis):
    """Ceed H1 Basis: finite element non tensor-product basis for H^1 discretizations."""

    # Constructor
    def __init__(self, ceed, topo, ncomp, nnodes,
                 nqpts, interp, grad, qref, qweight):

        # Setup arguments
        self._pointer = ffi.new("CeedBasis *")

        self._ceed = ceed

        interp_pointer = ffi.new("CeedScalar *")
        interp_pointer = ffi.cast(
            "CeedScalar *",
            interp.__array_interface__['data'][0])

        grad_pointer = ffi.new("CeedScalar *")
        grad_pointer = ffi.cast(
            "CeedScalar *",
            grad.__array_interface__['data'][0])

        qref_pointer = ffi.new("CeedScalar *")
        qref_pointer = ffi.cast(
            "CeedScalar *",
            qref.__array_interface__['data'][0])

        qweight_pointer = ffi.new("CeedScalar *")
        qweight_pointer = ffi.cast(
            "CeedScalar *",
            qweight.__array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedBasisCreateH1(self._ceed._pointer[0], topo, ncomp,
                                         nnodes, nqpts, interp_pointer,
                                         grad_pointer, qref_pointer,
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
