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
from abc import ABC
from .ceed_constants import REQUEST_IMMEDIATE, REQUEST_ORDERED, NOTRANSPOSE

# ------------------------------------------------------------------------------


class _OperatorBase(ABC):
    """Ceed Operator: composed FE-type operations on vectors."""

    # Destructor
    def __del__(self):
        # libCEED call
        err_code = lib.CeedOperatorDestroy(self._pointer)
        self._ceed._check_error(err_code)

    # Representation
    def __repr__(self):
        return "<CeedOperator instance at " + hex(id(self)) + ">"

    # String conversion for print() to stdout
    def __str__(self):
        """View an Operator via print()."""

        # libCEED call
        with tempfile.NamedTemporaryFile() as key_file:
            with open(key_file.name, 'r+') as stream_file:
                stream = ffi.cast("FILE *", stream_file)

                err_code = lib.CeedOperatorView(self._pointer[0], stream)
                self._ceed._check_error(err_code)

                stream_file.seek(0)
                out_string = stream_file.read()

        return out_string

    # Assemble linear diagonal
    def linear_assemble_diagonal(self, d, request=REQUEST_IMMEDIATE):
        """Assemble the diagonal of a square linear Operator

           Args:
             d: Vector to store assembled Operator diagonal
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorLinearAssembleDiagonal(self._pointer[0],
                                                          d._pointer[0], request)
        self._ceed._check_error(err_code)

    # Assemble add linear diagonal
    def linear_assemble_add_diagonal(self, d, request=REQUEST_IMMEDIATE):
        """Sum the diagonal of a square linear Operator into a Vector

           Args:
             d: Vector to store assembled Operator diagonal
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorLinearAssembleAddDiagonal(self._pointer[0],
                                                             d._pointer[0], request)
        self._ceed._check_error(err_code)

    # Assemble linear point block diagonal
    def linear_assemble_point_block_diagonal(
            self, d, request=REQUEST_IMMEDIATE):
        """Assemble the point block diagonal of a square linear Operator

           Args:
             d: Vector to store assembled Operator point block diagonal,
                  provided in row-major form with an ncomp*ncomp block
                  at each node
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorLinearAssemblePointBlockDiagonal(self._pointer[0],
                                                                    d._pointer[0], request)
        self._ceed._check_error(err_code)

    # Assemble linear point block diagonal
    def linear_assemble_add_point_block_diagonal(
            self, d, request=REQUEST_IMMEDIATE):
        """Sum the point block diagonal of a square linear Operator into a Vector

           Args:
             d: Vector to store assembled Operator point block diagonal,
                  provided in row-major form with an ncomp*ncomp block
                  at each node
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorLinearAssembleAddPointBlockDiagonal(self._pointer[0],
                                                                       d._pointer[0], request)
        self._ceed._check_error(err_code)

    # Apply CeedOperator
    def apply(self, u, v, request=REQUEST_IMMEDIATE):
        """Apply Operator to a vector.

           Args:
             u: Vector containing input state or CEED_VECTOR_NONE if there are no
                  active inputs
             v: Vector to store result of applying operator (must be distinct from u)
                  or CEED_VECTOR_NONE if there are no active outputs
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorApply(self._pointer[0], u._pointer[0], v._pointer[0],
                                         request)
        self._ceed._check_error(err_code)

    # Apply CeedOperator
    def apply_add(self, u, v, request=REQUEST_IMMEDIATE):
        """Apply Operator to a vector and add result to output vector.

           Args:
             u: Vector containing input state or CEED_VECTOR_NONE if there are no
                  active inputs
             v: Vector to sum in result of applying operator (must be distinct from u)
                  or CEED_VECTOR_NONE if there are no active outputs
             **request: Ceed request, default CEED_REQUEST_IMMEDIATE"""

        # libCEED call
        err_code = lib.CeedOperatorApplyAdd(self._pointer[0], u._pointer[0], v._pointer[0],
                                            request)
        self._ceed._check_error(err_code)

    # Create Multigrid Level
    def multigrid_create(self, p_mult_fine, rstr_coarse, basis_coarse):
        """ Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a Lagrange tensor basis for the active basis

           Args:
             p_mult_fine: L-vector multiplicity in parallel gather/scatter
             basis_coarse: Coarse grid active vector basis
             degree_coarse: Coarse grid basis polynomial order"""

        # Operator pointers
        opCoarsePointer = ffi.new("CeedOperator *")
        opProlongPointer = ffi.new("CeedOperator *")
        opRestrictPointer = ffi.new("CeedOperator *")

        # libCEED call
        lib.CeedOperatorMultigridLevelCreate(self._pointer[0],
                                             p_mult_fine._pointer[0],
                                             rstr_coarse._pointer[0],
                                             basis_coarse._pointer[0],
                                             opCoarsePointer,
                                             opProlongPointer,
                                             opRestrictPointer)

        # Wrap operators
        opCoarse = _OperatorWrap(
            self._ceed, opCoarsePointer)
        opProlong = _OperatorWrap(
            self._ceed, opProlongPointer)
        opRestrict = _OperatorWrap(
            self._ceed, opRestrictPointer)

        # Return
        return [opCoarse, opProlong, opRestrict]

    # Create Multigrid Level
    def multigrid_create_tensor_h1(self, p_mult_fine, rstr_coarse, basis_coarse,
                                   interp_C_to_F):
        """ Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a non-tensor basis for the active basis

           Args:
             p_mult_fine: L-vector multiplicity in parallel gather/scatter
             rstr_coarse: Coarse grid restriction
             basis_coarse: Coarse grid active vector basis
             interp_C_to_F: Matrix for coarse to fine interpolation"""

       # Setup arguments
        interpCtoF_pointer = ffi.new("CeedScalar *")
        interpCtoF_pointer = ffi.cast(
            "CeedScalar *",
            interp_C_to_F.__array_interface__['data'][0])

        # Operator pointers
        opCoarsePointer = ffi.new("CeedOperator *")
        opProlongPointer = ffi.new("CeedOperator *")
        opRestrictPointer = ffi.new("CeedOperator *")

        # libCEED call
        lib.CeedOperatorMultigridLevelCreateTensorH1(self._pointer[0],
                                                     p_mult_fine._pointer[0],
                                                     rstr_coarse._pointer[0],
                                                     basis_coarse._pointer[0],
                                                     interpCtoF_pointer,
                                                     opCoarsePointer,
                                                     opProlongPointer,
                                                     opRestrictPointer)

        # Wrap operators
        opCoarse = _OperatorWrap(
            self._ceed, opCoarsePointer)
        opProlong = _OperatorWrap(
            self._ceed, opProlongPointer)
        opRestrict = _OperatorWrap(
            self._ceed, opRestrictPointer)

        # Return
        return [opCoarse, opProlong, opRestrict]

    # Create Multigrid Level
    def multigrid_create_h1(self, p_mult_fine, rstr_coarse, basis_coarse,
                            interp_C_to_F):
        """ Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a Lagrange tensor basis for the active basis

           Args:
             p_mult_fine: L-vector multiplicity in parallel gather/scatter
             rstr_coarse: Coarse grid restriction
             basis_coarse: Coarse grid active vector basis
             interp_C_to_F: Matrix for coarse to fine interpolation"""

       # Setup arguments
        interpCtoF_pointer = ffi.new("CeedScalar *")
        interpCtoF_pointer = ffi.cast(
            "CeedScalar *",
            interp_C_to_F.__array_interface__['data'][0])

        # Operator pointers
        opCoarsePointer = ffi.new("CeedOperator *")
        opProlongPointer = ffi.new("CeedOperator *")
        opRestrictPointer = ffi.new("CeedOperator *")

        # libCEED call
        lib.CeedOperatorMultigridLevelCreateH1(self._pointer[0],
                                               p_mult_fine._pointer[0],
                                               rstr_coarse._pointer[0],
                                               basis_coarse._pointer[0],
                                               interpCtoF_pointer,
                                               opCoarsePointer,
                                               opProlongPointer,
                                               opRestrictPointer)

        # Wrap operators
        opCoarse = _OperatorWrap(
            self._ceed, opCoarsePointer)
        opProlong = _OperatorWrap(
            self._ceed, opProlongPointer)
        opRestrict = _OperatorWrap(
            self._ceed, opRestrictPointer)

        # Return
        return [opCoarse, opProlong, opRestrict]


# ------------------------------------------------------------------------------


class Operator(_OperatorBase):
    """Ceed Operator: composed FE-type operations on vectors."""

    # Constructor
    def __init__(self, ceed, qf, dqf=None, dqfT=None):
        # CeedOperator object
        self._pointer = ffi.new("CeedOperator *")

        # Reference to Ceed
        self._ceed = ceed

        # libCEED call
        err_code = lib.CeedOperatorCreate(self._ceed._pointer[0], qf._pointer[0],
                                          dqf._pointer[0] if dqf else ffi.NULL,
                                          dqfT._pointer[0] if dqfT else ffi.NULL,
                                          self._pointer)
        self._ceed._check_error(err_code)

    # Add field to CeedOperator
    def set_field(self, fieldname, restriction, basis, vector):
        """Provide a field to an Operator for use by its QFunction.

           Args:
             fieldname: name of the field (to be matched with the same name used
                          by QFunction)
             restriction: ElemRestriction
             basis: Basis in which the field resides or CEED_BASIS_COLLOCATED
                      if collocated with quadrature points
             vector: Vector to be used by Operator or CEED_VECTOR_ACTIVE
                       if field is active or CEED_VECTOR_NONE if using
                       CEED_EVAL_WEIGHT in the QFunction"""

        # libCEED call
        fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
        err_code = lib.CeedOperatorSetField(self._pointer[0], fieldnameAscii,
                                            restriction._pointer[0], basis._pointer[0],
                                            vector._pointer[0])
        self._ceed._check_error(err_code)

# ------------------------------------------------------------------------------


class CompositeOperator(_OperatorBase):
    """Ceed Composite Operator: composition of multiple Operators."""

    # Constructor
    def __init__(self, ceed):
        # CeedOperator object
        self._pointer = ffi.new("CeedOperator *")

        # Reference to Ceed
        self._ceed = ceed
        # libCEED call
        err_code = lib.CeedCompositeOperatorCreate(
            self._ceed._pointer[0], self._pointer)
        self._ceed._check_error(err_code)

    # Add sub operators
    def add_sub(self, subop):
        """Add a sub-operator to a composite CeedOperator.

           Args:
             subop: sub-operator Operator"""

        # libCEED call
        err_code = lib.CeedCompositeOperatorAddSub(
            self._pointer[0], subop._pointer[0])
        self._ceed._check_error(err_code)

# ------------------------------------------------------------------------------


class _OperatorWrap(Operator):
    """Wrap a CeedOperator pointer in a Operator object."""

    # Constructor
    def __init__(self, ceed, pointer):
        # CeedOperator object
        self._pointer = pointer

        # Reference to Ceed
        self._ceed = ceed

# ------------------------------------------------------------------------------
