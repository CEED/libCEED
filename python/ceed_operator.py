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

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedOperatorDestroy(self._pointer)

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

        lib.CeedOperatorView(self._pointer[0], stream)

        stream_file.seek(0)
        out_string = stream_file.read()

    return out_string

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
    lib.CeedOperatorApply(self._pointer[0], u._pointer[0], v._pointer[0],
                          request)

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
    lib.CeedOperatorApplyAdd(self._pointer[0], u._pointer[0], v._pointer[0],
                             request)

# ------------------------------------------------------------------------------
class Operator(_OperatorBase):
  """Ceed Operator: composed FE-type operations on vectors."""

  # Constructor
  def __init__(self, ceed, qf, dqf = None, dqfT = None):
    # CeedOperator object
    self._pointer = ffi.new("CeedOperator *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    lib.CeedOperatorCreate(self._ceed._pointer[0], qf._pointer[0],
                           dqf._pointer[0] if dqf else ffi.NULL,
                           dqfT._pointer[0] if dqfT else ffi.NULL,
                           self._pointer)

  # Add field to CeedOperator
  def set_field(self, fieldname, restriction, basis, vector, lmode=NOTRANSPOSE):
    """Provide a field to an Operator for use by its QFunction.

       Args:
         fieldname: name of the field (to be matched with the same name used
                      by QFunction)
         restriction: ElemRestriction
         basis: Basis in which the field resides or CEED_BASIS_COLLOCATED
                  if collocated with quadrature points
         vector: Vector to be used by Operator or CEED_VECTOR_ACTIVE
                   if field is active or CEED_VECTOR_NONE if using
                   CEED_EVAL_WEIGHT in the QFunction
         **lmode: CeedTransposeMode which specifies the ordering of the
                    components of the l-vector used by this CeedOperatorField,
                    CEED_NOTRANSPOSE indicates the component is the
                    outermost index and CEED_TRANSPOSE indicates the component
                    is the innermost index in ordering of the local vector,
                    default CEED_NOTRANSPOSE"""

    # libCEED call
    fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
    lib.CeedOperatorSetField(self._pointer[0], fieldnameAscii,
                             restriction._pointer[0], lmode, basis._pointer[0],
                             vector._pointer[0])

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
    lib.CeedCompositeOperatorCreate(self._ceed._pointer[0], self._pointer)

  # Add sub operators
  def add_sub(self, subop):
    """Add a sub-operator to a composite CeedOperator.

       Args:
         subop: sub-operator Operator"""

    # libCEED call
    lib.CeedCompositeOperatorAddSub(self._pointer[0], subop._pointer[0])

# ------------------------------------------------------------------------------
