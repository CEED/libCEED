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
import ctypes
import tempfile
from abc import ABC

# ------------------------------------------------------------------------------
class _QFunctionBase(ABC):
  """Ceed QFunction: point-wise operation at quadrature points for evaluating
       volumetric terms."""

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedQFunctionDestroy(self._pointer)

  # Representation
  def __repr__(self):
    return "<CeedQFunction instance at " + hex(id(self)) + ">"

  # String conversion for print() to stdout
  def __str__(self):
    """View a QFunction via print()."""

    # libCEED call
    with tempfile.NamedTemporaryFile() as key_file:
      with open(key_file.name, 'r+') as stream_file:
        stream = ffi.cast("FILE *", stream_file)

        lib.CeedQFunctionView(self._pointer[0], stream)

        stream_file.seek(0)
        out_string = stream_file.read()

    return out_string

  # Apply CeedQFunction
  def apply(self, q, inputs, outputs):
    """Apply the action of a QFunction.

       Args:
         q: number of quadrature points
         *inputs: array of input data vectors
         *outputs: array of output data vectors"""

    # Array of vectors
    invecs = ffi.new("CeedVector[16]")
    for i in range(min(16, len(inputs))):
      invecs[i] = inputs[i]._pointer[0]
    outvecs = ffi.new("CeedVector[16]")
    for i in range(min(16, len(outputs))):
      outvecs[i] = outputs[i]._pointer[0]

    # libCEED call
    lib.CeedQFunctionApply(self._pointer[0], q, invecs, outvecs)

    # Clean-up
    ffi.release(invecs)
    ffi.release(outvecs)


# ------------------------------------------------------------------------------
class QFunction(_QFunctionBase):
  """Ceed QFunction: point-wise operation at quadrature points for evaluating
       volumetric terms."""

  # Constructor
  def __init__(self, ceed, vlength, f, source):
    # libCEED object
    self._pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # Function pointer
    fpointer = ffi.cast("CeedQFunctionUser", ctypes.cast(f, ctypes.c_void_p).value)

    # libCEED call
    sourceAscii = ffi.new("char[]", source.encode('ascii'))
    lib.CeedQFunctionCreateInterior(self._ceed._pointer[0], vlength, fpointer,
                                    sourceAscii, self._pointer)

  # Set context data
  def set_context(self, ctx):
    """Set global context for a QFunction.

       Args:
         *ctx: Numpy array holding context data to set"""

    # Setup the numpy array for the libCEED call
    ctx_pointer = ffi.new("CeedScalar *")
    ctx_pointer = ffi.cast("void *", ctx.__array_interface__['data'][0])

    # libCEED call
    lib.CeedQFunctionSetContext(self._pointer[0], ctx_pointer, len(ctx))

  # Add fields to CeedQFunction
  def add_input(self, fieldname, size, emode):
    """Add a QFunction input.

       Args:
         fieldname: name of QFunction field
         size: size of QFunction field, (ncomp * dim) for CEED_EVAL_GRAD or
                 (ncomp * 1) for CEED_EVAL_NONE and CEED_EVAL_INTERP
         **emode: CEED_EVAL_NONE to use values directly,
                  CEED_EVAL_INTERP to use interpolated values,
                  CEED_EVAL_GRAD to use gradients."""

    # libCEED call
    fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
    lib.CeedQFunctionAddInput(self._pointer[0], fieldnameAscii, size, emode)

  def add_output(self, fieldname, size, emode):
    """Add a QFunction output.

       Args:
         fieldname: name of QFunction field
         size: size of QFunction field, (ncomp * dim) for CEED_EVAL_GRAD or
                 (ncomp * 1) for CEED_EVAL_NONE and CEED_EVAL_INTERP
         **emode: CEED_EVAL_NONE to use values directly,
                  CEED_EVAL_INTERP to use interpolated values,
                  CEED_EVAL_GRAD to use gradients."""

    # libCEED call
    fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
    lib.CeedQFunctionAddOutput(self._pointer[0], fieldnameAscii, size, emode)

# ------------------------------------------------------------------------------
class QFunctionByName(_QFunctionBase):
  """Ceed QFunction By Name: point-wise operation at quadrature points
       from a given gallery, for evaluating volumetric terms."""

  # Constructor
  def __init__(self, ceed, name):
    # libCEED object
    self._pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    nameAscii = ffi.new("char[]", name.encode('ascii'))
    lib.CeedQFunctionCreateInteriorByName(self._ceed._pointer[0], nameAscii,
                                          self._pointer)

# ------------------------------------------------------------------------------
class IdentityQFunction(_QFunctionBase):
  """Ceed Identity QFunction: identity qfunction operation."""

  # Constructor
  def __init__(self, ceed, size, inmode, outmode):
    # libCEED object
    self._pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    lib.CeedQFunctionCreateIdentity(self._ceed._pointer[0], size, inmode,
                                    outmode, self._pointer)

# ------------------------------------------------------------------------------
