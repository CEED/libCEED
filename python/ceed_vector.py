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
import contextlib
from .ceed_constants import MEM_HOST, COPY_VALUES, NORM_2

# ------------------------------------------------------------------------------
class Vector():
  """Ceed Vector: storing and manipulating vectors."""

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Constructor
  def __init__(self, ceed, size):
    # CeedVector object
    self._pointer = ffi.new("CeedVector *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    lib.CeedVectorCreate(self._ceed._pointer[0], size, self._pointer)

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedVectorDestroy(self._pointer)

  # Representation
  def __repr__(self):
    return "<CeedVector instance at " + hex(id(self)) + ">"

  # String conversion for print() to stdout
  def __str__(self):
    """View a Vector via print()."""

    # libCEED call
    fmt = ffi.new("char[]", "%f".encode('ascii'))
    with tempfile.NamedTemporaryFile() as key_file:
      with open(key_file.name, 'r+') as stream_file:
        stream = ffi.cast("FILE *", stream_file)

        lib.CeedVectorView(self._pointer[0], fmt, stream)

        stream_file.seek(0)
        out_string = stream_file.read()

    return out_string

  # Set Vector's data array
  def set_array(self, array, memtype=MEM_HOST, cmode=COPY_VALUES):
    """Set the array used by a Vector, freeing any previously allocated
       array if applicable.

       Args:
         *array: Numpy array to be used
         **memtype: memory type of the array being passed, default CEED_MEM_HOST
         **cmode: copy mode for the array, default CEED_COPY_VALUES"""

    # Setup the numpy array for the libCEED call
    array_pointer = ffi.new("CeedScalar *")
    array_pointer = ffi.cast("CeedScalar *", array.__array_interface__['data'][0])

    # libCEED call
    lib.CeedVectorSetArray(self._pointer[0], memtype, cmode, array_pointer)

  # Get Vector's data array
  def get_array(self, memtype=MEM_HOST):
    """Get read/write access to a Vector via the specified memory type.

       Args:
         **memtype: memory type of the array being passed, default CEED_MEM_HOST

       Returns:
         *array: Numpy array"""

    # Retrieve the length of the array
    length_pointer = ffi.new("CeedInt *")
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorGetArray(self._pointer[0], memtype, array_pointer)

    # Create buffer object from returned pointer
    buff = ffi.buffer(array_pointer[0], ffi.sizeof("CeedScalar") * length_pointer[0])
    # Return numpy array created from buffer
    return np.frombuffer(buff, dtype="float64")

  # Get Vector's data array in read-only mode
  def get_array_read(self, memtype=MEM_HOST):
    """Get read-only access to a Vector via the specified memory type.

       Args:
         **memtype: memory type of the array being passed, default CEED_MEM_HOST

       Returns:
         *array: Numpy array"""

    # Retrieve the length of the array
    length_pointer = ffi.new("CeedInt *")
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorGetArrayRead(self._pointer[0], memtype, array_pointer)

    # Create buffer object from returned pointer
    buff = ffi.buffer(array_pointer[0], ffi.sizeof("CeedScalar") * length_pointer[0])
    # Create numpy array from buffer
    ret = np.frombuffer(buff, dtype="float64")
    # Make the numpy array read-only
    ret.flags['WRITEABLE'] = False
    return ret

  # Restore the Vector's data array
  def restore_array(self):
    """Restore an array obtained using get_array()."""

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorRestoreArray(self._pointer[0], array_pointer)

  # Restore an array obtained using getArrayRead
  def restore_array_read(self):
    """Restore an array obtained using get_array_read()."""

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorRestoreArrayRead(self._pointer[0], array_pointer)

  @contextlib.contextmanager
  def array(self, *shape):
    """Context manager for array access.

    Args:
      shape (tuple): shape of returned numpy.array

    Returns:
      np.array: writable view of vector

    Examples:
      Constructing the identity inside a libceed.Vector:

      >>> vec = ceed.Vector(16)
      >>> with vec.array(4, 4) as x:
      >>>     x[...] = np.eye(4)
    """
    x = self.get_array()
    if shape:
      x = x.reshape(shape)
    yield x
    self.restore_array()

  @contextlib.contextmanager
  def array_read(self, *shape):
    """Context manager for read-only array access.

    Args:
      shape (tuple): shape of returned numpy.array

    Returns:
      np.array: read-only view of vector

    Examples:
      Constructing the identity inside a libceed.Vector:

      >>> vec = ceed.Vector(6)
      >>> vec.set_value(1.3)
      >>> with vec.array_read(2, 3) as x:
      >>>     print(x)
    """
    x = self.get_array_read()
    if shape:
      x = x.reshape(shape)
    yield x
    self.restore_array_read()

  # Get the length of a Vector
  def get_length(self):
    """Get the length of a Vector.

       Returns:
         length: length of the Vector"""

    length_pointer = ffi.new("CeedInt *")

    # libCEED call
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    return length_pointer[0]

  # Get the length of a Vector
  def __len__(self):
    """Get the length of a Vector.

       Returns:
         length: length of the Vector"""

    length_pointer = ffi.new("CeedInt *")

    # libCEED call
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    return length_pointer[0]

  # Set the Vector to a given constant value
  def set_value(self, value):
    """Set the Vector to a constant value.

       Args:
         value: value to be used"""

    # libCEED call
    lib.CeedVectorSetValue(self._pointer[0], value)

  # Sync the Vector to a specified memtype
  def sync_array(self, memtype=MEM_HOST):
    """Sync the Vector to a specified memtype.

       Args:
         **memtype: memtype to be synced"""

    # libCEED call
    lib.CeedVectorSyncArray(self._pointer[0], memtype)

  # Compute the norm of a vector
  def norm(self, normtype=NORM_2):
    """Get the norm of a Vector.

       Args:
         **normtype: type of norm to be computed"""

    norm_pointer = ffi.new("CeedScalar *")

    # libCEED call
    lib.CeedVectorNorm(self._pointer[0], normtype, norm_pointer)

    return norm_pointer[0]

# ------------------------------------------------------------------------------
class _VectorWrap(Vector):
  """Wrap a CeedVector pointer in a Vector object."""

  # Constructor
  def __init__(self, ceed, pointer):
    # CeedVector object
    self._pointer = pointer

    # Reference to Ceed
    self._ceed = ceed

# ------------------------------------------------------------------------------
