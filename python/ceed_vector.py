# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

from _ceed_cffi import ffi, lib
import tempfile
import numpy as np
import contextlib
from .ceed_constants import MEM_HOST, USE_POINTER, COPY_VALUES, NORM_2, scalar_types

# ------------------------------------------------------------------------------


class Vector():
    """Ceed Vector: storing and manipulating vectors."""

    # Constructor
    def __init__(self, ceed, size):
        # CeedVector object
        self._pointer = ffi.new("CeedVector *")

        # Reference to Ceed
        self._ceed = ceed

        # libCEED call
        err_code = lib.CeedVectorCreate(
            self._ceed._pointer[0], size, self._pointer)
        self._ceed._check_error(err_code)

    # Destructor
    def __del__(self):
        # libCEED call
        err_code = lib.CeedVectorDestroy(self._pointer)
        self._ceed._check_error(err_code)

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

                err_code = lib.CeedVectorView(self._pointer[0], fmt, stream)
                self._ceed._check_error(err_code)

                stream_file.seek(0)
                out_string = stream_file.read()

        return out_string

    # Copy the array from a vector into self
    def copy_from(self, vec_source):
        """Copies the array of vec_source into the array of self.

           Args:
             *vector: the Vector to copy from"""

        # libCEED call
        err_code = lib.CeedVectorCopy(vec_source._pointer[0], self._pointer[0])
        self._ceed._check_error(err_code)

    # Set Vector's data array
    def set_array(self, array, memtype=MEM_HOST, cmode=COPY_VALUES):
        """Set the array used by a Vector, freeing any previously allocated
           array if applicable.

           Args:
             *array: Numpy or Numba array to be used
             **memtype: memory type of the array being passed, default CEED_MEM_HOST
             **cmode: copy mode for the array, default CEED_COPY_VALUES"""

        # Store array reference if needed
        if cmode == USE_POINTER:
            self._array_reference = array
        else:
            self._array_reference = None

        # Setup the numpy array for the libCEED call
        array_pointer = ffi.new("CeedScalar *")
        if memtype == MEM_HOST:
            array_pointer = ffi.cast(
                "CeedScalar *",
                array.__array_interface__['data'][0])
        else:
            array_pointer = ffi.cast(
                "CeedScalar *",
                array.__cuda_array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedVectorSetArray(
            self._pointer[0], memtype, cmode, array_pointer)
        self._ceed._check_error(err_code)

    # Get Vector's data array
    def get_array(self, memtype=MEM_HOST):
        """Get read/write access to a Vector via the specified memory type.

           Args:
             **memtype: memory type of the array being passed, default CEED_MEM_HOST

           Returns:
             *array: Numpy or Numba array"""

        # Retrieve the length of the array
        length_pointer = ffi.new("CeedSize *")
        err_code = lib.CeedVectorGetLength(self._pointer[0], length_pointer)
        self._ceed._check_error(err_code)

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedVectorGetArray(
            self._pointer[0], memtype, array_pointer)
        self._ceed._check_error(err_code)

        # Return array created from buffer
        if memtype == MEM_HOST:
            # Create buffer object from returned pointer
            buff = ffi.buffer(
                array_pointer[0],
                ffi.sizeof("CeedScalar") *
                length_pointer[0])
            # return Numpy array
            return np.frombuffer(buff, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
        else:
            # CUDA array interface
            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
            import numba.cuda as nbcuda
            desc = {
                'shape': (length_pointer[0]),
                'typestr': '>f8',
                'data': (int(ffi.cast("intptr_t", array_pointer[0])), False),
                'version': 2
            }
            # return Numba array
            return nbcuda.from_cuda_array_interface(desc)

    # Get Vector's data array in read-only mode
    def get_array_read(self, memtype=MEM_HOST):
        """Get read-only access to a Vector via the specified memory type.

           Args:
             **memtype: memory type of the array being passed, default CEED_MEM_HOST

           Returns:
             *array: Numpy or Numba array"""

        # Retrieve the length of the array
        length_pointer = ffi.new("CeedSize *")
        err_code = lib.CeedVectorGetLength(self._pointer[0], length_pointer)
        self._ceed._check_error(err_code)

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedVectorGetArrayRead(
            self._pointer[0], memtype, array_pointer)
        self._ceed._check_error(err_code)

        # Return array created from buffer
        if memtype == MEM_HOST:
            # Create buffer object from returned pointer
            buff = ffi.buffer(
                array_pointer[0],
                ffi.sizeof("CeedScalar") *
                length_pointer[0])
            # return read only Numpy array
            ret = np.frombuffer(buff, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
            ret.flags['WRITEABLE'] = False
            return ret
        else:
            # CUDA array interface
            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
            import numba.cuda as nbcuda
            desc = {
                'shape': (length_pointer[0]),
                'typestr': '>f8',
                'data': (int(ffi.cast("intptr_t", array_pointer[0])), False),
                'version': 2
            }
            # return read only Numba array
            return nbcuda.from_cuda_array_interface(desc)

    # Get Vector's data array in write-only mode
    def get_array_write(self, memtype=MEM_HOST):
        """Get write-only access to a Vector via the specified memory type.
           All old values should be considered invalid.

           Args:
             **memtype: memory type of the array being passed, default CEED_MEM_HOST

           Returns:
             *array: Numpy or Numba array"""

        # Retrieve the length of the array
        length_pointer = ffi.new("CeedSize *")
        err_code = lib.CeedVectorGetLength(self._pointer[0], length_pointer)
        self._ceed._check_error(err_code)

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedVectorGetArrayWrite(
            self._pointer[0], memtype, array_pointer)
        self._ceed._check_error(err_code)

        # Return array created from buffer
        if memtype == MEM_HOST:
            # Create buffer object from returned pointer
            buff = ffi.buffer(
                array_pointer[0],
                ffi.sizeof("CeedScalar") *
                length_pointer[0])
            # return Numpy array
            return np.frombuffer(buff, dtype=scalar_types[lib.CEED_SCALAR_TYPE])
        else:
            # CUDA array interface
            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
            import numba.cuda as nbcuda
            desc = {
                'shape': (length_pointer[0]),
                'typestr': '>f8',
                'data': (int(ffi.cast("intptr_t", array_pointer[0])), False),
                'version': 2
            }
            # return Numba array
            return nbcuda.from_cuda_array_interface(desc)

    # Restore the Vector's data array
    def restore_array(self):
        """Restore an array obtained using get_array()."""

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedVectorRestoreArray(self._pointer[0], array_pointer)
        self._ceed._check_error(err_code)

    # Restore an array obtained using getArrayRead
    def restore_array_read(self):
        """Restore an array obtained using get_array_read()."""

        # Setup the pointer's pointer
        array_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedVectorRestoreArrayRead(
            self._pointer[0], array_pointer)
        self._ceed._check_error(err_code)

    @contextlib.contextmanager
    def array(self, *shape, memtype=MEM_HOST):
        """Context manager for array access.

        Args:
          shape (tuple): shape of returned numpy.array
          **memtype: memory type of the array being passed, default CEED_MEM_HOST


        Returns:
          np.array: writable view of vector

        Examples:
          Constructing the identity inside a libceed.Vector:

          >>> vec = ceed.Vector(16)
          >>> with vec.array(4, 4) as x:
          >>>     x[...] = np.eye(4)
        """
        x = self.get_array(memtype=memtype)
        if shape:
            x = x.reshape(shape)
        yield x
        self.restore_array()

    @contextlib.contextmanager
    def array_read(self, *shape, memtype=MEM_HOST):
        """Context manager for read-only array access.

        Args:
          shape (tuple): shape of returned numpy.array
          **memtype: memory type of the array being passed, default CEED_MEM_HOST

        Returns:
          np.array: read-only view of vector

        Examples:
          Viewing contents of a reshaped libceed.Vector view:

          >>> vec = ceed.Vector(6)
          >>> vec.set_value(1.3)
          >>> with vec.array_read(2, 3) as x:
          >>>     print(x)
        """
        x = self.get_array_read(memtype=memtype)
        if shape:
            x = x.reshape(shape)
        yield x
        self.restore_array_read()

    @contextlib.contextmanager
    def array_write(self, *shape, memtype=MEM_HOST):
        """Context manager for write-only array access.
           All old values should be considered invalid.

        Args:
          shape (tuple): shape of returned numpy.array
          **memtype: memory type of the array being passed, default CEED_MEM_HOST

        Returns:
          np.array: write-only view of vector

        Examples:
          Viewing contents of a reshaped libceed.Vector view:

          >>> vec = ceed.Vector(6)
          >>> vec.set_value(1.3)
          >>> with vec.array_read(2, 3) as x:
          >>>     print(x)
        """
        x = self.get_array_write(memtype=memtype)
        if shape:
            x = x.reshape(shape)
        yield x
        self.restore_array()

    # Get the length of a Vector
    def get_length(self):
        """Get the length of a Vector.

           Returns:
             length: length of the Vector"""

        length_pointer = ffi.new("CeedSize *")

        # libCEED call
        err_code = lib.CeedVectorGetLength(self._pointer[0], length_pointer)
        self._ceed._check_error(err_code)

        return length_pointer[0]

    # Get the length of a Vector
    def __len__(self):
        """Get the length of a Vector.

           Returns:
             length: length of the Vector"""

        length_pointer = ffi.new("CeedSize *")

        # libCEED call
        err_code = lib.CeedVectorGetLength(self._pointer[0], length_pointer)
        self._ceed._check_error(err_code)

        return length_pointer[0]

    # Set the Vector to a given constant value
    def set_value(self, value):
        """Set the Vector to a constant value.

           Args:
             value: value to be used"""

        # libCEED call
        err_code = lib.CeedVectorSetValue(self._pointer[0], value)
        self._ceed._check_error(err_code)

    # Sync the Vector to a specified memtype
    def sync_array(self, memtype=MEM_HOST):
        """Sync the Vector to a specified memtype.

           Args:
             **memtype: memtype to be synced"""

        # libCEED call
        err_code = lib.CeedVectorSyncArray(self._pointer[0], memtype)
        self._ceed._check_error(err_code)

    # Compute the norm of a vector
    def norm(self, normtype=NORM_2):
        """Get the norm of a Vector.

           Args:
             **normtype: type of norm to be computed"""

        norm_pointer = ffi.new("CeedScalar *")

        # libCEED call
        err_code = lib.CeedVectorNorm(self._pointer[0], normtype, norm_pointer)
        self._ceed._check_error(err_code)

        return norm_pointer[0]

    # Take the reciprocal of a vector
    def reciprocal(self):
        """Take the reciprocal of a Vector."""

        # libCEED call
        err_code = lib.CeedVectorReciprocal(self._pointer[0])
        self._ceed._check_error(err_code)

        return self

    # Compute self = alpha self
    def scale(self, alpha):
        """Compute self = alpha self."""

        # libCEED call
        err_code = lib.CeedVectorScale(self._pointer[0], alpha)
        self._ceed._check_error(err_code)

        return self

    # Compute self = alpha x + self
    def axpy(self, alpha, x):
        """Compute self = alpha x + self."""

        # libCEED call
        err_code = lib.CeedVectorAXPY(self._pointer[0], alpha, x._pointer[0])
        self._ceed._check_error(err_code)

        return self

    # Compute self = alpha x + beta self
    def axpby(self, alpha, beta, x):
        """Compute self = alpha x + beta self."""

        # libCEED call
        err_code = lib.CeedVectorAXPBY(
            self._pointer[0], alpha, beta, x._pointer[0])
        self._ceed._check_error(err_code)

        return self

    # Compute the pointwise multiplication self = x .* y
    def pointwise_mult(self, x, y):
        """Compute the pointwise multiplication self = x .* y."""

        # libCEED call
        err_code = lib.CeedVectorPointwiseMult(
            self._pointer[0], x._pointer[0], y._pointer[0]
        )
        self._ceed._check_error(err_code)

        return self

    def _state(self):
        """Return the modification state of the Vector.

        State is incremented each time the Vector is mutated, and is odd whenever a
        mutable reference has not been returned.
        """

        state_pointer = ffi.new("uint64_t *")
        err_code = lib.CeedVectorGetState(self._pointer[0], state_pointer)
        self._ceed._check_error(err_code)
        return state_pointer[0]

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
