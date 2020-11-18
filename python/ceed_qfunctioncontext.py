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
from .ceed_constants import MEM_HOST, USE_POINTER, COPY_VALUES

# ------------------------------------------------------------------------------


class QFunctionContext():
    """Ceed QFunction Context: stores Ceed QFunction user context data."""

    # Constructor
    def __init__(self, ceed):
        # CeedQFunctionContext object
        self._pointer = ffi.new("CeedQFunctionContext *")

        # Reference to Ceed
        self._ceed = ceed

        # libCEED call
        err_code = lib.CeedQFunctionContextCreate(
            self._ceed._pointer[0], self._pointer)
        self._ceed._check_error(err_code)

    # Destructor
    def __del__(self):
        # libCEED call
        err_code = lib.CeedQFunctionContextDestroy(self._pointer)
        self._ceed._check_error(err_code)

    # Representation
    def __repr__(self):
        return "<CeedQFunctionContext instance at " + hex(id(self)) + ">"

    # String conversion for print() to stdout
    def __str__(self):
        """View a QFunction Context via print()."""

        # libCEED call
        fmt = ffi.new("char[]", "%f".encode('ascii'))
        with tempfile.NamedTemporaryFile() as key_file:
            with open(key_file.name, 'r+') as stream_file:
                stream = ffi.cast("FILE *", stream_file)

                err_code = lib.CeedQFunctionContextView(
                    self._pointer[0], stream)
                self._ceed._check_error(err_code)

                stream_file.seek(0)
                out_string = stream_file.read()

        return out_string

    # Set QFunction Context's data
    def set_data(self, data, memtype=MEM_HOST, cmode=COPY_VALUES):
        """Set the data used by a QFunction Context, freeing any previously allocated
           data if applicable.

           Args:
             *data: Numpy or Numba array to be used
             **memtype: memory type of the array being passed, default CEED_MEM_HOST
             **cmode: copy mode for the array, default CEED_COPY_VALUES"""

        # Store array reference if needed
        if cmode == USE_POINTER:
            self._array_reference = data
        else:
            self._array_reference = None

        # Setup the numpy array for the libCEED call
        data_pointer = ffi.new("CeedScalar *")
        if memtype == MEM_HOST:
            data_pointer = ffi.cast(
                "void *",
                data.__array_interface__['data'][0])
        else:
            array_pointer = ffi.cast(
                "void *",
                data.__cuda_array_interface__['data'][0])

        # libCEED call
        err_code = lib.CeedQFunctionContextSetData(
            self._pointer[0],
            memtype,
            cmode,
            len(data) * ffi.sizeof("CeedScalar"),
            data_pointer)
        self._ceed._check_error(err_code)

    # Get QFunction Context's data
    def get_data(self, memtype=MEM_HOST):
        """Get read/write access to a QFunction Context via the specified memory type.

           Args:
             **memtype: memory type of the array being passed, default CEED_MEM_HOST

           Returns:
             *data: Numpy or Numba array"""

        # Retrieve the length of the array
        size_pointer = ffi.new("CeedInt *")
        err_code = lib.CeedQFunctionContextGetContextSize(
            self._pointer[0], size_pointer)
        self._ceed._check_error(err_code)

        # Setup the pointer's pointer
        data_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedQFunctionContextGetData(
            self._pointer[0], memtype, data_pointer)
        self._ceed._check_error(err_code)

        # Return array created from buffer
        if memtype == MEM_HOST:
            # Create buffer object from returned pointer
            buff = ffi.buffer(
                data_pointer[0],
                ffi.sizeof("CeedScalar") *
                length_pointer[0])
            # return Numpy array
            return np.frombuffer(buff, dtype="float64")
        else:
            # CUDA array interface
            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
            import numba.cuda as nbcuda
            desc = {
                'shape': (length_pointer[0]),
                'typestr': '>f8',
                'data': (int(ffi.cast("intptr_t", data_pointer[0])), False),
                'version': 2
            }
            # return Numba array
            return nbcuda.from_cuda_array_interface(desc)

    # Restore the QFunction Context's data
    def restore_data(self):
        """Restore an array obtained using get_data()."""

        # Setup the pointer's pointer
        data_pointer = ffi.new("CeedScalar **")

        # libCEED call
        err_code = lib.CeedQFunctionDataRestoreData(
            self._pointer[0], data_pointer)
        self._ceed._check_error(err_code)

    @contextlib.contextmanager
    def data(self, *shape, memtype=MEM_HOST):
        """Context manager for array access.

        Args:
          shape (tuple): shape of returned numpy.array
          **memtype: memory type of the data being passed, default CEED_MEM_HOST


        Returns:
          np.array: writable view of QFunction Context
        """
        x = self.get_data(memtype=memtype)
        if shape:
            x = x.reshape(shape)
        yield x
        self.restore_data()

# ------------------------------------------------------------------------------
