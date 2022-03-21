# Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

from _ceed_cffi import ffi, lib
import ctypes
import tempfile
from abc import ABC

# ------------------------------------------------------------------------------


class _QFunctionBase(ABC):
    """Ceed QFunction: point-wise operation at quadrature points for evaluating
         volumetric terms."""

    # Destructor
    def __del__(self):
        # libCEED call
        err_code = lib.CeedQFunctionDestroy(self._pointer)
        self._ceed._check_error(err_code)

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

                err_code = lib.CeedQFunctionView(self._pointer[0], stream)
                self._ceed._check_error(err_code)

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
        err_code = lib.CeedQFunctionApply(self._pointer[0], q, invecs, outvecs)
        self._ceed._check_error(err_code)

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
        fpointer = ffi.cast(
            "CeedQFunctionUser", ctypes.cast(
                f, ctypes.c_void_p).value)

        # libCEED call
        sourceAscii = ffi.new("char[]", source.encode('ascii'))
        err_code = lib.CeedQFunctionCreateInterior(self._ceed._pointer[0], vlength,
                                                   fpointer, sourceAscii, self._pointer)
        self._ceed._check_error(err_code)

    # Set context data
    def set_context(self, ctx):
        """Set global context for a QFunction.

           Args:
             ctx: Ceed User Context object holding context data"""

        # libCEED call
        err_code = lib.CeedQFunctionSetContext(
            self._pointer[0],
            ctx._pointer[0])
        self._ceed._check_error(err_code)

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
        err_code = lib.CeedQFunctionAddInput(
            self._pointer[0], fieldnameAscii, size, emode)
        self._ceed._check_error(err_code)

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
        err_code = lib.CeedQFunctionAddOutput(
            self._pointer[0], fieldnameAscii, size, emode)
        self._ceed._check_error(err_code)

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
        err_code = lib.CeedQFunctionCreateInteriorByName(self._ceed._pointer[0],
                                                         nameAscii, self._pointer)
        self._ceed._check_error(err_code)

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
        err_code = lib.CeedQFunctionCreateIdentity(self._ceed._pointer[0], size,
                                                   inmode, outmode, self._pointer)

# ------------------------------------------------------------------------------
