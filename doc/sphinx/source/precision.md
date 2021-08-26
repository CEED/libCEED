# Floating Point Precision

Currently, libCEED supports two options for {code}`CeedScalar` : double and single.  The default is to use 
double precision.  Users wishing to set {code}`CeedScalar` to single precision should edit `include/ceed/ceed.h` and change

```{literalinclude} ../../../include/ceed/ceed.h
:end-at: "#include \"ceed-f64.h\""
:language: c
:start-at: "#include \"ceed-f64.h\""
```

to include {code}`ceed-f32.h` instead, then recompile the library.

## Language-specific notes

 - **C**: {code}`CEED_SCALAR_TYPE` will be defined to match one of the values of the {code}`CeedScalarType` {code}`enum`, and can be used 
       for compile-time checking of {code}`CeedScalar`'s type; see, e.g., {code}`tests/t314-basis.c`.

 - **Fortran**: There is no definition of {code}`CeedScalar` available in the Fortran header.  The user is responsible for ensuring
            that data used in Fortran code is of the correct type ({code}`real*8` or {code}`real*4`) for libCEED's current configuration.

 - **Julia**: After compiling the single precision version of libCEED, instruct LibCEED.jl to use this library with the {code}`set_libceed_path!`
              function and restart the Julia session. LibCEED.jl will configure itself to use the appropriate type for {code}`CeedScalar`. 

 - **Python**: Make sure to replace the {code}`ceed-f64.h` inclusion rather than commenting it out, to guarantee that the Python
           bindings will pick the correct precision.
           The {c:func}`scalar_type()` function has been added to the {code}`Ceed` class for convenience.  It returns a string 
           corresponding to a numpy datatype matching that of {code}`CeedScalar`.

 - **Rust**: The {code}`Scalar` type corresponds to {code}`CeedScalar`.

**This is work in progress!**  The ability to use single precision is an initial step in ongoing development of mixed-precision support in libCEED.
A current GitHub [issue](https://github.com/CEED/libCEED/issues/778) contains discussions related to this development.
