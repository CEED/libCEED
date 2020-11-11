# Examples

LibCEED.jl includes three short examples, which are analogues of the two
examples in `libCEED/examples/ceed`.

These examples are:
- `ex1-volume-c.jl`, an almost one-to-one translation of `ex1-volume.c`, using
  the low-level C interface. This example uses low-level user Q-functions
  defined in `ex1-function-c.jl`.
- `ex1-volume.jl`, a higher-level more idiomatic version of `ex1-volume.c`,
  using user Q-functions defined using [`@interior_qf`](@ref).
- `ex2-surface.jl`, a higher-level, idiomatic version of `ex2-surface.c`.
