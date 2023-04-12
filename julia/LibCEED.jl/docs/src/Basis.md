# Basis

!!! info "Column-major vs. row-major storage"
    libCEED internally uses row-major (C convention) storage of matrices,
    while Julia uses column-major (Fortran convention) storage.

    LibCEED.jl will typically handle the conversion between these formats by
    transposing or permuting the dimensions of the input and output matrices
    and tensors.

```@docs
Basis
BasisCollocated
create_tensor_h1_lagrange_basis
create_tensor_h1_basis
create_h1_basis
create_hdiv_basis
create_hcurl_basis
apply!(b::Basis, nelem, tmode::TransposeMode, emode::EvalMode, u::LibCEED.AbstractCeedVector, v::LibCEED.AbstractCeedVector)
apply(b::Basis, u::AbstractVector; nelem=1, tmode=NOTRANSPOSE, emode=EVAL_INTERP)
getdimension
gettopology
getnumcomponents
getnumnodes
getnumnodes1d
getnumqpts
getnumqpts1d
getqref
getqweights
getinterp
getinterp1d
getgrad
getgrad1d
getdiv
getcurl
```
