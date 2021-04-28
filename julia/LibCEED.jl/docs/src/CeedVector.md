# CeedVector

```@docs
CeedVector
setvalue!
Base.setindex!(v::CeedVector, v2::CeedScalar)
Base.setindex!(v::CeedVector, v2::AbstractArray)
Base.Vector(v::CeedVector)
LinearAlgebra.norm(v::CeedVector, n::NormType)
LinearAlgebra.norm(v::CeedVector, p::Real)
@witharray
@witharray_read
witharray
witharray_read
setarray!
syncarray!
takearray!
scale!
LinearAlgebra.axpy!(a::Real, x::CeedVector, y::CeedVector)
pointwisemult!
```
