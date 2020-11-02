import LinearAlgebra: det

"""
    CeedDim(dim)

The singleton object of type `CeedDim{dim}`, used for dispatch to linear algebra operations
specialized for small matrices (1, 2, or 3 dimensions).
"""
struct CeedDim{dim} end
@inline CeedDim(dim) = CeedDim{Int(dim)}()

"""
    det(J, ::CeedDim{dim})

Specialized determinant calculations for matrices of size 1, 2, or 3.
"""
@inline det(J, ::CeedDim{1}) = @inbounds J[1]
@inline det(J, ::CeedDim{2}) = @inbounds J[1]*J[4] - J[3]*J[2]
#! format: off
@inline det(J, ::CeedDim{3}) = @inbounds (
    J[1]*(J[5]*J[9] - J[6]*J[8]) -
    J[2]*(J[4]*J[9] - J[6]*J[7]) +
    J[3]*(J[4]*J[8] - J[5]*J[7])
)
#! format: on

"""
    setvoigt(J::StaticArray{Tuple{D,D},T,2})
    setvoigt(J, ::CeedDim{dim})

Given a symmetric matrix `J`, return a `SVector` that encodes `J` using the [Voigt
convention](https://en.wikipedia.org/wiki/Voigt_notation).

The size of the symmetric matrix `J` must be known statically, either specified using
[`CeedDim`](@ref) or `StaticArray`.
"""
@inline setvoigt(J::StaticArray{Tuple{D,D}}) where {D} = setvoigt(J, CeedDim(D))
@inline setvoigt(J, ::CeedDim{1}) = @inbounds @SVector [J[1]]
@inline setvoigt(J, ::CeedDim{2}) = @inbounds @SVector [J[1], J[4], J[2]]
@inline setvoigt(J, ::CeedDim{3}) = @inbounds @SVector [J[1], J[5], J[9], J[6], J[3], J[2]]

@inline function setvoigt!(V, J, ::CeedDim{1})
    @inbounds V[1] = J[1]
end

@inline function setvoigt!(V, J, ::CeedDim{2})
    @inbounds begin
        V[1] = J[1]
        V[2] = J[4]
        V[3] = J[2]
    end
end

@inline function setvoigt!(V, J, ::CeedDim{3})
    @inbounds begin
        V[1] = J[1]
        V[2] = J[5]
        V[3] = J[9]
        V[4] = J[6]
        V[5] = J[3]
        V[6] = J[2]
    end
end

"""
    getvoigt(V, ::CeedDim{dim})

Given a vector `V` that encodes a symmetric matrix using the [Voigt
convention](https://en.wikipedia.org/wiki/Voigt_notation), return the corresponding
`SMatrix`.
"""
@inline getvoigt(V, ::CeedDim{1}) = @inbounds @SMatrix [V[1]]
@inline getvoigt(V, ::CeedDim{2}) = @inbounds @SMatrix [V[1] V[3]; V[3] V[2]]
@inline getvoigt(V, ::CeedDim{3}) = @inbounds @SMatrix [
    V[1] V[6] V[5]
    V[6] V[2] V[4]
    V[5] V[4] V[3]
]
@inline getvoigt(V::StaticArray{Tuple{1}}) = getvoigt(V, CeedDim(1))
@inline getvoigt(V::StaticArray{Tuple{3}}) = getvoigt(V, CeedDim(2))
@inline getvoigt(V::StaticArray{Tuple{6}}) = getvoigt(V, CeedDim(3))

@inline function getvoigt!(J, V, ::CeedDim{1})
    @inbounds J[1, 1] = V[1]
end

@inline function getvoigt!(J, V, ::CeedDim{2})
    @inbounds begin
        #! format: off
        J[1,1] = V[1] ; J[1,2] = V[3]
        J[2,1] = V[3] ; J[2,2] = V[2]
        #! format: on
    end
end

@inline function getvoigt!(J, V, ::CeedDim{3})
    @inbounds begin
        #! format: off
        J[1,1] = V[1] ; J[1,2] = V[6] ; J[1,3] = V[5]
        J[2,1] = V[6] ; J[2,2] = V[2] ; J[2,3] = V[4]
        J[3,1] = V[5] ; J[3,2] = V[4] ; J[3,3] = V[3]
        #! format: on
    end
end

function tmp_view(obj, view_fn)
    str = mktemp() do fname, f
        cf = Libc.FILE(f)
        er = view_fn(obj, cf.ptr)
        ccall(:fflush, Cint, (Ptr{Cvoid},), cf)
        seek(f, 0)
        read(f, String)
    end
    chomp(str)
end

function ceed_show(io::IO, obj, view_fn)
    print(io, tmp_view(obj[], view_fn))
end
