# COV_EXCL_START
using .CUDA, Cassette

#! format: off
const cudafuns = (
    :cos, :cospi, :sin, :sinpi, :tan,
    :acos, :asin, :atan,
    :cosh, :sinh, :tanh,
    :acosh, :asinh, :atanh,
    :log, :log10, :log1p, :log2,
    :exp, :exp2, :exp10, :expm1, :ldexp,
    :abs,
    :sqrt, :cbrt,
    :ceil, :floor,
)
#! format: on

Cassette.@context CeedCudaContext

@inline function Cassette.overdub(::CeedCudaContext, ::typeof(Core.kwfunc), f)
    return Core.kwfunc(f)
end
@inline function Cassette.overdub(::CeedCudaContext, ::typeof(Core.apply_type), args...)
    return Core.apply_type(args...)
end
@inline function Cassette.overdub(
    ::CeedCudaContext,
    ::typeof(StaticArrays.Size),
    x::Type{<:AbstractArray{<:Any,N}},
) where {N}
    return StaticArrays.Size(x)
end

for f in cudafuns
    @eval @inline function Cassette.overdub(
        ::CeedCudaContext,
        ::typeof(Base.$f),
        x::Union{Float32,Float64},
    )
        return CUDA.$f(x)
    end
end

function setarray!(v::CeedVector, mtype::MemType, cmode::CopyMode, arr::CuArray)
    ptr = Ptr{CeedScalar}(UInt64(pointer(arr)))
    C.CeedVectorSetArray(v[], mtype, cmode, ptr)
    if cmode == USE_POINTER
        v.arr = arr
    end
end

struct FieldsCuda
    inputs::NTuple{16,Int}
    outputs::NTuple{16,Int}
end

function generate_kernel(qf_name, kf, dims_in, dims_out)
    ninputs = length(dims_in)
    noutputs = length(dims_out)

    input_sz = prod.(dims_in)
    output_sz = prod.(dims_out)

    f_ins = [Symbol("rqi$i") for i = 1:ninputs]
    f_outs = [Symbol("rqo$i") for i = 1:noutputs]

    args = Vector{Union{Symbol,Expr}}(undef, ninputs + noutputs)
    def_ins = Vector{Expr}(undef, ninputs)
    f_ins_j = Vector{Union{Symbol,Expr}}(undef, ninputs)
    for i = 1:ninputs
        if length(dims_in[i]) == 0
            def_ins[i] = :(local $(f_ins[i]))
            f_ins_j[i] = f_ins[i]
            args[i] = f_ins[i]
        else
            def_ins[i] =
                :($(f_ins[i]) = LibCEED.MArray{Tuple{$(dims_in[i]...)},CeedScalar}(undef))
            f_ins_j[i] = :($(f_ins[i])[j])
            args[i] = :(LibCEED.SArray{Tuple{$(dims_in[i]...)},CeedScalar}($(f_ins[i])))
        end
    end
    for i = 1:noutputs
        args[ninputs+i] = f_outs[i]
    end

    def_outs = [
        :($(f_outs[i]) = LibCEED.MArray{Tuple{$(dims_out[i]...)},CeedScalar}(undef)) for
        i = 1:noutputs
    ]

    device_ptr_type = Core.LLVMPtr{CeedScalar,LibCEED.AS.Global}

    read_quads_in = [
        :(
            for j = 1:$(input_sz[i])
                $(f_ins_j[i]) = unsafe_load(
                    reinterpret($device_ptr_type, fields.inputs[$i]),
                    q + (j - 1)*Q,
                    a,
                )
            end
        ) for i = 1:ninputs
    ]

    write_quads_out = [
        :(
            for j = 1:$(output_sz[i])
                unsafe_store!(
                    reinterpret($device_ptr_type, fields.outputs[$i]),
                    $(f_outs[i])[j],
                    q + (j - 1)*Q,
                    a,
                )
            end
        ) for i = 1:noutputs
    ]

    qf = gensym(qf_name)
    quote
        function $qf(ctx_ptr, Q, fields)
            gd = LibCEED.gridDim()
            bi = LibCEED.blockIdx()
            bd = LibCEED.blockDim()
            ti = LibCEED.threadIdx()

            inc = bd.x*gd.x

            $(def_ins...)
            $(def_outs...)

            # Alignment for data read/write
            a = Val($(Base.datatype_alignment(CeedScalar)))

            # Cassette context for replacing intrinsics with CUDA versions
            ctx = LibCEED.CeedCudaContext()

            for q = (ti.x+(bi.x-1)*bd.x):inc:Q
                $(read_quads_in...)
                LibCEED.Cassette.overdub(ctx, $kf, ctx_ptr, $(args...))
                $(write_quads_out...)
            end
            return
        end
    end
end

function mk_cufunction(ceed, def_module, qf_name, kf, dims_in, dims_out)
    k_fn = Core.eval(def_module, generate_kernel(qf_name, kf, dims_in, dims_out))
    tt = Tuple{Ptr{Nothing},Int32,FieldsCuda}
    host_k = cufunction(k_fn, tt; maxregs=64)
    return host_k.fun.handle
end
# COV_EXCL_STOP
