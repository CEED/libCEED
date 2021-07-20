struct UserQFunction{F,K}
    f::F
    fptr::Ptr{Nothing}
    kf::K
    cuf::Union{Nothing,Ptr{Nothing}}
end

@inline function extract_context(ptr, ::Type{T}) where {T}
    unsafe_load(Ptr{T}(ptr))
end

@inline function extract_array(ptr, idx, dims)
    UnsafeArray(Ptr{CeedScalar}(unsafe_load(ptr, idx)), dims)
end

function generate_user_qfunction(
    ceed,
    def_module,
    qf_name,
    constants,
    array_names,
    ctx,
    dims_in,
    dims_out,
    body,
)
    idx = gensym(:i)
    Q = gensym(:Q)
    ctx_ptr = gensym(:ctx_ptr)
    in_ptr = gensym(:in_ptr)
    out_ptr = gensym(:out_ptr)

    const_assignments = Vector{Expr}(undef, length(constants))
    for (i, c) ∈ enumerate(constants)
        const_assignments[i] = :($(c[1]) = $(c[2]))
    end

    narrays = length(array_names)
    arrays = Vector{Expr}(undef, narrays)
    array_views = Vector{Expr}(undef, narrays)
    n_in = length(dims_in)
    for (i, arr_name) ∈ enumerate(array_names)
        i_inout = (i <= n_in) ? i : i - n_in
        dims = (i <= n_in) ? dims_in[i] : dims_out[i-n_in]
        ptr = (i <= n_in) ? in_ptr : out_ptr
        arr_name_gen = gensym(arr_name)
        arrays[i] = :($arr_name_gen = extract_array($ptr, $i_inout, (Int($Q), $(dims...))))
        ndims = length(dims)
        slice = Expr(:ref, arr_name_gen, idx, (:(:) for i = 1:ndims)...)
        if i <= n_in
            if ndims == 0
                array_views[i] = :($arr_name = $slice)
            else
                S = Tuple{dims...}
                array_views[i] = :($arr_name = LibCEED.SArray{$S}(@view $slice))
            end
        else
            array_views[i] = :($arr_name = @view $slice)
        end
    end

    if isnothing(ctx)
        ctx_assignment = nothing
    else
        ctx_assignment = :($(ctx.name) = extract_context($ctx_ptr, $(ctx.type)))
    end

    qf1 = gensym(qf_name)
    f = Core.eval(
        def_module,
        quote
            @inline function $qf1(
                $ctx_ptr::Ptr{Cvoid},
                $Q::CeedInt,
                $in_ptr::Ptr{Ptr{CeedScalar}},
                $out_ptr::Ptr{Ptr{CeedScalar}},
            )
                $(const_assignments...)
                $ctx_assignment
                $(arrays...)
                @inbounds @simd for $idx = 1:$Q
                    $(array_views...)
                    $body
                end
                CeedInt(0)
            end
        end,
    )
    f_qn = QuoteNode(f)
    rt = :CeedInt
    at = :(Core.svec(Ptr{Cvoid}, CeedInt, Ptr{Ptr{CeedScalar}}, Ptr{Ptr{CeedScalar}}))
    fptr = eval(Expr(:cfunction, Ptr{Cvoid}, f_qn, rt, at, QuoteNode(:ccall)))

    # COV_EXCL_START
    if iscuda(ceed)
        getresource(ceed) == "/gpu/cuda/gen" && error(
            string(
                "/gpu/cuda/gen is not compatible with user Q-functions defined with ",
                "libCEED.jl.\nPlease use a different backend, for example: /gpu/cuda/shared ",
                "or /gpu/cuda/ref",
            ),
        )
        if isdefined(@__MODULE__, :CUDA)
            !has_cuda() && error("No valid CUDA installation found")
            qf2 = gensym(qf_name)
            kf = Core.eval(
                def_module,
                quote
                    @inline function $qf2($ctx_ptr::Ptr{Cvoid}, $(array_names...))
                        $(const_assignments...)
                        $ctx_assignment
                        $body
                        nothing
                    end
                end,
            )
            cuf = mk_cufunction(ceed, def_module, qf_name, kf, dims_in, dims_out)
        else
            error(
                string(
                    "User Q-functions with CUDA backends require the CUDA.jl package to be ",
                    "loaded.\nThe libCEED backend is: $(getresource(ceed))\n",
                    "Please ensure that the CUDA.jl package is installed and loaded.",
                ),
            )
        end
    else
        kf = nothing
        cuf = nothing
    end
    # COV_EXCL_STOP

    UserQFunction(f, fptr, kf, cuf)
end

function meta_user_qfunction(ceed, def_module, qf, args)
    qf_name = Meta.quot(qf)

    ctx = nothing
    constants = Expr[]
    dims_in = Expr[]
    dims_out = Expr[]
    names_in = Symbol[]
    names_out = Symbol[]

    for a ∈ args[1:end-1]
        if Meta.isexpr(a, :(=))
            a1 = Meta.quot(a.args[1])
            a2 = esc(a.args[2])
            push!(constants, :(($a1, $a2)))
        elseif Meta.isexpr(a, :tuple)
            arr_name = a.args[1]
            inout = a.args[2].value
            ndim = length(a.args) - 3
            dims = Vector{Expr}(undef, ndim)
            for d = 1:ndim
                dims[d] = :(Int($(a.args[d+3])))
            end
            dims_expr = :(Int[$(esc.(a.args[4:end])...)])
            if inout == :in
                push!(dims_in, dims_expr)
                push!(names_in, arr_name)
            elseif inout == :out
                push!(dims_out, dims_expr)
                push!(names_out, arr_name)
            else
                error("Array specification must be either :in or :out. Given $inout.")
            end
        elseif Meta.isexpr(a, :(::))
            ctx = (name=a.args[1], type=a.args[2])
        else
            error("Bad argument to @user_qfunction")
        end
    end

    body = Meta.quot(args[end])

    return :(generate_user_qfunction(
        $ceed,
        $def_module,
        $qf_name,
        [$(constants...)],
        $([names_in; names_out]),
        $ctx,
        [$(dims_in...)],
        [$(dims_out...)],
        $body,
    ))
end

"""
    @interior_qf name=def

Creates a user-defined interior (volumetric) Q-function, and assigns it to a variable named
`name`. The definition of the Q-function is given as:
```
@interior_qf user_qf=(
    ceed::CEED,
    [const1=val1, const2=val2, ...],
    [ctx::ContextType],
    (I1, :in, EvalMode, dims...),
    (I2, :in, EvalMode, dims...),
    (O1, :out, EvalMode, dims...),
    body
)
```
The definitions of form `const=val` are used for definitions which will be compile-time
constants in the Q-function. For example, if `dim` is a variable set to the dimension of the
problem, then `dim=dim` will make `dim` available in the body of the Q-function as a
compile-time constant.

If the user wants to provide a context struct to the Q-function, that can be achieved by
optionally including `ctx::ContextType`, where `ContextType` is the type of the context
struct, and `ctx` is the name to which is will be bound in the body of the Q-function.

This is followed by the definition of the input and output arrays, which take the form
`(arr_name, (:in|:out), EvalMode, dims...)`. Each array will be bound to a variable named
`arr_name`. Input arrays should be tagged with :in, and output arrays with :out. An
`EvalMode` should be specified, followed by the dimensions of the array. If the array
consists of scalars (one number per Q-point) then `dims` should be omitted.

# Examples

- Q-function to compute the "Q-data" for the mass operator, which is given by the quadrature
  weight times the Jacobian determinant. The mesh Jacobian (the gradient of the nodal mesh
  points) and the quadrature weights are given as input arrays, and the Q-data is the output
  array. `dim` is given as a compile-time constant, and so the array `J` is statically
  sized, and therefore `det(J)` will automatically dispatch to an optimized implementation
  for the given dimension.
```
@interior_qf build_qfunc = (
    ceed, dim=dim,
    (J, :in, EVAL_GRAD, dim, dim),
    (w, :in, EVAL_WEIGHT),
    (qdata, :out, EVAL_NONE),
    qdata[] = w*det(J)
)
```
"""
macro interior_qf(args)
    if !Meta.isexpr(args, :(=))
        error("@interior_qf must be of form `qf = (body)`") # COV_EXCL_LINE
    end

    qf = args.args[1]
    user_qf = esc(qf)
    args = args.args[2].args
    ceed = esc(args[1])

    # Calculate field sizes
    fields_in = Expr[]
    fields_out = Expr[]
    for a ∈ args
        if Meta.isexpr(a, :tuple)
            field_name = String(a.args[1])
            inout = a.args[2].value
            evalmode = a.args[3]
            ndim = length(a.args) - 3
            dims = Vector{Expr}(undef, ndim)
            for d = 1:ndim
                dims[d] = esc(:(Int($(a.args[d+3]))))
            end
            sz_expr = :(prod(($(dims...),)))
            if inout == :in
                push!(fields_in, :(add_input!($user_qf, $field_name, $sz_expr, $evalmode)))
            elseif inout == :out
                push!(
                    fields_out,
                    :(add_output!($user_qf, $field_name, $sz_expr, $evalmode)),
                )
            end
        end
    end

    gen_user_qf = meta_user_qfunction(ceed, __module__, qf, args[2:end])

    quote
        $user_qf = create_interior_qfunction($ceed, $gen_user_qf)
        $(fields_in...)
        $(fields_out...)
    end
end
