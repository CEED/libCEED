mutable struct Context
    ref::RefValue{C.CeedQFunctionContext}
    data::Any
    function Context(ref::Ref{C.CeedQFunctionContext})
        obj = new(ref)
        finalizer(obj) do x
            # ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "Finalizing %s.\n", repr(x))
            C.CeedQFunctionContextDestroy(x.ref)
        end
        return obj
    end
end
Base.getindex(ctx::Context) = ctx.ref[]
Base.show(io::IO, ::MIME"text/plain", c::Context) =
    ceed_show(io, c, C.CeedQFunctionContextView)

"""
    Context(ceed::Ceed, data; mtype=MEM_HOST, cmode=USE_POINTER)

Create a `CeedQFunctionContext` object that allows user Q-functions to access an arbitrary
data object. `data` should be an instance of a mutable struct. If the copy mode `cmode` is
`USE_POINTER`, then the data will be preserved from the GC when assigned to a `QFunction`
object using `set_context!`.

Copy mode `OWN_POINTER` is not supported by this interface because Julia-allocated objects
cannot be freed from C.
"""
function Context(c::Ceed, data; mtype=MEM_HOST, cmode=USE_POINTER)
    ref = Ref{C.CeedQFunctionContext}()
    C.CeedQFunctionContextCreate(c[], ref)
    ctx = Context(ref)
    set_data!(ctx, mtype, cmode, data)
    return ctx
end

function set_data!(ctx::Context, mtype, cmode::CopyMode, data)
    # Store a reference to the context data so that it will not be GC'd before
    # it is accessed in the user Q-function.
    # A reference to the context object is stored in the QFunction object, and
    # references to the QFunctions are stored in the Operator.
    # This means that when `apply!(op, ...)` is called, the context data is
    # ensured to be valid.
    if cmode == USE_POINTER
        ctx.data = data
    elseif cmode == OWN_POINTER
        error("set_data!: copy mode OWN_POINTER is not supported")
    end

    C.CeedQFunctionContextSetData(
        ctx[],
        mtype,
        cmode,
        sizeof(data),
        pointer_from_objref(data),
    )
end
