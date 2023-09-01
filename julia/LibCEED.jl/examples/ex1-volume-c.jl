using LibCEED.C, Printf, UnsafeArrays
using LibCEED.C: CeedInt, CeedScalar

# A structure used to pass additional data to f_build_mass
mutable struct BuildContextC
    dim::CeedInt
    space_dim::CeedInt
end

# libCEED Q-function for building quadrature data for a mass operator
function f_build_mass_c(
    ctx_ptr::Ptr{Cvoid},
    Q::CeedInt,
    in_ptr::Ptr{Ptr{CeedScalar}},
    out_ptr::Ptr{Ptr{CeedScalar}},
)
    # in[0] is Jacobians with shape [dim, nc=dim, Q]
    # in[1] is quadrature weights, size (Q)
    ctx = unsafe_load(Ptr{BuildContextC}(ctx_ptr))
    J = UnsafeArray(unsafe_load(in_ptr, 1), (Int(Q), Int(ctx.dim^2)))
    w = UnsafeArray(unsafe_load(in_ptr, 2), (Int(Q),))
    qdata = UnsafeArray(unsafe_load(out_ptr, 1), (Int(Q),))
    if ctx.dim == 1
        @inbounds @simd for i = 1:Q
            qdata[i] = J[i]*w[i]
        end
    elseif ctx.dim == 2
        @inbounds @simd for i = 1:Q
            qdata[i] = (J[i, 1]*J[i, 4] - J[i, 2]*J[i, 3])*w[i]
        end
    elseif ctx.dim == 3
        @inbounds @simd for i = 1:Q
            qdata[i] =
                (
                    J[i, 1]*(J[i, 5]*J[i, 9] - J[i, 6]*J[i, 8]) -
                    J[i, 2]*(J[i, 4]*J[i, 9] - J[i, 6]*J[i, 7]) +
                    J[i, 3]*(J[i, 4]*J[i, 8] - J[i, 5]*J[i, 7])
                )*w[i]
        end
    else
        error("Bad dimension")
    end
    return CeedInt(0)
end

# libCEED Q-function for applying a mass operator
function f_apply_mass_c(
    ctx,
    Q::CeedInt,
    in_ptr::Ptr{Ptr{CeedScalar}},
    out_ptr::Ptr{Ptr{CeedScalar}},
)
    u = UnsafeArray(unsafe_load(in_ptr, 1), (Int(Q),))
    qdata = UnsafeArray(unsafe_load(in_ptr, 2), (Int(Q),))
    v = UnsafeArray(unsafe_load(out_ptr, 1), (Int(Q),))
    @inbounds @simd for i = 1:Q
        v[i] = qdata[i]*u[i]
    end
    return CeedInt(0)
end

function get_cartesian_mesh_size_c(dim, order, prob_size)
    dims = zeros(Int, dim)
    # Use the approximate formula:
    #    prob_size ~ num_elem * order^dim
    num_elem = div(prob_size, order^dim)
    s = 0 # find s: num_elem/2 < 2^s <= num_elem
    while num_elem > 1
        num_elem = div(num_elem, 2)
        s += 1
    end
    r = s%dim
    for d = 1:dim
        sd = div(s, dim)
        if r > 0
            sd += 1
            r -= 1
        end
        dims[d] = 2^sd
    end
    dims
end

function build_cartesian_restriction_c(
    ceed,
    dim,
    nxyz,
    order,
    ncomp,
    num_qpts;
    form_strided=false,
)
    p = order
    pp1 = p + 1
    nnodes = pp1^dim # number of scal. nodes per element
    elem_qpts = num_qpts^dim # number of qpts per element
    num_elem = 1
    scalar_size = 1

    nd = p*nxyz .+ 1
    num_elem = prod(nxyz)
    scalar_size = prod(nd)
    size = scalar_size*ncomp

    # elem:         0             1                 n-1
    #        |---*-...-*---|---*-...-*---|- ... -|--...--|
    # nnodes:   0   1    p-1  p  p+1       2*p             n*p

    el_nodes = zeros(C.CeedInt, num_elem*nnodes)
    exyz = zeros(Int, dim)
    @inbounds for e = 0:(num_elem-1)
        re = e
        for d = 1:dim
            exyz[d] = re%nxyz[d]
            re = div(re, nxyz[d])
        end
        for lnodes = 0:(nnodes-1)
            gnodes = 0
            gnodes_stride = 1
            rnodes = lnodes
            for d = 1:dim
                gnodes += (exyz[d]*p + rnodes%pp1)*gnodes_stride
                gnodes_stride *= nd[d]
                rnodes = div(rnodes, pp1)
            end
            el_nodes[e*nnodes+lnodes+1] = gnodes
        end
    end

    rstr = Ref{C.CeedElemRestriction}()
    C.CeedElemRestrictionCreate(
        ceed[],
        num_elem,
        nnodes,
        ncomp,
        scalar_size,
        ncomp*scalar_size,
        C.CEED_MEM_HOST,
        C.CEED_COPY_VALUES,
        el_nodes,
        rstr,
    )
    if form_strided
        rstr_i = Ref{C.CeedElemRestriction}()
        err = C.CeedElemRestrictionCreateStrided(
            ceed[],
            num_elem,
            elem_qpts,
            ncomp,
            ncomp*elem_qpts*num_elem,
            C.CEED_STRIDES_BACKEND[],
            rstr_i,
        )
        return size, rstr, rstr_i
    else
        return size, rstr
    end
end

function set_cartesian_mesh_coords_c(dim, nxyz, mesh_order, mesh_coords)
    p = mesh_order
    nd = p*nxyz .+ 1
    num_elem = prod(nxyz)
    scalar_size = prod(nd)

    coords_ref = Ref{Ptr{C.CeedScalar}}()
    C.CeedVectorGetArray(mesh_coords[], C.CEED_MEM_HOST, coords_ref)
    coords = unsafe_wrap(Array, coords_ref[], scalar_size*dim)

    nodes = zeros(C.CeedScalar, p + 1)
    # The H1 basis uses Lobatto quadrature points as nodes.
    C.CeedLobattoQuadrature(p + 1, nodes, C_NULL) # nodes are in [-1,1]
    nodes = 0.5 .+ 0.5*nodes
    for gsnodes = 0:(scalar_size-1)
        rnodes = gsnodes
        for d = 1:dim
            d1d = rnodes%nd[d]
            coords[gsnodes+scalar_size*(d-1)+1] = (div(d1d, p) + nodes[d1d%p+1])/nxyz[d]
            rnodes = div(rnodes, nd[d])
        end
    end
    C.CeedVectorRestoreArray(mesh_coords[], coords_ref)
end

function transform_mesh_coords_c(dim, mesh_size, mesh_coords)
    coords_ref = Ref{Ptr{C.CeedScalar}}()
    C.CeedVectorGetArray(mesh_coords[], C.CEED_MEM_HOST, coords_ref)
    coords = unsafe_wrap(Array, coords_ref[], mesh_size)

    if dim == 1
        for i = 1:mesh_size
            # map [0,1] to [0,1] varying the mesh density
            coords[i] = 0.5 + 1.0/sqrt(3.0)*sin((2.0/3.0)*pi*(coords[i] - 0.5))
        end
        exact_volume = 1
    else
        num_nodes = div(mesh_size, dim)
        for i = 1:num_nodes
            # map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
            # coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
            u = coords[i]
            v = coords[i+num_nodes]
            u = 1.0 + u
            v = pi/2*v
            coords[i] = u*cos(v)
            coords[i+num_nodes] = u*sin(v)
        end
        exact_volume = 3.0/4.0*pi
    end

    C.CeedVectorRestoreArray(mesh_coords[], coords_ref)
    return exact_volume
end

function run_ex1_c(; ceed_spec, dim, mesh_order, sol_order, num_qpts, prob_size)
    ncompx = dim
    prob_size < 0 && (prob_size = 256*1024)

    gallery = false

    ceed = Ref{C.Ceed}()
    C.CeedInit(ceed_spec, ceed)

    mesh_basis = Ref{C.CeedBasis}()
    sol_basis = Ref{C.CeedBasis}()
    C.CeedBasisCreateTensorH1Lagrange(
        ceed[],
        dim,
        ncompx,
        mesh_order + 1,
        num_qpts,
        C.CEED_GAUSS,
        mesh_basis,
    )
    C.CeedBasisCreateTensorH1Lagrange(
        ceed[],
        dim,
        1,
        sol_order + 1,
        num_qpts,
        C.CEED_GAUSS,
        sol_basis,
    )

    # Determine the mesh size based on the given approximate problem size.
    nxyz = get_cartesian_mesh_size_c(dim, sol_order, prob_size)
    println("Mesh size: ", nxyz)

    # Build CeedElemRestriction objects describing the mesh and solution discrete
    # representations.
    mesh_size, mesh_rstr =
        build_cartesian_restriction_c(ceed, dim, nxyz, mesh_order, ncompx, num_qpts)
    sol_size, sol_rstr, sol_rstr_i = build_cartesian_restriction_c(
        ceed,
        dim,
        nxyz,
        sol_order,
        1,
        num_qpts,
        form_strided=true,
    )
    println("Number of mesh nodes     : ", div(mesh_size, dim))
    println("Number of solution nodes : ", sol_size)

    # Create a C.CeedVector with the mesh coordinates.
    mesh_coords = Ref{C.CeedVector}()
    C.CeedVectorCreate(ceed[], mesh_size, mesh_coords)
    set_cartesian_mesh_coords_c(dim, nxyz, mesh_order, mesh_coords)
    # Apply a transformation to the mesh.
    exact_vol = transform_mesh_coords_c(dim, mesh_size, mesh_coords)

    # Create the Q-function that builds the mass operator (i.e. computes its
    # quadrature data) and set its context data.
    build_qfunc = Ref{C.CeedQFunction}()

    build_ctx = BuildContextC(dim, dim)
    qf_ctx = Ref{C.CeedQFunctionContext}()
    C.CeedQFunctionContextCreate(ceed[], qf_ctx)
    C.CeedQFunctionContextSetData(
        qf_ctx[],
        C.CEED_MEM_HOST,
        C.CEED_USE_POINTER,
        sizeof(build_ctx),
        pointer_from_objref(build_ctx),
    )

    if !gallery
        qf_build_mass = @cfunction(
            f_build_mass_c,
            C.CeedInt,
            (Ptr{Cvoid}, C.CeedInt, Ptr{Ptr{C.CeedScalar}}, Ptr{Ptr{C.CeedScalar}})
        )
        # This creates the QFunction directly.
        C.CeedQFunctionCreateInterior(ceed[], 1, qf_build_mass, "julia", build_qfunc)
        C.CeedQFunctionAddInput(build_qfunc[], "dx", ncompx*dim, C.CEED_EVAL_GRAD)
        C.CeedQFunctionAddInput(build_qfunc[], "weights", 1, C.CEED_EVAL_WEIGHT)
        C.CeedQFunctionAddOutput(build_qfunc[], "qdata", 1, C.CEED_EVAL_NONE)
        C.CeedQFunctionSetContext(build_qfunc[], qf_ctx[])
    else
        # This creates the QFunction via the gallery.
        name = "Mass$(dim)DBuild"
        C.CeedQFunctionCreateInteriorByName(ceed[], name, build_qfunc)
    end

    # Create the operator that builds the quadrature data for the mass operator.
    build_oper = Ref{C.CeedOperator}()
    C.CeedOperatorCreate(
        ceed[],
        build_qfunc[],
        C.CEED_QFUNCTION_NONE[],
        C.CEED_QFUNCTION_NONE[],
        build_oper,
    )
    C.CeedOperatorSetField(
        build_oper[],
        "dx",
        mesh_rstr[],
        mesh_basis[],
        C.CEED_VECTOR_ACTIVE[],
    )
    C.CeedOperatorSetField(
        build_oper[],
        "weights",
        C.CEED_ELEMRESTRICTION_NONE[],
        mesh_basis[],
        C.CEED_VECTOR_NONE[],
    )
    C.CeedOperatorSetField(
        build_oper[],
        "qdata",
        sol_rstr_i[],
        C.CEED_BASIS_NONE[],
        C.CEED_VECTOR_ACTIVE[],
    )

    # Compute the quadrature data for the mass operator.
    qdata = Ref{C.CeedVector}()
    elem_qpts = num_qpts^dim
    num_elem = prod(nxyz)
    C.CeedVectorCreate(ceed[], num_elem*elem_qpts, qdata)

    print("Computing the quadrature data for the mass operator ...")
    flush(stdout)
    GC.@preserve build_ctx C.CeedOperatorApply(
        build_oper[],
        mesh_coords[],
        qdata[],
        C.CEED_REQUEST_IMMEDIATE[],
    )
    println(" done.")

    # Create the Q-function that defines the action of the mass operator.
    apply_qfunc = Ref{C.CeedQFunction}()
    if !gallery
        qf_apply_mass = @cfunction(
            f_apply_mass_c,
            C.CeedInt,
            (Ptr{Cvoid}, C.CeedInt, Ptr{Ptr{C.CeedScalar}}, Ptr{Ptr{C.CeedScalar}})
        )
        # This creates the QFunction directly.
        C.CeedQFunctionCreateInterior(ceed[], 1, qf_apply_mass, "julia", apply_qfunc)
        C.CeedQFunctionAddInput(apply_qfunc[], "u", 1, C.CEED_EVAL_INTERP)
        C.CeedQFunctionAddInput(apply_qfunc[], "qdata", 1, C.CEED_EVAL_NONE)
        C.CeedQFunctionAddOutput(apply_qfunc[], "v", 1, C.CEED_EVAL_INTERP)
    else
        # This creates the QFunction via the gallery.
        C.CeedQFunctionCreateInteriorByName(ceed[], "MassApply", apply_qfunc)
    end

    # Create the mass operator.
    oper = Ref{C.CeedOperator}()
    C.CeedOperatorCreate(
        ceed[],
        apply_qfunc[],
        C.CEED_QFUNCTION_NONE[],
        C.CEED_QFUNCTION_NONE[],
        oper,
    )
    C.CeedOperatorSetField(oper[], "u", sol_rstr[], sol_basis[], C.CEED_VECTOR_ACTIVE[])
    C.CeedOperatorSetField(oper[], "qdata", sol_rstr_i[], C.CEED_BASIS_NONE[], qdata[])
    C.CeedOperatorSetField(oper[], "v", sol_rstr[], sol_basis[], C.CEED_VECTOR_ACTIVE[])

    # Compute the mesh volume using the mass operator: vol = 1^T \cdot M \cdot 1
    print("Computing the mesh volume using the formula: vol = 1^T.M.1 ...")
    flush(stdout)
    # Create auxiliary solution-size vectors.
    u = Ref{C.CeedVector}()
    v = Ref{C.CeedVector}()
    C.CeedVectorCreate(ceed[], sol_size, u)
    C.CeedVectorCreate(ceed[], sol_size, v)

    # Initialize 'u' with ones.
    C.CeedVectorSetValue(u[], 1.0)

    # Apply the mass operator: 'u' -> 'v'.
    C.CeedOperatorApply(oper[], u[], v[], C.CEED_REQUEST_IMMEDIATE[])

    # Compute and print the sum of the entries of 'v' giving the mesh volume.
    v_host_ref = Ref{Ptr{C.CeedScalar}}()
    C.CeedVectorGetArrayRead(v[], C.CEED_MEM_HOST, v_host_ref)
    v_host = unsafe_wrap(Array, v_host_ref[], sol_size)
    vol = sum(v_host)
    C.CeedVectorRestoreArrayRead(v[], v_host_ref)

    println(" done.")
    @printf("Exact mesh volume    : % .14g\n", exact_vol)
    @printf("Computed mesh volume : % .14g\n", vol)
    @printf("Volume error         : % .14g\n", vol - exact_vol)

    # Free dynamically allocated memory.
    C.CeedVectorDestroy(u)
    C.CeedVectorDestroy(v)
    C.CeedVectorDestroy(qdata)
    C.CeedVectorDestroy(mesh_coords)
    C.CeedOperatorDestroy(oper)
    C.CeedQFunctionDestroy(apply_qfunc)
    C.CeedOperatorDestroy(build_oper)
    C.CeedQFunctionDestroy(build_qfunc)
    C.CeedElemRestrictionDestroy(sol_rstr)
    C.CeedElemRestrictionDestroy(mesh_rstr)
    C.CeedElemRestrictionDestroy(sol_rstr_i)
    C.CeedBasisDestroy(sol_basis)
    C.CeedBasisDestroy(mesh_basis)
    C.CeedDestroy(ceed)
end

run_ex1_c(
    ceed_spec="/cpu/self",
    dim=3,
    mesh_order=4,
    sol_order=4,
    num_qpts=4 + 2,
    prob_size=-1,
)
