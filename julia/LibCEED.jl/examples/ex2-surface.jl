using LibCEED, Printf, StaticArrays

include("common.jl")

function transform_mesh_coords!(dim, mesh_size, mesh_coords)
    @witharray coords = mesh_coords begin
        @inbounds @simd for i = 1:mesh_size
            # map [0,1] to [0,1] varying the mesh density
            coords[i] = 0.5 + 1.0/sqrt(3.0)*sin((2.0/3.0)*pi*(coords[i] - 0.5))
        end
    end
    exact_sa = (dim == 1 ? 2 : dim == 2 ? 4 : 6)
end

function run_ex2(; ceed_spec, dim, mesh_order, sol_order, num_qpts, prob_size, gallery)
    ncompx = dim
    prob_size < 0 && (prob_size = 256*1024)

    mesh_order = max(mesh_order, sol_order)
    sol_order = mesh_order

    ceed = Ceed(ceed_spec)
    mesh_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, ncompx, mesh_order + 1, num_qpts, GAUSS)
    sol_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, 1, sol_order + 1, num_qpts, GAUSS)

    nxyz = get_cartesian_mesh_size(dim, sol_order, prob_size)
    println("Mesh size: ", nxyz)

    # Build CeedElemRestriction objects describing the mesh and solution discrete
    # representations.
    mesh_size, mesh_restr, _ = build_cartesian_restriction(
        ceed,
        dim,
        nxyz,
        mesh_order,
        ncompx,
        num_qpts,
        mode=RestrictionOnly,
    )
    sol_size, _, qdata_restr_i = build_cartesian_restriction(
        ceed,
        dim,
        nxyz,
        sol_order,
        div(dim*(dim + 1), 2),
        num_qpts,
        mode=StridedOnly,
    )
    sol_size, sol_restr, sol_restr_i = build_cartesian_restriction(
        ceed,
        dim,
        nxyz,
        sol_order,
        1,
        num_qpts,
        mode=RestrictionAndStrided,
    )
    println("Number of mesh nodes     : ", div(mesh_size, dim))
    println("Number of solution nodes : ", sol_size)

    # Create a CeedVector with the mesh coordinates.
    mesh_coords = CeedVector(ceed, mesh_size)
    set_cartesian_mesh_coords!(dim, nxyz, mesh_order, mesh_coords)

    # Apply a transformation to the mesh.
    exact_sa = transform_mesh_coords!(dim, mesh_size, mesh_coords)

    # Create the Q-function that builds the diffusion operator (i.e. computes its
    # quadrature data) and set its context data.
    if !gallery
        @interior_qf build_qfunc = (
            ceed,
            dim=dim,
            (J, :in, EVAL_GRAD, dim, dim),
            (w, :in, EVAL_WEIGHT),
            (qdata, :out, EVAL_NONE, dim*(dim + 1)÷2),
            begin
                Jinv = inv(J)
                qdata .= setvoigt(w*det(J)*Jinv*Jinv')
            end,
        )
    else
        build_qfunc = create_interior_qfunction(ceed, "Poisson$(dim)DBuild")
    end

    # Create the operator that builds the quadrature data for the diffusion
    # operator.
    build_oper = Operator(
        ceed,
        qf=build_qfunc,
        fields=[
            (gallery ? :dx : :J, mesh_restr, mesh_basis, CeedVectorActive()),
            (gallery ? :weights : :w, ElemRestrictionNone(), mesh_basis, CeedVectorNone()),
            (:qdata, qdata_restr_i, BasisNone(), CeedVectorActive()),
        ],
    )

    # Compute the quadrature data for the diffusion operator.
    elem_qpts = num_qpts^dim
    num_elem = prod(nxyz)
    qdata = CeedVector(ceed, num_elem*elem_qpts*div(dim*(dim + 1), 2))
    print("Computing the quadrature data for the diffusion operator ...")
    flush(stdout)
    apply!(build_oper, mesh_coords, qdata)
    println(" done.")

    # Create the Q-function that defines the action of the diffusion operator.
    if !gallery
        @interior_qf apply_qfunc = (
            ceed,
            dim=dim,
            (du, :in, EVAL_GRAD, dim),
            (qdata, :in, EVAL_NONE, dim*(dim + 1)÷2),
            (dv, :out, EVAL_GRAD, dim),
            begin
                dXdxdXdxT = getvoigt(qdata)
                dv .= dXdxdXdxT*du
            end,
        )
    else
        apply_qfunc = create_interior_qfunction(ceed, "Poisson$(dim)DApply")
    end

    # Create the diffusion operator.
    oper = Operator(
        ceed,
        qf=apply_qfunc,
        fields=[
            (:du, sol_restr, sol_basis, CeedVectorActive()),
            (:qdata, qdata_restr_i, BasisNone(), qdata),
            (:dv, sol_restr, sol_basis, CeedVectorActive()),
        ],
    )

    # Compute the mesh surface area using the diff operator:
    #                                             sa = 1^T \cdot abs( K \cdot x).
    print("Computing the mesh surface area using the formula: sa = 1^T.|K.x| ...")
    flush(stdout)

    # Create auxiliary solution-size vectors.
    u = CeedVector(ceed, sol_size)
    v = CeedVector(ceed, sol_size)
    # Initialize 'u' with sum of coordinates, x+y+z.
    u[] = 0.0
    @witharray_read(
        x_host = mesh_coords,
        size = (mesh_size÷dim, dim),
        @witharray(u_host = u, size = (sol_size, 1), sum!(u_host, x_host))
    )

    # Apply the diffusion operator: 'u' -> 'v'.
    apply!(oper, u, v)
    sa = witharray_read(x -> sum(abs, x), v, MEM_HOST)

    println(" done.")
    @printf("Exact mesh surface area    : % .14g\n", exact_sa)
    @printf("Computed mesh surface area : % .14g\n", sa)
    @printf("Surface area error         : % .14g\n", sa - exact_sa)
end

run_ex2(
    ceed_spec="/cpu/self",
    dim=3,
    mesh_order=4,
    sol_order=4,
    num_qpts=6,
    prob_size=-1,
    gallery=false,
)
