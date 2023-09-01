using LibCEED, Printf

include("common.jl")

function transform_mesh_coords!(dim, mesh_size, mesh_coords)
    @witharray coords = mesh_coords begin
        if dim == 1
            for i = 1:mesh_size
                # map [0,1] to [0,1] varying the mesh density
                coords[i] = 0.5 + 1.0/sqrt(3.0)*sin((2.0/3.0)*pi*(coords[i] - 0.5))
            end
            exact_volume = 1.0
        else
            num_nodes = mesh_size÷dim
            @inbounds @simd for i = 1:num_nodes
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
        return exact_volume
    end
end

function run_ex1(; ceed_spec, dim, mesh_order, sol_order, num_qpts, prob_size, gallery)
    ncompx = dim
    prob_size < 0 && (prob_size = 256*1024)

    ceed = Ceed(ceed_spec)
    mesh_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, ncompx, mesh_order + 1, num_qpts, GAUSS)
    sol_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, 1, sol_order + 1, num_qpts, GAUSS)

    # Determine the mesh size based on the given approximate problem size.
    nxyz = get_cartesian_mesh_size(dim, sol_order, prob_size)
    println("Mesh size: ", nxyz)

    # Build CeedElemRestriction objects describing the mesh and solution discrete
    # representations.
    mesh_size, mesh_rstr, _ =
        build_cartesian_restriction(ceed, dim, nxyz, mesh_order, ncompx, num_qpts)
    sol_size, sol_rstr, sol_rstr_i = build_cartesian_restriction(
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
    exact_vol = transform_mesh_coords!(dim, mesh_size, mesh_coords)

    # Create the Q-function that builds the mass operator (i.e. computes its
    # quadrature data) and set its context data.
    if !gallery
        @interior_qf build_qfunc = (
            ceed,
            dim=dim,
            (J, :in, EVAL_GRAD, dim, dim),
            (w, :in, EVAL_WEIGHT),
            (qdata, :out, EVAL_NONE),
            begin
                qdata .= w*det(J)
            end,
        )
    else
        build_qfunc = create_interior_qfunction(ceed, "Mass$(dim)DBuild")
    end

    # Create the operator that builds the quadrature data for the mass operator.
    build_oper = Operator(
        ceed,
        qf=build_qfunc,
        fields=[
            (gallery ? :dx : :J, mesh_rstr, mesh_basis, CeedVectorActive()),
            (gallery ? :weights : :w, ElemRestrictionNone(), mesh_basis, CeedVectorNone()),
            (:qdata, sol_rstr_i, BasisNone(), CeedVectorActive()),
        ],
    )

    # Compute the quadrature data for the mass operator.
    elem_qpts = num_qpts^dim
    num_elem = prod(nxyz)
    qdata = CeedVector(ceed, num_elem*elem_qpts)

    print("Computing the quadrature data for the mass operator ...")
    flush(stdout)
    apply!(build_oper, mesh_coords, qdata)
    println(" done.")

    # Create the Q-function that defines the action of the mass operator.
    if !gallery
        @interior_qf apply_qfunc = (
            ceed,
            (u, :in, EVAL_INTERP),
            (qdata, :in, EVAL_NONE),
            (v, :out, EVAL_INTERP),
            begin
                v .= qdata*u
            end,
        )
    else
        apply_qfunc = create_interior_qfunction(ceed, "MassApply")
    end

    # Create the mass operator.
    oper = Operator(
        ceed,
        qf=apply_qfunc,
        fields=[
            (:u, sol_rstr, sol_basis, CeedVectorActive()),
            (:qdata, sol_rstr_i, BasisNone(), qdata),
            (:v, sol_rstr, sol_basis, CeedVectorActive()),
        ],
    )

    # Compute the mesh volume using the mass operator: vol = 1^T \cdot M \cdot 1
    print("Computing the mesh volume using the formula: vol = 1^T.M.1 ...")
    flush(stdout)
    # Create auxiliary solution-size vectors.
    u = CeedVector(ceed, sol_size)
    v = CeedVector(ceed, sol_size)
    # Initialize 'u' with ones.
    u[] = 1.0
    # Apply the mass operator: 'u' -> 'v'.
    apply!(oper, u, v)
    # Compute and print the sum of the entries of 'v' giving the mesh volume.
    vol = witharray_read(sum, v, MEM_HOST)

    println(" done.")
    @printf("Exact mesh volume    : % .14g\n", exact_vol)
    @printf("Computed mesh volume : % .14g\n", vol)
    @printf("Volume error         : % .14g\n", vol - exact_vol)
end

run_ex1(
    ceed_spec="/cpu/self",
    dim=3,
    mesh_order=4,
    sol_order=4,
    num_qpts=4 + 2,
    prob_size=-1,
    gallery=false,
)
