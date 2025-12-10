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

function run_ex3(; ceed_spec, dim, mesh_order, sol_order, num_qpts, prob_size, gallery)
    # Main implementation goes here
    ncompx = dim
    prob_size < 0 && (prob_size = 256*1024)

    ceed = Ceed(ceed_spec)
    mesh_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, ncompx, mesh_order + 1, num_qpts, GAUSS)
    sol_basis =
        create_tensor_h1_lagrange_basis(ceed, dim, 1, sol_order + 1, num_qpts, GAUSS)

    nxyz = get_cartesian_mesh_size(dim, sol_order, prob_size)
    println("Mesh size:", nxyz)

    #Build CeedElemRestriction objects describing the mesh and solution discrete
    # mesh_rstr: for building (no qdata restriction needed)
    mesh_size, mesh_rstr, _ =
        build_cartesian_restriction(ceed, dim, nxyz, mesh_order, ncompx, num_qpts)

    # sol_rstr + sol_rstr_i: for solving
    sol_size, sol_rstr, sol_rstr_i = build_cartesian_restriction(
        ceed,
        dim,
        nxyz,
        sol_order,
        1,
        num_qpts,
        mode=RestrictionAndStrided,
    )

    # mesh_coords
    mesh_coords = CeedVector(ceed, mesh_size)
    set_cartesian_mesh_coords!(dim, nxyz, mesh_order, mesh_coords)
    exact_vol = transform_mesh_coords!(dim, mesh_size, mesh_coords)

    #Create the Q-function that builds the mass operator ( i.e it computes the quadrature data) and set its context data.
    num_q_comp = 1 + div(dim*(dim + 1), 2)

    @interior_qf build_qfunc = (
        ceed,
        dim=dim,
        (dx, :in, EVAL_GRAD, dim, dim),      # ← THIS LINE: dx input
        (weights, :in, EVAL_WEIGHT),         # ← weights input
        (qdata, :out, EVAL_NONE, num_q_comp), # ← qdata output
        begin
            # Compute determinant
            det_J = det(dx)

            # Store mass component
            qdata[1] = weights*det_J

            # Store diffusion components (J^T * J)
            idx = 2
            for i = 1:dim
                for j = i:dim
                    qdata[idx] = dx[:, i]'*dx[:, j]
                    idx += 1
                end
            end
        end,
    )

    #Create the operator that builds the quadrature data for the mass operator
    build_oper = Operator(
        ceed,
        qf=build_qfunc,
        fields=[
            (:dx, mesh_rstr, mesh_basis, CeedVectorActive()),
            (:weights, ElemRestrictionNone(), mesh_basis, CeedVectorNone()),
            (:qdata, sol_rstr_i, BasisNone(), CeedVectorActive()),
        ],
    )

    # Apply to get qdata
    elem_qpts = num_qpts^dim
    num_elem = prod(nxyz)
    qdata = CeedVector(ceed, num_elem*elem_qpts*num_q_comp)
    print("Computing the quadrature data for the mass operator ...")
    flush(stdout)
    apply!(build_oper, mesh_coords, qdata)
    println(" done.")

    #Create QFunction for applying the mass+diffusion operator
    @interior_qf apply_qfunc = (
        ceed,
        dim=dim,
        (u, :in, EVAL_INTERP),
        (du, :in, EVAL_GRAD, dim),
        (qdata, :in, EVAL_NONE, num_q_comp),
        (v, :out, EVAL_INTERP),
        (dv, :out, EVAL_GRAD, dim),
        begin
            # Apply mass: v = qdata[1] * u
            v .= qdata[1].*u

            # Apply diffusion: dv = (qdata[2:end]) * du
            # The qdata contains the symmetric diffusion tensor (J^T*J)
            # dv_i = sum_j (J^T*J)_{i,j} * du_j

            # For efficiency, rebuild the matrix from stored components
            idx = 2
            for i = 1:dim
                dv_i = 0.0
                for j = 1:dim
                    # Reconstruct symmetric matrix element
                    if j >= i
                        mat_idx = idx + div((j - 1)*j, 2) + (i - 1)
                    else
                        mat_idx = idx + div((i - 1)*i, 2) + (j - 1)
                    end
                    dv_i += qdata[mat_idx]*du[j]
                end
                dv[i] = dv_i
            end
        end,
    )
    apply_oper = Operator(
        ceed,
        qf=apply_qfunc,
        fields=[
            (:u, sol_rstr, sol_basis, CeedVectorActive()),
            (:du, sol_rstr, sol_basis, CeedVectorActive()),
            (:qdata, sol_rstr_i, BasisNone(), qdata),
            (:v, sol_rstr, sol_basis, CeedVectorActive()),
            (:dv, sol_rstr, sol_basis, CeedVectorActive()),
        ],
    )

    # # Compute the mesh volume using the massdiff operator
    print("Computing the mesh volume using the formula: vol = 1^T * (M + K) * 1...")
    flush(stdout)

    u = CeedVector(ceed, sol_size)
    v = CeedVector(ceed, sol_size)
    u[] = 1.0

    # Apply operator
    apply!(apply_oper, u, v)

    # Compute volume
    vol = witharray_read(sum, v, MEM_HOST)

    @printf("Exact mesh volume    : % .14g\n", exact_vol)
    @printf("Computed mesh volume : % .14g\n", vol)
    @printf("Volume error         : % .14g\n", vol - exact_vol)
end

# Entry point
run_ex3(
    ceed_spec="/cpu/self",
    dim=2,
    mesh_order=2,
    sol_order=2,
    num_qpts=3,
    prob_size=-1,
    gallery=false,
)
