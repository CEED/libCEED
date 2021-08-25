using Test, LibCEED, LinearAlgebra, StaticArrays

showstr(x) = sprint(show, MIME("text/plain"), x)
summarystr(x) = sprint(summary, x)
getoutput(fname) =
    chomp(read(joinpath(@__DIR__, "output", string(CeedScalar), fname), String))

function checkoutput(str, fname)
    if str != getoutput(fname)
        write(fname, str)
        return false
    end
    return true
end

mutable struct CtxData
    io::IOBuffer
    x::Vector{Float64}
end

const run_dev_tests = !isrelease() || ("--run-dev-tests" in ARGS)

if run_dev_tests
    include("rundevtests.jl")
end

if LibCEED.minimum_libceed_version > ceedversion() && !run_dev_tests
    @warn "Skipping tests because of incompatible libCEED versions."
else
    @testset "LibCEED Release Tests" begin
        @testset "LibCEED" begin
            @test ceedversion() isa VersionNumber
            @test isrelease() isa Bool
            @test isfile(get_libceed_path())
        end

        @testset "Ceed" begin
            res = "/cpu/self/ref/serial"
            c = Ceed(res)
            @test isdeterministic(c)
            @test getresource(c) == res
            @test !iscuda(c)
            @test get_preferred_memtype(c) == MEM_HOST
            @test_throws LibCEED.CeedError create_interior_qfunction(c, "")
            @test showstr(c) == """
                Ceed
                  Ceed Resource: $res
                  Preferred MemType: host"""
        end

        @testset "Context" begin
            c = Ceed()
            data = zeros(CeedScalar, 3)
            ctx = Context(c, data)
            @test showstr(ctx) == """
                CeedQFunctionContext
                  Context Data Size: $(sizeof(data))"""
            @test_throws Exception set_data!(ctx, MEM_HOST, OWN_POINTER, data)
        end

        @testset "CeedVector" begin
            n = 10
            c = Ceed()
            v = CeedVector(c, n)
            @test size(v) == (n,)
            @test length(v) == n
            @test axes(v) == (1:n,)
            @test ndims(v) == 1
            @test ndims(CeedVector) == 1

            v[] = 0.0
            @test @witharray(a = v, all(a .== 0.0))

            v1 = rand(CeedScalar, n)
            v2 = CeedVector(c, v1)
            @test @witharray_read(a = v2, mtype = MEM_HOST, a == v1)
            @test Vector(v2) == v1
            v[] = v1
            for p ∈ [1, 2, Inf]
                @test norm(v, p) ≈ norm(v1, p)
            end
            @test_throws Exception norm(v, 3)
            @test witharray_read(sum, v) == sum(v1)
            reciprocal!(v)
            @test @witharray(a = v, mtype = MEM_HOST, all(a .== CeedScalar(1.0)./v1))

            witharray(x -> x .= 1.0, v)
            @test @witharray(a = v, all(a .== 1.0))

            @test summarystr(v) == "$n-element CeedVector"
            @test sprint(show, v) == @witharray_read(a = v, sprint(show, a))
            io = IOBuffer()
            summary(io, v)
            println(io, ":")
            @witharray_read(a = v, Base.print_array(io, a))
            s1 = String(take!(io))
            @test showstr(v) == s1

            setarray!(v, MEM_HOST, USE_POINTER, v1)
            syncarray!(v, MEM_HOST)
            @test @witharray_read(a = v, a == v1)
            p = takearray!(v, MEM_HOST)
            @test p == pointer(v1)

            m = rand(CeedScalar, 10, 10)
            vm = CeedVector(c, vec(m))
            @test @witharray_read(a = vm, size = size(m), a == m)

            @test CeedVectorActive()[] == LibCEED.C.CEED_VECTOR_ACTIVE[]
            @test CeedVectorNone()[] == LibCEED.C.CEED_VECTOR_NONE[]

            w1 = rand(CeedScalar, n)
            w2 = rand(CeedScalar, n)
            w3 = rand(CeedScalar, n)

            cv1 = CeedVector(c, w1)
            cv2 = CeedVector(c, w2)
            cv3 = CeedVector(c, w3)

            alpha = rand(CeedScalar)

            scale!(cv1, alpha)
            w1 .*= alpha
            @test @witharray_read(a = cv1, a == w1)

            pointwisemult!(cv1, cv2, cv3)
            w1 .= w2.*w3
            @test @witharray_read(a = cv1, a == w1)

            axpy!(alpha, cv2, cv1)
            axpy!(alpha, w2, w1)
            @test @witharray_read(a = cv1, a ≈ w1)
        end

        @testset "Basis" begin
            c = Ceed()
            dim = 3
            ncomp = 1
            p = 4
            q = 6
            b1 = create_tensor_h1_lagrange_basis(c, dim, ncomp, p, q, GAUSS_LOBATTO)

            @test checkoutput(showstr(b1), "b1.out")
            @test getdimension(b1) == 3
            @test gettopology(b1) == HEX
            @test getnumcomponents(b1) == ncomp
            @test getnumnodes(b1) == p^dim
            @test getnumnodes1d(b1) == p
            @test getnumqpts(b1) == q^dim
            @test getnumqpts1d(b1) == q

            q1d, w1d = lobatto_quadrature(3, AbscissaAndWeights)
            @test q1d ≈ CeedScalar[-1.0, 0.0, 1.0]
            @test w1d ≈ CeedScalar[1/3, 4/3, 1/3]

            q1d, w1d = gauss_quadrature(3)
            @test q1d ≈ CeedScalar[-sqrt(3/5), 0.0, sqrt(3/5)]
            @test w1d ≈ CeedScalar[5/9, 8/9, 5/9]

            b1d = CeedScalar[1.0 0.0; 0.5 0.5; 0.0 1.0]
            d1d = CeedScalar[-0.5 0.5; -0.5 0.5; -0.5 0.5]
            q1d = CeedScalar[-1.0, 0.0, 1.0]
            w1d = CeedScalar[1/3, 4/3, 1/3]
            q, p = size(b1d)
            d2d = zeros(CeedScalar, 2, q*q, p*p)
            d2d[1, :, :] = kron(b1d, d1d)
            d2d[2, :, :] = kron(d1d, b1d)

            dim2 = 2
            b2 = create_tensor_h1_basis(c, dim2, 1, p, q, b1d, d1d, q1d, w1d)
            @test getinterp(b2) == kron(b1d, b1d)
            @test getinterp1d(b2) == b1d
            @test getgrad(b2) == d2d
            @test getgrad1d(b2) == d1d
            @test checkoutput(showstr(b2), "b2.out")

            b3 = create_h1_basis(c, LINE, 1, p, q, b1d, reshape(d1d, 1, q, p), q1d, w1d)
            @test getqref(b3) == q1d
            @test getqweights(b3) == w1d
            @test checkoutput(showstr(b3), "b3.out")

            v = rand(CeedScalar, 2)
            vq = apply(b3, v)
            vd = apply(b3, v; emode=EVAL_GRAD)
            @test vq ≈ b1d*v
            @test vd ≈ d1d*v

            @test BasisCollocated()[] == LibCEED.C.CEED_BASIS_COLLOCATED[]
        end

        @testset "Request" begin
            @test RequestImmediate()[] == LibCEED.C.CEED_REQUEST_IMMEDIATE[]
            @test RequestOrdered()[] == LibCEED.C.CEED_REQUEST_ORDERED[]
        end

        @testset "Misc" begin
            for dim = 1:3
                D = CeedDim(dim)
                J = rand(CeedScalar, dim, dim)
                @test det(J, D) ≈ det(J)
                J = J + J' # make symmetric
                @test setvoigt(SMatrix{dim,dim}(J)) == setvoigt(J, D)
                @test getvoigt(setvoigt(J, D)) == J
                V = zeros(CeedScalar, dim*(dim + 1)÷2)
                setvoigt!(V, J, D)
                @test V == setvoigt(J, D)
                J2 = zeros(CeedScalar, dim, dim)
                getvoigt!(J2, V, D)
                @test J2 == J
            end
        end

        @testset "QFunction" begin
            c = Ceed()

            id = create_identity_qfunction(c, 1, EVAL_INTERP, EVAL_INTERP)
            Q = 10
            v = rand(CeedScalar, Q)
            v1 = CeedVector(c, v)
            v2 = CeedVector(c, Q)
            apply!(id, Q, [v1], [v2])
            @test @witharray(a = v2, a == v)

            @test showstr(create_interior_qfunction(c, "Poisson3DApply")) == """
                Gallery CeedQFunction Poisson3DApply
                  2 Input Fields:
                    Input Field [0]:
                      Name: "du"
                      Size: 3
                      EvalMode: "gradient"
                    Input Field [1]:
                      Name: "qdata"
                      Size: 6
                      EvalMode: "none"
                  1 Output Field:
                    Output Field [0]:
                      Name: "dv"
                      Size: 3
                      EvalMode: "gradient\""""

            @interior_qf id2 = (c, (a, :in, EVAL_INTERP), (b, :out, EVAL_INTERP), b.=a)
            v2[] = 0.0
            apply!(id2, Q, [v1], [v2])
            @test @witharray(a = v2, a == v)

            ctxdata = CtxData(IOBuffer(), rand(CeedScalar, 3))
            ctx = Context(c, ctxdata)
            dim = 3
            @interior_qf qf = (
                c,
                dim=dim,
                ctxdata::CtxData,
                (a, :in, EVAL_GRAD, dim),
                (b, :in, EVAL_NONE),
                (c, :out, EVAL_INTERP),
                begin
                    c[] = b*sum(a)
                    show(ctxdata.io, MIME("text/plain"), ctxdata.x)
                end,
            )
            set_context!(qf, ctx)
            in_sz, out_sz = LibCEED.get_field_sizes(qf)
            @test in_sz == [dim, 1]
            @test out_sz == [1]
            v1 = rand(CeedScalar, dim)
            v2 = rand(CeedScalar, 1)
            cv1 = CeedVector(c, v1)
            cv2 = CeedVector(c, v2)
            cv3 = CeedVector(c, 1)
            apply!(qf, 1, [cv1, cv2], [cv3])
            @test String(take!(ctxdata.io)) == showstr(ctxdata.x)
            @test @witharray_read(v3 = cv3, v3[1] == v2[1]*sum(v1))

            @test QFunctionNone()[] == LibCEED.C.CEED_QFUNCTION_NONE[]
        end

        @testset "Operator" begin
            c = Ceed()
            @interior_qf id = (
                c,
                (input, :in, EVAL_INTERP),
                (output, :out, EVAL_INTERP),
                begin
                    output[] = input
                end,
            )
            b = create_tensor_h1_lagrange_basis(c, 3, 1, 3, 3, GAUSS_LOBATTO)
            n = getnumnodes(b)
            offsets = Vector{CeedInt}(0:n-1)
            r = create_elem_restriction(c, 1, n, 1, 1, n, offsets)
            op = Operator(
                c;
                qf=id,
                fields=[
                    (:input, r, b, CeedVectorActive()),
                    (:output, r, b, CeedVectorActive()),
                ],
            )
            @test showstr(op) == """
                CeedOperator
                  2 Fields
                  1 Input Field:
                    Input Field [0]:
                      Name: "input"
                      Active vector
                  1 Output Field:
                    Output Field [0]:
                      Name: "output"
                      Active vector"""

            v = rand(CeedScalar, n)
            v1 = CeedVector(c, v)
            v2 = CeedVector(c, n)
            apply!(op, v1, v2)
            @test @witharray_read(a1 = v1, @witharray_read(a2 = v2, a1 == a2))
            apply_add!(op, v1, v2)
            @test @witharray_read(a1 = v1, @witharray_read(a2 = v2, a1 + a1 == a2))

            diag_vector = create_lvector(r)
            LibCEED.assemble_diagonal!(op, diag_vector)
            @test @witharray_read(a = diag_vector, a == ones(n))
            # TODO: change this test after bug-fix in libCEED
            diag_vector[] = 0.0
            LibCEED.assemble_add_diagonal!(op, diag_vector)
            @test @witharray(a = diag_vector, a == fill(1.0, n))

            comp_op = create_composite_operator(c, [op])
            apply!(comp_op, v1, v2)
            @test @witharray_read(a1 = v1, @witharray_read(a2 = v2, a1 == a2))
        end

        @testset "ElemRestriction" begin
            c = Ceed()
            n = 10
            offsets = Vector{CeedInt}([0:n-1; n-1:2*n-2])
            lsize = 2*n - 1
            r = create_elem_restriction(c, 2, n, 1, lsize, lsize, offsets)
            @test getcompstride(r) == lsize
            @test getnumelements(r) == 2
            @test getelementsize(r) == n
            @test getlvectorsize(r) == lsize
            @test getnumcomponents(r) == 1
            @test length(create_lvector(r)) == lsize
            @test length(create_evector(r)) == 2*n
            lv, ev = create_vectors(r)
            @test length(lv) == lsize
            @test length(ev) == 2*n
            mult = getmultiplicity(r)
            mult2 = ones(lsize)
            mult2[n] = 2
            @test mult == mult2
            rand_lv = rand(CeedScalar, lsize)
            rand_ev = [rand_lv[1:n]; rand_lv[n:end]]
            @test apply(r, rand_lv) == rand_ev
            @test apply(r, rand_ev; tmode=TRANSPOSE) == rand_lv.*mult
            @test showstr(r) == string(
                "CeedElemRestriction from (19, 1) to 2 elements ",
                "with 10 nodes each and component stride 19",
            )

            strides = CeedInt[1, n, n]
            rs = create_elem_restriction_strided(c, 1, n, 1, n, strides)
            @test showstr(rs) == string(
                "CeedElemRestriction from (10, 1) to 1 elements ",
                "with 10 nodes each and strides [1, $n, $n]",
            )

            @test ElemRestrictionNone()[] == LibCEED.C.CEED_ELEMRESTRICTION_NONE[]
        end
    end
end
