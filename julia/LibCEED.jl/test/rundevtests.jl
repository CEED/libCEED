using Test, LibCEED, LinearAlgebra, StaticArrays

@testset "LibCEED Development Tests" begin
    @testset "QFunction" begin
        c = Ceed()
        @test showstr(create_interior_qfunction(c, "Poisson3DApply")) == """
             Gallery CeedQFunction - Poisson3DApply
               2 input fields:
                 Input field 0:
                   Name: "du"
                   Size: 3
                   EvalMode: "gradient"
                 Input field 1:
                   Name: "qdata"
                   Size: 6
                   EvalMode: "none"
               1 output field:
                 Output field 0:
                   Name: "dv"
                   Size: 3
                   EvalMode: "gradient\""""
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

        @interior_qf id2 = (c, (a, :in, EVAL_INTERP), (b, :out, EVAL_INTERP), b .= a)
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
               1 elements with 27 quadrature points each
               2 fields
               1 input field:
                 Input field 0:
                   Name: "input"
                   Size: 1
                   EvalMode: interpolation
                   Active vector
               1 output field:
                 Output field 0:
                   Name: "output"
                   Size: 1
                   EvalMode: interpolation
                   Active vector"""
    end
end
