using Test, LibCEED, LinearAlgebra, StaticArrays

function checkoutput(str, fname)
    if str != getoutput(fname)
        write(fname, str)
        return false
    end
    return true
end

@testset "LibCEED Development Tests" begin
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

        v = rand(CeedScalar, n)
        v1 = CeedVector(c, v)
        v2 = CeedVector(c, n)

        comp_op = create_composite_operator(c, [op])
        apply!(comp_op, v1, v2)
        @test @witharray_read(a1 = v1, @witharray_read(a2 = v2, a1 == a2))
    end
end
