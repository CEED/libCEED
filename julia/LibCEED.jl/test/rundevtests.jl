using Test, LibCEED, LinearAlgebra, StaticArrays

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
        @test showstr(op) == """
             CeedOperator
               1 elements with 27 quadrature points each
               2 Fields
               1 Input Field:
                 Input Field [0]:
                   Name: "input"
                   Active vector
               1 Output Field:
                 Output Field [0]:
                   Name: "output"
                   Active vector"""
    end
end
