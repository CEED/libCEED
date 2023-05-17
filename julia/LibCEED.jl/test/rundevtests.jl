using Test, LibCEED, LinearAlgebra, StaticArrays

include("buildmats.jl")

@testset "LibCEED Development Tests" begin
    @testset "Basis" begin
        c = Ceed()
        dim = 2
        ncomp = 1
        p1 = 4
        q1 = 4
        qref1 = Array{Float64}(undef, dim, q1)
        qweight1 = Array{Float64}(undef, q1)
        interp1, div1 = build_mats_hdiv(qref1, qweight1)
        b1 = create_hdiv_basis(c, QUAD, ncomp, p1, q1, interp1, div1, qref1, qweight1)

        u1 = ones(Float64, p1)
        v1 = apply(b1, u1)

        for i = 1:q1
            @test v1[i] ≈ -1.0
            @test v1[q1+i] ≈ 1.0
        end

        p2 = 3
        q2 = 4
        qref2 = Array{Float64}(undef, dim, q2)
        qweight2 = Array{Float64}(undef, q2)
        interp2, curl2 = build_mats_hcurl(qref2, qweight2)
        b2 = create_hcurl_basis(c, TRIANGLE, ncomp, p2, q2, interp2, curl2, qref2, qweight2)

        u2 = [1.0, 2.0, 1.0]
        v2 = apply(b2, u2)

        for i = 1:q2
            @test v2[i] ≈ 1.0
        end

        u2[1] = -1.0
        u2[2] = 1.0
        u2[3] = 2.0
        v2 = apply(b2, u2)

        for i = 1:q2
            @test v2[q2+i] ≈ 1.0
        end
    end
end
