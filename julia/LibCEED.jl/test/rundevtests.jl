using Test, LibCEED, LinearAlgebra, StaticArrays

@testset "LibCEED Development Tests" begin
    @test ceedversion() isa VersionNumber
    @test isrelease() == false

    @testset "CeedVector" begin
        n = 10
        c = Ceed()

        v1 = rand(CeedScalar, n)
        v2 = rand(CeedScalar, n)
        v3 = rand(CeedScalar, n)

        cv1 = CeedVector(c, v1)
        cv2 = CeedVector(c, v2)
        cv3 = CeedVector(c, v3)

        alpha = rand(CeedScalar)

        scale!(cv1, alpha)
        v1 .*= alpha
        @test @witharray_read(a = cv1, a == v1)

        pointwisemult!(cv1, cv2, cv3)
        v1 .= v2.*v3
        @test @witharray_read(a = cv1, a == v1)

        axpy!(alpha, cv2, cv1)
        axpy!(alpha, v2, v1)
        @test @witharray_read(a = cv1, a ≈ v1)
    end
end
