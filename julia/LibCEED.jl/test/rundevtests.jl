using Test, LibCEED, LinearAlgebra, StaticArrays

include("buildmats.jl")

function checkoutput(str, fname)
    if str != getoutput(fname)
        write(fname, str)
        return false
    end
    return true
end

@testset "LibCEED Development Tests" begin
    @testset "Basis" begin
        @test BasisNone()[] == LibCEED.C.CEED_BASIS_NONE[]

        c = Ceed()
        dim = 3
        ncomp = 1
        p = 4
        q = 6
        b1 = create_tensor_h1_lagrange_basis(c, dim, ncomp, p, q, GAUSS_LOBATTO)

        @test checkoutput(showstr(b1), "b1.out")

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
        @test checkoutput(showstr(b2), "b2.out")

        b3 = create_h1_basis(
            c,
            LINE,
            1,
            p,
            q,
            b1d,
            reshape(d1d, 1, q, p),
            reshape(q1d, 1, q),
            w1d,
        )
        @test checkoutput(showstr(b3), "b3.out")

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

    @testset "ElemRestriction" begin
        c = Ceed()
        nelem = 3
        elemsize = 2
        offsets = Array{CeedInt}(undef, elemsize, nelem)
        orients = Array{Bool}(undef, elemsize, nelem)
        for i = 1:nelem
            offsets[1, i] = i - 1
            offsets[2, i] = i
            # flip the dofs on element 1, 3, ...
            orients[1, i] = (i - 1)%2 > 0
            orients[2, i] = (i - 1)%2 > 0
        end
        r = create_elem_restriction_oriented(
            c,
            nelem,
            elemsize,
            1,
            1,
            nelem + 1,
            offsets,
            orients,
        )

        lv = Vector{CeedScalar}(undef, nelem + 1)
        for i = 1:nelem+1
            lv[i] = 10 + i - 1
        end

        ev = apply(r, lv)

        for i = 1:nelem
            for j = 1:elemsize
                k = j + elemsize*(i - 1)
                @test 10 + k÷2 == ev[k]*(-1)^((i - 1)%2)
            end
        end

        curlorients = Array{CeedInt8}(undef, 3*elemsize, nelem)
        for i = 1:nelem
            curlorients[1, i] = curlorients[6, i] = 0
            if (i - 1)%2 > 0
                # T = [0  -1]
                #     [-1  0]
                curlorients[2, i] = 0
                curlorients[3, i] = -1
                curlorients[4, i] = -1
                curlorients[5, i] = 0
            else
                # T = I
                curlorients[2, i] = 1
                curlorients[3, i] = 0
                curlorients[4, i] = 0
                curlorients[5, i] = 1
            end
        end
        r = create_elem_restriction_curl_oriented(
            c,
            nelem,
            elemsize,
            1,
            1,
            nelem + 1,
            offsets,
            curlorients,
        )

        ev = apply(r, lv)

        for i = 1:nelem
            for j = 1:elemsize
                k = j + elemsize*(i - 1)
                if (i - 1)%2 > 0
                    @test j == 2 || 10 + i == -ev[k]
                    @test j == 1 || 10 + i - 1 == -ev[k]
                else
                    @test 10 + k÷2 == ev[k]
                end
            end
        end
    end
end
