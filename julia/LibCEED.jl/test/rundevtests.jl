using Test, LibCEED, LinearAlgebra, StaticArrays

function checkoutput(str, fname)
    if str != getoutput(fname)
        write(fname, str)
        return false
    end
    return true
end

@testset "LibCEED Development Tests" begin end
