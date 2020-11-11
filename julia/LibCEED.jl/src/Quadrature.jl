@doc raw"""
    gauss_quadrature(q)

Return the Gauss-Legendre quadrature rule with `q` points (integrates polynomials of degree
$2q-1$ exactly).

A tuple `(x,w)` is returned.
"""
function gauss_quadrature(q)
    x = zeros(CeedScalar, q)
    w = zeros(CeedScalar, q)
    C.CeedGaussQuadrature(q, x, w)
    x, w
end

struct QuadratureMode{T} end
const Abscissa = QuadratureMode{:Abscissa}()
const AbscissaAndWeights = QuadratureMode{:AbscissaAndWeights}()

@doc raw"""
    lobatto_quadrature(q, mode::Mode=Abscissa)

Return the Gauss-Lobatto quadrature rule with `q` points (integrates polynomials of degree
$2q-3$ exactly).

If `mode` is `AbscissaAndWeights`, then both the weights and abscissa are returned as a
tuple `(x,w)`.

Otherwise, (if `mode` is `Abscissa`), then only the abscissa `x` are returned.
"""
function lobatto_quadrature(q, mode::Mode=Abscissa) where {Mode}
    return_weights = (mode == AbscissaAndWeights)
    x = zeros(CeedScalar, q)
    w = (return_weights) ? zeros(CeedScalar, q) : C_NULL
    C.CeedLobattoQuadrature(q, x, w)
    return_weights ? (x, w) : x
end
