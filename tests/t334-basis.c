/// @file
/// Test that quadrature formula is transformed
/// \test Test that quadrature formula is transformed
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
    CeedInt Q = 5;
    CeedScalar w_gauss[Q], w_ht[Q];
    CeedScalar q_gauss[Q], q_ht[Q];

    CeedGaussQuadrature(Q, q_gauss, w_gauss);
    CeedGaussHaleTrefethenQuadrature(1.4, Q, q_ht, w_ht);
    for (CeedInt i=0; i<Q; i++)
        printf("%g (%g) -> %g (%g)\n", q_gauss[i], w_gauss[i], q_ht[i], w_ht[i]);

    CeedScalar w_sum = 0;
    for (CeedInt i=0; i<Q; i++) w_sum += w_ht[i];
    if (fabs(w_sum - 2) > 1e-12) printf("Unexpected sum of weights %g\n", w_sum);
    return 0;
}