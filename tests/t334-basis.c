/// @file
/// Test that quadrature formula is transformed
/// \test Test that quadrature formula is transformed
#include <ceed.h>

int main(int argc, char **argv) {
    Ceed ceed;
    CeedInt Q = 16;
    CeedScalar *q_weight_1d;
    CeedBasis basis;

    CeedTransformQuadrature();

    CeedDestroy(&ceed)
    return 0;
}