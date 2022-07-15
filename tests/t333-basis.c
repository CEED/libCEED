/// @file
/// Test that Hale-Trefethen transform  works
/// \test Test that Hale-Trefethen transform works
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
    Ceed ceed;
    CeedInt Q = 16;
    CeedScalar rho;
    CeedScalar *g, *g_prime;

    CeedHaleTrefethenStripMap();


    CeedDestroy(&ceed);
    return 0;
}