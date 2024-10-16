#define FAKE_SYS_SCALE_ONE 1

// Note - files included this way cannot transitively include any files CUDA/ROCm won't compile
// These are bad
// #include <ceed.h>
// #include <math.h>
// #include <stddef.h>

// This is ok
#include <ceed/types.h>
