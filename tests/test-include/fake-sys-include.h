#define FAKE_SYS_SCALE_ONE 1

// Note - files included this way cannot transitively include any files CUDA/ROCm won't compile
// These are bad and need to be guarded
#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#include <stddef.h>
#endif

// These are ok
// Note - ceed/types.h should be used over ceed.h
//        ceed.h is replaced with ceed/types.h during JiT
#include <ceed.h>
#include <ceed/types.h>
