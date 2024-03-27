target_sources(CEED PUBLIC ${PROJECT_SOURCE_DIR}/backends/ceed-backend-weak.c)

file(GLOB_RECURSE ref_backend_sources ${PROJECT_SOURCE_DIR}/backends/ref/*)
target_sources(CEED PUBLIC ${ref_backend_sources})

file(GLOB_RECURSE opt_backend_sources ${PROJECT_SOURCE_DIR}/backends/opt/*)
target_sources(CEED PUBLIC ${opt_backend_sources})

file(GLOB_RECURSE blocked_backend_sources ${PROJECT_SOURCE_DIR}/backends/blocked/*)
target_sources(CEED PUBLIC ${blocked_backend_sources})

if (CEED_ENABLE_AVX)
  file(GLOB_RECURSE avx_backend_sources ${PROJECT_SOURCE_DIR}/backends/avx/*)
  target_sources(CEED PUBLIC ${avx_backend_sources})
endif()

if (CEED_ENABLE_HIP)
  file(GLOB_RECURSE hip_backend_sources 
    ${PROJECT_SOURCE_DIR}/backends/hip/*
    ${PROJECT_SOURCE_DIR}/backends/hip-gen/*
    ${PROJECT_SOURCE_DIR}/backends/hip-ref/*
    ${PROJECT_SOURCE_DIR}/backends/hip-shared/*
  )
  target_sources(CEED PUBLIC ${hip_backend_sources})
endif()

if (CEED_ENABLE_CUDA)
  file(GLOB_RECURSE cuda_backend_sources 
    ${PROJECT_SOURCE_DIR}/backends/cuda/*
    ${PROJECT_SOURCE_DIR}/backends/cuda-gen/*
    ${PROJECT_SOURCE_DIR}/backends/cuda-ref/*
    ${PROJECT_SOURCE_DIR}/backends/cuda-shared/*
  )
  target_sources(CEED PUBLIC ${cuda_backend_sources})
endif()
