cmake_minimum_required(VERSION 3.20)
project(libCEED LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if (CEED_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

file(GLOB_RECURSE interface_c_sources ${PROJECT_SOURCE_DIR}/interface/*.c)
file(GLOB_RECURSE interface_cpp_sources ${PROJECT_SOURCE_DIR}/interface/*.cpp)

if (NOT CEED_ENABLE_HIP)
  list(REMOVE_ITEM interface_c_sources ${PROJECT_SOURCE_DIR}/interface/ceed-hip.c)
endif()

if (NOT CEED_ENABLE_CUDA)
  list(REMOVE_ITEM interface_c_sources ${PROJECT_SOURCE_DIR}/interface/ceed-cuda.c)
endif()

file(GLOB_RECURSE gallery_sources ${PROJECT_SOURCE_DIR}/gallery/*.c)

add_library(CEED ${interface_c_sources} ${interface_cpp_sources} ${gallery_sources})
add_subdirectory(backends)
target_link_libraries(CEED PUBLIC m)
target_include_directories(CEED PUBLIC include)
target_compile_definitions(CEED PUBLIC CEED_JIT_SOURCE_ROOT_DEFAULT="${PROJECT_SOURCE_DIR}/include")

if (CEED_ENABLE_CUDA)
  target_link_libraries(CEED PUBLIC CUDA::cublas CUDA::cudart CUDA::nvrtc)
endif()

# TODO: which flags should be enabled for a given platform?
target_compile_options(CEED PRIVATE "-march=native" "-ffp-contract=fast")

if (CEED_ENABLE_TESTING)
  add_subdirectory(tests)
endif()

# TODO: figure out what to put for private_libs
set(CEED_PRIVATE_LIBS "??")

# generate pkgconfig file from template
configure_file(cmake/ceed.pc.in ceed.pc)

# TODO: install interface
