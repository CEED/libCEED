cmake_minimum_required(VERSION 3.20)
project(libCEED_consumer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(FetchContent)

# libraries that want to use libCEED need only append this in their CMakeLists.txt
FetchContent_Declare(
  libCEED
  GIT_REPOSITORY git@github.com:samuelpmish/libCEED.git
  GIT_TAG main
)
FetchContent_MakeAvailable(libCEED)

add_executable(my_executable main.c)

# and then link against the CEED target
target_link_libraries(my_executable PUBLIC CEED)
