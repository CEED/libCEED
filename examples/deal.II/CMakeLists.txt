CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT("bps")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_EXECUTABLE(bps bps.cc)
DEAL_II_SETUP_TARGET(bps)

TARGET_INCLUDE_DIRECTORIES(bps PUBLIC ${CEED_DIR}/include)
TARGET_LINK_LIBRARIES(bps ${CEED_DIR}/lib/libceed.so)
