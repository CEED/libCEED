CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

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

FILE(GLOB SOURCE_FILES "*.cc")

FOREACH ( source_file ${SOURCE_FILES} )
  GET_FILENAME_COMPONENT(file_name ${source_file} NAME)
  STRING( REPLACE ".cc" "" exec ${file_name} )

  DEAL_II_INITIALIZE_CACHED_VARIABLES()
  PROJECT(${exec})

  DEAL_II_INITIALIZE_CACHED_VARIABLES()

  ADD_EXECUTABLE(${exec} ${source_file})
  DEAL_II_SETUP_TARGET(${exec})

  TARGET_INCLUDE_DIRECTORIES(${exec} PUBLIC ${CEED_DIR}/include)
  TARGET_LINK_LIBRARIES(${exec} ${CEED_DIR}/lib/libceed.so)

ENDFOREACH ( source_file ${SOURCE_FILES} )
