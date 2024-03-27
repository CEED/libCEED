enable_testing()

file(GLOB_RECURSE all_tests ${PROJECT_SOURCE_DIR}/tests/*.c)

foreach(filename ${all_tests})
  get_filename_component(testname ${filename} NAME_WE)
  add_executable(${testname} ${filename})
  target_link_libraries(${testname} PUBLIC CEED)
  add_test(${testname} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${testname})		
endforeach(filename ${cpp_tests})