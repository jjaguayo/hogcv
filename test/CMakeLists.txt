# The test suite name is the project name with _test appended to it.
set(TEST_NAME ${PROJECT_NAME}_test)

# Append the project sources to the list of test sources, appending the
# "../src" relative path for each.
foreach(source ${PROJECT_SOURCES})
    list(APPEND TEST_SOURCES ../src/${source})
endforeach(source)

# Build the test executable.
add_executable(${TEST_NAME} ${TEST_SOURCES})
set(CMAKE_BUILD_TYPE Debug)

# gtest was not found with a find_package command and must be linked to
# explicitly.
target_link_libraries(${TEST_NAME} 
                      gtest_main 
                      gtest 
                      pthread 
                      gcov
                      ${QT_LIBRARIES} 
                      ${OpenCV_LIBS}
)

#target_link_libraries(${TEST_NAME} profile_rt)

# Add flags for debugging and for code coverage tests.
set_target_properties(${TEST_NAME} PROPERTIES COMPILE_FLAGS
    "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
set_target_properties(${TEST_NAME} PROPERTIES LINKER_FLAGS
    "-fprofile-arcs -ftest-coverage -lgcov")


add_custom_target(test
COMMAND ${TEST_NAME}
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Running tests.")

# We build a coverage make target using the external CodeCoverage.cmake
# file. The coverage test will skip files in the test subdirectory. This is
# because the gtest library's heavy use of macros makes the code coverage
# results for these files useless.
include(CodeCoverage)
setup_target_for_coverage(coverage ${TEST_NAME} coverage_results)

file(COPY ${PROJECT_SOURCE_DIR}/data/person_and_bike_001.bmp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
