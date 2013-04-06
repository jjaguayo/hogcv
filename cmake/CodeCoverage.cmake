# - Enable Code Coverage
#
# 2012-01-31, Lars Bilke
#
# Edited by D.M. on 2012-08-18 to be OS X compatible. Originally distributed
# with Boost license.
#
# USAGE:
# 1. Copy this file into your cmake modules path
# 2. Add the following line to your CMakeLists.txt:
# INCLUDE(CodeCoverage)
#
# 3. Use the function SETUP_TARGET_FOR_COVERAGE to create a custom make target
# which runs your test executable and produces a lcov code coverage report.
#

# Check prereqs
FIND_PROGRAM( GCOV_PATH gcov )
FIND_PROGRAM( LCOV_PATH lcov )
FIND_PROGRAM( GENHTML_PATH genhtml )
FIND_PROGRAM( GCOVR_PATH gcovr PATHS ${CMAKE_SOURCE_DIR}/tests)

IF(NOT GCOV_PATH)
MESSAGE(FATAL_ERROR "gcov not found! Aborting...")
ENDIF() # NOT GCOV_PATH

# XCode is backwards. The XCode GCC does NOT support coverage checking, but
# Clang does.
#IF(NOT CMAKE_COMPILER_IS_GNUCXX)
#MESSAGE(FATAL_ERROR "Compiler is not GNU gcc! Aborting...")
#ENDIF() # NOT CMAKE_COMPILER_IS_GNUCXX

IF ( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" )
  MESSAGE( WARNING "Code coverage results with an optimized (non-Debug) build may be misleading" )
ENDIF() # NOT CMAKE_BUILD_TYPE STREQUAL "Debug"

# Setup compiler options
ADD_DEFINITIONS(-fprofile-arcs -ftest-coverage)
LINK_LIBRARIES(gcov)

# Param _targetname The name of new the custom make target
# Param _testrunner The name of the target which runs the tests
# Param _outputname lcov output is generated as _outputname.info
# HTML report is generated in _outputname/index.html
# Optional fourth parameter is passed as arguments to _testrunner
# Pass them in list form, e.g.: "-j;2" for -j 2
FUNCTION(SETUP_TARGET_FOR_COVERAGE _targetname _testrunner _outputname)

IF(NOT LCOV_PATH)
MESSAGE(FATAL_ERROR "lcov not found! Aborting...")
ENDIF() # NOT LCOV_PATH

IF(NOT GENHTML_PATH)
MESSAGE(FATAL_ERROR "genhtml not found! Aborting...")
ENDIF() # NOT GENHTML_PATH

# Setup target
ADD_CUSTOM_TARGET(${_targetname}

# Cleanup lcov
${LCOV_PATH} --directory . --zerocounters 

# Run tests
COMMAND ${_testrunner} ${ARGV3}

# Capturing lcov counters and generating report
COMMAND ${LCOV_PATH} --directory . --capture --output-file ${_outputname}.info
COMMAND ${LCOV_PATH} --remove ${_outputname}.info 'test/*' '/usr/*' 'gtest*' 'include/*' --output-file ${_outputname}.info.cleaned
COMMAND ${GENHTML_PATH} -o ${_outputname} ${_outputname}.info.cleaned --legend -s
COMMAND ${CMAKE_COMMAND} -E remove ${_outputname}.info ${_outputname}.info.cleaned

        # Changed by D.M. on Aug 18 from CMAKE_BINARY_DIR.
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  SET(OPENCMD open)
ELSEIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  SET(OPENCMD xdg-open)
ENDIF()

# Open the report.
ADD_CUSTOM_COMMAND(TARGET ${_targetname} POST_BUILD
  COMMAND ${OPENCMD} ./${_outputname}/index.html;
  COMMENT "Open ./test/${_outputname}/index.html in your browser to view the coverage report."
)

ENDFUNCTION() # SETUP_TARGET_FOR_COVERAGE
