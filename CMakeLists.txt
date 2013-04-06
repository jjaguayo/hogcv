cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

SET(PROJECT_NAME hogcv)
PROJECT(${PROJECT_NAME})

# File with the main function. Separated as we will not be writing
# test cases for it
SET(MAIN_SOURCE
    hogcv.cpp
)

# All project source files that we will be writing test cases for.
set(PROJECT_SOURCES
    HOGCVController.cpp
    Parameters.cpp
    svm.cpp
)

# Separating QT source files. 
set(QT_SOURCES
    MainWin.cpp
)

# All test source files
set(TEST_SOURCES
    ParametersTest.cpp
    HOGCVControllerTest.cpp
    SVMGetFunctionsTest.cpp
    SVMCheckParamsTest.cpp
    SVMFreeMemoryTest.cpp
    SVMPredictProbabilityTest.cpp
    SVMPredictTest.cpp
    SVMCrossValidationTest.cpp
    SVMSaveAndLoadModelTest.cpp
    SVMTrainingTest.cpp
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Qt4 REQUIRED)

INCLUDE_DIRECTORIES(
    SYSTEM
    ${CMAKE_CURRENT_SOURCE_DIR}/hdrs
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}
)

# This is the path where opencv2 is put on a Mac.
# Need to change this path for other OSs
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  INCLUDE_DIRECTORIES("/opt/local/include/opencv2")
ELSEIF (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  INCLUDE_DIRECTORIES("/usr/include/opencv2")
ENDIF()

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})

SET(QTDLG_UIS
    "${PROJECT_SOURCE_DIR}/ui/MainWin.ui"
)

SET(QT_HEADERS
  "${PROJECT_SOURCE_DIR}/hdrs/MainWin.hpp"
)

ADD_SUBDIRECTORY(gtest-1.6.0)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)

