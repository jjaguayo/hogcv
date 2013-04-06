# Build the main executable.

QT4_WRAP_UI(QTDLG_UIS_HDRS ${QTDLG_UIS})
QT4_WRAP_CPP(QT_HEADERS_MOC ${QT_HEADERS})

add_executable(${PROJECT_NAME} 
               ${QT_HEADERS_MOC}
               ${QTDLG_UIS_HDRS}
               ${MAIN_SOURCE} 
               ${PROJECT_SOURCES}
               ${QT_SOURCES}
)

# Compiler flags for the standard build should be placed here.
set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")

TARGET_LINK_LIBRARIES(hogcv
  ${OpenCV_LIBS}
  ${QT_LIBRARIES}
)

# Set the directory for "make install" to place the binary file.
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
