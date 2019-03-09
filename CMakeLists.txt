cmake_minimum_required( VERSION 2.8 )
project( cmpro )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

set(DLIB_NO_GUI_SUPPORT 0)

# 寻找OpenCV库
find_package( OpenCV 3 REQUIRED )
find_package( dlib REQUIRED )
find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
# 添加头文件

include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories( ${SFML_INCLUDE_DIRS} )

add_executable( cmpro test1.cc detection_procession.cc detection_procession.h shape_processing.cc shape_processing.h showing_window.cc showing_window.h)


# 链接OpenCV库
target_link_libraries( cmpro ${OpenCV_LIBS} )
target_link_libraries( cmpro dlib::dlib )
target_link_libraries( cmpro sfml-graphics sfml-audio)
