cmake_minimum_required( VERSION 2.8 )
project( cmpro )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

set(DLIB_NO_GUI_SUPPORT 0)

# 寻找OpenCV库
find_package( OpenCV 3 REQUIRED )
find_package( dlib REQUIRED )


# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories( ${dlib_INCLUDE_DIRS} )

add_executable( cmpro cmpro.cpp )
add_executable( cmadd cmadd.cpp )

# 链接OpenCV库
target_link_libraries( cmpro ${OpenCV_LIBS} )
target_link_libraries( cmpro dlib::dlib )
target_link_libraries( cmadd ${OpenCV_LIBS} )
target_link_libraries( cmadd dlib::dlib )