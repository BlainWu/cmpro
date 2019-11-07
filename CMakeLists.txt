cmake_minimum_required( VERSION 3.14 )
project( cmpro )
set( CMAKE_CXX_COMPILER "g++" )
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(DLIB_NO_GUI_SUPPORT 0)

find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )
find_package( Eigen3 REQUIRED NO_MODULE)


include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(cmpro_rate
        test/cmpro_rate.cpp
        include/cmpro/common_include.h
        include/cmpro/shape_processing.h src/shape_processing.cpp
        include/cmpro/rate_method.h  src/rate_method.cpp
        include/cmpro/configure.h src/configure.cpp
        include/cmpro/coordinate_convert.h src/coordinate_convert.cpp
        include/cmpro/sequence_convert.h src/sequence_convert.cpp
        )
add_executable(cmpro_dpl
        test/cmpro_dpl.cpp
        include/cmpro/common_include.h
        include/cmpro/shape_processing.h src/shape_processing.cpp
        include/cmpro/configure.h src/configure.cpp
        src/deep_process.cpp include/cmpro/deep_process.h
        include/cmpro/sequence_convert.h src/sequence_convert.cpp
        )

target_link_libraries( cmpro_rate
        Eigen3::Eigen
        ${OpenCV_LIBS}
        dlib::dlib
        )

target_link_libraries( cmpro_dpl
        Eigen3::Eigen
        ${OpenCV_LIBS}
        dlib::dlib
        )