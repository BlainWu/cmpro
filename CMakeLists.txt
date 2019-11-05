cmake_minimum_required( VERSION 3.14 )
project( cmpro )
set( CMAKE_CXX_COMPILER "g++" )
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(DLIB_NO_GUI_SUPPORT 0)

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
find_package( dlib REQUIRED )
find_package( Eigen3 REQUIRED NO_MODULE)

#find_package(SFML 2.5 COMPONENTS graphics audio REQUIRED)
# 添加头文件

#set(TENSORFLOW_SOURCE_DIR /Users/kanae/Documents/soft/tensorflow)


include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories(                              # 头文件的搜索目录
#        ${TENSORFLOW_SOURCE_DIR}
#        ${TENSORFLOW_SOURCE_DIR}/bazel-genfiles
#        ${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow
#        ${TENSORFLOW_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/nsync/public
#        ${TENSORFLOW_SOURCE_DIR}/tensorflow/contrib/makefile/downloads/absl
#        ${TENSORFLOW_SOURCE_DIR}/tensorflow/contrib/makefile/gen/protobuf/include
#        /usr/local/include/eigen3
#        )
link_directories(${TENSORFLOW_SOURCE_DIR}/bazel-bin/tensorflow)
add_executable(cmpro_rate
        test/cmpro_rate.cpp
        include/cmpro/common_include.h
        include/cmpro/shape_processing.h src/shape_processing.cpp
        include/cmpro/rate_method.h  src/rate_method.cpp
        include/cmpro/configure.h src/configure.cpp
        )
add_executable(cmpro_rec
        test/cmpro_record.cpp
        include/cmpro/common_include.h
        include/cmpro/shape_processing.h src/shape_processing.cpp
        include/cmpro/rate_method.h  src/rate_method.cpp
        include/cmpro/configure.h src/configure.cpp
        include/cmpro/coordinate_convert.h src/coordinate_convert.cpp
        )
add_executable(cmpro_dpl
        test/cmpro_dpl.cpp
        include/cmpro/common_include.h
        include/cmpro/shape_processing.h src/shape_processing.cpp
        include/cmpro/learning_method.h  src/learning_method.cpp
        include/cmpro/configure.h src/configure.cpp
        include/cmpro/model.h src/model.cpp
        src/deep_process.cpp include/cmpro/deep_process.h
        include/cmpro/sequence_convert.h src/sequence_convert.cpp
        )


add_executable(freeze_model src/model_freeze.cpp)

#include_directories( ${SFML_INCLUDE_DIRS} )


# 链接OpenCV库

target_link_libraries( cmpro_rate sfml-graphics sfml-audio)
target_link_libraries( cmpro_rate
        Eigen3::Eigen
        ${OpenCV_LIBS}
        dlib::dlib
        )

target_link_libraries( cmpro_rec
        Eigen3::Eigen
        ${OpenCV_LIBS}
        dlib::dlib
        )

target_link_libraries( cmpro_dpl sfml-graphics sfml-audio)
#target_link_libraries( cmpro_dpl tensorflow_cc tensorflow_framework)
target_link_libraries( cmpro_dpl
        Eigen3::Eigen
        ${OpenCV_LIBS}
        dlib::dlib
        )