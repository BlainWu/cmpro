The cmpro Project
- Structure
1. config: the configure data files
2. include/cmpro: header files
3. model: model
4. src: source files
5. test: test files

- Environment build （the deep_learning pack）
1. package dependancy (tested on MacOS Mojave 10.14)
    1. install future
    2. install zlib
    3. install opencv
    4. install dlib
    5. install SFML
    6. upgrade cmake
    7. upgrade g++
    8. install eigen3 (from source)
    9. install bazel 0.26.1 (from source/exact version!)
2. tensorflow
    1. git clone (the git link of the latest version of tensorflow)
    2. ./configure
    - if CPU version : just press enter
    - if GPU version : install cuda and cuDNN (exact version)
    3. sh build_all_linux.sh
    4. see if the folder has libtensorflow_cc.so 
    and libtensorflow_framework.so, if not, cp one from 
    /usr/lib