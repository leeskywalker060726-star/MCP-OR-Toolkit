cmake_minimum_required(VERSION 3.12)
project(ExactOR_Agent)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 寻找 Python 和 Pybind11 依赖
find_package(pybind11 REQUIRED)

# 加上 NO_EXTRAS 解决 Windows MinGW 路径中文 bug
pybind11_add_module(exactor NO_EXTRAS exactor_wrapper.cpp)

# 针对不同平台优化编译体积和速度
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(exactor PRIVATE -O3 -fPIC -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(exactor PRIVATE /O2)
endif()