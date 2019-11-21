cmake_minimum_required(VERSION 3.14)
project(CalCameraOdomUrdf)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(CalCameraOdomUrdf main.cpp)
target_link_libraries (
        CalCameraOdomUrdf
        ${PCL_LIBRARIES}
)