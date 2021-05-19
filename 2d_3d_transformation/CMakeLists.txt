cmake_minimum_required(VERSION 3.19)
project(2d_3d_transformation)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

# Delare the executable target until from your sources
add_executable(2d_3d_transformation 2d_3d_transform.cpp)

# link your application with OpenCV libraries
target_link_libraries(2d_3d_transformation ${OpenCV_LIBS})
