cmake_minimum_required(VERSION 3.10)
project(sfm_reconstruction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust this if your OpenCV is in a different place
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED sfm viz highgui core)

include_directories(/usr/local/include/opencv4)

add_executable(sfm_reconstruction sfm_reconstruction.cpp)

target_link_libraries(sfm_reconstruction ${OpenCV_LIBS})

target_compile_definitions(sfm_reconstruction PRIVATE CERES_FOUND=1)
