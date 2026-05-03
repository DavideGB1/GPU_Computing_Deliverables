cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES native)

project(CudaMatrixProject LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(SpMV SpMV.cu matrix_format.c ./time/time_lib.c)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(SpMV
        PRIVATE
        m
        CUDA::cusparse
        OpenMP::OpenMP_C
)

target_compile_options(SpMV PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
)