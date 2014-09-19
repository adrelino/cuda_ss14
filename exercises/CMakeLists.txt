cmake_minimum_required(VERSION 2.8)

project(GPU_Course_Exercises)

# find cuda and openCV
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)

# pass compile options to nvcc
set(GENERAL_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;--use_fast_math;--compiler-options -Wall)

set(CUDA_NVCC_FLAGS ${GENERAL_NVCC_FLAGS})

#specify include directories
include_directories(common/include)

#specify library paths
link_directories()

# ex2
cuda_add_executable(
  ex2/squareArray
  ex2/squareArray.cu
  )

cuda_add_executable(
  ex2/addArrays
  ex2/addArrays.cu
  )

# ex3
cuda_add_executable(
  ex3/main
  ex3/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex3/main
  ${OpenCV_LIBS}
)

# ex4
cuda_add_executable(
  ex4/main
  ex4/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex4/main
  ${OpenCV_LIBS}
  )

# ex5
cuda_add_executable(
  ex5/main
  ex5/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex5/main
  ${OpenCV_LIBS}
  )

# ex6
cuda_add_executable(
  ex6/main
  ex6/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex6/main
  ${OpenCV_LIBS}
  )

# ex7
cuda_add_executable(
  ex7/main
  ex7/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex7/main
  ${OpenCV_LIBS}
  )

# ex7_bonus
cuda_add_executable(
  ex7_bonus/main
  ex7_bonus/main.cu
  common/src/helper.cu
  common/include/helper.h
  )
# link cublas
cuda_add_cublas_to_target(
  ex7_bonus/main
  )

target_link_libraries(
  ex7_bonus/main
  ${OpenCV_LIBS}
  )

# ex8
cuda_add_executable(
  ex8/main
  ex8/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex8/main
  ${OpenCV_LIBS}
  )

# ex9
cuda_add_executable(
  ex9/main
  ex9/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex9/main
  ${OpenCV_LIBS}
  )

# ex11
cuda_add_executable(
  ex11/main
  ex11/main.cu
  common/src/helper.cu
  common/include/helper.h
  common/include/opencv_helpers.h
  common/src/opencv_helpers.cpp
  common/include/common_kernels.cuh
  )

target_link_libraries(
  ex11/main
  ${OpenCV_LIBS}
  )

# ex12
cuda_add_executable(
  ex12/main
  ex12/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

target_link_libraries(
  ex12/main
  ${OpenCV_LIBS}
  )

# ex13_bonus
cuda_add_executable(
  ex13_bonus/main
  ex13_bonus/main.cu
  common/src/helper.cu
  common/include/helper.h
  common/include/opencv_helpers.h
  common/src/opencv_helpers.cpp
  common/include/common_kernels.cuh
  )

target_link_libraries(
  ex13_bonus/main
  ${OpenCV_LIBS}
  )

# ex14
cuda_add_executable(
  ex14/main
  ex14/main.cu
  common/src/helper.cu
  common/include/helper.h
  common/include/opencv_helpers.h
  common/src/opencv_helpers.cpp
  common/include/common_kernels.cuh
  )

cuda_add_cublas_to_target(
  ex14/main
  )

target_link_libraries(
  ex14/main
  ${OpenCV_LIBS}
  )

# ex15
cuda_add_executable(
  ex15/main
  ex15/main.cu
  common/src/helper.cu
  common/include/helper.h
  )

cuda_add_cublas_to_target(
  ex15/main
  )

target_link_libraries(
  ex15/main
  ${OpenCV_LIBS}
  )

# ex16
# ensure to compile it with nvcc compile flag '-arch=sm_12'; otherwise
# atomicAdd is not defined
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_12)

cuda_add_executable(
  ex16/main
  ex16/main.cu
  common/src/helper.cu
  common/include/helper.h
  common/include/opencv_helpers.h
  common/src/opencv_helpers.cpp
  common/include/common_kernels.cuh
  )

target_link_libraries(
  ex16/main
  ${OpenCV_LIBS}
  )

# set stuff back to most general flag settings
set(CUDA_NVCC_FLAGS ${GENERAL_NVCC_FLAGS})
