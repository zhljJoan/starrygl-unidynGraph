cmake_minimum_required(VERSION 3.16)

project(starry_unigraph_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED Python3_EXECUTABLE)
  set(Python3_EXECUTABLE "/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIRS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from torch.utils.cpp_extension import library_paths; print(';'.join(library_paths()))"
  OUTPUT_VARIABLE TORCH_LIBRARY_DIRS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE TORCH_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_INCLUDE_DIRS_RAW)
  message(FATAL_ERROR "Failed to query torch include paths from ${Python3_EXECUTABLE}")
endif()

if(NOT TORCH_LIBRARY_DIRS_RAW)
  message(FATAL_ERROR "Failed to query torch library paths from ${Python3_EXECUTABLE}")
endif()

separate_arguments(TORCH_INCLUDE_DIRS NATIVE_COMMAND "${TORCH_INCLUDE_DIRS_RAW}")
separate_arguments(TORCH_LIBRARY_DIRS NATIVE_COMMAND "${TORCH_LIBRARY_DIRS_RAW}")

find_package(OpenMP REQUIRED)

set(BTS_SAMPLER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/starry_unigraph/vendor/bts_sampler")
set(BTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/starry_unigraph/lib")

add_library(libstarrygl_sampler MODULE
  "${BTS_SAMPLER_ROOT}/sampler/export.cpp"
)

target_include_directories(libstarrygl_sampler PRIVATE
  "${BTS_SAMPLER_ROOT}/include"
  "${BTS_SAMPLER_ROOT}/sampler/include"
  "${Python3_INCLUDE_DIRS}"
  ${TORCH_INCLUDE_DIRS}
)

target_link_directories(libstarrygl_sampler PRIVATE
  ${TORCH_LIBRARY_DIRS}
)

target_compile_options(libstarrygl_sampler PRIVATE -O3)
target_compile_definitions(libstarrygl_sampler PRIVATE
  TORCH_EXTENSION_NAME=libstarrygl_sampler
  _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)

target_link_libraries(libstarrygl_sampler PRIVATE
  OpenMP::OpenMP_CXX
  c10
  torch
  torch_cpu
)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_LIBRARY_DIRS})
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(libstarrygl_sampler PRIVATE "${TORCH_PYTHON_LIBRARY}")
endif()

set_target_properties(libstarrygl_sampler PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${BTS_OUTPUT_DIR}"
  BUILD_RPATH "${TORCH_LIBRARY_DIRS_RAW}"
  INSTALL_RPATH "${TORCH_LIBRARY_DIRS_RAW}"
)
