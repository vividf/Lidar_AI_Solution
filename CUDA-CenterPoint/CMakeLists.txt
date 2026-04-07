# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.18)
project(centerpoint LANGUAGES CXX CUDA)

set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

# --------------------
# Build configuration
# --------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options(-Wextra -Wall -Wno-deprecated-declarations)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Using Debug Mode")
endif()

# --------------------
# CUDA Toolkit (modern CMake; works with CMake >= 3.18, incl. CMake 4.x)
# --------------------
# Backward-compat: if someone passes -DCUDA_TOOLKIT_ROOT_DIR=..., forward it.
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(CUDAToolkit REQUIRED)

# --------------------
# TensorRT (find headers + libnvinfer)
# --------------------
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT root (contains include/ and lib/)")

# Allow env var override (handy inside containers).
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()

# Backward-compat with this repo's old default layout.
if(NOT TENSORRT_ROOT AND arch STREQUAL "x86_64")
  set(TENSORRT_ROOT "/root/.kiwi/lib/TensorRT-8.5.3.1-cuda11x")
endif()

set(_tensorrt_hint_includes "")
set(_tensorrt_hint_libs "")
if(TENSORRT_ROOT)
  list(APPEND _tensorrt_hint_includes "${TENSORRT_ROOT}/include")
  list(APPEND _tensorrt_hint_libs     "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64")
endif()

# Try to discover TensorRT from the Python wheel (common in containers).
# This runs inside the build environment (your container), not on the host.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c"
            "import os,sys,glob\n\
try:\n\
 import tensorrt as trt\n\
 base=os.path.dirname(trt.__file__)\n\
 cand_inc=[]\n\
 for p in [base, os.path.dirname(base)]:\n\
  cand_inc += [os.path.join(p,'include'), os.path.join(p,'tensorrt','include')]\n\
 inc=''\n\
 for d in cand_inc:\n\
  if os.path.exists(os.path.join(d,'NvInfer.h')):\n\
   inc=os.path.abspath(d); break\n\
 cand_lib=[]\n\
 for p in [base, os.path.dirname(base)]:\n\
  cand_lib += [p, os.path.join(p,'lib'), os.path.join(p,'lib64'), os.path.join(p,'tensorrt','lib'), os.path.join(p,'tensorrt','lib64')]\n\
 libdir=''\n\
 for d in cand_lib:\n\
  if os.path.isdir(d) and glob.glob(os.path.join(d,'libnvinfer.so*')):\n\
   libdir=os.path.abspath(d); break\n\
 print(inc)\n\
 print(libdir)\n\
except Exception:\n\
 print('')\n\
 print('')\n"
    OUTPUT_VARIABLE _trt_py_out
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_trt_py_out)
    string(REPLACE "\r\n" "\n" _trt_py_out "${_trt_py_out}")
    string(REPLACE "\r" "\n" _trt_py_out "${_trt_py_out}")
    string(REGEX REPLACE "\n+$" "" _trt_py_out "${_trt_py_out}")
    string(REGEX MATCH "^[^\n]*" _trt_py_inc "${_trt_py_out}")
    string(REGEX REPLACE "^[^\n]*\n" "" _trt_py_rest "${_trt_py_out}")
    string(REGEX MATCH "^[^\n]*" _trt_py_lib "${_trt_py_rest}")

    if(_trt_py_inc AND EXISTS "${_trt_py_inc}/NvInfer.h")
      list(APPEND _tensorrt_hint_includes "${_trt_py_inc}")
    endif()
    if(_trt_py_lib)
      list(APPEND _tensorrt_hint_libs "${_trt_py_lib}")
    endif()
  endif()
endif()

# Allow explicit override without needing a "root" directory.
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Path containing NvInfer.h (overrides auto-detect)")
set(TENSORRT_NVINFER_LIBRARY "" CACHE FILEPATH "Full path to libnvinfer.so (overrides auto-detect)")

if(TENSORRT_INCLUDE_DIR)
  list(PREPEND _tensorrt_hint_includes "${TENSORRT_INCLUDE_DIR}")
endif()
if(TENSORRT_NVINFER_LIBRARY)
  get_filename_component(_trt_libdir "${TENSORRT_NVINFER_LIBRARY}" DIRECTORY)
  list(PREPEND _tensorrt_hint_libs "${_trt_libdir}")
endif()

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h NvInferRuntime.h NvOnnxConfig.h
  HINTS ${_tensorrt_hint_includes}
  PATHS
    /usr/include
    /usr/local/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
    /usr/src/tensorrt/include
    /usr/local/TensorRT/include
)

find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer libnvinfer.so
  HINTS ${_tensorrt_hint_libs}
  PATHS
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/src/tensorrt/lib
    /usr/src/tensorrt/lib64
    /usr/local/TensorRT/lib
    /usr/local/TensorRT/lib64
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
  message(FATAL_ERROR
    "TensorRT not found.\n"
    "  - Set -DTENSORRT_ROOT=/path/to/TensorRT (must contain include/ and lib/)\n"
    "  - Or install TensorRT development packages so NvInfer.h and libnvinfer.so are visible.\n"
    "Found include: ${TENSORRT_INCLUDE_DIR}\n"
    "Found lib:     ${TENSORRT_NVINFER_LIBRARY}\n"
  )
endif()

# --------------------
# libspconv
# --------------------
# Prefer the standalone 3DSparseConvolution library if present (supports CUDA 12+),
# but keep backward-compat with this demo's vendored prebuilt library.
set(SPCONV_ROOT "" CACHE PATH "Path to libspconv root (contains include/spconv and lib/<arch>_cuda*/libspconv.so)")
set(SPCONV_CUDA_VERSION "" CACHE STRING "CUDA version tag for libspconv folder name (e.g. 11.4, 12.8, 13.0)")

# Auto-default: use Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv when available.
if(NOT SPCONV_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/3DSparseConvolution/libspconv/include/spconv/engine.hpp")
  set(SPCONV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../libraries/3DSparseConvolution/libspconv")
endif()

# If user didn't specify the tag, derive a reasonable default from the toolkit.
if(NOT SPCONV_CUDA_VERSION)
  if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
    # Prebuilt binaries are released per minor version (e.g. 12.8); 12.8 generally works for 12.x toolkits.
    set(SPCONV_CUDA_VERSION "12.8")
  else()
    set(SPCONV_CUDA_VERSION "11.4")
  endif()
endif()

set(SPCONV_INCLUDE_DIR "")
set(SPCONV_LIB_DIR "")

if(SPCONV_ROOT)
  set(SPCONV_INCLUDE_DIR "${SPCONV_ROOT}/include")
  if(EXISTS "${SPCONV_ROOT}/lib/${arch}_cuda${SPCONV_CUDA_VERSION}")
    set(SPCONV_LIB_DIR "${SPCONV_ROOT}/lib/${arch}_cuda${SPCONV_CUDA_VERSION}")
  elseif(EXISTS "${SPCONV_ROOT}/lib/${arch}")
    # Backward-compat layout.
    set(SPCONV_LIB_DIR "${SPCONV_ROOT}/lib/${arch}")
  endif()
else()
  # Vendored default (legacy, typically built for CUDA 11.x and older protobuf)
  set(SPCONV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libspconv/include")
  set(SPCONV_LIB_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libspconv/lib/${arch}")
endif()

find_library(SPCONV_LIBRARY
  NAMES spconv libspconv.so
  HINTS "${SPCONV_LIB_DIR}"
  NO_DEFAULT_PATH
)

if(NOT SPCONV_INCLUDE_DIR OR NOT EXISTS "${SPCONV_INCLUDE_DIR}/spconv/engine.hpp")
  message(FATAL_ERROR "libspconv headers not found. SPCONV_INCLUDE_DIR=${SPCONV_INCLUDE_DIR}")
endif()
if(NOT SPCONV_LIBRARY)
  message(FATAL_ERROR "libspconv not found. Tried dir: ${SPCONV_LIB_DIR} (expected libspconv.so)")
endif()

# --------------------
# Target
# --------------------
file(GLOB_RECURSE SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
  "${TENSORRT_INCLUDE_DIR}"
  "${CUDAToolkit_INCLUDE_DIRS}"
  "${SPCONV_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  CUDA::cudart
  "${TENSORRT_NVINFER_LIBRARY}"
  "${SPCONV_LIBRARY}"
)

# libspconv may depend on protobuf (onnx parser). Link protobuf if available.
find_package(Protobuf QUIET)
if(Protobuf_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotobuf)
endif()

# Make runtime locate libspconv.so when it's not in a system path.
get_filename_component(_spconv_lib_dir "${SPCONV_LIBRARY}" DIRECTORY)
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${_spconv_lib_dir}"
  INSTALL_RPATH "${_spconv_lib_dir}"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G;-g>
    $<$<COMPILE_LANGUAGE:CXX>:-g>
  )
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()