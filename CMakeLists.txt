cmake_minimum_required(VERSION 3.18)
project(t10n_extension LANGUAGES CXX CUDA)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


#################################
# Compilation settings
#################################

option(USE_EASYLOGGING "Close easylogging" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    message(FATAL_ERROR "not tested on WIN32")
else(WIN32)
  set(EXTRA_CXX_FLAGS "-std=c++14    ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-Wall ${EXTRA_CXX_FLAGS}")
  #  set(EXTRA_CXX_FLAGS "-Werror ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-fPIE -fPIC   ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-fstack-protector    ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-DDCHECK_ALWAYS_ON   ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-Wformat -Wformat-security   ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-Wno-sign-compare            ${EXTRA_CXX_FLAGS}")
  set(EXTRA_CXX_FLAGS "-Wno-unused-function         ${EXTRA_CXX_FLAGS}")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_CXX_FLAGS}")
endif(WIN32)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-faligned-new COMPILER_SUPPORTS_ALIGNED_NEW)
if (COMPILER_SUPPORTS_ALIGNED_NEW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif()


#################################
# CMake module settings
#################################
set(T10N_CMAKE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${T10N_CMAKE_PATH})


#################################
# header and library settings
#################################

include(python)
include(pybind11)
include(openmp)
include(cuda)

include(third_party/parallel-hashmap)
if(USE_EASYLOGGING)
  include(third_party/easylogging)
  add_definitions(-DELPP_THREAD_SAFE)
else()
  add_definitions(-DDISABLE_THIRD_PARTY_LOGGING)
endif()
include(third_party/pcg_cpp)

include(numa)
include_directories(${NUMA_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_LIST_DIR})


# Find the libunwind library manually
find_library(LIBUNWIND_LIB NAMES unwind libunwind PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
if(NOT LIBUNWIND_LIB)
    message(FATAL_ERROR "libunwind not found")
endif()

#################################
# target settings
#################################
set(T10N_SRC
    ${EASYLOGGINGPP_SRC}
    "csrc/base/log.cc"
    "csrc/graph/build.cc"
    "csrc/graph/graph.cc"
    "csrc/graph/sampling_h.cc"
    "csrc/graph/sampling_g.cu"
    "csrc/python/py_xborder.cc"
    "csrc/python/py_seq_chunking.cc"
    "csrc/python/py_dist_chunking.cc"
    "csrc/python/py_graph.cc"
    "csrc/python/py_test.cc"
    "csrc/python/pybind11.cc")

add_definitions(-DMODULE_NAME=_C)
Python_add_library(_C MODULE "${T10N_SRC}" WITH_SOABI)
target_link_libraries(_C PRIVATE ${NUMA_LIBRARIES} ${LIBUNWIND_LIB})
cuda_set_for_cuda_target(_C)


## compiling test cases
include(test)
