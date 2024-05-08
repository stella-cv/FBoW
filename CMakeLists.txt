#
# The MIT License
#
# Copyright (c) 2017 Rafael Muñoz-Salinas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

# ------------------------------------------------------
#   BASIC CONFIGURATION
# ------------------------------------------------------

cmake_minimum_required(VERSION 3.12)

project(fbow CXX)

# Disable in-source build
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)

set(PROJECT_VERSION "0.0.1")
string(REGEX MATCHALL "[0-9]" PROJECT_VERSION_PARTS "${PROJECT_VERSION}")

list(GET PROJECT_VERSION_PARTS 0 PROJECT_VERSION_MAJOR)
list(GET PROJECT_VERSION_PARTS 1 PROJECT_VERSION_MINOR)
list(GET PROJECT_VERSION_PARTS 2 PROJECT_VERSION_PATCH)

set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

set(CMAKE_MACOSX_RPATH 1)

# ------------------------------------------------------
#   build type
# ------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# ------------------------------------------------------
#   library names and directories
# ------------------------------------------------------

set(PROJECT_DLL_VERSION "${PROJECT_VERSION_MAJOR}${PROJECT_VERSION_MINOR}${PROJECT_VERSION_PATCH}")
if(WIN32)
    set(RUNTIME_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" CACHE PATH "Directory for dlls and binaries")
    set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" CACHE PATH "Directory for binaries")
    set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" CACHE PATH "Directory for DLLs")
else()
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_INSTALL_PREFIX}/lib/cmake/" "/usr/lib/cmake" "/usr/local/lib/cmake")
endif()

option(BUILD_UTILS "Build utils" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_SHARED_LIBS "Build static libraries" ON)
option(USE_CONTRIB "Use xfeatures2d from OpenCV contrib modules" OFF)

# ------------------------------------------------------
#   find dependencies
# ------------------------------------------------------

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(OpenCV_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "OpenCV 3.0 or higher is required.")
endif()

if(USE_CONTRIB)
    add_definitions(-DUSE_CONTRIB)
endif()

# ------------------------------------------------------
#   PROJECT CONFIGURATION
# ------------------------------------------------------
 
# ------------------------------------------------------
#   uninstall target for "make uninstall"
# ------------------------------------------------------

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

# ------------------------------------------------------
#   create configuration file from .in file
#   (if you use windows take care with paths)
# ------------------------------------------------------

configure_file("${PROJECT_SOURCE_DIR}/config.cmake.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/cmake/${PROJECT_NAME}/")

# ------------------------------------------------------
#   program optimization and debug options
# ------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)

set(WARNINGS_ARE_ERRORS OFF CACHE BOOL "Treat warnings as errors")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang" OR MINGW)
    # profiling option
    set(ENABLE_PROFILING OFF CACHE BOOL "Enable profiling in the GCC/Clang compiler (Add flags: -g -pg)")
    # option for omitting frame pointer
    set(USE_OMIT_FRAME_POINTER ON BOOL "Enable -fomit-frame-pointer for GCC/Clang")

    # select optimization level
    if(CMAKE_SYSTEM_PROCESSOR MATCHES arm*)
        # ARM system
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC/Clang")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC/Clang")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES powerpc*)
        # PowerPC system
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC/Clang")
        set(USE_POWERPC ON CACHE BOOL "Enable PowerPC for GCC/Clang")
    endif ()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES amd64* OR CMAKE_SYSTEM_PROCESSOR MATCHES x86_64*)
        # 64bit system
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC/Clang")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC/Clang")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC/Clang")
        set(USE_SSE ON CACHE BOOL "Enable SSE for GCC/Clang")
        set(USE_SSE2 ON CACHE BOOL "Enable SSE2 for GCC/Clang")
        set(USE_SSE3 ON CACHE BOOL "Enable SSE3 for GCC/Clang")
        set(USE_SSE4 ON CACHE BOOL "Enable SSE4 for GCC/Clang")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES i686* OR CMAKE_SYSTEM_PROCESSOR MATCHES x86)
        # 32bit system
        set(USE_O3 ON CACHE BOOL "Enable -O3 for GCC/Clang")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC/Clang")
        set(USE_MMX ON CACHE BOOL "Enable MMX for GCC/Clang")
        set(USE_SSE OFF CACHE BOOL "Enable SSE for GCC/Clang")
        set(USE_SSE2 OFF CACHE BOOL "Enable SSE2 for GCC/Clang")
        set(USE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC/Clang")
        set(USE_SSE4 OFF CACHE BOOL "Enable SSE4 for GCC/Clang")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES aarch64*)
        # aarch64 system
        set(USE_O2 ON CACHE BOOL "Enable -O2 for GCC/Clang")
        set(USE_FAST_MATH OFF CACHE BOOL "Enable -ffast-math for GCC/Clang")
        set(USE_AVX OFF CACHE BOOL "Enable AVX for GCC/Clang" FORCE)
        set(USE_MMX OFF CACHE BOOL "Enable MMX for GCC/Clang" FORCE)
        set(USE_SSE OFF CACHE BOOL "Enable SSE for GCC/Clang" FORCE)
        set(USE_SSE2 OFF CACHE BOOL "Enable SSE2 for GCC/Clang" FORCE)
        set(USE_SSE3 OFF CACHE BOOL "Enable SSE3 for GCC/Clang" FORCE)
        set(USE_SSE4 OFF CACHE BOOL "Enable SSE4 for GCC/Clang" FORCE)
    else()
        set(USE_AVX ON CACHE BOOL "Enable AVX for GCC/Clang")
    endif()

    # warning options
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wall")
    # suppress warnings: ignoring attributes on template argument '__m256 {aka __vector(8) float}'
    set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wno-ignored-attributes")
    if(WARNINGS_ARE_ERRORS)
        set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Werror")
    endif()

    # -Wno-long-long is required in 64bit systems when including sytem headers
    if(CMAKE_SYSTEM_PROCESSOR MATCHES x86_64* OR CMAKE_SYSTEM_PROCESSOR MATCHES amd64*)
		set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -Wno-long-long")
    endif()

    # apply options
    if(NOT ENABLE_PROFILING AND USE_OMIT_FRAME_POINTER)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
    endif()
    if(USE_O2)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -O2")
    endif()
    if(USE_O3)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -O3")
    endif()
    if(USE_FAST_MATH)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -ffast-math")
    endif()
    if(USE_POWERPC)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -mcpu=G3 -mtune=G5")
    endif()

    # apply vectorization
    if(USE_MMX)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mmmx")
        add_definitions(-DUSE_MMX)
    endif()
    if(USE_SSE)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse")
        add_definitions(-DUSE_SSE)
    endif()
    if(USE_SSE2)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse2")
        add_definitions(-DUSE_SSE2)
    endif()
    if(USE_SSE3 AND NOT MINGW)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse3")
        add_definitions(-DUSE_SSE3)
    endif()
    if(USE_SSE4 AND NOT MINGW)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -msse4.1")
        add_definitions(-DUSE_SSE4)
    endif()
    IF(USE_AVX)
        set(VECTORIAL_INSTRUCTIONS "${VECTORIAL_INSTRUCTIONS} -mavx")
        add_definitions(-DUSE_AVX)
    endif()

    if(ENABLE_PROFILING)
        set(EXTRA_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -pg -g")
    else()
        if(NOT APPLE)
            set(EXTRA_CXX_FLAGS "${EXTRA_CXX_FLAGS} -ffunction-sections")
        endif()
    endif()

    set(CMAKE_CXX_FLAGS "${VECTORIAL_INSTRUCTIONS} ${EXTRA_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${EXTRA_CXX_FLAGS_RELEASE} -DNDEBUG -D_NDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g3 -DDEBUG -D_DEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g3 -DDEBUG -D_DEBUG")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS}")

# ------------------------------------------------------
#   fbow library
# ------------------------------------------------------

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

set(SOURCES
    src/fbow.cpp
    src/vocabulary_creator.cpp)

add_library(fbow ${SOURCES})

set_target_properties(fbow PROPERTIES
        DEFINE_SYMBOL FBOW_DLL_EXPORT
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_SOVERSION}
        CLEAN_DIRECT_OUTPUT 1
        OUTPUT_NAME ${PROJECT_NAME})

find_package(OpenMP)
if(OPENMP_FOUND)
    add_compile_options(-DUSE_OPENMP)
    target_link_libraries(fbow PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(fbow
    PUBLIC
        opencv_core
    PRIVATE
        opencv_features2d
        opencv_highgui
        "$<$<BOOL:${USE_CONTRIB}>:opencv_xfeatures2d>"
)
target_include_directories(fbow PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")

install(TARGETS fbow
        EXPORT fbowTargets
        LIBRARY DESTINATION lib)

install(EXPORT fbowTargets
        NAMESPACE fbow::
        DESTINATION share/cmake/fbow/)

export(EXPORT fbowTargets)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)

# ------------------------------------------------------
#   utility directories
# ------------------------------------------------------

if(BUILD_UTILS)
    add_subdirectory(utils)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ------------------------------------------------------
#   display status message for important variables
# ------------------------------------------------------

message(STATUS)
message(STATUS "----------------------------------------" )
message(STATUS "  General configuration for ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "----------------------------------------" )
message(STATUS)
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
message(STATUS)
message(STATUS "OpenCV_DIR = ${OpenCV_DIR}" )
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
message(STATUS)
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build dynamic libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build utils: ${BUILD_UTILS}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Use OpenMP: ${OPENMP_FOUND}")
message(STATUS)
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS)
message(STATUS "C++ flags (Common):${CMAKE_CXX_FLAGS}")
message(STATUS "C++ flags (Release):${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug):${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags (Relaese+Debug):${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS)
