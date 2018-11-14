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

add_definitions(-DNOMINMAX)

set(SOURCES fbow.cpp vocabulary_creator.cpp)
set(HEADERS fbow.h fbow_exports.h vocabulary_creator.h cpu.h)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
        DEFINE_SYMBOL FBOW_DLL_EXPORT
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_SOVERSION}
        CLEAN_DIRECT_OUTPUT 1
        OUTPUT_NAME ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${OpenMP_CXX_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin COMPONENT main
        LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
        ARCHIVE DESTINATION lib COMPONENT main)

if(INSTALL_FBOW_DEV)
    install(FILES ${HEADERS} DESTINATION include/fbow COMPONENT main)
endif()
