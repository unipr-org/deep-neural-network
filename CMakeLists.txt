cmake_minimum_required(VERSION 3.11)
project(ANN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(ann ${SOURCES})


if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

target_link_libraries(ann PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

# Used to turn on all types of logs. The default
# value of the variable DEBUG is OFF and
# can be activated using `cmake -DDEBUG=ON`
option(DEBUG "Enable debug mod" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test")
enable_testing()
