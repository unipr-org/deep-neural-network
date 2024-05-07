cmake_minimum_required(VERSION 3.11)
project(ANN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 \
    -fcrossjumping \
    -fcse-skip-blocks \
    -fdelete-null-pointer-checks \
    -fgcse \
    -fipa-bit-cp  -fipa-cp  -fipa-icf \
    -fipa-ra  -fipa-sra  -fipa-vrp \
    -fisolate-erroneous-paths-dereference \
    -flra-remat \
    -foptimize-sibling-calls \
    -fthread-jumps \
    -ftree-builtin-call-dce \
    -fvect-cost-model=dynamic \
    -floop-interchange \
    -floop-unroll-and-jam \
    -fpredictive-commoning \
    -ftree-partial-pre \
    -faggressive-loop-optimizations \
    -fsplit-wide-types-early \
    -fmodulo-sched \
    -fmodulo-sched-allow-regmoves \
    -fno-lifetime-dse \
")
include_directories(include)
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(dnn ${SOURCES})


if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

target_link_libraries(dnn PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

# Used to turn on all types of logs. The default
# value of the variable DEBUG is OFF and
# can be activated using `cmake -DDEBUG=ON`
option(DEBUG "Enable debug mod" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test")
enable_testing()

# Doxygen
find_package(Doxygen REQUIRED)
add_custom_target(
    documentation ALL
    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	COMMENT "Generating doc with Doxygen"
    VERBATIM
)
