enable_testing()

# Source files for tests
file(GLOB_RECURSE TEST_SOURCES "./*.cpp")

add_executable(test_ann ${TEST_SOURCES})

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

target_link_libraries(test_ann PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

add_test(
    NAME test_ann
    COMMAND test_ann
)

# Input files for tests
file(GLOB files_to_copy "${CMAKE_CURRENT_SOURCE_DIR}/test_*.txt")

foreach(file ${files_to_copy})
    get_filename_component(file_name ${file} NAME)
    add_custom_command(
        TARGET test_ann
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${file}"
                "${CMAKE_CURRENT_BINARY_DIR}/${file_name}"
    )
endforeach()

