enable_testing()

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

# Discover and add all test files
file(GLOB TEST_CASES "test_*.cpp")
file(GLOB DNN_SRC "../src/dnn/*.cpp")
foreach(TEST_CASE ${TEST_CASES})
    get_filename_component(TEST_NAME ${TEST_CASE} NAME_WE)
	add_executable(${TEST_NAME} ${TEST_CASE} ${DNN_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE spdlog::spdlog)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

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

