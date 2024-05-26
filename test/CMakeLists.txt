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

	# Determina il percorso del file txt associato al test
	set(TEST_TXT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.txt")

	# Verifica se il file txt associato esiste
	if(EXISTS ${TEST_TXT_FILE})
		# Copia il file txt nella directory di build
		configure_file(${TEST_TXT_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.txt COPYONLY)
	endif() 

	add_executable(${TEST_NAME} ${TEST_CASE} ${DNN_SRC})
	target_link_libraries(${TEST_NAME} PRIVATE spdlog::spdlog)
	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

endforeach()

set(DATASET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dataset)


# Controlla se la cartella dataset esiste
if(EXISTS ${DATASET_DIR})
    # Se la cartella esiste, copia i file
    file(COPY ${DATASET_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
else()
    message(WARNING "Dataset directory not found!")
endif()
