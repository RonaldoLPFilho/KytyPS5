set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()
# No git (or no repository): version-stamp.txt next to the sources ("<hash> <revision>"), so a local build keeps the
# stamp the driver pipeline cache and the shader warmup need (an "unknown" revision turns both off).
if(KYTY_GIT_REVISION STREQUAL "unknown" AND EXISTS "${GIT_WORKING_DIRECTORY}/version-stamp.txt")
	file(STRINGS "${GIT_WORKING_DIRECTORY}/version-stamp.txt" KYTY_VERSION_STAMP LIMIT_COUNT 1)
	string(REGEX MATCH "^([^ ]+) ([0-9a-f]+)$" KYTY_VERSION_STAMP_MATCH "${KYTY_VERSION_STAMP}")
	if(KYTY_VERSION_STAMP_MATCH)
		set(KYTY_GIT_HASH "${CMAKE_MATCH_1}")
		set(KYTY_GIT_REVISION "${CMAKE_MATCH_2}")
	endif()
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")
