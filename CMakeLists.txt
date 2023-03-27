cmake_minimum_required(VERSION 3.5)

project(simdx
        VERSION 0.1
        DESCRIPTION "Experiments with Intel SIMD"
        LANGUAGES CXX)

add_library(simdx INTERFACE)
target_include_directories(simdx INTERFACE include/simdx)

add_executable(test_compiles test/test_compiles.cc)
target_link_libraries(test_compiles PUBLIC simdx)
add_custom_target(check_compiles DEPENDS test_compiles)
add_custom_command(
	TARGET check_compiles
	COMMAND $<TARGET_FILE:test_compiles>
	COMMENT "Test whether the SIMD library compiles"
)

add_executable(test_intraregister_sort test/test_intraregister_sort.cc)
target_compile_options(test_intraregister_sort PUBLIC "-mavx512f")
target_link_libraries(test_intraregister_sort PUBLIC simdx)
add_custom_target(check_intraregister_sort DEPENDS test_intraregister_sort)
add_custom_command(
	TARGET check_intraregister_sort
	COMMAND $<TARGET_FILE:test_intraregister_sort>
	COMMENT "Test intraregister sort"
)

