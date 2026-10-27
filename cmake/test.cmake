set(TEST_REDUCE
    "csrc/test/test_reduce.cc"
    ${EASYLOGGINGPP_SRC})
add_executable(test_reduce.bin ${TEST_REDUCE})

set(TEST_RANDOM
    "csrc/test/test_random.cc"
    ${EASYLOGGINGPP_SRC})
add_executable(test_random.bin ${TEST_RANDOM})

set(TEST_BUILD_GRAPH "csrc/test/test_build_graph.cc" ${EASYLOGGINGPP_SRC})
add_executable(test_build_graph.bin ${TEST_BUILD_GRAPH})

set(TEST_GRAPH_SAMPLING "csrc/test/test_sampling_h.cc" ${EASYLOGGINGPP_SRC})
add_executable(test_sampling.bin ${TEST_GRAPH_SAMPLING})


set(TEST_CUDA "csrc/test/test_cuda.cu" ${EASYLOGGINGPP_SRC})
add_executable(test_cuda.bin ${TEST_CUDA})
cuda_set_for_cuda_target( test_cuda.bin )

set(TEST_CUDA "csrc/test/test_sampling_g.cu" ${EASYLOGGINGPP_SRC})
add_executable(test_sampling_g.bin ${TEST_CUDA})
cuda_set_for_cuda_target( test_sampling_g.bin )