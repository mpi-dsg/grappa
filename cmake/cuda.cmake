find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

function (cuda_set_for_cxx_target TARGET_NAME)

endfunction()

function (cuda_set_for_cuda_target TARGET_NAME)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
endfunction()