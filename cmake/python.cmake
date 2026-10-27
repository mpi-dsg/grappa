
function (python_exe RET CODE_LINE)
  execute_process(
    COMMAND
    "which" "python3"
    OUTPUT_VARIABLE WHICH_PYTHON
    RESULT_VARIABLE RET_CODE
    ERROR_VARIABLE  RET_MSG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT RET_CODE EQUAL 0)
	  message(FATAL_ERROR "${RET_CODE}: ${RET_MSG}")
  endif()
  execute_process(
    COMMAND
    "${WHICH_PYTHON}" "-c" "${CODE_LINE}" 
    OUTPUT_VARIABLE STD_OUT
    RESULT_VARIABLE RET_CODE
    ERROR_VARIABLE  RET_MSG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT RET_CODE EQUAL 0)
	  message(FATAL_ERROR "${RET_CODE}: ${RET_MSG}")
  endif()
  set(${RET} ${STD_OUT} PARENT_SCOPE)
endfunction()

macro (add_pytorch_cmake_prefix_path)
	python_exe(_PREFIX_PATH "import torch; print(torch.utils.cmake_prefix_path)" )	
	list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

macro (py_add_pybind11_cmake_prefix_path)
	python_exe(_PREFIX_PATH "import pybind11; print(pybind11.commands.get_cmake_dir())" )	
	list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

set(Python_EXECUTABLE python3)
find_package(Python COMPONENTS Interpreter Development.Module)
if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching")
endif()