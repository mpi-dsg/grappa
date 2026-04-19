/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#ifndef CSRC_PYTHON_PYUTIL_H_
#define CSRC_PYTHON_PYUTIL_H_

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>  // for posix_memalign
#include <vector>

#include "csrc/base/error.h"

namespace py = pybind11;

namespace t10n {
template <typename DataType>
int64_t get_py_array_num_ele(const py::array_t<DataType> input_array) {
  const py::ssize_t* shape = input_array.shape();
  return *shape;
}

template <typename DataType>
py::array_t<DataType> vector_to_pyarray(
    const std::vector<DataType>& input_vector) {
  size_t len = input_vector.size();
  // Create a NumPy array with the correct dimensions (1D in this case)
  py::array_t<DataType> result_array =
      py::array_t<DataType>({static_cast<pybind11::ssize_t>(len)});
  // Get a pointer to the NumPy array's data
  py::buffer_info buf = result_array.request();
  DataType* ptr = static_cast<DataType*>(buf.ptr);
  // Copy data from the std::vector to the NumPy array
  std::copy(input_vector.begin(), input_vector.end(), ptr);
  return result_array;
}

template <typename DataType>
py::array_t<DataType> create_py_array(size_t num_ele) {
  void* tmp;
  if (posix_memalign(&tmp, 1024, num_ele * sizeof(DataType))) {
    ABORT_IF_T(true, "posix_memalign failed");
  }

  return py::array_t<DataType>(
      py::buffer_info(tmp,
                      sizeof(DataType),  // itemsize
                      py::format_descriptor<DataType>::format(),
                      1,                                        // ndim
                      std::vector<size_t>{num_ele},             // shape
                      std::vector<size_t>{sizeof(DataType)}));  // stride
}
}  // namespace t10n
#endif  // CSRC_PYTHON_PYUTIL_H_
