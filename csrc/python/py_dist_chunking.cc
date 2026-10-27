/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#include <pybind11/numpy.h>

#include "csrc/base/random.h"
#include "csrc/python/py_dist_chunking.h"

namespace t10n {
namespace c_dist {

void random_assign_pid(int rank, pybind11::array_t<int32_t> local_slice,
                       int32_t num_part) {
  pybind11::buffer_info info = local_slice.request();
  const int64_t len = info.shape[0];
  int32_t* ptr = static_cast<int32_t*>(info.ptr);
  const int32_t& range_end = num_part - 1;

#pragma omp parallel for
  for (int64_t idx = 0; idx < len; ++idx) {
    ptr[idx] = uniform_pcg32<int32_t>(0, range_end);
  }
}

}  // namespace c_dist
}  // namespace t10n
