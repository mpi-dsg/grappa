/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_PYTHON_PY_DIST_CHUNKING_H_
#define CSRC_PYTHON_PY_DIST_CHUNKING_H_

#include <pybind11/numpy.h>

namespace t10n {
namespace c_dist {

void random_assign_pid(int rank, pybind11::array_t<int32_t> local_slice,
                       int32_t num_part);

}
}  // namespace t10n

#endif  // CSRC_PYTHON_PY_DIST_CHUNKING_H_
