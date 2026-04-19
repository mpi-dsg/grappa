/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#ifndef CSRC_PYTHON_PY_SEQ_CHUNKING_H_
#define CSRC_PYTHON_PY_SEQ_CHUNKING_H_

#include <pybind11/numpy.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

namespace t10n {
namespace c_seq {
void build(const std::string path, const int64_t n_n, const int64_t n_e,
           const int64_t feat_dim, const int64_t n_part,
           const py::array_t<int64_t> src_np, const py::array_t<int64_t> dst_np,
           const py::array_t<int64_t> nid2pid_np);

void split_node_feat(const std::string path, const int64_t n_n,
                     const int64_t feat_dim, const int64_t n_part,
                     const py::array_t<float> node_feat_np);

void split_node_data(const std::string path, const int64_t n_n,
                     const int64_t n_part,
                     const py::array_t<float> node_label_np,
                     const py::array_t<int8_t> train_mask_np,
                     const py::array_t<int8_t> valid_mask_np,
                     const py::array_t<int8_t> test_mask_np);

void print_string(const std::string str);
}  // namespace c_seq
}  // namespace t10n

#endif  // CSRC_PYTHON_PY_SEQ_CHUNKING_H_
