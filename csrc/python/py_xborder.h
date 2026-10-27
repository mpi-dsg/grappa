/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#ifndef CSRC_PYTHON_PY_XBORDER_H_
#define CSRC_PYTHON_PY_XBORDER_H_
#include <parallel_hashmap/phmap.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <iostream>
#include <vector>

namespace py = pybind11;
namespace t10n {
namespace xb {
class XBorder {
 public:
  XBorder(int64_t rank_, int64_t n_part_, std::vector<int64_t> prefix_sum_);

  std::vector<py::array_t<int64_t>> group_by_partition(
      const py::array_t<int64_t> uniq_nids, const py::array_t<int8_t> ret_mask);

  void mask_by_partition(const py::array_t<int64_t> uniq_nids,
                         const py::array_t<int8_t> ret_mask);

  void set_nbr_idx(int64_t idx) { this->nbr_idx = idx; }
  /*
      a group of ad_hoc functions should be used together
      for training usage
  */
  void ad_hoc_clear();
  void ad_hoc_build_id_mapping(int64_t pid,
                               const py::array_t<int64_t> gidx_arr);
  /*
    batch_feat should have the same shape as ids
    ids are the global ids
    ids_bacth_idx are the idx where these ids shows in the batch
  */
  void ad_hoc_fill_batch_feat(const py::array_t<float> batch_feat,
                              const std::vector<py::array_t<float>> feat_n,
                              const py::array_t<int64_t> ids);

  void ad_hoc_fill_batch_ideg(const py::array_t<int64_t> batch_ideg,
                              const std::vector<py::array_t<int64_t>> ideg_n,
                              const py::array_t<int64_t> ids);

  void ad_hoc_fill_batch_feat_v2(const py::array_t<float> batch_feat,
                                 const std::vector<py::array_t<float>> feat_n,
                                 const py::array_t<int64_t> ids,
                                 const py::array_t<int64_t> batch_idx);

 private:
  int64_t rank;
  int64_t n_part;
  int64_t nbr_idx;
  std::vector<int64_t> prefix_sum;
  // these two maps are used together
  std::vector<phmap::parallel_flat_hash_map<int64_t, int64_t>> gid_lid_n;
  phmap::parallel_flat_hash_map<int64_t, int64_t> gid2pid;
};
}  // namespace xb
}  // namespace t10n
#endif  // CSRC_PYTHON_PY_XBORDER_H_
