/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#include "csrc/python/py_xborder.h"

#include <immintrin.h>  // Header for intrinsics (x86)
#include <omp.h>

#include <algorithm>
#include <cassert>

#include "csrc/base/config.h"
#include "csrc/python/pyutil.h"

namespace t10n {

namespace xb {
XBorder::XBorder(int64_t rank_, int64_t n_part_,
                 std::vector<int64_t> prefix_sum_) {
  this->rank = rank_;
  this->n_part = n_part_;
  this->prefix_sum = prefix_sum_;
  assert(prefix_sum_.size() == n_part_ + 1);

  for (int i = 0; i < n_part_; i++) {
    gid_lid_n.push_back(phmap::parallel_flat_hash_map<int64_t, int64_t>());
  }
}

std::vector<py::array_t<int64_t>> XBorder::group_by_partition(
    const py::array_t<int64_t> uniq_nids, const py::array_t<int8_t> ret_mask) {
  const int64_t* uniq_ids_ptr =
      static_cast<const int64_t*>(uniq_nids.request().ptr);

  std::vector<int64_t>& cond = this->prefix_sum;
  int64_t n_part = cond.size() - 1;

  int64_t n_ids = get_py_array_num_ele(uniq_nids);

  int8_t* ret_mask_ptr = static_cast<int8_t*>(ret_mask.request().ptr);
  const int64_t stride = n_ids;

  std::vector<py::array_t<int64_t>> ret;
  std::vector<std::vector<int64_t>> res(n_part);
  for (size_t i = 0; i < n_part; i++) {
    res[i] = std::vector<int64_t>();
    res[i].reserve(n_ids / n_part);
  }
#pragma omp parallel
  {
    const int OMP_N_THREAD = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    for (int pidx = thread_id; pidx < n_part; pidx += OMP_N_THREAD) {
      int8_t* const& mask_ptr = ret_mask_ptr + pidx * stride;
      const int64_t& low = cond[pidx];
      const int64_t& up = cond[pidx + 1];
      for (size_t idx = 0; idx < n_ids; idx++) {
        const int64_t& iiid = uniq_ids_ptr[idx];
        const bool& flag = (iiid >= low && iiid < up);
        mask_ptr[idx] = flag;
        if (flag) {
          res[pidx].push_back(iiid);
        }
      }
    }
  }

  for (size_t i = 0; i < n_part; i++) {
    ret.push_back(vector_to_pyarray(res[i]));
  }
  return ret;
}

void XBorder::mask_by_partition(const py::array_t<int64_t> uniq_nids,
                                const py::array_t<int8_t> ret_mask) {
  const int64_t* uniq_ids_ptr =
      static_cast<const int64_t*>(uniq_nids.request().ptr);

  std::vector<int64_t>& cond = this->prefix_sum;
  int64_t n_part = this->n_part;

  int64_t n_ids = get_py_array_num_ele(uniq_nids);

  py::buffer_info info = ret_mask.request();
  // const int64_t in_n_part = info.shape[0];
  // const int64_t in_n_node = info.shape[1];

  int8_t* ret_mask_ptr = static_cast<int8_t*>(ret_mask.request().ptr);
  const int64_t stride = n_ids;
  // assert(in_n_part == this->n_part);
  // assert(in_n_node == n_ids);

#pragma omp parallel
  {
    const int OMP_N_THREAD = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    for (int pidx = thread_id; pidx < n_part; pidx += OMP_N_THREAD) {
      int8_t* const& mask_ptr = ret_mask_ptr + pidx * stride;
      const int64_t& low = cond[pidx];
      const int64_t& up = cond[pidx + 1];
      for (size_t idx = 0; idx < n_ids; idx++) {
        const int64_t& iiid = uniq_ids_ptr[idx];
        mask_ptr[idx] = (iiid >= low && iiid < up);
      }
    }
  }
}

void XBorder::ad_hoc_clear() {
  for (int i = 0; i < this->n_part; i++) {
    gid_lid_n[i].clear();
  }
  gid2pid.clear();
}

void XBorder::ad_hoc_build_id_mapping(int64_t pid,
                                      const py::array_t<int64_t> gidx_arr) {
  if (pid == this->rank || pid == this->nbr_idx)
    return;
  int64_t num = get_py_array_num_ele(gidx_arr);
  int64_t id = 0;
  const int64_t* d_ptr = static_cast<const int64_t*>(gidx_arr.request().ptr);
  for (int i = 0; i < num; i++) {
    gid_lid_n[pid][d_ptr[i]] = id;
    ++id;
    gid2pid[d_ptr[i]] = pid;
  }
}

void XBorder::ad_hoc_fill_batch_feat(
    const py::array_t<float> batch_feat,
    const std::vector<py::array_t<float>> feat_n,
    const py::array_t<int64_t> ids) {
  py::buffer_info info = batch_feat.request();
  // std::cout << "batch_feat shape(" << info.shape[0] << ", " << info.shape[1]
  //           << ")" << std::endl;
  const int64_t feat_dim = info.shape[1];
  float* batch_feat_ptr = static_cast<float*>(batch_feat.request().ptr);

  std::vector<float*> feat_n_ptr_vec(this->n_part);
  for (int i = 0; i < this->n_part; i++) {
    float* ptr = static_cast<float*>(feat_n[i].request().ptr);
    feat_n_ptr_vec[i] = ptr;
  }

  const int64_t* id_ptr = static_cast<const int64_t*>(ids.request().ptr);
  int64_t n_id = get_py_array_num_ele(ids);

  auto& cond = this->prefix_sum;

  const int64_t& low_1 = cond[this->rank];
  const int64_t& up_1 = cond[this->rank + 1];
  const int64_t& low_2 = cond[this->nbr_idx];
  const int64_t& up_2 = cond[this->nbr_idx + 1];

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    const int OMP_N_THREAD = omp_get_num_threads();

    const int64_t& length = (n_id + OMP_N_THREAD - 1) / OMP_N_THREAD;
    const int64_t& thrd_be = thread_id * length;
    const int64_t& thrd_en =
        (thrd_be + length) > n_id ? n_id : (thrd_be + length);

    for (int64_t i = thrd_be; i < thrd_en; i++) {
      const int64_t& oid = id_ptr[i];
      int64_t pid = -1;
      int64_t lid = -1;

      if (oid >= low_1 && oid < up_1) {
        pid = rank;
        lid = oid - low_1;
      } else if (oid >= low_2 && oid < up_2) {
        pid = nbr_idx;
        lid = oid - low_2;
      } else {
        pid = gid2pid[oid];
        lid = gid_lid_n[pid][oid];
      }

      const int64_t& b_idx = i;

      // do copy from feat_n[pid][lid] to batch_feat[b_idx]
      float* const to = batch_feat_ptr + b_idx * feat_dim;
      float* const start = feat_n_ptr_vec[pid] + lid * feat_dim;
      float* const end = start + feat_dim;
      std::copy(start, end, to);
    }
  }
}

void XBorder::ad_hoc_fill_batch_ideg(
    const py::array_t<int64_t> batch_ideg,
    const std::vector<py::array_t<int64_t>> ideg_n,
    const py::array_t<int64_t> ids) {
  py::buffer_info info = batch_ideg.request();
  int64_t* batch_ideg_ptr = static_cast<int64_t*>(batch_ideg.request().ptr);

  std::vector<int64_t*> ideg_n_ptr_vec(this->n_part);
  for (int i = 0; i < this->n_part; i++) {
    int64_t* ptr = static_cast<int64_t*>(ideg_n[i].request().ptr);
    ideg_n_ptr_vec[i] = ptr;
  }

  const int64_t* id_ptr = static_cast<const int64_t*>(ids.request().ptr);
  int64_t n_id = get_py_array_num_ele(ids);

  auto& cond = this->prefix_sum;

  const int64_t& low_1 = cond[this->rank];
  const int64_t& up_1 = cond[this->rank + 1];
  const int64_t& low_2 = cond[this->nbr_idx];
  const int64_t& up_2 = cond[this->nbr_idx + 1];

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    const int OMP_N_THREAD = omp_get_num_threads();

    const int64_t& length = (n_id + OMP_N_THREAD - 1) / OMP_N_THREAD;
    const int64_t& thrd_be = thread_id * length;
    const int64_t& thrd_en =
        (thrd_be + length) > n_id ? n_id : (thrd_be + length);

    for (int64_t i = thrd_be; i < thrd_en; i++) {
      const int64_t& oid = id_ptr[i];
      int64_t pid = -1;
      int64_t lid = -1;

      if (oid >= low_1 && oid < up_1) {
        pid = rank;
        lid = oid - low_1;
      } else if (oid >= low_2 && oid < up_2) {
        pid = nbr_idx;
        lid = oid - low_2;
      } else {
        pid = gid2pid[oid];
        lid = gid_lid_n[pid][oid];
      }

      const int64_t& b_idx = i;

      // do copy from feat_n[pid][lid] to batch_feat[b_idx]
      int64_t* const to = batch_ideg_ptr + b_idx;
      int64_t* const start = ideg_n_ptr_vec[pid] + lid;
      *to = *start;
    }
  }
}

void XBorder::ad_hoc_fill_batch_feat_v2(
    const py::array_t<float> batch_feat,
    const std::vector<py::array_t<float>> feat_n,
    const py::array_t<int64_t> ids, const py::array_t<int64_t> batch_idx) {
  py::buffer_info info = batch_feat.request();
  // std::cout << "batch_feat shape(" << info.shape[0] << ", " << info.shape[1]
  //           << ")" << std::endl;
  const int64_t feat_dim = info.shape[1];
  float* batch_feat_ptr = static_cast<float*>(batch_feat.request().ptr);

  std::vector<float*> feat_n_ptr_vec(this->n_part);
  for (int i = 0; i < this->n_part; i++) {
    float* ptr = static_cast<float*>(feat_n[i].request().ptr);
    feat_n_ptr_vec[i] = ptr;
  }

  const int64_t* id_ptr = static_cast<const int64_t*>(ids.request().ptr);
  const int64_t* batch_idx_ptr =
      static_cast<const int64_t*>(batch_idx.request().ptr);

  int64_t n_id = get_py_array_num_ele(ids);

  // #pragma omp parallel
  //   {
  //     const int OMP_N_THREAD = omp_get_num_threads();
  //     int thread_id = omp_get_thread_num();
  //     const int64_t& length = (n_id + OMP_N_THREAD - 1) / OMP_N_THREAD;
  //     const int64_t& thrd_be = thread_id * length;
  //     const int64_t& thrd_en =
  //         (thrd_be + length) > n_id ? n_id : (thrd_be + length);

  for (int64_t i = 0; i < n_id; i++) {
    const int64_t& oid = id_ptr[i];
    const int64_t& pid = gid2pid[oid];
    const int64_t& lid = gid_lid_n[pid][oid];
    const int64_t& b_idx = batch_idx_ptr[i];
    // do copy from feat_n[pid][lid] to batch_feat[b_idx]
    float* const to = batch_feat_ptr + b_idx * feat_dim;
    float* const start = feat_n_ptr_vec[pid] + lid * feat_dim;
    float* const end = start + feat_dim;
    std::copy(start, end, to);
  }
  //  }
}
}  // namespace xb
}  // namespace t10n
