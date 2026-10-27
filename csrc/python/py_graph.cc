/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#include <numa.h>
#include <pybind11/stl.h>
#include <chrono>  // NOLINT(*)
#include <memory>  // for unique_ptr<>
#include <utility>
#include <vector>

#include "csrc/base/config.h"
#include "csrc/base/error.h"
#include "csrc/graph/build.h"
#include "csrc/graph/sampling_h.h"
#include "csrc/python/py_graph.h"
#include "csrc/python/pyutil.h"

namespace t10n {

namespace pygraph {

HostGraph::HostGraph(const pybind11::array_t<int64_t> u,
                     const pybind11::array_t<int64_t> v, int node_id) {
  node_mask = nullptr;
  if (numa_available() >= 0) {
    node_mask = numa_allocate_nodemask();
    numa_bitmask_setbit(node_mask, node_id);
    numa_set_membind(node_mask);
  }
  csc_h<int64_t> ret(py_build_csc_h(u, v));
  csc.copy(ret);
  print_csc_h(csc);
}

HostGraph::HostGraph(const pybind11::array_t<int64_t> u,
                     const pybind11::array_t<int64_t> v) {
  node_mask = nullptr;
  csc_h<int64_t> ret(py_build_csc_h(u, v));
  csc.copy(ret);
  print_csc_h(csc);
}

HostGraph::~HostGraph() {
  if (numa_available() >= 0 && node_mask != nullptr) {
    numa_free_nodemask(node_mask);
  }
}

HostSampler::HostSampler(HostGraph* g_, std::vector<size_t> fanouts_,
                         const bool repeated_)
    : g(g_),
      fanouts(fanouts_),
      repeated(repeated_),
      count_resampling(false),
      nodes_ptr(nullptr),
      gideg_ptr(nullptr),
      batch_size(0),
      last_batch_size(0) {
  //    LOG(INFO) << __func__ << ": thread_id: " << std::this_thread::get_id();
}

HostSampler::HostSampler(HostGraph* g_, std::vector<size_t> fanouts_,
                         const bool repeated_, const bool count_resampling_)
    : g(g_),
      fanouts(fanouts_),
      repeated(repeated_),
      count_resampling(count_resampling_),
      nodes_ptr(nullptr),
      gideg_ptr(nullptr),
      batch_size(0),
      last_batch_size(0) {
  //    LOG(INFO) << __func__ << ": thread_id: " << std::this_thread::get_id();
}

HostSampler::~HostSampler() {
  if (worker.joinable()) {
    worker.join();
  }
}

/* a variation of sampling_graph_h()

  if gideg is not null, this function will be used for counting
  the number of resampling
*/

void HostSampler::batch_append_sampled_coo(Batch* b, const size_t fanout,
                                           const size_t n_source,
                                           const int64_t* source,
                                           const int64_t* gideg) {
  // LOG(DEBUG) << "in_, n_source=" << n_source;

  if (n_source < 1) {
      return;
  }
  chunk<size_t> pick_n_i_nbr(n_source, 0);
  chunk<size_t> source_i_deg(n_source, 0);

  collect_pick_n_and_ideg<int64_t>(g->csc, fanout, repeated, n_source, source,
                                   &pick_n_i_nbr, &source_i_deg);

  chunk<size_t> i_nbr_offset = exclusive_scan(pick_n_i_nbr);

  const size_t n_e = i_nbr_offset.back();

  if (n_e == 0) {
    std::vector<int64_t> none(0);
    b->layers.push_back(none);
    b->layers.push_back(none);
    b->input_nodes.push_back(none);
    return;
  }

  std::vector<int64_t> coo_row(n_e, 0);
  std::vector<int64_t> coo_col(n_e, 0);

  int64_t* row_ptr = coo_row.data();
  int64_t* col_ptr = coo_col.data();

  if (gideg == nullptr) {
    fill_sampling_result_to_coo<int64_t>(g->csc, repeated, n_source, source,
                                         &i_nbr_offset, &pick_n_i_nbr,
                                         &source_i_deg, row_ptr, col_ptr);
  } else {
    float resampling_count = 0.0;
    fill_w_reampling_sampling_result_to_coo<int64_t>(
        g->csc, repeated, n_source, source, gideg, &i_nbr_offset, &pick_n_i_nbr,
        &source_i_deg, row_ptr, col_ptr, &resampling_count);
    b->resampling[0] += resampling_count;
    // -------------
    //    debug 
    // -------------
    //    = 1.0
    b->resampling[1] += static_cast<float>(n_source); // [1] used as counter
  }

  b->layers.emplace_back(std::move(coo_row));
  b->layers.emplace_back(std::move(coo_col));
  i_nbr_offset.resize(0);

  //
  // find unique source nodes
  //
  int OMP_N_THREAD = 0;
#pragma omp parallel
  {
    const int& tid = omp_get_thread_num();
    if (tid == 0) {
      OMP_N_THREAD = omp_get_num_threads();
    }
  }

  std::vector<phmap::parallel_flat_hash_set<int64_t>> set_vec(OMP_N_THREAD);
  for (int i = 0; i < OMP_N_THREAD; i++) {
    set_vec.push_back(phmap::parallel_flat_hash_set<int64_t>());
  }

  chunk<size_t> num_per_thread(OMP_N_THREAD, 0);
#pragma omp parallel
  {
    const int& tid = omp_get_thread_num();
    for (size_t idx = 0; idx < n_e; idx++) {
      const int64_t& nid = row_ptr[idx];
      if (nid % OMP_N_THREAD == tid) {
        set_vec[tid].insert(nid);
      }
    }
#pragma omp barrier
    num_per_thread[tid] = set_vec[tid].size();
  }

  chunk<size_t> scan = exclusive_scan(num_per_thread);

  size_t total_n = scan.back();
  // LOG(DEBUG) << "total_n: " << total_n;
  std::vector<int64_t> inputs(total_n, 0);

#pragma omp parallel
  {
    const int& tid = omp_get_thread_num();
    size_t idx = scan[tid];
    for (auto& node : set_vec[tid]) {
      inputs[idx++] = node;
    }
  }

  b->input_nodes.emplace_back(std::move(inputs));
  // LOG(DEBUG) << "b->input_nodes.emplace_back(std::move(inputs)); "
  // << b->input_nodes.back().size();
}

void HostSampler::worker_func() {
  ABORT_IF_T(this->count_resampling,
             "if resampling, use worker_with_resampling_func");

  while (this->batch_idx.load() < this->total_batch_n) {
    if (this->q_len.load() < this->kMaxQLen) {
      std::unique_ptr<Batch> b_ptr(new Batch);

      size_t n_source = this->batch_size;
      if (this->batch_idx.load() + 1 == this->total_batch_n) {
        n_source = this->last_batch_size;
      }
      int64_t* source = this->nodes_ptr + this->batch_idx * this->batch_size;

      for (auto it = this->fanouts.rbegin(); it != this->fanouts.rend(); ++it) {
        // LOG(DEBUG) << "before_, n_source=" << n_source;
        this->batch_append_sampled_coo(b_ptr.get(), *it, n_source, source,
                                       nullptr);
        // LOG(DEBUG) << "after_, n_source=" << n_source;

        auto& last = b_ptr->input_nodes.back();
        n_source = last.size();
        source = last.data();
      }
      {
        lock_guard guard(&this->q_lock);
        this->q.push(std::move(b_ptr));
        this->batch_idx++;
        this->q_len++;
      }
    }
  }
}

void HostSampler::ReinitFrom(pybind11::array_t<int64_t> source_,
                             size_t batch_size_, size_t use_batch_n) {
  ABORT_IF_T(this->count_resampling, "if resampling, use ReinitWithGidegFrom");

  if (this->worker.joinable()) {
    this->worker.join();
  }

  while (true) {
    lock_guard guard(&q_lock);
    if (q.empty()) {
      pybind11::buffer_info info = source_.request();
      const size_t num = info.shape[0];
      this->nodes_ptr = static_cast<int64_t*>(info.ptr);
      this->batch_size = batch_size_;
      this->batch_idx.store(0);
      //this->total_batch_n = (num + batch_size_ - 1) / batch_size_;
      this->total_batch_n = use_batch_n;
      this->q_len.store(0);
      auto md = num % this->batch_size;
      this->last_batch_size = md == 0 ? this->batch_size : md;

      this->worker = std::thread(&HostSampler::worker_func, this);

      return;
    }
  }
}

// a variation of sampling_graph_h()

void HostSampler::worker_with_resampling_func() {
  ABORT_IF_T(!this->count_resampling, "if not resampling, use worker_func");

  while (this->batch_idx.load() < this->total_batch_n) {
    if (this->q_len.load() < this->kMaxQLen) {
      std::unique_ptr<Batch> b_ptr(new Batch);
      b_ptr->resampling = {0.0, 0.0};

      size_t n_source = this->batch_size;
      if (this->batch_idx.load() + 1 == this->total_batch_n) {
        n_source = this->last_batch_size;
      }
      int64_t* source = this->nodes_ptr + this->batch_idx * this->batch_size;
      int64_t* gideg = this->gideg_ptr  + this->batch_idx * this->batch_size;

      for (auto it = this->fanouts.rbegin(); it != this->fanouts.rend(); ++it) {
        // LOG(DEBUG) << "before_, n_source=" << n_source;
        this->batch_append_sampled_coo(b_ptr.get(), *it, n_source, source,
                                       gideg);

        // LOG(DEBUG) << "after_, n_source=" << n_source;
        auto& last = b_ptr->input_nodes.back();
        n_source = last.size();
        source = last.data();
	gideg = nullptr;
      }
      {
        lock_guard guard(&this->q_lock);
        this->q.push(std::move(b_ptr));
        this->batch_idx++;
        this->q_len++;
      }
    }
  }
}

void HostSampler::ReinitWithGidegFrom(pybind11::array_t<int64_t> source_,
                                      pybind11::array_t<int64_t> gideg_,
                                      size_t batch_size_, size_t use_batch_n) {
  ABORT_IF_T(!this->count_resampling, "if not resampling, use ReinitFrom");
  if (this->worker.joinable()) {
    this->worker.join();
  }

  while (true) {
    lock_guard guard(&q_lock);
    if (q.empty()) {
      pybind11::buffer_info info = source_.request();
      const size_t num = info.shape[0];
      this->nodes_ptr = static_cast<int64_t*>(info.ptr);

      info = gideg_.request();
      const size_t num2 = info.shape[0];
      this->gideg_ptr = static_cast<int64_t*>(info.ptr);

      ABORT_IF_T(num != num2, "source_ and gideg_ should have the same shape");

      this->batch_size = batch_size_;
      this->batch_idx.store(0);
      //this->total_batch_n = (num + batch_size_ - 1) / batch_size_;
      this->total_batch_n = use_batch_n;
      this->q_len.store(0);
      auto md = num % this->batch_size;
      this->last_batch_size = md == 0 ? this->batch_size : md;

      this->worker =
          std::thread(&HostSampler::worker_with_resampling_func, this);

      return;
    }
  }
}

Batch HostSampler::NextBatch() {
  // if batch_idx.load() == total_batch_n && q_len.load() == 0
  //    return
  while (true) {
    if (q_len.load() > 0) {
      lock_guard guard(&q_lock);
      std::unique_ptr<Batch> b_ptr = std::move(q.front());
      q.pop();
      q_len--;
      return *b_ptr;
    }
  }
}

BatchPy HostSampler::NextBatchPy() {
  // if batch_idx.load() == total_batch_n && q_len.load() == 0
  //    return
  while (true) {
    if (q_len.load() > 0) {
      lock_guard guard(&q_lock);
      std::unique_ptr<Batch> b_ptr = std::move(q.front());
      q.pop();
      q_len--;

      std::vector<pybind11::array_t<int64_t>> py_layers;
      pybind11::array_t<float> correct;
      for (auto& vec : b_ptr->layers) {
        py_layers.push_back(vector_to_pyarray(vec));
      }
      if (this->count_resampling) {
        correct = vector_to_pyarray(b_ptr->resampling);
      }
      // for (auto& vec : b_ptr->input_nodes) {
      //   py_input_nodes.push_back(vector_to_pyarray(vec));
      // }
      // return std::make_tuple(std::move(py_layers),
      // std::move(py_input_nodes));
      return std::make_tuple(py_layers, correct);
    }
  }
}

HostNeighborSampler::HostNeighborSampler(HostGraph* g_, const bool repeated_)
    : g(g_), repeated(repeated_) {}

BatchPy HostNeighborSampler::NextBatchPy(pybind11 ::array_t<int64_t> seeds,
                                         size_t fanout) {
  pybind11::buffer_info info = seeds.request();
  const size_t n_source = info.shape[0];
  const int64_t* source = static_cast<int64_t*>(info.ptr);

  coo_h<int64_t> ret =
      sampling_graph_h(this->g->csc, fanout, this->repeated, n_source, source);

  std::vector<pybind11::array_t<int64_t>> py_layers;
  pybind11::array_t<float> dum;
  py_layers.push_back(vector_to_pyarray(ret.row));
  py_layers.push_back(vector_to_pyarray(ret.col));
  return std::make_tuple(py_layers, dum);
}

csc_h<int64_t> py_build_csc_h(const pybind11::array_t<int64_t> u,
                              const pybind11::array_t<int64_t> v) {
  pybind11::buffer_info info = u.request();
  const size_t n_e = info.shape[0];
  const int64_t* ptr_u = static_cast<int64_t*>(u.request().ptr);
  const int64_t* ptr_v = static_cast<int64_t*>(v.request().ptr);

  return build_csc_h<int64_t>(n_e, ptr_u, ptr_v);
}

coo_h<int64_t> sampling_graph_h(
    const csc_h<int64_t>& in, const size_t fanout, const bool repeated,
    const pybind11::array_t<int64_t> source,
    const pybind11::array_t<int64_t> source_gideg,
    const pybind11::array_t<int64_t> total_resample) {
  coo_h<int64_t> ret;
  return ret;
}

}  // namespace pygraph

}  // namespace t10n
