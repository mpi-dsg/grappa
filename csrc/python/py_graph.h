/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#ifndef CSRC_PYTHON_PY_GRAPH_H_
#define CSRC_PYTHON_PY_GRAPH_H_

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>  // for unique_ptr<>
#include <queue>
#include <tuple>
#include <vector>

#include "csrc/base/util.h"
#include "csrc/graph/build.h"
#include "csrc/graph/graph.h"

namespace t10n {

namespace pygraph {

csc_h<int64_t> py_build_csc_h(const pybind11::array_t<int64_t> u,
                              const pybind11::array_t<int64_t> v);

class HostGraph {
 public:
  HostGraph(const pybind11::array_t<int64_t> u,
            const pybind11::array_t<int64_t> v, int node_id);

  HostGraph(const pybind11::array_t<int64_t> u,
            const pybind11::array_t<int64_t> v);

  ~HostGraph();

 public:
  csc_h<int64_t> csc;
  struct bitmask* node_mask;
};

class GpuGraph {
 public:
  explicit GpuGraph(const HostGraph& host_graph)
      : csc(move_to_gpu(&const_cast<HostGraph&>(host_graph).csc)) {
    // print_csc_h(host_graph.csc);
  }

 public:
  csc_g<int64_t> csc;
};

class __attribute__((visibility("default"))) Batch {
 public:
  std::vector<std::vector<int64_t>> get_layers() { return layers; }
  std::vector<std::vector<int64_t>> get_input_nodes() { return input_nodes; }

 public:
  // ..., u_l1, v_l1v, u_l0, v_l0
  std::vector<std::vector<int64_t>> layers;
  // ..., input_l1, input_l0
  std::vector<std::vector<int64_t>> input_nodes;

  std::vector<float> resampling;
};

/*
    BatchPy containes only Batch::layers
    if HostSampler is enabled count resampling
        BatchPy contains BatchPy+resampling_count(as py_arr)
*/

using BatchPy = std::tuple<std::vector<pybind11::array_t<int64_t>>, pybind11::array_t<float>>;

class HostSampler {
 public:
  HostSampler(HostGraph* g_, std::vector<size_t> fanouts_,
              const bool repeated_);
  HostSampler(HostGraph* g_, std::vector<size_t> fanouts_, const bool repeated_,
              const bool count_resampling);

  ~HostSampler();
  void ReinitFrom(pybind11::array_t<int64_t> source_, size_t batch_size_, size_t use_batch_n);
  void ReinitWithGidegFrom(pybind11::array_t<int64_t> source_,
                           pybind11::array_t<int64_t> gideg_,
                           size_t batch_size_, size_t use_batch_n);
  Batch NextBatch();
  BatchPy NextBatchPy();

 private:
  void batch_append_sampled_coo(Batch* b, const size_t fanout,
                                const size_t n_source, const int64_t* source,
                                const int64_t* gideg);
  void worker_func();
  void worker_with_resampling_func();

 private:
  HostGraph* g;
  std::vector<size_t> fanouts;
  const bool repeated;
  const bool count_resampling;
  int64_t* nodes_ptr;
  int64_t* gideg_ptr;
  int batch_size;
  int total_batch_n;

  std::atomic_int batch_idx;
  std::queue<std::unique_ptr<Batch>> q;
  const size_t kMaxQLen = 2;
  std::atomic_int q_len;

  tas_lock q_lock;
  std::thread worker;

  size_t last_batch_size;
};

class GpuSampler {
 public:
  explicit GpuSampler(std::vector<int> fanouts) {}
  void ReinitFrom() {}
  void NextBatch() {}
};

class HostNeighborSampler {
 public:
  explicit HostNeighborSampler(HostGraph* g_, const bool repeated_);
  ~HostNeighborSampler() {}
  BatchPy NextBatchPy(pybind11 ::array_t<int64_t> seeds, size_t fanouts);

 private:
  HostGraph* g;
  const bool repeated;
};

}  // namespace pygraph
}  // namespace t10n
#endif  // CSRC_PYTHON_PY_GRAPH_H_
