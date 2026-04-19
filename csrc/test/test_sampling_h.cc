/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#include "csrc/base/log.h"
#include "csrc/graph/build.h"
#include "csrc/graph/graph.h"
#include "csrc/graph/sampling_h.h"

INIT_LOG

void test(const t10n::csc_h<int64_t>& input, const std::vector<int64_t>& source,
          const size_t fanout, const bool repeated) {
  LOG(INFO) << __FILE__ << " " << __func__;
  LOG(INFO) << "repeated:" << repeated;
  LOG(INFO) << "fanout:" << fanout;
  LOG(INFO) << "source:";
  for (auto e : source) {
    LOG(INFO) << e;
  }
  const size_t n_source = source.size();

  t10n::coo_h<int64_t> subg_coo =
      t10n::sampling_graph_h(input, fanout, repeated, n_source, source.data());
  t10n::print_coo_h(subg_coo);
}

int main() {
  std::vector<int64_t> u{1, 2, 3, 4, 5, 100, 101, 101};
  std::vector<int64_t> v{10, 12, 200, 24, 200, 120, 200, 300};
  const int64_t* pu = static_cast<int64_t*>(u.data());
  const int64_t* pv = static_cast<int64_t*>(v.data());

  t10n::csc_h<int64_t> graph = t10n::build_csc_h(u.size(), pu, pv);

  t10n::print_csc_h<int64_t>(graph);

  bool repeated = true;
  // test(graph, {}, 0, repeated);
  // test(graph, {}, 3, repeated);
  // test(graph, {10}, 0, repeated);
  // test(graph, {10}, 3, repeated);
  // test(graph, {25}, 0, repeated);
  // test(graph, {25}, 233333333, repeated);
  // test(graph, {10, 200}, 1, repeated);
  // test(graph, {10, 200}, 2, repeated);
  test(graph, {10, 200, 100}, 0, repeated);
  test(graph, {10, 200, 100}, 2, repeated);

  repeated = false;
  // test(graph, {}, 0, repeated);
  // test(graph, {}, 3, repeated);
  // test(graph, {10}, 0, repeated);
  // test(graph, {10}, 3, repeated);
  // test(graph, {25}, 0, repeated);
  // test(graph, {25}, 233333333, repeated);
  // test(graph, {10, 200}, 1, repeated);
  // test(graph, {10, 200}, 2, repeated);
  test(graph, {10, 200, 100}, 0, repeated);
  test(graph, {10, 200, 100}, 2, repeated);
}
