/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#include "csrc/base/chunk.h"
#include "csrc/base/gchunk.h"
#include "csrc/base/log.h"
#include "csrc/graph/build.h"
#include "csrc/graph/graph.h"
#include "csrc/graph/sampling_g.cuh"
#include "csrc/graph/sampling_h.h"

INIT_LOG

void test(t10n::csc_g<int64_t>* input, t10n::chunk<int64_t>* source,
          const size_t fanout, const bool repeated) {
  LOG(INFO) << __FILE__ << " " << __func__;
  LOG(INFO) << "repeated:" << repeated;
  LOG(INFO) << "fanout:" << fanout;
  LOG(INFO) << "source:";
  for (auto e : *source) {
    LOG(INFO) << e;
  }
  const size_t n_source = source->size();
  t10n::gchunk<int64_t> seeds_g(n_source);
  t10n::copy_async_to(&seeds_g, source);

  t10n::coo_g<int64_t> subg_coo = t10n::sampling_graph_g<int64_t>(
      input, fanout, repeated, n_source, seeds_g.ptr());
  t10n::print_coo_g(subg_coo);
}

int main() {
  std::vector<int64_t> u{1, 2, 3, 4, 5, 100, 101, 101};
  std::vector<int64_t> v{10, 12, 200, 24, 200, 120, 200, 300};
  const int64_t* pu = static_cast<int64_t*>(u.data());
  const int64_t* pv = static_cast<int64_t*>(v.data());

  t10n::csc_h<int64_t> graph = t10n::build_csc_h(u.size(), pu, pv);
  t10n::csc_g<int64_t> csc_g = t10n::move_to_gpu(&graph);
  t10n::csc_h<int64_t> graph2 = t10n::move_to_host(&csc_g);

  t10n::print_csc_h<int64_t>(graph);
  t10n::print_csc_h<int64_t>(graph2);

  bool repeated = true;
  // t10n::chunk<int64_t> seeds{};
  // test(&csc_g, seeds, 0, repeated);
  // test(&csc_g, seeds, 3, repeated);

  t10n::chunk<int64_t> seeds2{10, 100};
  test(&csc_g, &seeds2, 0, repeated);
  test(&csc_g, &seeds2, 3, repeated);

  t10n::chunk<int64_t> seeds3{10, 100, 200};
  test(&csc_g, &seeds3, 0, repeated);
  test(&csc_g, &seeds3, 3, repeated);

  repeated = false;
  test(&csc_g, &seeds2, 2, repeated);
}
