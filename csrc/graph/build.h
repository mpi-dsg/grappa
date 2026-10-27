/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#ifndef CSRC_GRAPH_BUILD_H_
#define CSRC_GRAPH_BUILD_H_

#include <omp.h>

#include <parallel_hashmap/phmap.h>
#include <algorithm>
#include <vector>

#include "csrc/base/config.h"
#include "csrc/base/util.h"
#include "csrc/graph/graph.h"

namespace t10n {

template <typename IdxT>
csc_h<IdxT> build_csc_h(const size_t n_e, const IdxT* u, const IdxT* v) {
  csc_h<IdxT> ret;

  IdxT max_u = find_max<IdxT>(u, n_e);
  IdxT max_v = find_max<IdxT>(v, n_e);
  IdxT max_id = std::max(max_u, max_v);

  ret.n_e = n_e;
  ret.n_n = max_id + 1;
  // ret.hidx_ptr = chunk<IdxT>(ret.n_n + 1, 0);
  ret.hidx = chunk<IdxT>(ret.n_e, 0);

  std::vector<phmap::parallel_flat_hash_set<IdxT>> id2inbr(ret.n_n);
  for (size_t i = 0; i < ret.n_n; i++) {
    id2inbr.push_back(phmap::parallel_flat_hash_set<IdxT>());
  }

  chunk<IdxT> nnbr(ret.n_n, 0);  // use IdxT as the same as hidx_ptr

#pragma omp parallel
  {
    const IdxT& tid = omp_get_thread_num();
    const int OMP_N_THREAD = omp_get_num_threads();
    for (size_t idx = 0; idx < n_e; idx++) {
      const IdxT& dst = v[idx];
      if (dst % OMP_N_THREAD != tid) {
        continue;
      }
      auto& inbr = id2inbr[dst];
      inbr.insert(u[idx]);  // de-duplication
    }
  }

#pragma omp parallel for
  for (size_t col = 0; col < ret.n_n; col++) {
    nnbr[col] = static_cast<IdxT>(id2inbr[col].size());
  }

  ret.hidx_ptr = exclusive_scan<IdxT>(nnbr);

#pragma omp parallel for
  for (size_t col = 0; col < ret.n_n; col++) {
    size_t be = ret.hidx_ptr[col];
    for (const IdxT& inbr : id2inbr[col]) {
      ret.hidx[be++] = inbr;
    }
  }

  return ret;
}

}  // namespace t10n

#endif  // CSRC_GRAPH_BUILD_H_
