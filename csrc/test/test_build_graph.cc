/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#include "csrc/base/log.h"
#include "csrc/graph/build.h"
#include "csrc/graph/graph.h"

INIT_LOG

int main() {
  std::vector<int64_t> u{1, 2, 3, 4, 5, 100};
  std::vector<int64_t> v{10, 12, 200, 24, 200, 120};
  const int64_t* pu = static_cast<int64_t*>(u.data());
  const int64_t* pv = static_cast<int64_t*>(v.data());

  t10n::csc_h<int64_t> graph = t10n::build_csc_h(u.size(), pu, pv);

  t10n::print_csc_h<int64_t>(graph);
}
