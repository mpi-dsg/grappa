/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#include <iostream>

#include "csrc/base/chunk.h"
#include "csrc/base/error.h"
#include "csrc/base/log.h"
#include "csrc/base/random.h"
#include "csrc/base/util.h"

INIT_LOG

void tes_fill_n_rand_of_range(const size_t n, const int64_t& low,
                              const int64_t& up, bool repeated,
                              bool print_all) {
  std::vector<int64_t> buf(n, 0);
  t10n::fill_n_rand_of_range<int64_t>(buf.data(), n, low, up, repeated);
  if (print_all) {
    for (auto e : buf) {
      LOG(INFO) << e;
    }
  }
  LOG(INFO) << "test fill_n_rand_of_range:pick " << n << " [" << low << ", "
            << up << "], repeated=" << repeated;
}

int main() {
  const size_t kSample = 1 << 3;
#pragma omp parallel
  {
    // int tid = omp_get_thread_num();
    for (size_t idx = 0; idx < kSample; idx++) {
      // LOG(INFO) << "tid = " << tid << ", idx = " << idx
      //           << ", rand_std=" << t10n::uniform_std<int64_t>(0, 30);
      t10n::uniform_std<int64_t>(0, 30);
    }
  }

#pragma omp parallel
  {
    // int tid = omp_get_thread_num();
    for (size_t idx = 0; idx < kSample; idx++) {
      // LOG(INFO) << "tid = " << tid << ", idx = " << idx
      //           << ", rand_pcg=" << t10n::uniform_pcg32<int64_t>(0, 30);
      t10n::uniform_std<int64_t>(0, 30);
    }
  }

  bool print_all = true;

  for (int i = 0; i < 10; i++) {
    tes_fill_n_rand_of_range(3, 1, 4, false, print_all);
  }
  for (int i = 0; i < 10; i++) {
    tes_fill_n_rand_of_range(3, 1, 3, false, print_all);
  }

  tes_fill_n_rand_of_range(0, 0, 0, true, print_all);
  tes_fill_n_rand_of_range(0, 0, 0, false, print_all);
  tes_fill_n_rand_of_range(5, 1, 10, false, print_all);
  tes_fill_n_rand_of_range(5, 1, 10, true, print_all);
  // tes_fill_n_rand_of_range(10, 1, 5, false, print_all);
  tes_fill_n_rand_of_range(10, 1, 5, true, print_all);
  tes_fill_n_rand_of_range(5, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(5, 1, 1 << 16, false, print_all);

  print_all = false;
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, false, print_all);
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, false, print_all);
  tes_fill_n_rand_of_range(1 << 5, 1, 1 << 16, false, print_all);

  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, true, print_all);
  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, false, print_all);
  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, false, print_all);
  tes_fill_n_rand_of_range(1 << 15, 1, 1 << 16, false, print_all);
}
