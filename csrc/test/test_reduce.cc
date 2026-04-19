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
#include "csrc/base/util.h"

INIT_LOG

int main() {
  const size_t arr_len = 100;
  t10n::chunk<int64_t> test_arr(arr_len);
  for (size_t i = 0; i < arr_len; i++)
    test_arr[i] = i + 1;
  t10n::chunk<int64_t> inclusive_scan = t10n::inclusive_scan(test_arr);
  t10n::chunk<int64_t> exclusive_scan = t10n::exclusive_scan(test_arr);
  LOG(INFO) << "inclusive_scan";
  for (auto e : inclusive_scan) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  LOG(INFO) << "exclusive_scan";
  for (auto e : exclusive_scan) {
    std::cout << e << " ";
  }
  std::cout << std::endl;

  // test find_max
  for (size_t i = 0; i < arr_len; i++)
    test_arr[i] = (i % 7) * 13;
  int64_t max = t10n::find_max(test_arr);
  ABORT_IF_T(max != 78, "test_max fail");
}
