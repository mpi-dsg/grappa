/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#ifndef CSRC_BASE_UTIL_H_
#define CSRC_BASE_UTIL_H_

#include <omp.h>  // omp_get_thread_num
#include <algorithm>
#include <atomic>
#include "csrc/base/chunk.h"
#include "csrc/base/config.h"
#include "csrc/base/error.h"

namespace t10n {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
template <typename Tp>
inline chunk<Tp> inclusive_scan(const chunk<Tp>& in) {
  ABORT_IF_T(in.size() < 1, "in.size() > 0 is required");
  chunk<Tp> ret(in.size(), 0);

  Tp r = 0;
#pragma omp parallel for reduction(inscan, + : r)
  for (size_t i = 0; i < in.size(); i++) {
    r += in[i];
#pragma omp scan inclusive(r)
    ret[i] = r;
  }
  return ret;
}

template <typename Tp>
inline chunk<Tp> exclusive_scan(const chunk<Tp>& in) {
  ABORT_IF_T(in.size() < 1, "in.size() > 0 is required");
  chunk<Tp> ret(in.size() + 1, 0);

  Tp r = 0;
#pragma omp parallel for reduction(inscan, + : r)
  for (size_t i = 0; i < in.size(); i++) {
    r += in[i];
#pragma omp scan inclusive(r)
    ret[i + 1] = r;
  }
  return ret;
}
#pragma GCC diagnostic pop

template <typename Tp>
inline Tp find_max(const Tp* data, const size_t len) {
  ABORT_IF_T(len < 1, "len > 0 is required");
  Tp max_val = data[0];

#pragma omp parallel for reduction(max : max_val)
  for (size_t idx = 0; idx < len; ++idx) {
    max_val = std::max(max_val, data[idx]);
  }
  return max_val;
}

template <typename Tp>
inline Tp find_max(const chunk<Tp> in) {
  const Tp* data = static_cast<const Tp*>(in.data());
  size_t len = in.size();
  return find_max(data, len);
}

struct tas_lock {
  std::atomic<bool> lock_ = {false};

  void lock() {
    while (lock_.exchange(true, std::memory_order_acquire)) {}
  }

  void unlock() { lock_.store(false, std::memory_order_release); }
};

struct lock_guard {
  explicit lock_guard(tas_lock* lock_) {
    lock = lock_;
    lock->lock();
  }
  ~lock_guard() { lock->unlock(); }

 private:
  tas_lock* lock;
};

}  // namespace t10n

#endif  // CSRC_BASE_UTIL_H_
