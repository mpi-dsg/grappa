/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#ifndef CSRC_BASE_ERROR_H_
#define CSRC_BASE_ERROR_H_

#include <exception>  // for std::exception
#include <string>     // for std::string
#include <utility>    // for std::move

#include "csrc/base/log.h"

namespace t10n {

enum class ErrorCode : unsigned char { kOK = 0, kOOM = 10, kUnknown = 255 };

struct Error {
  Error() : c(ErrorCode::kOK), msg("") {}
  Error(const Error& s) noexcept : c(s.c), msg(s.msg) {}
  Error(Error&& s) noexcept : c(std::move(s.c)), msg(std::move(s.msg)) {}
  Error(ErrorCode c_, const std::string& message) : c(c_), msg(message) {}
  ~Error() {}

  bool ok() const { return (c == ErrorCode::kOK); }
  ErrorCode code() const { return c; }

  std::string ToString() {
    switch (c) {
    case ErrorCode::kOK:
      return "OK";
    case ErrorCode::kOOM:
      return "OOM";
    case ErrorCode::kUnknown:
      return "Unknown";
    }
    return "Unknown";
  }

  inline static Error OK() { return Error(); }
  static Error OOM(std::string msg = "") { return Error(ErrorCode::kOOM, msg); }
  static Error Unknown(std::string msg = "") {
    return Error(ErrorCode::kUnknown, msg);
  }

 private:
  ErrorCode c;
  std::string msg;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#define RET_IF_ERR(s)             \
  do {                            \
    if (!s.ok()) {                \
      LOG(ERROR) << s.ToString(); \
      return;                     \
    }                             \
  } while (0)

// TODO(t10n) Error is only valid under t10n::
#define RET_V_IF_ERR(v, s)        \
  do {                            \
    if (!s.ok()) {                \
      LOG(ERROR) << s.ToString(); \
      return (v);                 \
    }                             \
  } while (0)

#define ABORT_IF_ERR(s)           \
  do {                            \
    if (!s.ok()) {                \
      LOG(ERROR) << s.ToString(); \
      exit(1);                    \
    }                             \
  } while (0)

#define RET_IF_T(cond, msg) \
  do {                      \
    if ((cond)) {           \
      LOG(ERROR) << #msg;   \
      return;               \
    }                       \
  } while (0)

#define RET_V_IF_T(v, cond, msg) \
  do {                           \
    if ((cond)) {                \
      LOG(ERROR) << #msg;        \
      return (v);                \
    }                            \
  } while (0)

#define ABORT_IF_T(cond, msg) \
  do {                        \
    if ((cond)) {             \
      LOG(ERROR) << #msg;     \
      exit(1);                \
    }                         \
  } while (0)
#pragma GCC diagnostic pop

}  // namespace t10n

#endif  // CSRC_BASE_ERROR_H_
