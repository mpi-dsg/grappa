/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#ifndef CSRC_GRAPH_GRAPH_H_
#define CSRC_GRAPH_GRAPH_H_

#include <algorithm>
#include <limits>
#include <utility>  // std::move

#include "csrc/base/chunk.h"
#include "csrc/base/gchunk.h"
#include "csrc/base/log.h"

namespace t10n {

template <typename IdxT>
struct coo_h {
  chunk<IdxT> row;
  chunk<IdxT> col;
  size_t n_e;
  size_t n_n;
  coo_h() {}
  coo_h(size_t n_n_, size_t n_e_)
      : n_n(n_n_),
        n_e(n_e_),
        row(std::move(chunk<IdxT>(n_e_))),
        col(std::move(chunk<IdxT>(n_e_))) {}
  explicit coo_h(size_t n_e_)
      : n_n(std::numeric_limits<IdxT>::max()),
        n_e(n_e_),
        row(std::move(chunk<IdxT>(n_e_))),
        col(std::move(chunk<IdxT>(n_e_))) {}
  coo_h(coo_h&& other)
      : row(std::move(other.row)),
        col(std::move(other.col)),
        n_e(other.n_e),
        n_n(other.n_n) {  // move constructor
  }
};

template <typename IdxT>
struct csc_h {
  chunk<IdxT> hidx_ptr;
  chunk<IdxT> hidx;
  size_t n_e;
  size_t n_n;
  csc_h() {}
  csc_h(size_t n_n_, size_t n_e_)
      : n_n(n_n_),
        n_e(n_e_),
        hidx_ptr(std::move(chunk<IdxT>(n_n_ + 1))),
        hidx(std::move(chunk<IdxT>(n_e_))) {}
  csc_h(csc_h&& other)
      : hidx_ptr(std::move(other.hidx)),
        hidx(std::move(other.hidx)),
        n_e(other.n_e),
        n_n(other.n_n) {  // move constructor
  }
  void copy(const csc_h& other) {
    n_n = other.n_n;
    n_e = other.n_e;
    hidx_ptr.resize(other.hidx_ptr.size());
    hidx.resize(other.hidx.size());
    std::copy(other.hidx_ptr.begin(), other.hidx_ptr.end(), hidx_ptr.begin());
    std::copy(other.hidx.begin(), other.hidx.end(), hidx.begin());
  }
};

template <typename IdxT>
struct coo_g {
  gchunk<IdxT> row;
  gchunk<IdxT> col;
  size_t n_e;
  size_t n_n;
  coo_g() {}
  coo_g(size_t n_n_, size_t n_e_)
      : n_n(n_n_),
        n_e(n_e_),
        row(std::move(gchunk<IdxT>(n_e_))),
        col(std::move(gchunk<IdxT>(n_e_))) {}
  explicit coo_g(size_t n_e_)
      : n_n(std::numeric_limits<IdxT>::max()),
        n_e(n_e_),
        row(std::move(gchunk<IdxT>(n_e_))),
        col(std::move(gchunk<IdxT>(n_e_))) {}

  coo_g(coo_g&& other)
      : row(std::move(other.row)),
        col(std::move(other.col)),
        n_e(other.n_e),
        n_n(other.n_n) {  // move constructor
  }
};

template <typename IdxT>
struct csc_g {
  size_t n_n;
  size_t n_e;
  gchunk<IdxT> gidx_ptr;
  gchunk<IdxT> gidx;
  csc_g() {}
  csc_g(size_t n_n_, size_t n_e_)
      : n_n(n_n_),
        n_e(n_e_),
        gidx_ptr(std::move(gchunk<IdxT>(n_n_ + 1))),
        gidx(std::move(gchunk<IdxT>(n_e_))) {}
  csc_g(csc_g&& other)
      : n_n(other.n_n),
        n_e(other.n_e),
        gidx_ptr(std::move(other.gidx_ptr)),
        gidx(std::move(other.gidx)) {  // move constructor
  }
};

template <typename IdxT>
inline csc_g<IdxT> move_to_gpu(csc_h<IdxT>* src) {
  csc_g<IdxT> dest(src->n_n, src->n_e);
  copy_async_to<IdxT, IdxT>(&dest.gidx_ptr, &src->hidx_ptr);
  copy_async_to<IdxT, IdxT>(&dest.gidx, &src->hidx);
  return dest;
}

template <typename IdxT>
inline coo_g<IdxT> move_to_gpu(coo_h<IdxT>* src) {
  coo_g<IdxT> dest(src->n_n, src->n_e);
  copy_async_to<IdxT, IdxT>(&dest.row, &src->row);
  copy_async_to<IdxT, IdxT>(&dest.col, &src->col);
  return dest;
}

template <typename IdxT>
inline csc_h<IdxT> move_to_host(csc_g<IdxT>* src) {
  csc_h<IdxT> dest(src->n_n, src->n_e);
  copy_async_to<IdxT, IdxT>(&dest.hidx_ptr, &src->gidx_ptr);
  copy_async_to<IdxT, IdxT>(&dest.hidx, &src->gidx);
  return dest;
}

template <typename IdxT>
inline coo_h<IdxT> move_to_host(coo_g<IdxT>* src) {
  coo_h<IdxT> dest(src->n_n, src->n_e);
  copy_async_to<IdxT, IdxT>(&dest.row, &src->row);
  copy_async_to<IdxT, IdxT>(&dest.col, &src->col);
  return dest;
}

template <typename IdxT>
void print_csc_h(const csc_h<IdxT>& in) {
  LOG(INFO) << __func__;
  LOG(INFO) << "n_n = " << in.n_n << " == " << in.hidx_ptr.size() - 1;
  LOG(INFO) << "n_e = " << in.n_e << " == " << in.hidx.size();

  size_t counter = 0;
  for (size_t col = 0; col < in.n_n; col++) {
    for (size_t idx = in.hidx_ptr[col]; idx < in.hidx_ptr[col + 1]; idx++) {
      if (++counter > 3) {
        break;
      }
      LOG(INFO) << "(u, v) = " << in.hidx[idx] << ", " << col;
    }
  }
}

template <typename IdxT>
void print_coo_h(const coo_h<IdxT>& in) {
  LOG(INFO) << __func__;
  LOG(INFO) << "n_n = " << in.n_n;
  LOG(INFO) << "n_e = " << in.n_e << " == " << in.row.size();

  for (size_t idx = 0; idx < in.n_e; idx++) {
    LOG(INFO) << "(u, v) = " << in.row[idx] << ", " << in.col[idx];
  }
}

template <typename IdxT>
void print_coo_g(const coo_g<IdxT>& in) {
  coo_h<IdxT> h = move_to_host(&const_cast<coo_g<IdxT>&>(in));
  LOG(INFO) << __func__;
  print_coo_h(h);
}

}  // namespace t10n

#endif  // CSRC_GRAPH_GRAPH_H_
