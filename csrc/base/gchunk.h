/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_GCHUNK_H_
#define CSRC_BASE_GCHUNK_H_

#include <cuda_runtime.h>

#include "csrc/base/chunk.h"
#include "csrc/base/galloc.h"
#include "csrc/base/gerror.h"

namespace t10n {

template <class Tp, class Alloc = galloc<Tp>>
class gchunk {
 public:
  // try to be STL compatible
  typedef Tp value_type;
  typedef value_type* pointer;
  typedef value_type* iterator;
  typedef value_type& reference;
  typedef size_t size_type;

 private:
  iterator begin_;
  iterator end_;

 public:
  void* data() { return static_cast<void*>(begin_); }
  Tp* ptr() { return static_cast<Tp*>(begin_); }
  iterator begin() { return begin_; }
  iterator end() { return end_; }
  size_t size() const { return static_cast<size_type>(end_ - begin_); }
  bool empty() const { return begin_ == end_; }
  reference operator[](size_type n) { return *(begin() + n); }
  reference front() { return *begin_; }
  reference back() { return *end_; }
  gchunk() : begin_(nullptr), end_(nullptr) {}
  explicit gchunk(size_type n) {
    begin_ = Alloc::allocate(n);
    end_ = begin_ + n;
  }
  gchunk(gchunk&& other)
      : begin_(std::exchange(other.begin_, nullptr)),
        end_(std::exchange(other.end_, nullptr)) {}

  ~gchunk() {
    if (begin_)
      Alloc::deallocate(begin_, (end_ - begin_));
  }
};

template <typename U, typename V>
inline void copy_async_to(chunk<U>* h, gchunk<V>* d,
                          const cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpyAsync(h->data(), d->data(), sizeof(U) * h->size(),
                             cudaMemcpyDeviceToHost, stream));
}

template <typename U, typename V>
inline void copy_async_to(gchunk<U>* d, chunk<V>* h,
                          const cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpyAsync(d->data(), h->data(), sizeof(U) * d->size(),
                             cudaMemcpyHostToDevice, stream));
}

template <typename U, typename V>
inline void copy_sync_to(chunk<U>* h, gchunk<V>* d,
                         const cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpy(h->data(), d->data(), sizeof(U) * h->size(),
                        cudaMemcpyDeviceToHost, stream));
}

template <typename U, typename V>
inline void copy_sync_to(gchunk<U>* d, chunk<V>* h,
                         const cudaStream_t stream = 0) {
  CUDA_CHECK(cudaMemcpy(d->data(), h->data(), sizeof(U) * d->size(),
                        cudaMemcpyHostToDevice, stream));
}

}  // namespace t10n
#endif  // CSRC_BASE_GCHUNK_H_
