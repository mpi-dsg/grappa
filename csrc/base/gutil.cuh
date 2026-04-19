/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#ifndef CSRC_BASE_GUTIL_CUH_
#define CSRC_BASE_GUTIL_CUH_

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "csrc/base/error.h"
#include "csrc/base/galloc.h"
#include "csrc/base/gchunk.h"
#include "csrc/base/glauncher.cuh"

namespace t10n {

namespace internal {
struct gutil_max {
  template <typename T>
  DEVICE_INLINE T operator()(const T& a, const T& b) const {
    return (b < a) ? a : b;
  }
};
}  // namespace internal
template <typename Tp>
HOST_INLINE void gchunk_fill(gchunk<Tp>* g, Tp v) {
  t10n::LaunchKernel(
      g->size(),
      [] DEVICE(Tp * pa, Tp val, size_t len) {
        LANE_KERNEL_1D_LOOP(i, len) { pa[i] = val; }
      },
      g->ptr(), v, g->size());
}

template <typename IdxT>
HOST_INLINE void exclusive_scan(gchunk<IdxT>* in, gchunk<IdxT>* out) {
  ABORT_IF_T(out->size() != in->size() + 1,
             "exclusive_scan result has 1 more element than input");
  void* d_temp = NULL;
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, in->ptr(),
                                           out->ptr(), out->size()));

  gchunk<int8_t> temp(temp_bytes);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp.data(), temp_bytes, in->ptr(),
                                           out->ptr(), out->size()));
}

template <typename IdxT>
HOST_INLINE IdxT read_value(gchunk<IdxT>* in, size_t pos,
                            const cudaStream_t stream = 0) {
  IdxT ret = 0;
  CUDA_CHECK(cudaMemcpyAsync(&ret, (in->ptr() + pos), sizeof(IdxT),
                             cudaMemcpyDeviceToHost, stream));
  return ret;
}

template <typename IdxT>
HOST_INLINE IdxT find_max(gchunk<IdxT>* in) {
  gchunk<IdxT> ret(1);
  internal::gutil_max max_op;
  void* d_temp = NULL;
  size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp, temp_bytes, in->ptr(), ret.ptr(),
                                       in->size(), max_op, 0));

  gchunk<int8_t> temp(temp_bytes);
  CUDA_CHECK(cub::DeviceReduce::Reduce(temp.data(), temp_bytes, in->ptr(),
                                       ret.ptr(), in->size(), max_op, 0));
  return read_value(&ret, 0);
}

}  // namespace t10n
#endif  // CSRC_BASE_GUTIL_CUH_
