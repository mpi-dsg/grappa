/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#include "csrc/base/error.h"
#include "csrc/base/gchunk.h"
#include "csrc/base/glauncher.cuh"
#include "csrc/base/gutil.cuh"
#include "csrc/base/log.h"

INIT_LOG

int main() {
  const int64_t vec_len = 102400;
  t10n::gchunk<int64_t> a(vec_len);
  t10n::gchunk<int64_t> b(vec_len);
  t10n::gchunk<int64_t> c(vec_len);
  t10n::gchunk_fill(&a, 1l);
  t10n::gchunk_fill(&b, 2l);
  t10n::gchunk_fill(&c, 0l);

  t10n::LaunchKernel(
      vec_len,
      [] DEVICE(int64_t * pa, int64_t * pb, int64_t * pc, int64_t len) {
        LANE_KERNEL_1D_LOOP(i, len) { pc[i] = pa[i] + pb[i]; }
      },
      a.ptr(), b.ptr(), c.ptr(), vec_len);

  t10n::chunk<int64_t> res(vec_len, 0);
  t10n::copy_async_to(&res, &c);

  for (int i = 1; i < 5; ++i) {
    ABORT_IF_T(res[i] != 3, "test cuda failed");
  }
}
