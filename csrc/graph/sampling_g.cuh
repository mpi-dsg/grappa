/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#ifndef CSRC_GRAPH_SAMPLING_G_CUH_
#define CSRC_GRAPH_SAMPLING_G_CUH_

#include <curand_kernel.h>

#include <algorithm>

#include "csrc/base/glauncher.cuh"
#include "csrc/base/gutil.cuh"
#include "csrc/base/random.h"
#include "csrc/graph/graph.h"

namespace t10n {

template <typename IdxT>
HOST coo_g<IdxT> sampling_graph_g(csc_g<IdxT>* in, const size_t fanout,
                                  const bool repeated, const size_t n_source,
                                  const IdxT* source_g_ptr) {
  gchunk<size_t> pick_n_i_nbr(n_source);
  gchunk<size_t> i_nbr_offset(n_source + 1);
  gchunk<size_t> source_i_deg(n_source);
  gchunk_fill<size_t>(&pick_n_i_nbr, 0);
  gchunk_fill<size_t>(&i_nbr_offset, 0);
  gchunk_fill<size_t>(&source_i_deg, 0);

  LaunchKernel(
      n_source,
      [fanout, repeated, n_source, source_g_ptr] DEVICE(
          IdxT * csc_idx_ptr, IdxT * csc_idx, size_t * pick_n,
          size_t * src_i_deg) {
        LANE_KERNEL_1D_LOOP(idx, n_source) {
          const IdxT& nid = source_g_ptr[idx];
          const IdxT& i_deg = csc_idx_ptr[nid + 1] - csc_idx_ptr[nid];
          src_i_deg[idx] = i_deg;
          if (i_deg < fanout && repeated) {
            pick_n[idx] = (i_deg == 0 ? 0 : fanout);
          } else if (i_deg < fanout && !repeated) {
            pick_n[idx] = i_deg;
          } else {  // i_deg >= fanout, 0 is ok
            pick_n[idx] = fanout;
          }
        }
      },
      in->gidx_ptr.ptr(), in->gidx.ptr(), pick_n_i_nbr.ptr(),
      source_i_deg.ptr());

  exclusive_scan(&pick_n_i_nbr, &i_nbr_offset);

  size_t n_e = read_value<size_t>(&i_nbr_offset, i_nbr_offset.size() - 1);
  coo_g<IdxT> ret(static_cast<IdxT>(n_e));  // csc_h looks more natural

  // fill random number to dest
  ABORT_IF_T(!repeated, "repeated=false is not supported");
  uint64_t seed = uniform_pcg32<uint64_t>(0, 1 << 16);
  LaunchKernel(
      n_e,
      [n_e, seed] DEVICE(IdxT * rand_dest) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed + TID_1D, TID_1D, 0, &state);
        LANE_KERNEL_1D_LOOP(idx, n_e) { rand_dest[idx] = curand(&state); }
      },
      ret.row.ptr());

  // construct coo
  ABORT_IF_T(!repeated, "repeated=false is not supported");
  LaunchKernel(
      n_source,
      [n_source, source_g_ptr] DEVICE(
          IdxT * ret_row, IdxT * ret_col, size_t * nbr_offset, size_t * pick_n,
          size_t * src_i_deg, IdxT * csc_idx_ptr, IdxT * csc_idx) {
        LANE_KERNEL_1D_LOOP(idx, n_source) {
          const IdxT& nid = source_g_ptr[idx];
          // nid will sample pick_n_i_nbr[i] nbrs
          // and nbrs will be filled at ret.row[nid_offset...]
          const size_t& nid_offset = nbr_offset[idx];
          const size_t& p_n = pick_n[idx];
          const size_t& deg = src_i_deg[idx];

          for (int off = nid_offset; off < nid_offset + p_n; off++) {
            ret_col[off] = nid;
            // map random numbers to neighbor offset
            const IdxT& nbr_off = ret_row[off] % deg;
            const IdxT& orig_off = csc_idx_ptr[nid] + nbr_off;
            ret_row[off] = csc_idx[orig_off];
          }
        }
      },
      ret.row.ptr(), ret.col.ptr(), i_nbr_offset.ptr(), pick_n_i_nbr.ptr(),
      source_i_deg.ptr(), in->gidx_ptr.ptr(), in->gidx.ptr());

  IdxT m1 = find_max<IdxT>(&(ret.row));
  IdxT m2 = find_max<IdxT>(&(ret.col));
  ret.n_n = std::max(m1, m2);
  return ret;

  /*
    chunk<size_t> pick_n_i_nbr_h(n_source);
    chunk<size_t> i_nbr_offset_h(n_source + 1);
    chunk<size_t> source_i_deg_h(n_source);

    copy_async_to(&pick_n_i_nbr_h, &pick_n_i_nbr);
    copy_async_to(&source_i_deg_h, &source_i_deg);
    copy_async_to(&i_nbr_offset_h, &i_nbr_offset);

    print_coo_g(ret);

    LOG(INFO) << "pick_n_i_nbr_h";
    for (auto e : pick_n_i_nbr_h) {
      LOG(INFO) << e;
    }
    LOG(INFO) << "source_i_deg_h";
    for (auto e : source_i_deg_h) {
      LOG(INFO) << e;
    }

    LOG(INFO) << "i_nbr_offset_h";
    for (auto e : i_nbr_offset_h) {
      LOG(INFO) << e;
    }
  */
}

}  // namespace t10n

#endif  // CSRC_GRAPH_SAMPLING_G_CUH_
