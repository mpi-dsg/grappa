/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#ifndef CSRC_GRAPH_SAMPLING_H_H_
#define CSRC_GRAPH_SAMPLING_H_H_

#include <algorithm>  // for std::max
#include <cassert>
#include <cmath>
#include <numeric>
#include <omp.h>

#include "csrc/base/random.h"  // for fill_n_rand_of_range
#include "csrc/base/util.h"    // for exclusive_scan
#include "csrc/graph/graph.h"  // fill_n_rand_of_range

namespace t10n {

template <typename IdxT>
inline void collect_pick_n_and_ideg(const csc_h<IdxT>& in, const size_t fanout,
                                    const bool repeated, const size_t n_source,
                                    const IdxT* source,
                                    chunk<size_t>* pick_n_i_nbr,
                                    chunk<size_t>* source_i_deg) {
#pragma omp parallel for
  for (size_t idx = 0; idx < n_source; idx++) {
    const IdxT& nid = source[idx];
    const IdxT& i_deg = in.hidx_ptr[nid + 1] - in.hidx_ptr[nid];
    (*source_i_deg)[idx] = i_deg;
    if (i_deg < fanout && repeated) {
      (*pick_n_i_nbr)[idx] = (i_deg == 0 ? 0 : fanout);
    } else if (i_deg < fanout && !repeated) {
      (*pick_n_i_nbr)[idx] = i_deg;
    } else {  // i_deg >= fanout, 0 is ok
      (*pick_n_i_nbr)[idx] = fanout;
    }
  }
}

template <typename IdxT>
inline void fill_sampling_result_to_coo(
    const csc_h<IdxT>& in, const bool repeated, const size_t n_source,
    const IdxT* source, chunk<size_t>* i_nbr_offset,
    chunk<size_t>* pick_n_i_nbr, chunk<size_t>* source_i_deg, IdxT* coo_row,
    IdxT* coo_col) {
#pragma omp parallel for
  for (size_t idx = 0; idx < n_source; idx++) {
    const IdxT& nid = source[idx];
    // nid will sample pick_n_i_nbr[i] nbrs
    // and nbrs will be filled at ret.row[nid_offset...]
    const size_t& nid_offset = (*i_nbr_offset)[idx];
    const size_t& pick_n = (*pick_n_i_nbr)[idx];
    const IdxT& up = (*source_i_deg)[idx] - 1;  // if up < 0, pick_n should == 0
    fill_n_rand_of_range<IdxT>(coo_row + nid_offset, pick_n, 0, up, repeated);

    // construct coo of nid's part
    for (size_t off = nid_offset; off < nid_offset + pick_n; off++) {
      // fill back orig ids of nid's sampled in nbrs
      const size_t& orig_off = coo_row[off] + in.hidx_ptr[nid];
      coo_row[off] = in.hidx[orig_off];
      coo_col[off] = nid;
    }
  }
}

template <typename IdxT>
inline void fill_w_reampling_sampling_result_to_coo(
    const csc_h<IdxT>& in, const bool repeated, const size_t n_source,
    const IdxT* source, const IdxT* source_g_ideg, chunk<size_t>* i_nbr_offset,
    chunk<size_t>* pick_n_i_nbr, chunk<size_t>* source_i_deg, IdxT* coo_row,
    IdxT* coo_col, float* resampling_count) {
    if(n_source < 1){
      *resampling_count = 0.0;
      return;
    }
  //  printf("source: %x, source_g_ideg: %x, i_nbr_offset: %x, pick_n_i_nbr: %x, source_i_deg: %x, coo_row: %x, coo_col: %x, resampling_count: %x\n",
//		    source, source_g_ideg, i_nbr_offset, pick_n_i_nbr, source_i_deg, coo_row, coo_col, resampling_count);
    std::vector<float> scale_factor(omp_get_max_threads() + 1, 0.0);

  //printf("tag1\n");
  #pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  //printf("tag11: tid=%d, nthreads=%d\n", tid, nthreads);
  for (size_t idx = tid; idx < n_source; idx+=nthreads) {
    const IdxT& nid = source[idx];
   // printf("tag10.1: idx=%lu\n", idx);
    // nid will sample pick_n_i_nbr[i] nbrs
    // and nbrs will be filled at ret.row[nid_offset...]
    const size_t& nid_offset = (*i_nbr_offset)[idx];
   // printf("tag10.2: idx=%lu\n", idx);
    const size_t& pick_n = (*pick_n_i_nbr)[idx];
   // printf("tag10.3: idx=%lu\n", idx);
    IdxT up = (*source_i_deg)[idx] - 1;  // if up < 0, pick_n should == 0
   // printf("tag10.4: idx=%lu\n", idx);
    IdxT super_up = source_g_ideg[idx];
  //  printf("tag10.5: idx=%lu\n", idx);
    up = up < 0 ? 0 : up;
    super_up = super_up < 0 ? 0 : super_up;
  //  printf("tag11: idx=%lu, %ld, %ld\n", idx, up, super_up);
    float tmp = (up + 1.0) / (super_up + 1.0);
    // -------------
    // debug 3
    // -------------
    // float tmp = ((super_up + 1.0) / (up + 1.0) - 1.0) * pick_n;

  //  printf("tag11.2: idx=%lu\n", idx);
    if (std::isnan(tmp) || std::isinf(tmp) || tmp > 1.0 ){
  //  	printf("tag11.3: idx=%lu\n", idx);
        tmp = 1.0;
    }
//    printf("tag11.4: idx=%lu\n", idx);
    scale_factor[tid] += tmp;
//    printf("tag11.5: idx=%lu\n", idx);
    
 //   printf("tag12: idx=%lu\n", idx);
    fill_n_rand_of_range_from_super<IdxT>(
        coo_row + nid_offset, pick_n, 0, up, super_up, repeated);
    // printf("count: %f, up: %llu, down: %llu\n", count[idx], up + 1, super_up);

//    printf("tag13: idx=%lu\n", idx);
    // construct coo of nid's part
    for (size_t off = nid_offset; off < nid_offset + pick_n; off++) {
      // fill back orig ids of nid's sampled in nbrs
      const size_t& orig_off = coo_row[off] + in.hidx_ptr[nid];
      coo_row[off] = in.hidx[orig_off];
      coo_col[off] = nid;
    }
 //   printf("tag14: idx=%lu\n", idx);
  }
  }

 // printf("tag2\n");
  float sum = std::accumulate(scale_factor.begin(), scale_factor.end(), 0.0);
  // printf("scaling count %f, size: %llu\n", sum, count.size());
  *resampling_count = sum;
}

/*
    in    :
            the input graph of csc format
    fanout:
            for a node of interested, how many nbrs will be sampled

            assume in is directed graph, and nbrs from in edges are sampled
    repeated:
            if an nbr will be sampled repeated

    n_source:
            number of nodes that sampling start from

            source is not indicating the direction of edges,
            but means the start point / source of sampling processure
    source:
            pointer to the array contains of source nodes ids

*/
template <typename IdxT>
coo_h<IdxT> sampling_graph_h(const csc_h<IdxT>& in, const size_t fanout,
                             const bool repeated, const size_t n_source,
                             const IdxT* source) {
  // assume no duplication id in source

  if (n_source == 0) {
    coo_h<IdxT> ret;  // csc_h looks more natural
    ret.n_n = 0;
    ret.n_e = 0;
    return ret;
  }
  // pick_n_i_nbr[0] is for source[0]
  // it means the n_umber of i_n nbrs will be picked
  chunk<size_t> pick_n_i_nbr(n_source, 0);
  // the i_n deg_ree of source nodes
  chunk<size_t> source_i_deg(n_source, 0);

  collect_pick_n_and_ideg<IdxT>(in, fanout, repeated, n_source, source,
                                &pick_n_i_nbr, &source_i_deg);

  chunk<size_t> i_nbr_offset = exclusive_scan(pick_n_i_nbr);

  coo_h<IdxT> ret;  // csc_h looks more natural
  ret.n_e = i_nbr_offset.back();
  ret.row = chunk<IdxT>(ret.n_e, 0);
  ret.col = chunk<IdxT>(ret.n_e, 0);

  fill_sampling_result_to_coo<IdxT>(in, repeated, n_source, source,
                                    &i_nbr_offset, &pick_n_i_nbr, &source_i_deg,
                                    ret.row.data(), ret.col.data());

  IdxT max_u = find_max(ret.row);
  IdxT max_v = find_max(ret.col);

  IdxT max_id = std::max(max_u, max_v);
  ret.n_n = max_id + 1;

  return ret;
}

/*
    source_gideg:
            the g_lobal i_n degree of source nodes
            if this para is not null, it means the csc_h in is viewed as a
            partition of a graph, the in deg from csc_h could be smaller
            than the global in degree
   total_resample:
            this will be initialized as 0, and used to counting the number
            of times the sampler need to resample because miss an neighbor in
            local partition
*/

template <typename IdxT>
coo_h<IdxT> sampling_graph_h_count_resampling(
    const csc_h<IdxT>& in, const size_t fanout, const bool repeated,
    const size_t n_source, const IdxT* source, const IdxT* source_gideg,
    const IdxT* total_resample);

}  // namespace t10n

#endif  // CSRC_GRAPH_SAMPLING_H_H_
