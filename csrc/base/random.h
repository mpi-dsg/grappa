/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_RANDOM_H_
#define CSRC_BASE_RANDOM_H_

#include <parallel_hashmap/phmap.h>

#include <random>  // for std::random_device

#include <pcg_random.hpp>

#include "csrc/base/config.h"  // for std::numeric_limits<int64_t>::max()
#include "csrc/base/error.h"

namespace t10n {

// closed interval [low, up]
template <typename IdxT>
inline IdxT uniform_std(const IdxT& low, const IdxT& up) {
  static thread_local std::random_device rd;
  static thread_local std::mt19937 generator(rd());
  std::uniform_int_distribution<IdxT> distribution(low, up);
  return distribution(generator);
}

// closed interval [low, up]
template <typename IdxT>
inline IdxT uniform_pcg32(const IdxT& low, const IdxT& up) {
  static thread_local pcg_extras::seed_seq_from<std::random_device> seed_source;
  static thread_local pcg32 generator(seed_source);
  std::uniform_int_distribution<IdxT> distribution(low, up);
  return distribution(generator);
}

// closed interval [low, up]
template <typename IdxT>
inline void fill_n_rand_of_range(IdxT* const dest, const size_t& dest_len,
                                 const IdxT& low, const IdxT& up,
                                 const bool& repeated) {
  if (repeated) {
    for (size_t idx = 0; idx < dest_len; idx++) {
      dest[idx] = uniform_pcg32<IdxT>(low, up);
    }
  } else {
    const IdxT& new_up = up - low;
    ABORT_IF_T(dest_len > new_up + 1,
               "with repeated=false, desn_len_max == up - low + 1");
    phmap::parallel_flat_hash_map<IdxT, IdxT> pos2val;
    for (size_t idx = 0; idx < dest_len; idx++) {
      const IdxT& n_new_up = new_up - idx;
      const IdxT& sample = uniform_pcg32<IdxT>(0, n_new_up);
      auto it_sample = pos2val.find(sample);
      auto it_remove = pos2val.find(n_new_up);
      dest[idx] =
          (it_sample == pos2val.end()) ? low + sample : low + it_sample->second;

      const IdxT& to_fill =
          (it_remove == pos2val.end()) ? n_new_up : it_remove->second;
      if (it_sample == pos2val.end()) {
        pos2val[sample] = to_fill;
      } else {
        it_sample->second = to_fill;
      }
    }
  }
}

/*
    A variation of fill_n_rand_of_range
    this functions count the numbers of resampling when
    the valid rand is of closed interval [low, up],
    but the sampling range is [low, super_up]
*/
template <typename IdxT>
inline void fill_n_rand_of_range_from_super(IdxT* const dest,
                                               const size_t& dest_len,
                                               const IdxT& low, const IdxT& up,
                                               const IdxT& super_up,
                                               const bool& repeated) {
   // float correct_f = static_cast<float>(up)/static_cast<float>(super_up);

  if (repeated) {
    for (size_t idx = 0; idx < dest_len; idx++) {
      dest[idx] = uniform_pcg32<IdxT>(low, super_up);
      if (dest[idx] > up) {
        dest[idx] %= (up + 1);
      }
    }
  } else {
    const IdxT& new_up = up - low;  // use super_up - low when really counting
    ABORT_IF_T(dest_len > new_up + 1,
               "with repeated=false, desn_len_max == up - low + 1");
    phmap::parallel_flat_hash_map<IdxT, IdxT> pos2val;
    for (size_t idx = 0; idx < dest_len; idx++) {
      const IdxT& n_new_up = new_up - idx;
      const IdxT& sample = uniform_pcg32<IdxT>(0, n_new_up);

      auto it_sample = pos2val.find(sample);
      auto it_remove = pos2val.find(n_new_up);
      dest[idx] =
          (it_sample == pos2val.end()) ? low + sample : low + it_sample->second;

      const IdxT& to_fill =
          (it_remove == pos2val.end()) ? n_new_up : it_remove->second;
      if (it_sample == pos2val.end()) {
        pos2val[sample] = to_fill;
      } else {
        it_sample->second = to_fill;
      }
    }
    /*
        std::vector<bool> used(up, false);
        for (size_t idx = 0; idx < dest_len; idx++) {
          if(dest[idx] <= up){
            used[dest[idx]] = true;
          }
        }

        for (size_t idx = 0; idx < dest_len; idx++) {
          if(dest[idx] > up){
            counter ++;
            size_t f = dest[idx] % (up + 1);
            if(!used[f]){
                dest[idx] = f;
                used[f] = true;
            }else{
                for (size_t j = 0; j < dest_len; j++) {
                    if(!used[j]){
                        dest[idx] = j;
                        used[j] = true;
                        break;
                    }
                }
            }
          }
        }
    */
  }
  // return 0.0;
}

}  // namespace t10n
#endif  // CSRC_BASE_RANDOM_H_
