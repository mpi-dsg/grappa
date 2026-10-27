/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_CHUNK_H_
#define CSRC_BASE_CHUNK_H_
#include <vector>

namespace t10n {

template <class Tp>
using chunk = std::vector<Tp>;

}  // namespace t10n
#endif  // CSRC_BASE_CHUNK_H_
