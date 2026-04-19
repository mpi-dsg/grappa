/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 * 
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#ifndef CSRC_BASE_CHUNK_H_
#define CSRC_BASE_CHUNK_H_
#include <vector>

namespace t10n {

template <class Tp>
using chunk = std::vector<Tp>;

}  // namespace t10n
#endif  // CSRC_BASE_CHUNK_H_
