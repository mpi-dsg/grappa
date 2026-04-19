/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */
#include "csrc/base/log.h"

namespace t10n {

void InitLogging() {
#ifndef DISABLE_THIRD_PARTY_LOGGING
#ifdef USE_GLOG
  google::InitGoogleLogging();
#endif
#endif
}

}  // namespace t10n
