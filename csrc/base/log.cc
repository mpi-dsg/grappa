/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
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
