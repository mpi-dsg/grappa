/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */
#ifndef CSRC_BASE_LOG_H_
#define CSRC_BASE_LOG_H_

#ifdef DISABLE_THIRD_PARTY_LOGGING
#include <iostream>
#define LOG(tag) std::cout << "[" << #tag << "] "
#define INIT_LOG
#else
#ifdef USE_GLOG
#include <glog/logging.h>
#define INIT_LOG
#else                       // use easylogging++ on default
#include <easylogging++.h>  // for allocators
#define INIT_LOG INITIALIZE_EASYLOGGINGPP
#endif  // USE_GLOG
#endif  // DISABLE_THIRD_PARTY_LOGGING

namespace t10n {

void InitLogging();

}  // namespace t10n

#endif  // CSRC_BASE_LOG_H_
