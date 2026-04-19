/**
 *  Copyright (c) 2024-2026 by MPI-SWS, Germany
 *  All rights reserved.
 *
 *  Author: Chongyang Xu <cxu@mpi-sws.org>
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "csrc/base/log.h"
#include "csrc/python/py_dist_chunking.h"
#include "csrc/python/py_graph.h"
#include "csrc/python/py_seq_chunking.h"
#include "csrc/python/py_test.h"
#include "csrc/python/py_xborder.h"

INIT_LOG

#define UNW_LOCAL_ONLY

#include <libunwind.h>
#include <cxxabi.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

void print_stacktrace() {
    unw_cursor_t cursor;
    unw_context_t context;
    unw_getcontext(&context);
    unw_init_local(&cursor, &context);

    while (unw_step(&cursor) > 0) {
        char fname[256];
        unw_word_t offset, pc;
        unw_get_reg(&cursor, UNW_REG_IP, &pc);
        if (unw_get_proc_name(&cursor, fname, sizeof(fname), &offset) == 0) {
            int status;
            char *demangled = abi::__cxa_demangle(fname, NULL, NULL, &status);
            printf("0x%lx: (%s+0x%lx)\n", pc,
                   (status == 0 && demangled) ? demangled : fname,
                   offset);
            free(demangled);
        } else {
            printf("0x%lx: -- error: unable to obtain symbol name\n", pc);
        }
    }
}
void segfault_handler(int sig) {
    fprintf(stderr, "Caught signal %d (SIGSEGV)\n", sig);
    print_stacktrace();
    exit(1);
}

PYBIND11_MODULE(MODULE_NAME, m) {
  // Install handler at module import
  signal(SIGSEGV, segfault_handler);

  // binding apis for graph and sampler
  pybind11::module g = m.def_submodule(
      "graph", "t10n: this module contains graph data structures in cpp/cuda");

  py::class_<t10n::pygraph::Batch>(g, "Batch")
      .def(py::init<>())
      .def("get_layers", &t10n::pygraph::Batch::get_layers)
      .def("get_input_nodes", &t10n::pygraph::Batch::get_input_nodes);

  g.def("py_test_batch", &t10n::pygraph::py_test_batch, "test batch");

  py::class_<t10n::pygraph::HostGraph>(g, "HostGraph")
      .def(py::init<const pybind11::array_t<int64_t>,
                    const pybind11::array_t<int64_t>>())
      .def(py::init<const pybind11::array_t<int64_t>,
                    const pybind11::array_t<int64_t>, int>());

  py::class_<t10n::pygraph::GpuGraph>(g, "GpuGraph")
      .def(py::init<const t10n::pygraph::HostGraph&>());

  py::class_<t10n::pygraph::HostSampler>(g, "HostSampler")
      .def(py::init<t10n::pygraph::HostGraph*, std::vector<size_t>,
                    const bool>())
      .def(py::init<t10n::pygraph::HostGraph*, std::vector<size_t>, const bool,
                    const bool>())
      .def("c_reinit_from", &t10n::pygraph::HostSampler::ReinitFrom)
      .def("c_reinit_with_gideg_from",
           &t10n::pygraph::HostSampler::ReinitWithGidegFrom)
      .def("c_next_batch", &t10n::pygraph::HostSampler::NextBatch)
      .def("c_next_batch_py", &t10n::pygraph::HostSampler::NextBatchPy);

  py::class_<t10n::pygraph::GpuSampler>(g, "GpuSampler")
      .def(py::init<std::vector<int>>())
      .def("c_reinit_from", &t10n::pygraph::GpuSampler::ReinitFrom)
      .def("c_next_batch", &t10n::pygraph::GpuSampler::NextBatch);

  py::class_<t10n::pygraph::HostNeighborSampler>(g, "HostNeighborSampler")
      .def(py::init<t10n::pygraph::HostGraph*, const bool>())
      .def("c_next_batch_py", &t10n::pygraph::HostNeighborSampler::NextBatchPy);

  // binding apis for crossing partition border
  pybind11::module xb = m.def_submodule(
      "xborder",
      "t10n: this is ued when crossing (partition) borders is needed");
  py::class_<t10n::xb::XBorder>(xb, "XBorder")
      .def(py::init<int64_t, int64_t, std::vector<int64_t>>())
      .def("set_nbr_idx", &t10n::xb::XBorder::set_nbr_idx)
      .def("mask_by_partition", &t10n::xb::XBorder::mask_by_partition)
      .def("group_by_partition", &t10n::xb::XBorder::group_by_partition)
      .def("ad_hoc_clear", &t10n::xb::XBorder::ad_hoc_clear)
      .def("ad_hoc_build_id_mapping",
           &t10n::xb::XBorder::ad_hoc_build_id_mapping)
      .def("ad_hoc_fill_batch_feat", &t10n::xb::XBorder::ad_hoc_fill_batch_feat)
      .def("ad_hoc_fill_batch_ideg", &t10n::xb::XBorder::ad_hoc_fill_batch_ideg)
      .def("ad_hoc_fill_batch_feat_v2",
           &t10n::xb::XBorder::ad_hoc_fill_batch_feat_v2);

  // binding apis for splitting dataset
  pybind11::module c_seq =
      m.def_submodule("c_seq", "t10n: sequentially build graph into chunks ");
  c_seq.def("build", &t10n::c_seq::build, "build input graph into chunks");
  c_seq.def("split_node_feat", &t10n::c_seq::split_node_feat,
            "spltt node embeddings according to partition result");
  c_seq.def("split_node_data", &t10n::c_seq::split_node_data,
            "split node data label, masks according to partition result");
  c_seq.def("print_string", &t10n::c_seq::print_string,
            "used as test: print string from pybind module");

  // binding apis for splitting dataset
  pybind11::module c_dist =
      m.def_submodule("c_dist", "t10n: build graph into chunks in parallel");
  c_dist.def("random_assign_pid", &t10n::c_dist::random_assign_pid,
             "assign pid to node");
}
