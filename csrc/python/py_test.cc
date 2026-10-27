/**
 *  Copyright (c) 2024 by Data Systems Group, MPI-SWS
 */

#include "csrc/python/py_test.h"

namespace t10n {

namespace pygraph {

Batch py_test_batch() {
  Batch b;
  /*
    auto init0 = [](pybind11::array_t<int64_t> arr) {
      int64_t* p = static_cast<int64_t*>(arr.request().ptr);
      for (size_t i = 0; i < arr.request().shape[0]; i++) {
        p[i] = i;
      }
    };
  */
  auto init = [](std::vector<int64_t> arr) {
    int64_t* p = arr.data();
    for (size_t i = 0; i < arr.size(); i++) {
      p[i] = i;
    }
  };

  std::vector<int64_t> u0(3);
  std::vector<int64_t> v0(3);

  std::vector<int64_t> u1(5);
  std::vector<int64_t> v1(5);

  std::vector<int64_t> s0(1);
  std::vector<int64_t> s1(2);

  init(u0);
  init(v0);
  init(u1);
  init(v1);
  init(s0);
  init(s1);

  b.layers.push_back(u0);
  b.layers.push_back(u1);
  b.layers.push_back(v0);
  b.layers.push_back(v1);

  b.input_nodes.push_back(s0);
  b.input_nodes.push_back(s1);

  return b;
}

}  // namespace pygraph
}  // namespace t10n
