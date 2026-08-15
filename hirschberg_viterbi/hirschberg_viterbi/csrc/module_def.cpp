#include <Python.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include "hirschberg_viterbi.h"
#include "pruned_hirschberg_viterbi.h"

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the STABLE_TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace hirschberg_viterbi {
// Defines the operators
STABLE_TORCH_LIBRARY(hirschberg_viterbi, m) {
  m.def("viterbi(Tensor log_probs, Tensor targets, int blank=0) -> Tensor");
  m.def("hirschberg_viterbi(Tensor log_probs, Tensor targets, int blank=0, int soft_mem_limit=1000) -> Tensor");
//   m.def("mymul(Tensor a, Tensor b) -> Tensor");
//   m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// Registers CPU implementations for mymuladd, mymul, myadd_out
STABLE_TORCH_LIBRARY_IMPL(hirschberg_viterbi, CPU, m) {
  m.impl("viterbi", TORCH_BOX(&viterbi_cpu));
  m.impl("hirschberg_viterbi", TORCH_BOX(&hirschberg_viterbi_cpu));
//   m.impl("mymul", TORCH_BOX(&mymul_cpu));
//   m.impl("myadd_out", TORCH_BOX(&myadd_out_cpu));
}

}

/*

    m.def("pruned_viterbi",
        &pruned_viterbi,
        "Pruned Viterbi Alignment",
        py::arg("log_probs"),
        py::arg("targets"),
        py::kw_only(),
        py::arg("blank") = 0,
        py::arg("var_rat") = 3.7,
        py::arg("conf") = 0.99,
        py::arg("accuracy") = 0.97,
        py::arg("precision") = -1.0,
        py::arg("recall") = -1.0
    );

    m.def("pruned_hirschberg_viterbi",
        &pruned_hirschberg_viterbi,
        "Pruned Hirschberg Viterbi Alignment",
        py::arg("log_probs"),
        py::arg("targets"),
        py::kw_only(),
        py::arg("blank") = 0,
        py::arg("var_rat") = 3.7,
        py::arg("conf") = 0.99,
        py::arg("accuracy") = 0.97,
        py::arg("precision") = -1.0,
        py::arg("recall") = -1.0,
        py::arg("soft_mem_limit") = 1000LL
    );

*/