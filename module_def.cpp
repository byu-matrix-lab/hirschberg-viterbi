#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "hirschberg_viterbi.h"
#include "pruned_hirschberg_viterbi.h"

// TODO: maybe add templates for what is wanted here here
PYBIND11_MODULE(hirschberg_viterbi_impl, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("viterbi",
        &viterbi,
        "Base Viterbi Alignment",
        py::arg("log_probs"),
        py::arg("targets"),
        py::kw_only(),
        py::arg("blank") = 0
    );

    m.def("hirschberg_viterbi",
        &hirschberg_viterbi,
        "Hirschberg Viterbi Alignment",
        py::arg("log_probs"),
        py::arg("targets"),
        py::kw_only(),
        py::arg("blank") = 0,
        py::arg("soft_mem_limit") = 1000LL
    );

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
}
