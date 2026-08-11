# g++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) *.cpp -o hirschberg_viterbi_impl$(python3 -m pybind11 --extension-suffix)
python3 test.py
