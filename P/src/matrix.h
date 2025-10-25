#include <vector>

struct matrix_t {
    int rows;
    int cols;
    std::vector<int> data;

    matrix_t() = default;
    matrix_t(size_t r, size_t c) {
        resize(r, c);
    } 

    const int& at(int i, int j) const {
        return data[i * cols + j];
    }

    int& at(int i, int j) {
        return data[i * cols + j];
    }

    void resize(int r, int c) {
        rows = r;
        cols = c;
        data.resize(r * c, 0);
    }
};