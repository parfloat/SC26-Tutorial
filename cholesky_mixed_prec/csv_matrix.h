// csv_matrix.h -- read a square matrix from a comma-separated file (as written by gen_chol_matrices.py) into row-major doubles.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
static bool read_csv_matrix(const char *path, std::vector<double> &A, int &n)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    std::vector<char> line(1 << 20);
    int rows = 0; n = 0; A.clear();
    while (fgets(line.data(), (int)line.size(), f)) {
        char *p = line.data(), *e; int cols = 0;
        while (*p && *p != '\n') { A.push_back(strtod(p, &e)); if (e == p) break; p = e; if (*p == ',') ++p; ++cols; }
        if (rows == 0) n = cols;
        if (cols != n) { fprintf(stderr, "%s: row %d has %d entries, expected %d\n", path, rows, cols, n); fclose(f); return false; }
        ++rows;
    }
    fclose(f);
    if (rows != n) { fprintf(stderr, "%s: %d rows but %d columns\n", path, rows, n); return false; }
    return true;
}
