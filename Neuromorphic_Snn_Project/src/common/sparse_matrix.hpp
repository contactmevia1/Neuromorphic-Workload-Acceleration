/**
 * sparse_matrix.hpp
 * 
 * Compressed Sparse Row (CSR) matrix implementation.
 * 
 

#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <iostream>

template<typename T = float>
struct CSRMatrix {
    int num_rows = 0;
    int num_cols = 0;
    int num_nonzeros = 0;

    std::vector<T> values;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    std::vector<int> src_idx;

    CSRMatrix() = default;

  
    CSRMatrix(int rows, int cols, float density, 
              std::mt19937& rng, 
              T weight_min = 0.01f, 
              T weight_max = 0.5f) 
        : num_rows(rows), num_cols(cols) 
    {
        std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);
        std::uniform_real_distribution<T> dist_weight(weight_min, weight_max);

        bool show_progress = (rows > 2000);
        if (show_progress) {
            std::cout << "  Building CSR matrix..." << std::flush;
        }

        // Single-pass: store (row, col, weight) triples in a temporary vector
        // This is O(N^2) memory but only for the non-zero entries
        struct Conn { int row; int col; T weight; };
        std::vector<Conn> connections;
        connections.reserve(static_cast<size_t>(rows * cols * density * 1.5));

        for (int i = 0; i < rows; i++) {
            if (show_progress && rows > 5000 && i % (rows / 10) == 0) {
                std::cout << "." << std::flush;
            }
            for (int j = 0; j < cols; j++) {
                if (dist_prob(rng) < density) {
                    connections.push_back({i, j, dist_weight(rng)});
                }
            }
        }

        num_nonzeros = static_cast<int>(connections.size());

        // Build CSR arrays from connections
        values.resize(num_nonzeros);
        col_idx.resize(num_nonzeros);
        src_idx.resize(num_nonzeros);
        row_ptr.resize(rows + 1, 0);

        // Count entries per row
        for (const auto& c : connections) {
            row_ptr[c.row + 1]++;
        }

        // Cumulative sum to get row_ptr
        for (int i = 0; i < rows; i++) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Fill values, col_idx, src_idx
        // We need per-row offsets to handle multiple entries per row
        std::vector<int> row_offsets(rows, 0);
        for (int i = 0; i < rows; i++) {
            row_offsets[i] = row_ptr[i];
        }

        for (const auto& c : connections) {
            int idx = row_offsets[c.row]++;
            values[idx] = c.weight;
            col_idx[idx] = c.col;
            src_idx[idx] = c.row;
        }

        if (show_progress) {
            std::cout << " Done! " << num_nonzeros << " synapses." << std::endl;
        }
    }

    CSRMatrix<T> transpose() const {
        CSRMatrix<T> trans;
        trans.num_rows = num_cols;
        trans.num_cols = num_rows;
        trans.num_nonzeros = num_nonzeros;
        trans.values.resize(num_nonzeros);
        trans.col_idx.resize(num_nonzeros);
        trans.row_ptr.resize(num_cols + 1, 0);
        trans.src_idx.resize(num_nonzeros);

        std::vector<int> col_counts(num_cols, 0);
        for (int i = 0; i < num_nonzeros; i++) {
            col_counts[col_idx[i]]++;
        }

        trans.row_ptr[0] = 0;
        for (int j = 0; j < num_cols; j++) {
            trans.row_ptr[j + 1] = trans.row_ptr[j] + col_counts[j];
        }

        std::fill(col_counts.begin(), col_counts.end(), 0);

        for (int i = 0; i < num_rows; i++) {
            for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; idx++) {
                int j = col_idx[idx];
                int dest_idx = trans.row_ptr[j] + col_counts[j];
                trans.values[dest_idx] = values[idx];
                trans.col_idx[dest_idx] = i;
                trans.src_idx[dest_idx] = j;
                col_counts[j]++;
            }
        }

        return trans;
    }
};

#endif // SPARSE_MATRIX_HPP
