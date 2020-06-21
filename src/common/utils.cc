#include <algorithm>
#include <functional>
#include <thread>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <cstring>
#include "utils.h"

namespace nebula {

using namespace std;

// Unfold data.
void im2col(float* m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data,
            unsigned m_num_threads) {
    unsigned col_height = (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width = (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_channel = m_channel * m_filter_size * m_filter_size;

    vector<thread> threads;
    threads.reserve(m_num_threads);

    for(unsigned tid = 0; tid < m_num_threads; tid++) {
        threads.emplace_back(bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for (unsigned i = begin; i < end; i++) {
                unsigned offset_w = i % m_filter_size;
                unsigned offset_h = (i / m_filter_size) % m_filter_size;
                unsigned im_c = i / m_filter_size / m_filter_size;

                for (unsigned h = 0; h < col_height; h++) {
                    for (unsigned w = 0; w < col_width; w++) {
                        unsigned im_row = offset_h + h * m_stride;
                        unsigned im_col = offset_w + w * m_stride;
                        unsigned col_index = (i * col_height + h) * col_width + w;
                        im_row -= m_padding;
                        im_col -= m_padding;
                
                        if (im_row < 0 || im_col < 0 || im_row >= m_height || im_col >= m_width) {
                            m_col_data[col_index] = 0;
                        }
                        else {
                            m_col_data[col_index] =
                            m_im_data[im_col + m_width * (im_row + m_height * im_c)];
                        }
                    }   
                }
            }
        }, tid * col_channel / m_num_threads, (tid + 1) * col_channel / m_num_threads, tid));
    } for_each(threads.begin(), threads.end(), [](thread& t) { t.join(); });
}

// Fold data.
void col2im(float* m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data,
            unsigned m_num_threads) {
    unsigned col_height = (m_height + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_width = (m_width + 2 * m_padding - m_filter_size) / m_stride + 1;
    unsigned col_channel = m_channel * m_filter_size * m_filter_size;

    vector<thread> threads;
    threads.reserve(m_num_threads);

    for(unsigned tid = 0; tid < m_num_threads; tid++) {
        threads.emplace_back(bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                unsigned offset_w = i % m_filter_size;
                unsigned offset_h = (i / m_filter_size) % m_filter_size;
                unsigned im_c = i / m_filter_size / m_filter_size;

                for (unsigned h = 0; h < col_height; ++h) {
                    for (unsigned w = 0; w < col_width; ++w) {
                        unsigned im_row = offset_h + h * m_stride;
                        unsigned im_col = offset_w + w * m_stride;
                        unsigned col_index = (i * col_height + h) * col_width + w;
		                im_row -= m_padding;
		                im_col -= m_padding;

		                if (im_row < 0 || im_col < 0 || im_row >= m_height || im_col >= m_width) {
                            // Nothing to do
                        }
                        else {
                            m_im_data[im_col + m_width * (im_row + m_height * im_c)] +=
                            m_col_data[col_index];
                        }
                    }
                }
            }
        }, tid * col_channel / m_num_threads, (tid + 1) * col_channel / m_num_threads, tid));
    } for_each(threads.begin(), threads.end(), [](thread& t) { t.join(); });
}

void forward_bias(unsigned num_threads, float *m_output, float *m_bias,
                  unsigned m_channel, unsigned m_size, unsigned m_batch) {
    vector<thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                for(unsigned j = 0; j < m_channel; j++) {
                    for(unsigned k = 0; k < m_size; k++) {
                        m_output[(i * m_channel + j) * m_size + k] += m_bias[j];
                    }
                }
            }
        }, tid * m_batch / num_threads,
           (tid + 1) * m_batch / num_threads, tid));
    } for_each(threads.begin(), threads.end(), [](thread& t) { t.join(); });
}

} // namespace nebula
