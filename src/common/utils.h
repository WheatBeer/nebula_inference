#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>

namespace nebula {   

void im2col(float *m_im_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_col_data,
            unsigned m_num_threads);

// Fold data.
void col2im(float *m_col_data, unsigned m_channel, unsigned m_height, unsigned m_width,
            unsigned m_filter_size, unsigned m_stride, unsigned m_padding, float *m_im_data,
            unsigned m_num_threads);

void forward_bias(unsigned num_threads, float *m_output, float *m_bias,
                  unsigned m_channel, unsigned m_size, unsigned m_batch);

} // namespace nebula 
#endif
