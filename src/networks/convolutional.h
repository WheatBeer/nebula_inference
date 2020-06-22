#ifndef __CONVOLUTIONAL_H__
#define __CONVOLUTIONAL_H__

#include "config.h"

#include "network.h"

#include "convolutional_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
#include "pooling_layer.h"

namespace nebula {

class convolutional_t : public network_t {
public:
    convolutional_t(const std::string m_network_config);
    ~convolutional_t();

    // Run network.
    void run();
    // Print reulsts.
    void print_results();

private:
    // Initialize network.
    void init(const std::string m_network_config);
    // Initialize input data.
    void init_data(section_config_t &m_section_config);
    // Load batch data.
    void load_data(const unsigned m_batch_index);
};

} //namespace nebula
#endif
