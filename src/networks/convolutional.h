#ifndef __CONVOLUTIONAL_H__
#define __CONVOLUTIONAL_H__

#include "network.h"

#include "convolutional_layer.h"
#include "connected_layer.h"
#include "softmax_layer.h"
#include "pooling_layer.h"

namespace nebula {

class convolutional_t : public network_t {
public:
    convolutional_t() {}
    ~convolutional_t();

    // Run network.
    void run();
    // Initialize network.
    void init_network(const std::string m_network_config);
    //Initialize input data.
    void init_data(const std::string m_data_config);
    // Load batch data.
    void load_data(const unsigned m_batch_index);
    // Print reulsts.
    void print_results();
};

} //namespace nebula
#endif
