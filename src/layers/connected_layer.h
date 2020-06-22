#ifndef __CONNECTED_LAYER_H__
#define __CONNECTED_LAYER_H__

#include "layer.h"

namespace nebula {

class connected_layer_t : public layer_t {
public:
    connected_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~connected_layer_t();

    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_input_weight);
    // Initialize weight from scratch.
    void init_weight();
    // Forward propagation
    void forward();

    float *bias;                    // Bias.
    float *bias_update;             // Bias update.

    float *weight;                  // Weight 
    float *weight_update;           // Weight update
    unsigned weight_size;           // Weight size
};

} // namespace nebula
#endif

