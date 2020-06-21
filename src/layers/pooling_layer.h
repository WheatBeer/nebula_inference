#ifndef __POOLING_LAYER_H__
#define __POOLING_LAYER_H__

#include "layer.h"

namespace nebula {

class pooling_layer_t : public layer_t {
public:
    pooling_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type);
    ~pooling_layer_t();
    
    // Initialize layer.
    void init(section_config_t m_section_config);
    // Initialize weight from file.
    void init_weight(std::fstream &m_weight_file);
    // Initialize weight from scratch
    void init_weight();
    // Forward propagation.
    void forward();

    unsigned *index;
};

} // namespace nebula 
#endif
