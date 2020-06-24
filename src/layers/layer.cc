#include <cstring>
#include <functional>
#include "layer.h"

namespace nebula {

layer_t::layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_type(m_layer_type),
    activation_type(activation_type_t::UNDEFINED_ACTIVATION),
    input_height(1),
    input_width(1),
    input_channel(1),
    input_size(1),
    output_height(1),
    output_width(1),
    output_channel(1),
    output_size(0),
    output_data(NULL),
    delta(NULL),
    padding(0),
    num_filters(1),
    filter_size(1),
    stride(1), 
    group(1),
    prev_layer(m_prev_layer),
    next_layer(NULL),
    num_threads(1),
    network(m_network) {
    num_threads = network->num_threads;
    if(m_prev_layer) { m_prev_layer->next_layer = this; }
}

// Activation function
void layer_t::activate() {
    switch(activation_type) {
        case activation_type_t::ELU_ACTIVATION: { 
            elu_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::HARDTAN_ACTIVATION: {
            hardtan_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::LEAKY_ACTIVATION: { 
            leaky_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::LHTAN_ACTIVATION: {
            lhtan_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::LINEAR_ACTIVATION: { 
            // Nothing to do
            break;
        }
        case activation_type_t::LOGGY_ACTIVATION: { 
            loggy_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::LOGISTIC_ACTIVATION: { 
            logistic_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::PLSE_ACTIVATION: {
            plse_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::RAMP_ACTIVATION: { 
            ramp_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::RELIE_ACTIVATION: { 
            relie_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::RELU_ACTIVATION: { 
            relu_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::STAIR_ACTIVATION: {
            stair_activation(output_data, output_size * network->batch_size);
            break;
        }
        case activation_type_t::TANH_ACTIVATION: { 
            tanh_activation(output_data, output_size * network->batch_size);
            break;
        }
        default : {
            std::cerr << "Error: undefined activation type." << std::endl;
            exit(1);
        }
    }
}

} // namespace nebula
