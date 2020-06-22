#include <algorithm>
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>

#include "connected_layer.h"
#include "utils.h"
#include "gemm.h"

namespace nebula {

using namespace std;

connected_layer_t::connected_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    bias(NULL),
    bias_update(NULL),
    weight(NULL),
    weight_update(NULL),
    weight_size(0) {
}

connected_layer_t::~connected_layer_t() {
    delete [] bias;
    delete [] bias_update;
    delete [] weight;
    delete [] weight_update;
    delete [] output_data;
    delete [] delta;
}

void connected_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("output", &output_size);
    
    string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }

    // Initialize layer parameters.
    input_size = prev_layer ? prev_layer->output_size : network->input_size;
    weight_size = input_size * output_size;

    bias        = new float[output_size]();
    bias_update = new float[output_size]();

    weight        = new float[weight_size]();
    weight_update = new float[weight_size]();

    output_data = new float[output_size * network->batch_size]();
    delta       = new float[output_size * network->batch_size]();
}

// Initialize weight from weight file.
void connected_layer_t::init_weight(fstream &m_input_weight) {
    m_input_weight.read((char*)bias, output_size * sizeof(float));
    m_input_weight.read((char*)weight, weight_size * sizeof(float));
}

// Initialized weight from scratch.
void connected_layer_t::init_weight() {
    minstd_rand rng(random_device{}());
    uniform_real_distribution<float> dist(-1.0, 1.0);

    // Initialize weight 
    for(unsigned i = 0; i < weight_size; i++) {
        weight[i] = sqrt(2.0 / input_size) * dist(rng);
    }
}

void connected_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta , 0.0, output_size * network->batch_size * sizeof(float));
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
   
	// Matrix multiplication
    gemm(0, 1,
         network->batch_size, output_size, input_size,
         1.0,
         input_data, input_size,
         weight, input_size,
         1.0,
         output_data, output_size,
         num_threads);
    forward_bias(num_threads, output_data, bias, output_size, 1, network->batch_size);
   
    // Activate function
    activate();
}

} // namespace nebula
