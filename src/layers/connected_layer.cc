#include <algorithm>
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>

#include "connected_layer.h"
#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"

namespace nebula {

using namespace std;

connected_layer_t::connected_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    bias(NULL),
    bias_update(NULL),
    weight(NULL),
    weight_update(NULL),
    weight_size(0),
    batch_normalize(false),
    scale(NULL),
    scale_update(NULL),
    normalize_mean(NULL),
    rolling_mean(NULL),
    mean_delta(NULL),
    normalize_variance(NULL),
    rolling_variance(NULL),
    variance_delta(NULL),
    x(NULL),
    normalize_x(NULL) {
}

connected_layer_t::~connected_layer_t() {
    delete [] bias;
    delete [] bias_update;
    delete [] weight;
    delete [] weight_update;
    delete [] output_data;
    delete [] delta;
    if(batch_normalize) {
        delete [] scale;
        delete [] scale_update;
        delete [] normalize_mean;
        delete [] rolling_mean;
        delete [] mean_delta;
        delete [] normalize_variance;
        delete [] rolling_variance;
        delete [] variance_delta;
        delete [] x;
        delete [] normalize_x;
    }
}

void connected_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("output", &output_size);
    m_section_config.get_setting("batch_normalize", &batch_normalize);  
    
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

    if(batch_normalize) {
        scale        = new float[output_size]();
        scale_update = new float[output_size]();
        for(unsigned i = 0; i < output_size; i++) {
            scale[i] = 1.0;
        }

        normalize_mean = new float[output_size]();
        rolling_mean = new float[output_size]();
        mean_delta = new float[output_size]();

        normalize_variance = new float[output_size]();
        rolling_variance = new float[output_size]();
        variance_delta = new float[output_size]();

        x = new float[output_size * network->batch_size]();
        normalize_x = new float[output_size * network->batch_size]();
    }
}

// Initialize weight from weight file.
void connected_layer_t::init_weight(fstream &m_input_weight) {
    m_input_weight.read((char*)bias, output_size * sizeof(float));
    m_input_weight.read((char*)weight, weight_size * sizeof(float));
    
    if(batch_normalize) {
        m_input_weight.read((char*)scale, output_size * sizeof(float));
        m_input_weight.read((char*)rolling_mean, output_size * sizeof(float));
        m_input_weight.read((char*)rolling_variance, output_size * sizeof(float));
    }
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
    // Forward bias
    if(batch_normalize) {
        forward_batchnorm();
    } 
    forward_bias(num_threads, output_data, bias, output_size, 1, network->batch_size);
   
    // Activate function
    activate();
}

void connected_layer_t::forward(float *m_input_data) {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta , 0.0, output_size * network->batch_size * sizeof(float));
   
    float *input_data = m_input_data ? m_input_data :   
                        prev_layer ? prev_layer->output_data : network->input_data;
	// Matrix multiplication
    gemm(0, 1,
         network->batch_size, output_size, input_size,
         1.0,
         input_data, input_size,
         weight, input_size,
         1.0,
         output_data, output_size,
         num_threads);
    // Forward bias
    if(batch_normalize) {
        forward_batchnorm();
    } 
    forward_bias(num_threads, output_data, bias, output_size, 1, network->batch_size);
   
    // Activate function
    activate();
}

// Forward batch normalization.
void connected_layer_t::forward_batchnorm() {
    memcpy(x, output_data, output_size * network->batch_size * sizeof(float));
	// normalize all output data.
	batchnorm_normalize(num_threads, output_data, rolling_mean, rolling_variance, 
						output_size, 1, network->batch_size);
    batchnorm_scale_down(num_threads, output_data, scale, 
                         output_size, 1, network->batch_size);
}

void connected_layer_t::increment(int step){
    int num = (int)output_size * (int)network->batch_size * step;
    output_data += num; 
    delta += num;
    
    if(batch_normalize) {
        x += num;
        normalize_x += num; 
    }
}

} // namespace nebula
