#include <algorithm>
#include <fstream>
#include <functional>
#include <cstring>
#include <random>
#include <thread>
#include "convolutional_layer.h"

#include "utils.h"
#include "batchnorm.h"
#include "gemm.h"

namespace nebula {

using namespace std;

convolutional_layer_t::convolutional_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    workspace(NULL),
    workspace_size(0),
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

convolutional_layer_t::~convolutional_layer_t() {
    delete [] workspace;
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

void convolutional_layer_t::init(section_config_t m_section_config) {
    // Get layer settings.
    m_section_config.get_setting("filters", &num_filters);
    m_section_config.get_setting("size", &filter_size);
    m_section_config.get_setting("batch_normalize", &batch_normalize);  
    m_section_config.get_setting("padding", &padding);
    m_section_config.get_setting("stride", &stride);

    string activation_str;
    if(m_section_config.get_setting("activation", &activation_str)) {
        activation_type = (activation_type_t)get_type(activation_type_str, activation_str);
    }

    input_size = prev_layer ? prev_layer->output_size : network->input_size;

    input_height = prev_layer ? prev_layer->output_height : network->input_height;
    input_width = prev_layer ? prev_layer->output_width : network->input_width;
    input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;
    
    output_height = (input_height + 2 * padding - filter_size) / stride + 1;
    output_width = (input_width  + 2 * padding - filter_size) / stride + 1;
    output_channel = num_filters;
    output_size = output_height * output_width * output_channel;
    
    workspace_size = output_height * output_width * filter_size * filter_size * input_channel;
    weight_size = input_channel * num_filters * filter_size * filter_size / group;	   
    
    bias = new float[num_filters]();
    bias_update = new float[num_filters]();

    weight = new float[weight_size]();
    weight_update = new float[weight_size]();

    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();
    workspace = new float[workspace_size]();

    if(batch_normalize) {
        scale        = new float[num_filters]();
        scale_update = new float[num_filters]();
        for(unsigned i = 0; i < num_filters; i++) {
            scale[i] = 1.0;
        }

        normalize_mean = new float[num_filters]();
        rolling_mean = new float[num_filters]();
        mean_delta = new float[num_filters]();

        normalize_variance = new float[num_filters]();
        rolling_variance = new float[num_filters]();
        variance_delta = new float[num_filters]();

        x = new float[output_size * network->batch_size]();
        normalize_x = new float[output_size * network->batch_size]();
    }
}

void convolutional_layer_t::init_weight(fstream &m_input_weight) {
    m_input_weight.read((char*)bias, num_filters * sizeof(float));
    m_input_weight.read((char*)weight, weight_size * sizeof(float));
   
    if(batch_normalize) {
        m_input_weight.read((char*)scale, num_filters * sizeof(float));
        m_input_weight.read((char*)rolling_mean, num_filters * sizeof(float));
        m_input_weight.read((char*)rolling_variance, num_filters * sizeof(float));
    }
}

void convolutional_layer_t::init_weight() {
    minstd_rand rng(random_device{}());
    normal_distribution<float> dist(0.0, 1.0);
    
    for(unsigned i = 0; i < weight_size; i++) {
        weight[i] = sqrt(2.0 / (filter_size * filter_size * input_channel / group)) * dist(rng);
    }
}

void convolutional_layer_t::forward() {
    memset(output_data, 0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0, output_size * network->batch_size * sizeof(float));
    
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
    unsigned patch_size = filter_size * filter_size * input_channel/ group;
    unsigned num_patches = output_width * output_height;

	// Convolution
	for(unsigned i = 0; i < network->batch_size; i++){
		for(unsigned j =0 ; j < group ; j++){
			im2col(&input_data[(i * group + j) * input_channel / group * input_height * input_width], input_channel / group,
					input_height, input_width, filter_size, stride, padding, workspace,
					network->num_threads);
			gemm(0, 0,
					num_filters / group, num_patches, patch_size,
					1.0,
					&weight[j * weight_size / group], patch_size,
					workspace, num_patches, 
					1.0,
					&output_data[(i * group +j) * num_patches * num_filters / group], num_patches,
					network->num_threads);
		}
	}

    // Forward bias
    if(batch_normalize) {
        forward_batchnorm();
    } 
    forward_bias(num_threads, output_data, bias, num_filters, num_patches, network->batch_size);

    // Activate function
    activate();
}

// Forward batch normalization.
void convolutional_layer_t::forward_batchnorm() {
    unsigned num_patches = output_height * output_width;
    memcpy(x, output_data, output_size * network->batch_size * sizeof(float));
	// normalize all output data.
	batchnorm_normalize(num_threads, output_data, rolling_mean, rolling_variance, 
						output_channel, num_patches, network->batch_size);
    batchnorm_scale_down(num_threads, output_data, scale, 
                         output_channel, num_patches, network->batch_size);
}

} // namespace nebula
