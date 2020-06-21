#include <algorithm>
#include <functional>
#include <cstring>
#include <thread>

#include "pooling_layer.h"

namespace nebula {

using namespace std;

pooling_layer_t::pooling_layer_t(network_t *m_network, layer_t *m_prev_layer, layer_type_t m_layer_type) :
    layer_t(m_network, m_prev_layer, m_layer_type),
    index(NULL) {}

pooling_layer_t::~pooling_layer_t() {
    delete [] output_data;
    delete [] delta;
    
    if(layer_type == layer_type_t::MAXPOOL_LAYER) {
        delete [] index;
    }
}

void pooling_layer_t::init(section_config_t m_section_config) {
    // Get layer setting.
    m_section_config.get_setting("stride", &stride);
    // If the filters size is not set in the configure file,
    // Filter size is equal to stride.
    filter_size = stride;
    m_section_config.get_setting("size", &filter_size);

    padding = (filter_size - 1) / 2;
    m_section_config.get_setting("padding", &padding);

    // Set input parameters.
    input_height  = prev_layer ? prev_layer->output_height : network->input_height;
    input_width   = prev_layer ? prev_layer->output_width : network->input_width;
    input_channel = prev_layer ? prev_layer->output_channel : network->input_channel;
    input_size = prev_layer ? prev_layer->output_size : network->input_size;

    
    // Set output paramemters.
    output_height = (input_height + 2 * padding) / stride;
    output_width = (input_width + 2 * padding) / stride;
    output_channel = input_channel;
    output_size = output_height * output_width * output_channel;
    
    // Allocate memory for pooling layer.
    output_data = new float[output_size * network->batch_size]();
    delta = new float[output_size * network->batch_size]();
    
    if(layer_type == layer_type_t::MAXPOOL_LAYER) {
        index = new unsigned[output_size * network->batch_size]();
    }
}

void pooling_layer_t::init_weight(fstream &m_weight_file) {
    //Nothing to do
}

void pooling_layer_t::init_weight() {
    //Nothing to do
}

void pooling_layer_t::forward() {
    memset(output_data, 0.0, output_size * network->batch_size * sizeof(float));
    memset(delta, 0.0, output_size * network->batch_size * sizeof(float));
    
    float *input_data = prev_layer ? prev_layer->output_data : network->input_data;
    if(layer_type == layer_type_t::MAXPOOL_LAYER) {
        vector<thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned b = begin; b < end; b++) {
                    for(unsigned c = 0; c < output_channel; c++) {
                        for(unsigned h = 0; h < output_height; h++) {
                            for(unsigned w = 0; w < output_width; w++) {
                                unsigned output_index = w + output_width * (h + output_height * (c + output_channel * b));
                                float max_value = -1e8;
                                int max_index = -1;
                                for(unsigned i = 0; i < filter_size; i++) {
                                    for(unsigned j = 0; j < filter_size; j++) {
                                        unsigned current_height = h * stride + i - padding;
                                        unsigned current_width = w * stride + j - padding;
                                        unsigned input_index = current_width + input_width * (current_height + input_height * (c + b * input_channel));
                                        bool valid = ((current_height >= 0) && (current_height < input_height) && (current_width >= 0) && (current_width < input_width));
                                        float val = valid ? input_data[input_index] : -1e8;
                                        max_index = (val > max_value) ? input_index : max_index;
                                        max_value = (val > max_value) ? val : max_value;
                                    }
                                }
                                output_data[output_index] = max_value;
                                index[output_index] = max_index;
                            }
                        }
                    }
                }
            }, tid * network->batch_size / num_threads, (tid + 1) * network->batch_size / num_threads, tid));
        } for_each(threads.begin(), threads.end(), [](thread& t) { t.join();});
    }
    else if(layer_type == layer_type_t::AVGPOOL_LAYER) {
        vector<thread> threads;
        threads.reserve(num_threads);
        for(unsigned tid = 0; tid < num_threads; tid++) {
            threads.emplace_back(bind([&](const unsigned begin, const unsigned end, const unsigned tid) {
                for(unsigned b = begin; b < end; b++) {
                    for(unsigned c = 0; c < output_channel; c++) {
                        for(unsigned h = 0; h < output_height; h++) {
                            for(unsigned w = 0; w < output_width; w++) {
                                unsigned output_index = w + output_width * (h + output_height * (c + output_channel * b));
                                float val = 0.0;
                                for(unsigned i = 0; i < filter_size; i++) {
                                    for(unsigned j = 0; j < filter_size; j++) {
                                        unsigned current_height = h * stride + i - padding;
                                        unsigned current_width = w * stride + j - padding;
                                        unsigned input_index = current_width + input_width * (current_height + input_height * (c + b * input_channel));
                                        bool valid = ((current_height >= 0) && (current_height < input_height) && (current_width >= 0) && (current_width < input_width));
                                        val += valid ? input_data[input_index] : 0.0;
                                    }
                                }
                                output_data[output_index] = val / (filter_size * filter_size);
                            }
                        }
                    }
                }
            }, tid * network->batch_size / num_threads, (tid + 1) * network->batch_size / num_threads, tid));
        } for_each(threads.begin(), threads.end(), [](thread& t) { t.join();});
    }
    else {
        cerr << "undefined pooling layer type " << endl;
        exit(1);
    }
}

} // namespace nebula
