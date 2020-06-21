#include "layer.h"
#include "network.h"

namespace nebula {

network_t::network_t() 
	: num_threads(1), batch_size(1),
    input_height(1), input_width(1),
    input_channel(1), input_size(0),
    num_layers(0), num_classes(0),
    num_iterations(0), iteration(0), top_k(1),
    input_data(NULL), input_label(NULL),
    reference_label(NULL), input_layer(NULL),
    output_layer(NULL) {}

void network_t::forward() {
    for(unsigned i = 0; i < num_layers; i++) { 
		layers[i]->forward(); 
	}
}

// Initialize network.
void network_t::init(const std::string m_network_config,
                     const std::string m_data_config, 
					 const std::string m_input_weight) {
    std::cout << "Initializing network ..." << std::endl;
    // Initialize network.
    init_network(m_network_config);
    // Initialize input data.
    init_data(m_data_config);
    // Initialize weight.
    init_weight(m_input_weight);
}

// Initialize weight.
void network_t::init_weight(const std::string m_input_weight) {
	// Initialize weight from file.
	std::fstream weight_file;
	weight_file.open(m_input_weight.c_str(), std::fstream::in | std::fstream::binary);
	if(!weight_file.is_open()) {
		std::cerr << "Error: failed to open " << m_input_weight << std::endl;
		exit(1);
	}
	for(unsigned i = 0; i < layers.size(); i++) { 
		layers[i]->init_weight(weight_file); 
	}
	weight_file.close();
}

} // namespace nebula
