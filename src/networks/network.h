#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <fstream>
#include <iostream>
#include <vector>

namespace nebula {

class layer_t;

class network_t {
public:
    network_t();
    virtual ~network_t(){}

    // Forward propagation
    void forward();
    // Initialize network.
    void init(const std::string m_network_config,
              const std::string m_data_config, 
			  const std::string m_input_weight);
    // Run network.
    virtual void run() = 0;
    // Initialize network.
    virtual void init_network(const std::string m_network_config) = 0;
    // Initialize input data.
    virtual void init_data(const std::string m_data_config) = 0;
    // Load batch data.
    virtual void load_data(const unsigned m_batch_index) = 0;
    // Initialize weight.
    void init_weight(const std::string m_input_weight);
    // Print reulsts.
    virtual void print_results() = 0;

    unsigned num_threads;                   // Number of CPU threads

    unsigned batch_size;                    // Batch size
    unsigned input_height;                  // Input data height
    unsigned input_width;                   // Input data width
    unsigned input_channel;                 // Input data channel
    unsigned input_size;                    // Input data size

    unsigned num_layers;                    // Number of layers
    unsigned num_classes;                   // Number of output classes
    unsigned num_iterations;                // Number of iterations to run
    unsigned iteration;                     // Number of processed batches
    unsigned epoch_length;                  // Number of iterations in an epoch
    unsigned top_k;                         // Top-k indices for inference

    float *input_data;                      // Input data
    float *input_label;                     // Input label
    std::vector<std::string> inputs;        // List of input data
    std::vector<std::string> labels;        // List of labels

    unsigned *reference_label;              // Correct labels

    layer_t *input_layer;                   // Input layer
    layer_t *output_layer;                  // Output layer
    std::vector<layer_t*> layers;           // Network layers
};

} // namespace nebula
#endif
