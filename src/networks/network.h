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
    virtual ~network_t() {}

    // Run network.
    virtual void run() = 0;
    // Load batch data.
    virtual void load_data(const unsigned m_batch_index) = 0;
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

protected:
    // Forward propagation
    void forward();
    // Initialize network.
    virtual void init(const std::string m_network_config) = 0;
    // Initialize weight.
    void init_weight(const std::string m_input_weight);
};

} // namespace nebula
#endif
