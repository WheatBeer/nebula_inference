#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "convolutional.h"
#include "def.h"
#include "layer.h"

namespace nebula {

using namespace std;
using namespace cv;

convolutional_t::convolutional_t(const std::string m_network_config) {
    std::cout << "Initializing network ..." << std::endl;
    // Initialize network.
    init(m_network_config);
}

convolutional_t::~convolutional_t() {
    for(unsigned i = 0; i < layers.size(); i++) { delete layers[i]; }
    delete [] input_data;
    delete [] input_label;
    delete [] reference_label;
}

// Initialize network.
void convolutional_t::init(std::string m_network_config) {
    // Parse the configuration file.
    config_t config;
    config.parse(m_network_config);

    // Number of layers is equivalent to the size of sections.
    // -2 counts for generic network & dataset setting section.
    num_layers = config.sections.size() - 2;
    layers.reserve(num_layers);
    string input_weight;

    for(size_t i = 0; i < config.sections.size(); i++) {
        section_config_t section_config = config.sections[i];
        // Network configuration
        if(section_config.name == "net") {
            section_config.get_setting("num_threads", &num_threads);
            section_config.get_setting("height", &input_height);
            section_config.get_setting("width", &input_width);
            section_config.get_setting("channels", &input_channel);
            section_config.get_setting("batch", &batch_size);
            input_size = input_height * input_width * input_channel;
            section_config.get_setting("weight", &input_weight);
        }
        else if(section_config.name == "dataset") {
            init_data(section_config);
        }
        // Layer configuration
        else {
            layer_t *layer = NULL;
            if(section_config.name == "convolutional") {
                layer = new convolutional_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, layer_type_t::CONVOLUTIONAL_LAYER);
            }
            else if(section_config.name == "connected") {
                layer = new connected_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, layer_type_t::CONNECTED_LAYER);
            }
            else if(section_config.name == "maxpool") {
                layer = new pooling_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, layer_type_t::MAXPOOL_LAYER);
            }
            else if(section_config.name == "avgpool") {
                layer = new pooling_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, layer_type_t::AVGPOOL_LAYER);
            }
            else if(section_config.name == "softmax") {
                layer = new softmax_layer_t(this, layers.size()?layers[layers.size()-1]:NULL, layer_type_t::SOFTMAX_LAYER);
                output_layer = layer; // Softmax is output layer.
            }
            else {
                cerr << "Error: unknown layer type " << section_config.name << endl;
                exit(1);
            }
            // The first created layer becomes input layer.
            if(!layers.size()) { input_layer = layer; }
            // Initialize layer.
            layer->init(section_config);
            layers.push_back(layer); 
        }
        init_weight(input_weight);
    }
}

void convolutional_t::init_data(section_config_t &m_section_config) {
    // Input configuration
    string input_list, label_list;
    m_section_config.get_setting("inputs", &input_list);
    m_section_config.get_setting("labels", &label_list);
    m_section_config.get_setting("top_k", &top_k);
   
    // Read input list.
    fstream input_list_file;
    input_list_file.open(input_list.c_str(), fstream::in);
    if(!input_list_file.is_open()) {
        cerr << "Error: failed to open " << input_list << endl;
        exit(1);
    }
    string input;
    while(getline(input_list_file, input)) { inputs.push_back(input); }
    input_list_file.close();

    // Update epoch length 
    epoch_length = inputs.size() / batch_size;

    // Read label list.
    fstream label_list_file;
    label_list_file.open(label_list.c_str(), fstream::in);
    if(!label_list_file.is_open()) {
        cerr << "Error: failed to open " << label_list << endl;
        exit(1);
    }
    string label;
    while(getline(label_list_file, label)) { labels.push_back(label); }
    num_classes = labels.size();
    label_list_file.close();

    // Reserve memory for input data and labels.
    input_size = input_height * input_width * input_channel;
    input_data = new float[input_size*batch_size];
    input_label = new float[num_classes * batch_size]();
    reference_label = new unsigned[batch_size]();
}

// Run network.
void convolutional_t::run() {
    cout << "Running network ..." << endl;

    // Inference 
    unsigned batch_count = inputs.size() / batch_size - 1;
    for(iteration = 0; iteration < batch_count; iteration++) {
        // Loda batch data.
        load_data(iteration);
        // Forward propagation
        forward();
        // Print batch processing results.
        print_results();
    }
    cout << endl << "Network inference is done." << endl;
}


// Load batch data.
void convolutional_t::load_data(const unsigned m_batch_index) {
    // Load batch data.
    vector<string> batch_inputs;
    batch_inputs.reserve(batch_size);

    // Sequentially load batch data.
    for(unsigned i = 0; i < batch_size; i++) {
        batch_inputs.push_back(inputs[m_batch_index*batch_size + i]);
    }

    // Mark matching labels in the batch.
    memset(input_label, 0, batch_size * num_classes * sizeof(float)); 
    for(unsigned i = 0; i < batch_size; i++) {
        for(unsigned j = 0; j < num_classes; j++) {
            if(batch_inputs[i].find(labels[j]) != string::npos) {
                input_label[i*num_classes + j] = 1.0;
                reference_label[i] = j;
            }
        }
    }

    // Set opencv flag.
    // flag -1: IMREAD_UNCHANGED
    // flag  0: IMREAD_GRAYSCALE
    // flag  1: IMREAD_COLOR
    int opencv_flag = -1;
    if(input_channel == 1) { opencv_flag = 0; }
    else if(input_channel == 3) { opencv_flag = 1; }
    else {
        cerr << "Error: unsupported image channel " << input_channel << endl;
        exit(1);
    }

    // Load data in parallel.
    vector<thread> threads;
    threads.reserve(num_threads);
    for(unsigned tid = 0; tid < num_threads; tid++) {
        threads.emplace_back(bind([&](const unsigned begin, const unsigned end,
                                      const unsigned tid) {
            for(unsigned i = begin; i < end; i++) {
                Mat src, dst;
                // Check input data format.
                if(batch_inputs[i].find("png") != string::npos) { src = imread(batch_inputs[i], -1); }
                else { src = imread(batch_inputs[i], opencv_flag); }
                if(src.empty()) {
                    cerr << "Error: failed to load input " << inputs[i] << endl;
                    exit(1);
                }

                // Resize data.
                if((input_height != (unsigned)src.size().height) ||
                   (input_width  != (unsigned)src.size().width)) {
                    resize(src, dst, Size(input_width, input_height), 0, 0, INTER_LINEAR);
                }
                else { dst = src; }

                // Flatten data into 1-D array.
                unsigned height  = dst.size().height;
                unsigned width   = dst.size().width;
                unsigned channel = dst.channels();
                float *data = new float[height * width * channel]();

                for(unsigned h = 0; h < height; h++) {
                    for(unsigned c = 0; c < channel; c++) {
                        for(unsigned w = 0; w < width; w++) {
                            data[c * width * height + h * width + w] =
                            dst.data[h * dst.step + w * channel + c]/255.0;
                        }
                    }
                }

                for(unsigned i = 0; i < height * width; i++) {
                    swap(data[i], data[i + 2 * width * height]);
                }

                memcpy(input_data + i * input_size, data,
                       input_height * input_width * input_channel * sizeof(float));
                delete [] data;
            }
        }, tid * batch_size / num_threads, (tid + 1) * batch_size / num_threads, tid));
    } for_each(threads.begin(), threads.end(), [](thread& t) { t.join(); });
}

// Print results.
void convolutional_t::print_results() {
	// Array indices to sort out top-k classes.
	vector<unsigned> indices(num_classes);
	vector<unsigned> sorted(num_classes); {
        int x = 0;
        iota(sorted.begin(), sorted.end(), x++);
	}

	// Sort output neuron indices in decending order.
	static unsigned matches = 0;
	for(unsigned i = 0; i < batch_size; i++) {
        indices = sorted;
        sort(indices.begin(), indices.end(), [&](unsigned a, unsigned b) {
            return output_layer->output_data[i*num_classes + a] >
                    output_layer->output_data[i*num_classes + b];
        });
        for(unsigned k = 0; k < top_k; k++) {
            if(indices[k] == reference_label[i]) { matches++; }
        }
	}

	// Print results.
	cout << "Iteration #" << iteration
		 << " (data #" << ((iteration+1) * batch_size) << ")" 
		 << "  Accuracy: " << std::fixed << std::setprecision(6)
		 << (100.0 * matches/((iteration+1) * batch_size)) << "%" << endl;
}

} //namespace nebula
