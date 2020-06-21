#include <iostream>
#include <string>

#include "network.h"
#include "convolutional.h"

using namespace std;

int main(int argc, char **argv) {
    if(argc != 1) {
        cerr << "Usage: " << argv[0] << " [npusim_config]" << endl;
        exit(1);
    }
    /* Initialize the global neural network data -> must be done before npu_kernel::init() */ 
    string network_config = "./configs/alexnet.cfg";
	string data_config = "./datasets/imagenet/data.cfg";
	string input_weight = "./configs/alexnet.wgh";

	nebula::network_t *network = new nebula::convolutional_t();
	network->init(network_config, data_config, input_weight);

	network->run();
	network->print_results();
	
	delete network;
	return 0;
}
