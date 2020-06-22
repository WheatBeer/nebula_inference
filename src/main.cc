#include <iostream>
#include <string>

#include "network.h"
#include "convolutional.h"

using namespace std;

int main(int argc, char **argv) {
    if(argc != 1) {
        cerr << "Usage: " << argv[0] << endl;
        exit(1);
    }

	nebula::network_t *network = new nebula::convolutional_t("./configs/alexnet.cfg");

	network->run();
	
	delete network;
	return 0;
}
