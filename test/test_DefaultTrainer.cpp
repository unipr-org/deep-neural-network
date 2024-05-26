
#include "ANN.h"
#include "DefaultLoader.h"
#include "DefaultNetwork.h"
#include "DefaultTrainer.h"
#include <cstdlib>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
using namespace std;
using namespace spdlog;
using namespace ANN;
using namespace Training;
using namespace Utils;

/**
 * @brief Trains a single-layer perceptron (SLP) using a provided training set and tests it against a test set.
 *
 * This method initializes a neural network (SLP), trains it with a given training set, and evaluates its performance
 * using a test set. The training process involves loading existing network weights if available, otherwise initializing
 * a new network. The network is trained using the DefaultTrainer class.
 *
 * The training process includes the following steps:
 * 1. Load the training and test datasets from specified file paths.
 * 2. Initialize the neural network:
 *    - If a saved model exists (status_slp.txt), load it.
 *    - If no saved model exists, create a new network with a specified topology and randomize its weights.
 * 3. Set the activation function for the output neuron to identity.
 * 4. Train the network using the DefaultTrainer class with specified parameters (learning rate, epochs, and threshold).
 * 5. Close the dataset files after training.
 *
 * @throws runtime_error if the specified training set or test set files are not found.
 */
void test_TrainSLP() {
	info("START test_TrainSLP");
	
	string training_set_path = "./test/dataset/slp-example/training_set.txt";
	ifstream training_set(training_set_path);
	if (!training_set) {
		error("No such file " + training_set_path);
		throw runtime_error("No such file " + training_set_path);
	}

	string test_set_path = "./test/dataset/slp-example/test_set.txt";
	ifstream test_set(test_set_path);
	if (!test_set) {
		error("No such file " + test_set_path);
		throw runtime_error("No such file " + test_set_path);
	}

	DefaultNetwork net;
	DefaultLoader l;

	ifstream model("./model/status_slp.txt");
	if (!model) {
		vector<unsigned> topology = {1};
		net.createLayers(topology, 3);
		net.randomizeWeights();
		info("Created Network");
	} else {
		l.loadNetwork(net);
		info("Loaded Network");
		model.close();
	}	
	// cout << net << endl;
	net[net.getSize() - 1][0].setActivationFunctionId(ANN::ActivationFunctionID::IDENTITY);

	DefaultTrainer trainer;

	trainer.train(net, training_set, test_set, 0.000001, 3000, 0.0001);
	
	training_set.close();
	test_set.close();
}

void test_Train() {
	info("START test_Train");
	
	string training_set_path = "./test/dataset/training_set.txt";
	ifstream training_set(training_set_path);
	if (!training_set) {
		error("No such file " + training_set_path);
		throw runtime_error("No such file " + training_set_path);
	}

	string test_set_path = "./test/dataset/test_set.txt";
	ifstream test_set(test_set_path);
	if (!test_set) {
		error("No such file " + test_set_path);
		throw runtime_error("No such file " + test_set_path);
	}

	DefaultNetwork net;
	DefaultLoader l;

	ifstream model("./model/status.txt");
	if (!model) {
		vector<unsigned> topology = {3, 3, 1};
		net.createLayers(topology, 3);
		net.randomizeWeights();
		info("Created Network");
	} else {
		l.loadNetwork(net);
		info("Loaded Network");
		model.close();
	}	
	// cout << net << endl;
	net[net.getSize() - 1][0].setActivationFunctionId(ANN::ActivationFunctionID::IDENTITY);

	DefaultTrainer trainer;

	trainer.train(net, training_set, test_set, 0.000001, 3000, 0.0001);
	
	training_set.close();
	test_set.close();
}

int main(int argc, char *argv[]) {
#ifdef DEBUG
	spdlog::set_level(spdlog::level::trace);
#else
	spdlog::set_level(spdlog::level::info);
#endif // debug

	// test_TrainSLP(); 
	test_Train();
}
