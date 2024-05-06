
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

void test_Train() {
	info("START test_NewEvaluateTwoLayer");
	string training_set_path = "./test/test_DefaultTrainer.txt";

	ifstream training_set(training_set_path);
	if (!training_set) {
		error("No such file " + training_set_path);
		throw runtime_error("No such file " + training_set_path);
	}

	vector<unsigned> topology = {4, 6, 5, 3, 1};

	DefaultNetwork net;
	// net.createLayers(7, 3);
	net.createLayers(topology, 3);
	net.randomizeWeights();

	info("Created Network");
	cout << net;

	Training::DefaultTrainer trainer;
	debug("Created Trainer");
	trainer.train(net, training_set, 0.000001, 1000, 0.001);
	training_set.close();
}

int main(int argc, char *argv[]) {
#ifdef DEBUG
	spdlog::set_level(spdlog::level::trace);
#else
	spdlog::set_level(spdlog::level::info);
#endif // debug

	test_Train();
}
