
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

void test_Train() {
	info("START test_NewEvaluateTwoLayer");
	
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
	cout << net << endl;

	DefaultTrainer trainer;

	trainer.train(net, training_set, test_set, 0.000001, 200, 0.000000001);
	
	training_set.close();
	test_set.close();
}

int main(int argc, char *argv[]) {
#ifdef DEBUG
	spdlog::set_level(spdlog::level::trace);
#else
	spdlog::set_level(spdlog::level::info);
#endif // debug

	test_Train();
}
