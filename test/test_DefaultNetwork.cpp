#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultLoader.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <iostream>
#include <spdlog/common.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ANN;
using namespace std;
using namespace spdlog;
using namespace Utils;

void test(data_t expected, data_t value, data_t tolerance = 10E-7) {
	if (abs(expected - value) > tolerance)
		throw std::runtime_error("[test()] expected: " + std::to_string(expected) +
								 " got: " + std::to_string(value));
	else
		spdlog::info("[test()] success: " + std::to_string(value));
}

void test_EvaluateXOR() {
	info("START test_EvaluateTwoLayer");

	vector<weight_t> w11 = {1, 1, 0.5};
	vector<weight_t> w12 = {1, 1, 1.5};
	vector<weight_t> w21 = {1, -1, 1};
	DefaultNeuron n11(w11, heaviside);
	DefaultNeuron n12(w12, heaviside);
	DefaultNeuron n21(w21, heaviside);

	DefaultLayer l1({n11, n12});
	DefaultLayer l2({n21});
	DefaultNetwork net({l1, l2});

	cout << "Created Network:" << endl;
	cout << net;

	vector<data_t> input = {0, 0};
	vector<data_t> output(1);
	net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	test(0, output[0]);

	input = {0, 1};
	net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	test(1, output[0]);

	input = {1, 0};
	net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	test(1, output[0]);

	input = {1, 1};
	net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	test(0, output[0]);

	info("END test_EvaluateTwoLayer");
}

void test_EvaluateOneLayer() {
	info("START test_Evaluate");

	vector<weight_t> w = {1, 1, -1};
	DefaultNeuron dn(w, [=](weight_t x) -> weight_t { return x; });

	DefaultLayer dl(vector<DefaultNeuron>(1, std::move(dn)));
	DefaultNetwork d_net(vector<DefaultLayer>(1, dl));

	cout << "Created Network:" << endl;
	cout << d_net;

	vector<data_t> input = {1, 1};
	vector<data_t> output(1);
	d_net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	assert(output[0] == 3);

	input = {1, -1};
	d_net.evaluate(input, output);
	cout << "Evaluated output:" << endl;
	std::cout << output << std::endl;

	assert(output[0] == 1);

	info("END test_Evaluate");
}

void test_SaveStatus() {
	info("START test_SaveStatus");

	vector<weight_t> w11 = {1, 1, 0.5};
	vector<weight_t> w12 = {1, 1, 1.5};
	vector<weight_t> w21 = {1, -1, 1};
	DefaultNeuron n11(w11, heaviside);
	DefaultNeuron n12(w12, heaviside);
	DefaultNeuron n21(w21, heaviside);

	DefaultLayer l1({n11, n12});
	DefaultLayer l2({n21});
	DefaultNetwork net({l1, l2});

	DefaultLoader l;

	/* l.saveStatus(net); */

	std::cout << "net" << std::endl << net << std::endl;

	Network<DefaultLayer> *net2 = l.loadNetwork();

	auto net3 = l.loadNetwork();

	std::cout << "net2" << std::endl << net2 << std::endl;
	std::cout << "net3" << std::endl << net3 << std::endl;

	info("END test_SaveStatus");

	delete net2;
}

void test_NewEvaluateXOR() {
	info("START test_NewEvaluateXOR");

	vector<weight_t> w11 = {1, 1, 0.5};
	vector<weight_t> w12 = {1, 1, 1.5};
	vector<weight_t> w21 = {1, -1, 1};
	DefaultNeuron n11(w11, heaviside);
	DefaultNeuron n12(w12, heaviside);
	DefaultNeuron n21(w21, heaviside);

	DefaultLayer l1({n11, n12});
	DefaultLayer l2({n21});
	DefaultNetwork net({l1, l2});

	cout << "Created Network:" << endl;
	cout << net;

	vector<data_t> input = {0, 0};
	vector<vector<data_t>> output = net.getEmptyOutputVector();
	vector<vector<data_t>> pre_activations = net.getEmptyPreActivationVector();

	info("Starting the evaluation");

	net.evaluate(input, output, pre_activations);
	cout << "Evaluated output:" << endl;
	data_t &result = output[net.getSize() - 1][0];
	std::cout << result << std::endl;

	cout << "OUTPUT:" << endl;
	for (int i = 0; i < net.getSize(); ++i) {
		cout << "[" << i << "] " << output[i] << endl;
	}
	cout << "PRE_ACTIVATION:" << endl;
	for (int i = 0; i < net.getSize(); ++i) {
		cout << "[" << i << "] " << pre_activations[i] << endl;
	}
	test(0, result);

	input = {0, 1};
	net.evaluate(input, output, pre_activations);
	cout << "Evaluated output:" << endl;
	std::cout << result << std::endl;

	test(1, result);

	input = {1, 0};
	net.evaluate(input, output, pre_activations);
	cout << "Evaluated output:" << endl;
	std::cout << result << std::endl;

	test(1, result);

	input = {1, 1};
	net.evaluate(input, output, pre_activations);
	cout << "Evaluated output:" << endl;
	std::cout << result << std::endl;

	test(0, result);

	cout << "OUTPUT:" << endl;
	for (int i = 0; i < net.getSize(); ++i) {
		cout << "[" << i << "] " << output[i] << endl;
	}
	cout << "PRE_ACTIVATION:" << endl;
	for (int i = 0; i < net.getSize(); ++i) {
		cout << "[" << i << "] " << pre_activations[i] << endl;
	}

	info("END test_NewEvaluateXOR");
}

void test_getActivationDerivative() {
	info("START test_getActivationDerivative");

	DefaultNetwork net;
	vector<unsigned> topology = {1, 1};
	net.createLayers(topology, 3);
	net.randomizeWeights();

	net[0][0].setActivationFunctionId(ANN::ActivationFunctionID::TANH);
	net[1][0].setActivationFunctionId(ANN::ActivationFunctionID::IDENTITY);
    
    activationFunction_t g = net[1][0].getActivationFunction();
    std::cout << "Identity(10) = " << g(10) << std::endl;
    activationFunction_t g_d = getActivationDerivative(net[1][0].getActivationFunctionID());
    std::cout << "Identity Derivative(20) = " << g_d(20) << std::endl;

    activationFunction_t g1 = net[0][0].getActivationFunction();
    std::cout << "Tanh(10) = " << g1(10) << std::endl;
    std::cout << "Direct Tanh(10) = " << tanh(10) << std::endl;
    
    activationFunction_t g_d1 = getActivationDerivative(net[0][0].getActivationFunctionID());
    std::cout << "Tanh Derivative(1) = " << g_d1(1) << std::endl;
    std::cout << "Direct Tanh Derivative(1) = " << tanh_d(1) << std::endl;

	info("END test_getActivationDerivative");
}

int main(int argc, char *argv[]) {
#ifdef DEBUG
	spdlog::set_level(spdlog::level::trace);
#else
	spdlog::set_level(spdlog::level::info);
#endif // DEBUG

	// test_EvaluateOneLayer();
	// test_NewEvaluateXOR();

	test_getActivationDerivative();
	
	// test_SaveStatus();

	return 0;
}
