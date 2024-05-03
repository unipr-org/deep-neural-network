#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "DefaultLoader.h"
#include "Loader.h"
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
	
	l.saveStatus(net);

	std::cout << "net" << std::endl << net << std::endl;
	
	Network<DefaultLayer>* net2 = l.loadNetwork();

	auto net3 = l.loadNetwork();

	std::cout << "net2" << std::endl << net2 << std::endl;
	std::cout << "net3" << std::endl << net3 << std::endl;

	info("END test_SaveStatus");

	delete net2;
}

int main(int argc, char *argv[]) {
#ifdef DEBUG
	spdlog::set_level(spdlog::level::trace);
#else
	spdlog::set_level(spdlog::level::info);
#endif // DEBUG

	test_EvaluateOneLayer();
	test_EvaluateXOR();
	test_SaveStatus();

	return 0;
}
