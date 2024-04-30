#include "Network.h"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ConcreteNeuron.h"
#include "Neuron.h"
using namespace ANN;

void test_Constructor() {
	spdlog::info("START test_Constructor");

	std::vector<weight_t> w = {1, 2, 3};
	ConcreteNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created ConcreteNeuron cn");

	Neuron<> &n = cn;

	std::vector<weight_t> w_cn = n.getWeights();
	spdlog::info("having weights:");
	for (weight_t a : w_cn) {
		std::cout << a << " ";
	}
	std::cout << std::endl;

	spdlog::info("END test_Constructor");
}

#define abs(x) (x ? x > 0 : -x)

void test(Neuron<> &n, std::vector<weight_t> input, weight_t expected,
		  long double tolerance = 10E-10) {
	weight_t value = n.evaluate(input);

	if (abs(expected - value) > tolerance) {
		std::string msg =
			"While performing evaluate() expected: " + std::to_string(expected) + ", got: value";
		throw std::runtime_error(msg);
	} else {
		std::string msg = "Evaluated\n\t";

		msg += "g({";
		for (auto it = input.begin(); it != input.end(); ++it) {
			msg += std::to_string(*it);
			if (it != input.end() - 1)
				msg += ", ";
		}
		msg += "}";

		msg += " x ";

		std::vector<weight_t> w = n.getWeights();
		msg += "{";
		for (auto it = w.begin(); it != w.end(); ++it) {
			msg += std::to_string(*it);
			if (it != w.end() - 1)
				msg += ", ";
		}
		msg += "}) = " + std::to_string(value);

		spdlog::info(msg);
	}
}

void test_Evaluate() {
	spdlog::info("START test_Evaluate");

	std::vector<weight_t> w = {1.0, 0.5, 1};
	ConcreteNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created ConcreteNeuron cn");

	Neuron<> &n = cn;

	std::vector<weight_t> input;
	weight_t result, expected;

	input = {1.5, 1.5, 1.5};
	expected = 3.75;
	test(n, input, expected);

	input = {0, 0, 0};
	expected = 0;
	test(n, input, expected);

	input = {0, 1, 0};
	expected = 0;
	test(n, input, expected);

	spdlog::info("END test_Evaluate");
}

int main() {
	test_Constructor();
	test_Evaluate();

	return 0;
}
