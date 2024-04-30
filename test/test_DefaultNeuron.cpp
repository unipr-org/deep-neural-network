#include "Network.h"
#include "spdlog/spdlog.h"
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <assert.h>

#include "DefaultNeuron.h"
#include "Neuron.h"
using namespace ANN;

void test_Constructor() {
	spdlog::info("START test_Constructor");

	std::vector<weight_t> w = {1, 2, 3};
	DefaultNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created DefaultNeuron cn");

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
	DefaultNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created DefaultNeuron cn");

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


void test_SquareBracketsOperator() {
	spdlog::info("START test_SquareBracketsOperator(operator[])");
	std::vector<weight_t> w = {1.0, 0.5, 1};
	DefaultNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created DefaultNeuron cn");

	Neuron<> &n = cn;
	assert(n.getWeights().size() == w.size());
	for (size_t i = 0; i < w.size(); ++i) {
		assert(w[i] == n[i]);
	}
	for (size_t i = 0; i < w.size(); ++i) {
		w[i] *= 2;
		n[i] *= 2;
	}
	
	for (size_t i = 0; i < w.size(); ++i) {
		assert(w[i] == n[i]);
	}

	spdlog::info("END test_SquareBracketsOperator(operator[])");
}

void test_SetWeights() {
	spdlog::info("START test_SetWeights");

	std::vector<weight_t> w = {1.0, 0.5, 1};
	DefaultNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created DefaultNeuron cn");

	Neuron<> &n = cn;

	std::vector<weight_t> newWeights = {2.0, 1.0, 3};
	cn.setWeights(newWeights);

	for (size_t i = 0; i < newWeights.size(); ++i) {
		assert(cn[i] == newWeights[i]);
	}

	spdlog::info("END test_SetWeights");
}

void test_GetWeights() {
	spdlog::info("START test_GetWeights");

	std::vector<weight_t> w = {1.0, 0.5, 1};
	DefaultNeuron cn(w, [=](weight_t x) -> weight_t { return x; });
	spdlog::info("Created DefaultNeuron cn");

	Neuron<> &n = cn;

	auto& neuronWeights = cn.getWeights();
	assert(neuronWeights.size() == w.size());

	for (size_t i = 0; i < w.size(); ++i) {
		assert(neuronWeights[i] == w[i]);
	}

	neuronWeights[0] = 4;
	assert(cn[0] == 4);
	
	spdlog::info("END test_GetWeights");
}

int main() {
	test_Constructor();
	test_Evaluate();
	test_SquareBracketsOperator();
	test_SetWeights();
	test_GetWeights();
	return 0;
}
