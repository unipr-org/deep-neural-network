#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <iostream>
#include <vector>

using namespace ANN;
using namespace std;
using namespace spdlog;

template <typename T> std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
	size_t index = 0;

	os << "{";
	for (auto it = v.begin(); it != v.end(); ++it, ++index) {
		os << *it;
		if (index != v.size() - 1)
			os << ", ";
	}
	os << "}";
}

void test_Constructor() {
	spdlog::info("START test_Constructor");
	std::vector<weight_t> w = {1, 1, 1};
	std::vector<DefaultNeuron> vn;

	DefaultNeuron dn(w, [=](weight_t x) -> weight_t { return x; });

	size_t size = 10;
	for (int i = 0; i < size; ++i)
		vn.push_back(dn);

	DefaultLayer dl(vn);
	std::cout << dl;

	spdlog::info("END test_Constructor");
}

void test_Evaluate() {
	info("START test_Evaluate");

	vector<weight_t> w = {1, 1, 1};
	DefaultNeuron dn(w, [=](weight_t x) -> weight_t { return x; });

	DefaultLayer dl(vector<DefaultNeuron>(1, dn));
	DefaultNetwork d_net(vector<DefaultLayer>(1, dl));

	vector<data_t> input = {1, 1};
	vector<data_t> output(1);
	d_net.evaluate(input, output);

	std::cout << output << std::endl;

	info("END test_Evaluate");
}

int main(int argc, char *argv[]) {
	test_Constructor();
	test_Evaluate();
	return 0;
}
