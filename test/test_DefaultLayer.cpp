#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNeuron.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace ANN;
using namespace std;
using namespace spdlog;

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

int main(int argc, char *argv[]) {
	test_Constructor();
	return 0;
}
