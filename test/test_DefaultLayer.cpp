#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNeuron.h"
#include "spdlog/spdlog.h"
#include <vector>

using namespace ANN;

void test_Constructor() {
	spdlog::info("START test_Constructor");
	std::vector<weight_t> w = {1, 1, 1};
	std::vector<DefaultNeuron> vn;

	DefaultNeuron dn(w, [=](weight_t x) -> weight_t { return x; });
	vn.push_back(dn);

	DefaultLayer dl(vn);

	spdlog::info("END test_Constructor");
}

int main(int argc, char *argv[]) { return 0; }
