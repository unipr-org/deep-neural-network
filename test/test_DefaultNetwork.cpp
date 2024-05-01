#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNetwork.h"
#include "DefaultNeuron.h"
#include "spdlog/spdlog.h"
#include <cassert>
#include <iostream>
#include <spdlog/common.h>
#include <vector>

using namespace ANN;
using namespace std;
using namespace spdlog;

void test_Evaluate() {
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

int main(int argc, char *argv[]) {
	spdlog::set_level(spdlog::level::trace);
	test_Evaluate();
	return 0;
}
