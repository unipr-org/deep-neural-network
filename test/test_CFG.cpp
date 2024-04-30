#include "ANN.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <vector>

#include "Neuron.h"
#include "ConcreteNeuron.h"


int main() {
	using namespace ann;

	std::vector<weight_t> w = {1, 2, 4};
	ConcreteNeuron<> cn(w, [&](weight_t) -> weight_t {
		return 1.1;
	});
	
	Neuron<> &n = cn;
	
	std::cout<<n;

	return 0;
}
