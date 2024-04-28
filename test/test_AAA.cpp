#include "ANN.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
	vector<unsigned> topology = {2, 4, 1};

	ANN ann(topology);

	ann.print();
	ann.printCFG();

	return 0;
}
