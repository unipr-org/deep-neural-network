#ifndef LAYER_HH_INCLUSION_GUARD
#define LAYER_HH_INCLUSION_GUARD 1

#include <vector>
#include "Neuron.h"
using namespace std;

class Layer {
private:
    vector<Neuron> neurons;
    unsigned layerIndex;

public:
    Layer(int index, int numberOfNeurons, int numberOfOutputConnections);
    int getNumbersOfNeurons() { return neurons.size(); };
    void print();
    void printCFG(ofstream&);
    void printSubgraphCFG(ofstream&, unsigned);
};

#endif // LAYER_HH_INCLUSION_GUARD