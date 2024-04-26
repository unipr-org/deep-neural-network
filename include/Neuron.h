#ifndef NEURON_HH_INCLUSION_GUARD
#define NEURON_HH_INCLUSION_GUARD 1

#include <vector>
#include "include/Connection.h"
using namespace std;

class Neuron {
private:
    int neuronIndex;

    double activationFunction(double input);
    double activationFunctionDerivative(double input);
public:
    vector<Connection> connections;
    Neuron(int neuronIndex, int numberOfOutputConnections);

    int getIndex(){ return neuronIndex; }
    void print();
};

#endif // NEURON_HH_INCLUSION_GUARD