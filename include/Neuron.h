#ifndef NEURON_HH_INCLUSION_GUARD
#define NEURON_HH_INCLUSION_GUARD 1

#include <vector>
#include "Connection.h"
using namespace std;

class Neuron {
private:
    int neuronIndex;
    int layerIndex;
    double outputValue;
    bool isBias;

    double activationFunction(double input);
    double activationFunctionDerivative(double input);
public:
    vector<Connection> connections;
    Neuron(int neuronIndex, int layerIndex, int numberOfOutputConnections, bool bias);

    double getOutputValue();
    void setOutputValue(double value);

    int getIndex(){ return neuronIndex; }
    void print();
    void printCFG(ofstream&);
};

#endif // NEURON_HH_INCLUSION_GUARD