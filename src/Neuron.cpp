#include "include/Neuron.h"
#include <iostream>
#include <vector>

using namespace std;

Neuron::Neuron(int index, int numberOfOutputConnections){
    neuronIndex = index;
    if(numberOfOutputConnections > 0){
        for(int i = 0; i < numberOfOutputConnections; ++i){
            cout << "Adding connection from " << neuronIndex << " to " << i << endl;
            connections.push_back(Connection(i));
        }
    }
}
