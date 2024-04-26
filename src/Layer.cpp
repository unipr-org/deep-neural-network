#include "include/Layer.h"
#include <iostream>
#include <vector>

using namespace std;

Layer::Layer(int index, int numberOfNeurons, int numberOfOutputConnections){
    layerIndex = index;
    if(numberOfNeurons > 0){
        for(int i = 0; i < numberOfNeurons; ++i){
            neurons.push_back(Neuron(i, numberOfOutputConnections));
        }
    }
}
