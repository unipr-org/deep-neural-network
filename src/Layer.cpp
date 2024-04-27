#include "Layer.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

/**
 * @brief Layer constructor.
 *
 * Constructs a Layer object with the given index, number of neurons, and number of output connections per neuron.
 * It initializes the layer with neurons and their connections.
 *
 * @param index The index of the layer.
 * @param numberOfNeurons The number of neurons in the layer.
 * @param numberOfOutputConnections The number of output connections per neuron.
 */
Layer::Layer(int index, int numberOfNeurons, int numberOfOutputConnections){
    layerIndex = index;
    if(numberOfNeurons > 0){
        for(int i = 0; i < numberOfNeurons; ++i){
            neurons.push_back(Neuron(i, layerIndex, numberOfOutputConnections, false));
        }
        // Last layer has no bias
        if(numberOfOutputConnections > 0)
            neurons.push_back(Neuron(numberOfNeurons, layerIndex, numberOfOutputConnections, true)); // bias
    }
}

/**
 * @brief Prints information about the layer and its neurons.
 *
 * This method prints information about the layer, including its index,
 * and information about each neuron in the layer.
 */
void 
Layer::print(){
    cout << "Layer " << layerIndex << endl;
    for (unsigned i = 0; i < neurons.size(); ++i){
        neurons[i].print();
    }
}

/**
 * @brief Prints the configuration of the layer's neurons to a DOT file.
 *
 * This method prints the configuration of the layer's neurons to a DOT file format,
 * which can be used to visualize the connections between neurons in the layer.
 *
 * @param outFile The ofstream object representing the output file stream to write to.
 */
void 
Layer::printCFG(ofstream& outFile){
    for (unsigned i = 0; i < neurons.size(); ++i){
        neurons[i].printCFG(outFile);
    }
}