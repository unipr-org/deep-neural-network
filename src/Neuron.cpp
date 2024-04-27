#include "Neuron.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

/**
 * @brief Neuron constructor.
 *
 * Constructs a Neuron object with the given neuron index, layer index,
 * and number of output connections. It initializes the neuron's attributes
 * and creates output connections to other neurons.
 *
 * @param neuronI The index of the neuron.
 * @param layerI The index of the layer to which the neuron belongs.
 * @param numberOfOutputConnections The number of output connections for the neuron.
 */
Neuron::Neuron(int neuronI, int layerI, int numberOfOutputConnections, bool bias){
    neuronIndex = neuronI;
    layerIndex = layerI;
    isBias = bias;
    if(numberOfOutputConnections > 0){
        for(int i = 0; i < numberOfOutputConnections; ++i){
            // cout << "Adding connection from " << neuronIndex << " to " << i << endl;
            connections.push_back(Connection(i));
            if(isBias)
                setOutputValue(-1); // bias
            else 
                setOutputValue((double) rand() / RAND_MAX * 2.0 - 1.0); // random
        }
    }
}

/**
 * @brief Returns the output value of the neuron.
 *
 * @return The output value of the neuron.
 */
double 
Neuron::getOutputValue(){
    return outputValue;
}

/**
 * @brief Sets the output value of the neuron.
 *
 * @param value The value to set as the output value of the neuron.
 */
void 
Neuron::setOutputValue(double value){
    outputValue = value;
}

/**
 * @brief Prints information about the neuron and its connections.
 *
 * This method prints information about the neuron, including its index,
 * and its connections to other neurons along with their output values.
 */
void 
Neuron::print(){
    cout << "\tNeuron " << neuronIndex << endl;
    for (auto connection : connections) {
        cout << "\t- (o: " << outputValue << ", ";
        connection.print();
        cout << ")";
        cout << endl;
    }
}

/**
 * @brief Prints the configuration of the neuron to a DOT file.
 *
 * This method prints the configuration of the neuron to a DOT file format,
 * which can be used to visualize the connections of the neuron in a network.
 *
 * @param outFile The ofstream object representing the output file stream to write to.
 */
void 
Neuron::printCFG(ofstream& outFile){
    string name = to_string(layerIndex) + "_" + to_string(neuronIndex);

    if(isBias)
        outFile << "\"node" << name << "\" [label=\"bias\"];" << endl;
    else
        outFile << "\"node" << name << "\" [label=\"" << name << "\"];" << endl;
    
    for (auto connection : connections) {
        string nextName = to_string((layerIndex + 1)) + "_" + to_string(connection.getIndexNeuronLinked());

        outFile << "\"node" << name << "\" -> " << "\"node" << nextName << "\";" << endl;
    }
}