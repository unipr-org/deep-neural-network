#include "ANN.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace std;

/**
 * @brief Initializes an ANN object with the specified topology.
 *
 * This constructor initializes an ANN object with a given topology.
 * The topology is a sequence of integers representing the number of neurons
 * in each layer of the ANN.
 *
 * @param topology A constant reference to a vector of integers representing
 *                 the ANN topology.
 */
ANN::ANN(const vector<unsigned> &topology){
    unsigned last = topology.size() - 1;
    for(unsigned i = 0; i < last; ++i){
        layers.push_back(Layer(i, topology.at(i), topology.at(i + 1))); // inner layer
    }
    layers.push_back(Layer(last, topology.at(last), 0)); // last layer
}

/**
 * @brief Prints the Artificial Neural Network.
 *
 * This method prints the structure of the Artificial Neural Network (ANN) to the standard output.
 * It iterates over each layer of the ANN and prints the details of each layer.
 */
void 
ANN::print(){
    cout << endl
         << "Printing Artificial Neural Network:" << endl
         << endl;

    for (unsigned i = 0; i < layers.size(); ++i){
        layers[i].print();
        cout << endl;
    }
    cout << endl;
}

/**
 * @brief Prints the configuration of the ANN to a DOT file.
 *
 * This method prints the configuration of the Artificial Neural Network (ANN)
 * to a DOT file format, which can be used to visualize the network structure.
 * The DOT file is saved in a directory named "graph" and named "ann.dot".
 * Additionally, it provides instructions on how to generate a PNG image from the DOT file.
 */
void 
ANN::printCFG(){
    std::string folderPath = "./graph";
    std::string filename = folderPath + "/ann.dot";
    namespace fs = std::__fs::filesystem;

    if (!fs::exists(folderPath)) {
        if (!fs::create_directory(folderPath)) {
            std::cerr << "Error: Unable to create directory " << folderPath << endl;
            return;
        }
    }
    
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cerr << "Error: Unable to open " << filename << endl;
        return;
    }
    
    outFile << "digraph unix {" << endl;
    outFile << "rankdir=\"LR\";" << endl;
    outFile << "ranksep=3; // Imposta lo spazio tra i livelli a 3" << endl;
    outFile << "nodesep=1.0; // Imposta lo spazio tra i nodi a 1.0" << endl;
    
    for (unsigned i = 0; i < layers.size(); ++i){
        layers[i].printCFG(outFile);
    }

    outFile << "}" << endl;
    outFile.close();
    cout << filename << " generated successfully." << endl;
    cout << "To generate PNG please use: " << endl;
    cout << "dot -Tpng " + filename + " -o " + folderPath + "/ann.png && open " + folderPath + "/ann.png" << endl;
}