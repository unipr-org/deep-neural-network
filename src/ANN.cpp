#include "ANN.h"
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <iostream>
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
ANN::ANN(const vector<unsigned> &topology) {
  unsigned last = topology.size() - 1;
  for (unsigned i = 0; i < last; ++i) {
    layers.push_back(
        Layer(i, topology.at(i), topology.at(i + 1))); // inner layer
  }
  layers.push_back(Layer(last, topology.at(last), 0)); // last layer
}

/**
 * @brief Prints the Artificial Neural Network.
 *
 * This method prints the structure of the Artificial Neural Network (ANN) to
 * the standard output. It iterates over each layer of the ANN and prints the
 * details of each layer.
 */
void ANN::print() {
  spdlog::info("Printing Artificial Neural Network");

  for (unsigned i = 0; i < layers.size(); ++i) {
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
 * Additionally, it provides instructions on how to generate a PNG image from
 * the DOT file.
 */
void ANN::printCFG() {
  std::string folderPath = "./graph";
  std::string filename = folderPath + "/ann.dot";
  namespace fs = std::__fs::filesystem;

  if (!fs::exists(folderPath)) {
    if (!fs::create_directory(folderPath)) {
      spdlog::error("Unable to create directory {}", folderPath);
      return;
    }
  }

  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    spdlog::error("Unable to open {}", filename);
    return;
  }

  outFile << "digraph unix {" << endl;
  outFile << "rankdir=\"LR\";" << endl;
  outFile << "ranksep=4;" << endl;
  outFile << "nodesep=1.0;" << endl;

  for (unsigned i = 0; i < layers.size(); ++i)
    layers[i].printSubgraphCFG(outFile, layers.size());

  for (unsigned i = 0; i < layers.size(); ++i)
    layers[i].printCFG(outFile);

  outFile << "}" << endl;
  outFile.close();

  spdlog::info("{} generated successfully.", filename);
  spdlog::info("To generate PNG please use: ");
  cout << "dot -Tpng " + filename + " -o " + folderPath + "/ann.png && open " +
              folderPath + "/ann.png"
       << endl;
}
