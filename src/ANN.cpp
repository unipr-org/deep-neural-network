#include "include/ANN.h"
#include <iostream>
#include <vector>

using namespace std;

ANN::ANN(const vector<int> &topology){
    int last = topology.size() - 1;
    for(int i = 0; i < last; ++i){
        layers.push_back(Layer(i, topology.at(i), topology.at(i + 1))); // inner layer
    }
    layers.push_back(Layer(last, topology.at(last), 0)); // last layer
}
