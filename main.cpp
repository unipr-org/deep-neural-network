#include <iostream>
#include <vector>
#include "ANN.h"

//! To compile:
//! g++ -o main.out main.cpp ./src/ANN.cpp ./src/Connection.cpp ./src/Layer.cpp ./src/Neuron.cpp -I./include -O1

int main(){
    vector<unsigned> topology = {3, 5, 5, 1};

    ANN ann(topology);

    ann.print();
    ann.printCFG();

    return 0;
}