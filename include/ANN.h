#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD 1

#include <vector>
#include "Layer.h"
using namespace std;

class ANN {
private:
    vector<Layer> layers;

public:
    ANN(const vector<unsigned> &topology);
    void print();
    void printCFG();
};

#endif // ANN_HH_INCLUSION_GUARD