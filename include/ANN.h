#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD 1

#include "Layer.h"
#include <vector>

class ANN {
private:
  std::vector<Layer> layers;

public:
  ANN(const std::vector<unsigned> &topology);
  void print();
  void printCFG();
};

#endif // ANN_HH_INCLUSION_GUARD
