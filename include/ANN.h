#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD

#include <vector>
#include <functional>
namespace ANN {

using weight_t = long double;
using data_t = long double;
using activationFunction_t = std::function<data_t(data_t)>;

template <typename W_t = weight_t, typename D_t = data_t>
class ANN {
private:
public:
};
}

#endif // ANN_HH_INCLUSION_GUARD
