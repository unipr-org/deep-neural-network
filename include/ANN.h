#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD

#include <functional>

namespace ANN {

using weight_t = long double;
using data_t = long double;
using activationFunction_t = std::function<data_t(data_t)>;
} // namespace ANN

#endif // ANN_HH_INCLUSION_GUARD
