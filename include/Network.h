#ifndef NETWORK_HH_INCLUSION_GUARD
#define NETWORK_HH_INCLUSION_GUARD

#include <functional>
namespace ANN {

using weight_t = long double;
using data_t = long double;
using activationFunction_t = std::function<data_t(data_t)>;

template <typename W_t = weight_t, typename D_t = data_t> class Network {
  private:
  public:
};
} // namespace ANN

#endif // NETWORK_HH_INCLUSION_GUARD
