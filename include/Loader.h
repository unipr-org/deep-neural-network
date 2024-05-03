#ifndef LOADER_INCLUDE_GUARD
#define LOADER_INCLUDE_GUARD

#include "Network.h"
#include <vector>

namespace Utils {
template<typename Network_t = ANN::Network<>> 
class Loader{
    public:
        using network_t = Network_t;

        inline virtual void saveStatus(const network_t& net) const = 0;        
        inline virtual network_t* loadNetwork(const std::string& path = "./model/status.txt") const = 0;
};
} // namespace Utils

#endif // LOADER_INCLUDE_GUARD
