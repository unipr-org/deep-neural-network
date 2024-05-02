#ifndef LOADER_INCLUDE_GUARD
#define LOADER_INCLUDE_GUARD

#include "Network.h"
#include <vector>

namespace ANN {
template <typename Network_t = Network<>> class Loader{
    public:
        using network_t = Network_t; 
        
        inline virtual void saveStatus(Network& net) const = 0;
        inline virtual Network& loadStatus() const = 0;
};
} // namespace ANN

#endif // LOADER_INCLUDE_GUARD
