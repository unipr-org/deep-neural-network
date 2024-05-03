#ifndef TRAINING_INCLUDE_GUARD
#define TRAINING_INCLUDE_GUARD

#include "Network.h"
#include <fstream>

namespace Training {
    using network_t = ANN::Network<>;
    using stream_t = std::fstream;
} // namespace Training


#endif