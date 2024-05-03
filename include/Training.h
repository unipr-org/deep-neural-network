#ifndef TRAINING_INCLUDE_GUARD
#define TRAINING_INCLUDE_GUARD

#include "Network.h"
#include <fstream>


namespace Training {

/**
 * @brief Alias for the type of neural network used for training.
 */
using network_t = ANN::Network<>;

/**
 * @brief Alias for the type of data stream used for training.
 */
using stream_t = std::fstream;

} // namespace Training


#endif