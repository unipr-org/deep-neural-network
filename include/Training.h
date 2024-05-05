#ifndef TRAINING_INCLUDE_GUARD
#define TRAINING_INCLUDE_GUARD

#include "DefaultNetwork.h"
#include <fstream>

namespace Training {

/**
 * @brief Alias for the type of neural network used for training.
 */
using network_t = ANN::DefaultNetwork;

/**
 * @brief Alias for the type of data stream used for training.
 */
using stream_t = std::ifstream;

/**
 * @brief Alias for the type representing the tolerance for convergence during training.
 */
using tolerance_t = long double;

/**
 * @brief Alias for the type representing the step size for training.
 */
using step_t = long double;

} // namespace Training

#endif
