#ifndef LOADER_INCLUDE_GUARD
#define LOADER_INCLUDE_GUARD

#include "Network.h"

namespace Utils {

/**
 * @brief A template class representing a loader for neural networks.
 *
 * @tparam Network_t The type of neural network to load.
 */
template <typename Network_t = ANN::Network<>> class Loader {
  public:
	using network_t = Network_t; /**< Type definition for the neural network type. */

	/**
	 * @brief Saves the status of the neural network.
	 *
	 * @param net The neural network to save.
	 */
	inline virtual void saveStatus(const network_t &net) const = 0;

	/**
	 * @brief Loads a neural network from the specified path.
	 *
	 * @param path The path from which to load the neural network. Default is "./model/status.txt".
	 * @return A pointer to the loaded neural network.
	 */
	inline virtual network_t *loadNetwork(const std::string &path = "./model/status.txt") const = 0;
};
} // namespace Utils

#endif // LOADER_INCLUDE_GUARD
