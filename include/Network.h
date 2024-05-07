#ifndef NETWORK_INCLUDE_GUARD
#define NETWORK_INCLUDE_GUARD

#include "Layer.h"
#include <vector>

namespace ANN {

/**
 * @brief A template class representing a neural network.
 *
 * @tparam Layer_t The type of layer used in the network.
 */
template <typename Layer_t = Layer<>> class Network {
  public:
	using layer_t = Layer_t; /**< Type definition for layer type used in the network. */
	using layer_vector_t = std::vector<layer_t>; /**< Type definition for vector of layers. */
	using data_t =
		typename layer_t::data_t; /**< Type definition for data type used in the network. */
	using data_vector_t =
		typename layer_t::data_vector_t; /**< Type definition for vector of data. */
	using data_vv_t =
		typename layer_t::v_data_vector_t; /**< Type definition for vector of vectors of data. */
	using activationFunction_t = typename layer_t::activationFunction_t; 

	/**
	 * @brief Sets the layers of the network.
	 *
	 * @param layers The vector of layers to set.
	 */
	inline virtual void setLayers(const layer_vector_t &layers) = 0;

	/**
	 * @brief Sets a layer at a specific index in the network.
	 *
	 * @param layer The layer to set.
	 * @param index The index at which to set the layer.
	 */
	inline virtual void setLayer(const layer_t &layer, size_t index) = 0;

	/**
	 * @brief Gets the layers of the network.
	 *
	 * @return Reference to the vector of layers.
	 */
	inline virtual layer_vector_t &getLayers() = 0;

	/**
	 * @brief Gets the layers of the network (const version).
	 *
	 * @return Const reference to the vector of layers.
	 */
	inline virtual const layer_vector_t &getLayers() const = 0;

	/**
	 * @brief Gets the size of the network.
	 *
	 * @return The number of layers in the network.
	 */
	inline virtual size_t getSize() const = 0;

	/**
	 * @brief Evaluates the network with the given input data.
	 *
	 * @param input The input data vector.
	 * @param output The output data vector.
	 */
	inline virtual void evaluate(const data_vector_t &input, data_vector_t &output) const = 0;

	/**
	 * @brief Evaluates the network with the given input data and computes pre-activation values.
	 *
	 * @param input The input data vector.
	 * @param output The output data vector.
	 * @param neuronsPreactivations The pre-activation values of neurons in all layers (output
	 * parameter).
	 * @param layersOutputs The output values of neurons in all layers (output parameter).
	 */
	inline virtual void evaluate(const data_vector_t &input, data_vv_t &output,
								 data_vv_t &preActivation) const = 0;

	/**
	 * @brief Gets the status of the neural network.
	 *
	 * @return A string representing the status of the neural network.
	 *
	 * This function retrieves the status of the neural network and returns it as a string.
	 */
	inline virtual std::string getStatus() const = 0;

	/**
	 * @brief Accesses the layer at the specified index.
	 *
	 * @param index The index of the layer to access.
	 * @return Reference to the layer at the specified index.
	 */
	inline virtual layer_t &operator[](size_t index) = 0;

	/**
	 * @brief Accesses the layer at the specified index (const version).
	 *
	 * @param index The index of the layer to access.
	 * @return Const reference to the layer at the specified index.
	 */
	inline virtual const layer_t &operator[](size_t index) const = 0;

	/**
	 * @brief Outputs a textual representation of the network.
	 *
	 * @param os The output stream to write to.
	 * @return Reference to the output stream.
	 */
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;

	/**
	 * @brief Virtual destructor for Network class.
	 */
	inline virtual ~Network() {}
};

/**
 * @brief Output operator for Network pointers.
 *
 * @tparam Layer_t The type of layer used in the network.
 * @param os The output stream to write to.
 * @param n Pointer to the network object.
 * @return Reference to the output stream.
 */
template <typename Layer_t> std::ostream &operator<<(std::ostream &os, const Network<Layer_t> *n) {
	return n->operator<<(os);
}

/**
 * @brief Output operator for Network objects.
 *
 * @tparam Layer_t The type of layer used in the network.
 * @param os The output stream to write to.
 * @param n The network object.
 * @return Reference to the output stream.
 */
template <typename Layer_t> std::ostream &operator<<(std::ostream &os, const Network<Layer_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // NETWORK_INCLUDE_GUARD
