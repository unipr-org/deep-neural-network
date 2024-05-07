#ifndef LAYER_INCLUDE_GUARD
#define LAYER_INCLUDE_GUARD

#include "Neuron.h"
#include <cstddef>
#include <ostream>

namespace ANN {

/**
 * @brief A template class representing a layer in an artificial neural network.
 *
 * @tparam Neuron_t The type of neuron used in the layer.
 */
template <typename Neuron_t = Neuron<>> class Layer {
  public:
	using neuron_t = Neuron_t; /**< Type definition for neuron type used in the layer. */
	using weight_t = typename neuron_t::weight_t; /**< Type definition for neuron weight type. */
	using data_t = typename neuron_t::data_t;	  /**< Type definition for neuron data type. */
	using activationFunction_t =
		typename neuron_t::activationFunction_t; /**< Type definition for neuron activation function
													type. */
	using weight_vector_t =
		typename neuron_t::weight_vector_t; /**< Type definition for vector of weights. */
	using data_vector_t =
		typename neuron_t::data_vector_t; /**< Type definition for vector of data. */
	using v_data_vector_t =
		typename neuron_t::v_data_vector_t; /**< Type definition for vector of vectors of data. */
	using neuron_vector_t = std::vector<neuron_t>; /**< Type definition for vector of neurons. */

	/**
	 * @brief Sets the neurons of the layer.
	 *
	 * @param neurons The vector of neurons to set.
	 */
	inline virtual void setNeurons(const neuron_vector_t &neurons) = 0;

	/**
	 * @brief Sets a neuron at a specific index in the layer.
	 *
	 * @param neuron The neuron to set.
	 * @param index The index at which to set the neuron.
	 */
	inline virtual void setNeuron(const neuron_t &neuron, size_t index) = 0;

	/**
	 * @brief Gets the neurons of the layer (const version).
	 *
	 * @return Const reference to the vector of neurons.
	 */
	inline virtual const neuron_vector_t &getNeurons() const = 0;

	/**
	 * @brief Gets the neurons of the layer.
	 *
	 * @return Reference to the vector of neurons.
	 */
	inline virtual neuron_vector_t &getNeurons() = 0;

	/**
	 * @brief Gets the size of the layer.
	 *
	 * @return The number of neurons in the layer.
	 */
	inline virtual size_t getSize() const = 0;

	/**
	 * @brief Evaluates the layer with the given input data.
	 *
	 * @param input The input data vector.
	 * @param output The output data vector.
	 */
	inline virtual void evaluate(const data_vector_t &input, data_vector_t &output) const = 0;

	/**
	 * @brief Evaluates the layer with the given input and computes the pre-activation values of
	 * neurons.
	 *
	 * @param input The input vector to the layer.
	 * @param output The output vector of the layer (output parameter).
	 * @param neuronsPreactivations The pre-activation values of neurons in the layer (output
	 * parameter).
	 * @param layersOutputs The output values of neurons in all layers (output parameter).
	 */
	inline virtual void evaluate(const data_vector_t &input, data_vector_t &output,
								 data_vector_t &preActivation) const = 0;

	/**
	 * @brief Accesses the neuron at the specified index.
	 *
	 * @param index The index of the neuron to access.
	 * @return Reference to the neuron at the specified index.
	 */
	inline virtual neuron_t &operator[](size_t index) = 0;

	/**
	 * @brief Accesses the neuron at the specified index (const version).
	 *
	 * @param index The index of the neuron to access.
	 * @return Const reference to the neuron at the specified index.
	 */
	inline virtual const neuron_t &operator[](size_t index) const = 0;

	/**
	 * @brief Outputs a textual representation of the layer.
	 *
	 * @param os The output stream to write to.
	 * @return Reference to the output stream.
	 */
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;

	/**
	 * @brief Virtual destructor for Layer class.
	 */
	inline virtual ~Layer() {}
};

/**
 * @brief Output operator for Layer pointers.
 *
 * @tparam Neuron_t The type of neuron used in the layer.
 * @param os The output stream to write to.
 * @param n Pointer to the layer object.
 * @return Reference to the output stream.
 */
template <typename Neuron_t> std::ostream &operator<<(std::ostream &os, const Layer<Neuron_t> *n) {
	return n->operator<<(os);
}

/**
 * @brief Output operator for Layer objects.
 *
 * @tparam Neuron_t The type of neuron used in the layer.
 * @param os The output stream to write to.
 * @param n The layer object.
 * @return Reference to the output stream.
 */
template <typename Neuron_t> std::ostream &operator<<(std::ostream &os, const Layer<Neuron_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // LAYER_INCLUDE_GUARD
