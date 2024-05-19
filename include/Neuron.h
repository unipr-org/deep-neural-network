#ifndef NEURON_INCLUDE_GUARD
#define NEURON_INCLUDE_GUARD

#include <ostream>
#include <stddef.h>
#include <vector>

#include "ANN.h"

namespace ANN {

/**
 * @brief A template class representing a neuron in an artificial neural network.
 *
 * @tparam W_t The type of weights associated with the neuron.
 * @tparam D_t The type of data handled by the neuron.
 * @tparam AF_t The type of activation function used by the neuron.
 */
template <typename W_t = weight_t, typename D_t = data_t, typename AF_t = activationFunction_t>
class Neuron {
  public:
	using weight_t = W_t;			   /**< Type definition for neuron weights. */
	using data_t = D_t;				   /**< Type definition for neuron data. */
	using activationFunction_t = AF_t; /**< Type definition for neuron activation function. */

	using weight_vector_t = std::vector<weight_t>; /**< Type definition for vector of weights. */
	using data_vector_t = std::vector<data_t>;	   /**< Type definition for vector of data. */
	using v_data_vector_t =
		std::vector<data_vector_t>; /**< Type definition for vector of vectors of data. */

	/**
	 * @brief Sets the weights of the neuron.
	 *
	 * @param weights The vector of weights to set.
	 */
	inline virtual void setWeights(const weight_vector_t &) = 0;

	/**
	 * @brief Sets the activation function of the neuron.
	 *
	 * @param activationFunction The activation function to set.
	 */
	// inline virtual void setActivationFunction(const activationFunction_t &) = 0;

	/**
	 * @brief Sets the activation function ID of the neuron.
	 *
	 * @param activationFunction The activation function ID to set.
	 */
	inline virtual void setActivationFunctionId(const ActivationFunctionID &) = 0;

	/**
	 * @brief Gets the weights of the neuron.
	 *
	 * @return Reference to the vector of weights.
	 */
	inline virtual weight_vector_t &getWeights() = 0;

	/**
	 * @brief Gets the weights of the neuron (const version).
	 *
	 * @return Const reference to the vector of weights.
	 */
	inline virtual const weight_vector_t &getWeights() const = 0;

	/**
	 * @brief Gets the size of the neuron.
	 *
	 * @return The number of weights in the neuron.
	 */
	inline virtual size_t getSize() const = 0;

	/**
	 * @brief Gets the activation function of the neuron (const version).
	 *
	 * @return Const reference to the activation function.
	 */
	inline virtual const activationFunction_t &getActivationFunction() const = 0;

	/**
	 * @brief Gets the activation function ID of the neuron (const version).
	 *
	 * @return Const reference to the activation function ID.
	 */
	inline virtual const ActivationFunctionID &getActivationFunctionID() const = 0;

	/**
	 * @brief Accesses the weight at the specified index.
	 *
	 * @param index The index of the weight to access.
	 * @return Reference to the weight at the specified index.
	 */
	inline virtual weight_t &operator[](size_t index) = 0;

	/**
	 * @brief Accesses the weight at the specified index (const version).
	 *
	 * @param index The index of the weight to access.
	 * @return Const reference to the weight at the specified index.
	 */
	inline virtual const weight_t &operator[](size_t index) const = 0;

	// inline virtual void evaluate(const data_vector_t&, data_t&) const = 0;

	/**
	 * @brief Evaluates the neuron with the given input data.
	 *
	 * @param input The input data vector.
	 * @return The result of the neuron's evaluation.
	 */
	inline virtual data_t evaluate(const data_vector_t &input) const = 0;

	/**
	 * @brief Evaluates the neuron with the given input and calculates the pre-activation value.
	 *
	 * @param input The input vector to the neuron.
	 * @param neuronPreactivation The pre-activation value of the neuron (output parameter).
	 * @return The output value of the neuron after applying the activation function.
	 */
	inline virtual void evaluate(const data_vector_t &input, data_t &output,
								 data_t &pre_activation) const = 0;

	/**
	 * @brief Outputs a textual representation of the neuron.
	 *
	 * @param os The output stream to write to.
	 * @return Reference to the output stream.
	 */
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;

	/**
	 * @brief Virtual destructor for Neuron class.
	 */
	inline virtual ~Neuron() {}
};

/**
 * @brief Output operator for Neuron pointers.
 *
 * @tparam W_t The type of weights associated with the neuron.
 * @tparam D_t The type of data handled by the neuron.
 * @param os The output stream to write to.
 * @param n Pointer to the neuron object.
 * @return Reference to the output stream.
 */
template <typename W_t, typename D_t>
std::ostream &operator<<(std::ostream &os, const Neuron<W_t, D_t> *n) {
	return n->operator<<(os);
}

/**
 * @brief Output operator for Neuron objects.
 *
 * @tparam W_t The type of weights associated with the neuron.
 * @tparam D_t The type of data handled by the neuron.
 * @param os The output stream to write to.
 * @param n The neuron object.
 * @return Reference to the output stream.
 */
template <typename W_t, typename D_t>
std::ostream &operator<<(std::ostream &os, const Neuron<W_t, D_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // NEURON_INCLUDE_GUARD
