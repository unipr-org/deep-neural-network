#ifndef LAYER_HH_INCLUSION_GUARD
#define LAYER_HH_INCLUSION_GUARD

#include "Neuron.h"
#include <cstddef>
#include <ostream>

namespace ANN {

template <typename Neuron_t = Neuron<>> class Layer {
  public:
	using neuron_t = Neuron_t;
	using weight_t = typename neuron_t::weight_t;
	using data_t = typename neuron_t::data_t;
	using activationFunction_t = typename neuron_t::activationFunction_t;
	using weight_vector_t = typename neuron_t::weight_vector_t;
	using data_vector_t = typename neuron_t::data_vector_t;

	using neuron_vector_t = std::vector<neuron_t>;

	// Setter
	inline virtual void setNeurons(const neuron_vector_t &neurons) = 0;
	inline virtual void setNeuron(const neuron_t &neuron, size_t index) = 0;

	// Getter
	inline virtual const neuron_vector_t &getNeurons() const = 0;
	inline virtual neuron_vector_t &getNeurons() = 0;
	inline virtual size_t getSize() const = 0;

	inline virtual void evaluate(const data_vector_t &input, data_vector_t &output) const = 0;


	inline virtual neuron_t &operator[](size_t index) = 0;
	inline virtual const neuron_t &operator[](size_t index) const = 0;
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;

	inline virtual ~Layer() {}
};

template <typename Neuron_t> std::ostream &operator<<(std::ostream &os, const Layer<Neuron_t> *n) {
	return n->operator<<(os);
}

template <typename Neuron_t> std::ostream &operator<<(std::ostream &os, const Layer<Neuron_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // LAYER_HH_INCLUSION_GUARD
