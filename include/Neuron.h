#ifndef NEURON_HH_INCLUSION_GUARD
#define NEURON_HH_INCLUSION_GUARD

#include "Network.h"
#include <iostream>
#include <stddef.h>
#include <vector>

namespace ANN {

template <typename W_t = weight_t, typename D_t = data_t, typename AF_t = activationFunction_t>
class Neuron {
  public:
	using weight_t = W_t;
	using data_t = D_t;
	using activationFunction_t = AF_t;

	using weight_vector_t = std::vector<weight_t>;
	using data_vector_t = std::vector<data_t>;

	// Setter
	inline virtual void setWeights(const weight_vector_t &) = 0;
	inline virtual void setActivationFunction(const activationFunction_t &) = 0;

	// Getter
	inline virtual weight_vector_t &getWeights() = 0;
	inline virtual const activationFunction_t &getActivationFunction() const = 0;

	// inline virtual void evaluate(const data_vector_t&, data_t&) const = 0;
	inline virtual data_t evaluate(const data_vector_t &) const = 0;
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;
	inline virtual ~Neuron() {}
};

template <typename W_t, typename D_t>
std::ostream &operator<<(std::ostream &os, const Neuron<W_t, D_t> *n) {
	return n->operator<<(os);
}
template <typename W_t, typename D_t>
std::ostream &operator<<(std::ostream &os, const Neuron<W_t, D_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // NEURON_HH_INCLUSION_GUARD
