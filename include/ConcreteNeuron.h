#ifndef CONCRETE_NEURON_HH_INCLUSION_GUARD
#define CONCRETE_NEURON_HH_INCLUSION_GUARD

#include "Neuron.h"
#include <numeric>

namespace ANN {

class ConcreteNeuron : public Neuron<> {
	using typename Neuron<>::weight_vector_t;
	using typename Neuron<>::data_vector_t;
	using typename Neuron<>::activationFunction_t;

  private:
	weight_vector_t _weights;
	activationFunction_t _activationFunction;

  public:
	ConcreteNeuron() = delete;

	inline ConcreteNeuron(const weight_vector_t &weights,
						  const activationFunction_t &activationFunction)
		: _weights(weights), _activationFunction(activationFunction) {}

	// Setter
	inline void setWeights(const weight_vector_t &weights) override { _weights = weights; }

	inline void setActivationFunction(const activationFunction_t &activationFunction) override {
		_activationFunction = activationFunction;
	}

	// Getter
	inline weight_vector_t &getWeights() override { return _weights; }
	inline const activationFunction_t &getActivationFunction() const override {
		return _activationFunction;
	}

	inline data_t evaluate(const data_vector_t &input) const override {
		data_t innerProduct =
			std::inner_product(input.begin(), input.end(), _weights.begin(), data_t(0));
		return _activationFunction(innerProduct);
	}

	inline std::ostream &operator<<(std::ostream &os) const override {
		os << "Weights: [\n";
		for (auto i = _weights.begin(); i != _weights.end(); ++i)
			os << *i << " ";
		os << "\n]";
		return os;
	}
};

} // namespace ANN

#endif // NEURON_HH_INCLUSION_GUARD
