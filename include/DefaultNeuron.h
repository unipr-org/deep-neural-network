#ifndef DEFAULT_NEURON_INCLUDE_GUARD
#define DEFAULT_NEURON_INCLUDE_GUARD

#include "Neuron.h"
#include <cstddef>
#include <numeric>

namespace ANN {

class DefaultNeuron : public Neuron<> {
  public:
	using typename Neuron<>::weight_vector_t;
	using typename Neuron<>::data_vector_t;
	using typename Neuron<>::activationFunction_t;

  private:
	weight_vector_t _weights;
	activationFunction_t _activationFunction;

  public:
	DefaultNeuron() = delete;

	inline DefaultNeuron(const weight_vector_t &weights,
						 const activationFunction_t &activationFunction)
		: _weights(weights), _activationFunction(activationFunction) {}

	// Setter
	inline void setWeights(const weight_vector_t &weights) override { _weights = weights; }

	inline void setActivationFunction(const activationFunction_t &activationFunction) override {
		_activationFunction = activationFunction;
	}

	// Getter
	inline weight_vector_t &getWeights() override { return _weights; }
	inline const weight_vector_t &getWeights() const override { return _weights; }
	inline const activationFunction_t &getActivationFunction() const override {
		return _activationFunction;
	}

	inline data_t evaluate(const data_vector_t &input) const override {
		data_t innerProduct =
			std::inner_product(input.begin(), input.end(), _weights.begin(), data_t(0));
		return _activationFunction(innerProduct);
	}

	inline weight_t &operator[](size_t index) override { return _weights[index]; }
	inline const weight_t &operator[](size_t index) const override { return _weights[index]; }

	inline std::ostream &operator<<(std::ostream &os) const override {
		size_t index = 0;

		os << "Weights: {";
		for (auto i = _weights.begin(); i != _weights.end(); ++i, ++index) {
			os << *i;
			if (index != _weights.size() - 1)
				os << ", ";
		}
		os << "}";
		return os;
	}
};

} // namespace ANN

#endif // NEURON_INCLUDE_GUARD
