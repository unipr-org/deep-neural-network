#ifndef DEFAULT_LAYER_INCLUSION_GUARD
#define DEFAULT_LAYER_INCLUSION_GUARD

#include "DefaultNeuron.h"
#include "Layer.h"
#include <cstddef>

namespace ANN {

class DefaultLayer : public Layer<DefaultNeuron> {
  public:
	using Layer<DefaultNeuron>::neuron_t;
	using Layer<DefaultNeuron>::weight_t;
	using Layer<DefaultNeuron>::data_t;
	using Layer<DefaultNeuron>::activationFunction_t;
	using Layer<DefaultNeuron>::weight_vector_t;
	using Layer<DefaultNeuron>::data_vector_t;
	using Layer<DefaultNeuron>::neuron_vector_t;

  private:
	neuron_vector_t _neurons;

  public:
	DefaultLayer() = delete;
	inline DefaultLayer(const neuron_vector_t &neurons) : _neurons(neurons) {}

	inline void setNeurons(const neuron_vector_t &neurons) override { _neurons = neurons; }

	inline void setNeuron(const neuron_t &neuron, size_t index) override {
		_neurons[index] = neuron;
	}

	inline neuron_vector_t &getNeurons() override { return _neurons; }
	inline size_t getSize() const override { return _neurons.size(); }

	inline void evaluateOutput(const data_vector_t &input, data_vector_t &output) const override {
		size_t index = 0;
		for (auto it = _neurons.begin(); it != _neurons.end(); ++it, ++index) {
			data_t result = it->evaluate(input);
			output[index] = result;
		}
	}

	inline neuron_t &operator[](size_t index) override { return _neurons[index]; }

	inline std::ostream &operator<<(std::ostream &os) const override {
		size_t i = 0;

		for (auto it = _neurons.begin(); it != _neurons.end(); ++it, ++i) {
			os << "[" << i << "]: " << *it << std::endl;
		}

		return os;
	}
};
} // namespace ANN

#endif // DEFAULT_LAYER_INCLUSION_GUARD
