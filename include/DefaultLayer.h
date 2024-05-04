#ifndef DEFAULT_LAYER_INCLUDE_GUARD
#define DEFAULT_LAYER_INCLUDE_GUARD

#include "DefaultNeuron.h"
#include "Layer.h"
#include <cstddef>
#include <spdlog/spdlog.h>
#include <string>

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

	/**
	 * @brief Default constructor for DefaultLayer.
	 * 
	 * This constructor creates a DefaultLayer object using the default constructor.
	 */
	inline DefaultLayer() = default;
	
	/**
	 * @brief Explicit constructor for DefaultLayer with specified neurons.
	 * 
	 * @param neurons The vector of neurons to initialize the layer.
	 * 
	 * This constructor creates a DefaultLayer object with the specified neurons.
	 */
	inline explicit DefaultLayer(const neuron_vector_t &neurons) : _neurons(neurons) {}

	inline void setNeurons(const neuron_vector_t &neurons) override { _neurons = neurons; }

	inline void setNeuron(const neuron_t &neuron, size_t index) override {
		_neurons[index] = neuron;
	}

	inline neuron_vector_t &getNeurons() override { return _neurons; }
	inline const neuron_vector_t &getNeurons() const override { return _neurons; }
	inline size_t getSize() const override { return _neurons.size(); }

	inline void evaluate(const data_vector_t &input, data_vector_t &output) const override {
		size_t index = 0;

		std::string msg;
		msg = "[DefaultLayer::evaluate(const data_vector_t &)] input: ";
		msg += input;
		spdlog::debug(msg);

		for (auto it = _neurons.begin(); it != _neurons.end(); ++it, ++index) {
			msg = "[DefaultLayer::evaluate(const data_vector_t &)] Neuron [" +
				  std::to_string(index) + "]";
			spdlog::debug(msg);

			data_t result = it->evaluate(input);
			output[index] = result;
		}

		msg = "[DefaultLayer::evaluate(const data_vector_t &)] output: ";
		msg += output;
		spdlog::debug(msg);
	}

	inline void evaluate(const data_vector_t &input, data_vector_t &output, data_vector_t &neuronsPreactivations, data_vector_t &layerOutputs) const override {
		size_t index = 0;

		std::string msg;
		msg = "[DefaultLayer::evaluate(const data_vector_t &)] input: ";
		msg += input;
		spdlog::debug(msg);

		for (auto it = _neurons.begin(); it != _neurons.end(); ++it, ++index) {
			msg = "[DefaultLayer::evaluate(const data_vector_t &)] Neuron [" +
				  std::to_string(index) + "]";
			spdlog::debug(msg);

			data_t result = it->evaluate(input, neuronsPreactivations[index]);

			output[index] = result;
			layerOutputs[index] = result;
		}

		msg = "[DefaultLayer::evaluate(const data_vector_t &)] output: ";
		msg += output;
		spdlog::debug(msg);
	}


	inline neuron_t &operator[](size_t index) override { return _neurons[index]; }
	inline const neuron_t &operator[](size_t index) const override { return _neurons[index]; }

	inline std::ostream &operator<<(std::ostream &os) const override {
		size_t i = 0;

		for (auto it = _neurons.begin(); it != _neurons.end(); ++it, ++i) {
			os << "Neuron [" << i << "]: " << *it << std::endl;
		}

		return os;
	}
};
} // namespace ANN

#endif // DEFAULT_LAYER_INCLUDE_GUARD
