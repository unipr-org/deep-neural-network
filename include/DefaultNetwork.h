#ifndef DEFAULT_NETWORK_INCLUDE_GUARD
#define DEFAULT_NETWORK_INCLUDE_GUARD

#include "ANN.h"
#include "DefaultLayer.h"
#include "DefaultNeuron.h"
#include "Layer.h"
#include "Network.h"
#include "Neuron.h"
#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/stat.h>

namespace ANN {

class DefaultNetwork : public Network<DefaultLayer> {
  public:
	using Network<DefaultLayer>::data_t;
	using Network<DefaultLayer>::data_vector_t;
	using Network<DefaultLayer>::layer_t;
	using Network<DefaultLayer>::layer_vector_t;
	using Network<DefaultLayer>::activationFunction_t;
	activationFunction_t _activationFunctionDerivative = ANN::tanh_d;

  private:
	layer_vector_t _layers;

  public:
	/**
	 * @brief Default constructor for DefaultNetwork.
	 *
	 * This constructor creates a DefaultNetwork object using the default
	 * constructor.
	 */
	inline DefaultNetwork() = default;

	/**
	 * @brief Explicit constructor for DefaultNetwork with specified layers.
	 *
	 * @param layers The vector of layers to initialize the network.
	 *
	 * This constructor creates a DefaultNetwork object with the specified layers.
	 */
	inline explicit DefaultNetwork(const layer_vector_t &layers) : _layers(layers) {}

	inline void setLayers(const layer_vector_t &layers) override { _layers = layers; }

	inline void setLayer(const layer_t &layer, size_t index) override { _layers[index] = layer; }

	inline layer_vector_t &getLayers() override { return _layers; }

	inline const layer_vector_t &getLayers() const override { return _layers; }

	inline size_t getSize() const override { return _layers.size(); }

	inline void evaluate(const data_vector_t &input, data_vector_t &output) const override {
		data_vector_t layerInput;
		data_vector_t layerOutput(input);
		layerOutput.push_back(-1); // bias

		size_t index = 0;
		std::string msg;
		msg = "[DefaultNetwork::evaluate(const data_vector_t &)] input: ";
		msg += input;
		spdlog::debug(msg);

		for (auto it = _layers.begin(); it != _layers.end(); ++it, ++index) {
			layerInput = std::move(layerOutput);

			size_t layer_size = it->getSize();
			layerOutput = data_vector_t(layer_size + 1);
			layerOutput[layer_size] = -1; // bias

			msg = "[DefaultNetwork::evaluate(const data_vector_t &)] Layer [" +
				  std::to_string(index) + "]";
			spdlog::debug(msg);

			it->evaluate(layerInput, layerOutput);
		}

		layerOutput.pop_back();

		output = std::move(layerOutput);
	}

	inline void evaluate(const data_vector_t &input, data_vv_t &output,
						 data_vv_t &preActivation) const override {

		std::string msg;
		msg = "[DefaultNetwork::evaluate(const data_vector_t &input, data_vv_t "
			  "&output, data_vv_t "
			  "preActivation)] input: ";
		msg += input;
		spdlog::debug(msg);

		size_t index = 0;
		data_vector_t lastInput;
		data_vector_t lastOutput(input);
		lastOutput.push_back(-1);

		for (auto l = _layers.begin(); l != _layers.end(); ++l, ++index) {

			lastInput = std::move(lastOutput);

			// blanck empty vector to store the output of the new layer.
			// `size+1` refers to the input value of the bias in the next iteration.
			size_t layer_size = l->getSize();
			lastOutput = data_vector_t(layer_size + 1);
			lastOutput.back() = -1;

			msg = "[DefaultNetwork::evaluate(const data_vector_t &input, data_vv_t "
				  "&output, "
				  "data_vv_t preActivation)] Layer [" +
				  std::to_string(index) + "]";
			spdlog::debug(msg);

			l->evaluate(lastInput, lastOutput, preActivation[index]);

			std::copy(lastOutput.begin(), lastOutput.end(), output[index].begin());
		}
	}

	inline std::string getStatus() const override {
		std::string status = "";

		spdlog::debug("Saving topology");
		for (const auto &l : _layers) {
			status += std::to_string(l.getSize()) + " ";
		}

		status += "\n";

		spdlog::debug("Saving weights");
		for (const auto &l : _layers) {
			const auto &neurons = l.getNeurons();

			for (const auto &n : neurons) {
				const auto &weights = n.getWeights();

				for (const auto &w : weights) {
					std::stringstream ss;
					ss << std::setprecision(std::numeric_limits<long double>::max_digits10)
					   << std::scientific << w;
					std::string weight_str = ss.str();
					status += weight_str + " ";
				}
				status += "\n";
			}
		}

		return status;
	}

	inline const layer_t &operator[](size_t index) const override { return _layers[index]; }
	inline layer_t &operator[](size_t index) override { return _layers[index]; }

	inline std::ostream &operator<<(std::ostream &os) const override {
		size_t index = 0;
		for (const auto &l : _layers) {
			os << "Layer [" << ++index << "]\n";
			os << l;
		}
		return os;
	}

	// TODO scrivere documentazione
	inline data_vv_t &&getEmptyOutputVector() const {
		data_vv_t *result = new data_vv_t(_layers.size(), data_vector_t());

		for (size_t i = 0; i < _layers.size(); ++i) {
			(*result)[i] = std::move(data_vector_t(_layers[i].getSize() + 1));
		}

		return std::move(*result);
	}

	// TODO scrivere documentazione
	inline data_vv_t &&getEmptyPreActivationVector() const {
		data_vv_t *result = new data_vv_t(_layers.size(), data_vector_t());

		for (size_t i = 0; i < _layers.size(); ++i) {
			(*result)[i] = std::move(data_vector_t(_layers[i].getSize()));
		}

		return std::move(*result);
	}

	// TODO scrivere documentazione
	inline void randomizeWeights() {
		std::random_device rd;
		std::mt19937 gen(rd());
		double min_value = 10e-7;
		double max_value = -10e-7;

		std::uniform_real_distribution<double> dis(min_value, max_value);

		for (auto &l : _layers)
			for (auto &n : l.getNeurons())
				for (auto &w : n.getWeights())
					w = dis(gen);
	}

	// TODO scrivere documentazione
	inline void createLayers(size_t layers, size_t inputSize) {
		_layers = std::move(std::vector<DefaultLayer>(layers));

		size_t lastOutputSize = inputSize + 1;
		for (size_t i = 0; i < layers - 1; ++i) {
			_layers[i].createNeurons(5, lastOutputSize);
			lastOutputSize = _layers[i].getSize() + 1;
		}
		_layers[this->getSize() - 1].createNeurons(1, lastOutputSize);
	}

	// TODO scrivere documentazione
	inline void createLayers(const std::vector<unsigned> &topology, size_t inputSize) {
		_layers = std::move(std::vector<DefaultLayer>(topology.size()));

		unsigned last = topology.size() - 1;
		size_t lastOutputSize = inputSize + 1;

		for(unsigned i = 0; i < last; ++i){
			_layers[i].createNeurons(topology[i], lastOutputSize);
			lastOutputSize = _layers[i].getSize() + 1;
		}
		_layers[this->getSize() - 1].createNeurons(1, lastOutputSize);
	}
};

} // namespace ANN

#endif // DEFAULT_NETWORK_INCLUDE_GUARD
