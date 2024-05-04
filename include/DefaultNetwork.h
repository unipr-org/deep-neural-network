#ifndef DEFAULT_NETWORK_INCLUDE_GUARD
#define DEFAULT_NETWORK_INCLUDE_GUARD

#include "DefaultLayer.h"
#include "Layer.h"
#include "Network.h"
#include <cstddef>
#include <fstream>
#include <sys/stat.h>
#include <iostream>

namespace ANN {

class DefaultNetwork : public Network<DefaultLayer> {
  public:
	using Network<DefaultLayer>::data_t;
	using Network<DefaultLayer>::data_vector_t;
	using Network<DefaultLayer>::layer_t;
	using Network<DefaultLayer>::layer_vector_t;

  private:
	layer_vector_t _layers;

  public:

	/**
	 * @brief Default constructor for DefaultNetwork.
	 * 
	 * This constructor creates a DefaultNetwork object using the default constructor.
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

	inline void evaluate(const data_vector_t &input, data_vector_t &output, v_data_vector_t &neuronsPreactivations, v_data_vector_t &layersOutputs) const override {
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

			it->evaluate(layerInput, layerOutput, neuronsPreactivations[index], layersOutputs[index]);
		}

		layerOutput.pop_back();

		output = std::move(layerOutput);
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
			const auto& neurons = l.getNeurons();

			for(const auto &n : neurons) {
				const auto& weights = n.getWeights();

				for(const auto &w : weights) {
					status += std::to_string(w) + " ";
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
};

} // namespace ANN

#endif // DEFAULT_NETWORK_INCLUDE_GUARD
