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
	DefaultNetwork(const layer_vector_t &layers) : _layers(layers) {}

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

	inline virtual void saveStatus() const override {
		std::string folderPath = "./model";
		std::string filename = folderPath + "/status.txt";

		if (mkdir(folderPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
			if (errno != EEXIST) {
				spdlog::error("Unable to create directory {}", folderPath);
				return;
			}
		}
		
		std::ofstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			return;
		}

		spdlog::info("Saving topology");
		for (const auto &l : _layers) {
			model << l.getSize() << " ";
		}
		model << std::endl;

		spdlog::info("Saving weights");
		for (const auto &l : _layers) {
			auto neurons = l.getNeurons();

			for(const auto &n : neurons) {
				auto weights = n.getWeights();

				for(const auto &w : weights) {
					model << w << " ";
				}
				model << std::endl;
			}
		}

		spdlog::info("Status saved in {}", filename);

	}

	inline virtual void loadStatus() const override {
		std::string folderPath = "./model";
		std::string filename = folderPath + "/status.txt";

		std::ifstream model(filename);

		if (!model.is_open()) {
			spdlog::error("Unable to open {}", filename);
			return;
		}

		std::vector<int> topology;
		layer_vector_t layers;
		std::string line;

		getline(model, line); // topology
		std::istringstream iss(line);
		int value;
		
		while (iss >> value) {
			topology.push_back(value);
		}

		for(auto item : topology){
			int c = 0;
			DefaultLayer::neuron_vector_t neurons;
			while(c < item){
				getline(model, line);
				std::istringstream iss(line);
				data_t value;
				DefaultNeuron::weight_vector_t weights;
				
				while (iss >> value) {
					weights.push_back(value);
				}
				
				DefaultNeuron n(weights, heaviside);
				neurons.push_back(n);

				++c;
			}

			DefaultLayer l(neurons);
			layers.push_back(l);
		}

		model.close();

		DefaultNetwork net(layers);
		// Net loaded

		std::cout << net << std::endl;
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
