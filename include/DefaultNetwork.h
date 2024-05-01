#ifndef DEFAULT_NETWORK_INCLUDE_GUARD
#define DEFAULT_NETWORK_INCLUDE_GUARD

#include "DefaultLayer.h"
#include "Layer.h"
#include "Network.h"
#include <cstddef>

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

	// Setter
	inline void setLayers(const layer_vector_t &layers) override { _layers = layers; }

	inline void setLayer(const layer_t &layer, size_t index) override { _layers[index] = layer; }

	// Getter
	inline layer_vector_t &getLayers() override { return _layers; }

	inline size_t getSize() const override { return _layers.size(); }

	inline void evaluate(const data_vector_t &input, data_vector_t &output) const override {

		data_vector_t layerInput;
		data_vector_t layerOutput(input);
		layerOutput.push_back(-1); // bias

		for (auto it = _layers.begin(); it != _layers.end(); ++it) {

			layerInput = std::move(layerOutput);
			layerOutput = std::move(data_vector_t((*it).getSize() + 1));

			it->evaluate(layerInput, layerOutput);
		}

		layerOutput.pop_back();

		output = std::move(layerOutput);
	}

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
