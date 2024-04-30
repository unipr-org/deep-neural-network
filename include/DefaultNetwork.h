#ifndef DEFAULT_NETWORK_HH_INCLUSION_GUARD
#define DEFAULT_NETWORK_HH_INCLUSION_GUARD

#include "DefaultLayer.h"
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

	// Setter
	inline void setLayers(const layer_vector_t &layers) override { _layers = layers; }

	inline void setLayer(const layer_t &layer, size_t index) override { _layers[index] = layer; }

	// Getter
	inline layer_vector_t &getLayers() override { return _layers; }

	inline size_t getSize() const override { return _layers.size(); }

	inline void evaluate(const data_vector_t &input, data_vector_t &output) override {

		data_vector_t layerInput;
		data_vector_t layerOutput(input);

		for (auto it = _layers.begin(); it != _layers.end(); ++it) {

			layerInput = std::move(layerOutput);
			layerOutput = std::move(data_vector_t((*it).getSize()));

			it->evaluateOutput(layerInput, layerOutput);
		}

		output = std::move(layerOutput);
	}
};

} // namespace ANN

#endif // DEFAULT_NETWORK_HH_INCLUSION_GUARD
