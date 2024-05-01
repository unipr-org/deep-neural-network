#ifndef NETWORK_INCLUDE_GUARD
#define NETWORK_INCLUDE_GUARD

#include "Layer.h"
#include <vector>

namespace ANN {
template <typename Layer_t = Layer<>> class Network {
  public:
	using layer_t = Layer_t;
	using layer_vector_t = std::vector<layer_t>;
	using data_t = typename layer_t::data_t;
	using data_vector_t = typename layer_t::data_vector_t;

	// Setter
	inline virtual void setLayers(const layer_vector_t &layers) = 0;
	inline virtual void setLayer(const layer_t &layer, size_t index) = 0;

	// Getter
	inline virtual layer_vector_t &getLayers() = 0;
	inline virtual size_t getSize() const = 0;

	inline virtual void evaluate(const data_vector_t &input, data_vector_t &output) const = 0;

	inline virtual layer_t &operator[](size_t index) = 0;
	inline virtual std::ostream &operator<<(std::ostream &) const = 0;

	inline virtual ~Network() {}
};

template <typename Layer_t> std::ostream &operator<<(std::ostream &os, const Network<Layer_t> *n) {
	return n->operator<<(os);
}

template <typename Layer_t> std::ostream &operator<<(std::ostream &os, const Network<Layer_t> &n) {
	return n.operator<<(os);
}
} // namespace ANN

#endif // NETWORK_INCLUDE_GUARD
