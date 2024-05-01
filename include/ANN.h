#ifndef ANN_INCLUDE_GUARD
#define ANN_INCLUDE_GUARD

#include <cmath>
#include <functional>
#include <ostream>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace ANN {

using weight_t = long double;
using data_t = long double;
using activationFunction_t = std::function<data_t(data_t)>;

template <typename T> std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
	size_t index = 0;

	os << "{";
	for (auto it = v.begin(); it != v.end(); ++it, ++index) {
		os << *it;
		if (index != v.size() - 1)
			os << ", ";
	}
	os << "}";

	return os;
}

template <typename T> std::string &operator+=(std::string &string, const std::vector<T> &v) {
	size_t index = 0;

	string += "{";
	for (auto it = v.begin(); it != v.end(); ++it, ++index) {
		string += std::to_string(*it);
		if (index != v.size() - 1)
			string += ", ";
	}
	string += "}";

	return string;
}

inline data_t sigmoid(data_t x) {
	data_t result = 1 / (1 + exp(-x));
	return result;
}

inline data_t sigmoid_d(data_t x) {
	data_t result = sigmoid(x);
	return result * (1 - result);
}

inline data_t heaviside(data_t x) { return (x >= 0) ? 1 : 0; }
inline data_t identity(data_t x) { return x; }

inline data_t tanh(data_t x) {
	data_t e1 = exp(x);
	data_t e2 = exp(-x);

	return (e1 - e2) / (e1 + e2);
}

inline data_t tanh_d(data_t x) {
	data_t e1 = exp(x);
	data_t e2 = exp(-x);

	return (1 - pow(tanh(x), 2));
}

} // namespace ANN

#endif // ANN_INCLUDE_GUARD
