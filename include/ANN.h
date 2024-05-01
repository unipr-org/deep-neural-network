#ifndef ANN_HH_INCLUSION_GUARD
#define ANN_HH_INCLUSION_GUARD

#include <functional>
#include <ostream>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
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
		string += *it;
		if (index != v.size() - 1)
			string += ", ";
	}
	string += "}";

	return string;
}

} // namespace ANN

#endif // ANN_HH_INCLUSION_GUARD
